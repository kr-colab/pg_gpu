Phasing
=======

A ``HaplotypeMatrix`` stores two rows per diploid individual ``i``: rows ``2i``
and ``2i + 1`` are the individual's two copies. In phased data, each row
follows one chromosome across sites. If the data is unphased, a row does not
represent a real chromosome.

Many statistics use only per-site allele counts and are unaffected by
phasing (e.g., ``pi``, ``fst``, and the site frequency spectrum). Other
statistics assume the rows are real haplotypes (e.g., extended haplotype
homozygosity, haplotype LD).

The ``phased`` flag
-------------------

Every ``HaplotypeMatrix`` and ``StreamingHaplotypeMatrix`` has a
``phased`` flag. It records what you know about the data; pg_gpu does not
infer phasing from the input. Loaders set the flag from their ``phased``
argument, and you can change it later with ``set_phased()`` or
``set_unphased()``. These methods change only the flag; the underlying
data remain unchanged.

.. list-table::
   :header-rows: 1
   :widths: 45 15

   * - How the matrix was made
     - ``phased`` default
   * - ``HaplotypeMatrix.from_vcf``
     - ``False``
   * - ``HaplotypeMatrix.from_zarr`` (eager and streaming)
     - ``False``
   * - ``HaplotypeMatrix(haplotypes, positions, ...)``
     - ``False``
   * - ``HaplotypeMatrix.from_ts``
     - ``True``
   * - ``GenotypeMatrix.to_haplotype_matrix``
     - always ``False``

Subsets, population slices, and filtered copies inherit the flag of the source
matrix.

Loaders do not inspect phase information in the input. A VCF with
``0|1``-style genotypes still loads as ``phased=False`` by default: a
``|`` separator alone does not establish that the genotypes were phased
reliably. The flag is not saved by ``to_zarr``, so pass ``phased=True``
again when loading a store.

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   h = HaplotypeMatrix.from_vcf("phased.vcf.gz", phased=True)
   # or, after loading:
   h = HaplotypeMatrix.from_vcf("phased.vcf.gz").set_phased()

Simulated data
~~~~~~~~~~~~~~

Simulated haplotypes are usually phased by construction (e.g., msprime
output). ``from_ts`` already assumes the underlying data is phased and
sets the flag to ``True``.

Warnings for unphased data
--------------------------

A phase-dependent statistic emits an ``UnphasedDataWarning`` when it receives
a matrix with ``phased=False``. Each public call warns once, and the warning
points to the line that made the call. If an unphased alternative exists,
the warning message names it.

The warning does not change the result. You can:

* mark the matrix as phased if the data really are phased (or haploid);
* use an unphased alternative from the tables below, where one is available;
* silence the warning if you have checked that it is appropriate to do so:

.. code-block:: python

   import warnings
   from pg_gpu import UnphasedDataWarning
   warnings.filterwarnings("ignore", category=UnphasedDataWarning)

Which functions need phase
--------------------------

Functions not listed here do not depend on phase.

Phase-dependent only on a ``HaplotypeMatrix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions accept either matrix type. On a ``HaplotypeMatrix``, they
read rows as haplotypes and depend on phase. On a ``GenotypeMatrix``, they
use an unphased diploid calculation.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - On a ``GenotypeMatrix``
   * - ``selection.garud_h``
     - Diplotype frequencies (multilocus genotypes; Harris et al. 2018)
   * - ``distance_stats.pairwise_diffs``, ``dist_moments``, ``dist_var``,
       ``dist_skew``, ``dist_kurt``
     - Distances between individuals
   * - ``ld_statistics.zns``, ``omega``
     - Genotype LD (Rogers-Huff by default). On a ``HaplotypeMatrix``, these
       depend on phase with a haplotype estimator (``'r2'``, ``'sigma_d2'``,
       or the default ``'auto'``). ``estimator='rogers_huff'`` is the unphased alternative.

Haplotype-only functions
~~~~~~~~~~~~~~~~~~~~~~~~

These functions accept only haplotype data and warn when the matrix has
``phased=False``. Conditions are noted in the function column. The second
column lists an unphased alternative when one is available.

.. list-table:: Selection
   :header-rows: 1
   :widths: 35 65

   * - Phase-dependent
     - Phase-free alternative
   * - ``selection.moving_garud_h``
     - None available
   * - ``selection.ihs``, ``xpehh``, ``ehh_decay``
     - None available
   * - ``selection.nsl``, ``xpnsl``
     - None available

.. list-table:: Diversity and divergence
   :header-rows: 1
   :widths: 35 65

   * - Phase-dependent
     - Phase-free alternative
   * - ``diversity.haplotype_diversity``, ``haplotype_count``
     - None available
   * - ``diversity.diversity_stats``, only when ``'haplotype_diversity'`` is
       requested
     - Omit that statistic
   * - ``divergence.pairwise_distance_matrix``
     - ``distance_stats.pairwise_diffs_diploid`` (distances between
       individuals)
   * - ``divergence.snn``, ``dxy_min``, ``gmin``, ``dd``, ``dd_rank``
     - None. These functions warn only when they compute the distances
       themselves; passing ``distance_matrices=`` does not warn.
   * - ``divergence.distance_based_stats``
     - None. This function always computes the distances, so it warns.
   * - ``divergence.zx``
     - None available

.. list-table:: Distances, LD, and decomposition
   :header-rows: 1
   :widths: 35 65

   * - Phase-dependent
     - Phase-free alternative
   * - ``distance_stats.pairwise_diffs_haploid``
     - ``pairwise_diffs_diploid``
   * - ``ld_statistics.mu_ld``
     - None available
   * - ``HaplotypeMatrix.pairwise_r2``, ``windowed_r_squared`` with
       ``estimator='r2'`` (the default)
     - ``estimator='rogers_huff'`` (requires no missing calls;
       ``windowed_r_squared`` does not support ``pop`` with this estimator)
   * - ``HaplotypeMatrix.pairwise_LD_v``, ``locate_unlinked``
     - None available
   * - ``HaplotypeMatrix.tally_gpu_haplotypes*``,
       ``compute_ld_statistics_gpu_single_pop`` / ``_two_pops`` (eager and
       streaming)
     - ``moments_ld.compute_ld_statistics(use_genotypes=True)``
   * - ``moments_ld.compute_ld_statistics`` with ``use_genotypes=False`` on a
       ``haplotype_matrix=``
     - ``use_genotypes=True`` (the default; the unphased estimators of
       Ragsdale & Gravel 2020)
   * - ``decomposition.pca``, ``randomized_pca``
     - ``pca_dosage``, ``randomized_pca_dosage``
   * - ``decomposition.pairwise_distance``
     - ``pairwise_diffs_diploid``, ``relatedness.ibs``
   * - ``decomposition.local_pca``, ``local_pca_jackknife``, ``lostruct``
     - None available

When called with ``vcf_file=...`` and ``use_genotypes=False``,
``moments_ld.compute_ld_statistics`` loads the VCF with ``phased=True``.
Choosing the haplotype estimator in this form is treated as an assertion
that the VCF data are phased. If you pass ``haplotype_matrix=...`` instead,
the function checks that matrix's flag and warns when it is unphased.

Sample sets and population lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Caution is warranted when defining sample sets or population lists with
unphased data.

``relatedness.genetic_relatedness`` depends on phase when the sample set
contains only one of an individual's haplotypes. Because the default
``sample_sets`` is one sample per haplotype, on an unphased matrix
(``hm.phased`` is ``False``) the function warns when no ``sample_sets`` are
provided, or when a set contains only one of an individual's haplotypes,
such as ``[[0], [1, 2, 3]]``. It does not warn for sets such as
``[[0, 1], [2, 3]]``, as each set contains both of an individual's
haplotypes.

This same caveat applies to any statistic given a custom-made list of indices
from a ``HaplotypeMatrix`` to define a population. Populations from
``load_pop_file`` always contain whole individuals. In the example below, you
would get a different answer depending on whether the ``Haplotype_Matrix`` is
truly phased or not. However, no warning will be given.

.. code-block:: python

   # Rows 0, 2 and 4 are one haplotype each from individuals 0, 1 and 2
   diversity.pi(hm, population=[0, 2, 4])

Windowed scans
~~~~~~~~~~~~~~

``windowed_analysis``, ``windowed_statistics_fused``,
``windowed_statistics_fused_chunked``, and the ``WindowedAnalyzer.compute*``
methods warn once per scan if the requested statistics include any of:

``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``,
``haplotype_count``, ``mean_nsl``, ``zns``, ``omega``, ``mu_ld``,
``ld_decay``, ``mean_r2``, ``dist_var``, ``dist_skew``, ``dist_kurt``,
``local_pca``, ``local_pca_jackknife``.

The warning lists the requested statistics that depend on phase. Windowed
scans currently accept only a ``HaplotypeMatrix`` and do not provide an
estimator option for these statistics. Where an unphased alternative
exists, compute it on each window with the corresponding function:

* ``garud_*``: call ``selection.garud_h`` on a ``GenotypeMatrix``;
* ``dist_var``, ``dist_skew``, ``dist_kurt``: ``distance_stats.dist_moments``
  on a ``GenotypeMatrix``;
* ``zns``, ``omega``: ``estimator='rogers_huff'``, or a ``GenotypeMatrix``;
* ``mean_r2``, ``ld_decay``: ``pairwise_r2(estimator='rogers_huff')``
  (requires no missing calls).

There is no unphased alternative for ``haplotype_count``,
``mean_nsl``, ``mu_ld``, ``local_pca``, or ``local_pca_jackknife``.

A custom statistic passed as a function is not checked by the scan itself.
Any phase-dependent function it calls warns from the custom statistic.
