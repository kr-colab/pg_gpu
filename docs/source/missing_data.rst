Missing Data Handling
=====================

pg_gpu provides support for missing data across all population genetics
statistics. Missing data is encoded as ``-1`` in haplotype and genotype
matrices.

Missing Data Modes
------------------

Every function that operates on genetic data accepts a ``missing_data``
parameter with two options:

**include** (default)
   Use all sites, computing statistics from observed data only. Each site
   uses its own sample size (``n_valid``). For haplotype identity
   comparisons (e.g., Garud's H), missing values are treated as wildcards
   compatible with any allele.

**exclude**
   Drop entire sites that have any missing data in any sample. Only
   fully genotyped sites contribute to the result.

Simulation testing under the standard neutral model confirms that
``include`` mode is unbiased under MCAR (missing completely at random)
at missingness rates from 0--60% for pi, theta_w, theta_h, theta_l,
Tajima's D, dxy, Hudson FST, da, and all SFS-based estimators.

Basic Usage
-----------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence

   h = HaplotypeMatrix.from_vcf("data.vcf")

   # Default: per-site valid data
   pi = diversity.pi(h)

   # Conservative: only fully genotyped sites
   pi_excl = diversity.pi(h, missing_data='exclude')

How It Works
------------

Consider a site with 100 haplotypes where 10 are missing (``-1``):

* **include**: Computes allele frequencies from the 90 observed
  haplotypes. The site contributes to the result with ``n_valid = 90``.
* **exclude**: The site is dropped entirely because it has missing data.

For statistics that require a single sample size (e.g., Tajima's D
variance formula), the harmonic mean of per-site sample sizes is used.

Supported Statistics
--------------------

Every public function accepts the ``missing_data`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - include
     - exclude
   * - Diversity (pi, theta_w, theta_h, theta_l)
     - per-site n
     - filter sites
   * - Neutrality tests (tajimas_d, fay_wus_h, H*, E, DH)
     - per-site n, harmonic mean for variance
     - filter sites
   * - Divergence (dxy, fst_hudson, fst_weir_cockerham, da)
     - per-site n
     - filter sites
   * - SFS (sfs, joint_sfs, folded variants)
     - per-site n
     - filter sites
   * - Admixture (patterson_d, f2, f3)
     - per-site n
     - filter sites
   * - Selection scans (ihs, nsl, xpehh)
     - wildcard in shared-site length (SSL)
     - filter sites
   * - Haplotype stats (garud_h, haplotype_diversity)
     - wildcard match
     - filter sites
   * - Distance (pairwise_diffs, pca)
     - per-pair norm
     - filter sites
   * - LD (zns, omega)
     - per-site n
     - filter sites
   * - SFS estimators (FrequencySpectrum)
     - group by n
     - filter sites

Multiallelic Sites
------------------

pg_gpu encodes haplotype alleles as integers: ``0`` = reference,
``1`` = first alternate, ``2`` = second alternate, etc. Missing data
is ``-1``.

For derived allele counting (used by all SFS-based statistics),
any non-reference allele is treated as derived: allele values
1, 2, 3, etc. all contribute 1 to the derived allele count. This
is the standard "reference vs any alternate" folding used by most
population genetics tools.

This means multiallelic VCF sites are handled correctly without
filtering -- a site with REF=A, ALT=T,C where some samples carry
the C allele will count all non-reference haplotypes as derived.

If you need strictly biallelic data (e.g., for LD statistics or
compatibility with other tools), use ``apply_biallelic_filter()``:

.. code-block:: python

   h = h.apply_biallelic_filter()  # keeps only 0/1 sites

This is available on both ``HaplotypeMatrix`` and ``GenotypeMatrix``.
Note that ``GenotypeMatrix.from_vcf()`` applies this filter
automatically during loading.

LD Estimator Choice
-------------------

For LD statistics, ``zns()`` and ``omega()`` accept an ``estimator``
parameter independent of missing data handling:

* ``estimator='r2'`` (default): naive r-squared.
* ``estimator='sigma_d2'``: unbiased multinomial projection estimators
  (Ragsdale & Gravel 2019), computing sigma_D^2 = D^2/pi^2 with
  falling-factorial corrections. More robust with small or variable
  sample sizes.

.. code-block:: python

   from pg_gpu import ld_statistics

   # Default: naive r-squared
   zns = ld_statistics.zns(h)

   # Unbiased estimator
   zns_unbiased = ld_statistics.zns(h, estimator='sigma_d2')

Haplotype Identity and Missing Data
------------------------------------

For statistics based on haplotype identity (Garud's H, haplotype
diversity, haplotype count), missing values are treated as wildcards:
two haplotypes match if they agree at all positions where both are
non-missing.

.. code-block:: python

   # Haplotypes [0, 1, 0, 1] and [0, -1, 0, 1] are considered identical
   # because they match at positions 0, 2, 3 (position 1 is missing)

   from pg_gpu import selection
   h1, h12, h123, h2_h1 = selection.garud_h(h)

HaplotypeMatrix and GenotypeMatrix Utilities
--------------------------------------------

The same utilities are available on both ``HaplotypeMatrix`` and
``GenotypeMatrix`` -- substitute ``gm`` for ``h`` below if you are
working with diploid genotypes.

.. code-block:: python

   # Detect and count missing data
   missing_per_site = h.count_missing(axis=0)
   missing_per_sample = h.count_missing(axis=1)

   # Filter by missing data frequency
   h_clean = h.filter_variants_by_missing(max_missing_freq=0.1)

   # Summary statistics
   summary = h.summarize_missing_data()

Accessible Site Masks
---------------------

Genome accessibility masks (from BED files) define which sites are
callable in a sequencing experiment. This matters for normalization:
if only 60% of a region is accessible, per-base diversity estimates
should divide by the accessible base count, not the total span.

pg_gpu integrates accessibility masks into ``HaplotypeMatrix`` and
``GenotypeMatrix`` as a non-destructive filter. When a mask is set,
the ``haplotypes`` and ``positions`` properties transparently return
only variants within accessible regions. The original data is preserved
and the mask can be swapped or removed at any time.

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   h = HaplotypeMatrix.from_vcf("data.vcf.gz")
   print(h.num_variants)  # e.g. 50,000

   # Attach a mask -- only variants in accessible regions are visible
   h.set_accessible_mask("accessibility.bed", chrom="3L")
   print(h.num_variants)  # e.g. 42,000 (filtered)

   # n_total_sites is automatically set to the accessible base count
   print(h.n_total_sites)  # e.g. 30,000,000

   # Masks can also be set at load time
   h = HaplotypeMatrix.from_vcf("data.vcf.gz",
                                 accessible_bed="accessibility.bed")
   h = HaplotypeMatrix.from_zarr("data.zarr", region="3L:1-10000000",
                                  accessible_bed="accessibility.bed")

   # Remove the mask to restore all variants
   h.remove_accessible_mask()
   print(h.num_variants)  # back to 50,000

**Key behaviors:**

* ``set_accessible_mask()`` is non-destructive and returns ``self``
  for chaining. It automatically sets ``n_total_sites`` to the count
  of accessible bases.

* The mask covers the union of the BED's extent and the matrix's
  ``[chrom_start, chrom_end]`` range, so BED accessible bases that
  fall outside the variant range (common for variants-only VCFs) are
  not silently dropped. ``n_total_sites`` always equals the full BED
  accessible-base count.

* ``get_span('accessible')`` returns the accessible base count, used
  for per-base normalization. This matches the denominator used by
  ``allel.sequence_diversity(is_accessible=...)``.

* The mask stays on CPU and uses a lazy prefix-sum for O(1) range
  queries, so windowed analysis over many windows is efficient.

Site Count Properties
---------------------

After a mask is attached (or ``include_invariant=True`` was passed at
load time), three properties decompose the analysis universe:

* ``n_callable_sites`` -- alias for ``n_total_sites``; the BED span when
  masked, or the matrix length if loaded with ``include_invariant=True``.
* ``n_segregating_sites`` -- polymorphic sites in the matrix
  (``0 < derived_count < n_valid``).
* ``n_invariant_sites`` -- ``n_callable_sites - n_segregating_sites``;
  may include implied invariants outside the matrix when the VCF was
  variants-only.

These satisfy ``n_callable_sites == n_segregating_sites + n_invariant_sites``.
Note that ``num_variants`` is the *physical* matrix row count and is
generally not equal to either ``n_segregating_sites`` (which excludes
monomorphic rows) or ``n_callable_sites`` (which can include implied
invariants).

.. code-block:: python

   h.set_accessible_mask("accessibility.bed", chrom="3L")
   h.n_callable_sites          # e.g. 30,000,000 (BED total)
   h.n_segregating_sites       # e.g. 1,200,000  (polymorphic in matrix)
   h.n_invariant_sites         # e.g. 28,800,000 (callable - segregating)
   h.num_variants              # whatever rows are physically present

* ``get_subset()`` and ``get_population_matrix()`` read from the
  filtered properties, so child matrices automatically contain only
  accessible variants.

**Interaction with missing data modes:**

Accessibility masks and missing data modes are complementary. The mask
controls *which variants are visible* (a site-level filter based on
genome quality), while ``missing_data`` controls *how missing genotypes
at visible sites are handled* (a sample-level concern). Both can be
active simultaneously:

.. code-block:: python

   h = HaplotypeMatrix.from_vcf("data.vcf.gz",
                                 accessible_bed="mask.bed")
   pi = diversity.pi(h)  # uses accessible mask + per-site valid counts

Span Normalization
------------------

Rate estimators (pi, theta_w, dxy, etc.) accept a ``span_normalize``
parameter that controls *how results are expressed*. This is orthogonal
to missing data handling.

``span_normalize`` accepts ``True`` or ``False``:

* ``True`` (default): auto-detect the best denominator. If an accessible
  mask is set, divides by ``mask.total_accessible`` (the BED span).
  Otherwise divides by the genomic span (1-based inclusive,
  ``chrom_end - chrom_start + 1``).
* ``False``: return raw sum (used internally by composite statistics like
  Tajima's D, and by advanced users who need custom normalization).

.. code-block:: python

   # Per base pair (default -- auto-detects best denominator)
   pi = diversity.pi(h)

   # With accessible mask: auto uses accessible bases
   h.set_accessible_mask("mask.bed", chrom="3L")
   pi = diversity.pi(h)  # per accessible base, automatically

   # Raw sum (no normalization)
   pi_raw = diversity.pi(h, span_normalize=False)

Test statistics (Tajima's D, Fay-Wu's H, FST) do not accept
``span_normalize`` -- they are dimensionless by definition.

SFS Projection
--------------

When samples are missing at different sites the per-site sample size
varies, which complicates statistics that are sensitive to sample size
(notably theta estimators). *Hypergeometric projection* re-expresses an
observed SFS as the SFS that would have been seen if every site had
been called in exactly ``target_n`` randomly chosen samples. The
projection is unbiased and lets you build a single, comparable SFS from
data with mixed sample sizes -- and to compare populations that were
sequenced to different depths. ``FrequencySpectrum`` supports it
following Marth et al. (2004) / the implementation used in
:math:`\partial a \partial i` (Gutenkunst et al. 2009):

.. code-block:: python

   from pg_gpu.diversity import FrequencySpectrum

   fs = FrequencySpectrum(h, population="pop1")
   fs_proj = fs.project(target_n=50)   # project down to n=50
   pi_proj = fs_proj.theta("pi")        # any theta on the projected SFS

Component-Level Access
----------------------

For advanced use cases (e.g., custom windowed aggregation), raw pairwise
difference and comparison counts are available via separate functions:

.. code-block:: python

   from pg_gpu.diversity import pi_components
   from pg_gpu.divergence import dxy_components

   # Within-population: (total_diffs, total_comps, total_missing, n_sites)
   diffs, comps, missing, n = pi_components(h.haplotypes)
   pi_manual = diffs / comps

   # Between-population: (total_diffs, total_comps, n_sites)
   pop1_haps = h.haplotypes[h.sample_sets['pop1']]
   pop2_haps = h.haplotypes[h.sample_sets['pop2']]
   diffs, comps, n = dxy_components(pop1_haps, pop2_haps)
   dxy_manual = diffs / comps

Best Practices
--------------

1. **Use include mode** (default) for most analyses. It uses all
   available data at each site and is unbiased under MCAR.

2. **Use exclude mode** when you need all samples to be comparable
   at exactly the same sites (e.g., for certain LD analyses or when
   missingness is non-random).

3. **Use FrequencySpectrum** for theta estimators and neutrality tests
   when you want proper handling of variable sample sizes via group-by-n
   or SFS projection.

4. **Use estimator='sigma_d2'** for LD statistics (ZnS, Omega) when
   sample sizes are small or variable. The unbiased estimators correct
   the upward bias inherent in naive r^2.

5. **Check missingness patterns** before analysis with
   ``summarize_missing_data()`` and consider filtering sites with
   very high missing rates.

Example Workflow
----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence
   from pg_gpu.windowed_analysis import windowed_analysis

   # Load data with accessible mask
   h = HaplotypeMatrix.from_zarr("data.zarr", region="3L:1-10000000",
                                  accessible_bed="accessibility.bed")

   # Inspect missing data
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['missing_freq_overall']:.1%}")

   # Filter extreme missingness
   h = h.filter_variants_by_missing(max_missing_freq=0.5)

   # Scalar statistics (auto-normalized by accessible bases)
   pi = diversity.pi(h, population="pop1")
   tajd = diversity.tajimas_d(h, population="pop1")
   dxy = divergence.dxy(h, 'pop1', 'pop2')
   fst = divergence.fst_hudson(h, 'pop1', 'pop2')

   # Windowed analysis (accessible mask propagates per-window)
   df = windowed_analysis(h, window_size=50_000,
                          statistics=['pi', 'theta_w', 'tajimas_d',
                                      'fst', 'dxy'],
                          populations=['pop1', 'pop2'])
