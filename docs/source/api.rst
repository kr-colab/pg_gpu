API Reference
=============

HaplotypeMatrix
---------------

.. autoclass:: pg_gpu.HaplotypeMatrix
   :members:
   :undoc-members:
   :show-inheritance:

GenotypeMatrix
--------------

.. autoclass:: pg_gpu.GenotypeMatrix
   :members:
   :undoc-members:
   :show-inheritance:

LD Statistics
-------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd

.. autofunction:: pg_gpu.ld_statistics.dz

.. autofunction:: pg_gpu.ld_statistics.pi2

.. autofunction:: pg_gpu.ld_statistics.r

.. autofunction:: pg_gpu.ld_statistics.r_squared

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd_within

.. autofunction:: pg_gpu.ld_statistics.dd_between

.. autofunction:: pg_gpu.ld_statistics.compute_ld_statistics

.. autofunction:: pg_gpu.ld_statistics.zns

.. autofunction:: pg_gpu.ld_statistics.omega

.. autofunction:: pg_gpu.ld_statistics.mu_ld

Population Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

* Single population: ``populations=None`` or omit
* Two populations: ``populations=(0, 1)`` for DD
* Three indices: ``populations=(0, 0, 1)`` for Dz
* Four indices: ``populations=(0, 0, 1, 1)`` for pi2

Diversity Statistics
--------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.pi
.. autofunction:: pg_gpu.diversity.theta_w
.. autofunction:: pg_gpu.diversity.tajimas_d
.. autofunction:: pg_gpu.diversity.haplotype_diversity
.. autofunction:: pg_gpu.diversity.allele_frequency_spectrum
.. autofunction:: pg_gpu.diversity.segregating_sites
.. autofunction:: pg_gpu.diversity.singleton_count
.. autofunction:: pg_gpu.diversity.fay_wus_h
.. autofunction:: pg_gpu.diversity.heterozygosity_expected
.. autofunction:: pg_gpu.diversity.heterozygosity_observed
.. autofunction:: pg_gpu.diversity.inbreeding_coefficient

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.theta_l
.. autofunction:: pg_gpu.diversity.normalized_fay_wus_h
.. autofunction:: pg_gpu.diversity.zeng_e
.. autofunction:: pg_gpu.diversity.zeng_dh
.. autofunction:: pg_gpu.diversity.diversity_stats
.. autofunction:: pg_gpu.diversity.neutrality_tests
.. autofunction:: pg_gpu.diversity.theta_h
.. autofunction:: pg_gpu.diversity.max_daf
.. autofunction:: pg_gpu.diversity.haplotype_count
.. autofunction:: pg_gpu.diversity.daf_histogram
.. autofunction:: pg_gpu.diversity.diplotype_frequency_spectrum
.. autofunction:: pg_gpu.diversity.mu_var
.. autofunction:: pg_gpu.diversity.mu_sfs

Divergence Statistics
---------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.fst
.. autofunction:: pg_gpu.divergence.fst_hudson
.. autofunction:: pg_gpu.divergence.fst_weir_cockerham
.. autofunction:: pg_gpu.divergence.fst_nei
.. autofunction:: pg_gpu.divergence.dxy
.. autofunction:: pg_gpu.divergence.da

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.divergence_stats
.. autofunction:: pg_gpu.divergence.pairwise_fst
.. autofunction:: pg_gpu.divergence.pbs

Distance-Based Two-Population Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.snn
.. autofunction:: pg_gpu.divergence.dxy_min
.. autofunction:: pg_gpu.divergence.gmin
.. autofunction:: pg_gpu.divergence.dd
.. autofunction:: pg_gpu.divergence.dd_rank
.. autofunction:: pg_gpu.divergence.zx
.. autofunction:: pg_gpu.divergence.pairwise_distance_matrix
.. autofunction:: pg_gpu.divergence.distance_based_stats

Selection Scan Statistics
-------------------------

Haplotype-Based Selection Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.ihs
.. autofunction:: pg_gpu.selection.xpehh
.. autofunction:: pg_gpu.selection.nsl
.. autofunction:: pg_gpu.selection.xpnsl

EHH and Haplotype Homozygosity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.ehh_decay
.. autofunction:: pg_gpu.selection.garud_h
.. autofunction:: pg_gpu.selection.moving_garud_h

Standardization
~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.standardize
.. autofunction:: pg_gpu.selection.standardize_by_allele_count

Site Frequency Spectrum
-----------------------

Single Population
~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.sfs
.. autofunction:: pg_gpu.sfs.sfs_folded
.. autofunction:: pg_gpu.sfs.sfs_scaled
.. autofunction:: pg_gpu.sfs.sfs_folded_scaled

Joint (Two-Population)
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.joint_sfs
.. autofunction:: pg_gpu.sfs.joint_sfs_folded
.. autofunction:: pg_gpu.sfs.joint_sfs_scaled
.. autofunction:: pg_gpu.sfs.joint_sfs_folded_scaled

Scaling and Folding Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.scale_sfs
.. autofunction:: pg_gpu.sfs.scale_sfs_folded
.. autofunction:: pg_gpu.sfs.scale_joint_sfs
.. autofunction:: pg_gpu.sfs.scale_joint_sfs_folded
.. autofunction:: pg_gpu.sfs.fold_sfs
.. autofunction:: pg_gpu.sfs.fold_joint_sfs

Admixture and F-Statistics
--------------------------

Per-Variant Statistics
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.patterson_f2
.. autofunction:: pg_gpu.admixture.patterson_f3
.. autofunction:: pg_gpu.admixture.patterson_d

Windowed Statistics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.moving_patterson_f3
.. autofunction:: pg_gpu.admixture.moving_patterson_d

Block-Jackknife Averaged
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.average_patterson_f3
.. autofunction:: pg_gpu.admixture.average_patterson_d

Dimensionality Reduction and Distance
--------------------------------------

PCA
~~~

.. autofunction:: pg_gpu.decomposition.pca
.. autofunction:: pg_gpu.decomposition.randomized_pca

Distance
~~~~~~~~

.. autofunction:: pg_gpu.decomposition.pairwise_distance
.. autofunction:: pg_gpu.decomposition.pcoa

Local PCA (lostruct)
~~~~~~~~~~~~~~~~~~~~

GPU port of the ``lostruct`` method (`Li & Ralph 2019
<https://www.genetics.org/content/211/1/289>`_) for detecting genomic
regions where population structure differs from the chromosome-wide
pattern (inversions, introgression, low-recombination regions). 
.. Per-window eigendecomposition is batched with a single ``cp.linalg.eigh`` over a
.. stacked ``(n_windows, n_samples, n_samples)`` tensor.

.. autofunction:: pg_gpu.decomposition.local_pca
.. autofunction:: pg_gpu.decomposition.local_pca_jackknife
.. autofunction:: pg_gpu.decomposition.pc_dist
.. autofunction:: pg_gpu.decomposition.corners

.. autoclass:: pg_gpu.decomposition.LocalPCAResult
   :members: to_lostruct_matrix, n_windows, n_samples
   :show-inheritance:

Relatedness and Kinship
-----------------------

.. autofunction:: pg_gpu.relatedness.grm
.. autofunction:: pg_gpu.relatedness.ibs

Windowed Statistics (GPU-Native)
--------------------------------

The ``windowed_analysis()`` convenience function automatically routes
through fused CUDA kernels for maximum performance. A single kernel
launch processes all windows in parallel.

.. autofunction:: pg_gpu.windowed_analysis.windowed_analysis

.. autofunction:: pg_gpu.windowed_analysis.windowed_statistics_fused

Supported Fused Windowed Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-population (one kernel launch for all):

- ``pi`` -- nucleotide diversity
- ``theta_w`` -- Watterson's theta
- ``tajimas_d`` -- Tajima's D
- ``segregating_sites`` -- count of segregating sites
- ``singletons`` -- count of singletons

Two-population (one kernel launch for all):

- ``fst`` -- Hudson's FST (ratio of averages)
- ``fst_hudson`` -- alias for ``fst``
- ``fst_wc`` -- Weir-Cockerham FST (haploid)
- ``dxy`` -- absolute divergence
- ``da`` -- net divergence (Dxy - mean within-pop pi)

Selection scan statistics:

- ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1`` -- Garud's H statistics per window (prefix-sum hashing + shared-memory sort)
- ``mean_nsl`` -- mean nSL per window (per-site nSL + scatter binning)

Structure / dimensionality reduction:

- ``local_pca`` -- Li & Ralph (2019) local PCA. Vector-valued per window; dispatches to :func:`pg_gpu.decomposition.local_pca` and returns a :class:`~pg_gpu.decomposition.LocalPCAResult` instead of the scalar-stat DataFrame. Can be combined with scalar statistics in the same call; the scalar columns are merged onto ``result.windows``.

Legacy Functions
~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.windowed_analysis.windowed_statistics

Moments Integration (LD Inference)
-----------------------------------

GPU-accelerated drop-in replacement for ``moments.LD.Parsing.compute_ld_statistics()``.
Computes the 15 two-population LD statistics (DD, Dz, pi2) and 3 heterozygosity
statistics on GPU, returning output in the exact format moments expects for
demographic inference.

Requires the ``moments`` pixi environment: ``pixi install -e moments``.

.. autofunction:: pg_gpu.moments_ld.compute_ld_statistics

Distance Distribution Statistics
---------------------------------

.. autofunction:: pg_gpu.distance_stats.pairwise_diffs
.. autofunction:: pg_gpu.distance_stats.dist_moments
.. autofunction:: pg_gpu.distance_stats.dist_var
.. autofunction:: pg_gpu.distance_stats.dist_skew
.. autofunction:: pg_gpu.distance_stats.dist_kurt

Visualization
-------------

Site Frequency Spectrum
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_sfs
.. autofunction:: pg_gpu.plotting.plot_joint_sfs

Linkage Disequilibrium
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_pairwise_ld
.. autofunction:: pg_gpu.plotting.plot_ld_decay

PCA and Population Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_pca
.. autofunction:: pg_gpu.plotting.plot_pairwise_distance

Windowed Statistics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_windowed
.. autofunction:: pg_gpu.plotting.plot_windowed_panel

Haplotype Structure
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_haplotype_frequencies
.. autofunction:: pg_gpu.plotting.plot_variant_locator

FrequencySpectrum (SFS-Based Estimation)
----------------------------------------

.. autoclass:: pg_gpu.diversity.FrequencySpectrum
   :members:
   :undoc-members:
