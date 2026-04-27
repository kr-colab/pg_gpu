Changelog
=========

v0.1.0 (Current)
-----------------

First public release of pg_gpu.

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

* ``HaplotypeMatrix`` -- phased haplotype data (0/1 with -1 for missing).
  Loaders: ``from_vcf`` (with ``region=`` and ``samples=`` subsetting),
  ``from_zarr`` (auto-detects VCZ / scikit-allel / chromosome-grouped
  layouts), ``from_ts``, and direct NumPy construction. ``to_zarr`` writes
  VCZ by default. ``vcf_to_zarr`` provides multicore VCF-to-zarr conversion.
  Sample names from VCFs are preserved; ``load_pop_file('pops.txt')``
  assigns populations using stored sample names.

* ``GenotypeMatrix`` -- diploid genotypes (0/1/2). Same loaders and
  zarr round-trip as ``HaplotypeMatrix``. Many public functions
  auto-dispatch on input type (haplotype vs genotype).

Linkage Disequilibrium
~~~~~~~~~~~~~~~~~~~~~~

* Core statistics: ``r``, ``r_squared``, ``dd`` (D-squared), ``dz``,
  ``pi2`` (Ragsdale & Gravel 2019), ``zns`` (Kelly), ``omega``
  (Kim & Nielsen), ``mu_ld`` (RAiSD).
* LD pruning: ``locate_unlinked``; windowed :math:`r^2` decay:
  ``windowed_r_squared``.
* Two-population LD via ``compute_ld_statistics_gpu_two_pops`` with
  chunked GPU execution.

Diversity Statistics
~~~~~~~~~~~~~~~~~~~~

* Theta estimators: ``pi``, ``theta_w``, ``theta_h``, ``theta_l``,
  ``eta1``, ``eta1_star``, ``minus_eta1``, ``minus_eta1_star``.
* Neutrality tests: ``tajimas_d``, ``fay_wus_h``,
  ``normalized_fay_wus_h``, ``zeng_e``, ``zeng_dh``.
* Heterozygosity / inbreeding: ``heterozygosity_expected``,
  ``heterozygosity_observed``, ``inbreeding_coefficient``.
* Haplotype-level: ``haplotype_diversity``, ``haplotype_count``,
  ``daf_histogram``, ``diplotype_frequency_spectrum``.
* ``FrequencySpectrum`` class for custom weight functions, SFS
  projection, and the Achaz (2009) variance framework.
* All statistics accept ``missing_data='include' | 'exclude'`` and a
  unified ``span_normalize`` parameter that auto-detects the best
  denominator (accessible bases if mask set, else genomic span).

Divergence Statistics
~~~~~~~~~~~~~~~~~~~~~

* FST estimators: ``fst_hudson`` (ratio of averages),
  ``fst_weir_cockerham``, ``fst_nei``; ``pairwise_fst`` for multiple
  populations.
* Absolute / net divergence: ``dxy``, ``da``.
* Population Branch Statistic: ``pbs`` (normalized PBSn1).
* Distance-based two-population statistics (Schrider et al. 2018 and
  related): ``snn``, ``dxy_min``, ``gmin``, ``dd``, ``dd_rank``, ``zx``.
  Callers can pre-compute ``pairwise_distance_matrix`` once and pass it
  to multiple stats, or use the combined ``distance_based_stats``.

Selection Scans
~~~~~~~~~~~~~~~

* Haplotype-based: ``ihs`` (fused CUDA kernel, bitmask pair tracking,
  block-level EHH reductions), ``nsl``, ``xpehh``, ``xpnsl``,
  ``ehh_decay``.
* Garud's H: ``garud_h`` (H1, H12, H123, H2/H1) via GPU dot-product
  hashing of haplotypes; ``moving_garud_h`` uses cumulative prefix sums
  for O(1) per-window hash computation.
* Standardization: ``standardize``, ``standardize_by_allele_count``.
* Diploid variants: ``zns_diploid``, ``omega_diploid``,
  ``garud_h_diploid``, ``daf_histogram_diploid``.

Site Frequency Spectrum
~~~~~~~~~~~~~~~~~~~~~~~

* Unfolded and folded SFS: ``sfs``, ``sfs_folded``, ``sfs_scaled``,
  ``sfs_folded_scaled``.
* Two-population joint SFS: ``joint_sfs``, ``joint_sfs_folded``,
  ``joint_sfs_scaled``, ``joint_sfs_folded_scaled``.
* Folding utilities: ``fold_sfs``, ``fold_joint_sfs``.

Admixture / F-Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

* Per-variant: ``patterson_f2`` (F2 branch length), ``patterson_f3``
  (admixture test), ``patterson_d`` (ABBA-BABA).
* Windowed: ``moving_patterson_f3``, ``moving_patterson_d``.
* Block-jackknife with standard error: ``average_patterson_f3``,
  ``average_patterson_d``.

Resampling
~~~~~~~~~~

* Public ``pg_gpu.resampling`` module with ``block_jackknife`` and
  ``block_bootstrap`` for block-resampled standard errors / CIs on any scalar
  genome-wide statistic (genome-wide mean Tajima's D, ratio-of-sums
  estimators, etc.). Promotes the previously private ``_jackknife`` helper
  from ``admixture``. The weighted jackknife follows the Busing et al.
  (1999) delete-:math:`m_j` formulation for unequal block sizes.
* ``examples/sweep_tajimas_d_bootstrap.py`` -- 95% bootstrap CI on
  Tajima's D under a completed sweep, showing sweep-local vs distal mean
  difference CIs that exclude zero.

Dimensionality Reduction and Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``pca`` -- GPU-accelerated SVD PCA.
* ``randomized_pca`` -- truncated-SVD approximation for large
  datasets.
* ``pairwise_distance`` -- GPU-accelerated with memory-safe batching.
* ``pcoa`` -- classical MDS from a distance matrix.
* ``local_pca`` -- GPU port of Li & Ralph (2019) lostruct for detecting
  regions where population structure differs from the chromosome-wide
  pattern. Per-window top-k eigendecomposition via a single batched
  ``cp.linalg.eigh`` over a stacked
  ``(n_windows, n_samples, n_samples)`` tensor.
* ``pc_dist`` -- Frobenius distance between per-window low-rank
  covariance reps via the trace identity (no cov-matrix
  re-materialization). L1, L2, or no normalization.
* ``corners`` -- extreme-cluster selection in a 2D MDS embedding via
  Welzl's minimum enclosing circle.
* ``local_pca_jackknife`` -- delete-1 block jackknife standard error of local PCs,
  also GPU-batched with sign-aligned replicates.
* ``LocalPCAResult`` dataclass with ``.windows`` / ``.eigvals`` /
  ``.eigvecs`` / ``.sumsq`` plus ``.to_lostruct_matrix()`` for
  compatibility with the R ``lostruct::eigen_windows`` layout.

Relatedness and Kinship
~~~~~~~~~~~~~~~~~~~~~~~

* ``grm`` -- Genetic Relationship Matrix (Yang et al. 2011).
* ``ibs`` -- pairwise Identity-By-State proportions.

Fused Windowed Analysis
~~~~~~~~~~~~~~~~~~~~~~~

The ``windowed_analysis()`` convenience function routes through fused
CUDA kernels (one kernel launch for all windows) when using
non-overlapping windows with ``missing_data='include'``:

* Single-population: ``pi``, ``theta_w``, ``tajimas_d``,
  ``segregating_sites``, ``singletons``.
* Two-population: ``fst``, ``fst_hudson``, ``fst_wc``, ``dxy``, ``da``.
* Selection: ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``,
  ``mean_nsl``.
* Structure: ``local_pca`` (returns a ``LocalPCAResult``; scalar stats
  requested alongside are merged onto ``result.windows``).
* Structure: ``local_pca_jackknife`` computes delete-1 block jackknife
  standard error and populates ``LocalPCAResult.jackknife_se``. When both are
  requested together, per-window matrix preparation is shared.

Lower-level windowed entry points: ``windowed_statistics`` (scatter-add
aggregation) and ``windowed_statistics_fused`` (custom bin edges, one
thread block per window).

Distance Distribution Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``pairwise_diffs`` -- Hamming distance distributions (haploid or
  diploid).
* ``dist_var``, ``dist_skew``, ``dist_kurt`` -- moments of the
  pairwise-distance distribution (Schrider et al. 2018).
* ``dist_moments`` -- all three in one call.

diploSHIC / RAiSD Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``mu_var``, ``mu_sfs`` -- SNP density and SFS edge fraction (RAiSD).
* ``max_daf`` -- maximum derived allele frequency.

Visualization (``pg_gpu.plotting``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SFS: ``plot_sfs``, ``plot_joint_sfs``.
* LD: ``plot_pairwise_ld``, ``plot_ld_decay``.
* PCA / structure: ``plot_pca``, ``plot_pairwise_distance``.
* Windowed statistics: ``plot_windowed``, ``plot_windowed_panel``.
* Haplotypes: ``plot_haplotype_frequencies``, ``plot_variant_locator``.

Missing Data Handling
~~~~~~~~~~~~~~~~~~~~~

* Missing values are encoded as ``-1`` (haplotype) or ``-1`` sentinel
  (genotype).
* Every statistic accepts ``missing_data='include'`` (per-site valid
  data, default) or ``missing_data='exclude'`` (only fully genotyped
  sites). Simulation testing confirms ``include`` is unbiased under
  MCAR.
* LD projection estimator available via ``estimator='sigma_d2'`` on
  ``zns`` / ``omega``.

Moments Integration
~~~~~~~~~~~~~~~~~~~

``pg_gpu.moments_ld.compute_ld_statistics`` is a GPU drop-in for
``moments.LD.Parsing.compute_ld_statistics``. Returns the 15
two-population LD statistics and 3 heterozygosity statistics in the
exact layout moments expects for demographic inference. Requires the
``moments`` pixi environment: ``pixi install -e moments``.

Validation
~~~~~~~~~~

* Cross-validation script (``tests/validate_against_allel.py``)
  comparing 31 statistics against scikit-allel on real Ag1000G data
  (1M variants, 200 haplotypes). Divergence, diversity, and selection
  statistics match scikit-allel at machine precision; a timing table
  is included.
* Local PCA (lostruct) outputs validated against the R ``lostruct``
  package via frozen JSON references committed under ``tests/data/``.
  R is **not** a dependency of the pixi env or CI -- the comparison
  runs against the committed JSON. An optional ``requires_r`` test
  regenerates the references via rpy2 when R + lostruct are available
  locally.

Performance
~~~~~~~~~~~

All statistics run on CuPy with custom CUDA kernels for compute-bound
paths.

Scalar statistics at 1M variants, 200 haplotypes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Statistic
     - allel (s)
     - pg_gpu (s)
     - Speedup
   * - Weir-Cockerham FST
     - 9.85
     - 0.02
     - **468x**
   * - Patterson F2
     - 0.15
     - 0.009
     - **18x**
   * - nSL (255k variants)
     - 8.1
     - 0.56
     - **15x**
   * - Patterson F3
     - 0.14
     - 0.016
     - **9x**
   * - EHH decay (255k)
     - 0.06
     - 0.008
     - **8x**
   * - Hudson FST
     - 0.12
     - 0.017
     - **7x**
   * - iHS (255k variants)
     - 9.9
     - 1.5
     - **7x**
   * - Dxy
     - 0.07
     - 0.016
     - **4x**

Windowed statistics at 5.3M variants, 100kb windows, 200 haplotypes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Statistic
     - allel (s)
     - pg_gpu (s)
     - Speedup
   * - pi + theta_w + tajimas_d
     - 0.81
     - 0.013
     - **60x**
   * - All 5 single-pop stats
     - 0.81
     - 0.013
     - **60x**
   * - FST (Hudson)
     - 0.59
     - 0.18
     - **3x**
   * - All 12 stats together
     - n/a
     - 0.66
     - single call

Examples
~~~~~~~~

End-to-end demo scripts in ``examples/``:

* ``pg_gpu_tour.ipynb`` -- interactive tour using Anopheles gambiae X
  chromosome data.
* ``admixture_detection.py`` -- block-jackknife ABBA-BABA on simulated
  null and admixed msprime scenarios.
* ``accessibility_mask.py`` -- windowed :math:`\pi` with and without
  an accessibility mask over a low-:math:`\mu` "exon" region.
* ``ld_blocks.py`` -- LD-block partitioning via :math:`r^2` bridging
  scores.
* ``local_pca.py`` -- lostruct pipeline on a simulated partial
  selective sweep (``SweepGenicSelection`` with end frequency 0.5).

Infrastructure
~~~~~~~~~~~~~~

* pixi-based environment management; ``moments`` integration lives in
  a separate pixi feature.
* Shared ``_utils.py`` module for population extraction.
* Public API returns NumPy arrays (not CuPy) -- no need to call
  ``.get()`` on results.
