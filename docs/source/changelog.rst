Changelog
=========

v0.1.0 (Current)
-----------------

This release focuses on correctness auditing against scikit-allel, GPU kernel
optimization, and comprehensive fused windowed analysis.

Correctness Fixes
~~~~~~~~~~~~~~~~~

* **Divergence statistics audited against scikit-allel**

  - Hudson FST: switched from average-of-ratios to ratio-of-averages
  - Weir-Cockerham FST: corrected haploid variance components (h_bar=0), fixed s_squared divisor
  - Nei FST: switched to ratio-of-averages
  - All FST estimators and Da: removed incorrect clipping of negative values to zero
  - All divergence stats now match scikit-allel at machine precision

* **Diversity statistics fixes**

  - ``theta_h()`` / ``theta_l()``: fixed inclusion of non-segregating sites (fixed derived and monomorphic)
  - ``tajimas_d()``: fixed harmonic mean float-to-int truncation (199.999 -> 199 instead of 200)
  - Same fix in ``normalized_fay_wus_h()`` and ``zeng_e()`` via ``_effective_n_and_S()``

Data Loading and I/O
~~~~~~~~~~~~~~~~~~~~

* **Sample names stored** in ``HaplotypeMatrix`` -- ``from_vcf()`` now preserves
  VCF sample names as ``self.samples``, eliminating the need to re-read VCFs
  for population assignment.

* **Region queries** -- ``from_vcf(region='chr1:1M-2M')`` loads a genomic subset
  via tabix. ``from_vcf(samples=[...])`` loads a sample subset.

* **Zarr support** -- ``to_zarr()`` / ``from_zarr()`` for fast columnar data
  storage. Significantly faster than VCF for repeated loading.

* **Bio2zarr VCZ format** -- ``from_zarr()`` auto-detects VCZ (bio2zarr),
  scikit-allel, and chromosome-grouped zarr layouts. ``vcf_to_zarr()`` provides
  multicore VCF-to-zarr conversion via bio2zarr. ``to_zarr()`` now writes VCZ
  format by default. ``GenotypeMatrix`` gains ``from_zarr()`` / ``to_zarr()``.

* **Population file loading** -- ``load_pop_file('pops.txt')`` assigns populations
  from a tab-delimited file using stored sample names.

* **API consistency** -- all public functions now return NumPy arrays (not CuPy).
  Users no longer need to call ``.get()`` on results.

Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~

* **Fused windowed analysis** via ``windowed_analysis()``

  All supported statistics now route through fused CUDA kernels when using
  non-overlapping windows with ``missing_data='include'``. A single kernel
  launch processes all windows in parallel.

  Supported fused windowed stats:

  - Single-pop: ``pi``, ``theta_w``, ``tajimas_d``, ``segregating_sites``, ``singletons``
  - Two-pop: ``fst``, ``fst_hudson``, ``fst_wc``, ``dxy``, ``da``
  - Selection: ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``, ``mean_nsl``

  Windowed speedups at 5.3M variants (100kb windows, 200 haplotypes):

  - pi + theta_w + tajimas_d: **60x** vs allel
  - All 5 single-pop stats: **13-22x** vs allel
  - All 12 stats together: **0.66s** total

* **Fused iHS kernel**: one thread block per focal variant with bitmask-based
  pair tracking and block-level EHH reductions. Eliminates O(n_variants^2)
  histogram memory. Single kernel launch, no chunking.

  - iHS: **6.7x** faster than allel at 255k variants
  - Memory: O(n_variants) instead of O(n_variants^2)

* **Vectorized haplotype operations** via GPU dot-product hashing

  - ``haplotype_diversity()``: 37s -> 0.05s (**780x** internal speedup)
  - ``garud_h()``: 9s -> 3ms (**3000x** internal speedup)
  - ``moving_garud_h()``: 7.5s -> 50ms (**150x**), uses cumulative prefix sums
    for O(1) per-window hash computation

* **SFS optimization**: int32 bincount (30x faster than int64), ``cp.maximum``
  for missing data clamping

New Features
~~~~~~~~~~~~

* **New windowed statistics**

  - ``da`` (net divergence) in fused two-pop kernel
  - ``fst_wc`` (Weir-Cockerham) in fused two-pop kernel
  - ``fst_hudson`` as explicit alias for ``fst`` in fused path
  - ``garud_h1``/``garud_h12``/``garud_h123``/``garud_h2h1`` via fused kernel
    with shared-memory odd-even sort
  - ``mean_nsl`` via per-site nSL computation + scatter binning

* **Cross-validation script** (``tests/validate_against_allel.py``)

  Standalone script comparing 31 statistics against scikit-allel using real
  Ag1000G data (1M variants, 200 haplotypes). Includes timing comparison table.
  Run with: ``pixi run python tests/validate_against_allel.py``

* **Example notebook** (``examples/pg_gpu_tour.ipynb``)

  Interactive tour of all major features using Anopheles gambiae X chromosome
  data: VCF loading, GPU transfer, SFS, diversity, divergence, windowed scans,
  LD, PCA, and selection scans.

Performance Summary (vs scikit-allel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Windowed statistics at 5.3M variants, 100kb windows:

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


v0.1.0
------

New Modules
~~~~~~~~~~~

* **Selection Scan Statistics** (``pg_gpu.selection``)

  - ``ihs()`` - Integrated Haplotype Score with CUDA kernel acceleration
  - ``xpehh()`` - Cross-population Extended Haplotype Homozygosity
  - ``nsl()`` - Number of Segregating sites by Length
  - ``xpnsl()`` - Cross-population nSL
  - ``ehh_decay()`` - Extended Haplotype Homozygosity decay
  - ``garud_h()`` / ``moving_garud_h()`` - Garud's H1/H12/H123/H2H1 statistics
  - ``standardize()`` / ``standardize_by_allele_count()`` - Score normalization
  - 10-45x speedup over scikit-allel at typical dataset sizes

* **Site Frequency Spectrum** (``pg_gpu.sfs``)

  - ``sfs()`` / ``sfs_folded()`` / ``sfs_scaled()`` / ``sfs_folded_scaled()``
  - ``joint_sfs()`` / ``joint_sfs_folded()`` / ``joint_sfs_scaled()`` / ``joint_sfs_folded_scaled()``
  - ``fold_sfs()`` / ``fold_joint_sfs()`` - Folding utilities
  - Scaling utilities for neutral expectation comparison

* **Admixture / F-Statistics** (``pg_gpu.admixture``)

  - ``patterson_f2()`` - Branch length between populations
  - ``patterson_f3()`` - Three-population admixture test
  - ``patterson_d()`` - ABBA-BABA (D-statistic / F4)
  - ``moving_patterson_f3()`` / ``moving_patterson_d()`` - Windowed variants
  - ``average_patterson_f3()`` / ``average_patterson_d()`` - Block-jackknife with SE

New Functions
~~~~~~~~~~~~~

* **LD Statistics**

  - ``r()`` / ``r_squared()`` - Pearson correlation from haplotype counts
  - ``locate_unlinked()`` - GPU-accelerated LD pruning
  - ``windowed_r_squared()`` - Percentile of r-squared in distance bins

* **Diversity Statistics**

  - ``heterozygosity_expected()`` - Gene diversity per variant
  - ``heterozygosity_observed()`` - Observed heterozygosity (diploid)
  - ``inbreeding_coefficient()`` - Wright's F per variant

* **Divergence Statistics**

  - ``pbs()`` - Population Branch Statistic (normalized PBSn1)

* **Dimensionality Reduction** (``pg_gpu.decomposition``)

  - ``pca()`` - GPU-accelerated PCA via SVD (up to 56x faster than allel)
  - ``randomized_pca()`` - Truncated SVD approximation for large datasets
  - ``pairwise_distance()`` - GPU-accelerated with memory-safe batching
  - ``pcoa()`` - Principal Coordinate Analysis

* **GPU-Native Windowed Statistics** (``pg_gpu.windowed_analysis``)

  - ``windowed_statistics()`` - Compute pi, theta_w, tajimas_d, FST, Dxy across all windows in one GPU pass using scatter_add aggregation
  - ``windowed_statistics_fused()`` - Fused CUDA kernel variant with one thread block per window
  - Up to 4.3x speedup over allel for windowed FST at scale

* **Visualization** (``pg_gpu.plotting``)

  - ``plot_sfs()``, ``plot_joint_sfs()`` - SFS bar plots and heatmaps
  - ``plot_pairwise_ld()``, ``plot_ld_decay()`` - LD heatmap and decay curves
  - ``plot_pca()`` - PCA scatter with population labels
  - ``plot_pairwise_distance()`` - Distance matrix heatmap
  - ``plot_windowed()``, ``plot_windowed_panel()`` - Genome-wide windowed stat plots
  - ``plot_haplotype_frequencies()``, ``plot_variant_locator()`` - Haplotype and variant viz
  - Built on matplotlib + seaborn

* **GenotypeMatrix** (``pg_gpu.genotype_matrix``)

  - Diploid genotype storage (0/1/2 alt allele counts)
  - Conversion to/from HaplotypeMatrix, VCF loading

* **diploSHIC-Derived Statistics**

  - ``zns()``, ``omega()`` - Kelly's ZnS and Kim & Nielsen's Omega (GPU prefix-sum Omega)
  - ``mu_ld()`` - Haplotype pattern exclusivity (RAiSD)
  - ``mu_var()``, ``mu_sfs()`` - SNP density and SFS edge fraction (RAiSD)
  - ``theta_h()``, ``max_daf()``, ``haplotype_count()``, ``daf_histogram()``
  - ``dist_var()``, ``dist_skew()``, ``dist_kurt()`` - Pairwise distance moments
  - Diploid variants: ``zns_diploid()``, ``omega_diploid()``, ``garud_h_diploid()``, ``diplotype_frequency_spectrum()``, ``daf_histogram_diploid()``

Infrastructure
~~~~~~~~~~~~~~

* Migrated to pixi for unified environment management
* Shared ``_utils.py`` module for population extraction
* Comprehensive validation test suite against scikit-allel

