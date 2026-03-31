Changelog
=========

v0.1.0 (Current)
-----------------

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

Infrastructure
~~~~~~~~~~~~~~

* Migrated to pixi for unified environment management
* Shared ``_utils.py`` module for population extraction
* Comprehensive validation test suite against scikit-allel

