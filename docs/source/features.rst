Features
========

pg_gpu provides GPU-accelerated computation of population genetics statistics
using CuPy. All statistics return NumPy arrays and handle missing data
automatically. Below is a comprehensive catalog of every implemented statistic.

Diversity Statistics
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``pi``
     - Nucleotide diversity
     - Nei & Li (1979)
   * - ``theta_w``
     - Watterson's theta
     - Watterson (1975)
   * - ``tajimas_d``
     - Tajima's D neutrality test
     - Tajima (1989)
   * - ``fay_wus_h``
     - Fay & Wu's H (excess high-frequency derived alleles)
     - Fay & Wu (2000)
   * - ``normalized_fay_wus_h``
     - Normalized H (H*)
     - Zeng et al. (2006)
   * - ``theta_h``
     - Fay & Wu's theta_H
     - Fay & Wu (2000)
   * - ``theta_l``
     - Theta_L
     - Zeng et al. (2006)
   * - ``zeng_e``
     - Zeng's E neutrality test
     - Zeng et al. (2006)
   * - ``zeng_dh``
     - Zeng's DH joint test
     - Zeng et al. (2006)
   * - ``segregating_sites``
     - Count of segregating sites
     -
   * - ``singleton_count``
     - Count of singletons
     -
   * - ``haplotype_diversity``
     - Haplotype diversity (1 - sum of squared frequencies)
     -
   * - ``haplotype_count``
     - Number of distinct haplotypes
     -
   * - ``heterozygosity_expected``
     - Expected heterozygosity (gene diversity) per variant
     -
   * - ``heterozygosity_observed``
     - Observed heterozygosity per variant
     -
   * - ``inbreeding_coefficient``
     - Wright's F per variant
     - Wright (1951)
   * - ``allele_frequency_spectrum``
     - Allele frequency spectrum
     -
   * - ``max_daf``
     - Maximum derived allele frequency
     -
   * - ``daf_histogram``
     - Derived allele frequency histogram
     -
   * - ``diplotype_frequency_spectrum``
     - Diplotype (multi-locus genotype) frequency spectrum
     -
   * - ``diversity_stats``
     - All core diversity statistics in one call
     -

Divergence Statistics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``fst_hudson``
     - Hudson's FST
     - Hudson et al. (1992)
   * - ``fst_weir_cockerham``
     - Weir & Cockerham's FST (method of moments)
     - Weir & Cockerham (1984)
   * - ``fst_nei``
     - Nei's GST
     - Nei (1973)
   * - ``dxy``
     - Absolute divergence (mean pairwise differences between pops)
     - Nei (1987)
   * - ``da``
     - Net divergence (Dxy minus mean within-pop pi)
     - Nei & Li (1979)
   * - ``pbs``
     - Population Branch Statistic
     - Yi et al. (2010)
   * - ``pairwise_fst``
     - Pairwise FST matrix for multiple populations
     -

Distance-Based Two-Population Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``snn``
     - Nearest-neighbor statistic
     - Hudson (2000)
   * - ``dxy_min``
     - Minimum pairwise distance between populations
     - Geneva et al. (2015)
   * - ``gmin``
     - Gmin ratio (Dxy_min / Dxy_mean)
     - Geneva et al. (2015)
   * - ``dd``
     - Relative minimum divergence (dd1, dd2)
     - Schrider et al. (2018)
   * - ``dd_rank``
     - Rank of minimum between-pop distance in within-pop distribution
     - Schrider et al. (2018)
   * - ``zx``
     - ZnS ratio (within-pop LD / total LD)
     - Schrider et al. (2018)

Linkage Disequilibrium
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``r``
     - Pearson correlation between variant pairs
     -
   * - ``r_squared``
     - Squared correlation (r-squared) between variant pairs
     -
   * - ``dd`` (LD)
     - D-squared (two-locus LD statistic)
     - Ragsdale & Gravel (2019)
   * - ``dz``
     - Dz statistic (multi-population LD)
     - Ragsdale & Gravel (2019)
   * - ``pi2``
     - Two-locus nucleotide diversity
     - Ragsdale & Gravel (2019)
   * - ``zns``
     - Kelly's ZnS (mean pairwise r-squared)
     - Kelly (1997)
   * - ``omega``
     - Kim & Nielsen's Omega (partitioned LD)
     - Kim & Nielsen (2004)
   * - ``mu_ld``
     - Haplotype pattern exclusivity (RAiSD LD component)
     - Alachiotis & Pavlidis (2018)

Selection Scans
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``ihs``
     - Integrated Haplotype Score
     - Voight et al. (2006)
   * - ``nsl``
     - Number of Segregating Sites by Length
     - Ferrer-Admetlla et al. (2014)
   * - ``xpehh``
     - Cross-population Extended Haplotype Homozygosity
     - Sabeti et al. (2007)
   * - ``xpnsl``
     - Cross-population nSL
     - Szpiech et al. (2021)
   * - ``garud_h``
     - Garud's H1, H12, H123, H2/H1
     - Garud et al. (2015)
   * - ``moving_garud_h``
     - Garud's H in moving windows
     - Garud et al. (2015)
   * - ``ehh_decay``
     - Extended Haplotype Homozygosity decay
     - Sabeti et al. (2002)

Site Frequency Spectrum
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``sfs``
     - Unfolded SFS
     -
   * - ``sfs_folded``
     - Folded SFS (minor allele counts)
     -
   * - ``sfs_scaled``
     - Scaled unfolded SFS
     -
   * - ``sfs_folded_scaled``
     - Scaled folded SFS
     -
   * - ``joint_sfs``
     - Joint SFS (two populations)
     -
   * - ``joint_sfs_folded``
     - Folded joint SFS
     -
   * - ``joint_sfs_scaled``
     - Scaled joint SFS
     -
   * - ``joint_sfs_folded_scaled``
     - Scaled folded joint SFS
     -
   * - ``fold_sfs``
     - Fold an unfolded SFS
     -
   * - ``fold_joint_sfs``
     - Fold a joint SFS
     -

Admixture and F-Statistics
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``patterson_f2``
     - F2 branch length between two populations
     - Patterson et al. (2012)
   * - ``patterson_f3``
     - F3 admixture test
     - Patterson et al. (2012)
   * - ``patterson_d``
     - Patterson's D (ABBA-BABA)
     - Patterson et al. (2012)
   * - ``moving_patterson_f3``
     - Windowed F3
     - Patterson et al. (2012)
   * - ``moving_patterson_d``
     - Windowed D
     - Patterson et al. (2012)
   * - ``average_patterson_f3``
     - F3 with block-jackknife standard error
     - Patterson et al. (2012)
   * - ``average_patterson_d``
     - D with block-jackknife standard error
     - Patterson et al. (2012)

Resampling (Block Jackknife and Bootstrap)
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``block_jackknife``
     - Delete-1 block jackknife standard error; supports unequal block sizes
     - Busing et al. (1999)
   * - ``block_bootstrap``
     - Block bootstrap standard error and replicate distribution
     - Efron & Tibshirani (1993)

Both operate on pre-binned per-block values and a user-supplied statistic,
so any scalar aggregate (genome-wide mean Tajima's D, per-population :math:`\pi`,
ratio-of-sums estimators like normed F3 / D) can get a calibrated
standard error / CI with a single call.

FrequencySpectrum (Power-User SFS Interface)
---------------------------------------------

The ``FrequencySpectrum`` class provides direct access to SFS-based estimation
for custom weight functions, SFS projection, and the general Achaz (2009)
variance framework.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Method
     - Description
     - Reference
   * - ``FrequencySpectrum.theta``
     - Any theta estimator as weighted SFS dot product
     - Achaz (2009)
   * - ``FrequencySpectrum.neutrality_test``
     - Generalized neutrality test from any two theta estimators
     - Achaz (2009)
   * - ``FrequencySpectrum.project``
     - SFS projection via hypergeometric sampling
     - Gutenkunst et al. (2009)

Built-in estimators: ``pi``, ``watterson``, ``theta_h``, ``theta_l``,
``eta1``, ``eta1_star``, ``minus_eta1``, ``minus_eta1_star``. Custom weight
functions are also supported.

Dimensionality Reduction and Distance
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``pca``
     - Principal Component Analysis (GPU-accelerated SVD)
     - Patterson et al. (2006)
   * - ``randomized_pca``
     - Randomized PCA (truncated SVD approximation)
     - Halko et al. (2011)
   * - ``pairwise_distance``
     - Pairwise genetic distance (Euclidean, cityblock, etc.)
     -
   * - ``pcoa``
     - Principal Coordinate Analysis (classical MDS)
     -
   * - ``local_pca``
     - Per-window PCA (lostruct); GPU-batched ``eigh`` over stacked per-window Gram matrices
     - Li & Ralph (2019)
   * - ``local_pca_jackknife``
     - Delete-1 block jackknife standard error of local PCs (batched)
     - Li & Ralph (2019)
   * - ``pc_dist``
     - Frobenius distance between per-window low-rank covariance reps
     - Li & Ralph (2019)
   * - ``corners``
     - Extreme-cluster selection in a 2D MDS embedding (Welzl MEC)
     - Li & Ralph (2019)

Relatedness and Kinship
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``grm``
     - Genetic Relationship Matrix
     - Yang et al. (2011)
   * - ``ibs``
     - Pairwise Identity by State proportions
     -

Distance Distribution Statistics
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``pairwise_diffs``
     - Pairwise Hamming distances (haploid or diploid)
     -
   * - ``dist_var``
     - Variance of pairwise distance distribution
     - Schrider et al. (2018)
   * - ``dist_skew``
     - Skewness of pairwise distance distribution
     - Schrider et al. (2018)
   * - ``dist_kurt``
     - Excess kurtosis of pairwise distance distribution
     - Schrider et al. (2018)
   * - ``dist_moments``
     - Variance, skewness, and kurtosis in one call
     - Schrider et al. (2018)

Fused Windowed Statistics
-------------------------

The ``windowed_analysis()`` function computes statistics across all genomic
windows in a single GPU pass via fused CUDA kernels.

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Statistic
     - Description
   * - ``pi``
     - Nucleotide diversity per window
   * - ``theta_w``
     - Watterson's theta per window
   * - ``tajimas_d``
     - Tajima's D per window
   * - ``segregating_sites``
     - Segregating site count per window
   * - ``singletons``
     - Singleton count per window
   * - ``fst``
     - Hudson's FST per window
   * - ``fst_wc``
     - Weir-Cockerham FST per window
   * - ``dxy``
     - Absolute divergence per window
   * - ``da``
     - Net divergence per window
   * - ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``
     - Garud's H statistics per window
   * - ``mean_nsl``
     - Mean nSL per window
   * - ``local_pca``
     - Per-window local PCA (lostruct); returns a ``LocalPCAResult`` with eigvals, eigvecs, and window metadata
