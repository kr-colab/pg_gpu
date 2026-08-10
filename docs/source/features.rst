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
   * - ``max_daf``
     - Frequency of the most common alternate allele at a site
     -
   * - ``daf_histogram``
     - Histogram of alternate-allele frequencies, scaled to sum to 1, with
       one entry per alternate allele present
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
     - Hudson's FST. The usual choice.
     - Hudson et al. (1992)
   * - ``fst_tskit``
     - Same ingredients as Hudson's, combined the way tskit does it.
       Use this if you want to match ``TreeSequence.Fst``.
     -
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
     - Kelly's ZnS (mean pairwise LD); defaults to the unbiased
       :math:`\sigma_D^2` estimator on ``HaplotypeMatrix`` inputs,
       falls back to naive :math:`r^2` for pre-computed arrays.
     - Kelly (1997); Ragsdale & Gravel (2019)
   * - ``omega``
     - Kim & Nielsen's Omega (partitioned LD); same default policy
       as ``zns``.
     - Kim & Nielsen (2004); Ragsdale & Gravel (2019)
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
     - Integrated Haplotype Score. Each alternate allele is scored against
       the reference separately -- see the note below.
     - Voight et al. (2006)
   * - ``nsl``
     - Number of Segregating Sites by Length. Scores each alternate allele
       separately, same as ``ihs``.
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

``ihs`` and ``nsl`` ask whether one allele sits on unusually long
shared haplotypes. When a site has more than one alternate allele,
they score each alternate against the reference separately.

That changes the shape of what you get back. With ordinary two-allele
data you get one score per site, as usual::

   scores = selection.nsl(h)      # shape (n_variants,)

If any site in the matrix has more than two alleles, you get one column
per alternate allele instead::

   scores = selection.nsl(h)      # shape (n_variants, n_alt_alleles)
   scores[:, 0]                   # scores for allele 1
   scores[:, 1]                   # scores for allele 2

An entry is ``NaN`` where that allele is not present at that site, or
where it was filtered out by ``min_maf``. The ``min_maf`` cutoff is
applied to each alternate allele's own frequency, so a common allele is
never missed just because it has a high allele number.

``xpehh``, ``xpnsl``, ``ehh_decay``, and Garud's H compare whole
haplotypes rather than focusing on one allele, so they are unaffected
and always return their usual shape. Windowed ``mean_nsl`` averages all
the scores in a window, across sites and alleles alike.

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
     - Folded SFS (minor allele counts). Returned as ``float64`` -- see
       :ref:`sfs-conventions`.
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
     - Folded joint SFS. Folded once per site using the overall minor
       allele; shape ``(n1 + 1, n2 + 1)``.
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
   * - ``patterson_f4``
     - F4: tests whether four populations fit a given tree. Matches
       ``TreeSequence.f4``. Works on sites with any number of alleles.
     - Patterson et al. (2012)
   * - ``patterson_d``
     - Patterson's D (ABBA-BABA). Only works on two-allele sites; others
       are skipped, and a warning tells you how many.
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
     - PCA with one point per haplotype. Gives every allele its own
       column, so it works with any number of alleles. Takes a
       ``HaplotypeMatrix``.
     -
   * - ``randomized_pca``
     - Faster approximate version of ``pca``
     - Halko et al. (2011)
   * - ``pca_dosage``
     - PCA with one point per individual, counting alternate alleles as
       0, 1, or 2. The classical version, matching scikit-allel. Takes a
       ``GenotypeMatrix``.
     - Patterson et al. (2006)
   * - ``randomized_pca_dosage``
     - Faster approximate version of ``pca_dosage``
     - Halko et al. (2011)
   * - ``pairwise_distance``
     - Genetic distance between each pair of haplotypes, based on how
       many sites they differ at (``euclidean``, ``sqeuclidean``,
       ``cityblock``)
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

There are two PCAs because there are two genuinely different things
people mean by "PCA of genetic data", and they need different input.
``pca`` works on haplotypes and handles any number of alleles;
``pca_dosage`` works on diploid genotypes and is the classical version
you will find in textbooks and in scikit-allel. Each one raises a
``TypeError`` if handed the other's matrix, naming the one you want.

The reason ``pca`` cannot simply use the allele numbers is that they are
arbitrary labels. Whether an allele is called ``2`` or ``3`` carries no
meaning, but arithmetic on those numbers would treat ``3`` as bigger
than ``2``. So ``pca`` gives every allele its own column instead, which
makes the result independent of how the alleles were numbered.

``local_pca``, ``lostruct``, and ``local_pca_jackknife`` use the same
approach as ``pca`` and also take haplotypes. Because of that they no
longer match the R ``lostruct`` package number for number: pg_gpu keeps
one column per allele where R keeps one per site, and the two center
the data differently, which shifts the scale of the eigenvalues.

The directions the components point in still agree with R closely
(correlations above 0.999 in the test suite), so window-to-window
comparisons, the MDS plot, and outlier detection all behave the same.
It is the absolute eigenvalue scale that differs, not the structure you
would read off the plot.

``pairwise_distance`` counts, for each pair of haplotypes, how many
sites they differ at (skipping sites where either is missing). Call
that count :math:`m`. Then ``cityblock`` and ``sqeuclidean`` both
return :math:`m`, and ``euclidean`` returns :math:`\sqrt{m}`. Counting
mismatches this way means the answer does not depend on how the alleles
were numbered. Any other metric raises ``NotImplementedError``.

Relatedness and Kinship
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``genetic_relatedness``
     - How much two groups of samples share alleles, relative to the
       average across groups. Matches
       ``TreeSequence.genetic_relatedness``. Takes a ``HaplotypeMatrix``.
     -
   * - ``grm``
     - Genetic Relationship Matrix (GCTA). Takes a ``GenotypeMatrix``.
     - Yang et al. (2011)
   * - ``ibs``
     - Pairwise Identity by State proportions (PLINK). Takes a
       ``GenotypeMatrix``.
     -

``genetic_relatedness`` is the general-purpose one. It already counts
alleles separately, so it works on multiallelic sites without any
special handling. You choose what a "sample" means by grouping
haplotypes:

* leave the grouping out and every haplotype is its own sample, giving
  a haplotype-by-haplotype matrix;
* group haplotypes in pairs to get relatedness between individuals;
* group them by population to get relatedness between populations.

``grm`` and ``ibs`` are the two classic diploid measures -- the GCTA
relationship matrix and PLINK-style identity by state. Both need a
``GenotypeMatrix`` (or a streaming one). Handing them haplotypes raises
a ``TypeError`` that points you at the alternative. To use them on
haplotype data, convert first::

   gm = GenotypeMatrix.from_haplotype_matrix(h)
   k = grm(gm)

Or use ``genetic_relatedness`` instead, which needs no conversion.

Distance Distribution Statistics
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Function
     - Description
     - Reference
   * - ``pairwise_diffs``
     - Number of sites at which each pair differs, skipping sites where
       either one is missing. The haplotype version compares alleles
       directly, so it works with any number of alleles; the diploid
       version works on two-allele sites only.
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

Biobank-Scale Streaming
-----------------------

A *VCZ store* is a Zarr encoding of a VCF stored on disk: the genotype
matrix is split into compressed chunks, each chunk a small array of
samples by variants. ``pg_gpu`` reads VCZ stores; if your data
is in VCF you can convert it with the bio2zarr tools
(``vcf2zarr explode`` then ``vcf2zarr encode``). The streaming
codepath needs that VCZ layout because it relies on a fast
per-chunk decode, an operation that can't be done on a VCF. See
:doc:`tutorials/biobank_streaming` for the VCF→VCZ conversion
and a worked end-to-end example.

A VCZ store too large for the GPU (tens to hundreds of thousands
of haplotypes) opens via ``HaplotypeMatrix.from_zarr`` /
``GenotypeMatrix.from_zarr`` as a streaming view that walks the
chromosome chunk by chunk; every kernel listed below dispatches
on the streaming object the same way it would on a fully loaded
matrix -- the calling code is identical to the in-memory path.

What runs on a streaming matrix:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Entry point
     - How it streams
   * - ``windowed_analysis``
     - Each chunk's windows are computed on the GPU as the chunk arrives, then rows are concatenated; window grids are aligned to the chunk grid so no window straddles a chunk boundary.
   * - ``sfs.sfs``, ``sfs.joint_sfs``, ``sfs.sfs_folded``, ``sfs.joint_sfs_folded``
     - Each chunk contributes its own per-site frequency-spectrum counts; the chromosome-wide answer is the sum across chunks.
   * - ``sfs.project_joint_sfs``
     - Same per-chunk accumulation, but the joint SFS is projected (via hypergeometric sampling) to a small target grid as it is built, so the full ``(n1+1, n2+1)`` histogram is never materialized.
   * - ``HaplotypeMatrix.compute_ld_statistics_gpu_single_pop`` / ``_two_pops``
     - Variant-pair statistics within ``max_bp_dist`` are summed per chunk into the bp bins. Pairs that fall on opposite sides of a chunk boundary would be missed by naive per-chunk sums, so the last ``max_bp_dist`` of one chunk is carried forward and paired with the start of the next, counted exactly once. Returns moments-LD ``DD``, ``Dz``, ``pi²`` (3 stats single-pop, 15 stats two-pop).
   * - ``relatedness.ibs``, ``relatedness.grm`` (``StreamingGenotypeMatrix``)
     - Streamed along the variant axis; the individual axis is tiled into row blocks. ``(n_ind, n_ind)`` accumulators live on host so the output can exceed GPU memory. ``grm`` is a two-pass operation (first to calculate chromosome-wide allele frequencies, second to accumulate a per-chunk outer product).
   * - ``relatedness.genetic_relatedness`` (``StreamingHaplotypeMatrix``)
     - One pass along the variants, adding each chunk's contribution into an ``(n, n)`` result kept in CPU memory -- the same approach as the streaming GRM and IBS above.
   * - ``StreamingHaplotypeMatrix.materialize(region, sample_subset)``
     - Pulls one sub-region (and optionally a subset of haplotypes) of the chromosome into GPU memory for kernels that need every variant simultaneously -- ``pairwise_r2``, Garud's H, or any custom recipe. 
   * - ``zarr_io.allel_zarr_to_vcz``
     - Streaming converter from scikit-allel layout to VCZ for stores that pre-date bio2zarr.

Fused Windowed Statistics
-------------------------

The ``windowed_analysis()`` function computes statistics across all genomic
windows in a single GPU pass via fused CUDA kernels. That fast path
needs ``missing_data='include'``. With any other setting pg_gpu quietly
falls back to a slower route that produces the same numbers.

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
     - Number of mutations per window (a site with 3 alleles counts as 2)
   * - ``singletons``
     - Number of alternate alleles seen exactly once in the window
   * - ``theta_h``, ``fay_wu_h``
     - Fay & Wu's theta_H and H per window
   * - ``max_daf``
     - Frequency of the most common alternate allele in the window
   * - ``fst``, ``fst_hudson``
     - Hudson's FST per window
   * - ``fst_wc``
     - Weir-Cockerham FST per window (treats data as haploid; see below)
   * - ``dxy``
     - Absolute divergence per window
   * - ``da``
     - Net divergence per window
   * - ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``
     - Garud's H statistics per window
   * - ``haplotype_count``
     - Number of distinct haplotypes per window
   * - ``mean_nsl``
     - Mean nSL per window, averaged over every finite (site, allele) score
   * - ``zns``, ``omega``, ``mu_ld``
     - Per-window LD summaries
   * - ``mu_var``, ``mu_sfs``, ``daf_hist``
     - Features used by RAiSD and diploSHIC. ``daf_hist`` is a histogram
       of alternate-allele frequencies in the window, scaled to sum to 1.
       ``mu_sfs`` is the fraction of variable alleles sitting at the very
       low or very high end of the frequency range, and is 0.0 for a
       window with nothing variable in it. Both need
       ``missing_data='include'``.
   * - ``snp_dist_mean``, ``snp_dist_var``, ``snp_dist_min``, ``snp_dist_max``
     - Inter-SNP spacing summaries per window
   * - ``dist_var``, ``dist_skew``, ``dist_kurt``
     - Moments of the pairwise-distance distribution per window
   * - ``local_pca``
     - Per-window local PCA (lostruct); returns a ``LocalPCAResult`` with eigvals, eigvecs, and window metadata

The rule to expect is that a windowed statistic gives the same answer
as calling the plain function on just that window's variants. Alleles
are counted separately here too, the same as everywhere else. There are
two known exceptions:

* Windowed ``fst_wc`` treats the data as haploid, so it will not match
  the plain ``fst_weir_cockerham``, which treats it as diploid. This
  holds even for two-allele data.
* When data is missing, the windowed neutrality tests (``tajimas_d``,
  ``normalized_fay_wu_h``, ``zeng_e``, ``zeng_dh``) use the full sample
  size in their variance formula, while the plain versions use an
  average of the per-site sample sizes. Without missing data the two
  agree.

One more limit: the windowed kernels can handle at most 8 alleles at a
single site. Anything beyond that is skipped, and a
``MultiallelicCapWarning`` tells you how many sites that affected. DNA
has 4 bases, so in practice this never comes up. The plain
``diversity`` and ``divergence`` functions have no such limit.
