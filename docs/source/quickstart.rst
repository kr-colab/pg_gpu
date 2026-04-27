Quick Start Guide
=================

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

The primary container, ``HaplotypeMatrix``, holds *phased* haplotypes
(one row per haplotype, two rows per diploid sample). When loaded from a
diploid VCF, the two haploid copies of each sample become two rows; the
loader does not check whether your data are actually phased, so if the
VCF contains unphased calls the resulting matrix is best thought of as a
random-phasing of the genotypes. For unphased / diploid-aware analyses,
use ``GenotypeMatrix`` (see "Phased to Unphased" below).

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   # From VCF file (sample names are stored automatically). Diploid VCFs
   # are loaded as 2*n_samples haplotypes -- treated as phased.
   h = HaplotypeMatrix.from_vcf("data.vcf.gz")

   # Load a specific genomic region (requires tabix index)
   h = HaplotypeMatrix.from_vcf("data.vcf.gz", region="chr1:1000000-2000000")

   # Load a subset of samples
   h = HaplotypeMatrix.from_vcf("data.vcf.gz", samples=["ind1", "ind2", "ind3"])

   # From NumPy array (n_haplotypes, n_variants)
   import numpy as np
   hap = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)
   positions = np.array([100, 200, 300])
   h = HaplotypeMatrix(hap, positions, chrom_start=0, chrom_end=400)

Populations can be assigned from a tab-delimited file (sample, pop) or
manually via ``sample_sets``:

.. code-block:: python

   # From a population file (uses stored sample names)
   h.load_pop_file("pops.txt")

   # Or manually
   h.sample_sets = {
       "pop1": [0, 1, 2, 3],
       "pop2": [4, 5, 6, 7]
   }

Working with Zarr
~~~~~~~~~~~~~~~~~~

pg_gpu supports the VCZ zarr format (via `bio2zarr <https://sgkit-dev.github.io/bio2zarr/>`_)
as well as legacy scikit-allel zarr stores. The format is auto-detected on read.

**Converting VCF to Zarr** (recommended for large datasets):

.. code-block:: python

   # Convert VCF to VCZ zarr (uses bio2zarr, supports multicore)
   HaplotypeMatrix.vcf_to_zarr(
       "data.vcf.gz", "data.zarr",
       worker_processes=8,   # parallel conversion
   )

   # Load from zarr (much faster than VCF for repeated access)
   h = HaplotypeMatrix.from_zarr("data.zarr")

**Region queries and multi-chromosome stores:**

.. code-block:: python

   # Region queries work on all zarr layouts
   h = HaplotypeMatrix.from_zarr("data.zarr", region="chr1:1000000-2000000")

   # Chromosome-grouped stores (e.g., Ag1000G) require a region
   h = HaplotypeMatrix.from_zarr("ag1000g.zarr", region="3L:1-10000000")

**Quick save/reload** (for intermediate results):

.. code-block:: python

   # Save to VCZ format (default)
   h = HaplotypeMatrix.from_vcf("data.vcf.gz")
   h.to_zarr("data.zarr", contig_name="chr1")

   # Or legacy scikit-allel format
   h.to_zarr("data.zarr", format="scikit-allel")

LD Statistics
~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import ld_statistics

   # Pairwise r-squared matrix
   r2 = h.pairwise_r2()

   # Haplotype count-based statistics
   counts, n_valid = h.tally_gpu_haplotypes()
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
   r_vals = ld_statistics.r(counts, n_valid=n_valid)
   r2_vals = ld_statistics.r_squared(counts, n_valid=n_valid)

   # LD pruning
   unlinked = h.locate_unlinked(size=100, step=20, threshold=0.1)

   # Windowed r-squared
   result, pair_counts = h.windowed_r_squared(
       bp_bins=[0, 1000, 5000, 10000]
   )

   # Two-population LD
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=[0, 1000, 5000, 10000],
       pop1="pop1", pop2="pop2"
   )

Diversity Statistics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import diversity

   # Basic diversity
   pi_val = diversity.pi(h, span_normalize=True)
   theta = diversity.theta_w(h, span_normalize=True)
   tajd = diversity.tajimas_d(h)

   # Heterozygosity
   he = diversity.heterozygosity_expected(h)
   ho = diversity.heterozygosity_observed(h)
   f = diversity.inbreeding_coefficient(h)

   # Allele frequency spectrum
   afs = diversity.allele_frequency_spectrum(h)

   # Population-specific
   pi_pop1 = diversity.pi(h, population='pop1')

Theta Estimators and Neutrality Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pg_gpu ships eight theta estimators and five neutrality tests. They can
be called individually or batched (a single GPU pass for all requested
statistics):

.. code-block:: python

   from pg_gpu import diversity

   # Individual statistics
   diversity.pi(h, population="pop1")
   diversity.theta_w(h, population="pop1")
   diversity.tajimas_d(h, population="pop1")
   diversity.fay_wus_h(h, population="pop1")

   # Batched (single GPU pass for all)
   stats = diversity.diversity_stats(h, population="pop1",
       statistics=['pi', 'theta_w', 'theta_h', 'theta_l', 'tajimas_d'])

Available estimators: ``pi``, ``theta_w``, ``theta_h``, ``theta_l``,
``eta1``, ``eta1_star``, ``minus_eta1``, ``minus_eta1_star``. Available
tests: ``tajimas_d``, ``fay_wus_h``, ``normalized_fay_wus_h``,
``zeng_e``, ``zeng_dh``.

Divergence Statistics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import divergence

   fst = divergence.fst(h, 'pop1', 'pop2')
   dxy_val = divergence.dxy(h, 'pop1', 'pop2')

   # Population Branch Statistic (3 populations)
   pbs_vals = divergence.pbs(h, 'pop1', 'pop2', 'pop3', window_size=50)

   # Distance-based two-population statistics
   snn = divergence.snn(h, 'pop1', 'pop2')              # Hudson (2000)
   dmin = divergence.dxy_min(h, 'pop1', 'pop2')          # Geneva et al. (2015)
   g = divergence.gmin(h, 'pop1', 'pop2')                # Geneva et al. (2015)
   dd1, dd2 = divergence.dd(h, 'pop1', 'pop2')           # Schrider et al. (2018)
   r1, r2 = divergence.dd_rank(h, 'pop1', 'pop2')        # Schrider et al. (2018)
   zx_val = divergence.zx(h, 'pop1', 'pop2')             # Schrider et al. (2018)

   # Efficient: pre-compute distance matrix once, pass to multiple stats
   dm = divergence.pairwise_distance_matrix(h, 'pop1', 'pop2')
   snn = divergence.snn(h, 'pop1', 'pop2', distance_matrices=dm)
   g = divergence.gmin(h, 'pop1', 'pop2', distance_matrices=dm)
   dd1, dd2 = divergence.dd(h, 'pop1', 'pop2', distance_matrices=dm)

   # Or compute all distance-based stats in one call
   stats = divergence.distance_based_stats(h, 'pop1', 'pop2')

Selection Scans
~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import selection

   # Integrated haplotype score
   ihs_scores = selection.ihs(h)
   ihs_std = selection.standardize(ihs_scores)

   # Cross-population EHH
   xpehh_scores = selection.xpehh(h, 'pop1', 'pop2')

   # nSL (no distance weighting)
   nsl_scores = selection.nsl(h)

   # Garud's H statistics
   h1, h12, h123, h2_h1 = selection.garud_h(h)

   # EHH decay
   ehh = selection.ehh_decay(h)

Site Frequency Spectrum
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import sfs

   # Unfolded and folded SFS
   s = sfs.sfs(h)
   s_folded = sfs.sfs_folded(h)

   # Scaled SFS
   s_scaled = sfs.sfs_scaled(h)

   # Joint SFS (two populations)
   j = sfs.joint_sfs(h, 'pop1', 'pop2')
   j_folded = sfs.joint_sfs_folded(h, 'pop1', 'pop2')

Admixture / F-Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

Patterson's F-statistics are *ratio-of-sums* statistics: each call returns
the per-site numerator and denominator separately, so that resampling
(jackknife / bootstrap) can be applied across sites with the correct
covariance structure. Wrappers like ``average_patterson_d`` do the
combine-and-jackknife step for you.

.. code-block:: python

   from pg_gpu import admixture

   # Patterson's F2 (branch length)
   f2 = admixture.patterson_f2(h, 'pop1', 'pop2')

   # Patterson's F3 (admixture test): per-site numerator and denominator.
   # The point estimate of F3* is sum(numer) / sum(denom).
   f3_numer, f3_denom = admixture.patterson_f3(
       h, 'test_pop', 'source1', 'source2')
   f3_star = np.nansum(f3_numer) / np.nansum(f3_denom)

   # Patterson's D (ABBA-BABA): per-site numerator and denominator.
   d_numer, d_denom = admixture.patterson_d(
       h, 'popA', 'popB', 'popC', 'popD')

   # Block-jackknife wrapper for Patterson's D
   #   d_hat   point estimate
   #   d_se    jackknife standard error
   #   d_z     z-score
   #   v_block per-block leave-one-out values
   #   v_jack  jackknife pseudo-values
   d_hat, d_se, d_z, v_block, v_jack = admixture.average_patterson_d(
       h, 'popA', 'popB', 'popC', 'popD', blen=100
   )

Resampling (Block Jackknife and Bootstrap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``block_jackknife`` and ``block_bootstrap`` are general-purpose resampling
estimators: you bin your data into blocks (typically genomic windows or
chromosome chunks) and pass an array of per-block values plus a
``statistic`` callable that maps blocks to a scalar. They return a point
estimate, a standard error, and either jackknife pseudo-values or
bootstrap replicates.

For *ratio-of-sums* statistics (Patterson's F-statistics, FST, etc.), the
numerator and denominator must be resampled with the same block
assignments. Pass a tuple ``(numer_blocks, denom_blocks)`` and write
``statistic`` to consume both arrays.

.. code-block:: python

   import numpy as np
   from pg_gpu import (block_jackknife, block_bootstrap, windowed_analysis,
                       admixture)

   # 1) Jackknife standard error of a windowed mean. windowed_analysis gives per-window
   #    Tajima's D; treat each window as a block and take the mean.
   df = windowed_analysis(h, window_size=50_000, step_size=25_000,
                          statistics=['tajimas_d'], window_type='bp')
   tajd = df['tajimas_d'].to_numpy()
   tajd = tajd[np.isfinite(tajd)]
   jack_est, jack_se, _ = block_jackknife(tajd, statistic=np.mean)

   # 2) Bootstrap 95% CI on the same mean.
   boot_est, boot_se, reps = block_bootstrap(
       tajd, statistic=np.mean, n_replicates=2000, rng=0)
   lo, hi = np.quantile(reps, [0.025, 0.975])

   # 3) Ratio-of-sums pattern: bin Patterson's D numer/denom into blocks
   #    (here, equal-size chunks of consecutive sites) and bootstrap the
   #    ratio of sums.
   d_numer, d_denom = admixture.patterson_d(
       h, 'popA', 'popB', 'popC', 'popD')
   block_size = 1000
   n_blocks = len(d_numer) // block_size
   numer_blocks = d_numer[:n_blocks * block_size].reshape(n_blocks, -1).sum(1)
   denom_blocks = d_denom[:n_blocks * block_size].reshape(n_blocks, -1).sum(1)
   ratio_est, ratio_se, reps = block_bootstrap(
       (numer_blocks, denom_blocks),
       statistic=lambda numer, denom: np.sum(numer) / np.sum(denom),
       n_replicates=2000, rng=0,
   )

PCA and Distance
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import decomposition

   # PCA (GPU-accelerated SVD)
   coords, var_ratio = decomposition.pca(h, n_components=10)

   # Randomized PCA (faster for large datasets)
   coords, var_ratio = decomposition.randomized_pca(h, n_components=10)

   # Pairwise genetic distance
   dist = decomposition.pairwise_distance(h, metric='euclidean')

   # PCoA from distance matrix
   coords, var_ratio = decomposition.pcoa(dist)

Local PCA / lostruct
~~~~~~~~~~~~~~~~~~~~

GPU port of Li & Ralph's (2019) ``lostruct`` method for detecting
genomic regions where population structure differs from the
chromosome-wide pattern (inversions, introgression, low-recombination
regions). All four steps of the pipeline are available:

.. code-block:: python

   from pg_gpu.decomposition import local_pca, pc_dist, corners, pcoa

   # 1. Per-window top-k eigendecomposition (GPU-batched eigh)
   result = local_pca(h, window_size=500, window_type='snp', k=2)
   # result is a LocalPCAResult with:
   #   result.windows  -> DataFrame (chrom, start, end, center, n_variants, window_id)
   #   result.eigvals  -> (n_windows, k)
   #   result.eigvecs  -> (n_windows, k, n_samples)
   #   result.sumsq    -> (n_windows,)

   # 2. Frobenius distance between windows' low-rank covariance reps
   d = pc_dist(result, npc=2, normalize='L1')   # 'L1' | 'L2' | None

   # 3. MDS on the window-by-window distance matrix
   coords, _ = pcoa(d, n_components=2)

   # 4. Extreme-cluster detection in MDS space (Welzl MEC)
   corner_idx = corners(coords, prop=0.05, k=3)

   # Optional: delete-1 block jackknife standard error of local PCs (standalone)
   from pg_gpu.decomposition import local_pca_jackknife
   se = local_pca_jackknife(h, window_size=500, window_type='snp',
                            k=2, n_blocks=10)  # (n_windows, k)

Both ``local_pca`` and ``local_pca_jackknife`` can be requested through
``windowed_analysis``.  When both are requested together, per-window
matrix preparation is shared for efficiency.  Scalar statistics can be
mixed in the same call; the scalar columns are merged onto
``result.windows``.

.. code-block:: python

   from pg_gpu import windowed_analysis

   result = windowed_analysis(h, window_size=500, window_type='snp',
                              statistics=['local_pca', 'local_pca_jackknife'],
                              k=2, n_blocks=10)
   result.jackknife_se  # (n_windows, k)

For an end-to-end example with a simulated partial sweep, see
``examples/local_pca.py``.

Windowed Statistics (GPU-Native)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute statistics across all genomic windows in a single GPU pass via
fused CUDA kernels. The ``windowed_analysis()`` convenience function
automatically routes through fused kernels when possible.

.. code-block:: python

   from pg_gpu import windowed_analysis

   # Single-population diversity stats (fused: single kernel launch)
   results = windowed_analysis(
       h, window_size=100_000,
       statistics=['pi', 'theta_w', 'tajimas_d']
   )
   # results is a DataFrame with columns: chrom, start, end, center,
   # n_variants, window_id, pi, theta_w, tajimas_d

   # Two-population divergence stats
   results = windowed_analysis(
       h, window_size=100_000,
       statistics=['fst', 'fst_wc', 'dxy', 'da'],
       populations=['pop1', 'pop2']
   )

   # Everything at once -- diversity, divergence, and selection scans
   results = windowed_analysis(
       h, window_size=100_000,
       statistics=['pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                   'fst', 'fst_wc', 'dxy', 'da',
                   'garud_h1', 'garud_h12', 'mean_nsl'],
       populations=['pop1', 'pop2']
   )

Supported fused windowed statistics:

- **Single-pop**: ``pi``, ``theta_w``, ``tajimas_d``, ``segregating_sites``, ``singletons``
- **Two-pop**: ``fst``, ``fst_hudson``, ``fst_wc``, ``dxy``, ``da``
- **Selection**: ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``, ``mean_nsl``
- **Structure**: ``local_pca`` (lostruct); returns a ``LocalPCAResult`` rather than a scalar-stat DataFrame. See the `Local PCA / lostruct`_ section.

For advanced usage with custom bin edges, use ``windowed_statistics_fused()``
directly:

.. code-block:: python

   from pg_gpu.windowed_analysis import windowed_statistics_fused

   result = windowed_statistics_fused(
       h, bp_bins=[0, 10000, 20000, 30000, 40000, 50000],
       statistics=('pi', 'theta_w', 'tajimas_d'),
   )

Phased to Unphased
~~~~~~~~~~~~~~~~~~

Use ``GenotypeMatrix`` for unphased diploid genotypes (entries 0 / 1 / 2
counting alt alleles). Many functions auto-dispatch on input type, so
the same call works on either container:

.. code-block:: python

   from pg_gpu import GenotypeMatrix

   # Convert from haploid (pairs consecutive haplotypes)
   gm = GenotypeMatrix.from_haplotype_matrix(h)

   # Same functions work on both types
   h1, h12, h123, h2h1 = selection.garud_h(gm)  # uses diplotype frequencies
   z = ld_statistics.zns(gm)                      # uses genotype correlation
   hist, edges = diversity.daf_histogram(gm)       # DAF = sum / (2*n_ind)

   # Distance distribution moments
   var, skew, kurt = distance_stats.dist_moments(gm)

Custom Theta via FrequencySpectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom weight functions or hypergeometric SFS projection, build a
``FrequencySpectrum``. Any new theta estimator that can be written as a
weighted sum over the SFS is one ``fs.theta(weight_fn)`` call away.

.. code-block:: python

   from pg_gpu.diversity import FrequencySpectrum

   fs = FrequencySpectrum(h, population="pop1")
   fs.theta(my_custom_weight_fn)
   fs.project(target_n=50).theta("pi")

Missing Data and Accessibility Masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing genotypes are encoded as ``-1`` and handled automatically. Two
modes control how sites with missing data contribute to per-site
statistics:

.. code-block:: python

   # Missing data modes
   pi_include = diversity.pi(h, missing_data='include')   # default: per-site valid data
   pi_exclude = diversity.pi(h, missing_data='exclude')   # only fully genotyped sites

A complementary concern is *site accessibility*: in real data, only
some bases are reliably callable (high-coverage, non-repeat, etc.).
Without an accessibility mask, span-normalized statistics divide by
total genomic span and look spuriously low in low-callability regions.
With a mask attached, both per-site sums and the denominator are
restricted to accessible bases, so statistics report the true rate over
the regions you can actually see.

.. code-block:: python

   import numpy as np
   from pg_gpu import HaplotypeMatrix, diversity, windowed_analysis

   # Simulate or load a chromosome where part of the sequence is hard
   # to call. Build a boolean mask: True = accessible, False = excluded.
   accessible = np.ones(seq_length, dtype=bool)
   accessible[exon_start:exon_end] = False  # mark a region inaccessible

   hm_unmasked = HaplotypeMatrix.from_ts(ts)
   hm_masked   = HaplotypeMatrix.from_ts(ts).set_accessible_mask(accessible)

   # Genome-wide pi: unmasked divides by full sequence length,
   # masked divides by accessible bases only.
   diversity.pi(hm_unmasked)
   diversity.pi(hm_masked)

   # Windowed pi: fully-inaccessible windows return NaN so they don't
   # show up as a misleading dip; partially-accessible windows are
   # normalized by their accessible base count.
   df_unmasked = windowed_analysis(hm_unmasked, window_size=10_000,
                                   statistics=["pi"])
   df_masked   = windowed_analysis(hm_masked,   window_size=10_000,
                                   statistics=["pi"])

For an end-to-end demo (with simulated data and a side-by-side plot)
see ``examples/accessibility_mask.py``. See :doc:`missing_data` for
masks-and-modes details.
