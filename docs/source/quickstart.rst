Quick Start Guide
=================

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   # From VCF file (sample names are stored automatically)
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

Fast Reloading with Zarr
~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets, save to Zarr format after the first VCF load. Subsequent
loads are significantly faster.

.. code-block:: python

   # First time: load from VCF, save as Zarr
   h = HaplotypeMatrix.from_vcf("data.vcf.gz")
   h.to_zarr("data.zarr")

   # Subsequent runs: much faster
   h = HaplotypeMatrix.from_zarr("data.zarr")

   # Region queries work on Zarr too
   h = HaplotypeMatrix.from_zarr("data.zarr", region="chr1:1000000-2000000")

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

Divergence Statistics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import divergence

   fst = divergence.fst(h, 'pop1', 'pop2')
   dxy_val = divergence.dxy(h, 'pop1', 'pop2')

   # Population Branch Statistic (3 populations)
   pbs_vals = divergence.pbs(h, 'pop1', 'pop2', 'pop3', window_size=50)

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

.. code-block:: python

   from pg_gpu import admixture

   # Patterson's F2 (branch length)
   f2 = admixture.patterson_f2(h, 'pop1', 'pop2')

   # Patterson's F3 (admixture test)
   T, B = admixture.patterson_f3(h, 'test_pop', 'source1', 'source2')
   f3_star = np.nansum(T) / np.nansum(B)

   # Patterson's D (ABBA-BABA)
   num, den = admixture.patterson_d(h, 'popA', 'popB', 'popC', 'popD')

   # Block-jackknife with standard error
   d, se, z, vb, vj = admixture.average_patterson_d(
       h, 'popA', 'popB', 'popC', 'popD', blen=100
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
   # results is a DataFrame with columns: window_start, window_stop,
   # n_variants, pi, theta_w, tajimas_d

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

For advanced usage with custom bin edges, use ``windowed_statistics_fused()``
directly:

.. code-block:: python

   from pg_gpu.windowed_analysis import windowed_statistics_fused

   result = windowed_statistics_fused(
       h, bp_bins=[0, 10000, 20000, 30000, 40000, 50000],
       statistics=('pi', 'theta_w', 'tajimas_d'),
   )

Diploid Data
~~~~~~~~~~~~

Use ``GenotypeMatrix`` for diploid genotypes (0/1/2). Many functions
auto-dispatch based on input type:

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

Achaz Framework
~~~~~~~~~~~~~~~~

The Achaz (2009) framework treats all frequency-spectrum-based theta estimators
as linear combinations of the site frequency spectrum (SFS). Compute the SFS
once on GPU, then derive all estimators as trivial dot products. This is
faster when computing multiple statistics and enables custom estimators.

.. code-block:: python

   from pg_gpu import FrequencySpectrum

   # Build SFS from haplotype data (one GPU pass)
   fs = FrequencySpectrum(h, population="pop1")

   # Standard theta estimators
   fs.theta("pi")          # nucleotide diversity (Tajima 1983)
   fs.theta("watterson")   # Watterson's theta (Watterson 1975)
   fs.theta("theta_h")     # Fay & Wu's theta_H (Fay & Wu 2000)
   fs.theta("theta_l")     # theta_L (Zeng et al. 2006)
   fs.theta("eta1")        # singleton-based theta (Fu & Li 1993)

   # Neutrality tests (with proper variance)
   fs.tajimas_d()                    # Tajima's D
   fs.fay_wu_h()                     # Fay & Wu's H (unnormalized)
   fs.fay_wu_h(normalized=True)      # Fay & Wu's H* (Zeng et al. 2006)
   fs.zeng_e()                       # Zeng's E

   # All at once
   fs.all_thetas()   # dict of 8 theta estimators
   fs.all_tests()    # dict of 4 neutrality tests

   # Custom weight vector: any function w(n) -> array
   def rare_variant_weights(n):
       import numpy as np
       w = np.zeros(n + 1)
       w[1:4] = 1.0  # weight only singletons, doubletons, tripletons
       return w / np.sum(w[1:n])
   fs.theta(rare_variant_weights)

   # SFS projection for missing data (Gutenkunst et al. 2009)
   fs_projected = fs.project(target_n=50)
   fs_projected.theta("pi")

   # Batch computation via diversity module
   from pg_gpu import diversity
   stats = diversity.diversity_stats_fast(h, population="pop1")
   # Returns all 12 statistics in one call

Missing Data
~~~~~~~~~~~~

Missing data is encoded as -1 and handled automatically:

.. code-block:: python

   # Different missing data strategies
   pi_include = diversity.pi(h, missing_data='include')
   pi_exclude = diversity.pi(h, missing_data='exclude')
   pi_ignore = diversity.pi(h, missing_data='ignore')
