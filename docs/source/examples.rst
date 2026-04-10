Examples
========

Complete Workflow
-----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, ld_statistics, diversity, selection
   import numpy as np

   # Load VCF data
   h = HaplotypeMatrix.from_vcf("example.vcf")

   # Define populations
   h.sample_sets = {
       "CEU": [0, 1, 2, 3, 4],
       "YRI": [5, 6, 7, 8, 9]
   }

   # Diversity
   pi_ceu = diversity.pi(h, population="CEU")
   pi_yri = diversity.pi(h, population="YRI")

   # Divergence
   from pg_gpu import divergence
   fst = divergence.fst(h, "CEU", "YRI")

   # LD
   counts, n_valid = h.tally_gpu_haplotypes()
   r2 = ld_statistics.r_squared(counts, n_valid=n_valid)

   # Selection scans
   ihs_scores = selection.ihs(h, population="CEU")

Two-Population LD
-----------------

.. code-block:: python

   # Memory-efficient chunked computation
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=np.array([0, 1000, 5000, 10000, 50000]),
       pop1="CEU",
       pop2="YRI",
       chunk_size='auto'
   )

Batch Statistics
----------------

.. code-block:: python

   # Multiple LD statistics in one call
   results = ld_statistics.compute_ld_statistics(
       counts,
       statistics=['dd', 'dz', 'pi2', 'r_squared'],
   )

Integration with moments
------------------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   h = HaplotypeMatrix.from_vcf("data.vcf")
   h.sample_sets = {"pop1": list(range(10))}

   # Compute LD statistics with GPU acceleration
   # chunk_size='auto' adapts to available GPU memory
   ld_stats = h.compute_ld_statistics_gpu_single_pop(
       bp_bins=[0, 1000, 5000],
       chunk_size='auto'
   )

For a complete end-to-end workflow using ``pg_gpu`` as a drop-in replacement for
``moments.LD.Parsing.compute_ld_statistics()``, see
``examples/moments_integration_demo.py``.
Full details in :doc:`moments_integration`.

LD Pruning
----------

.. code-block:: python

   # Find variants in linkage equilibrium
   unlinked = h.locate_unlinked(size=100, step=20, threshold=0.1)
   h_pruned = h.get_subset(np.where(unlinked)[0])
   print(f"Kept {np.sum(unlinked)} of {h.num_variants} variants")

   # Windowed r-squared decay
   bins = np.arange(0, 100001, 1000)
   r2_decay, counts = h.windowed_r_squared(bins, percentile=50)

Selection Scan Pipeline
-----------------------

.. code-block:: python

   from pg_gpu import selection

   # iHS with standardization by allele count
   ihs_raw = selection.ihs(h)

   # Get allele counts for binned standardization
   dac = np.sum(h.haplotypes.get(), axis=0)
   ihs_std, bins = selection.standardize_by_allele_count(ihs_raw, dac)

   # Cross-population scans
   xpehh_scores = selection.xpehh(h, "CEU", "YRI")
   xpehh_std = selection.standardize(xpehh_scores)

   # Garud's H in sliding windows
   h1, h12, h123, h2_h1 = selection.moving_garud_h(h, size=200, step=50)

SFS and Admixture
-----------------

.. code-block:: python

   from pg_gpu import sfs, admixture

   # Joint SFS
   jsfs = sfs.joint_sfs(h, "CEU", "YRI")

   # Patterson's D with block-jackknife
   d, se, z, vb, vj = admixture.average_patterson_d(
       h, "popA", "popB", "popC", "popD", blen=100
   )
   print(f"D = {d:.4f}, SE = {se:.4f}, Z = {z:.2f}")

PBS (Population Branch Statistic)
---------------------------------

.. code-block:: python

   from pg_gpu import divergence

   # Detect selection in pop1 relative to pop2 and pop3
   pbs_vals = divergence.pbs(h, "pop1", "pop2", "pop3", window_size=50)

   # Un-normalized PBS
   pbs_raw = divergence.pbs(h, "pop1", "pop2", "pop3",
                            window_size=50, normed=False)

PCA and Dimensionality Reduction
---------------------------------

.. code-block:: python

   from pg_gpu import decomposition

   # GPU-accelerated PCA (up to 56x faster than allel)
   coords, var_ratio = decomposition.pca(h, n_components=10,
                                          scaler='patterson')

   # Randomized PCA for very large datasets
   coords, var_ratio = decomposition.randomized_pca(
       h, n_components=10, random_state=42)

   # Pairwise genetic distance (batched for memory safety)
   dist = decomposition.pairwise_distance(h, metric='euclidean')

   # PCoA from distance matrix
   coords, var_ratio = decomposition.pcoa(dist, n_components=5)

   # Population-specific PCA
   coords_ceu, _ = decomposition.pca(h, n_components=5, population="CEU")

GPU-Native Windowed Statistics
------------------------------

Compute multiple statistics across thousands of windows without Python loops:

.. code-block:: python

   from pg_gpu.windowed_analysis import windowed_statistics
   import numpy as np

   # Define windows across 1Mb region
   bp_bins = np.arange(0, 1_000_001, 10_000)  # 100 windows of 10kb

   # Compute 4 diversity stats in one GPU pass
   result = windowed_statistics(
       h, bp_bins,
       statistics=('pi', 'theta_w', 'tajimas_d', 'segregating_sites')
   )

   # Results are numpy arrays, one value per window
   print(f"Mean pi: {np.nanmean(result['pi']):.6f}")
   print(f"Mean Tajima's D: {np.nanmean(result['tajimas_d']):.3f}")

   # Windowed FST between populations
   result = windowed_statistics(
       h, bp_bins,
       statistics=('pi', 'fst', 'dxy'),
       pop1='CEU', pop2='YRI'
   )

Missing Data
------------

.. code-block:: python

   from pg_gpu import diversity

   # Check missing data summary
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['missing_freq_overall']:.1%}")

   # Different strategies
   pi_include = diversity.pi(h, missing_data='include')
   pi_exclude = diversity.pi(h, missing_data='exclude')

   # LD statistics handle missing data automatically
   counts, n_valid = h.tally_gpu_haplotypes()
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
