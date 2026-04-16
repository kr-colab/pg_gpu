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

Bootstrap CI on Tajima's D under a Sweep
-----------------------------------------

``examples/sweep_tajimas_d_bootstrap.py`` simulates a 10 Mb chromosome
with msprime's ``SweepGenicSelection`` targeting fixation at the midpoint,
computes windowed Tajima's D, and uses ``block_bootstrap`` to obtain 95%
confidence intervals for the mean Tajima's D in the sweep-local region,
and the distal region. Under a completed sweep the
sweep-local CI excludes zero; the distal CI
brackets the neutral expectation.

.. code-block:: bash

   pixi run python examples/sweep_tajimas_d_bootstrap.py
   pixi run python examples/sweep_tajimas_d_bootstrap.py --seed 7 --n-replicates 5000

Admixture Detection (end-to-end)
--------------------------------

``examples/admixture_detection.py`` is a self-contained demo: it
simulates two 4-population msprime tree sequences (one null, one with
a 10% C -> B admixture pulse), loads each into a ``HaplotypeMatrix`` via
``from_ts``, and computes ``average_patterson_d`` with a block-jackknife
95% CI. The null scenario's CI overlaps zero; the admixed scenario's
excludes it. A two-panel figure shows the per-block D distribution and
point estimates with CIs.

.. code-block:: bash

   pixi run python examples/admixture_detection.py
   pixi run python examples/admixture_detection.py --length 20_000_000 --samples 20

Accessibility Mask (end-to-end)
-------------------------------

``examples/accessibility_mask.py`` demonstrates what an accessibility
mask actually does to windowed statistics. It simulates a 1 Mb
chromosome with a 200 kb block of 100x lower mutation rate (a stand-in
for a low-callability exon), then computes windowed π twice — once
without a mask, once with the exon flagged inaccessible via an
in-memory numpy bool array. The unmasked trace shows a misleading dip
over the low-μ region; the masked trace drops those windows entirely
(NaN → visual gap) and the flanking π sits at its expected 4·Ne·μ
value. A two-panel figure with the excluded region shaded makes the
contrast visible at a glance.

.. code-block:: bash

   pixi run python examples/accessibility_mask.py
   pixi run python examples/accessibility_mask.py --window 20_000

Local PCA / lostruct (end-to-end)
---------------------------------

``examples/local_pca.py`` is a self-contained demo of the GPU lostruct
pipeline. It simulates a 10 Mb chromosome under ``msprime`` with
``SweepGenicSelection`` at the midpoint (final sweep frequency 0.5, i.e.
a partial / incomplete sweep), runs per-window local PCA, computes the
Frobenius pairwise distance between windows, and reduces with classical
MDS. 1D k-means (k=3) on the resulting MDS1 values partitions windows
into neutral / linked / sweep regimes, ordered by how far each
cluster's centroid sits from the chromosome-wide median. ``corners()``
picks the extreme windows on the MDS embedding, which coincide with
the sweep region. A two-panel figure shows the MDS scatter (colored by
regime) on the left and shared-x stacks of MDS1 and Garud H12 along
the chromosome on the right — the sweep cluster and H12 peak both
land on the sweep focal site.

.. code-block:: bash

   pixi run python examples/local_pca.py
   pixi run python examples/local_pca.py --window 300 --seed 7
   pixi run python examples/local_pca.py --s 0.05 --end-freq 0.3

Python API:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, windowed_analysis
   from pg_gpu.decomposition import pc_dist, pcoa, corners

   # `local_pca` statistic routes through the GPU-batched eigh pipeline
   result = windowed_analysis(
       hm, window_size=500, step_size=250,
       statistics=['local_pca'], window_type='snp', k=2)

   dist = pc_dist(result, npc=2, normalize='L1')
   mds, _ = pcoa(dist, n_components=2)
   extremes = corners(mds, prop=0.05, k=3)

   # With jackknife SE (shares per-window matrix prep with local_pca)
   result = windowed_analysis(
       hm, window_size=500, step_size=250,
       statistics=['local_pca', 'local_pca_jackknife'],
       window_type='snp', k=2, n_blocks=10)
   result.jackknife_se  # (n_windows, k) SE array

LD Block Partitioning (end-to-end)
----------------------------------

``examples/ld_blocks.py`` partitions a chromosome into LD blocks using
pg_gpu's GPU-fast pairwise r² as the input. It simulates a 1 Mb
chromosome with two recombination hotspots (so ground truth = 3 blocks),
computes the full r² matrix on the GPU, and locates block boundaries by
scanning a *bridging score*: at each candidate breakpoint the mean r² is
computed across a sliding (left-window, right-window) pair. The score is
high inside a block and dips at hotspots; ``scipy.signal.find_peaks``
then identifies the dips. A three-panel figure shows the r² heatmap with
detected boundaries, the bridging-score trace, and the simulated
recombination map for comparison.

.. code-block:: bash

   pixi run python examples/ld_blocks.py
   pixi run python examples/ld_blocks.py --window 200 --max-score 0.02

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

Utility Scripts
---------------

The top-level ``utils/`` directory contains standalone command-line scripts:

- ``vcf_to_zarr.py`` — convert a bgzipped VCF (or BCF) to a VCZ-format zarr store
  using ``HaplotypeMatrix.vcf_to_zarr``. Takes ``--workers`` to control parallelism.
- ``genome_scan.py`` — end-to-end genome scan workflow. Loads VCF or zarr data,
  optionally assigns populations from a tab-delimited file, computes windowed
  diversity / divergence / Garud's H and scalar summaries on the GPU, and writes
  a multi-panel scan figure (PDF).

Run either with ``pixi run python utils/<script>.py --help`` for usage.

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
