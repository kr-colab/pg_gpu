Local PCA / lostruct
====================

Packaged script: ``examples/local_pca.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/local_pca.py
   pixi run python examples/local_pca.py --window 300 --seed 7
   pixi run python examples/local_pca.py --s 0.05 --end-freq 0.3

Background
----------

Genome-wide PCA gives one picture of population structure averaged
over the whole genome. *Local* PCA computes one PCA per genomic
window and asks where along the genome the local picture deviates
from the chromosome-wide pattern. Outlier regions tend to flag
features whose local genealogy differs from the genome average:

* large structural variants (inversions, segmental duplications);
* recent introgression tracts;
* low-recombination regions where lineage sorting has run further;
* completed or partial selective sweeps.

`Li & Ralph (2019) <https://doi.org/10.1534/genetics.118.301747>`_
introduced this idea in the ``lostruct`` R package. ``pg_gpu`` ports
the four-step pipeline to the GPU, with the per-window
eigendecomposition fused into a single batched ``cupy.linalg.eigh``
call -- the original chokepoint at chromosome scale.

What the script does
--------------------

The packaged demo runs the full pipeline on simulated data where the
"truth" is known and the lostruct outliers are expected to land on
the sweep:

1. **Simulate.** A 10 Mb chromosome under ``msprime`` with
   ``SweepGenicSelection`` at the midpoint, configured as a partial
   sweep that reaches frequency 0.5 (so the local population
   structure differs from the rest of the chromosome but the variant
   itself is still segregating).
2. **Per-window local PCA.** Run ``windowed_analysis`` with the
   ``local_pca`` statistic, which does a single GPU pass over all
   windows and returns a ``LocalPCAResult`` containing the top-:math:`k`
   eigenvalues and eigenvectors per window.
3. **Window-by-window distance.** ``pc_dist`` computes the Frobenius
   distance between windows' low-rank covariance representations,
   producing an :math:`n_{\text{windows}} \times n_{\text{windows}}`
   distance matrix.
4. **MDS embedding.** Classical MDS (``pcoa``) reduces the
   window-by-window distances to a 2-D embedding.
5. **Cluster + outlier detection.** 1-D k-means on MDS1 partitions
   windows into neutral / linked / sweep regimes, ordered by how far
   each cluster's centroid sits from the chromosome-wide median.
   ``corners`` independently picks the extreme windows on the MDS
   embedding -- a complementary, distribution-free approach used by
   the original lostruct paper.

The output figure (``local_pca.png``) plots the MDS scatter coloured
by regime alongside MDS1 and Garud's H12 along the chromosome. The
sweep cluster on the MDS plot and the H12 peak both land on the
sweep focal site, validating the pipeline against an independent
selection summary.

Python API
----------

The whole pipeline in one call:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, lostruct

   res = lostruct(hm, window_size=500, step_size=250,
                  window_type='snp', k=2)

   res.local_pca      # LocalPCAResult: per-window eigvals / eigvecs / windows
   res.distance       # (n_windows, n_windows) Frobenius distance
   res.mds            # (n_windows, n_components) MDS coordinates
   res.corner_indices # (n_per_corner, n_corners) outlier-window indices

Pass ``jackknife=True`` to also compute the block-within-window
jackknife standard error of the per-window eigenvectors -- useful as a
signal-to-noise diagnostic for filtering noisy windows before
downstream analysis. The standard error is computed in the same window
pass as the base eigendecomposition, so this is much cheaper than
calling ``local_pca_jackknife`` separately:

.. code-block:: python

   res = lostruct(hm, window_size=500, step_size=250,
                  window_type='snp', k=2,
                  jackknife=True, n_blocks=10)
   res.jackknife_se   # (n_windows, k) per-PC standard error

If you want intermediate access -- e.g. to swap in a different
distance metric, or recompute MDS at a different ``n_components``
without redoing the per-window eigendecomposition -- you can call the
four constituent functions directly. ``lostruct()`` is a thin wrapper
over them:

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

   # With jackknife standard error (shares per-window matrix prep with local_pca)
   result = windowed_analysis(
       hm, window_size=500, step_size=250,
       statistics=['local_pca', 'local_pca_jackknife'],
       window_type='snp', k=2, n_blocks=10)
   result.jackknife_se  # (n_windows, k) standard error array

Why it's useful as a template
-----------------------------

Local PCA is most useful for chromosomes-scale exploratory work --
"are there any regions of this chromosome that look genuinely
different from the rest?" -- where you don't have a specific
candidate region in mind. A few directions to extend the template:

* **Windowing strategy.** SNP-count windows (``window_type='snp'``)
  give roughly equal-information windows; bp windows
  (``window_type='bp'``) make it easier to interpret distances along
  the genome at the cost of variable per-window SNP counts. Both are
  routed through the same fused kernel.
* **Jackknife standard error.** Add ``'local_pca_jackknife'`` to the
  ``statistics`` list to get per-window standard error on the eigenvalues at
  almost no extra cost (the per-window matrix prep is shared). Useful
  for filtering noisy windows before downstream MDS.
* **Other distance metrics.** ``pc_dist`` accepts
  ``normalize='L1' | 'L2' | None``; sweeping the choice and watching
  the MDS embedding is a good way to check whether outliers are
  robust.
