LD Block Partitioning
=====================

Packaged script: ``examples/ld_blocks.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/ld_blocks.py
   pixi run python examples/ld_blocks.py --window 200 --max-score 0.02

Background
----------

Linkage disequilibrium decays approximately exponentially with
recombination distance, but recombination rates are not uniform
along the genome -- they are concentrated in *hotspots* separated by
relatively cold regions. The result is that many genomes naturally
partition into blocks of high pairwise LD separated by sharp
boundaries at the hotspots. These blocks are useful for haplotype
analysis, association-test correction, fine-mapping, and as a unit
for block jackknife / bootstrap, where blocks separated by
recombination hotspots form approximately independent observations
and so give better-calibrated standard errors than arbitrary
fixed-size windows.

The bottleneck in inferring LD blocks at chromosome scale is the
:math:`n \times n` pairwise :math:`r^2` matrix, which is
:math:`O(V^2)` in the number of variants. For a chromosome with
hundreds of thousands of variants this is impractical on a CPU. On
a GPU it takes seconds: pg_gpu's ``pairwise_r2`` is a single fused
kernel that computes the full matrix in one pass. Once you have the
matrix, *finding* the blocks is comparatively cheap.

What the script does
--------------------

The demo simulates a scenario where the ground truth is known and
verifies that the algorithm recovers it:

1. **Simulate.** A 1 Mb chromosome under ``msprime`` with two
   recombination hotspots planted at known positions. Ground truth =
   exactly three LD blocks separated by the hotspots.
2. **Compute :math:`r^2`.** Call ``hm.pairwise_r2()`` to get the full
   matrix on the GPU.
3. **Find boundaries with a bridging score.** At each candidate
   breakpoint, compute the mean :math:`r^2` across a sliding
   ``(left-window, right-window)`` pair: high mean inside a block,
   low across a hotspot. ``scipy.signal.find_peaks`` flags the
   minima.
4. **Plot.** A three-panel figure shows the :math:`r^2` heatmap with
   the detected block boundaries overlaid, the bridging-score trace
   underneath, and the simulated recombination map for visual
   comparison. The algorithm recovers both hotspots cleanly.

Why it's useful as a template
-----------------------------

The pattern -- *compute the full LD matrix once on the GPU, run cheap
post-processing on it* -- is the right shape for any analysis that
needs the full LD structure of a region:

* LD-block detection (this tutorial).
* LD pruning diagnostics (which threshold is enough?).
* Haplotype-block visualisation for a candidate region.
* Detecting structural variants from anomalous block boundaries.

The bridging score is a simple choice but not the only one. The
tutorial code is short enough to fork and swap in a different
boundary detector -- e.g. ``scipy.cluster.hierarchy`` on the
:math:`r^2` distance matrix, or any change-point detection method
operating on the bridging score.
