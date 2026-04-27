Bootstrap CI on Tajima's D under a Sweep
========================================

Packaged script: ``examples/sweep_tajimas_d_bootstrap.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/sweep_tajimas_d_bootstrap.py
   pixi run python examples/sweep_tajimas_d_bootstrap.py --seed 7 --n-replicates 5000

Background
----------

Tajima's D contrasts two estimators of the population mutation rate
:math:`\theta`: the average pairwise diversity :math:`\hat{\pi}` and
Watterson's segregating-sites estimator :math:`\hat\theta_W`. Under
neutrality at equilibrium, :math:`\mathbb{E}[D] = 0`. A recent hard
sweep removes intermediate-frequency variation in the linked region
and leaves an excess of singletons as new mutations enter on the
sweep haplotype, so :math:`\hat\pi` decreases relative to
:math:`\hat\theta_W` and :math:`D` becomes negative in the local
neighborhood of the selected site.

Detecting the negative-D dip is the easy part. The harder part is
quantifying how confident you are that the dip is real, given that
windows from the same chromosome are correlated and that the mean
over a small number of windows has substantial sampling variance. The
block bootstrap addresses this directly: resample whole windows (so
LD between adjacent windows is preserved) and use the resulting
distribution of the per-region mean as an empirical sampling
distribution.

What the script does
--------------------

1. Simulate a 10 Mb chromosome under ``msprime`` with
   ``SweepGenicSelection`` targeting fixation at the chromosome
   midpoint. The default selection coefficient (``s=0.05``) gives a
   strong sweep with a clear local D dip.
2. Compute windowed Tajima's D with ``windowed_analysis``, taking
   each window as a "block" for the bootstrap.
3. Define two regions of interest: a *sweep-local* window around the
   selected site and a *distal* window on the same chromosome that
   should look neutral.
4. Use ``block_bootstrap`` (default 2000 replicates) on the
   per-window D values in each region to obtain 95 % confidence
   intervals on the mean Tajima's D.
5. Plot the per-window D track with the two regions highlighted, the
   bootstrap distributions side-by-side, and the point estimates with
   their CIs.

Under a completed sweep the sweep-local CI excludes zero (D is
significantly negative) while the distal CI brackets zero. The output
figure (``sweep_tajimas_d_bootstrap.png``) makes the contrast
visible at a glance.

Why it's useful as a template
-----------------------------

The recipe -- *windowed_analysis* -> *region selection* ->
*block_bootstrap on the per-window means* -- generalizes to any
windowed scalar statistic and any candidate region. Swap
``tajimas_d`` for ``fst``, ``garud_h12``, ``mean_nsl``, ``dxy`` etc.
to put a confidence interval on the effect size in any region of
interest, with no change to the surrounding code.

A few things to keep in mind when adapting the template:

* The bootstrap "block" is a window. Block size is therefore the
  ``window_size`` argument to ``windowed_analysis``. It should be
  large compared to the per-site LD scale -- if windows are too
  small, sample-to-sample correlation between blocks will deflate
  the bootstrap CI. For better-calibrated CIs in regions with very
  uneven recombination (or where the LD-decay scale is unknown),
  partition the chromosome into LD blocks first (see
  :doc:`ld_blocks`) and pass those as the bootstrap unit instead of
  fixed-size bp windows.
* ``block_bootstrap`` accepts a tuple ``(numer, denom)`` for
  ratio-of-sums statistics (FST, F3 / D, etc.); pass per-window
  numerator and denominator instead of a per-window scalar.
* For very small effects the default ``n_replicates=2000`` may give
  noisy CIs; bump to 5000-10000 for publication figures.
