Side-by-Side: scikit-allel vs pg_gpu
====================================

Packaged script: ``examples/scikit_allel_comparison.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/scikit_allel_comparison.py
   pixi run python examples/scikit_allel_comparison.py --small
   pixi run python examples/scikit_allel_comparison.py --no-plot

Background
----------

A common workflow on phased haplotype data is a windowed scan
producing several diversity statistics along a chromosome, plus an
LD-decay curve summarising linkage disequilibrium as a function of
physical distance. In ``scikit-allel`` each diversity stat needs a
separate call, and the LD-decay scan requires manually pairwise-r²,
distance binning, and median-per-bin aggregation. ``pg_gpu``
collapses the diversity stats into a single ``windowed_analysis(...)``
call and exposes the LD-decay path as ``HaplotypeMatrix.windowed_r_squared(estimator='rogers_huff')``,
both running in single GPU kernel passes.

This tutorial puts the two implementations next to each other on
real Anopheles gambiae X-chromosome data (n = 100 phased haplotypes,
~1M segregating sites), checks that they agree numerically, and
reports the wall-clock speedup.

What the script does
--------------------

1. Loads ``examples/data/gamb.X.phased.n100.zarr`` via
   ``HaplotypeMatrix.from_zarr`` and builds the views needed
   downstream: a ``(n_variants, 2)`` allele-count array for
   ``scikit-allel`` plus the ``HaplotypeMatrix`` itself for
   ``pg_gpu``.
2. Pins the windowing by computing the window edges once with
   ``allel.position_windows``; the resulting window array is passed
   explicitly to every ``scikit-allel`` call, and the per-window
   ``start``/``end`` columns of ``windowed_analysis``'s result are
   asserted to match.
3. Times both implementations with ``time.perf_counter``, after a
   warmup run that absorbs CuPy/JIT initialization. Three stats from
   pg_gpu: ``windowed_analysis(statistics=['pi', 'theta_w',
   'tajimas_d'])``. Three stats from scikit-allel:
   ``allel.windowed_diversity``, ``allel.windowed_watterson_theta``,
   ``allel.windowed_tajima_d``.
4. Verifies strict numerical agreement on the diversity statistics
   (NaN-aware, ``rtol=1e-5`` / ``atol=1e-8``). When the chromosome
   end isn't an exact multiple of ``window_size``, the trailing
   partial window is silently NaN-masked from the comparison (the
   two libraries normalize partial windows slightly differently --
   a single-window cosmetic effect, not a real numerical
   disagreement).
5. Subsamples ``--ld-snps`` (default 10,000) random SNPs and
   computes a pairwise LD-decay curve on each side. The pg_gpu
   path is one call:
   ``hm_sub.windowed_r_squared(bp_bins, percentile=50, estimator='rogers_huff')``;
   the scikit-allel path runs ``allel.rogers_huff_r``, squares it,
   builds a pair-distance array, and bins manually. Both compute
   the per-distance-bin median Rogers-Huff (2008) :math:`r^2`, so
   the two curves agree to floating-point precision (the float32
   precision floor scikit-allel sets internally). Times each side.
6. Plots a 5-panel figure: pi / theta_W / Tajima's D traces overlaid
   between the two implementations (agreement = visual identity),
   the LD-decay curves overlaid on a log distance axis, and a
   bottom panel of horizontal timing bars annotated with speedup
   ratios for both the windowed scan and the LD-decay scan.

Why it's useful as a template
------------------------------

The recipe -- *load once, build the right shapes for each library,
pass identical windows in, compare* -- is the right shape for any
multi-statistic windowed analysis. To adapt it:

* Swap the statistics list. ``windowed_analysis`` accepts any
  combination of single-pop summaries; the ``scikit-allel`` side
  needs one named call per statistic.
* Run on your own data. The script accepts any zarr store loadable
  by ``HaplotypeMatrix.from_zarr`` -- point ``ZARR_FULL`` at it.
* Time at different window sizes. The default 10 kb is a typical
  diversity-scan resolution; ``--window-size`` accepts any positive
  integer.
* Tune the LD scan for your scale of interest. ``--ld-snps``
  controls the random subsample; larger values give a less-noisy
  decay curve at the cost of quadratic compute. The bin edges
  (``LD_BP_BINS`` in the script) span 100 bp to 1 kb in 24 log-spaced
  steps -- appropriate for *Anopheles*-like populations where LD
  decays within a kilobase. For organisms with longer LD scales
  (humans, livestock) widen the range; for tighter LD scales
  (recombining viruses) shrink it.
