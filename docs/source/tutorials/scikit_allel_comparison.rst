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
producing several diversity statistics along a chromosome. In
``scikit-allel`` this needs three separate calls -- one per statistic
-- each rebuilding its own per-window iteration. ``pg_gpu`` collapses
the same three statistics into a single ``windowed_analysis(...)``
call that runs in one GPU kernel pass.

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
4. Verifies strict numerical agreement on all three statistics
   (NaN-aware, ``rtol=1e-5`` / ``atol=1e-8``). Trailing partial
   windows (where the chromosome end falls inside a window) are
   masked out of the comparison: the two libraries normalize the
   trailing partial window differently (allel divides by actual
   span; pg_gpu divides by the fixed window size), so a 1-window
   discrepancy there is uninteresting.
5. Plots a 4-panel figure: pi / theta_W / Tajima's D traces overlaid
   between the two implementations (so agreement = visual identity)
   plus horizontal timing bars at the bottom annotated with the
   speedup ratio.

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
