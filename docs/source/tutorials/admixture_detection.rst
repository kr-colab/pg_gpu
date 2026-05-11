Admixture Detection End-to-End
==============================

Packaged script: ``examples/admixture_detection.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/admixture_detection.py
   pixi run python examples/admixture_detection.py --length 20_000_000 --samples 20

Background
----------

Patterson's :math:`D` (also called the ABBA-BABA statistic) is the
canonical genome-wide test for gene flow between populations. Given a
rooted four-population topology :math:`(((A, B), C), D)` -- where
:math:`D` is the outgroup -- :math:`D` measures the excess of
``ABBA`` site patterns (derived in B and C, ancestral in A and D)
relative to ``BABA`` patterns (derived in A and C, ancestral in B
and D). Under the strict tree topology with no gene flow,
:math:`\mathbb{E}[D] = 0`. A positive :math:`D` is evidence of
admixture from C into B (or vice versa); a negative :math:`D` flips
the direction.

Genome-wide :math:`D` is a ratio of sums (numerator and denominator
each summed across sites), so the appropriate uncertainty measure is
not a per-site standard error but a *block jackknife*: drop one
contiguous chromosomal block at a time, recompute :math:`\hat D`, and
use the variance of the leave-one-out estimates. This handles the
correlation between linked sites correctly.

What the script does
--------------------

1. Simulate two 4-population ``msprime`` tree sequences sharing the
   same null topology -- one without admixture (the "null" scenario)
   and one with a 10 % :math:`C \to B` admixture pulse (the
   "admixed" scenario).
2. Load each tree sequence into a ``HaplotypeMatrix`` via
   ``HaplotypeMatrix.from_ts``.
3. Run ``admixture.average_patterson_d`` on each, which computes the
   per-site numerator and denominator of :math:`D`, partitions them
   into contiguous blocks, and returns the point estimate, the
   block-jackknife standard error, the z-score, and the per-block
   leave-one-out values.
4. Plot the per-block :math:`D` distribution and the point estimates
   with their 95 % confidence intervals (point estimate
   :math:`\pm\,1.96` standard errors) for both scenarios on a single
   figure.

The expected (and observed) outcome: the null scenario's CI brackets
zero, while the admixed scenario's CI excludes zero with a clearly
negative :math:`\hat D` consistent with :math:`C \to B` gene flow.
Also, you can see the per-block :math:`D` distribution is shifted to the left (more negative values) in the admixed scenario, showing the genome-wide signal of admixture.

Why it's useful as a template
-----------------------------

The structure -- *simulate (or load) two scenarios -> compute the same
statistic on each with a block-jackknife wrapper -> compare the
intervals* -- is the core of any null-vs-alternative power analysis or
calibration check. A few directions to extend:

* **Different statistic.** ``average_patterson_d`` has a sibling
  ``average_patterson_f3`` for the F3 admixture test; the two share
  the ratio-of-sums structure, so the wrapper code is interchangeable.
* **Different demographic model.** Edit the ``msprime`` model to add
  bottlenecks, varying admixture proportions, ghost populations, etc.
  This is also the right structure for calibrating a statistic on
  simulated null data before applying it to your real data.
* **Block size sensitivity.** The script uses a fixed ``blen=100``
  but real-data analyses typically check that the estimated standard
  error is stable across a range of block sizes (1-10 Mb is a common
  target). Sweeping ``blen`` and overlaying the resulting CIs is a
  one-line change.
