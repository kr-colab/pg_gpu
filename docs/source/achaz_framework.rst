Achaz Framework
===============

pg_gpu implements the generalized theta estimation framework from
`Achaz (2009) <https://doi.org/10.1534/genetics.109.104042>`_. This
framework unifies all frequency-spectrum-based theta estimators and
neutrality tests as linear combinations of the site frequency spectrum
(SFS).

Background
----------

Every standard theta estimator can be written as a weighted sum of the SFS:

.. math::

   \hat{\theta}_w = \sum_{i=1}^{n-1} w_i \cdot \xi_i

where :math:`\xi_i` is the number of variants with derived allele count
:math:`i` in a sample of :math:`n` haplotypes, and :math:`w_i` is a
weight vector specific to each estimator. The weight vectors in pg_gpu
incorporate all normalization factors (sample size corrections, frequency
weighting) so that theta is simply a dot product of weights and SFS counts.

Different weight vectors recover different estimators:

======================== ============================================= ====================
Estimator                Weight :math:`w_i`                            Reference
======================== ============================================= ====================
Watterson's theta        :math:`1 / a_1` where :math:`a_1 = \sum 1/j` Watterson 1975
Pi (nuc. diversity)      :math:`2i(n-i) / n(n-1)`                     Tajima 1983
Theta H                  :math:`2i^2 / n(n-1)`                        Fay & Wu 2000
Theta L                  :math:`i / (n-1)`                             Zeng et al. 2006
Eta1 (singletons)        :math:`\delta_{i,1} / a_1`                   Fu & Li 1993
======================== ============================================= ====================

Neutrality tests are contrasts between two estimators:

- **Tajima's D** = pi - theta_W (normalized)
- **Fay & Wu's H** = pi - theta_H
- **Zeng's E** = theta_L - theta_W (normalized)

The framework computes the SFS once on GPU, then derives all estimators
as trivial dot products. This is faster than calling individual functions
when computing multiple statistics.

Usage
-----

Basic usage:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, FrequencySpectrum

   h = HaplotypeMatrix.from_vcf("data.vcf.gz")
   h.sample_sets = {"pop1": [0, 1, 2, 3], "pop2": [4, 5, 6, 7]}

   # Build the SFS (one GPU pass)
   fs = FrequencySpectrum(h, population="pop1")

   # Compute any theta estimator by name
   pi = fs.theta("pi")
   tw = fs.theta("watterson")
   th = fs.theta("theta_h")

   # Compute neutrality tests
   D = fs.tajimas_d()
   H = fs.fay_wu_h()

   # Get everything at once
   all_thetas = fs.all_thetas()   # 8 estimators
   all_tests = fs.all_tests()     # 4 test statistics

Custom Weight Vectors
---------------------

Define your own estimator with any weight function:

.. code-block:: python

   import numpy as np

   # Exponential weight emphasizing rare variants
   def exponential_weights(n):
       w = np.zeros(n + 1)
       k = np.arange(1, n, dtype=np.float64)
       w[1:n] = np.exp(-0.5 * k)
       return w

   theta_custom = fs.theta(exponential_weights)

   # Generalized neutrality test between any two estimators
   T = fs.neutrality_test(exponential_weights, "watterson")

The weight function takes the sample size ``n`` and returns an array of
length ``n + 1``. Only indices 1 through n-1 are used (segregating sites).

Missing Data Strategies
-----------------------

When data contain missing genotypes (encoded as -1), different sites have
different effective sample sizes. pg_gpu offers three strategies for
handling this in theta estimation:

**Group by sample size (default):** Variants are grouped by their per-site
sample size and theta is estimated using sample-size-specific weights for
each group, retaining all data without discarding any sites. This is the
default ``missing_data='include'`` behavior.

.. code-block:: python

   fs = FrequencySpectrum(h, missing_data="include")  # default
   pi = fs.theta("pi")  # uses all sites with n_valid >= 2

**Hypergeometric projection:** Each variant's SFS contribution is projected
down to a common sample size using the hypergeometric distribution
(`Gutenkunst et al. 2009 <https://doi.org/10.1371/journal.pgen.1000695>`_),
producing a clean SFS at a single n at the cost of discarding sites with
too few observations.

.. code-block:: python

   fs = FrequencySpectrum(h, missing_data="include")

   # Let pg_gpu suggest a target that retains 95% of sites
   target = fs.suggest_projection_n(retain_fraction=0.95)

   # Project to the suggested sample size
   fs_proj = fs.project(target)
   pi_proj = fs_proj.theta("pi")

**Exclude incomplete sites:** Only sites with zero missing data across all
haplotypes are retained. At even modest missing rates in large samples this
discards nearly all data and produces catastrophically biased estimates.
Not recommended.

.. code-block:: python

   fs = FrequencySpectrum(h, missing_data="exclude")  # drops incomplete sites

Simulation-based validation (``debug/verify_missing_data_projection.py``)
shows that both the group-by-n and projection approaches are unbiased at
all missing rates tested (0--40%), while the exclude strategy is
catastrophically biased above 1% missing in a sample of 100 haplotypes.

Choosing a Projection Target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When projection is needed (e.g., for cross-population comparison at matched
sample sizes, or as input to demographic inference tools like moments or
dadi), use ``suggest_projection_n()`` to pick a target:

.. code-block:: python

   fs = FrequencySpectrum(h, missing_data="include")

   # Retain 95% of segregating sites (default)
   n95 = fs.suggest_projection_n(retain_fraction=0.95)

   # More conservative: retain 99%
   n99 = fs.suggest_projection_n(retain_fraction=0.99)

   # Project and compute
   fs_proj = fs.project(n95)
   stats = fs_proj.all_thetas()

The method picks the largest n such that at least ``retain_fraction`` of
segregating sites have per-site sample size >= n. Sites below the target
are dropped during projection. If there is no missing data, it returns
the full sample size.

Batch Computation
-----------------

The ``diversity_stats_fast()`` function uses the Achaz framework
internally to compute all statistics in one pass:

.. code-block:: python

   from pg_gpu import diversity

   stats = diversity.diversity_stats_fast(
       h, population="pop1",
       span_normalize=True,
       projection_n=50  # optional: project SFS first
   )
   # Returns dict with 12+ statistics:
   # pi, watterson, theta_h, theta_l, eta1, eta1_star,
   # minus_eta1, minus_eta1_star, segregating_sites,
   # tajimas_d, fay_wu_h, normalized_fay_wu_h, zeng_e

API Reference
-------------

.. autoclass:: pg_gpu.achaz.FrequencySpectrum
   :members:

.. autofunction:: pg_gpu.achaz.compute_sigma_ij

.. autofunction:: pg_gpu.achaz.project_sfs

.. autofunction:: pg_gpu.diversity.diversity_stats_fast
