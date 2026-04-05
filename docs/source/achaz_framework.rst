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

   \hat{\theta}_\omega = \frac{1}{\sum_i \omega_i} \sum_{i=1}^{n-1} \omega_i \cdot i \cdot \xi_i

where :math:`\xi_i` is the number of variants with derived allele count
:math:`i` in a sample of :math:`n` haplotypes, and :math:`\omega_i` is a
weight vector specific to each estimator.

Different weight vectors recover different estimators:

==================== ========================== ====================
Estimator            Weight :math:`\omega_i`    Reference
==================== ========================== ====================
Watterson's theta    :math:`1`                  Watterson 1975
Pi (nuc. diversity)  :math:`i(n-i)/\binom{n}{2}` Tajima 1983
Theta H              :math:`i^2/\binom{n}{2}`   Fay & Wu 2000
Theta L              :math:`i`                  Zeng et al. 2006
Eta1 (singletons)    :math:`\delta_{i,1}`       Fu & Li 1993
==================== ========================== ====================

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

SFS Projection
--------------

For datasets with missing data (variable per-site sample sizes), project
the SFS to a common sample size using hypergeometric sampling
(`Gutenkunst et al. 2009 <https://doi.org/10.1371/journal.pgen.1000695>`_):

.. code-block:: python

   # With missing data, sites have different sample sizes
   fs = FrequencySpectrum(h, population="pop1", missing_data="include")

   # Project to a common sample size
   fs_proj = fs.project(target_n=50)

   # All estimators now use the projected SFS
   pi_proj = fs_proj.theta("pi")

This enables principled cross-population comparison when populations have
different amounts of missing data.

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
