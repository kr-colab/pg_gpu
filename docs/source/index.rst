pg_gpu Documentation
====================

GPU-accelerated population genetics statistics for Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   achaz_framework
   api
   missing_data
   examples
   moments_integration
   changelog

Overview
--------

pg_gpu provides GPU-accelerated computation of population genetics statistics
using CuPy. It covers linkage disequilibrium, diversity, divergence, selection
scans, site frequency spectra, admixture statistics, and dimensionality
reduction (PCA, PCoA).

Key Features
~~~~~~~~~~~~

* **Fast GPU computation** using CuPy with fused CUDA kernels for compute-intensive operations
* **Comprehensive statistics**: LD (D, D-squared, Dz, pi2, r/r-squared), diversity (pi, theta, Tajima's D, heterozygosity, Fay & Wu's H), divergence (FST Hudson/Weir-Cockerham/Nei, Dxy, Da), selection scans (iHS, XP-EHH, nSL, XP-nSL, Garud's H, EHH decay), SFS (unfolded, folded, joint, scaled), admixture (Patterson's F2, F3, D)
* **Fused windowed analysis**: compute all statistics across all genomic windows in a single GPU pass -- up to 60x faster than scikit-allel
* **Automatic missing data handling** across all modules
* **Multi-population analyses** with flexible population specification
* **Achaz (2009) generalized framework**: compute all theta estimators and neutrality tests from a single SFS, with custom weight vectors and SFS projection for missing data
* **Validated against scikit-allel** -- 29 statistics verified at machine precision using real Ag1000G data

Installation
------------

.. code-block:: bash

   pixi install
   pixi shell

Quick Example
-------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, selection

   # Load data
   h = HaplotypeMatrix.from_vcf("data.vcf")

   # Diversity
   pi_val = diversity.pi(h)
   tajd = diversity.tajimas_d(h)

   # Selection scans
   ihs_scores = selection.ihs(h)

   # LD r-squared
   r2 = h.pairwise_r2()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
