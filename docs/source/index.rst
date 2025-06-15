pg_gpu Documentation
====================

GPU-accelerated population genetics statistics for Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Overview
--------

pg_gpu provides GPU-accelerated computation of linkage disequilibrium (LD) statistics, 
integrating with the `moments <https://github.com/moments-dev/moments>`_ population genetics package.

Key Features
~~~~~~~~~~~~

* **Fast GPU computation** using CuPy
* **Unified API** for all LD statistics
* **Automatic missing data handling**
* **Support for multi-population analyses**
* **Seamless integration** with moments

Installation
------------

.. code-block:: bash

   conda env create -f environment.yml
   conda activate pg_gpu
   pip install -e .

Quick Example
-------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, ld_statistics
   
   # Load data
   h = HaplotypeMatrix.from_vcf("data.vcf")
   
   # Compute LD statistics
   result = h.tally_gpu_haplotypes()
   if isinstance(result, tuple):
       counts, n_valid = result
   else:
       counts, n_valid = result, None
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`