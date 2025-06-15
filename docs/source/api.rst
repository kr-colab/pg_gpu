API Reference
=============

HaplotypeMatrix
---------------

.. autoclass:: pg_gpu.HaplotypeMatrix
   :members:
   :undoc-members:
   :show-inheritance:

LD Statistics Functions
-----------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd

.. autofunction:: pg_gpu.ld_statistics.dz

.. autofunction:: pg_gpu.ld_statistics.pi2

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd_within

.. autofunction:: pg_gpu.ld_statistics.dd_between

.. autofunction:: pg_gpu.ld_statistics.compute_ld_statistics

Parameters
----------

All LD statistics functions accept:

* **counts** : CuPy array of haplotype counts
* **populations** : Tuple specifying population indices (optional)
* **n_valid** : Valid sample counts for missing data (optional)

Population Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

* Single population: ``populations=None`` or omit
* Two populations: ``populations=(0, 1)`` for DD
* Three indices: ``populations=(0, 0, 1)`` for Dz
* Four indices: ``populations=(0, 0, 1, 1)`` for π₂