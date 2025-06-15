Quick Start Guide
=================

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import HaplotypeMatrix
   
   # From VCF file
   h = HaplotypeMatrix.from_vcf("data.vcf")
   
   # From NumPy array
   import numpy as np
   data = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)
   positions = np.array([100, 200])
   h = HaplotypeMatrix(data, positions)

Computing LD Statistics
~~~~~~~~~~~~~~~~~~~~~~~

Single Population
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pg_gpu import ld_statistics
   
   # Get haplotype counts
   result = h.tally_gpu_haplotypes()
   if isinstance(result, tuple):
       counts, n_valid = result
   else:
       counts, n_valid = result, None
   
   # Compute statistics
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
   dz_vals = ld_statistics.dz(counts, n_valid=n_valid)
   pi2_vals = ld_statistics.pi2(counts, n_valid=n_valid)

Two Populations
^^^^^^^^^^^^^^^

.. code-block:: python

   # Define populations
   h.sample_sets = {
       "pop1": [0, 1, 2, 3],
       "pop2": [4, 5, 6, 7]
   }
   
   # Compute between-population statistics
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=[0, 1000, 5000, 10000],
       pop1="pop1",
       pop2="pop2"
   )

Missing Data
~~~~~~~~~~~~

Missing data is handled automatically:

.. code-block:: python

   # Data with missing values (-1)
   data_missing = np.array([[0, -1, 1], [1, 0, -1]], dtype=np.int8)
   positions = np.array([100, 200])
   h = HaplotypeMatrix(data_missing, positions)
   
   # Compute statistics (automatic missing data handling)
   result = h.tally_gpu_haplotypes()
   if isinstance(result, tuple):
       counts, n_valid = result
   else:
       counts, n_valid = result, None
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)