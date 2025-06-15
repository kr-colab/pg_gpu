Examples
========

Complete Workflow
-----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, ld_statistics
   import numpy as np
   
   # Load VCF data
   h = HaplotypeMatrix.from_vcf("example.vcf")
   
   # Define populations
   h.sample_sets = {
       "CEU": [0, 1, 2, 3, 4],
       "YRI": [5, 6, 7, 8, 9]
   }
   
   # Compute within-population statistics
   counts_ceu = h.tally_gpu_haplotypes(pop="CEU")
   dd_ceu = ld_statistics.dd(counts_ceu)
   
   # Compute between-population statistics
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=np.array([0, 1000, 5000, 10000, 50000]),
       pop1="CEU",
       pop2="YRI"
   )

Batch Processing
----------------

.. code-block:: python

   # Compute multiple statistics at once
   results = ld_statistics.compute_ld_statistics(
       counts,
       statistics=['dd', 'dz', 'pi2'],
       populations={
           'dd': (0, 1),
           'dz': (0, 0, 1),
           'pi2': (0, 0, 1, 1)
       }
   )
   
   print(f"DD values: {results['dd']}")
   print(f"Dz values: {results['dz']}")
   print(f"π₂ values: {results['pi2']}")

Integration with moments
------------------------

.. code-block:: python

   import moments
   from pg_gpu import HaplotypeMatrix
   
   # Load data and compute LD statistics
   h = HaplotypeMatrix.from_vcf("data.vcf")
   h.sample_sets = {"pop1": list(range(10))}
   
   # Compute LD statistics with GPU acceleration
   ld_stats = h.compute_ld_statistics_gpu_single_pop(
       bp_bins=[0, 1000, 5000],
       pop="pop1"
   )
   
   # Use with moments demographic models
   # (Example integration pattern)