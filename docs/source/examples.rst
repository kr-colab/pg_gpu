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
   # Memory-efficient: only processes pairs within max distance
   # chunk_size='auto' adapts to available GPU memory
   stats = h.compute_ld_statistics_gpu_two_pops(
       bp_bins=np.array([0, 1000, 5000, 10000, 50000]),
       pop1="CEU",
       pop2="YRI",
       chunk_size='auto'  # or int for fixed size (e.g., 500_000)
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
   # Memory-efficient: only processes pairs within max distance
   # chunk_size='auto' adapts to available GPU memory
   ld_stats = h.compute_ld_statistics_gpu_single_pop(
       bp_bins=[0, 1000, 5000],
       chunk_size='auto'  # or int for fixed size (e.g., 500_000)
   )
   
   # Use with moments demographic models
   # (Example integration pattern)

Missing Data Examples
---------------------

Basic Missing Data Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity
   
   # Load data with missing values
   h = HaplotypeMatrix.from_vcf("data_missing.vcf")
   
   # Check missing data
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['missing_freq_overall']:.1%}")
   print(f"Variants with no missing: {summary['variants_with_no_missing']}")
   
   # Filter high-missing sites
   h_clean = h.filter_variants_by_missing(max_missing_freq=0.1)

Computing Statistics with Missing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different missing data strategies
   pi_include = diversity.pi(h, missing_data='include')
   pi_exclude = diversity.pi(h, missing_data='exclude')
   
   print(f"Pi (include): {pi_include:.4f}")
   print(f"Pi (exclude): {pi_exclude:.4f}")
   
   # Span normalization options
   pi_total = diversity.pi(h, span_normalize=True, span_denominator='total')
   pi_callable = diversity.pi(h, span_normalize=True, span_denominator='callable')
   
   print(f"Pi/bp (total span): {pi_total:.6f}")
   print(f"Pi/bp (callable): {pi_callable:.6f}")

LD Statistics with Missing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # LD statistics handle missing data automatically
   result = h.tally_gpu_haplotypes()
   if isinstance(result, tuple):
       counts, n_valid = result
   else:
       counts, n_valid = result, None
   
   # Compute LD with missing data
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
   
   # Multiple populations with missing data
   stats = ld_statistics.compute_ld_statistics(
       counts,
       statistics=['dd', 'dz'],
       populations={'dd': (0, 1), 'dz': (0, 0, 1)},
       n_valid=n_valid
   )