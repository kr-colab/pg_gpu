Missing Data Handling
=====================

pg_gpu provides comprehensive support for missing data in population genetics analyses.

Overview
--------

Missing data is represented as ``-1`` in haplotype matrices. All statistics functions automatically detect and handle missing data using three strategies:

* **include** - Include missing data in calculations (default for most statistics)
* **exclude** - Exclude sites/samples with missing data
* **ignore** - Treat missing data as a separate allele state

Basic Usage
-----------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity
   
   # Load data with missing values
   h = HaplotypeMatrix.from_vcf("data_with_missing.vcf")
   
   # Statistics automatically handle missing data
   pi = diversity.pi(h, missing_data='exclude')
   theta = diversity.theta_w(h, missing_data='include')

Span Normalization
------------------

When normalizing by genomic span, you can specify how to calculate the denominator:

.. code-block:: python

   # Total span (default)
   pi_total = diversity.pi(h, span_normalize=True, span_denominator='total')
   
   # Only sites present in data
   pi_sites = diversity.pi(h, span_normalize=True, span_denominator='sites')
   
   # Only callable sites (non-missing)
   pi_callable = diversity.pi(h, span_normalize=True, span_denominator='callable')

HaplotypeMatrix Utilities
-------------------------

The HaplotypeMatrix class provides several utilities for working with missing data:

.. code-block:: python

   # Check for missing data
   has_missing = h.has_missing()
   
   # Count missing data
   missing_per_site = h.count_missing(axis=0)
   missing_per_sample = h.count_missing(axis=1)
   
   # Filter variants by missing data frequency
   h_filtered = h.filter_variants_by_missing(max_missing_freq=0.1)
   
   # Get summary statistics
   summary = h.summarize_missing_data()
   print(f"Total missing: {summary['missing_freq_overall']:.1%}")

Statistics with Missing Data
----------------------------

Diversity Statistics
~~~~~~~~~~~~~~~~~~~~

All diversity statistics support the ``missing_data`` parameter:

.. code-block:: python

   # Nucleotide diversity
   pi = diversity.pi(h, missing_data='exclude')
   
   # Watterson's theta
   theta = diversity.theta_w(h, missing_data='include')
   
   # Tajima's D (always excludes missing)
   d = diversity.tajimas_d(h)
   
   # Haplotype diversity
   h_div = diversity.haplotype_diversity(h)

Divergence Statistics
~~~~~~~~~~~~~~~~~~~~~

Population divergence statistics also handle missing data:

.. code-block:: python

   # FST with missing data handling
   fst = divergence.fst(h, 'pop1', 'pop2', missing_data='exclude')
   
   # Dxy excluding missing sites
   dxy = divergence.dxy(h, 'pop1', 'pop2', missing_data='exclude')

LD Statistics
~~~~~~~~~~~~~

LD statistics automatically detect and handle missing data:

.. code-block:: python

   from pg_gpu import ld_statistics
   
   # Tally haplotypes (returns counts and n_valid for missing data)
   counts, n_valid = h.tally_gpu_haplotypes()
   
   # LD statistics use n_valid automatically
   dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
   dz_vals = ld_statistics.dz(counts, n_valid=n_valid)

Best Practices
--------------

1. **Check your data** - Use ``summarize_missing_data()`` to understand missingness patterns
2. **Choose appropriate strategy** - Use 'exclude' for conservative estimates, 'include' when missing is random
3. **Consider filtering** - Remove sites with high missing rates before analysis
4. **Document your choice** - Missing data handling affects results, document your approach

Example Workflow
----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence
   
   # Load and inspect data
   h = HaplotypeMatrix.from_vcf("data.vcf")
   summary = h.summarize_missing_data()
   print(f"Missing data: {summary['fraction_missing']:.1%}")
   
   # Filter high-missing sites
   h_filtered = h.filter_variants_by_missing(max_missing_freq=0.2)
   print(f"Kept {h_filtered.num_variants} of {h.num_variants} variants")
   
   # Compute statistics with explicit missing data handling
   results = {
       'pi': diversity.pi(h_filtered, missing_data='exclude'),
       'theta': diversity.theta_w(h_filtered, missing_data='exclude'),
       'fst': divergence.fst(h_filtered, 'pop1', 'pop2', missing_data='exclude')
   }