Moments Integration
===================

pg_gpu provides a GPU-accelerated drop-in replacement for
``moments.LD.Parsing.compute_ld_statistics()``, enabling fast computation
of two-population LD statistics (DD, Dz, pi2) for demographic inference.
The output format is identical to moments, so the downstream inference
pipeline (bootstrapping, optimization, Godambe confidence intervals) works
unchanged.

Installation
------------

moments is in a separate pixi environment to keep the default install
lightweight:

.. code-block:: bash

   pixi install -e moments
   pixi run -e moments python my_script.py

Usage
-----

The key function is ``pg_gpu.moments_ld.compute_ld_statistics()``. It accepts
the same arguments as ``moments.LD.Parsing.compute_ld_statistics()`` and
returns output in the same format.

.. code-block:: python

   from pg_gpu.moments_ld import compute_ld_statistics
   import moments.LD

   r_bins = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

   # Step 1: GPU-accelerated LD parsing (drop-in replacement)
   ld_stats = {}
   for i, vcf in enumerate(vcf_files):
       ld_stats[i] = compute_ld_statistics(
           vcf,
           rec_map_file="genetic_map.txt",
           pop_file="pops.txt",
           pops=["pop0", "pop1"],
           r_bins=r_bins,
       )

   # Step 2: Bootstrap means and variance-covariance (moments, unchanged)
   mv = moments.LD.Parsing.bootstrap_data(ld_stats)

   # Step 3: Demographic inference (moments, unchanged)
   demo_func = moments.LD.Demographics2D.split_mig
   p_guess = [0.1, 2, 0.075, 2, 10000]

   opt_params, LL = moments.LD.Inference.optimize_log_lbfgsb(
       p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins,
   )

   # Step 4: Convert to physical units
   physical = moments.LD.Util.rescale_params(
       opt_params, ["nu", "nu", "T", "m", "Ne"]
   )

What It Computes
----------------

For each pair of variants binned by recombination distance, the function
computes 15 two-locus LD statistics:

- **DD** (3 stats): :math:`D^2` within and between populations
- **Dz** (6 stats): three-locus LD correlations
- **pi2** (6 stats): four-locus joint LD statistics

Plus 3 single-locus heterozygosity statistics (H_0_0, H_0_1, H_1_1).

All statistics are validated against moments at machine precision
(max relative error < 1e-11).

Performance
-----------

LD parsing time across 3 replicate 1Mb regions (10 diploid individuals per population):

.. list-table::
   :header-rows: 1

   * - Populations
     - moments
     - pg_gpu
     - Speedup
   * - 2
     - 190s
     - 0.9s
     - **214x**
   * - 3
     - 638s
     - 2.9s
     - **219x**
   * - 4
     - 1630s
     - 7.3s
     - **224x**

Example
-------

See ``examples/moments_3pop_integration_demo.py`` for a complete working example
that simulates 200 replicate regions under a three-population model with
migration, computes LD statistics with pg_gpu, fits the model using the
``moments.Demes`` inference engine, computes Godambe standard errors, and plots
fitted vs observed LD curves:

.. code-block:: bash

   pixi run -e moments python examples/moments_3pop_integration_demo.py
