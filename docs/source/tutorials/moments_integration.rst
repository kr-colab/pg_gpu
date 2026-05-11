Demographic Inference with moments.LD
=====================================

Packaged script: ``examples/moments_integration_demo.py``

Run it from the repo root (note the ``-e moments`` env flag --
``moments`` lives in a separate pixi environment to keep the default
install lightweight):

.. code-block:: bash

   pixi install -e moments
   pixi run -e moments python examples/moments_integration_demo.py

What it shows
-------------

`moments.LD <https://moments.readthedocs.io/>`_ (`Ragsdale & Gravel
2019 <https://doi.org/10.1093/molbev/msz265>`_) infers demographic
histories by matching observed two-locus LD
statistics to expectations under a parameterized model. The bottleneck
in practice is *parsing* the LD statistics from genotype data: for
:math:`V` variants and :math:`P` populations the work is :math:`O(V^2
\binom{P+2}{2})`, which on whole-genome data is hours-to-days on a
CPU. Inference itself, by contrast, takes seconds.

``pg_gpu.moments_ld.compute_ld_statistics`` is a drop-in replacement
for ``moments.LD.Parsing.compute_ld_statistics``. Same arguments, same
output dictionary structure -- only the parsing step runs on the GPU.
Everything downstream (bootstrap variance-covariance, Demes /
parametric inference, Godambe confidence intervals) uses ``moments``
unchanged. The validation tests run pg_gpu's parser against
``moments``'s on the same VCFs and confirm machine precision agreement
(max relative error :math:`< 10^{-11}`).

The packaged script is an end-to-end demonstration: it simulates 200
replicate 1 Mb regions under a three-population model with recent
admixture using ``msprime``, parses each replicate with pg_gpu, fits
the demographic model with ``moments.LD`` via the Demes inference
engine, computes Godambe standard errors, and plots the fitted vs
observed LD decay curves. Total wall time for the parsing step on an
A100 is under five minutes -- on the same hardware, the CPU
``moments`` parser would take roughly a day.

Recipe
------

The core change between a ``moments.LD`` workflow and a ``pg_gpu``-accelerated workflow is:

.. code-block:: python

   from pg_gpu.moments_ld import compute_ld_statistics
   import moments.LD

   r_bins = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]

   # Step 1: GPU-accelerated LD parsing (the only part that changes
   # vs. a pure-moments workflow).
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

What it computes
----------------

For each pair of variants binned by recombination distance,
``compute_ld_statistics`` returns 15 two-locus LD statistics:

- **DD** (3 stats): :math:`D^2` within and between populations.
- **Dz** (6 stats): two-locus LD correlations involving three
  populations.
- **pi2** (6 stats): two-locus joint statistics involving four
  populations.

Plus 3 single-locus heterozygosity statistics (H_0_0, H_0_1, H_1_1).
These are exactly the statistics that ``moments.LD`` ingests.

Performance
-----------

Parsing time across 3 replicate 1 Mb regions (10 diploid individuals
per population):

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

The speedup compounds with the number of populations and the number
of replicate regions, which is exactly the regime where
demographic-LD parsing has historically been impractical on a single
machine.

Why it's useful as a template
-----------------------------

If you already have a ``moments.LD`` workflow, swapping in pg_gpu
parsing is a one-line change at the top of your script. If you
don't, the packaged demo is the easiest way to see the whole pipeline
end-to-end -- simulate, parse, infer, plot -- in one self-contained
file (~250 lines) that you can fork and adapt to your own
demographic model.
