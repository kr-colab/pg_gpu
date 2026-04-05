Missing Data Handling
=====================

pg_gpu provides comprehensive support for missing data across all
population genetics statistics. Missing data is encoded as ``-1`` in
haplotype and genotype matrices.

Missing Data Modes
------------------

Every function that operates on genetic data accepts a ``missing_data``
parameter with three options (plus a fourth for LD statistics):

**include** (default)
   Skip missing entries per site, computing statistics from observed
   data only. Each site uses its own sample size (``n_valid``). For
   haplotype identity comparisons (e.g., Garud's H, haplotype
   diversity), missing values are treated as wildcards compatible with
   any allele.

**exclude**
   Drop entire sites that have any missing data in any sample. Only
   fully genotyped sites contribute to the result.

**pairwise**
   Comparison-counting normalization inspired by pixy (Korunes & Samuk
   2021). Instead of averaging per-site estimates, computes
   ``sum(diffs) / sum(comps)`` across all sites, where sites with more
   observed data contribute proportionally more weight. Invariant sites
   can be included in the denominator when ``n_total_sites`` is set on
   the matrix (see :ref:`invariant-sites`).

**project** (LD statistics only)
   Unbiased multinomial projection estimators following Ragsdale & Gravel
   (2019). For each pair of sites, counts the four haplotype
   configurations among individuals observed at both sites, then applies
   falling-factorial corrections to obtain unbiased estimates of D² and
   pi2. The per-pair statistic is sigma_d^2 = D² / pi2, which avoids
   the upward bias of naive r² that worsens with small or variable
   sample sizes. Currently supported by ``zns()`` and ``omega()``.
   Requires a ``HaplotypeMatrix`` as input (not a pre-computed r² array).

   Reference: Ragsdale AP, Gravel S (2019) "Unbiased estimation of
   linkage disequilibrium from unphased data." *Mol Biol Evol*
   37(3):923-932.

The old ``'ignore'`` mode (which treated missing as reference allele)
has been removed because it silently biases allele frequencies.

Basic Usage
-----------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence

   h = HaplotypeMatrix.from_vcf("data.vcf")

   # Default: per-site valid data (include mode)
   pi = diversity.pi(h)

   # Conservative: only fully genotyped sites
   pi_excl = diversity.pi(h, missing_data='exclude')

   # Pixy-style comparison counting
   pi_pw = diversity.pi(h, missing_data='pairwise')

How Each Mode Works
-------------------

Consider a site with 100 haplotypes where 10 are missing (``-1``):

* **include**: Computes allele frequencies from the 90 observed
  haplotypes. The site contributes to the result with ``n_valid = 90``.
* **exclude**: The site is dropped entirely because it has missing data.
* **pairwise**: The site contributes ``C(90, 2) = 4005`` comparisons
  to the denominator and the observed pairwise differences to the
  numerator. A fully genotyped site would contribute ``C(100, 2) = 4950``
  comparisons, so it naturally gets more weight.

.. _invariant-sites:

Invariant Sites and Pairwise Mode
----------------------------------

The ``pairwise`` mode can account for invariant (monomorphic) sites in
the denominator, which is essential for unbiased estimation of pi and
dxy. Without invariant sites, diversity is overestimated because only
variable sites contribute comparisons.

To enable this, set ``n_total_sites`` on the matrix:

.. code-block:: python

   # From tree sequences: track total callable sites analytically
   h = HaplotypeMatrix.from_ts(ts, include_invariant=True)

   # From VCFs containing invariant sites (ALT=".")
   h = HaplotypeMatrix.from_vcf("allsites.vcf", include_invariant=True)

   # Set manually
   h.n_total_sites = 1_000_000

   # Now pairwise mode includes invariant sites in the denominator
   pi = diversity.pi(h, missing_data='pairwise')

If ``n_total_sites`` is not set, pairwise mode emits a warning and
computes from variant sites only.

Component Statistics
~~~~~~~~~~~~~~~~~~~~

In pairwise mode, ``pi()`` and ``dxy()`` can return the underlying
components for proper windowed aggregation:

.. code-block:: python

   result = diversity.pi(h, missing_data='pairwise',
                         span_normalize=False, return_components=True)
   print(result.total_diffs)    # sum of pairwise differences
   print(result.total_comps)    # sum of pairwise comparisons
   print(result.total_missing)  # comparisons lost to missing data
   print(result.value)          # total_diffs / total_comps

Supported Statistics
--------------------

Every public function accepts the ``missing_data`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 25 13 13 13 13

   * - Function
     - include
     - exclude
     - pairwise
     - project
   * - Diversity (pi, theta_w, theta_h, theta_l)
     - per-site n
     - filter sites
     - sum/sum
     - \-
   * - Neutrality tests (tajimas_d, fay_wus_h, H*, E, DH)
     - per-site n
     - filter sites
     - harmonic mean n
     - \-
   * - Divergence (dxy, fst, da)
     - per-site n
     - filter sites
     - sum/sum
     - \-
   * - SFS (sfs, joint_sfs, folded variants)
     - per-site n
     - filter sites
     - maps to include
     - \-
   * - Admixture (patterson_d, f2, f3)
     - per-site n
     - NaN at missing
     - maps to include
     - \-
   * - Selection scans (ihs, nsl, xpehh)
     - wildcard in SSL
     - filter sites
     - maps to include
     - \-
   * - Haplotype stats (garud_h, haplotype_diversity)
     - wildcard match
     - filter sites
     - maps to include
     - \-
   * - Distance (pairwise_diffs, pca)
     - per-pair norm
     - filter sites
     - maps to include
     - \-
   * - LD (zns, omega)
     - per-site n
     - filter sites
     - maps to include
     - unbiased sigma_d^2

Haplotype Identity and Missing Data
------------------------------------

For statistics based on haplotype identity (Garud's H, haplotype
diversity, haplotype count), missing values are treated as wildcards:
two haplotypes match if they agree at all positions where both are
non-missing.

.. code-block:: python

   # Haplotypes [0, 1, 0, 1] and [0, -1, 0, 1] are considered identical
   # because they match at positions 0, 2, 3 (position 1 is missing)

   from pg_gpu import selection
   h1, h12, h123, h2_h1 = selection.garud_h(h)

HaplotypeMatrix Utilities
-------------------------

.. code-block:: python

   # Detect and count missing data
   missing_per_site = h.count_missing(axis=0)
   missing_per_sample = h.count_missing(axis=1)

   # Filter by missing data frequency
   h_clean = h.filter_variants_by_missing(max_missing_freq=0.1)

   # Summary statistics
   summary = h.summarize_missing_data()

   # Invariant site info
   h.n_total_sites = 1_000_000
   print(h.has_invariant_info)    # True
   print(h.n_invariant_sites)     # total - variant count

Windowed Analysis
-----------------

The windowed analysis framework respects the ``missing_data`` parameter.
In pairwise mode, component columns (diffs, comps, missing) are included
in the output for proper sum-then-divide aggregation across windows:

.. code-block:: python

   from pg_gpu.windowed_analysis import StatisticsComputer, WindowedAnalyzer

   computer = StatisticsComputer(
       statistics=['pi', 'dxy'],
       populations=['pop1', 'pop2'],
       missing_data='pairwise'
   )

   # Component columns appear automatically:
   # pi_pop1_diffs, pi_pop1_comps, pi_pop1_missing, ...

Best Practices
--------------

1. **Use include mode** (default) for most analyses. It uses all
   available data at each site without bias.

2. **Use pairwise mode** when you need unbiased genome-wide estimates
   of pi or dxy, especially with variable missingness across sites.
   Set ``n_total_sites`` or use ``include_invariant=True`` to include
   invariant sites in the denominator.

3. **Use exclude mode** when you need all samples to be comparable
   at exactly the same sites (e.g., for certain LD analyses).

4. **Use project mode** for LD statistics (ZnS, Omega) when sample
   sizes are small or variable across sites. The unbiased estimators
   correct the upward bias inherent in naive r², which can be
   substantial when n < 50. This mode computes sigma_d^2 = D^2/pi2
   using falling-factorial estimators (Ragsdale & Gravel 2019).

   .. code-block:: python

      from pg_gpu import ld_statistics

      # Unbiased ZnS and Omega
      zns = ld_statistics.zns(h, missing_data='project')
      omega = ld_statistics.omega(h, missing_data='project')

      # Also works in windowed analysis
      from pg_gpu.windowed_analysis import windowed_analysis
      results = windowed_analysis(h, statistics=['zns', 'omega'],
                                  missing_data='project')

5. **Check missingness patterns** before analysis with
   ``summarize_missing_data()`` and consider filtering sites with
   very high missing rates.

Example Workflow
----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence

   # Load data with invariant sites for unbiased estimation
   h = HaplotypeMatrix.from_vcf("allsites.vcf", include_invariant=True)

   # Inspect missing data
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['fraction_missing']:.1%}")

   # Filter extreme missingness
   h = h.filter_variants_by_missing(max_missing_freq=0.5)

   # Compute diversity with pixy-style comparison counting
   pi = diversity.pi(h, missing_data='pairwise')
   dxy = divergence.dxy(h, 'pop1', 'pop2', missing_data='pairwise')
   fst = divergence.fst(h, 'pop1', 'pop2', missing_data='pairwise')

   # Or use default include mode
   tajd = diversity.tajimas_d(h)
   fwh = diversity.fay_wus_h(h)

Achaz Framework and SFS Projection
-----------------------------------

The :doc:`achaz_framework` provides an additional approach to missing data
through SFS projection. When computing theta estimators from the site
frequency spectrum, sites with different sample sizes can be handled in
two ways:

**Group by sample size (recommended default):** The ``FrequencySpectrum``
class groups variants by their per-site sample size and applies
sample-size-specific weight vectors to each group. This uses all available
data without discarding any sites.

**Hypergeometric projection:** Project all SFS contributions to a common
sample size using the hypergeometric distribution (Gutenkunst et al. 2009).
This is useful when a clean SFS at a single sample size is needed (e.g.,
for demographic inference with moments or dadi, or for cross-population
comparison at matched n).

.. code-block:: python

   from pg_gpu import FrequencySpectrum

   fs = FrequencySpectrum(h, population="pop1")

   # Group-by-n: uses all data (default)
   pi_include = fs.theta("pi")

   # Projection: suggest a target retaining 95% of sites
   target_n = fs.suggest_projection_n(retain_fraction=0.95)
   fs_proj = fs.project(target_n)
   pi_proj = fs_proj.theta("pi")

Simulation-based validation shows both approaches are unbiased under the
standard neutral model at missing rates from 0--40%. The exclude strategy
(``missing_data='exclude'``) is catastrophically biased above 1% missing
in samples of 100 haplotypes. See ``debug/verify_missing_data_projection.py``
for the full validation and figures.
