Missing Data Handling
=====================

pg_gpu provides support for missing data across all population genetics
statistics. Missing data is encoded as ``-1`` in haplotype and genotype
matrices.

Missing Data Modes
------------------

Every function that operates on genetic data accepts a ``missing_data``
parameter with two options:

**include** (default)
   Use all sites, computing statistics from observed data only. Each site
   uses its own sample size (``n_valid``). For haplotype identity
   comparisons (e.g., Garud's H), missing values are treated as wildcards
   compatible with any allele.

**exclude**
   Drop entire sites that have any missing data in any sample. Only
   fully genotyped sites contribute to the result.

Simulation testing under the standard neutral model confirms that
``include`` mode is unbiased under MCAR (missing completely at random)
at missingness rates from 0--60% for pi, theta_w, theta_h, theta_l,
Tajima's D, dxy, Hudson FST, da, and all Achaz SFS estimators.

Basic Usage
-----------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence

   h = HaplotypeMatrix.from_vcf("data.vcf")

   # Default: per-site valid data
   pi = diversity.pi(h)

   # Conservative: only fully genotyped sites
   pi_excl = diversity.pi(h, missing_data='exclude')

How It Works
------------

Consider a site with 100 haplotypes where 10 are missing (``-1``):

* **include**: Computes allele frequencies from the 90 observed
  haplotypes. The site contributes to the result with ``n_valid = 90``.
* **exclude**: The site is dropped entirely because it has missing data.

For statistics that require a single sample size (e.g., Tajima's D
variance formula), the harmonic mean of per-site sample sizes is used.

Supported Statistics
--------------------

Every public function accepts the ``missing_data`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - include
     - exclude
   * - Diversity (pi, theta_w, theta_h, theta_l)
     - per-site n
     - filter sites
   * - Neutrality tests (tajimas_d, fay_wus_h, H*, E, DH)
     - per-site n, harmonic mean for variance
     - filter sites
   * - Divergence (dxy, fst_hudson, fst_weir_cockerham, da)
     - per-site n
     - filter sites
   * - SFS (sfs, joint_sfs, folded variants)
     - per-site n
     - filter sites
   * - Admixture (patterson_d, f2, f3)
     - per-site n
     - filter sites
   * - Selection scans (ihs, nsl, xpehh)
     - wildcard in SSL
     - filter sites
   * - Haplotype stats (garud_h, haplotype_diversity)
     - wildcard match
     - filter sites
   * - Distance (pairwise_diffs, pca)
     - per-pair norm
     - filter sites
   * - LD (zns, omega)
     - per-site n
     - filter sites
   * - SFS estimators (FrequencySpectrum, diversity_stats_fast)
     - group by n
     - filter sites

LD Estimator Choice
-------------------

For LD statistics, ``zns()`` and ``omega()`` accept an ``estimator``
parameter independent of missing data handling:

* ``estimator='r2'`` (default): naive r-squared.
* ``estimator='sigma_d2'``: unbiased multinomial projection estimators
  (Ragsdale & Gravel 2019), computing sigma_D^2 = D^2/pi^2 with
  falling-factorial corrections. More robust with small or variable
  sample sizes.

.. code-block:: python

   from pg_gpu import ld_statistics

   # Default: naive r-squared
   zns = ld_statistics.zns(h)

   # Unbiased estimator
   zns_unbiased = ld_statistics.zns(h, estimator='sigma_d2')

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

Span Normalization
------------------

Rate estimators (pi, theta_w, dxy, etc.) accept a ``span_normalize``
parameter that controls *how results are expressed*. This is orthogonal
to missing data handling.

``span_normalize`` accepts:

* ``True`` (default): auto-detect the best denominator. If an accessible
  mask is set, divides by accessible bases. Otherwise divides by genomic
  span (chrom_end - chrom_start).
* ``False``: return raw sum (used internally by composite statistics like
  Tajima's D, and by advanced users who need custom normalization).
* ``'per_base'``: explicit genomic span.
* ``'accessible'``: explicit accessible base count (error if no mask).
* ``'per_variant'``: divide by number of variant sites.

.. code-block:: python

   # Per base pair (default -- auto-detects best denominator)
   pi = diversity.pi(h)

   # With accessible mask: auto uses accessible bases
   h.set_accessible_mask("mask.bed", chrom="3L")
   pi = diversity.pi(h)  # per accessible base, automatically

   # Raw sum (no normalization)
   pi_raw = diversity.pi(h, span_normalize=False)

   # Explicit mode
   pi_var = diversity.pi(h, span_normalize='per_variant')

Test statistics (Tajima's D, Fay-Wu's H, FST) do not accept
``span_normalize`` — they are dimensionless by definition.

Accessible Site Masks
---------------------

Genome accessibility masks (from BED files) define which sites are
callable in a sequencing experiment. This matters for normalization:
if only 60% of a region is accessible, per-base diversity estimates
should divide by the accessible base count, not the total span.

pg_gpu integrates accessibility masks into ``HaplotypeMatrix`` and
``GenotypeMatrix`` as a non-destructive filter. When a mask is set,
the ``haplotypes`` and ``positions`` properties transparently return
only variants within accessible regions. The original data is preserved
and the mask can be swapped or removed at any time.

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   h = HaplotypeMatrix.from_vcf("data.vcf.gz")
   print(h.num_variants)  # e.g. 50,000

   # Attach a mask -- only variants in accessible regions are visible
   h.set_accessible_mask("accessibility.bed", chrom="3L")
   print(h.num_variants)  # e.g. 42,000 (filtered)

   # n_total_sites is automatically set to the accessible base count
   print(h.n_total_sites)  # e.g. 30,000,000

   # Masks can also be set at load time
   h = HaplotypeMatrix.from_vcf("data.vcf.gz",
                                 accessible_bed="accessibility.bed")
   h = HaplotypeMatrix.from_zarr("data.zarr", region="3L:1-10000000",
                                  accessible_bed="accessibility.bed")

   # Remove the mask to restore all variants
   h.remove_accessible_mask()
   print(h.num_variants)  # back to 50,000

**Key behaviors:**

* ``set_accessible_mask()`` is non-destructive and returns ``self``
  for chaining. It automatically sets ``n_total_sites`` to the count
  of accessible bases.

* ``get_span('accessible')`` returns the accessible base count for a
  region, used for per-base normalization.

* The mask stays on CPU and uses a lazy prefix-sum for O(1) range
  queries, so windowed analysis over many windows is efficient.

* ``get_subset()`` and ``get_population_matrix()`` read from the
  filtered properties, so child matrices automatically contain only
  accessible variants.

**Interaction with missing data modes:**

Accessibility masks and missing data modes are complementary. The mask
controls *which variants are visible* (a site-level filter based on
genome quality), while ``missing_data`` controls *how missing genotypes
at visible sites are handled* (a sample-level concern). Both can be
active simultaneously:

.. code-block:: python

   h = HaplotypeMatrix.from_vcf("data.vcf.gz",
                                 accessible_bed="mask.bed")
   pi = diversity.pi(h)  # uses accessible mask + per-site valid counts

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
standard neutral model at missing rates from 0--60%.

Component-Level Access
----------------------

For advanced use cases (e.g., custom windowed aggregation), raw pairwise
difference and comparison counts are available via separate functions:

.. code-block:: python

   from pg_gpu.diversity import pi_components
   from pg_gpu.divergence import dxy_components

   # Within-population: (total_diffs, total_comps, total_missing, n_sites)
   diffs, comps, missing, n = pi_components(h.haplotypes)
   pi_manual = diffs / comps

   # Between-population: (total_diffs, total_comps, n_sites)
   pop1_haps = h.haplotypes[h.sample_sets['pop1']]
   pop2_haps = h.haplotypes[h.sample_sets['pop2']]
   diffs, comps, n = dxy_components(pop1_haps, pop2_haps)
   dxy_manual = diffs / comps

Best Practices
--------------

1. **Use include mode** (default) for most analyses. It uses all
   available data at each site and is unbiased under MCAR.

2. **Use exclude mode** when you need all samples to be comparable
   at exactly the same sites (e.g., for certain LD analyses or when
   missingness is non-random).

3. **Use the Achaz framework** (``FrequencySpectrum``) for theta
   estimators and neutrality tests when you want proper handling of
   variable sample sizes via group-by-n or SFS projection.

4. **Use estimator='sigma_d2'** for LD statistics (ZnS, Omega) when
   sample sizes are small or variable. The unbiased estimators correct
   the upward bias inherent in naive r^2.

5. **Check missingness patterns** before analysis with
   ``summarize_missing_data()`` and consider filtering sites with
   very high missing rates.

Example Workflow
----------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, diversity, divergence
   from pg_gpu.windowed_analysis import windowed_analysis

   # Load data with accessible mask
   h = HaplotypeMatrix.from_zarr("data.zarr", region="3L:1-10000000",
                                  accessible_bed="accessibility.bed")

   # Inspect missing data
   summary = h.summarize_missing_data()
   print(f"Missing: {summary['missing_freq_overall']:.1%}")

   # Filter extreme missingness
   h = h.filter_variants_by_missing(max_missing_freq=0.5)

   # Scalar statistics (auto-normalized by accessible bases)
   pi = diversity.pi(h, population="pop1")
   tajd = diversity.tajimas_d(h, population="pop1")
   dxy = divergence.dxy(h, 'pop1', 'pop2')
   fst = divergence.fst_hudson(h, 'pop1', 'pop2')

   # Windowed analysis (accessible mask propagates per-window)
   df = windowed_analysis(h, window_size=50_000,
                          statistics=['pi', 'theta_w', 'tajimas_d',
                                      'fst', 'dxy'],
                          populations=['pop1', 'pop2'])
