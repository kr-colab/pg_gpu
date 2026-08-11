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
Tajima's D, dxy, Hudson FST, da, and all SFS-based estimators.

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
   * - Admixture (patterson_d, f2, f3, f4)
     - per-site n
     - filter sites
   * - Selection scans (ihs, nsl, xpehh)
     - wildcard in shared-site length (SSL)
     - filter sites
   * - Haplotype stats (garud_h, haplotype_diversity)
     - wildcard match
     - filter sites
   * - Distance (pairwise_diffs, pca)
     - per-pair, over jointly non-missing sites
     - filter sites
   * - LD (zns, omega)
     - per-site n
     - filter sites
   * - SFS estimators (FrequencySpectrum)
     - group by n
     - filter sites

Multiallelic Sites
------------------

Most sites in a real dataset have two alleles: the reference base and
one alternate. A few have three or four. Those are called
**multiallelic** sites, and they need a little care.

pg_gpu stores each haplotype's allele as an integer. ``0``
means the reference base, ``1`` the first alternate, ``2`` the second,
and so on. Missing data is ``-1``.

Each allele is counted on its own
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose a site has reference ``A`` and two alternates, ``T`` and ``C``.
Across 10 haplotypes you observe 6 A's, 3 T's, and 1 C.

One way to handle this is to lump the alternates together and say "4 of
10 haplotypes are non-reference". **pg_gpu does not do that.** It keeps
T and C apart and counts each separately: T is at 3/10, C is at 1/10.

Lumping them would be misleading, because T and C are two different
mutations with two different histories. Treating them as one allele at
frequency 4/10 describes a variant that does not exist. Counting them
separately is also what ``tskit`` does, so :math:`\pi`, the theta
estimators, Tajima's D, Fay-Wu's H, Zeng's E, and every SFS function
give the same answer here that ``tskit`` would.

Two things follow from this.

**A three-allele site counts as two mutations.** It takes two mutations
to produce three different alleles, so that site adds 2 to
``segregating_sites`` rather than 1. In general a site with :math:`k`
alleles adds :math:`k - 1`. ``theta_w`` and Tajima's D use the same
count. On ordinary two-allele data this is just the number of variable
sites, as you would expect. On multiallelic data it is larger -- and
larger than what scikit-allel reports, which counts variable sites
rather than mutations.

**Frequencies are per allele.** For the site above, the frequency
spectrum gets one entry at 3 and another at 1 -- not a single entry at
4. Likewise ``max_daf`` reports the most common alternate (3/10 here),
and ``daf_histogram`` records one frequency for T and one for C.

If your data is entirely two-allele, none of this changes anything. A
site with one alternate has nothing to keep apart, so these rules give
the familiar answers.

.. _sfs-conventions:

Three SFS rules that also affect two-allele data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The frequency-spectrum functions follow ``tskit`` on three points.
Unlike everything above, these change the output even when every site
has just two alleles, so they are worth reading if you are comparing
against older results or against another tool.

* **Sites where nothing varies are left out.** If every haplotype
  carries the same allele, the site contributes nothing -- not to the
  first bin, not to the last. This changes ``sfs`` and
  ``daf_histogram`` whenever invariant sites are in your matrix. No
  theta estimator changes, because none of them uses those two bins.

* **Folded spectra are returned as floating point numbers** (``float64``) rather than
  integers. On two-allele data the values still come out whole, so
  only the array's type changes. On a multiallelic site each allele is
  folded on its own and contributes one half, so you will see genuine
  half-counts there. For example, a site with three alleles at counts
  3, 2, and 1 out of 6 folds to ``0.5`` in each of three bins.

* **``joint_sfs_folded`` folds the site once, not each population
  separately.** The site is folded using its overall minor allele, so
  every cell in the result refers to one specific allele.
  scikit-allel folds each population's axis on its own, which can put
  two different alleles into the same cell. The result is a full
  ``(n1 + 1, n2 + 1)`` grid rather than the smaller
  ``(n1 // 2 + 1, n2 // 2 + 1)``.

Which value to trust on multiallelic data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plain functions -- ``diversity.pi``, ``diversity.tajimas_d``, and
friends -- are the ones to trust. They match ``tskit`` exactly.

``FrequencySpectrum`` also offers :math:`\pi`, Tajima's D, and Fay-Wu's
H, but on multiallelic data these are close approximations rather than
exact. The reason is that a frequency spectrum records *how many* sites
sit at each frequency, but not *which alleles came from the same site*,
and you need that grouping to get :math:`\pi` exactly right. On
two-allele data the two routes agree exactly.

Statistics that only work on two-allele sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few statistics are only defined for sites with two alleles. When one
of them throws multiallelic sites away, it tells you how many with a
``BiallelicOnlyWarning``, so you do not mistake a partial answer for a
complete one.

That means ``admixture.patterson_d`` and the
``GenotypeMatrix`` loaders described below. If you want a statistic like
Patterson's D that does handle multiallelic sites, use
``admixture.patterson_f4``, which measures the same thing without the
restriction, at the expense of normalization (it is not bounded in [-1, 1]).

To turn the warning off:

.. code-block:: python

   import warnings
   from pg_gpu import BiallelicOnlyWarning

   warnings.filterwarnings("ignore", category=BiallelicOnlyWarning)

There is a second, unrelated warning called ``MultiallelicCapWarning``.
It does *not* mean a statistic is two-allele-only. It means a windowed
calculation that normally handles multiallelic sites just fine hit its
internal limit of 8 alleles at one site and discarded the site. DNA has 4 bases, so you will
almost certainly never see it.

Some other statistics are two-allele-only but never have to drop
anything, because the restriction was already applied when the data was
loaded. ``relatedness.grm``, ``relatedness.ibs``, the dosage PCA, and
``distance_stats.pairwise_diffs_diploid`` all take a ``GenotypeMatrix``,
which only ever holds two-allele sites. The LD statistics apply their
own filter.

What the ``GenotypeMatrix`` loaders keep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GenotypeMatrix.from_vcf``, ``GenotypeMatrix.from_zarr``, and
``GenotypeMatrix.from_haplotype_matrix`` all agree on what counts as a
two-allele site, so the same data gives you the same matrix no matter
which one you use.

* **A site is kept if at most two different alleles actually show up in
  the data.** What matters is how many distinct alleles are present,
  not which numbers they happen to be. A site where you only see
  alleles ``0`` and ``2`` is kept. So is a site showing only ``1`` and
  ``2``, which can happen after you subset to a few samples and the
  reference allele disappears.

* **The genotype value is a count of the alternate allele**, giving the
  usual 0, 1, or 2 per individual. Allele ``0`` is the reference and the
  alternate is the highest-numbered allele present.

* **Sites with three or more alleles cannot be written as a 0/1/2
  count**, so they are removed. ``from_vcf`` drops the site outright;
  the zarr loaders instead mark the whole site as missing (``-1``),
  which keeps the rows lined up for the streaming reader. Either way
  you get a ``BiallelicOnlyWarning`` with the count.

* **Sites with no variation (a single present allele) are kept.** Loaders load your data as it
  is; throwing away uninformative sites is
  ``restrict_to_segregating``'s job, not theirs.

Ordinary data using only alleles ``0`` and ``1`` is unaffected by all of
this and loads without any warning.

Restricting to biallelic or segregating sites yourself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two orthogonal filters are available. ``restrict_to_biallelic()`` (on a
``HaplotypeMatrix``) keeps sites with at most two distinct present alleles,
leaving the allele codes unchanged -- so ``{0,1}``, ``{0,2}`` and
reference-absent ``{1,2}`` sites are all retained -- and drops sites with three
or more alleles with a ``BiallelicOnlyWarning``. The LD statistics are defined on
biallelic sites and apply this themselves, so you rarely need to call it directly:

.. code-block:: python

   h = h.restrict_to_biallelic()      # drop >=3-allele sites, codes unchanged

``restrict_to_segregating()`` (on either matrix) is unrelated to allele count: it
drops sites where nothing varies, keeping those with at least two distinct
alleles present:

.. code-block:: python

   gm = gm.restrict_to_segregating()  # drop sites that aren't variable

LD Estimator Choice
-------------------

For LD statistics, ``zns()`` and ``omega()`` accept an ``estimator``
parameter independent of missing data handling:

* ``estimator='auto'`` (default): use the unbiased ``sigma_d2``
  estimator when the input is a ``HaplotypeMatrix``, and fall back to
  naive ``r2`` for pre-computed r² arrays or ``GenotypeMatrix`` inputs
  (where ``sigma_d2`` is not available).
* ``estimator='r2'``: always use naive r-squared. Convenient when you
  want the "classical" estimator regardless of input type.
* ``estimator='sigma_d2'``: always use the unbiased multinomial
  projection estimators (Ragsdale & Gravel 2019), computing
  :math:`\sigma_D^2 = D^2 / \pi_2` with falling-factorial corrections.
  More robust with small or variable sample sizes; requires a
  ``HaplotypeMatrix``.

.. code-block:: python

   from pg_gpu import ld_statistics

   # Default 'auto': unbiased sigma_d2 when given a HaplotypeMatrix
   zns = ld_statistics.zns(h)

   # Force naive r²
   zns_naive = ld_statistics.zns(h, estimator='r2')

   # Pre-computed r² array: 'auto' falls back to 'r2'
   r2 = h.pairwise_r2()
   zns_from_array = ld_statistics.zns(r2)

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

HaplotypeMatrix and GenotypeMatrix Utilities
--------------------------------------------

The same utilities are available on both ``HaplotypeMatrix`` and
``GenotypeMatrix`` -- substitute ``gm`` for ``h`` below if you are
working with diploid genotypes.

.. code-block:: python

   # Detect and count missing data
   missing_per_site = h.count_missing(axis=0)
   missing_per_sample = h.count_missing(axis=1)

   # Filter by missing data frequency
   h_clean = h.filter_variants_by_missing(max_missing_freq=0.1)

   # Summary statistics
   summary = h.summarize_missing_data()

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

* The mask covers the union of the BED's extent and the matrix's
  ``[chrom_start, chrom_end]`` range, so BED accessible bases that
  fall outside the variant range (common for variants-only VCFs) are
  not silently dropped. ``n_total_sites`` always equals the full BED
  accessible-base count.

* ``get_span('accessible')`` returns the accessible base count, used
  for per-base normalization. This matches the denominator used by
  ``allel.sequence_diversity(is_accessible=...)``.

* The mask stays on CPU and uses a lazy prefix-sum for O(1) range
  queries, so windowed analysis over many windows is efficient.

* A mask given to ``from_zarr`` as ``accessible_bed`` works the same
  whether the data is loaded all at once or streamed in pieces, and for
  both ``HaplotypeMatrix`` and ``GenotypeMatrix``. Streaming applies the
  mask to each piece as it arrives, so you get exactly the same numbers
  either way.

Site Count Properties
---------------------

After a mask is attached (or ``include_invariant=True`` was passed at
load time), three properties decompose the analysis universe:

* ``n_callable_sites`` -- alias for ``n_total_sites``; the BED span when
  masked, or the matrix length if loaded with ``include_invariant=True``.
* ``n_segregating_sites`` -- polymorphic sites in the matrix
  (``0 < derived_count < n_valid``).
* ``n_invariant_sites`` -- ``n_callable_sites - n_segregating_sites``;
  may include implied invariants outside the matrix when the VCF was
  variants-only.

These satisfy ``n_callable_sites == n_segregating_sites + n_invariant_sites``.
Note that ``num_variants`` is the *physical* matrix row count and is
generally not equal to either ``n_segregating_sites`` (which excludes
monomorphic rows) or ``n_callable_sites`` (which can include implied
invariants).

.. code-block:: python

   h.set_accessible_mask("accessibility.bed", chrom="3L")
   h.n_callable_sites          # e.g. 30,000,000 (BED total)
   h.n_segregating_sites       # e.g. 1,200,000  (polymorphic in matrix)
   h.n_invariant_sites         # e.g. 28,800,000 (callable - segregating)
   h.num_variants              # whatever rows are physically present

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

Span Normalization
------------------

Rate estimators (pi, theta_w, dxy, etc.) accept a ``span_normalize``
parameter that controls *how results are expressed*. This is orthogonal
to missing data handling.

``span_normalize`` accepts ``True`` or ``False``:

* ``True`` (default): auto-detect the best denominator. If an accessible
  mask is set, divides by ``mask.total_accessible`` (the BED span).
  Otherwise divides by the genomic span (1-based inclusive,
  ``chrom_end - chrom_start + 1``).
* ``False``: return raw sum (used internally by composite statistics like
  Tajima's D, and by advanced users who need custom normalization).

.. code-block:: python

   # Per base pair (default -- auto-detects best denominator)
   pi = diversity.pi(h)

   # With accessible mask: auto uses accessible bases
   h.set_accessible_mask("mask.bed", chrom="3L")
   pi = diversity.pi(h)  # per accessible base, automatically

   # Raw sum (no normalization)
   pi_raw = diversity.pi(h, span_normalize=False)

Test statistics (Tajima's D, Fay-Wu's H, FST) do not accept
``span_normalize`` -- they are dimensionless by definition.

SFS Projection
--------------

When samples are missing at different sites the per-site sample size
varies, which complicates statistics that are sensitive to sample size
(notably theta estimators). *Hypergeometric projection* re-expresses an
observed SFS as the SFS that would have been seen if every site had
been called in exactly ``target_n`` randomly chosen samples. The
projection is unbiased and lets you build a single, comparable SFS from
data with mixed sample sizes -- and to compare populations that were
sequenced to different depths. ``FrequencySpectrum`` supports it
following Marth et al. (2004) / the implementation used in
:math:`\partial a \partial i` (Gutenkunst et al. 2009):

.. code-block:: python

   from pg_gpu.diversity import FrequencySpectrum

   fs = FrequencySpectrum(h, population="pop1")
   fs_proj = fs.project(target_n=50)   # project down to n=50
   pi_proj = fs_proj.theta("pi")        # any theta on the projected SFS

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

3. **Use FrequencySpectrum** for theta estimators and neutrality tests
   when you want proper handling of variable sample sizes via group-by-n
   or SFS projection.

4. **The default ``estimator='auto'`` for ``zns()`` and ``omega()``
   uses the unbiased ``sigma_d2`` estimator on ``HaplotypeMatrix``
   inputs**, which corrects the upward bias inherent in naive
   :math:`r^2` -- particularly important with small or variable
   sample sizes. Pass ``estimator='r2'`` explicitly if you want the
   classical naive estimator instead (e.g. for backward comparison
   with a previous analysis).

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
