Changelog
=========

Unreleased
----------

Phase-dependent statistics warn on unphased data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added ``phased`` attribute to ``HaplotypeMatrix`` (which defaults to 
``False``). This tracks whether the haplotypes in the matrix 
represent true phased haplotypes or not. Each loader exposes a 
``phased`` keyword argument: ``from_vcf`` and ``from_zarr`` default 
to ``False``; ``from_ts`` defaults to ``True``. It is possible to
call ``set_phased()`` or ``set_unphased()`` to change it after
construction.

Statistics that depend on phased haplotypes emit an
``UnphasedDataWarning`` when given a ``HaplotypeMatrix`` with 
``phased=False``. These include Garud's H, iHS, XP-EHH, nSL, 
haplotype counts and diversity, haplotype r² and D, distance 
statistics between haplotypes, PCA and local PCA on haplotypes, and the
windowed scans that request any of these. Each call warns once and it 
names an unphased alternative where one exists. For
``garud_h``, ``pairwise_diffs``, the ``dist_*`` moments, ``zns`` and
``omega``, the unphased alternative is to pass a ``GenotypeMatrix``.
One small wrinkle: ``moments_ld.compute_ld_statistics`` with
``vcf_file=`` and ``use_genotypes=False`` now loads the VCF as phased, 
just to avoid the unphased data warning. See :doc:`phasing` for the full
list  of statistics and how to silence the warning.

Because ``from_vcf``, ``from_zarr``, the raw ``HaplotypeMatrix(...)``
constructor, and ``GenotypeMatrix.to_haplotype_matrix`` all give
``phased=False``, going forward all code that uses phased data without 
explicitly marking it as phased will start to see this warning. Load 
with ``phased=True`` or call ``set_phased()`` after loading.

Sites with more than two alleles are now handled correctly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistics used to lump every alternate allele into one "non-reference"
group. That is fine when a site has two alleles and wrong when it has
three or four. Each allele is now counted on its own, which is what
``tskit`` does. See :doc:`multiallelic` for what this means in
practice.

**If your data only has two alleles per site, nothing here changes your
results**, except where the next section says so.

* Diversity and frequency spectra: :math:`\pi`, ``theta_w`` /
  ``theta_h`` / ``theta_l``, ``tajimas_d``, ``fay_wus_h``, ``zeng_e``,
  ``segregating_sites``, ``singleton_count``,
  ``heterozygosity_expected``, and every SFS function.
  ``segregating_sites`` now counts mutations rather than variable
  sites, so a site with three alleles counts as 2. ``theta_w`` and
  Tajima's D use that count too.
* Divergence: ``dxy``, ``da``, ``fst_hudson``, ``fst_nei``,
  ``fst_weir_cockerham``, and ``pbs``. Added ``fst_tskit``, also
  available as ``fst(method='tskit')``.
* Admixture: ``patterson_f2`` and ``patterson_f3``. Added
  ``patterson_f4``. ``patterson_d`` only works on two-allele sites and
  now says so.
* Selection: ``ihs`` and ``nsl`` score each alternate allele against
  the reference separately, so they return one column per alternate
  allele when a site has more than two. ``min_maf`` now applies to each
  allele's own frequency. The promotion is matrix-wide: one site with a
  third allele anywhere makes the whole return two-dimensional, so
  ``plt.plot(pos, ihs)`` and ``np.argmax(ihs)`` change meaning; filter
  with ``restrict_to_biallelic`` first for the flat scan. A matrix with
  no alternate allele at all returns all ``NaN``.
* Distances: pairwise distances count how many sites two haplotypes
  differ at, instead of doing arithmetic on the allele numbers.
  ``decomposition.pairwise_distance`` uses the same count.
* Relatedness: added ``genetic_relatedness``, which matches
  ``tskit``'s function of the same name, including its ``proportion=True``
  default; ``polarised`` stays ``False`` by default, since a haplotype
  matrix has no ancestral state to assume one allele is derived. Under
  missing data, a pair's sum is rescaled to the full analyzed column count
  from the columns it actually shares, so it does not shrink with that
  pair's own call rate; a pair with no column called in both is NaN. With
  ``proportion=False``: an analysis with no column at all is 0.0, and NaN
  when the span is unmeasurable. With ``proportion=True``: NaN whenever no
  site segregates, whether or not any column exists.
* Windowed analysis: every windowed statistic now gives the same answer
  as running the plain function on that window's variants.
* LD statistics operate on only sites with two present alleles, but allow
  arbitrary integer coding. This is a break from parity with ``moments.LD``,
  which restricts to sites with ``{0, 1}`` coding only.
  ``pairwise_r2`` keeps its full square shape and returns dropped sites
  as ``NaN`` rows and columns, ``windowed_r_squared`` drops their pairs
  from its bins, and ``locate_unlinked`` returns ``False`` for them.
  ``moments_ld.compute_ld_statistics`` computes its heterozygosity
  terms on the same site set and 0/1 recoding as its LD sums, so both
  halves of the output describe the same sites.

Results that change even for two-allele data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read this section if you are comparing against older pg_gpu results.

* Haplotype identity with missing calls has one rule on every path.
  ``garud_h``, ``moving_garud_h``, the windowed Garud's H columns,
  ``haplotype_count``, ``haplotype_diversity``,
  ``diplotype_frequency_spectrum`` and ``mu_ld`` used to treat ``-1`` as
  a wildcard with first-match grouping. That grouping depended on row
  order and merged haplotypes that share no called site, so H1 went up.
  The windowed engine instead treated ``-1`` as one more allele, so
  almost every haplotype was distinct and H1 went down. Now the complete
  haplotypes (no missing call) set the distinct haplotypes, and EM
  splits each incomplete haplotype across the complete ones it matches
  at its called sites. This gives the maximum-likelihood frequencies when
  calls are missing completely at random, and does not depend on row
  order. An incomplete haplotype that matches no complete one is a
  distinct haplotype of its own. A window with no complete haplotype returns NaN.
  Values change on any data with missing calls; data with no missing
  calls is unchanged. See :doc:`missing_data`. The windowed
  ``haplotype_count`` column is now float64, since a window can be NaN.
* ``moving_garud_h`` with ``missing_data='exclude'`` dropped every site
  with a missing call from the whole matrix before it tiled the windows,
  so windows moved and the window count could shrink. Windows now stay
  on the full variant grid and each drops only its own missing sites,
  the same as ``windowed_analysis``.
* Haplotype rows now follow one order everywhere: sample ``i`` owns rows
  ``2i`` and ``2i + 1``. ``from_ts`` already used this order, while
  ``from_vcf`` and ``from_zarr`` grouped all of the first gametes ahead of
  the second ones. Statistics that reconstruct individuals -- ``ibs``,
  ``grm``, ``fst_weir_cockerham``, ``heterozygosity_observed``,
  ``GenotypeMatrix.from_haplotype_matrix``, and genotype-mode LD -- now
  read the same order the loaders write, so their values change for data
  loaded from VCF or zarr. Statistics that treat rows as independent
  gametes are unaffected, as long as the same rows are selected.
  ``sample_sets`` built by ``load_pop_file`` now lists ``2i`` and
  ``2i + 1`` for each member, and hand-written index lists that assumed
  the previous order need updating.
* Region strings now include their end position, matching samtools,
  tabix, and bcftools: ``region='X:1-1000000'`` keeps a variant at
  1,000,000, and ``'X:500-500'`` names one position. The zarr loaders,
  ``ZarrGenotypeSource``, and ``allel_zarr_to_vcz`` used to stop one
  base short, so a variant on the window edge went missing;
  ``from_vcf`` reads through tabix and was already inclusive.
  ``materialize(region=(left, right))`` is now a closed interval too --
  it keeps a variant sitting exactly on ``right``, matching the loader
  convention. Windowed-analysis windows remain half-open, and
  ``pg_gpu.zarr_io.parse_region`` still turns a region string into that
  half-open ``(start, stop)`` form. Every loader also takes the other
  samtools forms now: ``'X'`` selects a whole chromosome from a
  multi-contig store, ``'X:1000'`` runs from a position to the chromosome
  end, and thousands separators (``'X:1,000-2,000'``) are allowed.

* ``from_zarr`` now reports ``chrom_start`` / ``chrom_end`` as the
  requested region, matching ``from_vcf``: ``region='X:1-1000000'`` gives
  ``chrom_start=1`` and ``chrom_end=1000000`` for both ``HaplotypeMatrix``
  and ``GenotypeMatrix``, on the eager and streaming paths. With no
  ``region`` the bounds stay the first and last variant position (also as
  ``from_vcf`` does). The loaders used to set the bounds to the first and
  last variant even when a region was given, so a span-normalized
  statistic over a region used a denominator that stopped at the variant
  hull; it now covers the whole requested region. A streaming
  ``accessible_bed`` mask is likewise resolved over the full region rather
  than the variant span, so its accessible-base denominator changes to
  match.
* ``windowed_analysis`` on a streaming matrix now returns a row for every
  window in the requested region, including windows with no variants
  (reported with ``n_variants=0``), instead of dropping empty chunks. The
  streaming result now matches the eager path row-for-row over the same
  region. One caveat: a non-empty window whose sites are all missing under
  ``missing_data='exclude'`` is currently reported the same way as a
  genuinely empty window; distinguishing the two (e.g. zeros vs. NaN) is a
  possible follow-up.
* Row lists are validated at assignment and at resolution: assigning
  ``sample_sets`` and passing a row list as a population argument both
  raise ``ValueError`` for out-of-range or duplicated rows instead of
  silently producing wrong numbers. The statistics that pair rows into
  individuals (``fst_weir_cockerham``, ``heterozygosity_observed``,
  windowed ``fst_wc``, and ``GenotypeMatrix.from_haplotype_matrix``)
  warn when a list does not carry each sample's two rows adjacently;
  gamete statistics accept any list, as before. An empty set also
  raises -- a population must name at least one row, matching tskit --
  and ``load_pop_file`` drops a population with no member in the matrix
  instead of storing it empty. Duplicates are a hard error, so
  bootstrapping over individuals by repeating rows in a set is not
  possible; resample windows or blocks instead.
* Streaming matrices follow one row space end to end: a stream's
  ``sample_sets`` and ``materialize(sample_subset=...)`` both speak
  haplotype rows on a haplotype stream and individual rows on a genotype
  stream, so ``materialize(sample_subset=stream.sample_sets[pop])``
  works on either class. This changes what a genotype stream's
  ``sample_subset`` means: it took haplotype columns before and takes
  individual rows now, so callers that passed columns must halve their
  values. The genotype stream's sets previously held
  haplotype columns that indexed past its individual axis. Assigning
  ``sample_sets`` on a stream validates like the eager classes, and pop
  files validate once at stream construction.
* Windowed ``fst_wc`` now gives the same estimate as
  ``divergence.fst_weir_cockerham``. It used to treat the data as haploid
  and lump every alternate allele together, so it returned a different
  number from the plain function on the same variants. Expect windowed
  values to move: the old ones were off by a small fixed amount, which is
  under a few percent once FST is above about 0.02 and much larger when
  FST is near zero. The plain function is unchanged.
* Frequency spectra: sites where nothing varies no longer contribute,
  folded spectra are returned as ``float64`` rather than integers, and
  ``joint_sfs_folded`` returns a full ``(n1 + 1, n2 + 1)`` grid folded
  by the site's overall minor allele.
* ``pca`` and ``randomized_pca`` now take haplotypes and give every
  allele its own column. The old diploid behavior is now
  ``pca_dosage`` / ``randomized_pca_dosage``, which take a
  ``GenotypeMatrix``. The ``scaler`` argument is gone, and each
  function raises a ``TypeError`` on the wrong matrix type.
  ``pca`` accumulates its Gram over variant chunks when the standardized
  matrix would not fit in GPU memory, so the exact decomposition runs at
  any matrix size.
  ``local_pca`` and ``lostruct`` changed the same way. They no longer
  match the R ``lostruct`` package number for number -- the eigenvalue
  scale differs -- though the component directions still agree closely,
  so plots and outlier detection are unaffected. Window eigenvectors
  are sign-aligned against the previous window, so individual entries
  can flip sign relative to older output, and ``LocalPCAResult`` no
  longer carries a ``scaler`` field.
* ``grm`` and ``ibs`` now require a ``GenotypeMatrix`` and reject
  haplotypes. ``grm(GenotypeMatrix, population=...)`` used to raise an
  error and now works.
* Windowed ``daf_hist`` is now a histogram normalized to sum to 1, and
  windowed ``mu_sfs`` divides by the number of variable sites.
* These statistics now return NaN when they are undefined. Before, they
  returned 0.0, which looks like a real value:

  - ``zns``: no pair of sites has a defined value;
  - ``omega``: fewer than five sites have a defined value, or no
    partition has a positive cross-partition LD sum;
  - ``mu_ld``: fewer than two segregating sites, or no haplotype;
  - ``mu_sfs``: no variable site;
  - ``mu_var``: a window length that is not positive;
  - ``haplotype_diversity``: fewer than two haplotypes.

  ``max_daf(missing_data='exclude')`` returned ``-1.0``, not even a
  plausible-looking real value, when no site was complete; it is NaN now.
  The windowed engines (``windowed_analysis``, ``windowed_statistics_fused``,
  ``windowed_statistics_fused_chunked``) had the same gap one level up: a
  window whose sites were all excluded gave ``0.0`` rather than NaN,
  indistinguishable from a window with no variant to begin with. Both now
  agree with the scalar function.

  With the default ``sigma_d2`` estimator, ``zns`` and ``omega`` leave
  out a pair that they cannot estimate (fewer than four haplotypes
  called at both sites, or a pi2 estimate of 0). Before, ``zns`` counted
  a pair with a pi2 estimate of 0 as 0, and ``omega`` counted every such
  pair as 0. ``mu_ld`` now uses the segregating sites only,
  as RAiSD does. Monomorphic sites no longer move its split point, so
  its value changes for data that contains invariant sites.
  ``zx`` is NaN when one of its ZnS terms is NaN.

  The windowed columns change the same way. The windowed ``zns``,
  ``omega``, ``mu_ld`` and ``dist_*`` columns skipped every window with
  fewer than four variants. Now a window with one to three variants
  gets the value that the scalar function gives for its variants.
* The scalar ``max_daf(missing_data='include')`` now needs at least two
  calls at a site to count it, matching ``pi``, ``theta_w``, and both
  windowed engines, which already worked this way. A single stray call
  of the derived allele used to be enough to set the whole-region value
  to that call's own frequency, as high as 1.0; it now takes the best
  value among sites with their own pair, the same answer the windowed
  path already gave.
* In ``windowed_statistics``, ``singletons`` now counts only alternate
  alleles seen once (it used to also count reference alleles seen
  once), and ``segregating_sites`` counts mutations.
* ``dxy(span_normalize=False)`` and ``da(span_normalize=False)`` return
  the raw sum, matching every other statistic. They used to divide by
  the number of sites.
* ``fst_hudson``, ``fst_tskit``, ``da``, and ``pbs`` were biased upward
  under missing data when a population had only one call at a site: the
  site counted toward the between-population term without counting
  toward that population's own within-population term, inflating Hb
  relative to Hw. A site now needs its own pair (at least two calls) in
  both populations to count toward any of these, in every engine
  (scalar, windowed, and the fused kernels). ``dxy`` is unaffected -- it
  has no within term to stay aligned with, so it keeps needing only one
  call per side. A population with a single haplotype can no longer
  produce a defined ``fst_hudson``/``da``/``pbs`` value against anything,
  since it never has its own pair; NaN where data allowed a number
  before.
* Scalar ``mu_var`` divides by ``get_span()``, the length that ``pi``
  uses. It used to divide by ``chrom_end - chrom_start``, one base short
  of the inclusive region, and it ignored an accessible mask. With or
  without a mask, or with ``n_total_sites``, it now agrees with windowed
  ``mu_var``.
* The bp window grid now covers the base at ``chrom_end``. When
  ``chrom_end - chrom_start`` was a multiple of ``step_size``, a variant at
  ``chrom_end`` was in no window; a 1-bp trailing window now holds it, and
  a one-variant matrix with no bounds gets one window on every engine.
  Bounds are inclusive, so a half-open ``chrom_end`` (one past the last
  base) now adds an empty 1-bp window.
* With ``n_total_sites`` set and no mask, the scatter engines gave each
  window a share of ``n_total_sites`` over ``chrom_end - chrom_start``
  bases, one short of the inclusive region. They now divide by
  ``chrom_end - chrom_start + 1``, so the shares add up to
  ``n_total_sites``.
* With ``n_total_sites`` set and no mask, each windowed engine divided a
  window by a different number: the scatter engines by the window's
  share of ``n_total_sites``, the fused and chunked engines by the window
  length, and the per-window fallback by the ``n_total_sites`` of the
  whole region (so its per-base values were too low by about the number
  of windows). Every engine now uses the share, in proportion to the
  window length, for ``bp``, ``snp`` and ``regions`` windows. The shares
  of windows that tile the region add up to ``n_total_sites``, the
  denominator of the scalar function. Windowed values from the fused,
  chunked and fallback engines change for such a matrix.
* ``include_invariant=True`` (``HaplotypeMatrix.from_vcf``,
  ``GenotypeMatrix.from_vcf``, ``GenotypeMatrix.from_zarr``) now makes
  the record positions the accessible mask, so each window divides by
  its own called bases. The streaming ``GenotypeMatrix.from_zarr`` path
  used to ignore the flag, so ``streaming='auto'`` gave a different
  denominator when the data did not fit in memory; both paths now build
  the same mask. ``n_total_sites`` is the count of distinct
  called positions (a split multiallelic site counts once), and only
  those inside ``accessible_bed`` when one is given. With the mask,
  ``ihs`` and ``xpehh`` scale gaps by their called fraction, not by
  ``gap_scale``.
* ``GenotypeMatrix.from_vcf`` set its region bounds from the records it
  kept after dropping multiallelic ones, so a multiallelic first or last
  record shortened the region. The bounds now span every loaded record,
  as in ``HaplotypeMatrix.from_vcf``.
* ``ihs`` and ``xpehh`` read a matrix's accessible mask as if it
  started at position 1. A mask that starts elsewhere (a region load, or
  a BED that does not start at the first base) gave each gap the
  accessible count of the wrong bases. They now use the mask's offset.
* ``get_subset_from_range`` with a range that starts before the
  accessible mask does shifted the subset's mask, so it marked the wrong
  bases accessible. The subset's mask now keeps its own start.
* ``get_subset_from_range`` on a matrix with ``n_total_sites`` and no
  mask dropped ``n_total_sites``, so the subset divided by its length.
  It now gets its share of ``n_total_sites``, by the rule the windowed
  engines use, so a windowed value equals the plain function on the
  window's range.
* Streaming chunks and ``materialize`` slices got neither the stream's
  accessible mask (``materialize``) nor its ``n_total_sites`` (both).
  Each now gets the mask, or else its share of ``n_total_sites``, so a
  streamed value equals the eager one. ``windowed_analysis`` on a stream
  that has a mask now ignores ``accessible_bed``, as it does on an eager
  matrix; it used to put the BED over the stream's mask.
* A bool ``is_accessible`` array has element 0 at position 1, as in
  scikit-allel, for ``ihs``, ``xpehh``, ``windowed_statistics_fused`` and
  ``windowed_statistics_fused_chunked``. The windowed functions read
  element 0 as position 0, one base off from ``ihs`` and ``xpehh``.
* The biallelic and segregating filters keep the parent matrix's
  chromosome bounds. The old filter moved the bounds to the first and
  last surviving variant, which quietly changed the denominator of any
  span-normalized statistic computed afterwards.
  ``exclude_missing_sites`` (and so every statistic under
  ``missing_data='exclude'``) and ``filter_variants_by_missing`` follow
  the same rule and also keep the accessibility mask: filtering sites
  changes neither the chromosome extent nor which bases are
  accessible.
* ``decomposition.pairwise_distance`` supports ``euclidean``,
  ``sqeuclidean``, and ``cityblock``. Other metrics now raise
  ``NotImplementedError``, and passing a ``GenotypeMatrix`` raises a
  ``TypeError``.
* ``pairwise_r2`` and ``ld_statistics.r2_matrix_diploid`` return
  ``NaN`` instead of ``0`` for a pair whose r-squared is undefined -- a
  site where nothing varies; for diploid dosages that includes an
  all-heterozygous site -- with either estimator. Reductions over a
  stored matrix need
  ``nanmean`` or friends now.
* ``windowed_analysis`` with exactly two populations names the
  two-population columns the same way on every path (``fst_hudson``);
  the non-fused fallback used to suffix them
  (``fst_hudson_pop1_pop2``).
* ``fst_weir_cockerham`` with a population holding a single row
  returns 0 (after the unpaired-rows warning) where it used to fall
  back to a haploid estimate. One row is half an individual and gives
  the diploid analysis no within-population variance to estimate.

Bug fixes
~~~~~~~~~

* The per-window fallback of ``windowed_analysis`` (used under
  ``missing_data='exclude'`` for most statistics, for ``snp`` windows,
  and for requests the fast engines cannot serve) skipped a window or
  region with no variant, while the scatter and fused engines gave it a
  row, so the rows of a request depended on the engine. The fallback now
  gives every window a row, with the values the other engines give an
  empty window (for example ``pi`` 0.0, ``tajimas_d`` NaN). A custom
  statistic gets NaN on an empty window, without a call. With two or
  more populations, its empty rows also use the per-population column
  names, as its other rows do.
* A scalar statistic given a streaming matrix failed deep inside with a
  bare ``AttributeError`` (no attribute ``device``), which did not say
  what to do. Every function that needs the whole matrix in memory now
  raises ``NotImplementedError`` that points to ``windowed_analysis``
  for a genome scan and to ``.materialize(region=(lo, hi))`` for a
  sub-region. ``mu_var``, which returned a value on a stream, now
  rejects one too: the stream's variant count ignores its accessible
  mask, while its span reads it. ``pca_dosage``,
  ``randomized_pca_dosage`` and ``sigma_d2_geno`` raise the same error
  in place of a ``TypeError`` about the matrix class.
* ``pairwise_diffs_diploid`` (and so ``dist_moments`` on a
  ``GenotypeMatrix``) raised ``TypeError`` for a population given as a
  list of rows. It now selects rows as the haploid path does.
* ``windowed_analysis`` with ``missing_data='exclude'`` raised
  ``Unknown statistic`` for ``zns``, ``omega``, ``mu_ld``, ``mu_var``,
  ``mu_sfs``, ``daf_hist``, the ``snp_dist_*`` and ``dist_*``
  statistics, and ``mean_nsl``. It also raised for ``theta_h``,
  ``singletons``, ``fay_wu_h`` and ``max_daf`` when they were requested
  with a Garud statistic, or with two populations and no two-population
  statistic. These requests now use the fused engine, which keeps only
  the sites with no missing call in the rows the request reads. Called
  directly, ``windowed_statistics_fused`` and
  ``windowed_statistics_fused_chunked`` ignored ``'exclude'`` for every
  statistic except the Garud family and ``omega``; they now apply it to
  every statistic. Under ``'exclude'``, ``n_variants`` now counts only
  the complete sites on every path; a Garud-only request and the
  per-window fallback counted every site. Called with a two-population
  statistic and no ``pop1`` or ``pop2``, the fused engines raised
  ``TypeError`` under ``'exclude'``; they now raise the same
  ``ValueError`` as under ``'include'``.
* ``windowed_analysis`` raised ``TypeError`` when a population was given
  as a list of rows and the request took the per-window fallback (for
  example ``['pi', 'garud_h1']`` with ``missing_data='exclude'``). The
  fallback now accepts row lists and names their columns as it names
  the columns of a named population.
* ``tally_gpu_haplotypes`` and ``tally_gpu_haplotypes_two_pops`` gave
  wrong counts on complete data when more than 127 haplotypes carried
  the alternate allele at both sites of a pair. The pair count used an
  ``int8`` matrix product, which wraps above 127. The tally now uses
  ``float32``, which is exact up to 2^24 haplotypes, also with
  ``CUPY_TF32=1``, and ``float64`` above that. Both methods also counted
  raw allele codes, so ``{0,2}`` and ``{1,2}`` codings gave wrong
  counts. They now recode each site to a 0/1 indicator first, as
  ``windowed_r_squared`` does. Counts are ``int32`` on every path.
  ``tally_gpu_haplotypes_two_pops_with_missing`` no longer loops over
  pairs in Python. ``tally_gpu_haplotypes_two_pops`` now looks for
  missing calls only in the two populations, so missing data in a third
  population no longer turns ``n_valid1`` and ``n_valid2`` into arrays.
* ``pairwise_r2`` and the naive ``r2`` estimator behind ``zns``/``omega``
  computed the joint 11-frequency over the pairwise-complete sample but
  each site's own frequency over its separate, larger marginal sample,
  biasing ``D`` and ``r2`` under structured missing data with
  ``missing_data='include'``. Both frequencies now come from the same
  pairwise-complete sample. The diploid dosage-correlation path
  (``_r2_matrix_diploid``, reached by ``zns``/``omega`` on a
  ``GenotypeMatrix`` with ``estimator='r2'``) had the same bug in its
  continuous form -- mean/variance from each site's own marginal valid
  set instead of the pair's jointly-valid individuals -- and is fixed
  the same way.
* ``omega`` and ``zns`` mishandled a single pair left undefined by the
  fix above (both sites otherwise fine, just no -- or locally
  degenerate -- jointly-valid sample): the NaN propagated through
  ``omega``'s prefix sum and collapsed results to ``0.0``, and both
  statistics divided by a closed-form pair count that assumed every
  pair was defined. Both now count only their actually-defined pairs,
  in the full-matrix path and in ``zns``'s tiled naive-``r2`` path
  (``_zns_tiled``/``_zns_from_precomputed``).
* ``get_population_matrix`` returned a subset with no sample names and no
  QC ``fields``. It now keeps both: per-variant fields keep every variant
  of the subset, and per-genotype fields and ``samples`` keep the
  population's individuals. On a ``HaplotypeMatrix`` that works only when
  the population's rows pair into whole individuals (sample ``i`` at rows
  ``2i`` and ``2i + 1``); otherwise the subset drops them and warns.
  ``samples`` or a per-genotype field that does not hold one entry per
  individual is also dropped with a warning, since picking by individual
  would mislabel it. ``metadata=False`` skips the copy.
* ``zeng_dh``, scalar and windowed, returned 0.0 ("no sweep") when
  Tajima's D or Fay & Wu's H was undefined because of too few
  segregating sites. It now returns NaN in that case.
* ``diversity_stats`` returned NaN for ``fay_wus_h`` with fewer than three
  segregating sites, a gate meant only for the variance-based tests. It
  now matches ``fay_wus_h``.
* Windowed ``fay_wu_h`` was a raw sum, not a per-base rate, when the
  request went to the fused engine (for example together with a Garud's H
  statistic), so the same column changed units with the rest of the
  request. It is now per base on every engine, like ``pi`` and
  ``theta_h``.
* ``FrequencySpectrum.neutrality_test`` used the sample size with the
  most sites for the Achaz variance. It now uses the harmonic mean of the
  per-site sample sizes, as ``tajimas_d``, ``normalized_fay_wu_h`` and
  ``zeng_e`` do, so it matches them under missing data.
* ``FrequencySpectrum.theta`` raises ``ValueError`` for a custom weight
  vector whose length is not ``n + 1``; it used to cut the vector to fit.
  ``FrequencySpectrum.project`` warns, with a count, when it leaves out
  sites that have fewer valid haplotypes than the target.
* The two-population distance statistics (``snn``, ``dxy_min``,
  ``gmin``, ``dd``, ``dd_rank`` and ``distance_based_stats``) used raw
  difference counts, and a pair can only differ at the sites both
  haplotypes call. So under missing data a haplotype with many missing
  calls looked close to everything and could set ``dxy_min``. Each
  pair's count is now scaled to all sites (``pairwise_distance_matrix``
  returns the scaled values), and a pair with no site in common is
  skipped. ``dd`` now takes pi from the same distances, so it matches
  ``distance_based_stats``; before, the two disagreed under missing data.
  Results on data with no missing call are unchanged.
* A pair with no site called in both samples gets NaN, not 0.0, from
  ``pairwise_diffs`` (haploid and diploid),
  ``decomposition.pairwise_distance`` and ``relatedness.ibs``.
  ``dist_var``, ``dist_skew`` and ``dist_kurt``, and their windowed
  columns, skip these pairs; a moment is NaN when missing data leaves
  too few defined pairs for it.
  ``pcoa`` raises ``ValueError`` on a NaN distance. ``lostruct`` gives a
  window that ``pc_dist`` cannot compare (too few variants, or no
  variation) a NaN ``mds`` row; before, such a window made ``pcoa`` fail
  with ``LinAlgError``. ``lostruct`` now raises ``ValueError`` when fewer
  than two windows can be compared, or when corners are requested and
  the MDS has fewer than two dimensions.
* ``grm`` divided every pair by the dataset-wide count of called,
  polymorphic SNPs instead of that pair's own jointly-called count, so an
  individual's relatedness (including to itself) shrank with its own
  missing-call rate. It now divides each pair by its own count, and gives
  NaN for a pair with no jointly-called polymorphic SNP, and for the
  whole matrix when no SNP is polymorphic (before, both were 0.0). Values
  on data with no missing call are unchanged.
* ``pbs`` counted a population's own diversity at a site where another
  population in the pair had no data, which pulled that pair's FST down;
  a single such site could change the sign of PBS. PBS now uses, for all
  three FSTs, only the sites where all three populations have data, so the
  three branches describe the same loci. Sites where every population has
  at least one call are unchanged.
* ``garud_h`` on a ``GenotypeMatrix`` ignored ``missing_data``, and
  ``windowed_analysis`` with a Garud's H statistic or ``haplotype_count``
  under ``missing_data='exclude'`` raised ``Unknown statistic``. Both now
  apply ``missing_data``.
* ``fst_hudson``, ``fst_weir_cockerham``, ``fst_tskit``, and ``fst_nei``
  returned ``0.0`` for an undefined ratio (no site with data in both
  populations) while every windowed engine already used ``NaN`` for the
  same case; they now return ``NaN`` too.
* Several ``HaplotypeMatrix``/``GenotypeMatrix`` methods that subset or
  convert a matrix by variant (``get_subset``, ``get_subset_from_range``,
  ``restrict_to_biallelic``, ``restrict_to_segregating``,
  ``exclude_missing_sites``, ``filter_variants_by_missing``,
  ``from_haplotype_matrix``, ``to_haplotype_matrix``) dropped ``samples``
  and/or ``fields`` on the result instead of carrying/slicing them as
  ``filter`` already did. All now match ``filter``.
* Windowed ``fst_wc`` could only be computed by the fused CUDA kernel
  (``missing_data='include'``) or a slow per-window scalar loop
  (``'exclude'``), since nothing exposed Weir-Cockerham's per-site
  decomposition for the vectorized scatter engine to reuse. It now shares
  that decomposition with the scalar function and runs through the same
  fast scatter path as the other two-population statistics under both
  modes.
* ``da``, and the windowed scatter engine's ``fst``/``fst_hudson``, computed
  their within-population terms independently of the other population
  instead of restricting every term to sites where both populations have
  data (as ``fst_hudson`` and the fused kernel already did), so they gave
  the wrong answer whenever one population was entirely missing at a site.
  All three now share one site set, matching the fused kernel.
* ``fst_tskit`` summed its within-population term over every site with no
  such restriction at all, so it could count a site's within-population
  diversity toward Hw even where the other population had no data there
  to compare against. It now shares the same site set as ``fst_hudson``.
* Windowed Garud's H made three float64 copies of the haplotype matrix
  (109 GB each for 2,940 haplotypes across 4.7 million sites) and, for
  a Garud-only request, a transposed int8 copy on top. Each window is
  now hashed straight from the int8 matrix, in batches sized to free
  GPU memory, and Garud-only scans skip the transposed copy.
* A window with no variants borrowed the next window's first variant
  and reported its Garud H; a grid whose windows all lay past the last
  variant raised ``IndexError``. Such a window now reports one
  haplotype (``garud_h1``, ``garud_h12`` and ``garud_h123`` of 1,
  ``garud_h2h1`` of 0, ``haplotype_count`` of 1) on tile and sliding
  grids alike.
* Windowed Garud's H sorted each window's haplotypes in shared memory
  and left every haplotype past the 2,048th unsorted, so haplotype
  counts and H statistics were wrong for larger cohorts, and the kernel
  failed outright past 3,072 haplotypes. The sort now runs in global
  memory with no limit on the haplotype count.
* ``haplotype_diversity``, ``haplotype_count``, ``garud_h`` and
  ``moving_garud_h`` no longer make a float copy of the matrix; they
  share the exact haplotype hash of the windowed scan.
  ``moving_garud_h`` rejects windows outside the matrix instead of
  reading past it.
* Windowed ``tajimas_d``, ``normalized_fay_wu_h``, ``zeng_e``, and
  ``zeng_dh`` computed their null variance from the nominal haplotype
  count instead of the harmonic mean of per-site valid counts the scalar
  functions use, so values diverged from the scalar reference under
  missing data. All four windowed engines now use the same per-window
  effective sample size the scalar functions do.
* The fused engine's ``tajimas_d`` built a temporary proportional to the
  window-overlap depth times the variant count, large enough to run out
  of memory on large, deeply overlapping window grids; it and the fused
  ``theta_w`` now come from a small per-window accumulator computed
  inside the kernel itself, and agree with the scalar reference under
  missing data (previously only ``tajimas_d``'s variance term did).
* ``HaplotypeMatrix.pairwise_r2`` and ``windowed_r_squared`` rejected
  ``estimator='auto'`` -- the default name for the LD estimator
  everywhere else -- with an "unknown estimator" error. They accept it
  now (it means ``'r2'`` there), sharing the one estimator resolver.
* ``mu_ld`` and the ``rogers_huff_r`` family raised a bare
  ``AttributeError``/``TypeError`` on a streaming matrix; they now give
  the same "call ``materialize``" message ``zns``/``omega`` do.
* ``diversity_stats`` listed the Achaz eta-family estimators (``eta1``,
  ``eta1_star``, ``minus_eta1``, ``minus_eta1_star``) as available but
  raised ``Unknown statistic`` when asked for them. It now computes them
  through the per-allele scalar path, giving the same values as
  ``FrequencySpectrum.theta`` on both biallelic and multiallelic data.
* ``zns`` and ``omega`` accepted ``estimator='rogers_huff'`` on a
  ``HaplotypeMatrix`` and silently computed naive ``r2`` instead. They
  now pair adjacent haplotypes into 0/1/2 dosages and compute the
  dosage correlation.
* ``zns`` and ``omega`` reject an estimator their input cannot compute:
  the estimator names raise on a pre-computed r² array
  (``estimator='auto'`` there means ``r2``, as documented), and a
  streaming matrix gets a clear error pointing at ``materialize``.
* ``zns`` under the default ``sigma_d2`` estimator silently ignored
  ``missing_data='exclude'`` and returned the ``include`` result,
  because the estimator was smuggled through the ``missing_data``
  argument. ``exclude`` is now applied before the estimator on every
  path, matching ``omega``.
* ``zns`` and ``omega`` on a ``GenotypeMatrix`` scored sites with no
  dosage variance -- monomorphic sites, and all-heterozygous sites --
  as r^2 = 0 instead of undefined, which diluted ZnS and inflated
  omega. Both statistics now drop those sites, as the haplotype path
  already did.
* ``patterson_d`` kept sites where one of the four populations had no
  called gametes, treating the missing population as frequency 0, which
  can flip D's sign. Such sites are now excluded from both the
  numerator and denominator (num = den = 0), matching scikit-allel,
  which returns nan there. ``moving_patterson_d`` and
  ``average_patterson_d`` share the fix. ``missing_data='exclude'`` was
  already correct.
* ``HaplotypeMatrix.from_ts`` laid out rows in ``ts.samples()`` order and
  assumed each individual's two nodes were adjacent there. tskit does not
  promise that (``simplify`` renumbers samples freely), so on a reordered
  tree sequence every statistic that rebuilds individuals from adjacent
  rows (``fst_weir_cockerham``, ``heterozygosity_observed``,
  ``GenotypeMatrix.from_haplotype_matrix``) was silently wrong. Rows now
  follow the individuals, so individual ``i`` owns rows ``2i`` and
  ``2i + 1`` whatever the sample order; ordinary msprime output is
  unchanged.
* ``divergence.zx`` computed each of its three ZnS values on a different
  site set, because ``zns`` drops the sites that are multiallelic within
  the matrix it is handed. The whole matrix is now restricted to
  biallelic sites before any ZnS is computed, and one
  ``BiallelicOnlyWarning`` reports the count.
* Pairwise distances treated a missing genotype as if it were a
  reference homozygote, so a missing call could count as a difference.
  Missing data is now skipped on both sides of the calculation.
* ``GenotypeMatrix.from_zarr`` ignored ``accessible_bed`` when
  streaming, so streaming ``grm`` and ``ibs`` silently used every
  variant while the non-streaming path filtered correctly. Both now
  apply the mask, and results agree exactly.
* The three ``GenotypeMatrix`` loaders disagreed about which sites to
  keep and how to build genotype values, so the same data could give
  different matrices. They now share one definition. Sites showing only
  alleles ``0`` and ``2``, or only ``1`` and ``2``, are kept rather
  than dropped, and sites where nothing varies are kept too.
* ``moments_ld.compute_ld_statistics`` ran out of GPU memory for three
  and four populations: the automatic pair-chunk size assumed the work
  per pair grew in proportion to the number of populations, but the
  number of LD statistics grows faster (3, 15, 45, 105 for one to four
  populations), so it chose a chunk several times too large. The
  estimate now scales with the statistic count.
  ``compute_ld_statistics`` also gained ``chunk_size`` and
  ``available_memory_bytes`` arguments to override the estimate; they do
  not change results, only memory use.
* ``windowed_analysis`` ignored ``window_type='regions'`` and ``regions``
  kwarg for statistics that went to the scatter or fused paths; only the
  fallback used them. Now ``window_type`` defaults to ``'bp'`` and if
  ``'regions'`` is specified, it is sent to the proper engines. An error is
  raised if no engine can compute the requested set of statistics or if
  ``regions`` is not passed alongside ``window_type='regions'`` and vice-versa.
  An error is raised when ``window_type='snp'`` or ``'regions'`` for a
  ``StreamingHaplotypeMatrix``.
* ``fay_wus_h`` and the ``fay_wus_h`` key of ``diversity_stats`` gave
  NaN for a region with no variants, while windowed ``fay_wu_h`` and
  ``FrequencySpectrum.fay_wu_h`` gave 0.0. H is a difference of two thetas
  that are both 0 there, and it has no variance term, so every path now
  gives 0.0. ``normalized_fay_wus_h`` and ``zeng_dh`` stay NaN.
* ``zns`` and ``omega`` on a ``GenotypeMatrix`` with
  ``missing_data='exclude'`` raised ``ValueError: genotypes cannot be
  empty`` when no site had a call for every sample, since the filtering
  step built a zero-column ``GenotypeMatrix`` and the constructor rejects
  that. They now give the same no-pair result (NaN) as every other
  undefined case there, without constructing the empty matrix.
* A BED file with more than one contig was merged into one mask when no
  contig was named, so other contigs' intervals counted as accessible
  bases of the data. Eager ``from_zarr`` now takes the store's contig (as
  streaming does), streaming ``windowed_analysis`` the stream's, and
  ``compute_ld_statistics`` the VCF's. Elsewhere (``set_accessible_mask``,
  eager ``windowed_analysis``), a multi-contig BED with no ``chrom`` raises
  ``ValueError``. A BED that names the contig differently, such as ``chr1``
  for the data's ``1``, is matched by name without ``chr`` and case, with a
  ``ContigNameWarning``. ``from_vcf`` and ``from_zarr`` with
  ``accessible_bed`` on data with more than one contig and no ``region``
  raise ``ValueError``; ``from_vcf`` used to take the first variant's
  contig and drop the other contigs' intervals.
* ``window_type='regions'`` now computes every windowed statistic. The theta
  estimators and neutrality tests (``theta_l``, ``zeng_e``, ``zeng_dh``,
  ``normalized_fay_wu_h``) and the per-site statistics (``mean_nsl``,
  ``daf_hist``, ``mu_sfs``) used to raise ``NotImplementedError`` for region
  windows; they are now served over regions by the scatter and fused engines.
  Those engines share one window-membership rule that maps each window to its
  contiguous variant range -- handling overlapping, nested or unordered
  regions -- so a region's values match a bp grid of the same windows.
* ``windowed_analysis`` ignored an unknown keyword argument when the
  scatter or fused engine served the request, so a misspelled name, or
  ``population=`` in place of ``populations=``, changed nothing and gave
  no error. It now raises ``TypeError`` for any name that no engine reads,
  whichever engine serves the request, and suggests the closest valid
  name. ``population``, ``k``, ``n_blocks``, ``batch_size`` and
  ``aggregate`` are valid only with ``local_pca`` or
  ``local_pca_jackknife``.
* Windowed ``mu_var`` gave NaN for a window with accessible bases but no
  variants, while the scalar ``mu_var`` gives 0.0 there. The fused engine
  (and the chunked engine, which shares its code for this statistic) now
  give 0.0 too, keeping NaN only for a window with no accessible base.
* ``daf_histogram`` on a ``GenotypeMatrix`` put monomorphic sites and
  sites with no valid call in bin 0, and ignored ``missing_data``, so
  ``'exclude'`` gave the same result as ``'include'``. A site with no
  present derived allele, or no call at all, now contributes nothing.
* ``daf_histogram`` and the windowed ``daf_hist`` counted a site fixed for
  the derived allele at frequency 1.0, in the top bin, while ``sfs``
  already left such a site out. A fixed-derived site now contributes
  nothing in both, matching ``sfs``.
* ``pi``, ``theta_w``, ``theta_h``, ``theta_l`` and the Achaz eta-family
  (``diversity_stats``'s ``eta1``, ``eta1_star``, ``minus_eta1``,
  ``minus_eta1_star``) silently treated a site where the population has
  fewer than two calls as invariant: the site added 0 to the raw sum but
  stayed in the span denominator, biasing the default per-base rate
  toward 0 under missing data. The raw sum is now rescaled to the sites
  where the population actually had two calls before the span divides it;
  the rate is NaN when variants exist but none qualifies, and 0.0 (as
  before) with no variants at all (``span_normalize=False`` is
  unaffected; it already returned the plain raw sum).
* Windowed ``pi``, ``theta_w``, ``theta_h`` and ``theta_l`` (the scatter
  engine) had the same per-window bias and now get the same rescale.
* ``dxy`` and ``da`` had the same bias: a site where a population lacked
  enough calls (one call each for ``dxy``, two for ``da``) added 0 to the
  raw sum but stayed in the span denominator. Both are now rescaled the
  same way, and ``dxy(per_site=True)`` gives NaN rather than 0.0 at a site
  where either population has no call.
* Windowed ``dxy``/``da`` (the scatter engine) had the same per-window
  bias and now get the same rescale.
* Windowed ``pi``, ``theta_w`` and ``theta_h`` (the fused CUDA engine, used
  when a request mixes one of these with a Garud's H statistic, a diploSHIC
  statistic, or ``mean_nsl``) had the same per-window bias and now get the
  same rescale.
* Windowed ``dxy``/``da`` (both fused CUDA engines) had the same per-window
  bias and now get the same rescale, using dxy's and da's own gates as the
  scatter engine does.
* ``fay_wus_h`` is a difference of two estimators, not a rate, and has no
  ``span_normalize`` parameter at the scalar level. Windowed ``fay_wu_h``
  divided it by the window span anyway, giving a value that depended on
  window size and didn't match the scalar. It is now the raw difference
  on every engine, matching ``diversity.fay_wus_h`` on the window's
  variants exactly, like the other neutrality tests. The scatter engine's
  ``zeng_dh`` reused the same span-divided value for its own H term and
  had the same bias; it is now the same raw difference too.
* ``zeng_dh`` combined Tajima's D with the raw, unnormalized H
  (``pi - theta_h``), whose scale depends on the sample's
  segregating-site count and isn't comparable across windows or
  datasets. It now combines D with the normalized H*
  (``normalized_fay_wus_h``), matching the statistic Zeng et al. (2006)
  actually uses in their joint test.
* ``HaplotypeMatrix.n_segregating_sites`` summed allele codes and called a
  site polymorphic when the sum fell strictly between 0 and the valid
  count, so a multiallelic site could be missed at some allele
  frequencies (and so could ``n_invariant_sites``, which is computed from
  it). It now counts a site as segregating when at least two distinct
  alleles are present among the non-missing haplotypes.

New warnings
~~~~~~~~~~~~

* ``BiallelicOnlyWarning`` -- something that only works on two-allele
  sites threw multiallelic sites away, and tells you how many. Comes
  from ``patterson_d``, the ``GenotypeMatrix`` loaders and
  ``from_haplotype_matrix``, and the LD statistics that restrict to
  two-allele sites (``zns``, ``omega``, ``pairwise_r2``,
  ``pairwise_LD_v``, ``locate_unlinked``, ``windowed_r_squared``).
  The ``tally_gpu_haplotypes*`` methods keep multiallelic sites, count
  the highest allele against all others, and warn with the number of
  such sites.
* ``FewCompleteHaplotypesWarning`` -- 10% or more of the windows in a
  ``windowed_analysis`` or ``moving_garud_h`` call have few haplotypes
  with no missing call (fewer than 25, or fewer than half the sample
  when it is under 50). Garud's H and ``haplotype_count`` come out too
  low there. ``windowed_analysis`` now returns ``n_complete_haplotypes``
  with any Garud statistic or ``haplotype_count``, and
  ``min_complete_haplotypes`` sets thinner windows to NaN. See
  :doc:`missing_data`.
* ``UnpairedRowsWarning`` -- a statistic that pairs rows into
  individuals got a population list that does not keep each sample's
  two rows together. The row-validation entry above says when it
  fires.
* ``MultiallelicCapWarning`` -- a windowed calculation skipped sites
  with more than 8 alleles. You will not see this with DNA.

Removed
~~~~~~~
* The ``windowed_analysis.windowed_statistics`` scatter engine is
  removed. It was a lower-level entry point that the ``windowed_analysis``
  dispatcher never called. Its only capability not already in
  ``windowed_analysis`` -- expected heterozygosity -- is now the
  ``het_expected`` statistic, the per-window mean of per-site
  ``heterozygosity_expected`` over the sites with a defined value. Use
  ``windowed_analysis`` (or ``windowed_statistics_fused`` for the custom
  bin-edge interface).
* The ``align_bp`` argument on the streaming windowed path (unused; it
  defaulted to ``chunk_bp``). The streaming windowed path now requires
  ``window_size`` to be no larger than ``chunk_bp`` and to divide it
  evenly, raising a clear error otherwise.
* The deprecated LD wrappers ``ld_statistics.DD``, ``DD_two_pops``,
  ``Dz_two_pops``, and ``pi2_two_pops``. Use ``dd`` / ``dz`` / ``pi2``
  with the ``populations=`` argument.
* ``pg_gpu/utils.py`` and its ``read_vcf`` helper (unused, superseded by
  ``HaplotypeMatrix.from_vcf``).
* ``diversity.allele_frequency_spectrum`` and
  ``HaplotypeMatrix.allele_frequency_spectrum``. Use the ``sfs`` module
  instead.
* The ``scaler`` argument on the PCA functions.
* ``apply_biallelic_filter`` on both matrix classes. Use
  ``restrict_to_biallelic``, which keeps sites with at most two
  distinct present alleles whatever their coding -- including sites
  where nothing varies, which the old filter dropped -- or
  ``restrict_to_segregating`` to drop the non-varying ones.
* The ``ac_filter`` argument on
  ``compute_ld_statistics_gpu_single_pop`` / ``..._two_pops`` (eager
  and streaming) and ``moments_ld.compute_ld_statistics``. The LD
  pipeline always restricts to two-present-allele sites now.
* ``HaplotypeMatrix`` input to ``rogers_huff_r`` /
  ``rogers_huff_r_squared``. Rogers-Huff r is a diploid dosage
  correlation: convert with ``GenotypeMatrix.from_haplotype_matrix``
  first, or call ``pairwise_r2(estimator='rogers_huff')`` on the
  haplotype matrix, which converts internally.
* The ``genotype_matrix_or_haplotype_matrix`` keyword on ``grm`` and
  ``ibs``; the parameter is named ``genotype_matrix`` now

v0.1.0
------

First public release of pg_gpu.

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

* ``HaplotypeMatrix`` -- phased haplotype data (0/1 with -1 for missing).
  Loaders: ``from_vcf`` (with ``region=`` and ``samples=`` subsetting),
  ``from_zarr`` (auto-detects VCZ / scikit-allel / chromosome-grouped
  layouts), ``from_ts``, and direct NumPy construction. ``to_zarr`` writes
  VCZ by default. ``vcf_to_zarr`` provides multicore VCF-to-zarr conversion.
  Sample names from VCFs are preserved; ``load_pop_file('pops.txt')``
  assigns populations using stored sample names.

* ``GenotypeMatrix`` -- diploid genotypes (0/1/2). Same loaders and
  zarr round-trip as ``HaplotypeMatrix``. Many public functions
  auto-dispatch on input type (haplotype vs genotype).

Linkage Disequilibrium
~~~~~~~~~~~~~~~~~~~~~~

* Core statistics: ``r``, ``r_squared``, ``dd`` (D-squared), ``dz``,
  ``pi2`` (Ragsdale & Gravel 2019), ``zns`` (Kelly), ``omega``
  (Kim & Nielsen), ``mu_ld`` (RAiSD).
* LD pruning: ``locate_unlinked``; windowed :math:`r^2` decay:
  ``windowed_r_squared``.
* Two-population LD via ``compute_ld_statistics_gpu_two_pops`` with
  chunked GPU execution.

Diversity Statistics
~~~~~~~~~~~~~~~~~~~~

* Theta estimators: ``pi``, ``theta_w``, ``theta_h``, ``theta_l``,
  ``eta1``, ``eta1_star``, ``minus_eta1``, ``minus_eta1_star``.
* Neutrality tests: ``tajimas_d``, ``fay_wus_h``,
  ``normalized_fay_wus_h``, ``zeng_e``, ``zeng_dh``.
* Heterozygosity / inbreeding: ``heterozygosity_expected``,
  ``heterozygosity_observed``, ``inbreeding_coefficient``.
* Haplotype-level: ``haplotype_diversity``, ``haplotype_count``,
  ``daf_histogram``, ``diplotype_frequency_spectrum``.
* ``FrequencySpectrum`` class for custom weight functions, SFS
  projection, and the Achaz (2009) variance framework.
* All statistics accept ``missing_data='include' | 'exclude'`` and a
  unified ``span_normalize`` parameter that auto-detects the best
  denominator (accessible bases if mask set, else genomic span).

Divergence Statistics
~~~~~~~~~~~~~~~~~~~~~

* FST estimators: ``fst_hudson`` (ratio of averages),
  ``fst_weir_cockerham``, ``fst_nei``; ``pairwise_fst`` for multiple
  populations.
* Absolute / net divergence: ``dxy``, ``da``.
* Population Branch Statistic: ``pbs`` (normalized PBSn1).
* Distance-based two-population statistics (Schrider et al. 2018 and
  related): ``snn``, ``dxy_min``, ``gmin``, ``dd``, ``dd_rank``, ``zx``.
  Callers can pre-compute ``pairwise_distance_matrix`` once and pass it
  to multiple stats, or use the combined ``distance_based_stats``.

Selection Scans
~~~~~~~~~~~~~~~

* Haplotype-based: ``ihs`` (fused CUDA kernel, bitmask pair tracking,
  block-level EHH reductions), ``nsl``, ``xpehh``, ``xpnsl``,
  ``ehh_decay``.
* Garud's H: ``garud_h`` (H1, H12, H123, H2/H1) via GPU dot-product
  hashing of haplotypes; ``moving_garud_h`` uses cumulative prefix sums
  for O(1) per-window hash computation.
* Standardization: ``standardize``, ``standardize_by_allele_count``.
* Diploid variants: ``zns_diploid``, ``omega_diploid``,
  ``garud_h_diploid``, ``daf_histogram_diploid``.

Site Frequency Spectrum
~~~~~~~~~~~~~~~~~~~~~~~

* Unfolded and folded SFS: ``sfs``, ``sfs_folded``, ``sfs_scaled``,
  ``sfs_folded_scaled``.
* Two-population joint SFS: ``joint_sfs``, ``joint_sfs_folded``,
  ``joint_sfs_scaled``, ``joint_sfs_folded_scaled``.
* Folding utilities: ``fold_sfs``, ``fold_joint_sfs``.

Admixture / F-Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

* Per-variant: ``patterson_f2`` (F2 branch length), ``patterson_f3``
  (admixture test), ``patterson_d`` (ABBA-BABA).
* Windowed: ``moving_patterson_f3``, ``moving_patterson_d``.
* Block-jackknife with standard error: ``average_patterson_f3``,
  ``average_patterson_d``.

Resampling
~~~~~~~~~~

* Public ``pg_gpu.resampling`` module with ``block_jackknife`` and
  ``block_bootstrap`` for block-resampled standard errors / CIs on any scalar
  genome-wide statistic (genome-wide mean Tajima's D, ratio-of-sums
  estimators, etc.). Promotes the previously private ``_jackknife`` helper
  from ``admixture``. The weighted jackknife follows the Busing et al.
  (1999) delete-:math:`m_j` formulation for unequal block sizes.
* ``examples/sweep_tajimas_d_bootstrap.py`` -- 95% bootstrap CI on
  Tajima's D under a completed sweep, showing sweep-local vs distal mean
  difference CIs that exclude zero.

Dimensionality Reduction and Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``pca`` -- GPU-accelerated SVD PCA.
* ``randomized_pca`` -- truncated-SVD approximation for large
  datasets.
* ``pairwise_distance`` -- GPU-accelerated with memory-safe batching.
* ``pcoa`` -- classical MDS from a distance matrix.
* ``local_pca`` -- GPU port of Li & Ralph (2019) lostruct for detecting
  regions where population structure differs from the chromosome-wide
  pattern. Per-window top-k eigendecomposition via a single batched
  ``cp.linalg.eigh`` over a stacked
  ``(n_windows, n_samples, n_samples)`` tensor.
* ``pc_dist`` -- Frobenius distance between per-window low-rank
  covariance reps via the trace identity (no cov-matrix
  re-materialization). L1, L2, or no normalization.
* ``corners`` -- extreme-cluster selection in a 2D MDS embedding via
  Welzl's minimum enclosing circle.
* ``local_pca_jackknife`` -- delete-1 block jackknife standard error of local PCs,
  also GPU-batched with sign-aligned replicates.
* ``LocalPCAResult`` dataclass with ``.windows`` / ``.eigvals`` /
  ``.eigvecs`` / ``.sumsq`` plus ``.to_lostruct_matrix()`` for
  compatibility with the R ``lostruct::eigen_windows`` layout.

Relatedness and Kinship
~~~~~~~~~~~~~~~~~~~~~~~

* ``grm`` -- Genetic Relationship Matrix (Yang et al. 2011).
* ``ibs`` -- pairwise Identity-By-State proportions.

Fused Windowed Analysis
~~~~~~~~~~~~~~~~~~~~~~~

The ``windowed_analysis()`` convenience function routes through fused
CUDA kernels (one kernel launch for all windows) when using
non-overlapping windows with ``missing_data='include'``:

* Single-population: ``pi``, ``theta_w``, ``tajimas_d``,
  ``segregating_sites``, ``singletons``.
* Two-population: ``fst``, ``fst_hudson``, ``fst_wc``, ``dxy``, ``da``.
* Selection: ``garud_h1``, ``garud_h12``, ``garud_h123``, ``garud_h2h1``,
  ``mean_nsl``.
* Structure: ``local_pca`` (returns a ``LocalPCAResult``; scalar stats
  requested alongside are merged onto ``result.windows``).
* Structure: ``local_pca_jackknife`` computes delete-1 block jackknife
  standard error and populates ``LocalPCAResult.jackknife_se``. When both are
  requested together, per-window matrix preparation is shared.

Lower-level windowed entry points: ``windowed_statistics`` (scatter-add
aggregation) and ``windowed_statistics_fused`` (custom bin edges, one
thread block per window).

Distance Distribution Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``pairwise_diffs`` -- Hamming distance distributions (haploid or
  diploid).
* ``dist_var``, ``dist_skew``, ``dist_kurt`` -- moments of the
  pairwise-distance distribution (Schrider et al. 2018).
* ``dist_moments`` -- all three in one call.

diploSHIC / RAiSD Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``mu_var``, ``mu_sfs`` -- SNP density and SFS edge fraction (RAiSD).
* ``max_daf`` -- maximum derived allele frequency.

Visualization (``pg_gpu.plotting``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SFS: ``plot_sfs``, ``plot_joint_sfs``.
* LD: ``plot_pairwise_ld``, ``plot_ld_decay``.
* PCA / structure: ``plot_pca``, ``plot_pairwise_distance``.
* Windowed statistics: ``plot_windowed``, ``plot_windowed_panel``.
* Haplotypes: ``plot_haplotype_frequencies``, ``plot_variant_locator``.

Missing Data Handling
~~~~~~~~~~~~~~~~~~~~~

* Missing values are encoded as ``-1`` (haplotype) or ``-1`` sentinel
  (genotype).
* Every statistic accepts ``missing_data='include'`` (per-site valid
  data, default) or ``missing_data='exclude'`` (only fully genotyped
  sites). Simulation testing confirms ``include`` is unbiased under
  MCAR.
* LD projection estimator available via ``estimator='sigma_d2'`` on
  ``zns`` / ``omega``. The default is ``estimator='auto'``, which
  resolves to ``'sigma_d2'`` on ``HaplotypeMatrix`` inputs (the
  recommended path; uses the unbiased Ragsdale & Gravel 2019
  estimators) and falls back to ``'r2'`` for pre-computed r² arrays
  or ``GenotypeMatrix`` inputs. ``windowed_analysis`` follows the
  same default for windowed ``zns`` and ``omega``.

Moments Integration
~~~~~~~~~~~~~~~~~~~

``pg_gpu.moments_ld.compute_ld_statistics`` is a GPU drop-in for
``moments.LD.Parsing.compute_ld_statistics``. Returns the 15
two-population LD statistics and 3 heterozygosity statistics in the
exact layout moments expects for demographic inference. Requires the
``moments`` pixi environment: ``pixi install -e moments``.

Validation
~~~~~~~~~~

* Cross-validation script (``tests/validate_against_allel.py``)
  comparing 31 statistics against scikit-allel on real Ag1000G data
  (1M variants, 200 haplotypes). Divergence, diversity, and selection
  statistics match scikit-allel at machine precision; a timing table
  is included.
* Local PCA (lostruct) outputs validated against the R ``lostruct``
  package via frozen JSON references committed under ``tests/data/``.
  R is **not** a dependency of the pixi env or CI -- the comparison
  runs against the committed JSON. An optional ``requires_r`` test
  regenerates the references via rpy2 when R + lostruct are available
  locally.

Performance
~~~~~~~~~~~

All statistics run on CuPy with custom CUDA kernels for compute-bound
paths.

Scalar statistics at 1M variants, 200 haplotypes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Statistic
     - allel (s)
     - pg_gpu (s)
     - Speedup
   * - Weir-Cockerham FST
     - 9.85
     - 0.02
     - **468x**
   * - Patterson F2
     - 0.15
     - 0.009
     - **18x**
   * - nSL (255k variants)
     - 8.1
     - 0.56
     - **15x**
   * - Patterson F3
     - 0.14
     - 0.016
     - **9x**
   * - EHH decay (255k)
     - 0.06
     - 0.008
     - **8x**
   * - Hudson FST
     - 0.12
     - 0.017
     - **7x**
   * - iHS (255k variants)
     - 9.9
     - 1.5
     - **7x**
   * - Dxy
     - 0.07
     - 0.016
     - **4x**

Windowed statistics at 5.3M variants, 100kb windows, 200 haplotypes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Statistic
     - allel (s)
     - pg_gpu (s)
     - Speedup
   * - pi + theta_w + tajimas_d
     - 0.81
     - 0.013
     - **60x**
   * - All 5 single-pop stats
     - 0.81
     - 0.013
     - **60x**
   * - FST (Hudson)
     - 0.59
     - 0.18
     - **3x**
   * - All 12 stats together
     - n/a
     - 0.66
     - single call

Examples
~~~~~~~~

End-to-end demo scripts in ``examples/``:

* ``pg_gpu_tour.ipynb`` -- interactive tour using Anopheles gambiae X
  chromosome data.
* ``admixture_detection.py`` -- block-jackknife ABBA-BABA on simulated
  null and admixed msprime scenarios.
* ``accessibility_mask.py`` -- windowed :math:`\pi` with and without
  an accessibility mask over a low-:math:`\mu` "exon" region.
* ``ld_blocks.py`` -- LD-block partitioning via :math:`r^2` bridging
  scores.
* ``local_pca.py`` -- lostruct pipeline on a simulated partial
  selective sweep (``SweepGenicSelection`` with end frequency 0.5).

Infrastructure
~~~~~~~~~~~~~~

* pixi-based environment management; ``moments`` integration lives in
  a separate pixi feature.
* Shared ``_utils.py`` module for population extraction.
* Public API returns NumPy arrays (not CuPy) -- no need to call
  ``.get()`` on results.
