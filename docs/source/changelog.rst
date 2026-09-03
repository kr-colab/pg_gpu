Changelog
=========

Unreleased
----------

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
  ``tskit``'s function of the same name.
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
  ``from_vcf`` reads through tabix and was already inclusive. The
  ``materialize(region=(left, right))`` tuple and windowed analysis
  are unchanged: those intervals stay half-open, and
  ``pg_gpu.zarr_io.parse_region`` turns a region string into that
  half-open form. Every loader also takes the other samtools forms now:
  ``'X'`` selects a whole chromosome from a multi-contig store,
  ``'X:1000'`` runs from a position to the chromosome end, and
  thousands separators (``'X:1,000-2,000'``) are allowed.
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
  windowed ``mu_sfs`` divides by the number of variable sites and
  returns 0.0 instead of NaN for an empty window.
* In ``windowed_statistics``, ``singletons`` now counts only alternate
  alleles seen once (it used to also count reference alleles seen
  once), and ``segregating_sites`` counts mutations.
* ``dxy(span_normalize=False)`` and ``da(span_normalize=False)`` return
  the raw sum, matching every other statistic. They used to divide by
  the number of sites.
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

New warnings
~~~~~~~~~~~~

* ``BiallelicOnlyWarning`` -- something that only works on two-allele
  sites threw multiallelic sites away, and tells you how many. Comes
  from ``patterson_d``, the ``GenotypeMatrix`` loaders and
  ``from_haplotype_matrix``, and the LD statistics that restrict to
  two-allele sites (``zns``, ``omega``, ``pairwise_r2``,
  ``pairwise_LD_v``, ``locate_unlinked``, ``windowed_r_squared``).
* ``UnpairedRowsWarning`` -- a statistic that pairs rows into
  individuals got a population list that does not keep each sample's
  two rows together. The row-validation entry above says when it
  fires.
* ``MultiallelicCapWarning`` -- a windowed calculation skipped sites
  with more than 8 alleles. You will not see this with DNA.

Removed
~~~~~~~

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
  ``ibs``; the parameter is named ``genotype_matrix`` now.

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
