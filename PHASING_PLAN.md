# Plan: phasing awareness, Part 2: which functions need phase, and what to do about it

## Context

Part 1 gives `HaplotypeMatrix` a `phased` flag. It is implemented as uncommitted changes on
`add-phase`. VCF, zarr and the raw constructor default to `False`, and `from_ts`
defaults to `True`. Nothing reads the flag yet.

Part 2 decides what each statistic does with an unphased HaplotypeMatrix. For each
function this file keeps three things apart:
1. **Today:** what it accepts, whether its result depends on phase, and whether
   pg_gpu already has a phase-free route.
2. **Feasible:** whether a version on unphased diploid genotypes (diplotypes / 0-1-2
   dosage) could exist in principle, and whether it would estimate the same thing.
3. **Proposed:** what Part 2 does, and what becomes a follow-up issue.

Decision carried over: Part 2 is **warning only**. It adds no new GenotypeMatrix
dispatch and changes no numbers. Diploid implementations are follow-ups.

## Column key

| Column | Values |
|---|---|
| **Accepts** | `HM`, `GM`, `sHM` (streaming HM), `arrays`. How it is enforced: *dispatch* (`isinstance` branch), *TypeError* (explicit rejection), *converts* (turns the other type into its own), or *attr* (no check; the other class fails on a missing attribute such as `.haplotypes`). |
| **Phase (HM)** | Test: does the output change when alleles are swapped between rows `2i` and `2i+1` at heterozygous sites, independently per site? **No-site**: per-site allele counts only. **No-pairs**: rebuilds individuals from rows `2i`/`2i+1`, which needs the pairing but not the phase. **Yes**: changes under re-phasing. **Partial**: depends on a mode or argument. A GM never carries phase, so every GM path is phase-free. |
| **Route today** | An existing phase-free way to get the same or a comparable answer. |
| **Diplotype version feasible?** | **Exists**: pg_gpu already has it. **Same**: not in pg_gpu yet, but the same estimand on dosage, mostly plumbing. **Analogue**: well-defined but a different statistic (different null, needs its own docs). **Weak**: computable, but it dilutes the signal the statistic exists to detect. **None known**: no established formulation. |
| **Proposed** | **Warn**: emit `UnphasedDataWarning`. **Warn→X**: the warning names route X. **FU-n**: follow-up issue, priority n (§4). |

**Scope assumption.** Every "No" label assumes populations and sample sets are unions
of whole individuals. A population that holds row `2i` but not `2i+1` makes even
per-site statistics (`pi`, `fst`, `sfs`) depend on which allele the VCF listed first.
That is a pairing problem rather than a phase-dependent statistic, and Part 2 doesn't
warn about it. `genetic_relatedness` is the one place it's handled, because its
default (`sample_sets=None`) always splits individuals.

**Rogers-Huff caveat.** The dosage-based Rogers-Huff route (`_dosage_from_matrix`,
[ld_statistics.py:431-437](pg_gpu/ld_statistics.py#L431-L437)) raises `ValueError` on
missing calls. So `pairwise_r2('rogers_huff')`, `windowed_r_squared('rogers_huff')`
(which also rejects `pop`) and `rogers_huff_r` work only on complete data. `zns` and
`omega` avoid this, because they go through `_r2_matrix_diploid`. Warning text that
points at Rogers-Huff says "(requires no missing calls)" wherever that applies.

---

## 1. Phase-dependent functions (the Part 2 scope)

### selection
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `garud_h` | HM, GM (dispatch) | Yes | GM → diplotype freqs (`_garud_h_diploid`) | Exists (G12/G123, Harris et al. 2018) | Warn→GM, HM branch only |
| `moving_garud_h` | HM (attr) | Yes | none | Same: `_garud_h_diploid` per window | Warn; FU-1 |
| `ihs`, `xpehh`, `ehh_decay` | HM (attr) | Yes (EHH) | none | Analogue: unphased EHH on genotype homozygosity (Klassmann & Gautier 2022; rehh `phased=FALSE`). Needs its own standardization. | Warn; FU-3 |
| `nsl`, `xpnsl` | HM (attr) | Yes (shared-suffix length) | none | Analogue: unphased nSL/XP-nSL in selscan 2.0 (Szpiech 2024, *Bioinformatics*; **verify the citation**) | Warn; FU-3 |

### diversity
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `haplotype_count`, `haplotype_diversity` | HM (attr) | Yes | none (`diplotype_frequency_spectrum` counts diplotypes, but has a different output) | Analogue: distinct-diplotype count and diversity over individuals. The EM frequencies already exist for `garud_h`. | Warn; FU-2 |
| `diversity_stats` | HM (attr) | Partial: Yes only when `'haplotype_diversity'` is requested ([diversity.py:1148](pg_gpu/diversity.py#L1148)) | omit that stat | as `haplotype_diversity` | Warn, only when that stat is requested |

### divergence
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `pairwise_distance_matrix` | HM (attr) | Yes (gamete distances) | `pairwise_diffs_diploid` (individual distances, condensed form, no pop split) | Same at the individual level | Warn→`pairwise_diffs_diploid` |
| `snn`, `dxy_min`, `gmin`, `dd`, `dd_rank`, `distance_based_stats` | HM (attr) | Yes (min / rank / nearest neighbour of gamete distances) | none | Weak: these target single introgressed haplotypes, and an individual distance averages two gametes | Warn only when the function computes the distances itself; silent when `distance_matrices` is passed in |
| `zx` | HM (attr) | Yes: a ratio of haplotype ZnS terms (`_zns_biallelic`, [divergence.py:1394-1401](pg_gpu/divergence.py#L1394-L1401)), not a distance statistic | none (scalar `zns(estimator='rogers_huff')` exists, but `zx` exposes no estimator) | Same: thread an `estimator` through to `_zns_biallelic` | Warn; FU-1 |

### distance_stats
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `pairwise_diffs_haploid` | HM (attr) | Yes | `pairwise_diffs_diploid` | Exists | Warn→`pairwise_diffs_diploid` |
| `pairwise_diffs`, `dist_moments`, `dist_var`, `dist_skew`, `dist_kurt` | HM, GM (dispatch) | Yes on HM. Var/skew/kurt always. The mean too when there's missing data, since each pair is normalized by its own jointly called sites. | GM input | Exists | Warn→GM, HM branch only |

### ld_statistics / moments_ld
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `zns`, `omega` | HM, GM, r² array (dispatch) | Partial: Yes for `'r2'` / `'sigma_d2'` (`'auto'` resolves to `sigma_d2` on an HM); No for `'rogers_huff'` | `estimator='rogers_huff'`, or a GM | Exists | Warn→`rogers_huff`/GM, HM haplotype-estimator branch only |
| `mu_ld` | HM (attr; sHM rejected) | Yes | none | Analogue: the same exclusivity count on diplotype labels. Not a published form (RAiSD defines it on haplotypes). | Warn |
| `moments_ld.compute_ld_statistics` | VCF path, HM, GM | Partial: No with `use_genotypes=True` (default); Yes with `False` | `use_genotypes=True` | Exists (Ragsdale & Gravel 2020, *MBE*: unphased estimators) | Warn→`use_genotypes=True`, only when `False`, including the VCF path, whose HM loads as unphased |

### decomposition
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `pca`, `randomized_pca` | HM (GM → TypeError) | Yes, mildly: per-gamete coordinates, and gamete-gamete inner products shift under re-phasing | `pca_dosage`, `randomized_pca_dosage` | Exists | Warn→`pca_dosage` |
| `pairwise_distance` | HM (GM → TypeError) | Yes (gamete distances) | `pairwise_diffs_diploid`, `ibs` (individual level) | Exists at the individual level | Warn→`pairwise_diffs_diploid` |
| `local_pca`, `local_pca_jackknife`, `lostruct` | HM only (attr). An sHM fails because `WindowIterator` calls `get_subset`. | Yes, mildly (same as `pca`) | none | Same: lostruct was formulated on 0/1/2 genotype matrices (Li & Ralph 2019) | Warn; FU-2 |

### relatedness
| Function | Accepts | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|---|
| `genetic_relatedness` | HM, sHM (other → TypeError) | Partial: No when every sample set is a union of whole individuals; Yes otherwise, including the default `sample_sets=None` | whole-individual `sample_sets`; `grm` on a GM for an individual GRM (different normalization) | Exists (pass individual pairs) | Warn→whole-individual sets, only when `sample_sets is None` or some set splits an individual (3b) |

### HaplotypeMatrix / StreamingHaplotypeMatrix methods
| Method | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|
| `pairwise_r2`, `windowed_r_squared` | Partial: Yes for `'r2'` (default); No for `'rogers_huff'` | `estimator='rogers_huff'` (no missing calls; `windowed_r_squared` also rejects `pop`) | Exists | Warn→`rogers_huff`, `'r2'` branch only |
| `pairwise_LD_v` | Yes (haplotype D) | none for D; `rogers_huff_r` on a GM gives r, not D | Analogue | Warn |
| `locate_unlinked` | Yes (haplotype r²) | none | Same: Rogers-Huff r², which is what scikit-allel's `locate_unlinked` uses | Warn; FU-1 |
| `tally_gpu_haplotypes*`, `compute_ld_statistics_gpu_single_pop/two_pops` (eager and streaming) | Yes | `moments_ld.compute_ld_statistics(use_genotypes=True)` | Exists | Warn→`moments_ld` genotype path |

### windowed_analysis stats
The public scan entry points are `windowed_analysis`, `windowed_statistics_fused`,
`windowed_statistics_fused_chunked` and `WindowedAnalyzer.compute*`. They accept an
HM (plus an sHM for `windowed_analysis`), never a GM. `windowed_statistics`
([windowed_analysis.py:3002](pg_gpu/windowed_analysis.py#L3002)) computes no stat
below, so it needs no warning. A scan is phase-dependent if it requests any of:

| Stat | Phase (HM) | Route today | Diplotype version feasible? | Proposed |
|---|---|---|---|---|
| `garud_h1/h12/h123/h2h1` | Yes | scalar `garud_h(GM)` only | Same (diplotype G-stats), but needs GM input to windowed_analysis | Warn; FU-4 |
| `haplotype_count` | Yes | none | Analogue (see diversity) | Warn; FU-4 |
| `mean_nsl` | Yes | none | Analogue (see `nsl`) | Warn; FU-3 |
| `mu_ld` | Yes | none | Analogue (see `mu_ld`) | Warn |
| `zns`, `omega` | Yes. Per-window code always uses haplotype `sigma_d2`/r², with no estimator option. | scalar `zns/omega(estimator='rogers_huff')` | Same: add an estimator option to the scan | Warn; FU-1 |
| `ld_decay`, `mean_r2` | Yes (`_compute_mean_r2` uses haplotype `'r2'`) | `pairwise_r2(estimator='rogers_huff')` by hand | Same: pass the estimator through | Warn; FU-1 |
| `dist_var/skew/kurt` | Yes | scalar `dist_moments(GM)` | Same, but needs GM input to windowed_analysis | Warn; FU-4 |
| `local_pca`, `local_pca_jackknife` | Yes, mildly | none | Same (see decomposition) | Warn; FU-2 |

## 2. Phase-free functions (no Part 2 change)

Listed so the map is complete, under the scope assumption above. A diplotype version
would compute the same allele counts, so feasibility doesn't apply.

| Group | Functions | Accepts | Phase (HM) |
|---|---|---|---|
| diversity | `pi`, `theta_w`, `tajimas_d`, `segregating_sites`, `singleton_count`, `neutrality_tests`, `fay_wus_h`, `normalized_fay_wus_h`, `theta_h`, `theta_l`, `zeng_e`, `zeng_dh`, `max_daf`, `heterozygosity_expected`, `mu_var`, `mu_sfs`; `diversity_stats` without `'haplotype_diversity'` | HM (attr) | No-site |
| | `daf_histogram` | HM, GM (dispatch) | No-site |
| | `heterozygosity_observed`, `inbreeding_coefficient` | HM (attr) | No-pairs |
| | `diplotype_frequency_spectrum` | GM | n/a (GM) |
| divergence | `fst`, `fst_hudson`, `fst_tskit`, `fst_nei`, `dxy`, `da`, `pi_within_population`, `divergence_stats`, `pairwise_fst`, `pbs` | HM (attr) | No-site |
| | `fst_weir_cockerham` | HM (attr) | No-pairs |
| admixture | `patterson_f2/f3/f4/d`, `moving_patterson_f3/d`, `average_patterson_f3/d` | HM (attr) | No-site |
| sfs | `sfs`, `sfs_folded`, `joint_sfs`, `project_joint_sfs`, `joint_sfs_folded`, `*_scaled` | HM, sHM (dispatch) | No-site |
| ld (GM) | `rogers_huff_r`, `rogers_huff_r_squared` | GM (HM → TypeError) | n/a (GM) |
| | `ld_statistics_genotype.sigma_d2_geno` | GM; HM converts ([ld_statistics_genotype.py:1556-1563](pg_gpu/ld_statistics_genotype.py#L1556-L1563)) | No (works on dosage) |
| ld (counts) | `dd`, `dz`, `pi2`, `r`, `r_squared`, `d_prime`, `ld_statistics.compute_ld_statistics` | count arrays | inherited from whoever built the counts (covered by `tally_*` above) |
| decomposition | `pca_dosage`, `randomized_pca_dosage` | GM (HM → TypeError) | n/a (GM) |
| | `pcoa`, `pc_dist`, `corners` | arrays / results | n/a |
| relatedness | `grm`, `ibs` | GM, sGM (HM → TypeError) | n/a (GM) |
| distance_stats | `pairwise_diffs_diploid` | GM | n/a (GM) |
| selection | `standardize`, `standardize_by_allele_count` | arrays | n/a |
| HM methods | `diversity`, `watersons_theta`, `Tajimas_D` | HM | No-site |
| windowed stats | `pi`, `theta_w`, `tajimas_d`, `segregating_sites`, `singletons`, `n_variants`, `theta_h`, `fay_wu_h`, `max_daf`, `dxy`, `fst*`, `da`, `daf_hist`, `mu_var`, `mu_sfs`, `snp_dist_*`; `fst_wc` (No-pairs) | HM, sHM | No-site / No-pairs |

---

## 3. Part 2 implementation: the warning

### 3a. Class and helper ([pg_gpu/_warnings.py](pg_gpu/_warnings.py))
- `class UnphasedDataWarning(UserWarning)`. The docstring follows `BiallelicOnlyWarning`:
  what the warning means, the fix (`hm.set_phased()` if the data really is phased, or
  the route), and the silence snippet.
- `_warn_unphased(matrix, *, context, alternative=None, stacklevel=3)`. It does
  nothing unless `getattr(matrix, 'phased', True) is False`, so call sites can call it
  unconditionally; a GM or an array has no `phased`. The message is
  `"{context} depends on haplotype phase, but this HaplotypeMatrix is not marked
  phased; "` + (`alternative` or `"results assume the rows are true haplotypes"`) +
  the `set_phased()` hint + the silence snippet. The phrase "depends on haplotype
  phase" is what the pyproject filter matches on, and a test pins it (3e).
- No warning-suppression context manager. Nothing in `pg_gpu/` uses
  `catch_warnings`, and it changes process-global filter state.
- Export `UnphasedDataWarning` from [pg_gpu/__init__.py](pg_gpu/__init__.py).

### 3b. Call sites: one rule
**Public functions warn; pg_gpu's own code calls private bodies.**
- Each §1 function whose Proposed column says "Warn" calls `_warn_unphased` once, at
  the top of its phase-dependent branch, after any GM dispatch has returned.
  `context` is the function's public name. The Route-today text becomes `alternative`.
- Wherever pg_gpu itself calls a warned public function, the public function becomes a
  thin wrapper (warn, then call `_<name>`), and the internal caller calls `_<name>`.
  The warning then fires once, at the user's call, with `stacklevel=3` pointing at the
  user's code. This also removes the need for the `phased=True` override on per-window
  matrices that an earlier draft proposed: no matrix's flag is ever set to anything
  but the truth.

Internal call sites that need a private body (from tracing the call graph):

| Public function | Called internally by | Private body |
|---|---|---|
| `pairwise_diffs_haploid` | `pairwise_diffs` → `dist_moments` → `dist_var/skew/kurt`; windowed `dist_*` | `_pairwise_diffs_haploid`; also `_pairwise_diffs`, `_dist_moments` |
| `pairwise_distance_matrix` | `_resolve_distance_matrices` (from `snn`, ...), `distance_based_stats` | `_pairwise_distance_matrix` |
| `nsl` | windowed `mean_nsl` ([windowed_analysis.py:2401](pg_gpu/windowed_analysis.py#L2401)) | `_nsl` |
| `haplotype_diversity` | `diversity_stats` | `_haplotype_diversity` |
| `local_pca` | `lostruct`; windowed `local_pca` ([windowed_analysis.py:1345](pg_gpu/windowed_analysis.py#L1345)) | `_local_pca` (`_local_pca_with_jackknife` already exists) |
| `HaplotypeMatrix.pairwise_r2` (`'r2'` path) | `_resolve_r2_matrix` (from `zns`/`omega` with `estimator='r2'`, [ld_statistics.py:1048](pg_gpu/ld_statistics.py#L1048)); `_compute_mean_r2` | `_haplotype_r2(matrix)` |
| `omega`, `mu_ld` | per-window loop ([windowed_analysis.py:2541](pg_gpu/windowed_analysis.py#L2541)) | `_omega`, `_mu_ld` |
| `windowed_statistics_fused`, `windowed_statistics_fused_chunked` | `_windowed_analysis` ([windowed_analysis.py:1488-1501](pg_gpu/windowed_analysis.py#L1488-L1501)); `_chunked` → `windowed_statistics_fused` ([:2918](pg_gpu/windowed_analysis.py#L2918)) | `_windowed_statistics_fused`, `_windowed_statistics_fused_chunked` |
| `WindowedAnalyzer.compute*` | `_windowed_analysis` fallback | `_compute*` |

Calls the review checked that can't produce a duplicate warning: `xpehh`/`xpnsl`,
`moving_garud_h`, the `tally_*` and `compute_ld_statistics_gpu_*` methods (eager and
streaming), `local_pca_jackknife`, streaming `genetic_relatedness`, per-window `zns`,
and `zx` (it calls the private `_zns_biallelic`).

**Distance stats with precomputed matrices.** `snn`/`dxy_min`/`gmin`/`dd`/`dd_rank`
warn only on the path where `_resolve_distance_matrices` computes the distances. With
`distance_matrices` passed in they compute nothing and stay silent.
`distance_based_stats` warns once and calls `_pairwise_distance_matrix`.

**`genetic_relatedness` ([relatedness.py:13](pg_gpu/relatedness.py#L13)).**
- **Why only some calls warn.** A set that holds both gametes of every individual it
  touches has allele counts equal to the sum of those individuals' dosages, which
  re-phasing can't change. A set that holds row `2i` without `2i+1` follows one gamete
  across sites, and unphased data doesn't define that. The default `sample_sets=None`
  (one set per haplotype) always does this.
- **Check.** Add `_splits_individual(rows)` to `_warnings.py`. It tests set membership,
  not order. Convert with the existing `_rows_as_numpy`, since sets may be lists,
  tuples, numpy or cupy arrays, then
  `s = set(arr.tolist()); any((r ^ 1) not in s for r in s)`. Don't reuse
  `paired_rows_problem` ([_warnings.py:180](pg_gpu/_warnings.py#L180)): it requires
  *consecutive* pairs and would wrongly flag `[0, 2, 1, 3]`.
- **Placement.** Use its own `isinstance(haplotype_matrix, (StreamingHaplotypeMatrix,
  HaplotypeMatrix))` block, after the validation loop at
  [relatedness.py:75-80](pg_gpu/relatedness.py#L75-L80) and before the streaming
  dispatch at line 85. It can't go inside that loop's gate, which requires
  `sample_sets is not None`, so the default case would never warn. Warn if
  `sample_sets is None` or any set splits an individual. One call covers eager and
  streaming.

### 3c. Windowed scans: one warning per scan
- Add `_PHASE_DEPENDENT_STATS` (the §1 windowed table) next to `_FUSED_GARUD_STATS` in
  [windowed_analysis.py](pg_gpu/windowed_analysis.py).
- Each public entry point (`windowed_analysis`, `windowed_statistics_fused`,
  `windowed_statistics_fused_chunked`, `WindowedAnalyzer.compute*`) checks
  `set(statistics) & _PHASE_DEPENDENT_STATS` once, before any dispatch, and warns
  naming those stats.
- **Streaming.** `windowed_analysis` warns once using the stream's flag before
  `_stream_windowed_analysis` loops over chunks. The per-chunk `_windowed_analysis` is
  private and calls private bodies, so it never warns. The same holds for the other
  internal routes in the 3b table.

### 3d. Docs
- New [docs/source/phasing.rst](docs/source/phasing.rst), linked from `index.rst`. It
  covers the flag and loader defaults, the scope assumption (whole-individual
  populations), a trimmed version of §1 (function → needs phase? → route, with the
  Rogers-Huff missing-data caveat), and how to fix or silence the warning.
- [api.rst](docs/source/api.rst) Warnings section: `autoclass UnphasedDataWarning`.
- Add a one-line "Depends on haplotype phase; see :doc:`phasing`" to each warned
  function's docstring.
- Mark phase-dependent stats with a short marker ("†: needs phased haplotypes, see
  :doc:`phasing`") in the windowed-stat lists users pick names from: the "Fused
  Windowed Statistics" table at [features.rst:574](docs/source/features.rst#L574) and
  the supported-stats list at [quickstart.rst:510](docs/source/quickstart.rst#L510).
- `phasing.rst` is the only full list. Other pages get markers and links, never a copy
  of the table.
- [changelog.rst](docs/source/changelog.rst): a behavior-change entry.
- [quickstart.rst](docs/source/quickstart.rst) "Phased to Unphased": link to
  `phasing.rst`.

### 3e. Tests
- [pyproject.toml](pyproject.toml) `filterwarnings`: add
  `"ignore:.* depends on haplotype phase:UserWarning"`. It matches by message, like the
  biallelic entry, so pg_gpu isn't imported early. Fixtures load VCFs as unphased.
- New `tests/test_unphased_warning.py`, modeled on
  [test_biallelic_only_warning.py](tests/test_biallelic_only_warning.py):
  - **Helper:** it subclasses `UserWarning`. A test pins the filter phrase
    ("depends on haplotype phase") so the pyproject entry can't silently stop
    matching. It does nothing for a GM, an array, and a phased HM, and it can be
    silenced by category.
  - **Positive:** parametrized over every "Warn" row in §1. On an unphased HM each
    emits exactly one warning (`catch_warnings(record=True)` +
    `simplefilter("always")`), and the warning's `filename` is the test file, which
    checks `stacklevel`.
  - **Negative:** the same calls are silent after `set_phased()`. A sample of §2
    functions is silent (`pi`, `fst_hudson`, `sfs`, `heterozygosity_observed`,
    `fst_weir_cockerham`, `diversity_stats(['pi'])`). Phase-free branches are silent:
    `zns(estimator='rogers_huff')`, `pairwise_r2(estimator='rogers_huff')`,
    `moments_ld(use_genotypes=True)`, `garud_h(GM)`, `dist_moments(GM)`.
  - **`genetic_relatedness` sets:** `[[0, 1], [2, 3]]` and `[[0, 2, 1, 3]]` are silent.
    `[[0, 2]]` and `sample_sets=None` warn. A cupy-array set works. Run on both eager
    and streaming.
  - **Once per call (nested routes):**
    - `dist_var`, `dist_moments`, `pairwise_diffs`;
    - `omega(hm, estimator='r2')`;
    - `distance_based_stats`;
    - `lostruct`, both branches;
    - `local_pca_jackknife`;
    - `diversity_stats(['haplotype_diversity'])`.

    `snn(..., distance_matrices=dm)` emits none.
  - **Once per scan.** Each case below must be a valid request; `StatisticsComputer`
    rejects some combinations, e.g. `omega` together with `mean_r2`.
    - Fused `['garud_h12', 'pi']` → 1. Fused `['omega', 'mu_ld', 'dist_var']` → 1.
    - `['mean_nsl']` → 1.
    - `['mean_r2']` with `custom_stat_kwargs={'mean_r2': {'max_dist': ...}}` (the
      fallback route) → 1.
    - `['local_pca']` → 1.
    - `windowed_statistics_fused_chunked` with a Garud stat → 1.
    - Streaming `windowed_analysis` over several chunks → 1.
    - `['pi', 'fst']` → 0.
  - **`from_ts`** matrices never warn.
- **Pinned set:** assert `_PHASE_DEPENDENT_STATS == {...explicit list...}`, with a
  comment: "update phasing.rst and the † markers in features.rst/quickstart.rst when
  this changes".
- **Re-phasing test (checks the classification against behavior):**
  - **Fixture:** an msprime tree sequence loaded with `from_ts` (so haplotypes are
    shared), with short windows, populations made of whole individuals, and a fixed
    seed. It must hold both shared haplotypes and enough heterozygous sites.
    Otherwise `haplotype_count`/`garud_*` don't change when every haplotype is already
    distinct.
  - **Operation:** randomly swap alleles between rows `2i` and `2i+1` at heterozygous
    sites.
  - **Assertion:** every windowed stat changes **iff** it is in
    `_PHASE_DEPENDENT_STATS`. "Unchanged" means `allclose` with a tolerance (scatter-add
    order), `nan == nan` allowed. `local_pca` results are compared through their
    eigenvalues.
  - Optionally, extend this to the scalar §1/§2 functions.

## 4. Follow-up issues (not Part 2)

Ranked by value over cost. Each is a separate issue, and the Part 2 warning names the
route once it exists.

| Priority | Work | Why this rank |
|---|---|---|
| **FU-1** | Reuse existing phase-free code: `locate_unlinked(estimator=)`, an estimator option for windowed `zns`/`omega`/`mean_r2`/`ld_decay` and for `zx`, and a `moving_garud_h` GM dispatch | Same estimand, reuses existing kernels. The Rogers-Huff dosage path rejects missing calls, so either route through `_r2_matrix_diploid` (as `zns`/`omega` do) or document the restriction. |
| **FU-2** | GM input for `local_pca`/`lostruct`/`local_pca_jackknife` and the windowed `local_pca` stats (dosage rows); GM dispatch for `haplotype_count`/`haplotype_diversity` (diplotype counts, following the `garud_h` precedent) | High user value, since lostruct users usually have unphased data. Moderate plumbing. The `haplotype_*` change returns a different estimand under the same name, so the docs must say so. |
| **FU-3** | Unphased EHH/nSL family: `ihs`, `xpehh`, `ehh_decay` (Klassmann & Gautier 2022), and `nsl`, `xpnsl`, windowed `mean_nsl` (selscan 2.0) | Real demand, but a new kernel plus a new standardization. Validate against rehh/selscan. |
| **FU-4** | GM input to `windowed_analysis` for the diplotype-capable stats (`garud_*`, `haplotype_count`, `dist_*`) | Large: touches the fused and streaming engines |
| not planned | `mu_ld`, `pairwise_LD_v`, and the gamete-distance introgression stats (`snn`, `dxy_min`, `gmin`, `dd`, `dd_rank`) | No established diploid form, or one that defeats the statistic's purpose. Warn only. |

## Verification
- `pixi run pytest tests/test_unphased_warning.py -x`, then the full suite, on a GPU
  node via Slurm per the `ucd-hpc` skill. The full suite confirms that the
  private-body refactors changed no numbers.
- `tests/test_examples_run.py` and `make -C docs html` (no new Sphinx warnings).
- By hand: run `examples/` with `-W always::pg_gpu.UnphasedDataWarning`. Warnings should
  appear only on §1 functions, once per call.

---

## Review notes

### N1. Phase-free counterparts are hard to find (open)
`pairwise_diffs_diploid`, `pca_dosage` and `rogers_huff_r` are listed in §2 and appear
in §1 only as routes. **Proposal (not yet applied):** under each §1 module table, add
a "phase-free counterpart" row so each pair sits side by side. §2 stays the complete
list.

### N2. `genetic_relatedness` "splits an individual" check (resolved)
The first draft reused `paired_rows_problem`, which wrongly flags `[0, 2, 1, 3]`. It
also placed the check inside a gate that skips `sample_sets=None`. Both are fixed in
3b.

### N3. First-principles review (resolved into the plan)
An independent review traced the call graphs and found five blockers, now fixed above:
- the windowed entry points call each other, and streaming calls once per chunk;
- `mean_nsl` calls the public `nsl`;
- the `dist_*` chain would warn up to four times;
- `zx` is a ZnS ratio, not a distance stat, and would never have warned;
- the `genetic_relatedness` default case was skipped.

It also found these, fixed above too: `omega('r2')` double-warning, the missing
windowed `local_pca` stats, `WindowedAnalyzer`, `diversity_stats` being Partial, the
whole-individual scope assumption, the Rogers-Huff missing-data limit, unphased nSL in
selscan 2.0, invalid scan combinations in the tests, and fragility in the re-phasing
test. The selscan 2.0 citation was given from memory and still needs checking.
