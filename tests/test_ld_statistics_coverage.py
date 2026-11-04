"""Residual coverage for ld_statistics.py internal r2 / mu_ld / dispatcher
branches. The >=3-population LD moment dispatch is covered separately, since it
needs the moments environment as an oracle.

Each gap is pinned to an independent oracle: the public ``zns`` (validated
against moments/allel elsewhere) for the precomputed-ZnS helper, a numpy
correlation for the diploid r2 matrix, the single-stat functions for the
multi-stat dispatcher, and hand values for mu_ld and the population-data
unpacker.
"""
import cupy as cp
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.ld_statistics import (
    compute_ld_statistics, dd, dz, mu_ld, omega, pi2, r, r_squared, zns,
    _get_pop_data, _r2_matrix_diploid, _resolve_r2_matrix,
    _zns_from_precomputed,
)


def _agree(a, b):
    np.testing.assert_allclose(
        cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a),
        cp.asnumpy(b) if isinstance(b, cp.ndarray) else np.asarray(b),
        rtol=1e-9, atol=1e-12, equal_nan=True)


# A biallelic, fully-segregating matrix (every column carries both alleles).
_X = np.array([
    [0, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
], dtype=np.int8)
_POS = np.array([100, 200, 300, 400, 500], dtype=np.int64)


def _hm():
    m = HaplotypeMatrix(_X.copy(), _POS.copy(), 0, 1000)
    m.transfer_to_gpu()
    return m


def _precomputed(hm):
    """Build (hap_clean, valid_mask) the way windowed_analysis feeds
    _zns_from_precomputed: 0/1 indicator with missing zeroed, float64 mask."""
    ind = hm._biallelic_indicator()
    hap_clean = cp.where(ind >= 0, ind, 0).astype(cp.float64)
    valid_mask = (ind >= 0).astype(cp.float64)
    return hap_clean, valid_mask


# ── _zns_from_precomputed (live windowed helper) vs the public zns ─────
@pytest.mark.parametrize("use_projection,estimator", [
    (False, "r2"), (True, "sigma_d2"),
], ids=["naive", "projection"])
def test_zns_from_precomputed_matches_public_zns(use_projection, estimator):
    hm = _hm()
    hc, vm = _precomputed(hm)
    z = _zns_from_precomputed(hc, vm, 0, hm.num_variants,
                              use_projection=use_projection)
    _agree(z, zns(hm, estimator=estimator))


def test_zns_from_precomputed_too_few_sites_returns_nan():
    hm = _hm()
    hc, vm = _precomputed(hm)
    # A single-column range has < 2 segregating sites -> NaN.
    assert np.isnan(_zns_from_precomputed(hc, vm, 0, 1))


def test_zns_naive_matches_direct_pairwise_correlation():
    # Independent oracle for the naive tile math: on complete data ZnS is the
    # mean over distinct site pairs of the squared column correlation.
    hm = _hm()
    Xf = _X.astype(np.float64)
    m = Xf.shape[1]
    r2s = [np.corrcoef(Xf[:, i], Xf[:, j])[0, 1] ** 2
           for i in range(m) for j in range(i + 1, m)]
    _agree(zns(hm, estimator="r2"), float(np.mean(r2s)))


def test_pairwise_r2_pairwise_complete_under_missing_data():
    """r2 under missing_data='include' must use one sample set -- the pair's
    jointly-valid haplotypes -- for both the joint 11-frequency and each
    site's own frequency, not a site's separate (larger) marginal valid set.

    Site 0 is missing on haplotypes 0-3, present (0,0,0,0,1,1,1,1,0,0,0,0) on
    4-15. Site 1 is missing on haplotypes 12-15, present
    (1,1,1,1,0,0,0,0,1,1,1,1) on 0-11. Their only shared valid haplotypes are
    4-11, where the two columns are identical -- a perfect correlation, r2
    exactly 1. Each site's own marginal frequency (1/3 and 2/3, computed over
    its own full valid set) differs from its frequency restricted to that
    shared set (1/2 for both); using the marginal frequencies for D and the
    r2 denominator gives 25/16 -- not just wrong, but impossible for a real
    r2, which cannot exceed 1.
    """
    hap = np.array([
        [-1, 1], [-1, 1], [-1, 1], [-1, 1],
        [0, 0], [0, 0], [0, 0], [0, 0],
        [1, 1], [1, 1], [1, 1], [1, 1],
        [0, -1], [0, -1], [0, -1], [0, -1],
    ], dtype=np.int8)
    pos = np.array([100, 200], dtype=np.int64)
    hm = HaplotypeMatrix(hap, pos, 0, 1000)
    hm.transfer_to_gpu()

    r2 = cp.asnumpy(hm.pairwise_r2())
    assert np.isclose(r2[0, 1], 1.0, rtol=1e-9, atol=1e-12)

    # A 2-site matrix has exactly one pair, so ZnS (the tiled naive path) is
    # that pair's r2 -- checks the tiled and dense paths agree.
    _agree(zns(hm, estimator="r2"), 1.0)


def test_pairwise_r2_matches_pairwise_complete_correlation_multi_site():
    """General oracle: with several sites each missing a distinct block of
    haplotypes (so every pair overlaps on a different subset), r2 under
    missing_data='include' must equal the direct Pearson correlation squared
    computed independently in numpy, restricted to each pair's own jointly-
    valid haplotypes."""
    rng = np.random.RandomState(0)
    n_hap, n_var = 24, 5
    hap = rng.randint(0, 2, size=(n_hap, n_var)).astype(np.int8)
    block = n_hap // n_var
    for v in range(n_var):
        hap[v * block:(v + 1) * block, v] = -1
    pos = ((np.arange(n_var) + 1) * 100).astype(np.int64)
    hm = HaplotypeMatrix(hap, pos, 0, 1000)
    hm.transfer_to_gpu()

    r2 = cp.asnumpy(hm.pairwise_r2())
    hap_f = hap.astype(np.float64)
    expected = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i + 1, n_var):
            valid = (hap[:, i] >= 0) & (hap[:, j] >= 0)
            ai, aj = hap_f[valid, i], hap_f[valid, j]
            # The fix should not be exercised on a degenerate (zero-variance)
            # pair -- assert the construction avoids that rather than
            # silently skip it.
            assert ai.std() > 0 and aj.std() > 0, (i, j)
            expected[i, j] = expected[j, i] = np.corrcoef(ai, aj)[0, 1] ** 2

    for i in range(n_var):
        for j in range(i + 1, n_var):
            assert np.isclose(r2[i, j], expected[i, j], rtol=1e-9, atol=1e-9), (i, j)

    iu = np.triu_indices(n_var, k=1)
    _agree(zns(hm, estimator="r2"), float(expected[iu].mean()))


def test_zns_tiled_excludes_undefined_pair_unlike_closed_form_count():
    """Counterpart to the multi-site test above, which deliberately avoids a
    degenerate pair: here sites 0 and 1 share no valid haplotype at all (an
    undefined pair, not a measured zero), while every other pair is defined.
    _zns_tiled must exclude it from both the sum and the pair count, not
    divide by the closed-form m*(m-1) that assumes every pair is defined."""
    hap = np.array([
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [0, -1, 0, 0],
        [1, -1, 1, 0],
        [-1, 1, 0, 1],
        [-1, 0, 1, 1],
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
    ], dtype=np.int8)
    pos = np.array([100, 200, 300, 400], dtype=np.int64)
    hm = HaplotypeMatrix(hap, pos, 0, 1000)
    hm.transfer_to_gpu()

    r2 = cp.asnumpy(hm.pairwise_r2())
    assert np.isnan(r2[0, 1]) and np.isnan(r2[1, 0])

    finite = ~np.isnan(r2)
    np.fill_diagonal(finite, False)
    expected = (np.nansum(r2) - np.nansum(np.diag(r2))) / finite.sum()

    _agree(zns(hm, estimator="r2"), float(expected))


@pytest.mark.parametrize("use_projection", [False, True], ids=["naive", "proj"])
def test_zns_from_precomputed_tiling_invariant(use_projection):
    # tile_size is an implementation detail: a small tile forces the
    # cross-tile (off-diagonal) accumulation path, which must match the
    # single-tile result.
    hm = _hm()
    hc, vm = _precomputed(hm)
    m = hm.num_variants
    single = _zns_from_precomputed(hc, vm, 0, m, tile_size=512,
                                   use_projection=use_projection)
    multi = _zns_from_precomputed(hc, vm, 0, m, tile_size=2,
                                  use_projection=use_projection)
    _agree(multi, single)


# ── mu_ld ──────────────────────────────────────────────────────────────
def test_mu_ld_too_few_variants_returns_nan():
    hm = HaplotypeMatrix(_X[:, :1].copy(), _POS[:1].copy(), 0, 1000)
    assert np.isnan(mu_ld(hm))


def test_mu_ld_fully_exclusive_patterns():
    # Left half and right half patterns pair one-to-one, so every left pattern
    # maps to exactly one right pattern (and vice versa) -> mu_ld == 1.
    X = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ], dtype=np.int8)
    hm = HaplotypeMatrix(X, np.array([1, 2, 3, 4], dtype=np.int64), 0, 10)
    assert mu_ld(hm) == pytest.approx(1.0)


def test_mu_ld_partial_exclusivity():
    # Left patterns: {00: h0,h1}, {11: h2,h3}. Right patterns: {00: h0,h2,h3},
    # {11: h1}. Left 11 -> only right 00 (exclusive); left 00 -> {00,11} (not).
    # Symmetric on the right -> (1/2 + 1/2) / 2 = 0.5.
    X = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ], dtype=np.int8)
    hm = HaplotypeMatrix(X, np.array([1, 2, 3, 4], dtype=np.int64), 0, 10)
    assert mu_ld(hm) == pytest.approx(0.5)


def test_mu_ld_exclude_missing_runs():
    X = _X.copy()
    X[0, 0] = -1
    hm = HaplotypeMatrix(X, _POS.copy(), 0, 1000)
    val = mu_ld(hm, missing_data="exclude")
    assert 0.0 <= float(val) <= 1.0


# ── _r2_matrix_diploid (dosage correlation r^2) ────────────────────────
def test_r2_matrix_diploid_value_and_array_input():
    geno = np.array([[0, 1], [1, 1], [2, 0], [1, 2], [0, 0]], dtype=np.int8)
    # Raw ndarray input (not a GenotypeMatrix) exercises the coercion branch.
    r2 = _r2_matrix_diploid(geno)
    expected = np.corrcoef(geno[:, 0], geno[:, 1])[0, 1] ** 2
    assert float(cp.asnumpy(r2)[0, 1]) == pytest.approx(expected)
    # Same result via the GenotypeMatrix path.
    gm = GenotypeMatrix(geno.copy(), np.array([1, 2], dtype=np.int64))
    _agree(r2, _r2_matrix_diploid(gm))


def test_r2_matrix_diploid_zero_variance_site_is_nan():
    # Column 1 is monomorphic (constant dosage) -> its row/column are NaN.
    geno = np.array([[0, 1], [1, 1], [2, 1], [1, 1]], dtype=np.int8)
    r2 = cp.asnumpy(_r2_matrix_diploid(geno))
    assert np.isnan(r2[0, 1]) and np.isnan(r2[1, 0])


def test_r2_matrix_diploid_no_overlap_is_nan_not_zero():
    # Column 0 valid only on individuals 0-1, column 1 only on 2-3: zero
    # individuals jointly valid, so this pair's correlation cannot be
    # estimated at all -- nan, not a measured zero (each column is still
    # individually polymorphic, so this isn't the whole-site-NaN case above).
    geno = np.array([[0, -1], [1, -1], [-1, 5], [-1, 9]], dtype=np.int8)
    r2 = cp.asnumpy(_r2_matrix_diploid(geno))
    assert np.isnan(r2[0, 1]) and np.isnan(r2[1, 0])


def test_r2_matrix_diploid_locally_degenerate_pair_is_nan():
    # Site 0 valid on individuals 0-3 as [0, 1, 1, 1] (polymorphic overall).
    # Site 1 valid on individuals 1-4 as [5, 5, 5, 9] (polymorphic overall).
    # Their shared individuals are 1-3, where site 0 reads [1, 1, 1] --
    # constant in that specific jointly-valid subsample, even though neither
    # site is globally monomorphic. The pair's correlation is 0/0, not 0.
    geno = np.array([
        [0, -1],
        [1, 5],
        [1, 5],
        [1, 5],
        [-1, 9],
    ], dtype=np.int8)
    r2 = cp.asnumpy(_r2_matrix_diploid(geno))
    assert np.isnan(r2[0, 1]) and np.isnan(r2[1, 0])


def _hand_r2_matrix_one_scattered_undefined_pair():
    """5x5 r2 matrix, every pair defined except (1, 3), to check that omega
    and zns exclude a single undefined pair from their own sums and pair
    counts rather than treating it as a dropped site (_drop_undefined_sites
    only drops whole rows) or letting it propagate through a cumulative sum."""
    upper = {
        (0, 1): 0.1, (0, 2): 0.2, (0, 3): 0.3, (0, 4): 0.4,
        (1, 2): 0.5, (1, 3): None, (1, 4): 0.6,
        (2, 3): 0.7, (2, 4): 0.8,
        (3, 4): 0.9,
    }
    r2 = np.zeros((5, 5))
    for (i, j), v in upper.items():
        r2[i, j] = r2[j, i] = np.nan if v is None else v
    return r2


def test_omega_excludes_single_undefined_pair_not_whole_sites():
    r2 = _hand_r2_matrix_one_scattered_undefined_pair()
    assert omega(cp.asarray(r2)) == pytest.approx(0.7589285714285713)


def test_zns_excludes_single_undefined_pair_from_pair_count():
    r2 = _hand_r2_matrix_one_scattered_undefined_pair()
    assert zns(cp.asarray(r2)) == pytest.approx(0.5)


def test_r2_matrix_diploid_pairwise_complete_under_missing_data():
    """Same bug as pairwise_r2's, in its continuous-dosage form: mean/variance
    must come from the pair's jointly-valid individuals, not each site's own
    (possibly larger) marginal valid set.

    Site 0 is missing on individuals 0-3, dosage (0,0,0,0,2,2,2,2,0,0,0,0) on
    4-15. Site 1 is missing on individuals 12-15, dosage
    (2,2,2,2,0,0,0,0,2,2,2,2) on 0-11. Their only shared valid individuals are
    4-11, where the two columns are identical -- a perfect correlation, r2
    exactly 1. Each site's own marginal mean (computed over its own full
    valid set) differs from its mean restricted to the shared set, so using
    marginal mean/variance for centering would not give r2 == 1 here.
    """
    geno = np.array([
        [-1, 2], [-1, 2], [-1, 2], [-1, 2],
        [0, 0], [0, 0], [0, 0], [0, 0],
        [2, 2], [2, 2], [2, 2], [2, 2],
        [0, -1], [0, -1], [0, -1], [0, -1],
    ], dtype=np.int8)
    r2 = cp.asnumpy(_r2_matrix_diploid(geno))
    assert np.isclose(r2[0, 1], 1.0, rtol=1e-9, atol=1e-12)

    gm = GenotypeMatrix(geno, np.array([1, 2], dtype=np.int64))
    _agree(zns(gm, estimator="r2"), 1.0)
    assert np.isnan(omega(gm, estimator="r2"))  # fewer than 5 sites


def test_r2_matrix_diploid_matches_pairwise_complete_correlation_multi_site():
    """General oracle: with several sites each missing a distinct block of
    individuals (so every pair overlaps on a different subset), r2 must equal
    the direct Pearson correlation squared computed independently in numpy,
    restricted to each pair's own jointly-valid individuals."""
    rng = np.random.RandomState(1)
    n_ind, n_var = 24, 5
    geno = rng.randint(0, 3, size=(n_ind, n_var)).astype(np.int8)
    block = n_ind // n_var
    for v in range(n_var):
        geno[v * block:(v + 1) * block, v] = -1
    r2 = cp.asnumpy(_r2_matrix_diploid(geno))

    geno_f = geno.astype(np.float64)
    expected = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i + 1, n_var):
            valid = (geno[:, i] >= 0) & (geno[:, j] >= 0)
            ai, aj = geno_f[valid, i], geno_f[valid, j]
            assert ai.std() > 0 and aj.std() > 0, (i, j)
            expected[i, j] = expected[j, i] = np.corrcoef(ai, aj)[0, 1] ** 2

    for i in range(n_var):
        for j in range(i + 1, n_var):
            assert np.isclose(r2[i, j], expected[i, j], rtol=1e-9, atol=1e-9), (i, j)

    gm = GenotypeMatrix(geno, (np.arange(n_var) + 1).astype(np.int64))
    iu = np.triu_indices(n_var, k=1)
    _agree(zns(gm, estimator="r2"), float(expected[iu].mean()))


# ── _resolve_r2_matrix (passthrough + dispatch) ────────────────────────
def test_resolve_r2_matrix_passthrough_and_dispatch():
    arr = np.array([[0.0, 0.25], [0.25, 0.0]])
    out = _resolve_r2_matrix(arr)          # non-array -> coerced to cupy float64
    assert isinstance(out, cp.ndarray) and out.dtype == cp.float64
    _agree(out, arr)
    # An already-cupy array is passed straight through (identity).
    carr = cp.asarray(arr)
    assert _resolve_r2_matrix(carr) is carr
    geno = np.array([[0, 1], [1, 1], [2, 0], [1, 2], [0, 0]], dtype=np.int8)
    gm = GenotypeMatrix(geno, np.array([1, 2], dtype=np.int64))
    _agree(_resolve_r2_matrix(gm), _r2_matrix_diploid(gm))


def test_resolve_r2_matrix_exclude_drops_missing_sites():
    # A fully-missing extra column is dropped under missing_data='exclude',
    # so the result equals resolving the matrix without that column.
    X = np.column_stack([_X, np.full(_X.shape[0], -1, dtype=np.int8)])
    pos = np.append(_POS, 600).astype(np.int64)
    hm = HaplotypeMatrix(X, pos, 0, 1000)
    hm.transfer_to_gpu()
    hm_complete = _hm()  # the same matrix without the missing column
    _agree(_resolve_r2_matrix(hm, missing_data="exclude"),
           _resolve_r2_matrix(hm_complete, missing_data="include"))


def test_resolve_r2_matrix_exclude_genotype_path():
    # GenotypeMatrix exclude branch: a fully-missing column is dropped.
    geno = np.array([[0, 1, -1], [1, 1, -1], [2, 0, -1], [1, 2, -1],
                     [0, 0, -1]], dtype=np.int8)
    gm = GenotypeMatrix(geno, np.array([1, 2, 3], dtype=np.int64))
    gm_complete = GenotypeMatrix(geno[:, :2].copy(),
                                 np.array([1, 2], dtype=np.int64))
    _agree(_resolve_r2_matrix(gm, missing_data="exclude"),
           _resolve_r2_matrix(gm_complete, missing_data="include"))


# ── compute_ld_statistics multi-stat dispatcher ────────────────────────
def test_compute_ld_statistics_multi_stat():
    counts = cp.array([[4, 3, 2, 1], [5, 0, 0, 5], [3, 3, 2, 2]],
                      dtype=cp.float64)
    res = compute_ld_statistics(
        counts, statistics=["r", "r_squared", "dd", "dz", "pi2"])
    _agree(res["r"], r(counts))
    _agree(res["r_squared"], r_squared(counts))
    _agree(res["dd"], dd(counts, None, None))
    _agree(res["dz"], dz(counts, None, None))
    _agree(res["pi2"], pi2(counts, None, None))


def test_compute_ld_statistics_unknown_raises():
    counts = cp.array([[4, 3, 2, 1]], dtype=cp.float64)
    with pytest.raises(ValueError, match="Unknown statistic"):
        compute_ld_statistics(counts, statistics=["bogus"])


# ── _get_pop_data n_valid tuple / 2D / scalar handling ─────────────────
def test_get_pop_data_nvalid_forms():
    counts = cp.array([[4, 3, 2, 1, 5, 0, 0, 5]], dtype=cp.float64)  # 2 pops
    # tuple with an explicit per-pop count
    *_, n = _get_pop_data(counts, (cp.array([10.0]), cp.array([9.0])), 1)
    _agree(n, cp.array([9.0]))
    # tuple entry None / out of range -> falls back to the pop's count sum
    *_, n = _get_pop_data(counts, (None,), 0)
    _agree(n, cp.array([10.0]))
    # 2D n_valid -> column select
    *_, n = _get_pop_data(counts, cp.array([[10.0, 12.0]]), 1)
    _agree(n, cp.array([12.0]))
    # 1D array n_valid (not tuple, not 2D) -> used as-is
    *_, n = _get_pop_data(counts, cp.array([7.0]), 0)
    _agree(n, cp.array([7.0]))
    # None -> sum of that pop's four counts
    *_, n = _get_pop_data(counts, None, 0)
    _agree(n, cp.array([10.0]))


# ── dd/dz/pi2 undefined-pair sentinel (NaN below each threshold) ───────
def test_dd_single_below_threshold_is_nan():
    below = cp.array([[1, 1, 1, 0]], dtype=cp.float64)  # n=3 < 4
    above = cp.array([[2, 2, 2, 2]], dtype=cp.float64)  # n=8 >= 4
    assert bool(cp.isnan(dd(below)[0]))
    assert bool(cp.isfinite(dd(above)[0]))


def test_dd_between_below_threshold_is_nan():
    # pop0 well-sampled (n=8); pop1 has only one valid haplotype (n=1 < 2).
    counts = cp.array([[2, 2, 2, 2, 1, 0, 0, 0]], dtype=cp.float64)
    assert bool(cp.isnan(dd(counts, populations=(0, 1))[0]))
    well_sampled = cp.array([[2, 2, 2, 2, 2, 2, 2, 2]], dtype=cp.float64)
    assert bool(cp.isfinite(dd(well_sampled, populations=(0, 1))[0]))


def test_dz_single_below_threshold_is_nan():
    below = cp.array([[1, 1, 1, 0]], dtype=cp.float64)  # n=3 < 4
    above = cp.array([[2, 2, 2, 2]], dtype=cp.float64)
    assert bool(cp.isnan(dz(below)[0]))
    assert bool(cp.isfinite(dz(above)[0]))


class TestDzMultiThresholds:
    """Each of _dz_multi's four population-pattern branches gates on its own
    threshold; below it the result is NaN, not 0.0."""

    def test_iij_pattern(self):
        # Dz(i,i,j) needs n_i>=3 & n_j>=1.
        below = cp.array([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=cp.float64)  # n_i=2
        above = cp.array([[2, 1, 1, 2, 1, 0, 0, 0]], dtype=cp.float64)  # n_i=6
        assert bool(cp.isnan(dz(below, populations=(0, 0, 1))[0]))
        assert bool(cp.isfinite(dz(above, populations=(0, 0, 1))[0]))

    def test_iji_pattern(self):
        # Dz(i,j,i) shares the iij branch's threshold with the roles swapped.
        below = cp.array([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=cp.float64)
        above = cp.array([[2, 1, 1, 2, 1, 0, 0, 0]], dtype=cp.float64)
        assert bool(cp.isnan(dz(below, populations=(0, 1, 0))[0]))
        assert bool(cp.isfinite(dz(above, populations=(0, 1, 0))[0]))

    def test_ijj_pattern(self):
        # Dz(i,j,j) needs n_i>=2 & n_j>=2.
        below = cp.array([[2, 2, 2, 2, 1, 0, 0, 0]], dtype=cp.float64)  # n_j=1
        above = cp.array([[2, 2, 2, 2, 1, 1, 0, 0]], dtype=cp.float64)  # n_j=2
        assert bool(cp.isnan(dz(below, populations=(0, 1, 1))[0]))
        assert bool(cp.isfinite(dz(above, populations=(0, 1, 1))[0]))

    def test_all_different_pattern(self):
        # Dz(i,j,k) needs n_i>=2 & n_j>=1 & n_k>=1.
        below = cp.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
                         dtype=cp.float64)  # n_i=1 < 2
        above = cp.array([[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
                         dtype=cp.float64)  # n_i=2
        assert bool(cp.isnan(dz(below, populations=(0, 1, 2))[0]))
        assert bool(cp.isfinite(dz(above, populations=(0, 1, 2))[0]))


def test_pi2_single_below_threshold_is_nan():
    below = cp.array([[1, 1, 1, 0]], dtype=cp.float64)  # n=3 < 4
    above = cp.array([[2, 2, 2, 2]], dtype=cp.float64)
    assert bool(cp.isnan(pi2(below)[0]))
    assert bool(cp.isfinite(pi2(above)[0]))


class TestPi2MultiThresholds:
    def test_two_distinct_pops_each_need_at_least_two(self):
        # pi2(i,i,k,k) needs n_i>=2 & n_k>=2.
        below = cp.array([[2, 2, 2, 2, 1, 0, 0, 0]], dtype=cp.float64)  # n_k=1
        above = cp.array([[2, 2, 2, 2, 1, 1, 0, 0]], dtype=cp.float64)  # n_k=2
        assert bool(cp.isnan(pi2(below, populations=(0, 0, 1, 1))[0]))
        assert bool(cp.isfinite(pi2(above, populations=(0, 0, 1, 1))[0]))

    def test_all_four_distinct_pops_each_need_at_least_one(self):
        below = cp.array(
            [[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
            dtype=cp.float64)  # pop3 (last) has n=0
        above = cp.array(
            [[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
            dtype=cp.float64)  # pop3 has n=1
        assert bool(cp.isnan(pi2(below, populations=(0, 1, 2, 3))[0]))
        assert bool(cp.isfinite(pi2(above, populations=(0, 1, 2, 3))[0]))

    def test_orbit_uniformly_undefined_when_one_population_is_empty(self):
        # pi2(i,i,i,j) appears in several structurally different formula
        # branches across its symmetry orbit (_pi2_iikl, _pi2_ijkk), but
        # population j's multiplicity (appearing exactly once) is invariant
        # across every arrangement, so every branch requires n_j>=1 and the
        # whole orbit is undefined together when j is entirely missing.
        counts = cp.array(
            [[2, 2, 2, 2, 0, 0, 0, 0]], dtype=cp.float64)  # pop1 (j) has n=0
        assert bool(cp.isnan(pi2(counts, populations=(0, 0, 0, 1))[0]))
        assert bool(cp.isnan(pi2(counts, populations=(1, 1, 1, 0))[0]))
