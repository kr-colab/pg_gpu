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
    compute_ld_statistics, dd, dz, mu_ld, pi2, r, r_squared, zns,
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


def test_zns_from_precomputed_too_few_sites_returns_zero():
    hm = _hm()
    hc, vm = _precomputed(hm)
    # A single-column range has < 2 segregating sites -> 0.0.
    assert _zns_from_precomputed(hc, vm, 0, 1) == 0.0


def test_zns_naive_matches_direct_pairwise_correlation():
    # Independent oracle for the naive tile math: on complete data ZnS is the
    # mean over distinct site pairs of the squared column correlation.
    hm = _hm()
    Xf = _X.astype(np.float64)
    m = Xf.shape[1]
    r2s = [np.corrcoef(Xf[:, i], Xf[:, j])[0, 1] ** 2
           for i in range(m) for j in range(i + 1, m)]
    _agree(zns(hm, estimator="r2"), float(np.mean(r2s)))


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
def test_mu_ld_too_few_variants_returns_zero():
    hm = HaplotypeMatrix(_X[:, :1].copy(), _POS[:1].copy(), 0, 1000)
    assert mu_ld(hm) == 0.0


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
