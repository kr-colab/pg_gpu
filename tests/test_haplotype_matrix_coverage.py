"""Residual coverage for haplotype_matrix.py.

Covers the missing-data pairwise haplotype tallies (single- and two-population),
the missing-data introspection helpers, pairwise_r2 dropping multiallelic sites,
accessible-mask span/reset, and streaming-LD chunk stitching. Each gap is pinned
to an independent oracle -- a plain numpy hand-masked reference, a hand count, or
the eager (non-streaming) result -- rather than a golden constant.
"""
import numpy as np
import cupy as cp
import pytest

from pg_gpu import BiallelicOnlyWarning, HaplotypeMatrix
from pg_gpu.accessible import AccessibleMask


def _host(a):
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


# A 0/1 matrix with scattered missing (-1); 8 haplotypes x 5 variants.
X_MISS = np.array([
    [0, 1, 0, 1, 0],
    [1, 1, -1, 0, 1],
    [0, 0, 1, 1, -1],
    [1, -1, 0, 0, 1],
    [0, 1, 1, -1, 0],
    [1, 0, -1, 1, 1],
    [-1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
], dtype=np.int8)
POS5 = np.array([100, 200, 300, 400, 500], dtype=np.int64)


def _hm(X, pos, *, gpu=False, **kw):
    m = HaplotypeMatrix(X.copy(), pos.copy(), **kw)
    if gpu:
        m.transfer_to_gpu()
    return m


def _ref_tally(X):
    """Per upper-triangle pair [n11, n10, n01, n00] and n_valid, counting only
    haplotypes non-missing (!= -1) at both loci. Independent numpy reference."""
    n_var = X.shape[1]
    ii, jj = np.triu_indices(n_var, k=1)
    counts, n_valid = [], []
    for i, j in zip(ii, jj):
        valid = (X[:, i] != -1) & (X[:, j] != -1)
        a, b = X[valid, i], X[valid, j]
        counts.append([
            int(np.sum((a == 1) & (b == 1))),
            int(np.sum((a == 1) & (b == 0))),
            int(np.sum((a == 0) & (b == 1))),
            int(np.sum((a == 0) & (b == 0))),
        ])
        n_valid.append(int(valid.sum()))
    return np.array(counts), np.array(n_valid)


# ── A. Missing-data pairwise haplotype tallies ─────────────────────────
def test_tally_single_pop_with_missing():
    hm = _hm(X_MISS, POS5, gpu=True)
    counts, n_valid = hm.tally_gpu_haplotypes()
    ref_c, ref_v = _ref_tally(X_MISS)
    np.testing.assert_array_equal(_host(counts), ref_c)
    np.testing.assert_array_equal(_host(n_valid), ref_v)


def test_tally_two_pops_with_missing():
    sample_sets = {"p1": [0, 1, 2, 3], "p2": [4, 5, 6, 7]}
    hm = _hm(X_MISS, POS5, gpu=True, sample_sets=sample_sets)
    counts, v1, v2 = hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    c1, rv1 = _ref_tally(X_MISS[sample_sets["p1"]])
    c2, rv2 = _ref_tally(X_MISS[sample_sets["p2"]])
    np.testing.assert_array_equal(_host(counts)[:, :4], c1)
    np.testing.assert_array_equal(_host(counts)[:, 4:], c2)
    np.testing.assert_array_equal(_host(v1), rv1)
    np.testing.assert_array_equal(_host(v2), rv2)
    forced = hm.tally_gpu_haplotypes_two_pops_with_missing("p1", "p2")
    for a, b in zip(forced, (counts, v1, v2)):
        np.testing.assert_array_equal(_host(a), _host(b))


def test_tally_tuple_row_lists():
    # Tuple row lists are valid sample sets; they must select rows, not index
    # one axis per element.
    hm = _hm(X_MISS, POS5, gpu=True, sample_sets={"p1": (0, 1, 2, 3), "p2": (4, 5, 6, 7)})
    counts, n_valid = hm.tally_gpu_haplotypes(pop="p1")
    ref_c, ref_v = _ref_tally(X_MISS[:4])
    np.testing.assert_array_equal(_host(counts), ref_c)
    np.testing.assert_array_equal(_host(n_valid), ref_v)
    counts2, _, _ = hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    np.testing.assert_array_equal(_host(counts2)[:, :4], ref_c)
    np.testing.assert_array_equal(_host(counts2)[:, 4:], _ref_tally(X_MISS[4:])[0])


def test_tally_two_pops_all_missing_pop_pair():
    # Pop1 is entirely missing at variant 0, so any pair (0, j) has n_valid1==0
    # and pop1's counts are zero, while pop2 is tallied normally.
    X = np.array([
        [-1, 0, 1],   # p1
        [-1, 1, 0],   # p1
        [0, 1, 1],    # p2
        [1, 0, 0],    # p2
    ], dtype=np.int8)
    hm = _hm(X, np.array([100, 200, 300], dtype=np.int64), gpu=True,
             sample_sets={"p1": [0, 1], "p2": [2, 3]})
    counts, v1, v2 = hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    c1, rv1 = _ref_tally(X[[0, 1]])
    c2, rv2 = _ref_tally(X[[2, 3]])
    np.testing.assert_array_equal(_host(counts)[:, :4], c1)
    np.testing.assert_array_equal(_host(counts)[:, 4:], c2)
    np.testing.assert_array_equal(_host(v1), rv1)
    np.testing.assert_array_equal(_host(v2), rv2)
    # Upper-triangle pair order is (0,1), (0,2), (1,2); the first two involve
    # variant 0, where pop1 is fully missing.
    assert _host(v1)[0] == 0 and _host(v1)[1] == 0
    assert np.all(_host(counts)[:2, :4] == 0)


def test_tally_pop_validation_raises():
    hm = _hm(X_MISS, POS5, gpu=True)  # no sample_sets
    with pytest.raises(ValueError, match="sample_sets must be defined"):
        hm.tally_gpu_haplotypes(pop="p1")
    with pytest.raises(ValueError, match="sample_sets must be defined"):
        hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    hm2 = _hm(X_MISS, POS5, gpu=True, sample_sets={"p1": [0, 1]})
    with pytest.raises(KeyError):
        hm2.tally_gpu_haplotypes(pop="nope")
    with pytest.raises(KeyError):
        hm2.tally_gpu_haplotypes_two_pops("p1", "nope")


# Complete 0/1 data, 300 haplotypes x 6 variants, where many pairs share far
# more than 127 carriers: an int8 sample-contracting matmul wraps there.
# Column 0 is fixed for the alt allele.
X_MANY = (np.random.default_rng(0).random((300, 6)) < 0.8).astype(np.int8)
X_MANY[:, 0] = 1
POS6 = np.arange(1, 7, dtype=np.int64) * 100


def test_tally_complete_data_many_carriers():
    X = X_MANY
    hm = _hm(X, POS6, gpu=True, sample_sets={"p1": list(range(200)), "p2": list(range(200, 300))})
    counts, n_valid = hm.tally_gpu_haplotypes()
    assert n_valid is None
    np.testing.assert_array_equal(_host(counts), _ref_tally(X)[0])
    counts2, v1, v2 = hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    assert v1 is None and v2 is None
    np.testing.assert_array_equal(_host(counts2)[:, :4], _ref_tally(X[:200])[0])
    np.testing.assert_array_equal(_host(counts2)[:, 4:], _ref_tally(X[200:])[0])


def test_tally_complete_and_missing_paths_agree():
    # One all-missing haplotype sends the tally down the missing-aware path but
    # leaves every pair's counts unchanged, so both paths must agree exactly.
    X = X_MANY
    Xm = np.vstack([X, np.full((1, X.shape[1]), -1, np.int8)])
    c_fast, _ = _hm(X, POS6, gpu=True).tally_gpu_haplotypes()
    c_miss, n_valid = _hm(Xm, POS6, gpu=True).tally_gpu_haplotypes()
    np.testing.assert_array_equal(_host(c_fast), _host(c_miss))
    assert c_fast.dtype == c_miss.dtype
    np.testing.assert_array_equal(_host(n_valid), X.shape[0])


@pytest.mark.parametrize("X01", [X_MISS, np.where(X_MISS < 0, 0, X_MISS).astype(np.int8)],
                         ids=["missing", "complete"])
@pytest.mark.parametrize("ref_code,alt_code", [(0, 2), (1, 2)])
def test_tally_allele_coding_invariant(X01, ref_code, alt_code):
    # {0,2} and reference-absent {1,2} codings must tally like {0,1}.
    Xc = np.where(X01 == 1, alt_code, np.where(X01 == 0, ref_code, -1)).astype(np.int8)
    sample_sets = {"p1": [0, 1, 2, 3], "p2": [4, 5, 6, 7]}
    hm = _hm(Xc, POS5, gpu=True, sample_sets=sample_sets)
    ref_c, _ = _ref_tally(X01)
    np.testing.assert_array_equal(_host(hm.tally_gpu_haplotypes()[0]), ref_c)
    counts, _, _ = hm.tally_gpu_haplotypes_two_pops("p1", "p2")
    np.testing.assert_array_equal(_host(counts)[:, :4], _ref_tally(X01[:4])[0])
    np.testing.assert_array_equal(_host(counts)[:, 4:], _ref_tally(X01[4:])[0])


def test_tally_all_missing_site():
    X = X_MISS.copy()
    X[:, 2] = -1
    counts, n_valid = _hm(X, POS5, gpu=True).tally_gpu_haplotypes()
    ref_c, ref_v = _ref_tally(X)
    np.testing.assert_array_equal(_host(counts), ref_c)
    np.testing.assert_array_equal(_host(n_valid), ref_v)


def test_tally_multiallelic_site_warns_and_lumps():
    # Column 1 has three alleles: the highest (2) counts as 1, both others as 0.
    X = np.array([[0, 0], [1, 1], [1, 2], [0, 2]], dtype=np.int8)
    hm = _hm(X, np.array([100, 200], dtype=np.int64), gpu=True)
    with pytest.warns(BiallelicOnlyWarning, match="tally_gpu_haplotypes"):
        counts, _ = hm.tally_gpu_haplotypes()
    lumped = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(_host(counts), _ref_tally(lumped)[0])


@pytest.mark.parametrize("n,missing", [(2**24, False), (2**24 + 1, False), (2**24 + 2, True)])
def test_tally_exact_at_float32_limit(n, missing):
    # float32 holds integers exactly only up to 2**24, so 2**24 + 1 carriers
    # pins the switch to float64. One missing call at variant 0 pins the
    # missing-data path above the limit too.
    X = cp.ones((n, 3), dtype=cp.int8)
    if missing:
        X[0, 0] = -1
    counts, n_valid = HaplotypeMatrix._tally_pairs_impl(X)
    n_ok = n - 1 if missing else n
    np.testing.assert_array_equal(_host(counts), [[n_ok, 0, 0, 0], [n_ok, 0, 0, 0], [n, 0, 0, 0]])
    if missing:
        np.testing.assert_array_equal(_host(n_valid), [n_ok, n_ok, n])


# ── D. Missing-data introspection ──────────────────────────────────────
def test_missing_introspection_gpu():
    hm = _hm(X_MISS, POS5, gpu=True)
    miss = X_MISS < 0
    np.testing.assert_array_equal(_host(hm.is_missing()), miss)
    np.testing.assert_array_equal(_host(hm.is_missing(axis=0)), miss.any(0))
    np.testing.assert_array_equal(_host(hm.is_missing(axis=1)), miss.any(1))
    np.testing.assert_array_equal(_host(hm.is_called(axis=0)), ~miss.any(0))
    assert int(_host(hm.count_missing())) == int(miss.sum())
    np.testing.assert_array_equal(_host(hm.count_missing(axis=0)), miss.sum(0))
    np.testing.assert_array_equal(_host(hm.count_called(axis=1)), (~miss).sum(1))


def test_missing_introspection_cpu():
    hm = _hm(X_MISS, POS5)  # CPU-resident (numpy)
    assert hm.device == "CPU"
    miss = X_MISS < 0
    np.testing.assert_array_equal(hm.is_missing(axis=0), miss.any(0))
    np.testing.assert_array_equal(hm.is_called(axis=1), ~miss.any(1))
    assert int(hm.count_missing()) == int(miss.sum())
    np.testing.assert_array_equal(hm.count_called(axis=0), (~miss).sum(0))


def test_summarize_missing_data():
    hm = _hm(X_MISS, POS5, gpu=True)
    miss = X_MISS < 0
    s = hm.summarize_missing_data()
    assert s["total_missing_calls"] == int(miss.sum())
    assert s["total_calls"] == X_MISS.size
    assert s["missing_freq_overall"] == pytest.approx(miss.sum() / X_MISS.size)
    assert s["variants_with_no_missing"] == int(np.sum(miss.sum(0) == 0))
    assert s["samples_with_no_missing"] == int(np.sum(miss.sum(1) == 0))
    assert s["max_missing_per_variant"] == int(miss.sum(0).max())
    assert s["max_missing_per_sample"] == int(miss.sum(1).max())


def test_pairwise_r2_drops_multiallelic_site():
    # Six haplotypes = three diploid individuals (rows paired 0-1, 2-3, 4-5).
    # Column 2 carries a third allele (value 2) -> multiallelic, dropped from
    # the diploid-dosage conversion, so its row/column come back NaN. The
    # biallelic columns are built to have dosage variance across individuals so
    # the surviving block is finite.
    Xr = np.array([
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 2, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
    ], dtype=np.int8)
    hm = _hm(Xr, np.array([100, 200, 300, 400], dtype=np.int64), gpu=True)
    with pytest.warns(BiallelicOnlyWarning):
        r2 = _host(hm.pairwise_r2(estimator="rogers_huff"))
    bad = 2
    off = ~np.eye(4, dtype=bool)
    assert np.all(np.isnan(r2[bad][off[bad]]))       # whole row NaN off-diagonal
    assert np.all(np.isnan(r2[:, bad][off[:, bad]]))  # whole column NaN
    # The surviving block must equal rogers_huff r2 on the matrix with the
    # multiallelic column removed -- an independent oracle for the values, not
    # just their finiteness.
    good = [0, 1, 3]
    block = r2[np.ix_(good, good)]
    sub = _hm(Xr[:, good], np.array([100, 200, 400], dtype=np.int64), gpu=True)
    r2_sub = _host(sub.pairwise_r2(estimator="rogers_huff"))
    np.testing.assert_allclose(block, r2_sub)


# ── B. Accessible-mask span and reset ──────────────────────────────────
def test_accessible_bases_no_bounds_and_remove():
    # No chrom bounds -> _accessible_bases_in_range counts the whole mask.
    mask = np.ones(400, dtype=bool)
    mask[:100] = False  # 300 accessible bases
    hm = _hm(X_MISS, POS5)
    hm.set_accessible_mask(AccessibleMask(mask, offset=0))
    assert hm.n_total_sites == 300
    assert hm.accessible_mask is not None
    hm.remove_accessible_mask()
    assert hm.accessible_mask is None
    assert hm.n_total_sites is None


def test_get_span_modes():
    span_callable = int(POS5.max() - POS5.min()) + 1
    # auto -> n_total_sites when set and no mask.
    hm = _hm(X_MISS, POS5, n_total_sites=1234)
    assert hm.get_span("auto") == 1234
    # callable span (max - min + 1 of positions), GPU and CPU paths, against an
    # independent value rather than each other.
    hm_cpu = _hm(X_MISS, POS5)
    assert hm_cpu.get_span("callable") == span_callable
    hm_gpu = _hm(X_MISS, POS5, gpu=True)
    assert hm_gpu.get_span("callable") == span_callable
    # No mask / no n_total_sites / no bounds: auto and per_base fall to callable.
    assert hm_cpu.get_span("auto") == span_callable
    assert hm_cpu.get_span("per_base") == span_callable
    # WITH chrom bounds (no mask, no n_total_sites): per_base and auto use the
    # inclusive span end - start + 1.
    hm_b = _hm(X_MISS, POS5, chrom_start=1000, chrom_end=6000)
    assert hm_b.get_span("per_base") == 6000 - 1000 + 1
    assert hm_b.get_span("auto") == 6000 - 1000 + 1
    with pytest.raises(ValueError, match="Invalid span mode"):
        hm_cpu.get_span("bogus")
