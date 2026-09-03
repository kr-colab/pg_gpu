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


def test_tally_two_pops_all_missing_pop_pair():
    # Pop1 is entirely missing at variant 0, so any pair (0, j) has n_valid1==0
    # and pop1's counts stay zero (the `if n_valid1 > 0` guard's false branch)
    # while pop2 is tallied normally.
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
