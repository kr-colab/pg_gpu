"""Unit tests for the haplotype hashing kernels behind Garud's H and haplotype counts."""
import cupy as cp
import numpy as np
import pytest

from pg_gpu._haplotype_hash import (
    garud_h_windows, garud_walk, hash_weights, row_hashes, sort_window_hashes,
    window_hashes,
)

from .conftest import founder_haplotypes, garud_reference


def _walk(hap_cpu):
    n_var = hap_cpu.shape[1]
    h1, h2 = window_hashes(cp.asarray(hap_cpu), *hash_weights(n_var), [0], [n_var])
    return [float(s[0]) for s in garud_walk(*sort_window_hashes(h1, h2))]


def test_empty_window_hashes_to_zero():
    rng = np.random.default_rng(0)
    hap = cp.asarray(rng.integers(0, 2, size=(6, 20), dtype=np.int8))
    h1, h2 = window_hashes(hap, *hash_weights(20), [0, 5, 20, 3, 4], [0, 5, 20, 3, 12])
    assert h1.shape == (5, 6)
    np.testing.assert_array_equal(h1[:4].get(), 0.0)
    np.testing.assert_array_equal(h2[:4].get(), 0.0)
    assert np.all(h1[4].get() != 0.0)


def test_hashes_match_numpy_sum():
    rng = np.random.default_rng(1)
    hap_cpu = rng.integers(-1, 3, size=(7, 300), dtype=np.int8)
    w1, w2 = hash_weights(300)
    h1, h2 = window_hashes(cp.asarray(hap_cpu), w1, w2, [0, 100], [100, 300])
    for k, (lo, hi) in enumerate([(0, 100), (100, 300)]):
        ref1 = hap_cpu[:, lo:hi].astype(np.float64) @ w1.get()[lo:hi]
        ref2 = hap_cpu[:, lo:hi].astype(np.float64) @ w2.get()[lo:hi]
        np.testing.assert_allclose(h1[k].get(), ref1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h2[k].get(), ref2, rtol=1e-12, atol=1e-12)


def test_layouts_and_dtypes_bit_identical():
    rng = np.random.default_rng(2)
    base = cp.asarray(rng.integers(0, 2, size=(9, 500), dtype=np.int8))
    w1, w2 = hash_weights(500)
    ws, we = [0, 250, 100], [250, 500, 400]
    ref1, ref2 = window_hashes(base, w1, w2, ws, we)
    variants = [
        cp.asfortranarray(base.astype(cp.int32)),           # from_ts layout
        cp.asfortranarray(base),                            # from_vcf layout
        base.astype(cp.int64),
        cp.ascontiguousarray(base.T).T,                     # transposed-copy view
    ]
    for v in variants:
        h1, h2 = window_hashes(v, w1, w2, ws, we)
        np.testing.assert_array_equal(h1.get(), ref1.get())
        np.testing.assert_array_equal(h2.get(), ref2.get())
    # A column-strided view hashes like its contiguous copy.
    strided = base[:, ::2]
    n = strided.shape[1]
    a1, a2 = window_hashes(strided, *hash_weights(n), [0], [n])
    b1, b2 = window_hashes(cp.ascontiguousarray(strided), *hash_weights(n), [0], [n])
    np.testing.assert_array_equal(a1.get(), b1.get())
    np.testing.assert_array_equal(a2.get(), b2.get())


def test_window_ranges_outside_matrix_raise():
    hap = cp.zeros((3, 10), dtype=cp.int8)
    w1, w2 = hash_weights(10)
    with pytest.raises(ValueError):
        window_hashes(hap, w1, w2, [-1], [5])
    with pytest.raises(ValueError):
        window_hashes(hap, w1, w2, [0], [11])


def test_short_rows_hash_like_a_single_window():
    rng = np.random.default_rng(3)
    hap = cp.asarray(rng.integers(0, 2, size=(8, 1000), dtype=np.int8))
    h1, h2 = window_hashes(hap, *hash_weights(1000), [0], [1000])
    r1, r2 = row_hashes(hap)
    np.testing.assert_array_equal(h1[0].get(), r1.get())
    np.testing.assert_array_equal(h2[0].get(), r2.get())


def test_row_hashes_segments_keep_identical_rows_identical():
    rng = np.random.default_rng(4)
    n_var = 150_000                     # spans three row segments
    hap_cpu = founder_haplotypes(rng, 12, n_var, 3)
    r1, r2 = row_hashes(cp.asarray(hap_cpu))
    _, key_labels = np.unique(np.stack([r1.get(), r2.get()], axis=1), axis=0,
                              return_inverse=True)
    _, row_labels = np.unique(hap_cpu, axis=0, return_inverse=True)
    np.testing.assert_array_equal(key_labels[:, None] == key_labels[None, :],
                                  row_labels[:, None] == row_labels[None, :])


@pytest.mark.parametrize("hap_cpu,expected", [
    (np.zeros((10, 5), dtype=np.int8), (1.0, 1.0, 1.0, 0.0, 1)),
    (np.eye(4, dtype=np.int8), (0.25, 6 / 16, 10 / 16, 0.75, 4)),
    (np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=np.int8),
     (0.5, 1.0, 1.0, 0.5, 2)),
])
def test_walk_closed_forms(hap_cpu, expected):
    np.testing.assert_allclose(_walk(hap_cpu), expected, rtol=1e-12)


def test_walk_matches_reference_past_shared_memory_limits():
    rng = np.random.default_rng(5)
    hap_cpu = founder_haplotypes(rng, 5000, 64, 50)
    np.testing.assert_allclose(_walk(hap_cpu), garud_reference(hap_cpu), rtol=1e-12)


def test_garud_h_windows_no_windows():
    hap = cp.zeros((4, 10), dtype=cp.int8)
    out = garud_h_windows(hap, [], [])
    assert len(out) == 5
    assert all(o.shape == (0,) for o in out)


def test_garud_h_windows_no_variants():
    hap = cp.zeros((5, 0), dtype=cp.int8)
    out = garud_h_windows(hap, [0, 0], [0, 0])
    for o, expected in zip(out, (1.0, 1.0, 1.0, 0.0, 1.0)):
        np.testing.assert_array_equal(o, expected)
