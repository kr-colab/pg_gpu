"""Unit tests for pg_gpu.ld_pipeline.iter_pairs_within_distance."""
import cupy as cp
import numpy as np
import pytest

from pg_gpu.ld_pipeline import iter_pairs_within_distance


def _collect(positions_gpu, max_dist, chunk_size):
    """Run the generator and return concatenated (idx_i, idx_j) as numpy arrays."""
    ii_parts = []
    jj_parts = []
    for ci, cj in iter_pairs_within_distance(positions_gpu, max_dist, chunk_size):
        ii_parts.append(ci.get())
        jj_parts.append(cj.get())
    if not ii_parts:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(ii_parts), np.concatenate(jj_parts)


def _naive_pairs(positions, max_dist):
    """CPU oracle: exhaustive (i, j) enumeration with i < j and pos[j]-pos[i] <= max_dist."""
    n = len(positions)
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if positions[j] - positions[i] <= max_dist:
                pairs.add((i, j))
    return pairs


@pytest.mark.parametrize("n", [2, 10, 50])
@pytest.mark.parametrize("max_dist_factor", [0.1, 0.5, 1.0, 10.0])
def test_matches_naive_oracle(n, max_dist_factor):
    rng = np.random.default_rng(seed=(n * 997 + int(max_dist_factor * 1000)))
    # Sorted integer positions with gaps.
    positions = np.sort(rng.integers(0, 10 * n, size=n)).astype(np.int64)
    positions = np.unique(positions)
    if len(positions) < 2:
        pytest.skip("duplicates collapsed input")
    span = positions[-1] - positions[0]
    max_dist = max(1, int(span * max_dist_factor))

    pos_gpu = cp.asarray(positions)
    idx_i, idx_j = _collect(pos_gpu, max_dist, chunk_size=7)

    got = set(zip(idx_i.tolist(), idx_j.tolist()))
    expected = _naive_pairs(positions, max_dist)
    assert got == expected
    # All pairs have i < j
    assert np.all(idx_i < idx_j)
    # All within distance
    assert np.all(positions[idx_j] - positions[idx_i] <= max_dist)


def test_chunk_size_independence():
    rng = np.random.default_rng(seed=12345)
    positions = np.sort(rng.integers(0, 100_000, size=500)).astype(np.int64)
    positions = np.unique(positions)
    pos_gpu = cp.asarray(positions)
    max_dist = 5000

    def collect_sorted(cs):
        idx_i, idx_j = _collect(pos_gpu, max_dist, chunk_size=cs)
        order = np.lexsort((idx_j, idx_i))
        return idx_i[order], idx_j[order]

    refs = collect_sorted(1)
    for cs in (1, 100, 10_000, 1_000_000):
        got = collect_sorted(cs)
        np.testing.assert_array_equal(got[0], refs[0])
        np.testing.assert_array_equal(got[1], refs[1])


def test_empty_inputs():
    # n=0
    empty = cp.asarray(np.empty(0, dtype=np.int64))
    assert list(iter_pairs_within_distance(empty, 100, 10)) == []
    # n=1
    single = cp.asarray(np.array([42], dtype=np.int64))
    assert list(iter_pairs_within_distance(single, 100, 10)) == []


def test_max_dist_zero_admits_no_pairs_when_positions_unique():
    positions = cp.asarray(np.arange(10, dtype=np.int64))
    idx_i, idx_j = _collect(positions, max_dist=0, chunk_size=4)
    assert idx_i.size == 0 and idx_j.size == 0


def test_max_dist_huge_admits_every_pair():
    n = 25
    positions = cp.asarray(np.arange(n, dtype=np.int64) * 10)
    idx_i, idx_j = _collect(positions, max_dist=10**9, chunk_size=7)
    assert idx_i.size == n * (n - 1) // 2
    got = set(zip(idx_i.tolist(), idx_j.tolist()))
    expected = {(i, j) for i in range(n) for j in range(i + 1, n)}
    assert got == expected


def test_total_pair_count_matches_cumulative():
    rng = np.random.default_rng(seed=99)
    positions = np.sort(rng.integers(0, 50_000, size=300)).astype(np.int64)
    positions = np.unique(positions)
    pos_gpu = cp.asarray(positions)
    max_dist = 2500

    n = len(positions)
    upper = pos_gpu + max_dist
    j_max = cp.searchsorted(pos_gpu, upper, side='right')
    pairs_per_variant = cp.maximum(0, j_max - cp.arange(n, dtype=cp.int64) - 1)
    expected_total = int(cp.sum(pairs_per_variant).get())

    total = sum(int(ci.size) for ci, _ in
                iter_pairs_within_distance(pos_gpu, max_dist, chunk_size=250))
    assert total == expected_total
