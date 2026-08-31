"""Unit tests for the LD pair-pipeline helpers.

The chunk-size estimator is pure arithmetic on an explicit memory budget, so
these tests pass ``available_memory_bytes`` and never query the GPU or try to
provoke an out-of-memory condition (which is platform-dependent).
"""
import pytest

from pg_gpu.ld_pipeline import estimate_ld_chunk_size, ld_names


class TestEstimateLdChunkSize:
    def test_single_pop_matches_historical_anchor(self):
        # The workspace term is anchored so that at one population (n_ld == 3)
        # it equals the historical 150 * num_pops, leaving P=1 unchanged.
        assert len(ld_names(1)) == 3
        H, budget = 50, 1050 * 1_000_000  # bytes_per_pair = (4*50 + 150)*3 = 1050
        assert estimate_ld_chunk_size(H, budget, num_pops=1) == 1_000_000

    def test_chunk_shrinks_as_populations_grow(self):
        # More populations -> more LD statistics (3/15/45/105) -> heavier
        # per-pair working set -> a smaller chunk for the same budget.
        H, budget = 100, 5_000_000_000
        sizes = [estimate_ld_chunk_size(H, budget, num_pops=p) for p in (1, 2, 3, 4)]
        assert sizes == sorted(sizes, reverse=True)
        assert len(set(sizes)) == 4  # strictly decreasing, none clamped here

    def test_scales_with_statistic_count(self):
        # Lock the 50 * n_ld workspace term for four populations (n_ld == 105).
        H, budget = 100, 5_000_000_000
        bytes_per_pair = (4 * H * 4 + 50 * len(ld_names(4))) * 3
        assert estimate_ld_chunk_size(H, budget, num_pops=4) == budget // bytes_per_pair

    def test_estimate_never_exceeds_capacity(self):
        # On a tight budget where fewer than 100k pairs fit, the estimate
        # returns what fits rather than being lifted to a fixed minimum (the
        # removed 100k floor), which would have overshot memory.
        H, budget = 100, 1_000_000_000
        bytes_per_pair = (4 * H * 4 + 50 * len(ld_names(4))) * 3
        fit = budget // bytes_per_pair
        assert fit < 100_000
        assert estimate_ld_chunk_size(H, budget, num_pops=4) == fit

    def test_ceiling_caps_large_budgets(self):
        assert estimate_ld_chunk_size(50, 10**12, num_pops=1) == 10_000_000
