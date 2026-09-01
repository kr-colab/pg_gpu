"""Unit tests for the LD pair-pipeline helpers.

The chunk-size estimator is pure arithmetic on an explicit memory budget, so
these tests pass ``available_memory_bytes`` and never query the GPU or try to
provoke an out-of-memory condition (which is platform-dependent).
"""
import pytest

from pg_gpu.ld_pipeline import estimate_ld_chunk_size, ld_names


class TestEstimateLdChunkSize:
    def test_chunk_shrinks_as_statistics_grow(self):
        # More statistics -> heavier per-pair working set -> a smaller chunk for
        # the same budget. The LD basis grows 3/15/45/105 with the populations,
        # and gathered rows grow with them (H per pop times pops).
        H, budget = 100, 5_000_000_000
        sizes = [estimate_ld_chunk_size(H * p, len(ld_names(p)), budget)
                 for p in (1, 2, 3, 4)]
        assert sizes == sorted(sizes, reverse=True)
        assert len(set(sizes)) == 4  # strictly decreasing, none clamped here

    def test_scales_with_statistic_count(self):
        # Lock the 50 * n_stats workspace term for a 105-statistic basis.
        rows, budget, n_stats = 400, 5_000_000_000, len(ld_names(4))
        bytes_per_pair = (4 * rows + 50 * n_stats) * 3
        assert (estimate_ld_chunk_size(rows, n_stats, budget)
                == budget // bytes_per_pair)

    def test_rejects_nonpositive_statistic_count(self):
        with pytest.raises(ValueError):
            estimate_ld_chunk_size(400, 0, 1_000_000_000)

    def test_returns_what_fits_between_floor_and_ceiling(self):
        # A budget admitting more than the floor and less than the ceiling is
        # passed through unclamped.
        rows, budget, n_stats = 400, 1_000_000_000, len(ld_names(4))
        bytes_per_pair = (4 * rows + 50 * n_stats) * 3
        fit = budget // bytes_per_pair
        assert 1_000 < fit < 10_000_000
        assert estimate_ld_chunk_size(rows, n_stats, budget) == fit

    def test_floor_applied_on_tight_budget(self):
        # A budget too small for the floor still returns 1k pairs rather than a
        # handful, which would launch far too many kernels.
        assert estimate_ld_chunk_size(400, len(ld_names(4)), 1_000_000) == 1_000

    def test_ceiling_caps_large_budgets(self):
        assert estimate_ld_chunk_size(50, 3, 10**12) == 10_000_000

    def test_float_budget_gives_int_chunk(self):
        # A float budget must not leak a float chunk size.
        assert isinstance(estimate_ld_chunk_size(400, 105, 5.0e9), int)
