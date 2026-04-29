"""Tests for ``engine='auto'`` in ``local_pca``.

The auto-selector picks between ``'dense-eigh'`` (faster but holds the whole
Gram stack on the GPU) and ``'streaming-dense'`` (bounded memory, ~2x slower
per window) based on the estimated Gram-stack peak vs free GPU memory.
"""

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import (
    _estimate_n_windows,
    _pick_dense_engine,
    local_pca,
)
from pg_gpu.windowed_analysis import WindowParams


@pytest.fixture
def small_hm():
    rng = np.random.default_rng(0)
    n_hap = 20
    n_var = 600
    hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return HaplotypeMatrix(hap, pos, 0, n_var * 1000)


# ---------------------------------------------------------------------------
# _estimate_n_windows
# ---------------------------------------------------------------------------


class TestEstimateNWindows:

    def test_snp_windows_exact(self, small_hm):
        # 600 variants, window=100, step=100 -> 6 windows.
        params = WindowParams(window_type='snp', window_size=100, step_size=100)
        assert _estimate_n_windows(small_hm, params) == 6

    def test_snp_windows_with_overlap(self, small_hm):
        # 600 variants, window=100, step=50 -> floor((600-100)/50)+1 = 11.
        params = WindowParams(window_type='snp', window_size=100, step_size=50)
        assert _estimate_n_windows(small_hm, params) == 11

    def test_snp_window_larger_than_data(self, small_hm):
        params = WindowParams(window_type='snp', window_size=10_000, step_size=10_000)
        assert _estimate_n_windows(small_hm, params) == 0

    def test_bp_windows_upper_bound(self, small_hm):
        # positions span 0..599_000 with step_size=100_000 -> ceil-ish bound.
        params = WindowParams(window_type='bp', window_size=100_000, step_size=100_000)
        n = _estimate_n_windows(small_hm, params)
        # The bound is span // step + 1 = 599_000 // 100_000 + 1 = 6.
        assert n == 6


# ---------------------------------------------------------------------------
# _pick_dense_engine
# ---------------------------------------------------------------------------


class TestPickDenseEngine:
    """Pure-function tests with synthetic free_bytes — no real GPU memory state."""

    def test_picks_dense_eigh_when_stack_fits(self, small_hm):
        params = WindowParams(window_type='snp', window_size=100, step_size=100)
        # 6 windows * 20^2 * 8 = 19_200 bytes. peak = 3x = 57_600. Plenty of room.
        engine = _pick_dense_engine(small_hm, params, free_bytes=10**9)
        assert engine == 'dense-eigh'

    def test_picks_streaming_when_stack_exceeds_budget(self, small_hm):
        params = WindowParams(window_type='snp', window_size=100, step_size=100)
        # Force tiny "free" memory so the budget can't accommodate even a tiny stack.
        engine = _pick_dense_engine(small_hm, params, free_bytes=10_000)
        assert engine == 'streaming-dense'

    def test_picks_dense_eigh_for_zero_window_estimate(self, small_hm):
        # Window larger than the data -> 0 windows. dense-eigh handles this fine.
        params = WindowParams(window_type='snp', window_size=10_000, step_size=10_000)
        engine = _pick_dense_engine(small_hm, params, free_bytes=10_000)
        assert engine == 'dense-eigh'

    def test_threshold_boundary(self, small_hm):
        # Construct free_bytes so that peak == budget_fraction * free_bytes
        # exactly; the strict-less-than comparison should pick streaming.
        params = WindowParams(window_type='snp', window_size=100, step_size=100)
        n_windows = _estimate_n_windows(small_hm, params)
        peak = 3 * n_windows * small_hm.num_haplotypes ** 2 * 8
        # peak < 0.5 * free  iff  free > 2 * peak. Set free = 2 * peak: NOT less than.
        free = 2 * peak
        assert _pick_dense_engine(small_hm, params, free_bytes=free) == 'streaming-dense'
        # Bump free up by a margin: now strictly less than -> dense-eigh.
        assert _pick_dense_engine(small_hm, params, free_bytes=free + 1) == 'dense-eigh'


# ---------------------------------------------------------------------------
# Integration: engine='auto' produces the same numerics as the engine it picks.
# ---------------------------------------------------------------------------


class TestAutoEngineIntegration:

    def test_auto_matches_dense_eigh_on_small_workload(self, small_hm):
        # Small workload always picks dense-eigh on any reasonable GPU.
        # Auto must produce bit-identical output to an explicit dense-eigh call.
        explicit = local_pca(small_hm, window_size=100, window_type='snp', k=2,
                              engine='dense-eigh')
        auto = local_pca(small_hm, window_size=100, window_type='snp', k=2,
                          engine='auto')
        np.testing.assert_array_equal(auto.eigvals, explicit.eigvals)
        np.testing.assert_array_equal(auto.sumsq, explicit.sumsq)

    def test_auto_rejects_unknown_engine(self, small_hm):
        with pytest.raises(ValueError, match="engine must be one of"):
            local_pca(small_hm, window_size=100, window_type='snp', k=2,
                       engine='bogus')

    def test_auto_is_a_valid_engine(self, small_hm):
        # Should not raise.
        local_pca(small_hm, window_size=100, window_type='snp', k=2, engine='auto')
