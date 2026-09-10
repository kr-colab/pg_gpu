"""Coverage for windowed_analysis error and device-handling branches:
the unknown-statistic, unknown-population, unknown-window-type, and
missing-regions raises, and the fused engine's CPU->GPU transfer.
"""
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import (
    StatisticsComputer, WindowIterator, WindowParams, windowed_analysis,
    windowed_statistics_fused,
)


def _matrix(n_hap=8, n_var=12, seed=0, sample_sets=None):
    """A small biallelic matrix, left on the CPU."""
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, size=(n_hap, n_var)).astype(np.int8)
    pos = (np.arange(1, n_var + 1) * 100).astype(np.int64)
    return HaplotypeMatrix(hap, pos, sample_sets=sample_sets)


# ── unknown statistic ──────────────────────────────────────────────────
def test_unknown_statistic_raises_in_computer():
    with pytest.raises(ValueError, match="Unknown statistic"):
        StatisticsComputer(["not_a_real_stat"])


def test_unknown_statistic_raises_through_public_api():
    with pytest.raises(ValueError, match="Unknown statistic"):
        windowed_analysis(_matrix(), window_size=100,
                          statistics=["not_a_real_stat"])


# ── unknown population ─────────────────────────────────────────────────
def test_unknown_population_raises():
    m = _matrix(sample_sets={"p1": [0, 1, 2, 3]})
    sc = StatisticsComputer(["pi"])
    with pytest.raises(ValueError, match="not found in sample_sets"):
        sc._get_population_matrix(m, "nope")


# ── unknown window type ────────────────────────────────────────────────
def test_unknown_window_type_raises():
    it = WindowIterator(_matrix(), WindowParams(
        window_type="bogus", window_size=100, step_size=50))
    with pytest.raises(ValueError, match="Unknown window type"):
        iter(it)


# ── region windows without regions ─────────────────────────────────────
def test_region_windows_require_regions():
    it = WindowIterator(_matrix(), WindowParams(
        window_type="regions", window_size=0, step_size=0, regions=None))
    # _iter_region_windows is a generator, so the raise fires on consumption.
    with pytest.raises(ValueError, match="Regions must be provided"):
        list(it)


# ── fused engine transfers a CPU matrix to the GPU ─────────────────────
def test_fused_transfers_cpu_matrix():
    # population= also drives the single-population-subset branch.
    m = _matrix(n_var=12, sample_sets={"p1": [0, 1, 2, 3]})
    assert m.device == "CPU"
    bp_bins = np.array([0.0, 1300.0])
    result = windowed_statistics_fused(m, bp_bins=bp_bins,
                                       statistics=("pi",), population="p1")
    assert m.device == "GPU"
    assert np.isfinite(result["pi"][0])
