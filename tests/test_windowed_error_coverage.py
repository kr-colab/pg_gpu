"""Coverage for windowed_analysis error and device-handling branches:
the unknown-statistic, unknown-population, unknown-window-type,
unknown-keyword and missing-regions raises, and the fused engine's CPU->GPU
transfer.
"""
import numpy as np
import pandas as pd
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import (
    StatisticsComputer, WindowData, WindowIterator, WindowParams,
    windowed_analysis, windowed_statistics_fused,
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
    sc = StatisticsComputer(["pi"], populations=["nope"])
    window = WindowData("1", 0, 100, 50, m, m.num_variants, 0)
    with pytest.raises(ValueError, match="not found in sample_sets"):
        sc.compute(window)


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


# ── forward-looking: window_type/statistic combinations deferred to later ──
# Each case below raises NotImplementedError today and marks work the issue
# leaves for the future. When one is implemented its dispatch guard goes away
# and this test fails -- the signal to drop that case here and from the guard.
#
# window_type='regions' no longer defers any statistic: the scatter engine
# serves the scatter-only theta estimators and neutrality tests over region
# windows, and the fused engine's per-site range stats (mean_nsl, daf_hist,
# mu_sfs, het_expected) reduce each region's variant range, so all now have a
# regions path (see TestRegionsScatterStats).

# snp windows route to the per-window fallback, so stats that aren't
# implemented there still raise.
_SNP_UNSUPPORTED = ["theta_l", "zeng_e", "theta_h", "max_daf", "mean_nsl",
                    "zns"]


@pytest.mark.parametrize("window_type,stat",
                         [("snp", s) for s in _SNP_UNSUPPORTED])
def test_deferred_window_type_stats_raise(window_type, stat):
    m = _matrix(n_var=40)
    regions = (pd.DataFrame({"start": [100], "end": [3000]})
               if window_type == "regions" else None)
    with pytest.raises(NotImplementedError):
        windowed_analysis(m, window_size=10, window_type=window_type,
                          statistics=[stat], regions=regions)


# ── unknown keyword arguments ──────────────────────────────────────────
# Only the fallback and the local-PCA dispatch read **kwargs, so the check
# runs before the engine is chosen; each request below goes to a different
# engine (scatter, fused, fallback).
@pytest.mark.parametrize("statistics", [["pi"], ["pi", "mu_var"], ["n_variants"]])
def test_unknown_kwarg_raises_on_every_engine(statistics):
    m = _matrix(sample_sets={"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]})
    with pytest.raises(TypeError, match="Did you mean 'populations'"):
        windowed_analysis(m, window_size=400, statistics=statistics,
                          popualtions=["a"])


def test_population_without_local_pca_raises():
    m = _matrix(sample_sets={"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]})
    with pytest.raises(TypeError, match="apply only to the 'local_pca'"):
        windowed_analysis(m, window_size=400, statistics=["pi"],
                          population="a")


def test_local_pca_kwargs_accepted_with_local_pca():
    m = _matrix(n_var=40, sample_sets={"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]})
    result = windowed_analysis(m, window_size=1000, statistics=["local_pca"],
                               population="a", k=2)
    assert result.windows is not None


def test_fallback_kwargs_accepted_on_fast_engine():
    # Names that only the fallback uses are valid on every request.
    m = _matrix()
    df = windowed_analysis(m, window_size=400, statistics=["pi"],
                           progress_bar=False, ld_bins=[0, 100],
                           custom_stat_kwargs={})
    assert len(df) > 0
