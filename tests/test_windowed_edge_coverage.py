"""Edge-case coverage for windowed_analysis.py: zero-span and all-missing
matrices, the SNP window-count edge, the populations-without-sample_sets
warning, the empty-window callable name fallback, the less-common single-pop
stat outputs, multiallelic max_daf, and the two-population exclude /
normalization paths.
"""
import warnings

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import (
    StatisticsComputer, WindowData, WindowedAnalyzer, WindowIterator,
    WindowParams, windowed_analysis,
)


def _matrix(n_hap=8, n_var=12, seed=0, multiallelic=False, sample_sets=None,
            same_position=False, **kw):
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, size=(n_hap, n_var)).astype(np.int8)
    if multiallelic:
        hap[0, 0] = 2
    if same_position:
        pos = np.full(n_var, 500, dtype=np.int64)      # zero-span
    else:
        pos = (np.arange(1, n_var + 1) * 100).astype(np.int64)
    return HaplotypeMatrix(hap, pos, sample_sets=sample_sets, **kw)


_TWO_POP = {"p1": [0, 1, 2, 3], "p2": [4, 5, 6, 7]}


# ── zero-span (chrom_end <= chrom_start) -> empty DataFrame ─────────────
def test_zero_span_single_pop_returns_empty():
    m = _matrix(n_var=5, same_position=True)
    assert windowed_analysis(m, window_size=100, statistics=["pi"]).empty


def test_zero_span_two_pop_returns_empty():
    m = _matrix(n_var=5, same_position=True, sample_sets=_TWO_POP)
    df = windowed_analysis(m, window_size=100, statistics=["dxy"],
                           populations=["p1", "p2"])
    assert df.empty


# ── WindowIterator.count_windows edges ─────────────────────────────────
def test_count_windows_snp_fewer_variants_than_window():
    # n_variants (12) <= window_size (50) -> a single window.
    m = _matrix(n_var=12)
    it = WindowIterator(m, WindowParams(window_type="snp", window_size=50,
                                        step_size=25))
    assert it.count_windows() == 1


def test_count_windows_regions():
    m = _matrix(n_var=12)
    it = WindowIterator(m, WindowParams(window_type="regions", window_size=0,
                                        step_size=0,
                                        regions=[("1", 0, 600), ("1", 600, 1200)]))
    assert it.count_windows() == 2


# ── StatisticsComputer empty-window callable name fallback ─────────────
def test_statistics_computer_empty_window_callable_name_fallback():
    m = _matrix()

    def my_stat(window):
        return 1.0

    res = StatisticsComputer(statistics=[my_stat]).compute(
        WindowData("1", 0, 100, 50, m, 0, 0))
    assert "my_stat" in res and np.isnan(res["my_stat"])


# ── multiallelic max_daf ───────────────────────────────────────────────
def test_max_daf_multiallelic():
    m = _matrix(multiallelic=True)
    df = windowed_analysis(m, window_size=600, statistics=["max_daf"])
    assert "max_daf" in df.columns
    assert np.all(df["max_daf"].to_numpy() >= 0)


def test_max_daf_no_derived_alleles():
    # An all-ancestral (monomorphic) matrix has no derived-allele column, so
    # max_daf takes the empty-derived branch and returns 0 for every window.
    m = HaplotypeMatrix(np.zeros((8, 6), dtype=np.int8),
                        (np.arange(1, 7) * 100).astype(np.int64))
    df = windowed_analysis(m, window_size=300, statistics=["max_daf"])
    assert np.all(df["max_daf"].to_numpy() == 0)


# ── populations set but matrix has no sample_sets -> warns (then raises) ─
def test_populations_without_sample_sets_warns():
    m = _matrix()
    m.sample_sets = {}  # empty sample_sets sticks; the default is {'all': ...}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="not found"):
            WindowedAnalyzer(window_size=600, statistics=["pi"],
                             populations=["p1"], progress_bar=False).compute(m)
    assert any("sample_sets" in str(w.message) for w in caught)


# ── two-population exclude-missing + n_total_sites normalization ────────
def test_two_pop_exclude_missing_with_n_total_sites():
    m = _matrix(n_var=12, sample_sets=_TWO_POP, n_total_sites=5000)
    hap = m.haplotypes
    import cupy as cp
    hap = cp.asnumpy(hap) if isinstance(hap, cp.ndarray) else np.asarray(hap)
    hap[0, 0] = -1  # a missing call -> exclude drops that site
    m2 = HaplotypeMatrix(hap, np.asarray(m.positions.get() if hasattr(m.positions, "get")
                                         else m.positions),
                         sample_sets=_TWO_POP, n_total_sites=5000)
    df = windowed_analysis(m2, window_size=600, statistics=["dxy"],
                           populations=["p1", "p2"], missing_data="exclude")
    assert "dxy_p1_p2" in df.columns or "dxy" in df.columns
    assert len(df) >= 1


def test_exclude_missing_all_sites_returns_empty():
    # Every call missing -> exclude drops all sites -> empty DataFrame.
    hap = np.full((8, 6), -1, dtype=np.int8)
    pos = (np.arange(1, 7) * 100).astype(np.int64)
    m = HaplotypeMatrix(hap, pos)
    assert windowed_analysis(m, window_size=300, statistics=["pi"],
                             missing_data="exclude").empty


def test_two_pop_exclude_missing_all_sites_returns_empty():
    hap = np.full((8, 6), -1, dtype=np.int8)
    pos = (np.arange(1, 7) * 100).astype(np.int64)
    m = HaplotypeMatrix(hap, pos, sample_sets=_TWO_POP)
    df = windowed_analysis(m, window_size=300, statistics=["dxy"],
                           populations=["p1", "p2"], missing_data="exclude")
    assert df.empty


# ── scatter path: the less-common single-pop stat outputs ──────────────
def test_scatter_all_single_pop_stat_outputs():
    # Requesting the full scatter-single set exercises the per-stat output
    # branches (singletons, fay_wu_h, normalized_fay_wu_h, zeng_e, zeng_dh,
    # max_daf) that the common pi/theta_w tests skip.
    # n_total_sites drives the proportional span-normalization branch.
    m = _matrix(n_var=30, seed=3, n_total_sites=8000)
    stats = ["pi", "theta_w", "theta_h", "theta_l", "segregating_sites",
             "singletons", "fay_wu_h", "normalized_fay_wu_h", "zeng_e",
             "zeng_dh", "max_daf", "tajimas_d"]
    df = windowed_analysis(m, window_size=1500, statistics=stats)
    for col in stats:
        assert col in df.columns


# ── fused-path edges: non-overlapping windows + fst_wc ─────────────────
def test_fused_non_overlapping_theta_fay_mu():
    # Mixing mu_var forces the fused CUDA path; step == window exercises the
    # non-overlapping-window branches for theta_h / fay_wu_h / mu_var.
    m = _matrix(n_var=20)
    df = windowed_analysis(m, window_size=1000, step_size=1000,
                           statistics=["theta_h", "fay_wu_h", "mu_var"])
    for col in ("theta_h", "fay_wu_h", "mu_var"):
        assert col in df.columns
    assert len(df) >= 2


def test_fused_two_pop_fst_wc():
    m = _matrix(n_var=20, sample_sets=_TWO_POP)
    df = windowed_analysis(m, window_size=1000, step_size=1000,
                           statistics=["fst_wc"], populations=["p1", "p2"])
    assert any("fst_wc" in c for c in df.columns)
    assert len(df) >= 2
