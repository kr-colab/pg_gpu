"""Edge-case coverage for windowed_analysis.py: zero-span and all-missing
matrices, the SNP window-count edge, the populations-without-sample_sets
warning, the empty-window callable name fallback, the less-common single-pop
stat outputs, multiallelic max_daf, the two-population exclude /
normalization paths, and Fay & Wu's H on a window with no variants.
"""
import warnings

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, divergence, diversity
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
        pos = np.full(n_var, 500, dtype=np.int64)      # one-base region
    else:
        pos = (np.arange(1, n_var + 1) * 100).astype(np.int64)
    return HaplotypeMatrix(hap, pos, sample_sets=sample_sets, **kw)


_TWO_POP = {"p1": [0, 1, 2, 3], "p2": [4, 5, 6, 7]}


# ── one-base region (chrom_start == chrom_end) -> one 1-bp window ───────
def test_one_base_region_single_pop_one_window():
    m = _matrix(n_var=5, same_position=True)
    df = windowed_analysis(m, window_size=100, statistics=["pi"])
    assert df[["start", "end", "n_variants"]].values.tolist() == [[500, 501, 5]]


def test_one_base_region_two_pop_one_window():
    m = _matrix(n_var=5, same_position=True, sample_sets=_TWO_POP)
    df = windowed_analysis(m, window_size=100, statistics=["dxy"],
                           populations=["p1", "p2"])
    assert df[["start", "end", "n_variants"]].values.tolist() == [[500, 501, 5]]


@pytest.mark.parametrize("stat", ["pi", "dxy"])
def test_n_total_sites_share_counts_both_bounds(stat):
    # A scatter window's share of n_total_sites divides by the region's
    # base count, chrom_end - chrom_start + 1. For a one-base region that
    # count is 1, not 0, and the window gets all of n_total_sites.
    m = _matrix(n_var=5, same_position=True, sample_sets=_TWO_POP,
                n_total_sites=1000)
    kw = {"populations": ["p1", "p2"]} if stat == "dxy" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        df = windowed_analysis(m, window_size=100, statistics=[stat], **kw)
    scalar = (divergence.dxy(m, "p1", "p2") if stat == "dxy"
              else diversity.pi(m))
    assert df[stat].iloc[0] == pytest.approx(scalar)


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

    # A custom statistic gets NaN on a window with no variant, uncalled.
    empty = m.get_subset(np.arange(0))
    res = StatisticsComputer(statistics=[my_stat]).compute(
        WindowData("1", 0, 100, 50, empty, 0, 0))
    assert "my_stat" in res and np.isnan(res["my_stat"])


# ── multiallelic max_daf ───────────────────────────────────────────────
def test_max_daf_multiallelic():
    # A single window over the whole matrix must reproduce the scalar max_daf,
    # which is the frequency of the most common derived allele per site --
    # a per-allele (not total non-ancestral) quantity on multiallelic sites.
    m = _matrix(multiallelic=True)
    df = windowed_analysis(m, window_size=100_000, statistics=["max_daf"])
    assert len(df) == 1
    assert df["max_daf"].iloc[0] == pytest.approx(float(diversity.max_daf(m)))


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
        with pytest.raises(ValueError, match="not found in sample_sets"):
            WindowedAnalyzer(window_size=600, statistics=["pi"],
                             populations=["p1"], progress_bar=False).compute(m)
    assert any("sample_sets" in str(w.message) for w in caught)


# ── two-population exclude-missing + n_total_sites normalization ────────
def test_two_pop_exclude_missing_drops_missing_sites():
    # missing_data='exclude' drops every site with a missing call: the windowed
    # dxy then differs from 'include' and equals dxy on the matrix with that
    # site already removed. The n_total_sites call drives the span-norm branch.
    rng = np.random.default_rng(0)
    hap = rng.integers(0, 2, size=(8, 12)).astype(np.int8)
    pos = (np.arange(1, 13) * 100).astype(np.int64)
    hap[0, 0] = -1                                    # missing in p1 at site 0
    m = HaplotypeMatrix(hap.copy(), pos, sample_sets=_TWO_POP,
                        n_total_sites=5000)
    complete = HaplotypeMatrix(hap[:, 1:].copy(), pos[1:], sample_sets=_TWO_POP)

    def dxy(mat, missing_data, **kw):
        return float(windowed_analysis(
            mat, window_size=100_000, statistics=["dxy"],
            populations=["p1", "p2"], missing_data=missing_data, **kw
        ).iloc[0]["dxy"])

    excl = dxy(m, "exclude", span_normalize=False)
    assert excl != dxy(m, "include", span_normalize=False)
    assert excl == pytest.approx(dxy(complete, "include", span_normalize=False))
    assert np.isfinite(dxy(m, "exclude"))             # n_total_sites normalized


def test_exclude_missing_all_sites_yields_empty_windows():
    # Every call missing -> exclude drops all sites, but the window grid is
    # still anchored, so each window is emitted with n_variants=0 rather than
    # vanishing. Each window had sites before 'exclude' removed them all, so
    # pi is undefined (NaN), not 0 -- 0 is reserved for a window with no
    # variant to begin with.
    hap = np.full((8, 6), -1, dtype=np.int8)
    pos = (np.arange(1, 7) * 100).astype(np.int64)
    m = HaplotypeMatrix(hap, pos)
    df = windowed_analysis(m, window_size=300, statistics=["pi"],
                           missing_data="exclude")
    assert not df.empty
    assert (df["n_variants"] == 0).all()
    assert df["pi"].isna().all()


def test_two_pop_exclude_missing_all_sites_yields_empty_windows():
    hap = np.full((8, 6), -1, dtype=np.int8)
    pos = (np.arange(1, 7) * 100).astype(np.int64)
    m = HaplotypeMatrix(hap, pos, sample_sets=_TWO_POP)
    df = windowed_analysis(m, window_size=300, statistics=["dxy"],
                           populations=["p1", "p2"], missing_data="exclude")
    assert not df.empty
    assert (df["n_variants"] == 0).all()
    assert df["dxy"].isna().all()


# ── scatter path: the less-common single-pop stat outputs ──────────────
def test_scatter_all_single_pop_stat_outputs():
    # Requesting the full scatter-single set exercises every per-stat output
    # branch (singletons, fay_wu_h, normalized_fay_wu_h, zeng_e, zeng_dh,
    # max_daf) that the common pi/theta_w tests skip. span_normalize=False
    # pins the core stats to their independent scalar estimators; the
    # normalized call then exercises the n_total_sites span branch.
    m = _matrix(n_var=30, seed=3, n_total_sites=8000)
    stats = ["pi", "theta_w", "theta_h", "theta_l", "segregating_sites",
             "singletons", "fay_wu_h", "normalized_fay_wu_h", "zeng_e",
             "zeng_dh", "max_daf", "tajimas_d"]
    r = windowed_analysis(m, window_size=100_000, statistics=stats,
                          span_normalize=False).iloc[0]
    for col in stats:
        assert col in r.index
    assert r["pi"] == pytest.approx(float(diversity.pi(m, span_normalize=False)))
    assert r["theta_w"] == pytest.approx(
        float(diversity.theta_w(m, span_normalize=False)))
    assert r["tajimas_d"] == pytest.approx(float(diversity.tajimas_d(m)))
    assert r["fay_wu_h"] == pytest.approx(float(diversity.fay_wus_h(m)))
    assert r["max_daf"] == pytest.approx(float(diversity.max_daf(m)))
    # One window covers the whole region, so its share of n_total_sites is
    # all of it.
    norm = windowed_analysis(
        m, window_size=100_000, statistics=["pi"]).iloc[0]["pi"]
    assert norm == pytest.approx(r["pi"] / 8000)


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



# ── Fay & Wu's H on a window with no variants ──────────────────────────
@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("statistics", [["fay_wu_h"], ["fay_wu_h", "zns"]])
def test_fay_wu_h_empty_window_matches_scalar(statistics, missing_data):
    # Variants only in [1, 1000]; the windows [1001, 2001) and [2001, 3000]
    # are empty. ["fay_wu_h"] runs the scatter engine, the zns pair the
    # fused. The one missing call makes 'exclude' drop a site.
    rng = np.random.default_rng(1)
    hap = rng.integers(0, 2, (8, 10)).astype(np.int8)
    hap[0, 3] = -1
    hm = HaplotypeMatrix(hap, np.arange(10) * 100 + 1, 1, 3000)
    df = windowed_analysis(hm, 1000, statistics=statistics,
                           missing_data=missing_data, span_normalize=False)
    for start, end, value in zip(df["start"], df["end"], df["fay_wu_h"]):
        sub = hm.get_subset_from_range(int(start), int(end))
        assert value == pytest.approx(
            diversity.fay_wus_h(sub, missing_data=missing_data))
