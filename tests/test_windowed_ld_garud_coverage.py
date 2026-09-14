"""Coverage for the windowed LD and Garud-H paths of windowed_analysis.py.

Two oracle strategies, both non-tautological:

* Single window covering the whole matrix. A window wider than the sequence
  makes the windowed value the statistic of every variant at once, so it must
  equal the corresponding scalar estimator (`ld_statistics.zns/omega/mu_ld`,
  `selection.garud_h`, `divergence.fst_hudson/dxy`, `_compute_mean_r2`). This
  pins the fused windowed engine against the independently-tested scalar path.

* Multiple windows for Garud H. Non-overlapping and overlapping grids both
  hash each window from its own variants. Each window's H is compared to
  `selection.garud_h` on the bp-sliced submatrix, and a window with no
  variants must report a single haplotype.
"""
import cupy as cp
import numpy as np
import pytest

import pg_gpu._memutil as _memutil
from pg_gpu import (
    HaplotypeMatrix, distance_stats, divergence, ld_statistics, selection,
)
from pg_gpu.windowed_analysis import (
    WindowedAnalyzer, _compute_mean_r2, windowed_analysis,
)

from .conftest import founder_haplotypes, garud_reference, simulate_hm

# The fused windowed engine drops multiallelic sites (as does every scalar
# estimator here), so the oracle equalities hold; silence the notice.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pg_gpu._warnings.BiallelicOnlyWarning")

# Wider than the fixture's sequence length, so a single window covers every
# variant.
_WHOLE = 1_000_000
_GARUD = ["garud_h1", "garud_h12", "garud_h123", "garud_h2h1"]


@pytest.fixture(scope="module")
def hm():
    """LD-rich single-population matrix (recombination gives real LD)."""
    return simulate_hm(n_samples=30, seq_length=50_000, seed=7)


@pytest.fixture(scope="module")
def hm_twopop():
    m = simulate_hm(n_samples=30, seq_length=50_000, seed=7)
    n = m.num_haplotypes
    m.sample_sets = {"p1": list(range(n // 2)), "p2": list(range(n // 2, n))}
    return m


def _single(m, statistics, **kwargs):
    """The one row of a single whole-matrix window."""
    df = windowed_analysis(m, window_size=_WHOLE, step_size=_WHOLE,
                           statistics=statistics, **kwargs)
    assert len(df) == 1
    return df.iloc[0]


def _garud_ref(m, start, end):
    """`selection.garud_h` on the variants a right-open bp window covers."""
    return selection.garud_h(m.get_subset_from_range(int(start), int(end)))


# ── whole-matrix window == scalar LD estimators ────────────────────────
def test_windowed_zns_omega_mu_ld_match_scalar(hm):
    r = _single(hm, ["zns", "omega", "mu_ld"])
    np.testing.assert_allclose(
        [r["zns"], r["omega"], r["mu_ld"]],
        [float(ld_statistics.zns(hm)), float(ld_statistics.omega(hm)),
         float(ld_statistics.mu_ld(hm))], rtol=1e-9, atol=1e-12)


def test_windowed_mean_r2_matches_scalar(hm):
    analyzer = WindowedAnalyzer(
        window_type="bp", window_size=_WHOLE, step_size=_WHOLE,
        statistics=["mean_r2"],
        custom_stat_kwargs={"mean_r2": {"max_dist": 20000}})
    df = analyzer.compute(hm)
    assert len(df) == 1
    np.testing.assert_allclose(df.iloc[0]["mean_r2"],
                               float(_compute_mean_r2(hm, 20000)),
                               rtol=1e-9, atol=1e-12)


def test_windowed_ld_decay_matches_mean_r2(hm):
    # ld_decay dispatches to _compute_mean_r2 with the given max_distance, so
    # the whole-matrix window equals that scalar at the same distance.
    analyzer = WindowedAnalyzer(
        window_type="bp", window_size=_WHOLE, step_size=_WHOLE,
        statistics=["ld_decay"],
        custom_stat_kwargs={"ld_decay": {"max_distance": 20000}})
    df = analyzer.compute(hm)
    np.testing.assert_allclose(df.iloc[0]["ld_decay"],
                               float(_compute_mean_r2(hm, 20000)),
                               rtol=1e-9, atol=1e-12)


def test_windowed_distribution_moments_match_scalar(hm):
    r = _single(hm, ["dist_var", "dist_skew", "dist_kurt"])
    np.testing.assert_allclose(
        [r["dist_var"], r["dist_skew"], r["dist_kurt"]],
        [float(distance_stats.dist_var(hm)),
         float(distance_stats.dist_skew(hm)),
         float(distance_stats.dist_kurt(hm))], rtol=1e-9, atol=1e-12)


# ── whole-matrix window == scalar Garud H ──────────────────────────────
def test_windowed_garud_matches_scalar(hm):
    r = _single(hm, _GARUD)
    np.testing.assert_allclose([r[s] for s in _GARUD],
                               [float(x) for x in selection.garud_h(hm)],
                               rtol=1e-9, atol=1e-12)


# ── whole-matrix window == scalar two-population divergence ─────────────
def test_windowed_two_pop_fst_dxy_match_scalar(hm_twopop):
    # zns forces the fused path, which emits per-variant fst / dxy components.
    r = _single(hm_twopop, ["fst_hudson", "dxy", "zns"],
                populations=["p1", "p2"])
    np.testing.assert_allclose(
        [r["fst_hudson"], r["dxy"]],
        [float(divergence.fst_hudson(hm_twopop, "p1", "p2")),
         float(divergence.dxy(hm_twopop, "p1", "p2"))],
        rtol=1e-9, atol=1e-12)


# ── multi-window Garud H: tile and sliding grids ────────────────────────
@pytest.mark.parametrize("window,step", [
    (10_000, 10_000),   # non-overlapping
    (10_000, 5_000),    # overlapping
], ids=["tile", "sliding"])
def test_garud_multiwindow_matches_per_window_scalar(hm, window, step):
    df = windowed_analysis(hm, window_size=window, step_size=step,
                           statistics=_GARUD)
    assert len(df) > 1
    compared = 0
    for _, row in df.iterrows():
        ref = _garud_ref(hm, row["start"], row["end"])
        np.testing.assert_allclose([row[s] for s in _GARUD],
                                   [float(x) for x in ref],
                                   rtol=1e-9, atol=1e-9)
        compared += 1
    assert compared > 1


_GARUD_COUNT = _GARUD + ["haplotype_count"]


def _assert_rows_match_reference(df, hap, pos):
    """Every window's statistics equal the exact unique-row reference.

    ``hap`` may live on either device; only the window's columns move to
    the host.
    """
    for _, row in df.iterrows():
        lo, hi = np.searchsorted(pos, [row["start"], row["end"]])
        rows = hap[:, lo:hi]
        ref = garud_reference(rows.get() if hasattr(rows, "get") else rows)
        np.testing.assert_allclose([row[s] for s in _GARUD], ref[:4],
                                   rtol=1e-12)
        assert row["haplotype_count"] == ref[4]


@pytest.mark.parametrize("pos,bounds,window,step,n_empty", [
    ([10, 210], (0, 300), 100, 100, 1),    # variant-free middle tile
    ([10, 20], (0, 400), 100, 100, 3),     # grid runs on past the variants
    ([10, 20], (0, 400), 100, 50, 7),      # sliding grid, variant-free overlaps
    ([10, 20], (200, 400), 100, 100, 2),   # every window empty
])
def test_garud_windows_without_variants(pos, bounds, window, step, n_empty):
    # A window with no variants has no differentiating sites: one distinct
    # haplotype, H1 = H12 = H123 = 1, H2/H1 = 0.
    hap = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int8)
    pos = np.array(pos, dtype=np.int64)
    m = HaplotypeMatrix(hap, pos, *bounds)
    df = windowed_analysis(m, window_size=window, step_size=step,
                           statistics=_GARUD_COUNT)
    assert (df["n_variants"] == 0).sum() == n_empty
    _assert_rows_match_reference(df, hap, pos)


@pytest.mark.parametrize("window,step", [(10_000, 10_000), (10_000, 5_000)],
                         ids=["tile", "sliding"])
def test_garud_window_batches_are_independent(hm, window, step, monkeypatch):
    single = windowed_analysis(hm, window_size=window, step_size=step,
                               statistics=_GARUD_COUNT)
    monkeypatch.setattr(_memutil, "estimate_garud_window_batch",
                        lambda n_hap, memory_fraction=0.3: 3)
    batched = windowed_analysis(hm, window_size=window, step_size=step,
                                statistics=_GARUD_COUNT)
    assert len(single) > 3
    for s in _GARUD_COUNT:
        np.testing.assert_array_equal(single[s].to_numpy(),
                                      batched[s].to_numpy())


def test_garud_many_haplotypes_exact():
    # 2,500 haplotypes drawn from 50 patterns: more haplotypes than a thread
    # block could sort in shared memory, so this pins the global sort.
    n_hap, n_var = 2500, 3000
    hap = founder_haplotypes(np.random.default_rng(11), n_hap, n_var, 50)
    pos = np.arange(1, n_var + 1, dtype=np.int64) * 10
    m = HaplotypeMatrix(hap, pos, 0, n_var * 10 + 10)
    df = windowed_analysis(m, window_size=5_000, step_size=5_000,
                           statistics=_GARUD_COUNT)
    assert len(df) >= 6
    _assert_rows_match_reference(df, hap, pos)


def test_garud_only_request_matches_combined_request(hm):
    # A Garud-only request skips the fused engine's transposed copy of the
    # matrix; the Garud columns must not depend on the other statistics
    # requested alongside them.
    alone = windowed_analysis(hm, window_size=10_000, step_size=10_000,
                              statistics=_GARUD_COUNT)
    combined = windowed_analysis(hm, window_size=10_000, step_size=10_000,
                                 statistics=_GARUD_COUNT + ["pi"])
    for s in _GARUD_COUNT:
        np.testing.assert_array_equal(alone[s].to_numpy(),
                                      combined[s].to_numpy())


@pytest.mark.slow
def test_garud_cohort_scale_stress():
    # 3,000 haplotypes x 2,000,000 variants (6 GB of int8, built on the GPU)
    # on a 20 kb tile grid and a 100 kb / 10 kb sliding grid; spot-check
    # windows against exact unique rows.
    n_hap, n_var = 3000, 2_000_000
    rng = cp.random.default_rng(3)
    founders = (rng.random((200, n_var), dtype=cp.float32) < 0.5).astype(cp.int8)
    hap = founders[rng.integers(0, 200, size=n_hap)]
    del founders
    pos = np.arange(1, n_var + 1, dtype=np.int64) * 10
    m = HaplotypeMatrix(hap, pos, 0, 20_000_000)
    for window, step in [(20_000, 20_000), (100_000, 10_000)]:
        df = windowed_analysis(m, window_size=window, step_size=step,
                               statistics=_GARUD_COUNT)
        _assert_rows_match_reference(df.iloc[[0, len(df) // 2, len(df) - 1]],
                                     hap, pos)
