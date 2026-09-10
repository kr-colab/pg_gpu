"""Coverage for the windowed LD and Garud-H paths of windowed_analysis.py.

Two oracle strategies, both non-tautological:

* Single window covering the whole matrix. A window wider than the sequence
  makes the windowed value the statistic of every variant at once, so it must
  equal the corresponding scalar estimator (`ld_statistics.zns/omega/mu_ld`,
  `selection.garud_h`, `divergence.fst_hudson/dxy`, `_compute_mean_r2`). This
  pins the fused windowed engine against the independently-tested scalar path.

* Multiple windows for Garud H. Non-overlapping windows drive the tile /
  reduceat assembly; overlapping windows drive the sliding prefix-sum path.
  Each window's H is compared to `selection.garud_h` on the bp-sliced
  submatrix, exercising both per-window assembly routes.
"""
import cupy as cp
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, divergence, ld_statistics, selection
from pg_gpu.windowed_analysis import (
    WindowedAnalyzer, _compute_mean_r2, windowed_analysis,
)

from .conftest import simulate_hm

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


def _positions(m):
    pos = m.positions
    return pos.get() if hasattr(pos, "get") else np.asarray(pos)


def _single(m, statistics, **kwargs):
    """The one row of a single whole-matrix window."""
    df = windowed_analysis(m, window_size=_WHOLE, step_size=_WHOLE,
                           statistics=statistics, **kwargs)
    assert len(df) == 1
    return df.iloc[0]


def _garud_ref(m, start, end):
    """`selection.garud_h` on the variants a bp window covers."""
    pos = _positions(m)
    mask = (pos >= start) & (pos < end)
    sub = HaplotypeMatrix(m.haplotypes[:, cp.asarray(mask)], pos[mask])
    return selection.garud_h(sub)


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


def test_windowed_ld_decay_is_finite(hm):
    analyzer = WindowedAnalyzer(
        window_type="bp", window_size=_WHOLE, step_size=_WHOLE,
        statistics=["ld_decay"],
        custom_stat_kwargs={"ld_decay": {"max_distance": 20000}})
    df = analyzer.compute(hm)
    assert np.isfinite(float(df.iloc[0]["ld_decay"]))


def test_windowed_distribution_moments_finite(hm):
    r = _single(hm, ["dist_var", "dist_skew", "dist_kurt"])
    assert all(np.isfinite(float(r[c]))
               for c in ["dist_var", "dist_skew", "dist_kurt"])


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


# ── multi-window Garud H: tile (reduceat) and sliding (prefix-sum) ──────
@pytest.mark.parametrize("window,step", [
    (10_000, 10_000),   # non-overlapping -> tile / reduceat assembly
    (10_000, 5_000),    # overlapping     -> sliding prefix-sum assembly
], ids=["tile", "sliding"])
def test_garud_multiwindow_matches_per_window_scalar(hm, window, step):
    df = windowed_analysis(hm, window_size=window, step_size=step,
                           statistics=_GARUD)
    assert len(df) > 1
    compared = 0
    for _, row in df.iterrows():
        if row["n_variants"] == 0:
            continue
        ref = _garud_ref(hm, row["start"], row["end"])
        np.testing.assert_allclose([row[s] for s in _GARUD],
                                   [float(x) for x in ref],
                                   rtol=1e-9, atol=1e-9)
        compared += 1
    assert compared > 1
