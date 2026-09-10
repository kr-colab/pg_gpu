"""Coverage for the chunked fused windowed engine.

`windowed_statistics_fused_chunked` splits the variant axis into memory-safe
chunks and accumulates each chunk's kernel output; it must reproduce the
single-shot `windowed_statistics_fused` result. Each comparison test forces a
small chunk size (so the accumulation loop runs many times) and checks the two
engines agree, or drives a validation / delegation branch that only the chunked
engine has. Both engines are called with `_win_starts` / `_win_stops`, the way
the public dispatcher invokes them and the way overlapping or sparse windows
must be expressed.
"""
import numpy as np
import pytest

import pg_gpu._memutil as _memutil
from pg_gpu import ld_statistics
from pg_gpu.windowed_analysis import (
    _bp_grid_bounds, _bp_window_grid, windowed_analysis,
    windowed_statistics_fused, windowed_statistics_fused_chunked,
)

from .conftest import simulate_hm

pytestmark = pytest.mark.filterwarnings(
    "ignore::pg_gpu._warnings.BiallelicOnlyWarning")

# Below the fixture's variant count and above any single window's, so the
# forced chunk size splits the matrix into several genuine chunks.
_CHUNK = 256


@pytest.fixture(scope="module")
def twopop_hm():
    hm = simulate_hm(n_samples=40, seq_length=100_000, seed=11)
    n = hm.num_haplotypes
    hm.sample_sets = {"p1": list(range(n // 2)), "p2": list(range(n // 2, n))}
    hm.transfer_to_gpu()
    assert hm.num_variants > _CHUNK        # the chunk loop must iterate
    return hm


def _grid(hm, window_size, step_size):
    """The window arrays the dispatcher builds for a bp grid."""
    pos = hm.positions
    pos = pos.get() if hasattr(pos, "get") else np.asarray(pos)
    cs, ce = _bp_grid_bounds(hm, pos)
    ws, we = _bp_window_grid(cs, ce, window_size, step_size)
    return {"bp_bins": np.concatenate([ws, [we[-1]]]),
            "_win_starts": ws, "_win_stops": we}


def _compare(hm, statistics, monkeypatch, window_size=10_000,
             step_size=10_000, **kwargs):
    common = dict(statistics=tuple(statistics),
                  **_grid(hm, window_size, step_size), **kwargs)
    single = windowed_statistics_fused(hm, **common)
    monkeypatch.setattr(_memutil, "estimate_fused_chunk_size",
                        lambda n, memory_fraction=0.35: _CHUNK)
    chunked = windowed_statistics_fused_chunked(hm, **common)
    for k in statistics:
        np.testing.assert_allclose(single[k], chunked[k], rtol=1e-10,
                                   atol=1e-14, equal_nan=True, err_msg=k)


# ── two-population: fst_hudson column + per_base=False dxy/da ───────────
def test_twopop_fst_hudson_and_perbase_false(twopop_hm, monkeypatch):
    _compare(twopop_hm, ["fst", "fst_hudson", "dxy", "da"], monkeypatch,
             per_base=False, pop1="p1", pop2="p2")


# ── single-population stats the existing chunked tests don't request ────
def test_single_pop_extra_stats(twopop_hm, monkeypatch):
    # theta_h / fay_wu_h use the ancestral-weighted accumulator; max_daf is
    # accumulated with element-wise max (not a sum) across chunks.
    _compare(twopop_hm, ["theta_h", "fay_wu_h", "singletons", "max_daf"],
             monkeypatch)


def test_single_pop_perbase_false(twopop_hm, monkeypatch):
    _compare(twopop_hm, ["pi", "theta_w", "theta_h"], monkeypatch,
             per_base=False)


# ── scatter/LD stats delegate to the single-shot engine ────────────────
def test_remaining_stats_delegated(twopop_hm, monkeypatch):
    # zns is not a chunked kernel output; the chunked engine forwards it to
    # windowed_statistics_fused. _compare shows the forwarded column matches
    # the single-shot engine (and pi still accumulates across chunks); the
    # whole-matrix window below independently pins the forwarded zns value to
    # the ld_statistics.zns scalar, so the delegation cannot forward a wrong
    # number undetected.
    _compare(twopop_hm, ["pi", "zns"], monkeypatch)
    pos = twopop_hm.positions
    pos = pos.get() if hasattr(pos, "get") else np.asarray(pos)
    whole = int(pos.max()) * 10
    r = windowed_analysis(twopop_hm, window_size=whole, step_size=whole,
                          statistics=["zns"]).iloc[0]
    np.testing.assert_allclose(r["zns"], float(ld_statistics.zns(twopop_hm)),
                               rtol=1e-9, atol=1e-12)


# ── sparse windows: chunks in a between-window gap overlap nothing ──────
def test_sparse_windows_empty_overlap(twopop_hm, monkeypatch):
    # step > window_size leaves variant gaps; chunks landing in a gap match
    # no window and take the empty-overlap continue (both the single- and
    # two-population loops, hence a single- and a two-pop statistic).
    _compare(twopop_hm, ["pi", "fst_hudson", "dxy"], monkeypatch,
             window_size=5_000, step_size=50_000, per_base=False,
             pop1="p1", pop2="p2")


# ── population validation (before the chunk loop) ──────────────────────
def test_chunked_requires_pop1_pop2(twopop_hm):
    grid = _grid(twopop_hm, 10_000, 10_000)
    with pytest.raises(ValueError, match="pop1 and pop2 required"):
        windowed_statistics_fused_chunked(
            twopop_hm, statistics=("fst_hudson",), **grid)


def test_chunked_unknown_population_raises(twopop_hm):
    grid = _grid(twopop_hm, 10_000, 10_000)
    with pytest.raises(ValueError, match="not found in sample_sets"):
        windowed_statistics_fused_chunked(
            twopop_hm, statistics=("fst_hudson",),
            pop1="nope", pop2="p2", **grid)
