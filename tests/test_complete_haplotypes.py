"""The complete-haplotype count behind Garud's H and haplotype_count.

Haplotype-identity statistics take their groups from the haplotypes with no
missing call in a window. windowed_analysis reports that count as
``n_complete_haplotypes``, warns when many windows have few of them, and can
set those windows to NaN through ``min_complete_haplotypes``.
"""
import warnings

import cupy as cp
import numpy as np
import pytest

from pg_gpu import FewCompleteHaplotypesWarning, HaplotypeMatrix, selection
from pg_gpu.windowed_analysis import windowed_analysis, windowed_statistics_fused_chunked

from .conftest import simulate_hm

GARUD = ["garud_h1", "garud_h12", "garud_h123", "garud_h2h1", "haplotype_count"]
SPACING, WIN_SITES = 10, 100
WIN = WIN_SITES * SPACING


def _hap(n, rate, seed=0, n_var=400):
    rng = np.random.default_rng(seed)
    hap = (rng.random((n, n_var)) < 0.3).astype(np.int8)
    hap[rng.random(hap.shape) < rate] = -1
    return hap


def _hm(hap, sample_sets=None):
    n_var = hap.shape[1]
    return HaplotypeMatrix(hap.copy(), np.arange(n_var, dtype=np.int64) * SPACING,
                           0, n_var * SPACING, sample_sets=sample_sets)


def _ref_complete(hap):
    n, m = hap.shape
    return (~(hap < 0).reshape(n, m // WIN_SITES, WIN_SITES).any(axis=2)).sum(axis=0)


def _few_complete(record):
    return [w for w in record if issubclass(w.category, FewCompleteHaplotypesWarning)]


def _run(*args, **kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = windowed_analysis(*args, **kwargs)
    return out, _few_complete(record)


# ---------------------------------------------------------------------------
# The column, on every engine
# ---------------------------------------------------------------------------

def test_fused_column_counts_rows_with_no_missing_call():
    hap = _hap(200, 0.01)
    df, _ = _run(_hm(hap), window_size=WIN, statistics=["garud_h12"])
    np.testing.assert_array_equal(df["n_complete_haplotypes"], _ref_complete(hap))


def test_chunked_engine_reports_the_column():
    hap = _hap(200, 0.01)
    edges = np.arange(0, hap.shape[1] * SPACING + 1, WIN)
    res = windowed_statistics_fused_chunked(_hm(hap), bp_bins=edges,
                                            statistics=("pi", "garud_h1"))
    np.testing.assert_array_equal(res["n_complete_haplotypes"], _ref_complete(hap))


def test_exclude_counts_every_row():
    hap = _hap(200, 0.01)
    df, few = _run(_hm(hap), window_size=WIN, statistics=GARUD, missing_data="exclude")
    np.testing.assert_array_equal(df["n_complete_haplotypes"], 200)
    assert not few


def test_fallback_names_the_column_per_population():
    # 'pi' with a Garud statistic under 'exclude' takes the per-window
    # fallback, which suffixes per-population columns.
    hap = _hap(60, 0.01)
    pops = {"p1": list(range(40)), "p2": list(range(40, 60))}
    df, _ = _run(_hm(hap, pops), window_size=WIN, statistics=["pi", "garud_h1"],
                 populations=["p1", "p2"], missing_data="exclude")
    np.testing.assert_array_equal(df["n_complete_haplotypes_p1"], 40)
    np.testing.assert_array_equal(df["n_complete_haplotypes_p2"], 20)


def test_fused_column_follows_the_population():
    hap = _hap(60, 0.01)
    pops = {"p1": list(range(40)), "p2": list(range(40, 60))}
    df, _ = _run(_hm(hap, pops), window_size=WIN, statistics=["garud_h1"], populations=["p1"])
    np.testing.assert_array_equal(df["n_complete_haplotypes"], _ref_complete(hap[:40]))


def test_population_given_as_rows():
    # A matrix with no sample_sets; the row count behind the warning comes
    # from the row list.
    hap = _hap(60, 0.03)
    rows = list(range(30))
    df, few = _run(_hm(hap), window_size=WIN, statistics=["garud_h1"], populations=[rows])
    np.testing.assert_array_equal(df["n_complete_haplotypes"], _ref_complete(hap[rows]))
    assert len(few) == 1
    assert f"of {len(rows)} haplotypes" in str(few[0].message)

    df, few = _run(_hm(hap), window_size=WIN, statistics=["garud_h1"], populations=[rows],
                   missing_data="exclude")
    np.testing.assert_array_equal(df["n_complete_haplotypes"], len(rows))
    assert not few


def test_streaming_reports_the_column_and_warns_once(tmp_path):
    hm = simulate_hm(n_samples=50, seq_length=50_000, seed=42, mutation_model='binary')
    hap = cp.asnumpy(hm.haplotypes).astype(np.int8)
    pos = cp.asnumpy(hm.positions)
    hap[np.random.default_rng(3).random(hap.shape) < 0.05] = -1
    hm = HaplotypeMatrix(hap, pos, 0, 50_000)
    hm.samples = [f"s{i}" for i in range(hap.shape[0] // 2)]
    path = str(tmp_path / "missing.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")

    df, few = _run(HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=10_000),
                   window_size=5_000, statistics=["garud_h12"])
    for k, (lo, hi) in enumerate(zip(df["start"], df["end"])):
        cols = (pos >= lo) & (pos < hi)
        assert df["n_complete_haplotypes"].iloc[k] == (~(hap[:, cols] < 0).any(axis=1)).sum()
    assert len(few) == 1


# ---------------------------------------------------------------------------
# The warning and the cutoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,rate", [(20, 0.0), (20, 0.001), (200, 0.01)])
def test_no_warning_when_windows_are_well_anchored(n, rate):
    # With n = 20 the bound is half the sample, so data with few or no
    # missing calls does not warn.
    _, few = _run(_hm(_hap(n, rate)), window_size=WIN, statistics=["garud_h12"])
    assert not few


def test_warning_when_most_windows_are_thin():
    hap = _hap(200, 0.03)
    assert (_ref_complete(hap) < 25).all()
    df, few = _run(_hm(hap), window_size=WIN, statistics=["garud_h12"])
    assert len(few) == 1
    assert few[0].filename == __file__
    assert "4 of 4 windows" in str(few[0].message)


def test_min_complete_haplotypes_sets_thin_windows_to_nan():
    hap = _hap(200, 0.03)
    n_complete = _ref_complete(hap)
    cutoff = int(np.median(n_complete)) + 1
    df, few = _run(_hm(hap), window_size=WIN, statistics=GARUD,
                   min_complete_haplotypes=cutoff)
    ref, _ = _run(_hm(hap), window_size=WIN, statistics=GARUD)
    thin = n_complete < cutoff
    assert thin.any() and not thin.all()
    assert df.loc[thin, GARUD].isna().all().all()
    np.testing.assert_array_equal(df.loc[~thin, GARUD], ref.loc[~thin, GARUD])
    assert not few


def test_moving_garud_h_warns_on_thin_windows():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        selection.moving_garud_h(_hm(_hap(200, 0.03)), size=WIN_SITES)
        selection.moving_garud_h(_hm(_hap(200, 0.0)), size=WIN_SITES)
    few = _few_complete(record)
    assert len(few) == 1
    assert few[0].filename == __file__


def test_fallback_accepts_populations_given_as_rows():
    # The per-window fallback ('pi' with a Garud statistic under 'exclude')
    # gives row-list populations the values it gives the same rows by name.
    hap = _hap(60, 0.01)
    r1, r2 = list(range(30)), list(range(30, 60))
    stats = ["garud_h1", "pi"]
    rows, _ = _run(_hm(hap), window_size=WIN, statistics=stats, populations=[r1, r2],
                   missing_data="exclude")
    named, _ = _run(_hm(hap, {"a": r1, "b": r2}), window_size=WIN, statistics=stats,
                    populations=["a", "b"], missing_data="exclude")
    for stat in stats + ["n_complete_haplotypes"]:
        for rows_pop, name in ((r1, "a"), (r2, "b")):
            np.testing.assert_array_equal(rows[f"{stat}_{rows_pop}"], named[f"{stat}_{name}"])
