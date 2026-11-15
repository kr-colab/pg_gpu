"""A windowed request that names two or more populations.

Each single-population statistic gets one column for each population, named
``<statistic>_<population>``, and its value is the scalar function on that
population's rows of the window. Under 'exclude' a site counts for a
statistic when the populations of that statistic have no missing call
there, so the value does not depend on what else is in the request or on
the engine that serves it. ``n_variants`` is one column for the row: under
'exclude' it counts the sites complete in every named population.

The matrix has three groups of rows. p1 is complete. p2 has missing calls
at some sites. The last rows are in no population and have missing calls at
other sites, so "every named population" and "every row" give different
counts.
"""
import importlib
import warnings

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, divergence, diversity, windowed_analysis

# pg_gpu.windowed_analysis names the function, so the module needs its path.
wa = importlib.import_module("pg_gpu.windowed_analysis")

N_VAR, SPACING, WINDOW = 120, 100, 3000
POPS = {"p1": list(range(8)), "p2": list(range(8, 16))}
P2_MISSING = np.arange(5, N_VAR, 9)        # sites where p2 has a missing call
OUTSIDE_MISSING = np.arange(2, N_VAR, 7)   # sites where an unnamed row has one
WINDOWS = [(lo, lo + WINDOW) for lo in range(0, N_VAR * SPACING, WINDOW)]


def _hap():
    rng = np.random.default_rng(3)
    hap = rng.integers(0, 2, (20, N_VAR)).astype(np.int8)
    hap[9, P2_MISSING] = -1
    hap[17, OUTSIDE_MISSING] = -1
    return hap


POSITIONS = np.arange(N_VAR) * SPACING + 1


def _hm(sample_sets=POPS, hap=None):
    return HaplotypeMatrix(_hap() if hap is None else hap, POSITIONS, 0,
                           N_VAR * SPACING - 1, sample_sets=dict(sample_sets))


def _scalar(fn, population, missing_data, **kw):
    """``fn`` on each window, on the rows of ``population``."""
    hm = _hm()
    return np.array([fn(hm.get_subset_from_range(lo, hi), population=population,
                        missing_data=missing_data, **kw) for lo, hi in WINDOWS])


def _force_chunked(monkeypatch):
    """Send the dispatcher's fused requests to the chunked engine, with
    chunks smaller than the matrix. The chunked engine hands some statistics
    back to the single-shot one, with its site set; those calls go through."""
    monkeypatch.setattr("pg_gpu._memutil.estimate_fused_chunk_size",
                        lambda n, memory_fraction=0.35: 40)
    single_shot = wa.windowed_statistics_fused

    def fused(*args, **kwargs):
        engine = single_shot if "_sites" in kwargs else wa.windowed_statistics_fused_chunked
        return engine(*args, **kwargs)

    monkeypatch.setattr(wa, "windowed_statistics_fused", fused)


# Requests with pi and a companion. The companion decides which engine
# serves pi for one population, or adds a two-population statistic to the
# request; pi must not notice either.
REQUESTS = {
    "scatter": ["pi", "theta_w"],
    "fused": ["pi", "mu_var"],
    "garud": ["pi", "garud_h1"],
    "with_two_population": ["pi", "fst"],
    "with_two_population_and_fused": ["pi", "fst", "zns"],
}
# The requests that reach the fused engine, which has a chunked form.
FUSED_REQUESTS = ("fused", "garud", "with_two_population_and_fused")
CASES = ([(name, False) for name in REQUESTS]
         + [(name, True) for name in FUSED_REQUESTS])


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("request_name,chunked", CASES,
                         ids=[n + ("-chunked" if c else "") for n, c in CASES])
def test_pi_of_each_population_is_the_scalar_value(request_name, chunked, missing_data, monkeypatch):
    if chunked:
        _force_chunked(monkeypatch)
    df = windowed_analysis(_hm(), WINDOW, statistics=REQUESTS[request_name],
                           populations=["p1", "p2"], missing_data=missing_data,
                           span_normalize=False)
    assert "pi" not in df.columns
    for pop in ("p1", "p2"):
        np.testing.assert_allclose(
            df[f"pi_{pop}"], _scalar(diversity.pi, pop, missing_data, span_normalize=False),
            rtol=1e-10, err_msg=pop)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
def test_every_single_population_column_gets_the_suffix(missing_data):
    stats = ["pi", "tajimas_d", "singletons", "zns", "garud_h1", "mean_nsl", "daf_hist", "fst", "dxy"]
    df = windowed_analysis(_hm(), WINDOW, statistics=stats, populations=["p1", "p2"],
                           missing_data=missing_data)
    columns = [c for c in df.columns if c not in wa.CANONICAL_WINDOW_PREFIX]
    assert {"fst", "dxy"} <= set(columns)
    single = [c for c in columns if c not in ("fst", "dxy")]
    assert all(c.endswith(("_p1", "_p2")) for c in single), single
    for name in ("pi", "tajimas_d", "singletons", "zns", "garud_h1", "mean_nsl",
                 "daf_bin_0", "n_complete_haplotypes"):
        assert {f"{name}_p1", f"{name}_p2"} <= set(columns), name


@pytest.mark.parametrize("stats", [["n_variants"], ["pi"], ["pi", "fst", "zns"]],
                         ids=["alone", "scatter", "fused"])
def test_n_variants_counts_sites_complete_in_every_named_population(stats):
    in_window = [(POSITIONS >= lo) & (POSITIONS < hi) for lo, hi in WINDOWS]
    complete = np.ones(N_VAR, dtype=bool)
    complete[P2_MISSING] = False        # the unnamed rows' missing calls do not count
    exclude = windowed_analysis(_hm(), WINDOW, statistics=stats, populations=["p1", "p2"],
                                missing_data="exclude")
    np.testing.assert_array_equal(exclude["n_variants"], [(w & complete).sum() for w in in_window])
    include = windowed_analysis(_hm(), WINDOW, statistics=stats, populations=["p1", "p2"],
                                missing_data="include")
    np.testing.assert_array_equal(include["n_variants"], [w.sum() for w in in_window])


def test_three_populations():
    sets = {**POPS, "p3": [16, 18, 19]}
    df = windowed_analysis(_hm(sets), WINDOW, statistics=["pi", "fst"],
                           populations=["p1", "p2", "p3"])
    assert {"pi_p1", "pi_p2", "pi_p3", "fst_p1_p2", "fst_p1_p3", "fst_p2_p3"} <= set(df.columns)


def test_population_given_as_rows():
    rows = [list(range(8)), list(range(8, 16))]
    df = windowed_analysis(_hm(), WINDOW, statistics=["pi"], populations=rows, span_normalize=False)
    named = windowed_analysis(_hm(), WINDOW, statistics=["pi"], populations=["p1", "p2"],
                              span_normalize=False)
    for row_list, name in zip(rows, ("p1", "p2")):
        np.testing.assert_allclose(df[f"pi_{row_list}"], named[f"pi_{name}"], rtol=1e-12)


def test_custom_statistic_stays_one_column():
    def n_rows(window):
        return window.matrix.num_haplotypes

    df = windowed_analysis(_hm(), WINDOW, statistics=["pi", n_rows], populations=["p1", "p2"])
    assert {"pi_p1", "pi_p2", "n_rows"} <= set(df.columns)
    assert (df["n_rows"] == 20).all()


def test_population_with_no_complete_site_gets_a_nan_column():
    hap = _hap()
    hap[9, :] = -1
    df = windowed_analysis(_hm(hap=hap), WINDOW, statistics=["pi"], populations=["p1", "p2"],
                           missing_data="exclude")
    assert len(df) == len(WINDOWS)
    assert np.isfinite(df["pi_p1"]).all()
    assert np.isnan(df["pi_p2"]).all()
    assert (df["n_variants"] == 0).all()


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("stats,window_type", [
    (["pi"], "bp"), (["pi", "garud_h1"], "bp"), (["pi", "mu_var"], "bp"),
    (["pi"], "snp"), (["pi", "garud_h1"], "snp"),
], ids=["scatter", "garud", "fused", "snp", "snp-garud"])
def test_one_named_population_gives_bare_columns(stats, window_type, missing_data):
    size = WINDOW if window_type == "bp" else 30
    df = windowed_analysis(_hm(), size, statistics=stats, populations=["p1"],
                           missing_data=missing_data, window_type=window_type)
    assert set(stats) <= set(df.columns)
    assert not [c for c in df.columns if c.endswith("_p1")]


def test_snp_windows_name_columns_as_bp_windows_do():
    df = windowed_analysis(_hm(), 30, statistics=["pi", "fst"], populations=["p1", "p2"],
                           window_type="snp")
    assert {"pi_p1", "pi_p2", "fst"} <= set(df.columns)


def test_duplicate_populations_raise():
    with pytest.raises(ValueError, match="distinct"):
        windowed_analysis(_hm(), WINDOW, statistics=["pi"], populations=["p1", "p1"])


def test_mix_without_one_engine_is_served_for_each_population():
    # zns (fused engine) and a callable (fallback) have no common engine, so
    # the request raises with one population. With two, zns runs in the
    # per-population requests and the callable in the request for the rest.
    def n_sites(window):
        return window.matrix.num_variants

    with pytest.raises(ValueError, match="No single windowed engine"):
        windowed_analysis(_hm(), WINDOW, statistics=["zns", n_sites], populations=["p1"])
    df = windowed_analysis(_hm(), WINDOW, statistics=["zns", n_sites], populations=["p1", "p2"])
    assert {"zns_p1", "zns_p2", "n_sites"} <= set(df.columns)
