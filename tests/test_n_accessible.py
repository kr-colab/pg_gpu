"""The n_accessible column of windowed_analysis.

It is each window's base count, the denominator of the per-base statistics
in that window: the accessible bases when the matrix has a mask, else the
window's share of n_total_sites, else the window length. Every engine must
report the same count for a window, and must divide by the count it reports.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from pg_gpu import HaplotypeMatrix, windowed_analysis

from .conftest import force_chunked_engine

# pg_gpu.windowed_analysis names the function, so the module needs its path.
wa = importlib.import_module("pg_gpu.windowed_analysis")

N_VAR, LENGTH, WINDOW = 400, 20_000, 5_000
POPS = {"p1": list(range(6)), "p2": list(range(6, 12))}
# Accessible: positions 1-7000 and 9301-19000 (BED is 0-based, half-open).
BED_INTERVALS = [(0, 7000), (9300, 19000)]
N_TOTAL_SITES = 15_000
STARTS = np.arange(0, LENGTH, WINDOW)


def _data():
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(np.arange(1, LENGTH), N_VAR, replace=False))
    hap = rng.integers(0, 2, (12, N_VAR)).astype(np.int8)
    hap[rng.random(hap.shape) < 0.02] = -1
    return hap, pos


def _bed(tmp_path):
    path = tmp_path / "accessible.bed"
    path.write_text("".join(f"1\t{lo}\t{hi}\n" for lo, hi in BED_INTERVALS))
    return str(path)


def _hm(kind, tmp_path):
    """A matrix whose denominator comes from a mask, from n_total_sites, or
    from the window length."""
    hap, pos = _data()
    hm = HaplotypeMatrix(hap, pos, 0, LENGTH - 1, sample_sets=dict(POPS))
    if kind == "mask":
        hm.set_accessible_mask(_bed(tmp_path), chrom="1")
    elif kind == "n_total_sites":
        hm.n_total_sites = N_TOTAL_SITES
    return hm


def _accessible_bases(start, end):
    """Accessible bases at positions in [start, end), counted one by one."""
    accessible = np.zeros(LENGTH + 2, dtype=bool)
    for lo, hi in BED_INTERVALS:
        accessible[lo + 1:hi + 1] = True      # BED [lo, hi) is positions lo+1..hi
    return int(accessible[max(start, 1):end].sum())


def _expected(kind):
    """The count of each WINDOW-wide window, from no pg_gpu code."""
    if kind == "mask":
        return [_accessible_bases(lo, lo + WINDOW) for lo in STARTS]
    # Four equal windows tile the region: a quarter of n_total_sites each,
    # or the window length.
    return [N_TOTAL_SITES / 4 if kind == "n_total_sites" else WINDOW] * 4


def test_the_mask_oracle_counts_the_bed_bases():
    assert _expected("mask") == [4999, 2700, 5000, 4001]


def test_without_a_mask_it_is_the_window_length(tmp_path):
    # A window size that does not divide the region leaves a short last window.
    df = windowed_analysis(_hm("none", tmp_path), 6000, statistics=["pi"])
    np.testing.assert_array_equal(df["n_accessible"], [6000, 6000, 6000, 2000])
    np.testing.assert_array_equal(df["n_accessible"], df["end"] - df["start"])


@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
def test_column_follows_n_variants_and_is_float(kind, tmp_path):
    df = windowed_analysis(_hm(kind, tmp_path), WINDOW, statistics=["pi"])
    assert list(df.columns[4:6]) == ["n_variants", "n_accessible"]
    assert df["n_accessible"].dtype == np.float64


# The same request on every path that makes windows. Each returns the frame.
def _scatter(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["pi"], missing_data=md)


def _fused(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["pi", "mu_var"], missing_data=md)


def _fallback(hm, md):
    return wa.WindowedAnalyzer(window_size=WINDOW, step_size=WINDOW, statistics=["pi"],
                               missing_data=md, progress_bar=False).compute(hm)


def _two_population_scatter(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["fst"], populations=["p1", "p2"],
                             missing_data=md)


def _per_population_split(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["pi", "fst", "mu_var"],
                             populations=["p1", "p2"], missing_data=md)


def _raw(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["pi"], missing_data=md,
                             span_normalize=False)


def _named(hm, md):
    return windowed_analysis(hm, WINDOW, statistics=["n_accessible", "n_variants"],
                             missing_data=md)


PATHS = [_scatter, _fused, _fallback, _two_population_scatter, _per_population_split, _raw, _named]


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
@pytest.mark.parametrize("path", PATHS, ids=lambda f: f.__name__.strip("_"))
def test_every_path_reports_the_count(path, kind, missing_data, tmp_path):
    np.testing.assert_allclose(path(_hm(kind, tmp_path), missing_data)["n_accessible"],
                               _expected(kind), rtol=1e-12)


@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
def test_chunked_engine_reports_the_count(kind, tmp_path, monkeypatch):
    force_chunked_engine(monkeypatch, chunk=60)
    np.testing.assert_allclose(_fused(_hm(kind, tmp_path), "include")["n_accessible"],
                               _expected(kind), rtol=1e-12)


@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
def test_regions_windows(kind, tmp_path):
    regions = pd.DataFrame({"start": [0, 6000, 9000], "end": [5000, 9500, 20000]})
    kw = dict(window_type="regions", regions=regions)
    fast = windowed_analysis(_hm(kind, tmp_path), WINDOW, statistics=["pi"], **kw)
    fallback = wa.WindowedAnalyzer(statistics=["pi"], progress_bar=False, **kw).compute(
        _hm(kind, tmp_path))
    np.testing.assert_allclose(fast["n_accessible"], fallback["n_accessible"], rtol=1e-12)
    if kind == "mask":
        np.testing.assert_array_equal(
            fast["n_accessible"],
            [_accessible_bases(lo, hi) for lo, hi in zip(regions["start"], regions["end"])])
    if kind == "none":
        np.testing.assert_array_equal(fast["n_accessible"], regions["end"] - regions["start"])


@pytest.mark.parametrize("kind", ["mask", "none"])
def test_snp_windows_count_from_first_to_last_variant(kind, tmp_path):
    # A snp window's end is its last variant, so its bases run to end + 1.
    df = windowed_analysis(_hm(kind, tmp_path), 100, statistics=["pi"], window_type="snp")
    if kind == "mask":
        expected = [_accessible_bases(lo, hi + 1) for lo, hi in zip(df["start"], df["end"])]
    else:
        expected = df["end"] - df["start"] + 1
    np.testing.assert_array_equal(df["n_accessible"], expected)


@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
@pytest.mark.parametrize("path", [_scatter, _fused, _fallback], ids=lambda f: f.__name__.strip("_"))
def test_per_base_statistic_is_the_raw_sum_over_the_count(path, kind, tmp_path):
    # No site here has every call missing, so the per-base value has no
    # rescale for sites without data and is the raw sum over n_accessible.
    per_base = path(_hm(kind, tmp_path), "include")
    raw = windowed_analysis(_hm(kind, tmp_path), WINDOW, statistics=["pi"], span_normalize=False)
    np.testing.assert_allclose(per_base["pi"] * per_base["n_accessible"], raw["pi"], rtol=1e-10)


@pytest.mark.parametrize("population", [None, "p1"], ids=["all-rows", "population"])
@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
def test_local_pca_windows_keep_the_column(kind, population, tmp_path):
    # Local PCA windows a population's rows, which carry no mask; the count
    # must still come from the whole matrix.
    kw = {} if population is None else {"population": population}
    result = windowed_analysis(_hm(kind, tmp_path), WINDOW, statistics=["local_pca", "pi"], **kw)
    np.testing.assert_allclose(result.windows["n_accessible"], _expected(kind), rtol=1e-12)
    assert "pi" in result.windows.columns


@pytest.mark.parametrize("kind", ["mask", "n_total_sites", "none"])
def test_stream_matches_eager(kind, tmp_path):
    hap, pos = _data()
    hm = HaplotypeMatrix(hap, pos, 0, LENGTH - 1)
    hm.samples = [f"s{i}" for i in range(6)]
    path = str(tmp_path / "stream.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    kw = dict(accessible_bed=_bed(tmp_path), chrom="1") if kind == "mask" else {}
    frames = []
    for streaming in ("never", "always"):
        # Chunks of three windows leave a short last chunk in the stream.
        m = HaplotypeMatrix.from_zarr(path, region=f"1:1-{LENGTH}", streaming=streaming,
                                      chunk_bp=3 * WINDOW)
        if kind == "n_total_sites":
            m.n_total_sites = N_TOTAL_SITES
        frames.append(windowed_analysis(m, WINDOW, statistics=["pi"], **kw))
    eager, stream = frames
    np.testing.assert_array_equal(eager[["start", "end"]], stream[["start", "end"]])
    np.testing.assert_allclose(stream["n_accessible"], eager["n_accessible"], rtol=1e-12)


def test_the_fallback_takes_the_name_as_a_statistic(tmp_path):
    # windowed_analysis drops the name, because the column is always there;
    # the analyzer, used directly, must know it as it knows n_variants.
    df = wa.WindowedAnalyzer(window_size=WINDOW, step_size=WINDOW, progress_bar=False,
                             statistics=["n_accessible", "n_variants"]).compute(_hm("mask", tmp_path))
    np.testing.assert_array_equal(df["n_accessible"], _expected("mask"))
