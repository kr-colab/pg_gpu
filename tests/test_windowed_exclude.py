"""windowed_analysis under missing_data='exclude' for statistics the per-window
fallback does not compute: the diploS/HIC statistics, mean_nsl, theta_h,
singletons, fay_wu_h and max_daf.

'exclude' keeps the sites with no missing call in the rows the request
reads. Each windowed value must equal the scalar statistic on that window's
complete sites, the window grid must not move, and n_variants counts the
complete sites.
"""
import cupy as cp
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, diversity, distance_stats, ld_statistics, selection
from pg_gpu.windowed_analysis import (
    windowed_analysis, windowed_statistics_fused, windowed_statistics_fused_chunked,
)

from .conftest import founder_haplotypes, simulate_hm

N_HAP, N_VAR, SPACING, WIN = 60, 600, 10, 1500
POPS = {"p1": list(range(30)), "p2": list(range(30, 60))}
WINDOWS = [(lo, lo + WIN) for lo in range(0, N_VAR * SPACING, WIN)]


def _data(seed=0, rate=0.004):
    # Founder haplotypes give real LD; private flips keep sites segregating
    # at varied frequencies.
    rng = np.random.default_rng(seed)
    hap = founder_haplotypes(rng, N_HAP, N_VAR, 8)
    hap ^= (rng.random(hap.shape) < 0.02).astype(np.int8)
    hap[rng.random(hap.shape) < rate] = -1
    # One site missing only in p2: complete for p1 alone, not for both.
    hap[:, 7] = np.where(np.arange(N_HAP) < 30, hap[:, 7] % 2, -1)
    return hap


def _hm(hap, sample_sets=None):
    return HaplotypeMatrix(hap.copy(), np.arange(N_VAR, dtype=np.int64) * SPACING,
                           0, N_VAR * SPACING, sample_sets=sample_sets)


def _gaps(sub):
    return np.diff(cp.asnumpy(sub.positions)).astype(np.float64)


def _mean_nsl(sub, nsl_by_pos):
    v = np.array([nsl_by_pos[p] for p in cp.asnumpy(sub.positions)])
    return v[np.isfinite(v)].mean()


# Scalar reference for each statistic on one window's complete sites.
SCALAR = {
    "zns": lambda sub, nsl: ld_statistics.zns(sub),
    "omega": lambda sub, nsl: ld_statistics.omega(sub),
    "mu_ld": lambda sub, nsl: ld_statistics.mu_ld(sub),
    "mu_sfs": lambda sub, nsl: diversity.mu_sfs(sub),
    "mu_var": lambda sub, nsl: sub.num_variants / WIN,
    "mean_nsl": _mean_nsl,
    **{s: (lambda sub, nsl, i=i: distance_stats.dist_moments(sub)[i])
       for i, s in enumerate(("dist_var", "dist_skew", "dist_kurt"))},
    **{s: (lambda sub, nsl, f=f: f(_gaps(sub)))
       for s, f in (("snp_dist_mean", np.mean), ("snp_dist_var", np.var),
                    ("snp_dist_min", np.min), ("snp_dist_max", np.max))},
}


@pytest.fixture(scope="module")
def case():
    hap = _data()
    complete = _hm(hap).exclude_missing_sites()
    nsl = np.asarray(selection.nsl(complete), dtype=np.float64).reshape(complete.num_variants, -1)[:, 0]
    return hap, complete, dict(zip(cp.asnumpy(complete.positions), nsl))


def _exclude(hap, stats, sample_sets=None, **kw):
    return windowed_analysis(_hm(hap, sample_sets), window_size=WIN, step_size=WIN,
                             statistics=stats, missing_data="exclude", **kw)


def _include(matrix, stats, **kw):
    return windowed_analysis(matrix, window_size=WIN, step_size=WIN, statistics=stats, **kw)


@pytest.mark.parametrize("stat", list(SCALAR))
def test_exclude_matches_scalar_on_complete_sites(case, stat):
    hap, complete, nsl_by_pos = case
    df = _exclude(hap, [stat])
    assert list(df["start"]) == [lo for lo, _ in WINDOWS]
    for k, (lo, hi) in enumerate(WINDOWS):
        sub = complete.get_subset_from_range(lo, hi)
        assert df["n_variants"].iloc[k] == sub.num_variants
        np.testing.assert_allclose(df[stat].iloc[k], SCALAR[stat](sub, nsl_by_pos),
                                   rtol=1e-9, atol=1e-12)


def test_mixed_exclude_request_matches_single_requests(case):
    # A request mixing these statistics with pi gives each column the value
    # it has alone, and pi the value from the complete sites.
    hap, complete, _ = case
    stats = list(SCALAR)
    mixed = _exclude(hap, stats + ["pi"])
    for s in stats:
        np.testing.assert_allclose(mixed[s], _exclude(hap, [s])[s], rtol=1e-12)
    np.testing.assert_allclose(mixed["pi"], _include(complete, ["pi"])["pi"], rtol=1e-12)


def test_exclude_daf_hist_matches_scalar(case):
    hap, complete, _ = case
    df = _exclude(hap, ["daf_hist"])
    cols = sorted((c for c in df.columns if c.startswith("daf_bin_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    for k, (lo, hi) in enumerate(WINDOWS):
        sub = complete.get_subset_from_range(lo, hi)
        np.testing.assert_allclose(df[cols].iloc[k].to_numpy(dtype=np.float64),
                                   np.asarray(diversity.daf_histogram(sub)[0], dtype=np.float64),
                                   rtol=1e-12)


@pytest.mark.parametrize("stat", ["theta_h", "singletons", "fay_wu_h", "max_daf"])
def test_exclude_single_population_statistics_outside_the_fallback(case, stat):
    # With a Garud statistic, or with two populations and no two-population
    # statistic, these requests skip the scatter engine; the fallback does not
    # know them, so the fused engine serves them on the complete sites.
    hap, complete, _ = case
    with_garud = _exclude(hap, [stat, "garud_h1"])
    np.testing.assert_allclose(with_garud[stat], _include(complete, [stat])[stat], rtol=1e-12)

    two_pops = _exclude(hap, [stat], sample_sets=POPS, populations=["p1", "p2"])
    p1_complete = _hm(hap, POPS).exclude_missing_sites(populations=["p1"])
    np.testing.assert_allclose(two_pops[stat],
                               _include(p1_complete, [stat], populations=["p1"])[stat],
                               rtol=1e-12)


@pytest.mark.parametrize("engine", [windowed_statistics_fused, windowed_statistics_fused_chunked])
def test_direct_engines_apply_exclude(case, engine):
    # Each engine, called directly, drops incomplete sites under 'exclude'
    # and gives the same values as 'include' on the complete sites.
    hap, complete, _ = case
    stats = ("zns", "mu_sfs", "pi", "mean_nsl", "snp_dist_mean")
    edges = np.arange(0, N_VAR * SPACING + 1, WIN)
    got = engine(_hm(hap), bp_bins=edges, statistics=stats, missing_data="exclude")
    ref = engine(complete, bp_bins=edges, statistics=stats, missing_data="include")
    for s in stats + ("n_variants",):
        np.testing.assert_allclose(np.asarray(got[s], dtype=np.float64),
                                   np.asarray(ref[s], dtype=np.float64), equal_nan=True)


def test_chunked_and_unchunked_agree_with_two_populations(case):
    # The chunked engine hands zns to the unchunked one; that hand-off must
    # not filter again over a different set of rows. Rows 40-59 belong to
    # neither population, so a second filter over every row would drop the
    # sites missing only there.
    hap, _, _ = case
    pops = {"p1": list(range(20)), "p2": list(range(20, 40))}
    edges = np.arange(0, N_VAR * SPACING + 1, WIN)
    kw = dict(bp_bins=edges, statistics=("pi", "fst", "zns"), pop1="p1", pop2="p2",
              missing_data="exclude")
    chunked = windowed_statistics_fused_chunked(_hm(hap, pops), **kw)
    plain = windowed_statistics_fused(_hm(hap, pops), **kw)
    for s in ("pi", "fst", "zns", "n_variants"):
        np.testing.assert_allclose(np.asarray(chunked[s], dtype=np.float64),
                                   np.asarray(plain[s], dtype=np.float64),
                                   rtol=1e-12, equal_nan=True)


def test_exclude_judges_completeness_within_the_population(case):
    # Site 7 is complete for p1 but not for p1 and p2 together. A
    # single-population request on p1 keeps it; a request that also reads
    # p2 through a two-population statistic drops it.
    hap, _, _ = case
    hm = _hm(hap, POPS)
    p1_complete = hm.exclude_missing_sites(populations=["p1"])
    both_complete = hm.exclude_missing_sites(populations=["p1", "p2"])
    site7 = 7 * SPACING
    assert site7 in cp.asnumpy(p1_complete.positions)
    assert site7 not in cp.asnumpy(both_complete.positions)

    single = _exclude(hap, ["zns"], sample_sets=POPS, populations=["p1"])
    ref1 = _include(p1_complete, ["zns"], populations=["p1"])
    np.testing.assert_allclose(single["zns"], ref1["zns"], rtol=1e-12)
    np.testing.assert_array_equal(single["n_variants"], ref1["n_variants"])

    mixed = _exclude(hap, ["zns", "fst"], sample_sets=POPS, populations=["p1", "p2"])
    ref2 = _include(both_complete, ["zns", "fst"], populations=["p1", "p2"])
    np.testing.assert_allclose(mixed["zns"], ref2["zns"], rtol=1e-12)
    np.testing.assert_allclose(mixed["fst"], ref2["fst"], rtol=1e-12, equal_nan=True)


def test_exclude_keeps_the_grid_when_an_end_site_is_incomplete():
    # The last site is incomplete. Without chromosome bounds the grid comes
    # from the variant hull, which must be taken before the filter.
    hap = _data(seed=3)
    hap[0, -1] = -1
    hm = HaplotypeMatrix(hap.copy(), np.arange(N_VAR, dtype=np.int64) * SPACING)
    inc = windowed_analysis(hm, window_size=WIN, step_size=WIN, statistics=["mu_sfs"])
    exc = windowed_analysis(hm, window_size=WIN, step_size=WIN, statistics=["mu_sfs"],
                            missing_data="exclude")
    np.testing.assert_array_equal(inc["start"], exc["start"])
    np.testing.assert_array_equal(inc["end"], exc["end"])


def test_streaming_exclude_matches_scalar(tmp_path):
    hm = simulate_hm(n_samples=20, seq_length=50_000, seed=42, mutation_model='binary')
    hap = cp.asnumpy(hm.haplotypes).astype(np.int8)
    pos = cp.asnumpy(hm.positions)
    rng = np.random.default_rng(71)
    hap[rng.random(hap.shape) < 0.002] = -1
    hm = HaplotypeMatrix(hap, pos, 0, 50_000)
    hm.samples = [f"s{i}" for i in range(hap.shape[0] // 2)]
    path = str(tmp_path / "missing.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")

    stream = windowed_analysis(
        HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=10_000),
        window_size=5_000, step_size=5_000, statistics=["zns", "mu_sfs", "snp_dist_mean"],
        missing_data="exclude")
    complete = hm.exclude_missing_sites()
    assert complete.num_variants < hm.num_variants
    for k, (lo, hi) in enumerate(zip(stream["start"], stream["end"])):
        sub = complete.get_subset_from_range(lo, hi)
        assert stream["n_variants"].iloc[k] == sub.num_variants
        np.testing.assert_allclose(stream["zns"].iloc[k], ld_statistics.zns(sub), rtol=1e-9)
        np.testing.assert_allclose(stream["mu_sfs"].iloc[k], diversity.mu_sfs(sub), rtol=1e-12)
        np.testing.assert_allclose(stream["snp_dist_mean"].iloc[k], _gaps(sub).mean(), rtol=1e-12)
