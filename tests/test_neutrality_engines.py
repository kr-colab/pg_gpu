"""theta_l and the Achaz neutrality tests on every windowed engine.

theta_l, normalized_fay_wu_h, zeng_e and zeng_dh are computed by the scatter
engine, both fused engines and the per-window fallback, and diversity_stats
knows zeng_dh. Each path must give the scalar function's value on the
window, so a request can mix these names with any other.
"""

import functools

import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, diversity
from pg_gpu.windowed_analysis import (
    WindowedAnalyzer, windowed_analysis, windowed_statistics_fused_chunked)

SCALAR = {
    'theta_l': diversity.theta_l,
    'normalized_fay_wu_h': diversity.normalized_fay_wus_h,
    'zeng_e': diversity.zeng_e,
    'zeng_dh': diversity.zeng_dh,
}
NAMES = sorted(SCALAR)
LENGTH, WINDOW = 400_000, 100_000
POPS = {'a': list(range(0, 24)), 'b': list(range(24, 40))}


@functools.lru_cache(maxsize=None)
def _data():
    # The default mutation model gives some multiallelic sites, where
    # theta_l is not (pi + theta_h) / 2.
    ts = msprime.sim_mutations(
        msprime.sim_ancestry(20, sequence_length=LENGTH, population_size=1e4,
                             recombination_rate=1e-8, random_seed=11),
        rate=4e-8, random_seed=11)
    hap = ts.genotype_matrix().T.astype(np.int8)
    assert hap.max() > 1
    rng = np.random.default_rng(11)
    hap[rng.random(hap.shape) < 0.03] = -1
    return hap, ts.tables.sites.position.astype(np.int64) + 1


def _hm(mask=False):
    hap, pos = _data()
    hm = HaplotypeMatrix(hap.copy(), pos.copy(), 1, LENGTH, sample_sets=POPS)
    if mask:
        accessible = np.ones(LENGTH, dtype=bool)
        accessible[150_000:260_000] = False
        hm.set_accessible_mask(accessible)
    return hm


def _expected(name, hm, **kw):
    """The scalar function on each window of ``hm``."""
    return np.array([
        SCALAR[name](hm.get_subset_from_range(lo, lo + WINDOW), **kw)
        for lo in range(1, LENGTH, WINDOW)])


def _check(got, name, missing_data, mask=False):
    np.testing.assert_allclose(
        np.asarray(got, dtype=float),
        _expected(name, _hm(mask), missing_data=missing_data),
        rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("mask", [False, True], ids=["no-mask", "mask"])
@pytest.mark.parametrize("extra", [None, "mu_var", "garud_h1"],
                         ids=["alone", "with-mu_var", "with-garud_h1"])
def test_bp_windows_match_scalar(extra, mask, missing_data):
    # Alone, the scatter engine serves the names. mu_var sends the request
    # to the fused engine under both modes; garud_h1 sends it to the fused
    # engine under 'include' and to the fallback under 'exclude'.
    stats = NAMES + ([extra] if extra else [])
    df = windowed_analysis(_hm(mask), WINDOW, statistics=stats,
                           missing_data=missing_data)
    for name in NAMES:
        _check(df[name], name, missing_data, mask=mask)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
def test_chunked_engine_matches_scalar(monkeypatch, missing_data):
    monkeypatch.setattr("pg_gpu._memutil.estimate_fused_chunk_size",
                        lambda n, memory_fraction=0.35: 150)
    bins = np.arange(1, LENGTH + 2, WINDOW, dtype=float)
    got = windowed_statistics_fused_chunked(
        _hm(), bins, statistics=tuple(NAMES), missing_data=missing_data)
    for name in NAMES:
        _check(got[name], name, missing_data)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
def test_fallback_matches_scalar(missing_data):
    df = WindowedAnalyzer(window_size=WINDOW, statistics=NAMES,
                          missing_data=missing_data,
                          progress_bar=False).compute(_hm())
    for name in NAMES:
        _check(df[name], name, missing_data)


@pytest.mark.parametrize("extra", [None, "mu_var"], ids=["scatter", "fused"])
def test_region_windows_match_scalar(extra):
    import pandas as pd
    regions = pd.DataFrame({'start': [250_001, 1, 40_001],
                            'end': [400_001, 120_001, 90_001]})
    hm = _hm()
    df = windowed_analysis(hm, window_type='regions', regions=regions,
                           statistics=NAMES + ([extra] if extra else []))
    for name in NAMES:
        ref = [SCALAR[name](hm.get_subset_from_range(int(r.start), int(r.end)))
               for r in regions.itertuples()]
        np.testing.assert_allclose(df[name].to_numpy(float), ref, rtol=1e-9,
                                   atol=1e-12, equal_nan=True, err_msg=name)


def test_snp_windows_accept_the_names():
    hm = _hm()
    df = windowed_analysis(hm, 200, window_type='snp', statistics=NAMES)
    # An snp window spans its first to its last variant.
    first = hm.get_subset(np.arange(200))
    pos = cp.asnumpy(first.positions)
    window = HaplotypeMatrix(cp.asnumpy(first.haplotypes), pos,
                             int(pos[0]), int(pos[-1]))
    for name in NAMES:
        np.testing.assert_allclose(df[name].iloc[0], SCALAR[name](window),
                                   rtol=1e-9, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
def test_two_population_request_uses_the_first_population(missing_data):
    # With a two-population statistic the fused engine serves the request;
    # single-population columns are for the first population, on the sites
    # the request keeps (complete in both populations under 'exclude').
    df = windowed_analysis(_hm(), WINDOW, statistics=NAMES + ['dxy', 'mu_var'],
                           populations=['a', 'b'], missing_data=missing_data)
    hm = _hm()
    if missing_data == 'exclude':
        hm = hm.exclude_missing_sites(populations=['a', 'b'])
    for name in NAMES:
        np.testing.assert_allclose(
            df[name].to_numpy(float), _expected(name, hm, population='a'),
            rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=name)


def test_stream_matches_eager(tmp_path):
    hm = _hm()
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "s.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    stream = HaplotypeMatrix.from_zarr(path, region=f"1:1-{LENGTH}",
                                       streaming="always", chunk_bp=200_000)
    stats = NAMES + ['mu_var']
    eager = windowed_analysis(_hm(), WINDOW, statistics=stats)
    streamed = windowed_analysis(stream, WINDOW, statistics=stats)
    for name in NAMES:
        np.testing.assert_allclose(streamed[name], eager[name], rtol=1e-9,
                                   equal_nan=True, err_msg=name)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
def test_diversity_stats_knows_zeng_dh(missing_data):
    hm = _hm()
    got = diversity.diversity_stats(
        hm, statistics=['zeng_dh', 'pi'], missing_data=missing_data)
    assert list(got) == ['zeng_dh', 'pi']
    np.testing.assert_allclose(
        got['zeng_dh'], diversity.zeng_dh(hm, missing_data=missing_data),
        rtol=1e-12, equal_nan=True)
