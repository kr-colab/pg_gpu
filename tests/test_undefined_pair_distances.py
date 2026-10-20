"""A pair of samples with no site called in both has no distance.

The pairwise distance and IBS functions give NaN for such a pair, not 0.0,
which would mark the pair as identical (or, for IBS, as sharing nothing).
Each consumer skips the pair or refuses it: the distance moments use the
defined pairs, and pcoa raises.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, relatedness, windowed_analysis
from pg_gpu.decomposition import lostruct, pairwise_distance, pcoa
from pg_gpu.distance_stats import dist_moments, pairwise_diffs_diploid, pairwise_diffs_haploid
from pg_gpu.windowed_analysis import windowed_statistics_fused_chunked

# Haplotypes 0 and 1 are never called at the same site, so the first pair
# in condensed order, (0, 1), has no distance.
HAP = np.array([[0, 0, 0, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0]], dtype=np.int8)
# Per-site differences of the five defined pairs: (0,2) (0,3) (1,2) (1,3) (2,3).
HAP_DEFINED = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

# Individuals 0 and 1 are never called at the same site.
GENO = np.array([[0, 0, 0, -1, -1, -1],
                 [-1, -1, -1, 2, 2, 2],
                 [0, 1, 0, 1, 1, 1],
                 [2, 2, 2, 0, 0, 0]], dtype=np.int8)
GENO_DEFINED = np.array([1 / 3, 1.0, 1.0, 1.0, 1.0])


def _hm(hap=HAP):
    return HaplotypeMatrix(hap.copy(), np.arange(hap.shape[1]) * 10 + 1, 1, hap.shape[1] * 10)


def _gm():
    return GenotypeMatrix(GENO.copy(), np.arange(6) * 10 + 1, 1, 60)


def _moments(d):
    # Sample variance, and the population skewness and excess kurtosis.
    return np.var(d, ddof=1), stats.skew(d), stats.kurtosis(d)


def test_haploid_distances_are_nan_for_an_unshared_pair():
    d = pairwise_diffs_haploid(_hm())
    assert np.isnan(d[0])
    np.testing.assert_array_equal(d[1:], HAP_DEFINED)


def test_diploid_distances_are_nan_for_an_unshared_pair():
    d = pairwise_diffs_diploid(_gm())
    assert np.isnan(d[0])
    np.testing.assert_allclose(d[1:], GENO_DEFINED)


def test_dist_moments_skip_the_unshared_pair():
    np.testing.assert_allclose(dist_moments(_hm()), _moments(HAP_DEFINED), rtol=1e-12)


def test_dist_moments_skip_the_unshared_pair_on_genotypes():
    np.testing.assert_allclose(dist_moments(_gm()), _moments(GENO_DEFINED), rtol=1e-12)


def test_dist_moments_are_nan_when_no_pair_is_defined():
    # Three haplotypes, each called at a different site only.
    hap = np.array([[0, -1, -1], [-1, 1, -1], [-1, -1, 0]], dtype=np.int8)
    assert np.isnan(dist_moments(_hm(hap))).all()


@pytest.mark.parametrize("metric", ["cityblock", "sqeuclidean", "euclidean"])
def test_pairwise_distance_is_nan_for_an_unshared_pair(metric):
    # The defined pairs share every site they call, so the scaled count is
    # the per-site difference times the site count.
    counts = HAP_DEFINED * HAP.shape[1]
    d = pairwise_distance(_hm(), metric=metric)
    assert np.isnan(d[0])
    np.testing.assert_allclose(d[1:], np.sqrt(counts) if metric == "euclidean" else counts)


@pytest.mark.parametrize("square", [False, True], ids=["condensed", "square"])
def test_pcoa_raises_for_an_undefined_distance(square):
    from scipy.spatial.distance import squareform
    d = pairwise_distance(_hm())
    with pytest.raises(ValueError, match=r"undefined \(NaN\): 1 of 6 pairs"):
        pcoa(squareform(d, checks=False) if square else d)


def test_windowed_distance_moments_skip_the_unshared_pair():
    # One window holds every site, so each column is the moment of the five
    # defined pairs.
    expected = _moments(HAP_DEFINED)
    stats = ["dist_var", "dist_skew", "dist_kurt"]
    df = windowed_analysis(_hm(), window_size=60, statistics=stats)
    np.testing.assert_allclose(df[stats].iloc[0].to_numpy(), expected, rtol=1e-12)
    chunked = windowed_statistics_fused_chunked(_hm(), bp_bins=np.array([1, 61]),
                                                statistics=tuple(stats))
    np.testing.assert_allclose([chunked[s][0] for s in stats], expected, rtol=1e-12)


def test_ibs_is_nan_for_an_unshared_pair():
    m = relatedness.ibs(_gm())
    assert np.isnan(m[0, 1]) and np.isnan(m[1, 0])
    np.testing.assert_array_equal(np.diag(m), 1.0)
    assert np.isfinite(m[np.triu_indices(4, 1)][1:]).all()


def test_streaming_ibs_is_nan_for_an_unshared_pair(tmp_path):
    gm = _gm()
    gm.samples = [f"s{i}" for i in range(4)]
    path = str(tmp_path / "unshared.vcz")
    gm.to_zarr(path, format="vcz", contig_name="1")
    eager = relatedness.ibs(GenotypeMatrix.from_zarr(path, streaming="never"))
    stream = relatedness.ibs(GenotypeMatrix.from_zarr(path, streaming="always", chunk_bp=20))
    assert np.isnan(stream[0, 1]) and np.isnan(stream[1, 0])
    np.testing.assert_allclose(stream, eager, equal_nan=True)


def test_lostruct_skips_a_window_with_no_local_pca():
    # The first region holds one variant, too few for a local PCA, so it has
    # no distance to any window. The MDS uses the other windows, and the
    # invalid window gets a NaN row.
    rng = np.random.default_rng(3)
    hap = (rng.random((20, 200)) < 0.3).astype(np.int8)
    hm = HaplotypeMatrix(hap, np.arange(200) * 100, 0, 20_000)
    starts = [0] + list(range(1000, 20_000, 1000))
    regions = pd.DataFrame({"chrom": 1, "start": starts, "end": [100] + [s + 1000 for s in starts[1:]]})
    res = lostruct(hm, k=2, window_type="regions", regions=regions,
                   window_size=0, step_size=0, n_corners=3, random_state=0)
    assert np.isnan(res.mds[0]).all()
    assert np.isfinite(res.mds[1:]).all()
    ref, evr = pcoa(res.distance[1:, 1:], n_components=2)
    np.testing.assert_allclose(res.mds[1:], ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.explained_variance_ratio, evr, rtol=1e-12)
    assert 0 not in res.corner_indices


def test_lostruct_needs_two_windows_with_a_local_pca():
    # Two of the three regions hold one variant each.
    rng = np.random.default_rng(3)
    hm = HaplotypeMatrix((rng.random((20, 200)) < 0.3).astype(np.int8),
                         np.arange(200) * 100, 0, 20_000)
    regions = pd.DataFrame({"chrom": 1, "start": [0, 1000, 5000], "end": [100, 3000, 5100]})
    with pytest.raises(ValueError, match="at least two windows"):
        lostruct(hm, k=2, window_type="regions", regions=regions,
                 window_size=0, step_size=0, n_corners=0)
