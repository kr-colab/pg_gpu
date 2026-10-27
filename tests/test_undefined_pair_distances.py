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
from pg_gpu.decomposition import corners, lostruct, pairwise_distance, pcoa
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
OFF_DIAGONAL = ~np.eye(4, dtype=bool)


def _hm(hap=HAP):
    return HaplotypeMatrix(hap.copy(), np.arange(hap.shape[1]) * 10 + 1, 1, hap.shape[1] * 10)


def _gm(geno=GENO):
    return GenotypeMatrix(geno.copy(), np.arange(geno.shape[1]) * 10 + 1, 1, geno.shape[1] * 10)


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


@pytest.mark.parametrize("kind", ["haplotypes", "genotypes"])
def test_dist_moments_are_nan_when_no_pair_is_defined(kind):
    # Three samples, each called at a different site only.
    x = np.array([[0, -1, -1], [-1, 1, -1], [-1, -1, 0]], dtype=np.int8)
    m = _hm(x) if kind == "haplotypes" else _gm(np.where(x < 0, -1, 2 * x).astype(np.int8))
    assert np.isnan(dist_moments(m)).all()


def test_dist_moments_are_nan_when_missing_data_leaves_too_few_pairs():
    # Two complete haplotypes have one pair, too few for a variance: 0.0 by
    # the small-sample rule. When missing data leaves too few defined pairs,
    # the moments it cannot define are NaN instead.
    assert dist_moments(_hm(np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int8))) == (0.0, 0.0, 0.0)
    assert np.isnan(dist_moments(_hm(np.array([[0, -1], [-1, 1]], dtype=np.int8)))).all()
    # Four haplotypes, one defined pair (0, 3).
    one = np.array([[0, -1, -1], [-1, 1, -1], [-1, -1, 0], [0, -1, -1]], dtype=np.int8)
    assert np.isnan(dist_moments(_hm(one))).all()
    # Four haplotypes, two defined pairs: (0, 1) at distance 0, (1, 2) at 1.
    two = np.array([[0, -1, -1], [0, 0, -1], [-1, 1, -1], [-1, -1, 0]], dtype=np.int8)
    var, skew, kurt = dist_moments(_hm(two))
    assert var == pytest.approx(0.5)
    assert np.isnan(skew) and np.isnan(kurt)
    # Three defined pairs, all at distance 0: no spread, so skew is 0.0,
    # but kurt needs a fourth pair.
    three = np.array([[0, 0, 0], [0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=np.int8)
    var, skew, kurt = dist_moments(_hm(three))
    assert (var, skew) == (0.0, 0.0) and np.isnan(kurt)


def test_exclude_with_no_complete_site_gives_nan():
    # Every site of HAP and GENO has a missing call.
    assert np.isnan(pairwise_diffs_haploid(_hm(), missing_data="exclude")).all()
    assert np.isnan(pairwise_diffs_diploid(_gm(), missing_data="exclude")).all()
    assert np.isnan(pairwise_distance(_hm(), missing_data="exclude")).all()
    assert np.isnan(dist_moments(_hm(), missing_data="exclude")).all()
    assert np.isnan(relatedness.ibs(_gm(), missing_data="exclude")[OFF_DIAGONAL]).all()


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
    with pytest.raises(ValueError, match=r"undefined \(NaN\): 1 of 6 pairs, in 2 of 4 rows"):
        pcoa(squareform(d, checks=False) if square else d)


def test_pcoa_counts_a_nan_on_the_diagonal():
    with pytest.raises(ValueError, match="0 of 3 pairs, in 1 of 3 rows"):
        pcoa(np.array([[0.0, 1.0, 2.0], [1.0, np.nan, 1.5], [2.0, 1.5, 0.0]]))


def test_windowed_distance_moments_skip_the_unshared_pair():
    # The first window holds HAP, so each column is the moment of its five
    # defined pairs. In the second window each haplotype calls a different
    # site, so no pair is defined.
    block = np.full((4, 4), -1, dtype=np.int8)
    np.fill_diagonal(block, 0)
    hm = _hm(np.hstack([HAP, block]))
    expected = _moments(HAP_DEFINED)
    stats = ["dist_var", "dist_skew", "dist_kurt"]
    df = windowed_analysis(hm, window_size=60, statistics=stats)
    np.testing.assert_allclose(df[stats].iloc[0].to_numpy(), expected, rtol=1e-12)
    assert np.isnan(df[stats].iloc[1].to_numpy()).all()
    chunked = windowed_statistics_fused_chunked(hm, bp_bins=np.array([1, 61, 101]),
                                                statistics=tuple(stats))
    np.testing.assert_allclose([chunked[s][0] for s in stats], expected, rtol=1e-12)
    assert np.isnan([chunked[s][1] for s in stats]).all()


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
    excluded = relatedness.ibs(GenotypeMatrix.from_zarr(path, streaming="always", chunk_bp=20),
                               missing_data="exclude")
    assert np.isnan(excluded[OFF_DIAGONAL]).all()


def _lostruct_hm():
    rng = np.random.default_rng(3)
    return HaplotypeMatrix((rng.random((20, 200)) < 0.3).astype(np.int8),
                           np.arange(200) * 100, 0, 20_000)


@pytest.mark.parametrize("engine,jackknife", [
    ("dense-eigh", False), ("streaming-dense", False), ("dense-eigh", True)])
def test_lostruct_skips_a_window_with_no_local_pca(engine, jackknife):
    # The first region holds one variant, too few for a local PCA, so it has
    # no distance to any window. The MDS uses the other windows, and the
    # invalid window gets a NaN row.
    starts = [0] + list(range(1000, 20_000, 1000))
    regions = pd.DataFrame({"chrom": 1, "start": starts, "end": [100] + [s + 1000 for s in starts[1:]]})
    res = lostruct(_lostruct_hm(), k=2, window_type="regions", regions=regions,
                   window_size=0, step_size=0, n_corners=3, random_state=0,
                   engine=engine, jackknife=jackknife, n_blocks=2)
    assert np.isnan(res.mds[0]).all()
    assert np.isfinite(res.mds[1:]).all()
    ref, evr = pcoa(res.distance[1:, 1:], n_components=2)
    np.testing.assert_allclose(res.mds[1:], ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.explained_variance_ratio, evr, rtol=1e-12)
    assert 0 not in res.corner_indices


def test_lostruct_needs_two_windows_with_a_local_pca():
    # Two of the three regions hold one variant each.
    regions = pd.DataFrame({"chrom": 1, "start": [0, 1000, 5000], "end": [100, 3000, 5100]})
    with pytest.raises(ValueError, match="at least two windows"):
        lostruct(_lostruct_hm(), k=2, window_type="regions", regions=regions,
                 window_size=0, step_size=0, n_corners=0)


def test_lostruct_corners_need_a_two_dimensional_mds():
    # Two valid windows give a one-dimensional MDS, too few for corners.
    regions = pd.DataFrame({"chrom": 1, "start": [0, 1000, 5000, 9000],
                            "end": [100, 3000, 7000, 9100]})
    with pytest.raises(ValueError, match="give 1 MDS dimension"):
        lostruct(_lostruct_hm(), k=2, window_type="regions", regions=regions,
                 window_size=0, step_size=0, n_corners=3)


@pytest.mark.parametrize("n_components", [1, 3])
def test_lostruct_corners_use_the_first_two_mds_axes(n_components):
    regions = pd.DataFrame({"chrom": 1, "start": list(range(0, 20_000, 1000)),
                            "end": list(range(1000, 20_001, 1000))})
    kw = dict(k=2, window_type="regions", regions=regions, window_size=0, step_size=0,
              n_corners=3, n_components=n_components, random_state=0)
    if n_components < 2:
        with pytest.raises(ValueError, match="n_components=1 was requested"):
            lostruct(_lostruct_hm(), **kw)
    else:
        res = lostruct(_lostruct_hm(), **kw)
        assert res.mds.shape[1] == 3
        expected = corners(res.mds[:, :2], prop=0.05, k=3, random_state=0)
        np.testing.assert_array_equal(res.corner_indices, expected)
