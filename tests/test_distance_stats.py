"""dist_moments: the values, and its rules for small or flat samples.

dist_moments derives variance, skewness, and kurtosis from the pairwise
Hamming distances (C(n, 2) of them). A moment with too few pairs, or
distances with no spread, has a fixed value; the first tests pin each of
those. The rest compare the values with scipy on the same distances.
"""

import numpy as np
import pytest
from scipy import stats

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.distance_stats import (
    dist_moments, pairwise_diffs, pairwise_diffs_diploid, pairwise_diffs_haploid,
)


def _hm(hap):
    hap = np.asarray(hap, dtype=np.int8)
    pos = np.arange(hap.shape[1]) * 100
    return HaplotypeMatrix(hap, pos, 0, hap.shape[1] * 100)


class TestDistMomentsSmallN:

    def test_single_pair_returns_zeros(self):
        # 2 haplotypes -> 1 pairwise distance -> too few to define variance.
        assert dist_moments(_hm([[0, 1, 0], [1, 0, 1]])) == (0.0, 0.0, 0.0)

    def test_zero_variance_returns_zeros(self):
        # Identical haplotypes -> every pairwise distance 0 -> no spread, so
        # skewness and kurtosis are undefined and returned as 0.
        assert dist_moments(_hm([[0, 1], [0, 1], [0, 1]])) == (0.0, 0.0, 0.0)

    def test_equal_distances_that_the_mean_rounds_return_zeros(self):
        # Three alleles at one site of ten: every pair differs there, so
        # all three distances are 0.1. (0.1 + 0.1 + 0.1) / 3 is not 0.1 in
        # floating point, so a test on the variance would see a tiny
        # spread and give a skewness of rounding noise.
        hap = np.zeros((3, 10), dtype=np.int8)
        hap[:, 0] = [0, 1, 2]
        diffs = pairwise_diffs(_hm(hap))
        assert diffs.tolist() == [0.1, 0.1, 0.1]
        assert np.mean(diffs) != 0.1
        assert dist_moments(_hm(hap)) == (0.0, 0.0, 0.0)

    def test_three_haplotypes_skip_kurtosis(self):
        # 3 haplotypes -> 3 distances with spread -> variance and skewness
        # defined, but n < 4 leaves kurtosis at 0.
        var, skew, kurt = dist_moments(
            _hm([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]]))
        assert var > 0.0
        assert kurt == 0.0

    def test_four_haplotypes_full_moments(self):
        # >= 4 haplotypes -> 6 distances -> all three moments computed.
        var, skew, kurt = dist_moments(
            _hm([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]))
        assert var > 0.0
        assert np.isfinite(skew) and np.isfinite(kurt)


# ── the moments against scipy on the same distances ───────────────────
def _host_moments(diffs):
    """The moments of the defined distances, by scipy, with the small-sample
    rules of dist_moments."""
    d = diffs[np.isfinite(diffs)]
    fill = 0.0 if d.size == diffs.size else np.nan
    same = d.size >= 2 and np.ptp(d) == 0
    var = fill if d.size < 2 else (0.0 if same else np.var(d, ddof=1))
    skew = fill if d.size < 3 else (0.0 if same else stats.skew(d))
    kurt = fill if d.size < 4 else (0.0 if same else stats.kurtosis(d))
    return var, skew, kurt


def _random_matrix(kind, n_rows, missing, seed):
    rng = np.random.default_rng(seed)
    high = 3 if kind == "genotypes" else 2
    calls = rng.integers(0, high, (n_rows, 60)).astype(np.int8)
    calls[rng.random(calls.shape) < missing] = -1
    pos = np.arange(60) * 100 + 1
    if kind == "genotypes":
        return GenotypeMatrix(calls, pos, 0, 6000)
    return HaplotypeMatrix(calls, pos, 0, 6000, phased=True)


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("missing", [0.0, 0.1, 0.6], ids=["complete", "some-missing", "sparse"])
@pytest.mark.parametrize("n_rows", [2, 3, 4, 5, 40])
@pytest.mark.parametrize("kind", ["haplotypes", "genotypes"])
def test_moments_equal_the_host_formulas(kind, n_rows, missing, missing_data):
    # The moments are reduced on the device from the same condensed
    # distances that pairwise_diffs returns to the host.
    matrix = _random_matrix(kind, n_rows, missing, seed=n_rows)
    diffs = pairwise_diffs(matrix, missing_data=missing_data)
    assert isinstance(diffs, np.ndarray) and diffs.shape == (n_rows * (n_rows - 1) // 2,)
    np.testing.assert_allclose(dist_moments(matrix, missing_data=missing_data),
                               _host_moments(diffs), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_moments_of_a_population():
    matrix = _random_matrix("haplotypes", 30, 0.05, seed=1)
    matrix.sample_sets = {"p": list(range(5, 25))}
    rows = HaplotypeMatrix(matrix.haplotypes[5:25].copy(), matrix.positions, 0, 6000, phased=True)
    np.testing.assert_allclose(dist_moments(matrix, population="p"), dist_moments(rows), rtol=1e-12)


@pytest.mark.parametrize("function", [pairwise_diffs_haploid, pairwise_diffs_diploid],
                         ids=lambda f: f.__name__)
def test_distance_functions_reject_the_other_matrix_class(function):
    # Haplotype rows read as individuals, or the reverse, would give
    # distances of the wrong thing with no error.
    wrong = "genotypes" if function is pairwise_diffs_haploid else "haplotypes"
    with pytest.raises(TypeError, match="expected a"):
        function(_random_matrix(wrong, 6, 0.0, seed=0))

