"""Small-n early returns of dist_moments.

dist_moments derives variance, skewness, and kurtosis from the condensed
vector of pairwise Hamming distances (length C(n_hap, 2)). It short-circuits
when too few pairs exist to define a higher moment, or when the distances have
no spread; these tests pin each of those returns.
"""

import numpy as np

from pg_gpu import HaplotypeMatrix
from pg_gpu.distance_stats import dist_moments


def _hm(hap):
    hap = np.asarray(hap, dtype=np.int8)
    pos = np.arange(hap.shape[1]) * 100
    return HaplotypeMatrix(hap, pos, 0, hap.shape[1] * 100)


class TestDistMomentsSmallN:

    def test_single_pair_returns_zeros(self):
        # 2 haplotypes -> 1 pairwise distance -> too few to define variance.
        assert dist_moments(_hm([[0, 1, 0], [1, 0, 1]])) == (0.0, 0.0, 0.0)

    def test_zero_variance_returns_zeros(self):
        # Identical haplotypes -> every pairwise distance 0 -> m2 == 0, so
        # skewness and kurtosis are undefined and returned as 0.
        assert dist_moments(_hm([[0, 1], [0, 1], [0, 1]])) == (0.0, 0.0, 0.0)

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
