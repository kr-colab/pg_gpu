"""
Unit tests for selection scan statistics.
"""

import pytest
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix, selection


class TestStandardize:
    """Test standardization utilities."""

    def test_standardize_basic(self):
        score = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = selection.standardize(score)
        assert np.abs(np.nanmean(result)) < 1e-10
        assert np.abs(np.nanstd(result) - 1.0) < 1e-10

    def test_standardize_with_nan(self):
        score = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = selection.standardize(score)
        assert np.isnan(result[1])
        assert np.isnan(result[3])
        valid = result[~np.isnan(result)]
        assert np.abs(np.mean(valid)) < 1e-10

    def test_standardize_by_allele_count(self):
        np.random.seed(42)
        score = np.random.randn(100)
        aac = np.random.randint(1, 20, size=100)
        result, bins = selection.standardize_by_allele_count(
            score, aac, n_bins=5)
        assert result.shape == score.shape
        assert len(bins) >= 2


class TestGarudH:
    """Test Garud's H statistics."""

    def test_garud_h_identical_haplotypes(self):
        """All identical haplotypes: H1 = 1, H12 = 1, H123 = 1."""
        hap = np.zeros((10, 5), dtype=np.int8)
        pos = np.arange(5) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 5000)
        h1, h12, h123, h2_h1 = selection.garud_h(matrix)
        assert np.isclose(h1, 1.0)
        assert np.isclose(h12, 1.0)
        assert np.isclose(h123, 1.0)
        assert np.isclose(h2_h1, 0.0)

    def test_garud_h_all_different(self):
        """Each haplotype unique: H1 = n * (1/n)^2 = 1/n."""
        n = 4
        hap = np.eye(n, dtype=np.int8)
        pos = np.arange(n) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n * 1000)
        h1, h12, h123, h2_h1 = selection.garud_h(matrix)
        assert np.isclose(h1, n * (1.0 / n) ** 2)  # 0.25
        assert np.isclose(h12, (2.0 / n) ** 2 + (n - 2) * (1.0 / n) ** 2)

    def test_garud_h_two_groups(self):
        """Two equally frequent haplotypes."""
        hap = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.int8)
        pos = np.arange(3) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 3000)
        h1, h12, h123, h2_h1 = selection.garud_h(matrix)
        assert np.isclose(h1, 0.5)  # 0.5^2 + 0.5^2
        assert np.isclose(h12, 1.0)  # (0.5 + 0.5)^2
        assert np.isclose(h2_h1, 0.5)


class TestMovingGarudH:
    """Test moving window Garud's H."""

    def test_moving_garud_h_basic(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 100), dtype=np.int8)
        pos = np.arange(100) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 100000)

        h1, h12, h123, h2_h1 = selection.moving_garud_h(matrix, size=20)
        n_windows = (100 - 20) // 20 + 1
        assert h1.shape[0] == n_windows
        assert np.all(h1 >= 0) and np.all(h1 <= 1)
        assert np.all(h12 >= h1)


class TestEHHDecay:
    """Test EHH decay computation."""

    def test_ehh_decay_known(self):
        """Reproduce allel's test case."""
        # allel convention: (n_variants, n_haplotypes)
        # pg_gpu convention: (n_haplotypes, n_variants)
        h_allel = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0]])
        hap = h_allel.T.astype(np.int8)  # (4, 5)
        pos = np.arange(5) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 5000)

        ehh = selection.ehh_decay(matrix)
        expected = np.array([2/6, 2/6, 1/6, 1/6, 0])
        np.testing.assert_array_almost_equal(ehh, expected)

    def test_ehh_decay_identical(self):
        """All identical haplotypes: EHH = 1 everywhere."""
        hap = np.zeros((5, 10), dtype=np.int8)
        pos = np.arange(10) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 10000)
        ehh = selection.ehh_decay(matrix)
        np.testing.assert_array_almost_equal(ehh, np.ones(10))

    def test_ehh_decay_truncate(self):
        """Test truncation of trailing zeros."""
        h_allel = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0]])
        hap = h_allel.T.astype(np.int8)
        pos = np.arange(5) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 5000)
        ehh = selection.ehh_decay(matrix, truncate=True)
        assert ehh[-1] > 0 or len(ehh) < 5

    def test_ehh_decay_returns_full_length_with_zero_tail(self):
        """Issue #94: post-fix ehh_decay must still return shape (n_variants,)
        with exactly zero past the largest shared-prefix length."""
        rng = np.random.default_rng(42)
        n_hap, n_var = 20, 1000
        hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(n_var, dtype=np.int64)
        matrix = HaplotypeMatrix(hap, pos, 0, n_var)
        ehh = selection.ehh_decay(matrix)
        assert ehh.shape == (n_var,)
        nonzero = np.nonzero(ehh)[0]
        if len(nonzero) > 0:
            assert np.all(ehh[nonzero[-1] + 1:] == 0.0)


class TestNSL:
    """Test nSL computation."""

    def test_nsl_output_shape(self):
        """nSL returns correct shape and dtype."""
        np.random.seed(42)
        n_hap, n_var = 20, 100
        hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)

        score = selection.nsl(matrix)
        assert isinstance(score, np.ndarray)
        assert score.shape == (n_var,)
        assert score.dtype == np.float64

    def test_nsl_symmetric_data(self):
        """With balanced alleles, nSL should be near zero."""
        # 4 haps: 2 with allele 0, 2 with allele 1, all constant
        hap = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]], dtype=np.int8)
        pos = np.arange(5) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 5000)
        score = selection.nsl(matrix)
        # log(nsl1/nsl0) should be 0 when both allele classes have same SSL
        valid = score[~np.isnan(score)]
        if len(valid) > 0:
            np.testing.assert_array_almost_equal(valid, 0.0, decimal=10)


class TestXPNSL:
    """Test cross-population nSL."""

    def test_xpnsl_output_shape(self):
        np.random.seed(42)
        n_var = 50
        hap = np.random.randint(0, 2, (20, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000,
                                 sample_sets={'pop1': list(range(10)),
                                              'pop2': list(range(10, 20))})
        score = selection.xpnsl(matrix, 'pop1', 'pop2')
        assert isinstance(score, np.ndarray)
        assert score.shape == (n_var,)


class TestIHS:
    """Test integrated haplotype score."""

    def test_ihs_output_shape(self):
        np.random.seed(42)
        n_hap, n_var = 20, 100
        hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(1, n_var + 1) * 10
        matrix = HaplotypeMatrix(hap, pos, 0, (n_var + 1) * 10)

        score = selection.ihs(matrix)
        assert isinstance(score, np.ndarray)
        assert score.shape == (n_var,)
        assert score.dtype == np.float64

    def test_ihs_known_data(self):
        """Reproduce allel's test_ihs_data: score at core variant = log(5.5/1.5)."""
        hap1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],  # core variant index 9
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        hap2 = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],  # core variant index 9
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        # allel format: (n_variants, n_haplotypes)
        h = np.hstack([hap1, hap2])
        # pg_gpu format: (n_haplotypes, n_variants)
        hap = h.T.astype(np.int8)
        pos = np.arange(1, h.shape[0] + 1)
        matrix = HaplotypeMatrix(hap, pos, 0, h.shape[0] + 1)

        score = selection.ihs(matrix, include_edges=True)
        expected = np.log(5.5 / 1.5)
        assert np.isclose(score[9], expected), \
            f"Expected {expected}, got {score[9]}"


class TestXPEHH:
    """Test cross-population EHH."""

    def test_xpehh_output_shape(self):
        np.random.seed(42)
        n_var = 50
        hap = np.random.randint(0, 2, (20, n_var), dtype=np.int8)
        pos = np.arange(1, n_var + 1) * 10
        matrix = HaplotypeMatrix(hap, pos, 0, (n_var + 1) * 10,
                                 sample_sets={'pop1': list(range(10)),
                                              'pop2': list(range(10, 20))})
        score = selection.xpehh(matrix, 'pop1', 'pop2')
        assert isinstance(score, np.ndarray)
        assert score.shape == (n_var,)

    def test_xpehh_known_data(self):
        """Reproduce allel's test_xpehh_data: score at core = -log(5.5/1.5)."""
        hap1 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        hap2 = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        # pg_gpu: (n_haplotypes, n_variants)
        h1 = hap1.T.astype(np.int8)
        h2 = hap2.T.astype(np.int8)
        n_var = hap1.shape[0]
        pos = np.arange(1, n_var + 1)

        combined = np.vstack([h1, h2])
        matrix = HaplotypeMatrix(
            combined, pos, 0, n_var + 1,
            sample_sets={'pop1': list(range(h1.shape[0])),
                         'pop2': list(range(h1.shape[0], h1.shape[0] + h2.shape[0]))}
        )

        score = selection.xpehh(matrix, 'pop1', 'pop2', include_edges=True)
        expected = -np.log(5.5 / 1.5)
        assert np.isclose(score[9], expected, rtol=1e-5), \
            f"Expected {expected}, got {score[9]}"
