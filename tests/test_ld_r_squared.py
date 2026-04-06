"""
Tests for r and r-squared LD statistics and integration functions.
"""

import pytest
import numpy as np
import cupy as cp
import allel
from pg_gpu import HaplotypeMatrix, ld_statistics


class TestRSquaredFromCounts:
    """Test r and r_squared functions in ld_statistics module."""

    def test_r_squared_perfect_ld(self):
        """Two perfectly correlated loci: r²=1."""
        # 10 haplotypes, loci A and B identical
        # n11=5, n10=0, n01=0, n00=5 -> D = 5*5/(10²) = 0.25
        # p_A=0.5, p_B=0.5, denom=0.0625, r²=1.0
        counts = cp.array([[5, 0, 0, 5]], dtype=cp.int32)
        r2 = ld_statistics.r_squared(counts)
        np.testing.assert_allclose(r2.get(), [1.0], rtol=1e-10)

    def test_r_squared_no_ld(self):
        """Independent loci: r²=0."""
        # p_A=0.5, p_B=0.5, all 4 haplotypes equally frequent
        # n11=25, n10=25, n01=25, n00=25
        counts = cp.array([[25, 25, 25, 25]], dtype=cp.int32)
        r2 = ld_statistics.r_squared(counts)
        np.testing.assert_allclose(r2.get(), [0.0], atol=1e-10)

    def test_r_negative_correlation(self):
        """Perfectly anti-correlated: r=-1, r²=1."""
        # n11=0, n10=5, n01=5, n00=0
        counts = cp.array([[0, 5, 5, 0]], dtype=cp.int32)
        r_val = ld_statistics.r(counts)
        r2_val = ld_statistics.r_squared(counts)
        np.testing.assert_allclose(r_val.get(), [-1.0], rtol=1e-10)
        np.testing.assert_allclose(r2_val.get(), [1.0], rtol=1e-10)

    def test_r_squared_monomorphic(self):
        """Monomorphic at one locus: r² undefined (NaN)."""
        # p_A = 1.0, p_B = 0.5 -> denom has p_A*(1-p_A)=0
        counts = cp.array([[5, 5, 0, 0]], dtype=cp.int32)
        r2 = ld_statistics.r_squared(counts)
        assert np.isnan(r2.get()[0])

    def test_r_squared_batch(self):
        """Multiple pairs computed in parallel."""
        counts = cp.array([
            [5, 0, 0, 5],    # perfect LD
            [25, 25, 25, 25], # no LD
            [0, 5, 5, 0],    # perfect anti-LD
        ], dtype=cp.int32)
        r2 = ld_statistics.r_squared(counts)
        expected = [1.0, 0.0, 1.0]
        np.testing.assert_allclose(r2.get(), expected, atol=1e-10)


class TestRSquaredVsAllel:
    """Validate r² against scikit-allel's rogers_huff_r."""

    def test_r_squared_vs_pairwise_r2(self):
        """Verify r_squared from counts matches pairwise_r2 matrix method."""
        np.random.seed(42)
        n_hap = 40
        n_var = 20
        hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)

        # method 1: pairwise_r2 matrix (uses freq-based formula)
        r2_mat = matrix.pairwise_r2().get()
        idx_i, idx_j = np.triu_indices(n_var, k=1)
        r2_from_mat = r2_mat[idx_i, idx_j]

        # method 2: r_squared from haplotype counts
        counts, n_valid = matrix.tally_gpu_haplotypes()
        r2_from_counts = ld_statistics.r_squared(counts, n_valid=n_valid).get()

        valid = ~np.isnan(r2_from_counts)
        np.testing.assert_allclose(
            r2_from_counts[valid], r2_from_mat[valid], rtol=1e-5,
            err_msg="r_squared from counts does not match pairwise_r2 matrix"
        )


class TestPairwiseR2:
    """Test existing pairwise_r2 method on HaplotypeMatrix."""

    def test_pairwise_r2_shape(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (10, 20), dtype=np.int8)
        pos = np.arange(20) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 20000)
        r2 = matrix.pairwise_r2().get()
        assert r2.shape == (20, 20)
        # diagonal should be 0
        np.testing.assert_array_almost_equal(np.diag(r2), 0.0)

    def test_pairwise_r2_consistency(self):
        """pairwise_r2 matrix should match ld_statistics.r_squared."""
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 10), dtype=np.int8)
        pos = np.arange(10) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 10000)

        r2_mat = matrix.pairwise_r2().get()
        counts, n_valid = matrix.tally_gpu_haplotypes()
        r2_flat = ld_statistics.r_squared(counts, n_valid=n_valid).get()

        # extract upper triangle from matrix
        idx_i, idx_j = np.triu_indices(10, k=1)
        r2_from_mat = r2_mat[idx_i, idx_j]

        valid = ~np.isnan(r2_flat)
        np.testing.assert_allclose(
            r2_from_mat[valid], r2_flat[valid], rtol=1e-5)


class TestLocateUnlinked:
    """Test LD pruning function."""

    def test_locate_unlinked_basic(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 50), dtype=np.int8)
        pos = np.arange(50) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 50000)

        loc = matrix.locate_unlinked(size=10, step=5, threshold=0.1)
        assert loc.shape == (50,)
        assert loc.dtype == bool
        assert np.sum(loc) > 0  # should keep some variants
        assert np.sum(loc) <= 50

    def test_locate_unlinked_high_threshold(self):
        """With threshold=1.0, all variants should be kept."""
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 20), dtype=np.int8)
        pos = np.arange(20) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 20000)

        loc = matrix.locate_unlinked(size=10, step=5, threshold=1.0)
        assert np.all(loc)


class TestWindowedRSquared:
    """Test windowed r-squared computation."""

    def test_windowed_r_squared_basic(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 50), dtype=np.int8)
        pos = np.arange(50) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 50000)

        bp_bins = [0, 10000, 20000, 50000]
        result, counts = matrix.windowed_r_squared(bp_bins)

        assert result.shape == (3,)  # 3 bins
        assert counts.shape == (3,)
        assert np.sum(counts) > 0
        # r² values should be in [0, 1] where not NaN
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)
        assert np.all(result[valid] <= 1)


class TestDPrime:
    """Test Lewontin's D' statistic."""

    def test_d_prime_perfect_ld(self):
        """Perfect positive LD: D' = 1."""
        counts = cp.array([[5, 0, 0, 5]], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        np.testing.assert_allclose(dp.get(), [1.0], rtol=1e-10)

    def test_d_prime_perfect_negative_ld(self):
        """Perfect negative LD: D' = -1."""
        counts = cp.array([[0, 5, 5, 0]], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        np.testing.assert_allclose(dp.get(), [-1.0], rtol=1e-10)

    def test_d_prime_no_ld(self):
        """Independent loci: D' = 0."""
        counts = cp.array([[25, 25, 25, 25]], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        np.testing.assert_allclose(dp.get(), [0.0], atol=1e-10)

    def test_d_prime_range(self):
        """D' should be in [-1, 1]."""
        np.random.seed(42)
        n_hap = 40
        n_var = 20
        hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)
        counts, n_valid = matrix.tally_gpu_haplotypes()
        dp = ld_statistics.d_prime(counts, n_valid=n_valid).get()
        valid = ~np.isnan(dp)
        assert np.all(dp[valid] >= -1.0 - 1e-10)
        assert np.all(dp[valid] <= 1.0 + 1e-10)

    def test_d_prime_monomorphic_is_nan(self):
        """Monomorphic at one locus: D' undefined."""
        counts = cp.array([[5, 5, 0, 0]], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        assert np.isnan(dp.get()[0])

    def test_d_prime_batch(self):
        """Multiple pairs computed in parallel."""
        counts = cp.array([
            [5, 0, 0, 5],     # perfect positive
            [25, 25, 25, 25],  # independent
            [0, 5, 5, 0],     # perfect negative
        ], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        expected = [1.0, 0.0, -1.0]
        np.testing.assert_allclose(dp.get(), expected, atol=1e-10)

    def test_d_prime_asymmetric_frequencies(self):
        """D' with unequal allele frequencies."""
        # p_A=0.8, p_B=0.3, D>0
        # n11=28, n10=52, n01=2, n00=18 -> n=100
        # p_A=0.8, q_A=0.2, p_B=0.3, q_B=0.7
        # D = (28*18 - 52*2)/10000 = (504-104)/10000 = 0.04
        # D_max = min(0.8*0.7, 0.2*0.3) = min(0.56, 0.06) = 0.06
        # D' = 0.04/0.06 = 2/3
        counts = cp.array([[28, 52, 2, 18]], dtype=cp.int32)
        dp = ld_statistics.d_prime(counts)
        np.testing.assert_allclose(dp.get(), [2.0 / 3.0], rtol=1e-10)
