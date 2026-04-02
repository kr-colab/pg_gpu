"""Tests for pg_gpu.relatedness (GRM and IBS)."""

import numpy as np
import pytest
from pg_gpu import HaplotypeMatrix, relatedness


@pytest.fixture
def small_haplotype_matrix():
    """Small test dataset: 6 haplotypes (3 diploids), 20 variants."""
    np.random.seed(42)
    hap = np.random.randint(0, 2, size=(6, 20)).astype(np.int8)
    positions = np.arange(20) * 1000
    return HaplotypeMatrix(hap, positions)


def _reference_grm(hap):
    """Compute GRM from haplotypes using numpy (reference implementation)."""
    n_ind = hap.shape[0] // 2
    g = hap[:n_ind, :] + hap[n_ind:, :]
    g = g.astype(float)
    p = g.mean(axis=0) / 2
    poly = (p > 0) & (p < 1)
    g_p = g[:, poly]
    p_p = p[poly]
    centered = g_p - 2 * p_p
    scale = np.sqrt(2 * p_p * (1 - p_p))
    std = centered / scale
    return (std @ std.T) / poly.sum()


def _reference_ibs(hap):
    """Compute IBS from haplotypes using numpy (reference implementation)."""
    n_ind = hap.shape[0] // 2
    g = hap[:n_ind, :] + hap[n_ind:, :]
    n_snps = g.shape[1]
    ibs_mat = np.eye(n_ind)
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            ibs_val = (2 - np.abs(g[i] - g[j])).sum() / (2 * n_snps)
            ibs_mat[i, j] = ibs_val
            ibs_mat[j, i] = ibs_val
    return ibs_mat


class TestGRM:
    def test_shape(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        assert grm.shape == (3, 3)

    def test_symmetric(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        np.testing.assert_allclose(grm, grm.T)

    def test_matches_reference(self, small_haplotype_matrix):
        grm_pg = relatedness.grm(small_haplotype_matrix)
        hap = small_haplotype_matrix.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()
        grm_ref = _reference_grm(hap)
        np.testing.assert_allclose(grm_pg, grm_ref, atol=1e-10)

    def test_identical_individuals(self):
        """Two identical individuals should have GRM off-diagonal = diagonal."""
        # pg_gpu layout: [allele1_ind0, allele1_ind1, allele2_ind0, allele2_ind1]
        hap = np.array([[0, 1, 0, 1, 0],   # ind0 allele1
                         [0, 1, 0, 1, 0],   # ind1 allele1 (same as ind0)
                         [1, 0, 1, 0, 1],   # ind0 allele2
                         [1, 0, 1, 0, 1]], dtype=np.int8)  # ind1 allele2 (same)
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        grm = relatedness.grm(hm)
        np.testing.assert_allclose(grm[0, 1], grm[0, 0], atol=1e-10)

    def test_returns_numpy(self, small_haplotype_matrix):
        grm = relatedness.grm(small_haplotype_matrix)
        assert isinstance(grm, np.ndarray)


class TestIBS:
    def test_shape(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert ibs_mat.shape == (3, 3)

    def test_diagonal_is_one(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        np.testing.assert_allclose(ibs_mat.diagonal(), 1.0)

    def test_symmetric(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        np.testing.assert_allclose(ibs_mat, ibs_mat.T)

    def test_range(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert np.all(ibs_mat >= 0)
        assert np.all(ibs_mat <= 1)

    def test_matches_reference(self, small_haplotype_matrix):
        ibs_pg = relatedness.ibs(small_haplotype_matrix)
        hap = small_haplotype_matrix.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()
        ibs_ref = _reference_ibs(hap)
        np.testing.assert_allclose(ibs_pg, ibs_ref, atol=1e-10)

    def test_identical_individuals(self):
        # pg_gpu layout: [allele1_ind0, allele1_ind1, allele2_ind0, allele2_ind1]
        hap = np.array([[0, 1, 0, 1, 0],   # ind0 allele1
                         [0, 1, 0, 1, 0],   # ind1 allele1
                         [1, 0, 1, 0, 1],   # ind0 allele2
                         [1, 0, 1, 0, 1]], dtype=np.int8)  # ind1 allele2
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        ibs_mat = relatedness.ibs(hm)
        assert ibs_mat[0, 1] == 1.0

    def test_completely_different(self):
        # ind0 = 0/0 at all sites, ind1 = 2/2 at all sites
        hap = np.array([[0, 0, 0, 0, 0],   # ind0 allele1
                         [1, 1, 1, 1, 1],   # ind1 allele1
                         [0, 0, 0, 0, 0],   # ind0 allele2
                         [1, 1, 1, 1, 1]], dtype=np.int8)  # ind1 allele2
        hm = HaplotypeMatrix(hap, np.arange(5) * 100)
        ibs_mat = relatedness.ibs(hm)
        assert ibs_mat[0, 1] == 0.0

    def test_returns_numpy(self, small_haplotype_matrix):
        ibs_mat = relatedness.ibs(small_haplotype_matrix)
        assert isinstance(ibs_mat, np.ndarray)
