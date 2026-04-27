"""
Tests for diploSHIC-derived statistics: GenotypeMatrix, distance moments,
ZnS, Omega, theta_h, mu stats, diploid variants.
"""

import pytest
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix, GenotypeMatrix
from pg_gpu import ld_statistics, diversity, selection, distance_stats


@pytest.fixture
def hap_data():
    np.random.seed(42)
    n_hap, n_var = 40, 100
    hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    return HaplotypeMatrix(hap, pos, 0, n_var * 1000)


@pytest.fixture
def geno_data(hap_data):
    return GenotypeMatrix.from_haplotype_matrix(hap_data)


# ---------------------------------------------------------------------------
# GenotypeMatrix
# ---------------------------------------------------------------------------

class TestGenotypeMatrix:

    def test_from_haplotype_matrix(self, hap_data):
        gm = GenotypeMatrix.from_haplotype_matrix(hap_data)
        assert gm.num_individuals == 20  # 40 haplotypes / 2
        assert gm.num_variants == 100
        # values should be 0, 1, or 2
        gm_cpu = gm.genotypes if isinstance(gm.genotypes, np.ndarray) else gm.genotypes.get()
        assert np.all((gm_cpu >= 0) & (gm_cpu <= 2))

    def test_gpu_transfer(self, geno_data):
        geno_data.transfer_to_gpu()
        assert geno_data.device == 'GPU'
        assert isinstance(geno_data.genotypes, cp.ndarray)
        geno_data.transfer_to_cpu()
        assert geno_data.device == 'CPU'
        assert isinstance(geno_data.genotypes, np.ndarray)

    def test_roundtrip(self, hap_data):
        """HaplotypeMatrix -> GenotypeMatrix -> HaplotypeMatrix preserves allele counts."""
        gm = GenotypeMatrix.from_haplotype_matrix(hap_data)
        hm2 = gm.to_haplotype_matrix()
        # per-variant allele counts should match
        hap1 = hap_data.haplotypes
        hap2 = hm2.haplotypes
        if isinstance(hap1, cp.ndarray):
            hap1 = hap1.get()
        if isinstance(hap2, cp.ndarray):
            hap2 = hap2.get()
        dac1 = np.sum(hap1, axis=0)
        dac2 = np.sum(hap2, axis=0)
        np.testing.assert_array_equal(dac1, dac2)

    def test_shape(self, geno_data):
        assert geno_data.shape == (20, 100)


# ---------------------------------------------------------------------------
# Pairwise Distance Distribution
# ---------------------------------------------------------------------------

class TestDistanceStats:

    def test_pairwise_diffs_haploid_shape(self, hap_data):
        diffs = distance_stats.pairwise_diffs_haploid(hap_data)
        n = 40
        expected_pairs = n * (n - 1) // 2
        assert diffs.shape == (expected_pairs,)

    def test_pairwise_diffs_diploid_shape(self, geno_data):
        diffs = distance_stats.pairwise_diffs_diploid(geno_data)
        n = 20
        expected_pairs = n * (n - 1) // 2
        assert diffs.shape == (expected_pairs,)

    def test_pairwise_diffs_identical(self):
        """Identical haplotypes should have zero distance."""
        hap = np.zeros((4, 10), dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(10) * 100, 0, 1000)
        diffs = distance_stats.pairwise_diffs_haploid(matrix)
        np.testing.assert_array_almost_equal(np.asarray(diffs), 0.0)

    def test_dist_var(self, hap_data):
        v = distance_stats.dist_var(hap_data)
        assert isinstance(v, float)
        assert v >= 0

    def test_dist_skew(self, hap_data):
        s = distance_stats.dist_skew(hap_data)
        assert isinstance(s, float)

    def test_dist_kurt(self, hap_data):
        k = distance_stats.dist_kurt(hap_data)
        assert isinstance(k, float)

    def test_dist_moments_diploid(self, geno_data):
        v = distance_stats.dist_var(geno_data)
        s = distance_stats.dist_skew(geno_data)
        k = distance_stats.dist_kurt(geno_data)
        assert isinstance(v, float)
        assert isinstance(s, float)
        assert isinstance(k, float)


# ---------------------------------------------------------------------------
# ZnS and Omega
# ---------------------------------------------------------------------------

class TestZnSOmega:

    def test_zns_range(self, hap_data):
        r2 = hap_data.pairwise_r2()
        z = ld_statistics.zns(r2)
        assert 0 <= z <= 1

    def test_zns_perfect_ld(self):
        """All identical haplotypes: r2 matrix is all 0 (no variation), ZnS=0."""
        hap = np.zeros((10, 5), dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(5) * 100, 0, 500)
        r2 = matrix.pairwise_r2()
        z = ld_statistics.zns(r2)
        assert z == 0.0

    def test_omega_range(self, hap_data):
        r2 = hap_data.pairwise_r2()
        o = ld_statistics.omega(r2)
        assert o >= 0

    def test_omega_small_matrix(self):
        """Omega requires at least 4 SNPs."""
        r2 = cp.zeros((3, 3), dtype=cp.float64)
        assert ld_statistics.omega(r2) == 0.0

    def test_zns_diploid(self, geno_data):
        z = ld_statistics.zns_diploid(geno_data)
        assert 0 <= z <= 1

    def test_omega_diploid(self, geno_data):
        o = ld_statistics.omega_diploid(geno_data)
        assert o >= 0

    def test_zns_default_is_sigma_d2_for_haplotype_matrix(self, hap_data):
        # Default 'auto' should resolve to sigma_d2 for HaplotypeMatrix
        # inputs, matching an explicit estimator='sigma_d2' call.
        z_default = ld_statistics.zns(hap_data)
        z_sigma_d2 = ld_statistics.zns(hap_data, estimator='sigma_d2')
        z_r2 = ld_statistics.zns(hap_data, estimator='r2')
        assert z_default == z_sigma_d2
        # The unbiased estimator is generally smaller than naive r²,
        # but at minimum the two are not numerically identical for any
        # realistic dataset (they differ in their finite-sample bias).
        assert z_default != z_r2

    def test_zns_default_falls_back_to_r2_for_array(self, hap_data):
        # Pre-computed r² array: 'auto' must fall back to 'r2' since
        # sigma_d2 needs the haplotype data.
        r2 = hap_data.pairwise_r2()
        z_auto = ld_statistics.zns(r2)
        z_r2 = ld_statistics.zns(r2, estimator='r2')
        assert z_auto == z_r2

    def test_zns_sigma_d2_requires_haplotype_matrix(self, hap_data):
        r2 = hap_data.pairwise_r2()
        with pytest.raises(ValueError, match="HaplotypeMatrix"):
            ld_statistics.zns(r2, estimator='sigma_d2')

    def test_omega_default_is_sigma_d2_for_haplotype_matrix(self, hap_data):
        o_default = ld_statistics.omega(hap_data)
        o_sigma_d2 = ld_statistics.omega(hap_data, estimator='sigma_d2')
        assert o_default == o_sigma_d2

    def test_zns_unknown_estimator_raises(self, hap_data):
        with pytest.raises(ValueError, match="Unknown estimator"):
            ld_statistics.zns(hap_data, estimator='nonsense')


# ---------------------------------------------------------------------------
# mu_ld
# ---------------------------------------------------------------------------

class TestMuLD:

    def test_mu_ld_range(self, hap_data):
        ml = ld_statistics.mu_ld(hap_data)
        assert 0 <= ml <= 1

    def test_mu_ld_identical(self):
        """All identical: one pattern per half, exclusive, mu_ld=1."""
        hap = np.zeros((10, 10), dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(10) * 100, 0, 1000)
        ml = ld_statistics.mu_ld(matrix)
        assert ml == 1.0


# ---------------------------------------------------------------------------
# Diversity: theta_h, max_daf, haplotype_count, daf_histogram
# ---------------------------------------------------------------------------

class TestDiversityNewStats:

    def test_theta_h(self, hap_data):
        th = diversity.theta_h(hap_data)
        assert isinstance(th, float)
        assert th >= 0

    def test_theta_h_monomorphic(self):
        """No variation: theta_h = 0."""
        hap = np.zeros((10, 5), dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(5) * 100, 0, 500)
        th = diversity.theta_h(matrix)
        assert th == 0.0

    def test_max_daf(self, hap_data):
        md = diversity.max_daf(hap_data)
        assert 0 <= md <= 1

    def test_haplotype_count(self, hap_data):
        hc = diversity.haplotype_count(hap_data)
        assert 1 <= hc <= 40

    def test_haplotype_count_identical(self):
        hap = np.zeros((10, 5), dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(5) * 100, 0, 500)
        assert diversity.haplotype_count(matrix) == 1

    def test_daf_histogram(self, hap_data):
        hist, edges = diversity.daf_histogram(hap_data, n_bins=20)
        assert hist.shape == (20,)
        assert edges.shape == (21,)
        np.testing.assert_allclose(hist.sum(), 1.0, atol=1e-10)

    def test_daf_histogram_diploid(self, geno_data):
        hist, edges = diversity.daf_histogram_diploid(geno_data, n_bins=10)
        assert hist.shape == (10,)
        np.testing.assert_allclose(hist.sum(), 1.0, atol=1e-10)

    def test_mu_var(self, hap_data):
        mv = diversity.mu_var(hap_data)
        assert mv > 0

    def test_mu_sfs(self, hap_data):
        ms = diversity.mu_sfs(hap_data)
        assert 0 <= ms <= 1


# ---------------------------------------------------------------------------
# Diploid Garud's H
# ---------------------------------------------------------------------------

class TestDiploidGarudH:

    def test_garud_h_diploid(self, geno_data):
        h1, h12, h123, h2_h1 = selection.garud_h_diploid(geno_data)
        assert 0 <= h1 <= 1
        assert h12 >= h1
        assert 0 <= h2_h1 <= 1

    def test_garud_h_diploid_identical(self):
        """All identical diplotypes: H1=1, H12=1, H2/H1=0."""
        geno = np.zeros((5, 10), dtype=np.int8)
        pos = np.arange(10) * 100
        gm = GenotypeMatrix(geno, pos, 0, 1000)
        h1, h12, h123, h2_h1 = selection.garud_h_diploid(gm)
        assert np.isclose(h1, 1.0)
        assert np.isclose(h12, 1.0)
        assert np.isclose(h2_h1, 0.0)


# ---------------------------------------------------------------------------
# Diplotype frequency spectrum
# ---------------------------------------------------------------------------

class TestDiplotypeFreqSpec:

    def test_diplotype_freq_spec(self, geno_data):
        freqs, n_d = diversity.diplotype_frequency_spectrum(geno_data)
        assert n_d >= 1
        assert len(freqs) == n_d
        np.testing.assert_allclose(freqs.sum(), 1.0, atol=1e-10)
        # sorted descending
        assert np.all(freqs[:-1] >= freqs[1:])

    def test_diplotype_freq_spec_identical(self):
        geno = np.ones((5, 10), dtype=np.int8)
        pos = np.arange(10) * 100
        gm = GenotypeMatrix(geno, pos, 0, 1000)
        freqs, n_d = diversity.diplotype_frequency_spectrum(gm)
        assert n_d == 1
        assert freqs[0] == 1.0
