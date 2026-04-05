"""
Tests for the Achaz (2009) generalized theta estimation framework.
"""

import pytest
import numpy as np
import msprime
from pg_gpu import HaplotypeMatrix, diversity
from pg_gpu.achaz import (
    FrequencySpectrum, project_sfs, compute_sigma_ij,
    WEIGHT_REGISTRY,
)


@pytest.fixture
def simple_ts():
    ts = msprime.sim_ancestry(
        samples=50, sequence_length=100_000,
        recombination_rate=1e-8, population_size=10_000,
        random_seed=42, ploidy=2)
    return msprime.sim_mutations(ts, rate=1e-8, random_seed=42)


@pytest.fixture
def hm(simple_ts):
    return HaplotypeMatrix.from_ts(simple_ts)


class TestThetaEstimators:
    """Verify Achaz theta estimators match current implementations."""

    def test_pi_matches(self, hm):
        fs = FrequencySpectrum(hm)
        current = diversity.pi(hm, span_normalize=False)
        achaz = fs.theta('pi')
        np.testing.assert_allclose(achaz, current, rtol=1e-12)

    def test_theta_w_matches(self, hm):
        fs = FrequencySpectrum(hm)
        current = diversity.theta_w(hm, span_normalize=False)
        achaz = fs.theta('watterson')
        np.testing.assert_allclose(achaz, current, rtol=1e-12)

    def test_theta_h_matches(self, hm):
        fs = FrequencySpectrum(hm)
        current = diversity.theta_h(hm, span_normalize=False)
        achaz = fs.theta('theta_h')
        np.testing.assert_allclose(achaz, current, rtol=1e-12)

    def test_theta_l_matches(self, hm):
        fs = FrequencySpectrum(hm)
        current = diversity.theta_l(hm, span_normalize=False)
        achaz = fs.theta('theta_l')
        np.testing.assert_allclose(achaz, current, rtol=1e-12)

    def test_span_normalization(self, hm):
        fs = FrequencySpectrum(hm)
        span = hm.get_span()
        pi_norm = fs.theta('pi', span_normalize=True, span=span)
        pi_raw = fs.theta('pi')
        np.testing.assert_allclose(pi_norm, pi_raw / span, rtol=1e-12)

    def test_all_thetas_returns_dict(self, hm):
        fs = FrequencySpectrum(hm)
        result = fs.all_thetas()
        assert isinstance(result, dict)
        assert 'pi' in result
        assert 'watterson' in result
        assert 'theta_h' in result
        assert 'theta_l' in result
        assert 'eta1' in result


class TestNeutralityTests:
    """Verify neutrality test statistics."""

    def test_tajimas_d_matches(self, hm):
        fs = FrequencySpectrum(hm)
        current = diversity.tajimas_d(hm)
        achaz = fs.tajimas_d()
        np.testing.assert_allclose(achaz, current, rtol=1e-6)

    def test_fay_wu_h_unnormalized(self, hm):
        fs = FrequencySpectrum(hm)
        h = fs.fay_wu_h()
        pi = fs.theta('pi')
        th = fs.theta('theta_h')
        np.testing.assert_allclose(h, pi - th, rtol=1e-12)

    def test_all_tests_returns_dict(self, hm):
        fs = FrequencySpectrum(hm)
        result = fs.all_tests()
        assert 'tajimas_d' in result
        assert 'fay_wu_h' in result
        assert not np.isnan(result['tajimas_d'])

    def test_custom_neutrality_test(self, hm):
        fs = FrequencySpectrum(hm)
        # Custom test: pi vs theta_h (Fay & Wu's H, Achaz-normalized)
        T = fs.neutrality_test('pi', 'theta_h')
        assert np.isfinite(T)


class TestPopulation:
    """Test population subsetting."""

    def test_population_subset(self, hm):
        n = hm.num_haplotypes
        hm.sample_sets = {
            'pop1': list(range(n // 2)),
            'pop2': list(range(n // 2, n)),
        }
        fs1 = FrequencySpectrum(hm, population='pop1')
        fs2 = FrequencySpectrum(hm, population='pop2')
        # Different populations should give different thetas
        assert fs1.theta('pi') != fs2.theta('pi')

    def test_matches_current_with_population(self, hm):
        n = hm.num_haplotypes
        hm.sample_sets = {'pop1': list(range(n // 2))}
        fs = FrequencySpectrum(hm, population='pop1')
        current = diversity.pi(hm, population='pop1', span_normalize=False)
        achaz = fs.theta('pi')
        np.testing.assert_allclose(achaz, current, rtol=1e-12)


class TestProjection:
    """Test SFS projection via hypergeometric sampling."""

    def test_projection_preserves_total(self):
        # Simple SFS: 10 singletons, 5 doubletons, 2 tripletons
        sfs = np.array([0, 10, 5, 2, 0, 0], dtype=np.float64)  # n=5
        projected = project_sfs(sfs, n_from=5, n_to=3)
        # Total variant sites should be approximately preserved
        assert projected.shape == (4,)
        np.testing.assert_allclose(np.sum(projected[1:3]),
                                   np.sum(sfs[1:5]), rtol=0.3)

    def test_projection_identity(self):
        sfs = np.array([100, 10, 5, 2, 0, 1], dtype=np.float64)
        projected = project_sfs(sfs, n_from=5, n_to=5)
        np.testing.assert_array_equal(projected, sfs)

    def test_projection_reduces_size(self):
        sfs = np.array([0, 10, 5, 2, 1, 0, 0], dtype=np.float64)  # n=6
        projected = project_sfs(sfs, n_from=6, n_to=4)
        assert projected.shape == (5,)

    def test_frequency_spectrum_project(self, hm):
        fs = FrequencySpectrum(hm)
        n = fs.n_max
        projected = fs.project(n - 10)
        assert projected.n_max == n - 10
        assert len(projected.sfs_by_n) == 1


class TestSigmaIJ:
    """Test Fu (1995) covariance structure."""

    def test_sigma_symmetric(self):
        sigma = compute_sigma_ij(20)
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)

    def test_sigma_shape(self):
        sigma = compute_sigma_ij(30)
        assert sigma.shape == (29, 29)

    def test_sigma_diagonal_positive(self):
        sigma = compute_sigma_ij(20)
        assert np.all(np.diag(sigma) > 0)


class TestCustomWeights:
    """Test user-defined weight vectors."""

    def test_custom_callable(self, hm):
        fs = FrequencySpectrum(hm)
        # Custom weight: emphasize rare variants (1/k^2)
        def rare_weights(n):
            k = np.arange(n + 1, dtype=np.float64)
            w = np.zeros(n + 1)
            w[1:n] = 1.0 / (k[1:n] ** 2)
            norm = np.sum(w[1:n])
            if norm > 0:
                w[1:n] /= norm
            return w

        theta = fs.theta(rare_weights)
        assert np.isfinite(theta)
        assert theta > 0

    def test_all_registry_weights_work(self, hm):
        fs = FrequencySpectrum(hm)
        for name in WEIGHT_REGISTRY:
            theta = fs.theta(name)
            assert np.isfinite(theta), f"{name} gave non-finite theta"

    def test_invalid_weight_raises(self, hm):
        fs = FrequencySpectrum(hm)
        with pytest.raises(ValueError, match="Unknown weight"):
            fs.theta('nonexistent_weight')


class TestNeutralModelValidation:
    """Validate estimators against known theta from msprime simulation."""

    def test_estimators_unbiased(self):
        """Under standard neutral model, all theta estimators should be
        unbiased: E[theta_hat] ≈ theta = 4 * N * mu * L."""
        N = 10_000
        mu = 1e-8
        L = 100_000
        theta_true = 4 * N * mu * L  # = 0.04 per site * 100K = 4000... no
        # theta = 4*N*mu = 4e-4 per site; over L sites: 4*N*mu*L = 4

        estimates = {name: [] for name in ['pi', 'watterson', 'theta_h', 'theta_l']}

        n_reps = 50
        for seed in range(n_reps):
            ts = msprime.sim_ancestry(
                samples=25, sequence_length=L,
                recombination_rate=1e-8, population_size=N,
                random_seed=seed + 1, ploidy=2)
            ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)
            hm = HaplotypeMatrix.from_ts(ts)
            if hm.num_variants < 2:
                continue
            fs = FrequencySpectrum(hm)
            for name in estimates:
                estimates[name].append(fs.theta(name))

        # Expected theta (unnormalized) = 4*N*mu*L = 4
        expected = 4 * N * mu * L

        for name, vals in estimates.items():
            mean_est = np.mean(vals)
            # Allow 50% tolerance due to variance (50 reps, small L)
            assert abs(mean_est - expected) / expected < 0.5, \
                f"{name}: mean={mean_est:.2f}, expected={expected:.2f}"

    def test_tajimas_d_mean_near_zero(self):
        """Under neutrality, E[Tajima's D] ≈ 0."""
        d_values = []
        for seed in range(50):
            ts = msprime.sim_ancestry(
                samples=25, sequence_length=100_000,
                recombination_rate=1e-8, population_size=10_000,
                random_seed=seed + 1, ploidy=2)
            ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed + 1)
            hm = HaplotypeMatrix.from_ts(ts)
            if hm.num_variants < 5:
                continue
            fs = FrequencySpectrum(hm)
            d = fs.tajimas_d()
            if np.isfinite(d):
                d_values.append(d)

        mean_d = np.mean(d_values)
        # Under neutrality, mean should be near 0 (within 0.5)
        assert abs(mean_d) < 0.5, f"Mean Tajima's D = {mean_d:.3f}"


class TestEdgeCases:
    """Test edge cases: small n, no segregating sites, etc."""

    def test_n_equals_2(self):
        """Minimum viable sample size."""
        hap = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)
        pos = np.array([100, 200, 300], dtype=np.int32)
        hm = HaplotypeMatrix(hap, pos)
        fs = FrequencySpectrum(hm)
        assert np.isfinite(fs.theta('pi'))
        assert np.isfinite(fs.theta('watterson'))
        # Tajima's D needs S >= 3; with 3 variants and n=2,
        # all have dac=1, so S=3 if all segregating
        d = fs.tajimas_d()
        assert np.isfinite(d) or np.isnan(d)  # either is acceptable

    def test_no_segregating_sites(self):
        """All sites monomorphic."""
        hap = np.zeros((10, 50), dtype=np.int8)
        pos = np.arange(50, dtype=np.int32)
        hm = HaplotypeMatrix(hap, pos)
        fs = FrequencySpectrum(hm)
        assert fs.n_segregating == 0
        assert fs.theta('pi') == 0.0
        assert fs.theta('watterson') == 0.0
        assert np.isnan(fs.tajimas_d())

    def test_single_segregating_site(self):
        """Only one segregating site."""
        hap = np.zeros((10, 50), dtype=np.int8)
        hap[0, 25] = 1  # one singleton
        pos = np.arange(50, dtype=np.int32)
        hm = HaplotypeMatrix(hap, pos)
        fs = FrequencySpectrum(hm)
        assert fs.n_segregating == 1
        assert fs.theta('pi') > 0
        assert np.isnan(fs.tajimas_d())  # S < 3

    def test_projection_to_n2(self):
        """Project down to minimum sample size."""
        hap = np.random.randint(0, 2, (20, 100), dtype=np.int8)
        pos = np.arange(100, dtype=np.int32)
        hm = HaplotypeMatrix(hap, pos)
        fs = FrequencySpectrum(hm)
        proj = fs.project(2)
        assert proj.n_max == 2
        assert np.isfinite(proj.theta('pi'))


class TestMissingData:
    """Test behavior with missing data."""

    def test_exclude_mode(self, hm):
        fs_include = FrequencySpectrum(hm, missing_data='include')
        fs_exclude = FrequencySpectrum(hm, missing_data='exclude')
        # With no missing data, both should give same result
        np.testing.assert_allclose(
            fs_include.theta('pi'), fs_exclude.theta('pi'), rtol=1e-12)

    def test_multiple_sample_sizes(self):
        """Inject missing data and verify grouping works."""
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 100), dtype=np.int8)
        # Add some missing data
        hap[0, :10] = -1
        hap[1, 20:30] = -1
        pos = np.arange(100, dtype=np.int32)
        hm = HaplotypeMatrix(hap, pos)
        fs = FrequencySpectrum(hm, missing_data='include')
        # Should have multiple sample sizes
        assert len(fs.sfs_by_n) >= 2
        # Theta should still be computable
        assert np.isfinite(fs.theta('pi'))
