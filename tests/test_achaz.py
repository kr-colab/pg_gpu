"""
Tests for the Achaz (2009) generalized theta estimation framework.
"""

import pytest
import numpy as np
import msprime
from pg_gpu import HaplotypeMatrix, diversity
from pg_gpu.diversity import (
    FrequencySpectrum, project_sfs, compute_sigma_ij,
    WEIGHT_REGISTRY,
)


@pytest.fixture
def simple_ts():
    ts = msprime.sim_ancestry(
        samples=50, sequence_length=100_000,
        recombination_rate=1e-8, population_size=10_000,
        random_seed=42, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
    # These tests assert scalar diversity == FrequencySpectrum (SFS) values.
    # That equivalence only holds on biallelic data: at multiallelic sites the
    # scalar mutation-count Watterson and the per-allele SFS diverge
    # (kr-colab/pg_gpu#100). Nothing above forces biallelic (the default JC
    # mutation model can produce triallelic sites), so guard it explicitly --
    # if a future rate/seed change adds a >2-allele site this fails loudly
    # rather than letting the equivalence break silently.
    G = ts.genotype_matrix()
    assert all(np.unique(G[i]).size <= 2 for i in range(ts.num_sites)), \
        "test_achaz fixture must stay biallelic (see kr-colab/pg_gpu#100)"
    return ts


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

    def test_projection_rejects_upsampling(self):
        with pytest.raises(ValueError, match="Cannot project up"):
            project_sfs(np.zeros(6), n_from=5, n_to=6)

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


def _multi_size_fs():
    """A FrequencySpectrum whose sites have several per-site sample sizes
    (scattered missing data) and some segregating alleles."""
    rng = np.random.default_rng(0)
    hap = (rng.random((20, 80)) < 0.3).astype(np.int8)
    hap[rng.random((20, 80)) < 0.1] = -1
    hm = HaplotypeMatrix(hap, np.arange(80) * 100, 0, 8000)
    return FrequencySpectrum(hm)


class TestFrequencySpectrumEdges:
    """Multi-sample-size, projection, and degenerate-input paths."""

    def test_suggest_projection_n_single_size(self, hm):
        fs = FrequencySpectrum(hm)
        assert len(fs.sfs_by_n) == 1
        assert fs.suggest_projection_n() == fs.n_max

    def test_suggest_projection_n_multi_size(self):
        fs = _multi_size_fs()
        assert len(fs.sfs_by_n) > 1
        ns = sorted(fs.sfs_by_n)
        # Retaining everything needs the smallest size; a valid key otherwise;
        # an unreachable fraction falls back to the smallest.
        assert fs.suggest_projection_n(retain_fraction=1.0) == ns[0]
        assert fs.suggest_projection_n(retain_fraction=0.5) in fs.sfs_by_n
        assert fs.suggest_projection_n(retain_fraction=2.0) == ns[0]

    def test_suggest_projection_n_no_segregating(self):
        # Several sample sizes but nothing segregates -> falls back to n_max.
        hap = np.zeros((10, 8), dtype=np.int8)
        hap[0, :4] = -1  # sites 0-3 have n_valid 9, sites 4-7 have 10
        fs = FrequencySpectrum(HaplotypeMatrix(hap, np.arange(8) * 100, 0, 800))
        assert fs.n_segregating == 0 and len(fs.sfs_by_n) > 1
        assert fs.suggest_projection_n() == fs.n_max

    def test_sfs_method_paths(self, hm):
        single = FrequencySpectrum(hm)
        np.testing.assert_array_equal(single.sfs(),
                                      list(single.sfs_by_n.values())[0])
        multi = _multi_size_fs()
        np.testing.assert_array_equal(multi.sfs(), multi.sfs_by_n[multi.n_max])
        target = min(multi.sfs_by_n)
        assert multi.sfs(n=target).shape == (target + 1,)

    def test_project_combines_sample_sizes(self):
        multi = _multi_size_fs()
        ns = sorted(multi.sfs_by_n)
        target = ns[len(ns) // 2]  # larger sizes project down, smaller skip
        proj = multi.project(target)
        assert proj.n_max == target
        assert list(proj.sfs_by_n) == [target]

    def test_empty_spectrum(self):
        # Under 'exclude', a missing call at every site leaves no complete
        # site, so the spectrum is empty.
        hap = np.zeros((6, 4), dtype=np.int8)
        hap[0, :] = -1  # haplotype 0 missing at every site
        fs = FrequencySpectrum(HaplotypeMatrix(hap, np.arange(4) * 100, 0, 400),
                               missing_data='exclude')
        assert fs.n_max == 0 and fs.n_segregating == 0
        np.testing.assert_array_equal(fs.sfs(), np.array([]))

    def test_skips_sites_with_one_called_haplotype(self):
        # Site 1 has a single non-missing haplotype (n_valid == 1) -> skipped.
        hap = np.array([[0, 0], [1, -1], [1, -1], [0, -1]], dtype=np.int8)
        fs = FrequencySpectrum(HaplotypeMatrix(hap, np.array([0, 100]), 0, 200))
        assert 1 not in fs.sfs_by_n and 4 in fs.sfs_by_n

    def test_invariant_site_correction(self, hm):
        n_seg = FrequencySpectrum(hm).n_segregating
        fs = FrequencySpectrum(hm, n_total_sites=n_seg + 50)
        # The 50 monomorphic sites land in bin 0 at the top sample size.
        assert fs.sfs_by_n[fs.n_max][0] == 50


class TestEtaScalarPath:
    """The eta-family through diversity_stats (the per-allele scalar path):
    absolute values from the Achaz definition and multiallelic per-allele
    decomposition."""

    ETA = ['eta1', 'eta1_star', 'minus_eta1', 'minus_eta1_star']

    def test_absolute_values_match_achaz_definition(self):
        # n=8 haplotypes; one column per known derived-allele count.
        n = 8
        counts = [1, 1, 1, 7, 3, 3, 2, 6]
        hap = np.zeros((n, len(counts)), dtype=np.int8)
        for j, c in enumerate(counts):
            hap[:c, j] = 1
        hm = HaplotypeMatrix(hap, np.arange(len(counts)) * 100)

        a1 = np.sum(1.0 / np.arange(1, n))
        cnt = np.array(counts)
        expected = {
            'eta1': np.sum(cnt == 1) / a1,
            'eta1_star': np.sum((cnt == 1) | (cnt == n - 1)) / a1,
            'minus_eta1': np.sum((cnt >= 2) & (cnt < n)) / (a1 - 1),
            'minus_eta1_star': (np.sum((cnt >= 2) & (cnt <= n - 2))
                                / (a1 - 1 - 1.0 / (n - 1))),
        }
        got = diversity.diversity_stats(hm, statistics=self.ETA,
                                        span_normalize=False)
        for name in self.ETA:
            np.testing.assert_allclose(got[name], expected[name], rtol=1e-12,
                                       err_msg=name)

    @pytest.mark.parametrize("name", ETA)
    def test_multiallelic_equals_per_allele_split(self, name):
        # A multiallelic site contributes per derived allele, so eta on data
        # with >2-allele sites equals eta on the same data split into one
        # biallelic indicator column per derived allele.
        rng = np.random.default_rng(3)
        n = 12
        hap = rng.integers(0, 3, size=(n, 40)).astype(np.int8)
        assert any(len(np.unique(hap[:, j])) > 2 for j in range(hap.shape[1]))

        split = np.stack(
            [(hap[:, j] == a).astype(np.int8)
             for j in range(hap.shape[1])
             for a in np.unique(hap[:, j]) if a > 0],
            axis=1)

        hm = HaplotypeMatrix(hap, np.arange(hap.shape[1]) * 100)
        hm_split = HaplotypeMatrix(split, np.arange(split.shape[1]) * 100)
        multi = diversity.diversity_stats(hm, statistics=[name],
                                          span_normalize=False)[name]
        decomp = diversity.diversity_stats(hm_split, statistics=[name],
                                           span_normalize=False)[name]
        np.testing.assert_allclose(multi, decomp, rtol=1e-12)
