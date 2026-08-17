"""
Tests for PBS, PCA, pairwise_distance, and PCoA.
Validates against scikit-allel where applicable.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import divergence, decomposition


def _allele_counts(hap):
    """Helper: allele counts (n_variants, 2) from haplotype array."""
    n = hap.shape[0]
    dac = np.sum(hap, axis=0)
    return np.column_stack([n - dac, dac])


# ---------------------------------------------------------------------------
# PBS tests
# ---------------------------------------------------------------------------

class TestPBS:
    """Test Population Branch Statistic."""

    @pytest.fixture
    def three_pop_matrix(self):
        np.random.seed(42)
        n_var = 100
        pops = {}
        for name in ['pop1', 'pop2', 'pop3']:
            pops[name] = np.random.randint(0, 2, (10, n_var), dtype=np.int8)
        combined = np.vstack([pops[k] for k in ['pop1', 'pop2', 'pop3']])
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(
            combined, pos, 0, n_var * 1000,
            sample_sets={'pop1': list(range(10)),
                         'pop2': list(range(10, 20)),
                         'pop3': list(range(20, 30))}
        )
        return matrix, pops

    def test_pbs_output_shape(self, three_pop_matrix):
        matrix, _ = three_pop_matrix
        result = divergence.pbs(matrix, 'pop1', 'pop2', 'pop3',
                                window_size=20)
        n_windows = (100 - 20) // 20 + 1
        assert result.shape == (n_windows,)

    def test_pbs_vs_allel(self, three_pop_matrix):
        matrix, pops = three_pop_matrix

        # pg_gpu
        result_pg = divergence.pbs(matrix, 'pop1', 'pop2', 'pop3',
                                   window_size=20, normed=True)

        # allel
        ac1 = _allele_counts(pops['pop1'])
        ac2 = _allele_counts(pops['pop2'])
        ac3 = _allele_counts(pops['pop3'])
        result_allel = allel.pbs(ac1, ac2, ac3, window_size=20, normed=True)

        both_valid = ~np.isnan(result_pg) & ~np.isnan(result_allel)
        if np.sum(both_valid) > 0:
            np.testing.assert_allclose(
                result_pg[both_valid], result_allel[both_valid],
                rtol=1e-3,
                err_msg="PBS does not match allel"
            )

    def test_pbs_unnormed(self, three_pop_matrix):
        matrix, pops = three_pop_matrix
        result_pg = divergence.pbs(matrix, 'pop1', 'pop2', 'pop3',
                                   window_size=20, normed=False)
        result_normed = divergence.pbs(matrix, 'pop1', 'pop2', 'pop3',
                                       window_size=20, normed=True)
        # normed values should generally be smaller
        valid = ~np.isnan(result_pg) & ~np.isnan(result_normed)
        if np.sum(valid) > 0:
            assert not np.allclose(result_pg[valid], result_normed[valid])


# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------

class TestPCA:
    """Test PCA functions."""

    @pytest.fixture
    def pca_data(self):
        np.random.seed(42)
        n_hap = 40
        n_var = 100
        hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        return HaplotypeMatrix(hap, pos, 0, n_var * 1000)

    def test_pca_output_shape(self, pca_data):
        coords, var_ratio = decomposition.pca(pca_data, n_components=5)
        assert coords.shape == (40, 5)
        assert var_ratio.shape == (5,)
        assert np.all(var_ratio >= 0)
        assert np.sum(var_ratio) <= 1.0 + 1e-10

    def test_pca_vs_allel(self, pca_data):
        """Verify PCA produces similar variance explained as allel."""
        hap = pca_data.haplotypes
        if hasattr(hap, 'get'):
            hap = hap.get()

        # pg_gpu
        _, var_pg = decomposition.pca(pca_data, n_components=5,
                                       scaler='patterson')

        # allel: needs (n_variants, n_samples) genotype array
        # use haplotypes directly as "genotypes" (0/1)
        gn = hap.T.astype('i1')
        _, model = allel.pca(gn, n_components=5, scaler='patterson')
        var_allel = model.explained_variance_ratio_

        # variance ratios should correlate highly
        corr = np.corrcoef(var_pg, var_allel)[0, 1]
        assert corr > 0.9, f"PCA variance correlation: {corr}"

    def test_pca_scalers(self, pca_data):
        """All scalers should produce valid output."""
        for scaler in ['patterson', 'standard', None]:
            coords, var_ratio = decomposition.pca(
                pca_data, n_components=3, scaler=scaler)
            assert coords.shape == (40, 3)
            assert not np.any(np.isnan(coords))

    def test_pca_with_population(self, pca_data):
        pca_data.sample_sets = {'sub': list(range(20))}
        coords, _ = decomposition.pca(pca_data, n_components=3,
                                       population='sub')
        assert coords.shape == (20, 3)


class TestRandomizedPCA:
    """Test randomized PCA."""

    def test_randomized_pca_shape(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (50, 200), dtype=np.int8)
        pos = np.arange(200) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 200000)

        coords, var_ratio = decomposition.randomized_pca(
            matrix, n_components=5, random_state=42)
        assert coords.shape == (50, 5)
        assert var_ratio.shape == (5,)

    def test_randomized_vs_full_pca(self):
        """Randomized PCA should approximate full PCA."""
        np.random.seed(42)
        hap = np.random.randint(0, 2, (30, 100), dtype=np.int8)
        pos = np.arange(100) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 100000)

        coords_full, var_full = decomposition.pca(matrix, n_components=5)
        coords_rand, var_rand = decomposition.randomized_pca(
            matrix, n_components=5, random_state=42)

        # variance explained should be similar
        np.testing.assert_allclose(var_full, var_rand, atol=0.05)


# ---------------------------------------------------------------------------
# Distance tests
# ---------------------------------------------------------------------------

class TestPairwiseDistance:
    """Test pairwise distance computation."""

    def test_euclidean(self):
        hap = np.array([[0, 0, 1],
                         [0, 1, 1],
                         [1, 1, 0]], dtype=np.int8)
        pos = np.arange(3) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 3000)

        dist = decomposition.pairwise_distance(matrix, metric='euclidean')
        assert dist.shape == (3,)  # 3 choose 2

        # manual check
        d01 = np.sqrt(0 + 1 + 0)  # 1.0
        d02 = np.sqrt(1 + 1 + 1)  # sqrt(3)
        d12 = np.sqrt(1 + 0 + 1)  # sqrt(2)
        np.testing.assert_allclose(dist, [d01, d02, d12], rtol=1e-10)

    def test_cityblock(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (10, 50), dtype=np.int8)
        pos = np.arange(50) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 50000)

        dist = decomposition.pairwise_distance(matrix, metric='cityblock')
        assert dist.shape == (45,)  # 10 choose 2

    def test_vs_scipy(self):
        """Compare GPU distance against scipy pdist."""
        from scipy.spatial.distance import pdist
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 50), dtype=np.int8)
        pos = np.arange(50) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 50000)

        for metric in ['euclidean', 'cityblock', 'sqeuclidean']:
            dist_pg = decomposition.pairwise_distance(matrix, metric=metric)
            dist_scipy = pdist(hap.astype(float), metric=metric)
            np.testing.assert_allclose(dist_pg, dist_scipy, rtol=1e-10,
                                      err_msg=f"{metric} mismatch")

    def test_missing_data_modes(self):
        from scipy.spatial.distance import pdist
        hap = np.array([
            [0, 0, 1, -1],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ], dtype=np.int8)
        matrix = HaplotypeMatrix(hap, np.arange(4) * 1000, 0, 4000)

        excluded = decomposition.pairwise_distance(
            matrix, metric='sqeuclidean', missing_data='exclude')
        expected = pdist(hap[:, :3].astype(float), metric='sqeuclidean')
        np.testing.assert_allclose(excluded, expected, rtol=1e-10)

        included = decomposition.pairwise_distance(
            matrix, metric='sqeuclidean', missing_data='include')
        np.testing.assert_allclose(included, [4.0 / 3.0, 4.0, 3.0],
                                   rtol=1e-10)


# ---------------------------------------------------------------------------
# PCoA tests
# ---------------------------------------------------------------------------

class TestPCoA:
    """Test Principal Coordinate Analysis."""

    def test_pcoa_basic(self):
        np.random.seed(42)
        hap = np.random.randint(0, 2, (20, 50), dtype=np.int8)
        pos = np.arange(50) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 50000)

        dist = decomposition.pairwise_distance(matrix)
        coords, var_ratio = decomposition.pcoa(dist)

        assert coords.shape[0] == 20
        assert len(var_ratio) > 0
        assert np.all(var_ratio >= 0)

    def test_pcoa_vs_allel(self):
        """Compare PCoA against allel."""
        np.random.seed(42)
        hap = np.random.randint(0, 2, (15, 40), dtype=np.int8)
        pos = np.arange(40) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 40000)

        dist_pg = decomposition.pairwise_distance(matrix, metric='euclidean')
        coords_pg, var_pg = decomposition.pcoa(dist_pg)

        coords_allel, var_allel = allel.pcoa(dist_pg)

        # eigenvalues should match
        n_comp = min(len(var_pg), len(var_allel))
        np.testing.assert_allclose(
            var_pg[:n_comp], var_allel[:n_comp], rtol=1e-5,
            err_msg="PCoA variance ratios differ from allel"
        )
