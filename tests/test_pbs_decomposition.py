"""
Tests for PBS, PCA, pairwise_distance, and PCoA.
Validates against scikit-allel where applicable.
"""

import pytest
import numpy as np
import cupy as cp
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

    # The Patterson-vs-scikit-allel comparison moved to pca_dosage (the diploid
    # biallelic GCTA PCA on a GenotypeMatrix); pca is now the tskit PCA of
    # genetic_relatedness, pinned in test_decomposition_multiallelic.py.

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

    @pytest.mark.parametrize("metric", ['euclidean', 'cityblock', 'sqeuclidean'])
    @pytest.mark.parametrize("missing_data", ['include', 'exclude'])
    def test_biallelic_bit_identical_to_premultiallelic(self, metric, missing_data):
        # The mismatch-count rewrite must not change biallelic output, including
        # the missing-data normalization path (which scipy cannot oracle since it
        # treats -1 as a value). Pin bit-for-bit against the pre-multiallelic
        # per-pair formula reproduced here.
        def premultiallelic(hap):
            hap = cp.asarray(hap)
            if missing_data == 'exclude':
                hap = hap[:, cp.sum(hap < 0, axis=0) == 0]
            X = cp.where(hap >= 0, hap, 0).astype(cp.float64)
            valid = (hap >= 0).astype(cp.float64)
            has_missing = bool(cp.any(hap < 0))
            n, nv = X.shape
            ii, jj = cp.triu_indices(n, k=1)
            if has_missing:
                joint = valid[ii] * valid[jj]
                njoint = cp.sum(joint, axis=1)
            if metric == 'cityblock':
                raw = cp.sum(cp.abs(X[ii] - X[jj]) * (joint if has_missing else 1.0), axis=1)
            else:
                raw = cp.sum(((X[ii] - X[jj]) ** 2) * (joint if has_missing else 1.0), axis=1)
            d = cp.where(njoint > 0, raw * nv / njoint, 0.0) if has_missing else raw
            if metric == 'euclidean':
                d = cp.sqrt(d)
            return d.get()

        rng = np.random.RandomState(7)
        hap = rng.randint(0, 2, (16, 60)).astype(np.int8)
        hap[rng.random(hap.shape) < 0.15] = -1
        matrix = HaplotypeMatrix(hap, np.arange(60) * 10, 0, 600)
        new = decomposition.pairwise_distance(
            matrix, metric=metric, missing_data=missing_data)
        np.testing.assert_array_equal(new, premultiallelic(hap))


def _host_pairwise(hap, metric, missing_data='include'):
    """Independent host reference: the allele-mismatch count m per pair, scaled to
    the full variant span over jointly non-missing sites. euclidean = sqrt(m)."""
    hap = np.asarray(hap)
    if missing_data == 'exclude':
        hap = hap[:, (hap >= 0).all(axis=0)]
    n, nv = hap.shape
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            valid = (hap[i] >= 0) & (hap[j] >= 0)
            njoint = int(valid.sum())
            m = int(((hap[i] != hap[j]) & valid).sum())
            d = (m * nv / njoint) if njoint > 0 else 0.0
            out.append(np.sqrt(d) if metric == 'euclidean' else d)
    return np.array(out)


class TestPairwiseDistanceMultiallelic:
    """pairwise_distance is a label-independent allele-mismatch distance, correct
    on multiallelic sites and unchanged on biallelic data."""

    def _hm(self, hap):
        return HaplotypeMatrix(hap, np.arange(hap.shape[1]) * 10, 0,
                               hap.shape[1] * 10)

    @pytest.mark.parametrize("metric", ['euclidean', 'cityblock', 'sqeuclidean'])
    @pytest.mark.parametrize("missing_data", ['include', 'exclude'])
    def test_matches_host_reference(self, metric, missing_data):
        rng = np.random.RandomState(1)
        hap = rng.randint(0, 4, (14, 50)).astype(np.int8)
        hap[rng.random(hap.shape) < 0.12] = -1
        dist = decomposition.pairwise_distance(
            self._hm(hap), metric=metric, missing_data=missing_data)
        np.testing.assert_allclose(
            dist, _host_pairwise(hap, metric, missing_data), rtol=1e-12)

    @pytest.mark.parametrize("metric", ['euclidean', 'cityblock', 'sqeuclidean'])
    def test_label_independence(self, metric):
        rng = np.random.RandomState(2)
        hap = rng.randint(0, 4, (12, 40)).astype(np.int8)
        perm = np.array([2, 0, 3, 1], dtype=np.int8)   # relabel alleles 0..3
        d0 = decomposition.pairwise_distance(self._hm(hap), metric=metric)
        d1 = decomposition.pairwise_distance(self._hm(perm[hap]), metric=metric)
        np.testing.assert_array_equal(d0, d1)

    @pytest.mark.parametrize("metric", ['euclidean', 'cityblock', 'sqeuclidean'])
    def test_biallelic_index_reduction(self, metric):
        # A 2-allele site coded {0,2} gives the same distances as {0,1}.
        rng = np.random.RandomState(3)
        h2 = rng.choice([0, 2], (10, 30)).astype(np.int8)
        h1 = np.where(h2 == 2, 1, 0).astype(np.int8)
        d2 = decomposition.pairwise_distance(self._hm(h2), metric=metric)
        d1 = decomposition.pairwise_distance(self._hm(h1), metric=metric)
        np.testing.assert_array_equal(d2, d1)

    def test_metrics_collapse_on_categorical(self):
        # On allele-index data cityblock == sqeuclidean == m and euclidean == sqrt(m).
        rng = np.random.RandomState(4)
        hap = rng.randint(0, 4, (10, 45)).astype(np.int8)
        hm = self._hm(hap)
        cb = decomposition.pairwise_distance(hm, metric='cityblock')
        sq = decomposition.pairwise_distance(hm, metric='sqeuclidean')
        eu = decomposition.pairwise_distance(hm, metric='euclidean')
        np.testing.assert_array_equal(cb, sq)
        np.testing.assert_allclose(eu, np.sqrt(sq), rtol=1e-12)

    def test_unsupported_metric_raises(self):
        rng = np.random.RandomState(5)
        hm = self._hm(rng.randint(0, 3, (6, 20)).astype(np.int8))
        with pytest.raises(NotImplementedError, match="euclidean"):
            decomposition.pairwise_distance(hm, metric='correlation')

    def test_rejects_genotype_matrix(self):
        from pg_gpu import GenotypeMatrix
        gm = GenotypeMatrix(
            np.random.RandomState(6).randint(0, 3, (6, 20)).astype(np.int8),
            np.arange(20) * 10)
        with pytest.raises(TypeError, match="HaplotypeMatrix"):
            decomposition.pairwise_distance(gm)


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
