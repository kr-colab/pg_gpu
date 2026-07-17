"""Multiallelic-correctness tests for decomposition PCA.

pca / randomized_pca use the all-allele centered standardization (every present
allele including the reference, centered by its frequency, no variance scaling),
so the Gram X @ X.T normalized by the number of segregating sites is exactly the
site-mode genetic_relatedness matrix that tskit's PCA decomposes. These tests pin
that convention: the Gram and explained variance against tskit, the randomized
approximation against full pca, the GenotypeMatrix rejection, and the per-window
all-allele path used by local_pca / lostruct / jackknife.
"""
import numpy as np
import cupy as cp
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import (
    _prepare_centered, _window_gram, pca, randomized_pca,
    local_pca, local_pca_jackknife, lostruct,
)


class TestDecompositionMultiallelic:

    def test_pca_gram_matches_tskit_relatedness(self, multiallelic_hm):
        # The centered all-allele Gram (proportion-normalized) is exactly the
        # site-mode genetic_relatedness matrix -- the matrix tskit's PCA
        # decomposes. This pins the whole tskit convention.
        ts, hm = multiallelic_hm
        X, n_seg = _prepare_centered(hm)
        C = cp.asnumpy((X @ X.T) / n_seg)
        n = ts.num_samples
        sets = [[s] for s in ts.samples()]
        idx = [(i, j) for i in range(n) for j in range(n)]
        G = np.asarray(ts.genetic_relatedness(
            sets, indexes=idx, mode='site', centre=True, polarised=False,
            proportion=True)).reshape(n, n)
        np.testing.assert_allclose(C, G, atol=1e-9, rtol=1e-6)

    def test_pca_explained_variance_matches_tskit(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        n = ts.num_samples
        sets = [[s] for s in ts.samples()]
        idx = [(i, j) for i in range(n) for j in range(n)]
        G = np.asarray(ts.genetic_relatedness(
            sets, indexes=idx, mode='site', centre=True, polarised=False,
            proportion=True)).reshape(n, n)
        evals = np.sort(np.linalg.eigvalsh(G))[::-1]
        _, evr = pca(hm, n_components=4)
        np.testing.assert_allclose(evr, evals[:4] / evals.sum(),
                                   atol=1e-9, rtol=1e-6)

    def test_randomized_pca_matches_pca(self, multiallelic_hm):
        # Randomized PCA shares pca's all-allele standardization, so it should
        # recover the same explained-variance ratios and the same top subspace
        # (columns equal up to sign).
        _, hm = multiallelic_hm
        coords, evr = pca(hm, n_components=4)
        rcoords, revr = randomized_pca(hm, n_components=4, n_iter=7,
                                       random_state=0)
        np.testing.assert_allclose(revr, evr, atol=1e-6, rtol=1e-5)
        for j in range(coords.shape[1]):
            corr = np.corrcoef(coords[:, j], rcoords[:, j])[0, 1]
            assert abs(corr) > 1 - 1e-6

    def test_pca_rejects_genotype_matrix(self):
        from pg_gpu import GenotypeMatrix
        gm = GenotypeMatrix(np.random.RandomState(0).randint(0, 3, (5, 20)).astype(np.int8),
                            np.arange(20) * 100)
        with pytest.raises(TypeError, match="pca_dosage"):
            pca(gm)

    def test_randomized_pca_rejects_genotype_matrix(self):
        from pg_gpu import GenotypeMatrix
        gm = GenotypeMatrix(np.random.RandomState(0).randint(0, 3, (5, 20)).astype(np.int8),
                            np.arange(20) * 100)
        with pytest.raises(TypeError, match="randomized_pca_dosage"):
            randomized_pca(gm)


def _windowed_multiallelic_hm(seed=0, n=40, nv=120):
    """Multiallelic HaplotypeMatrix with positions, enough sites for several
    windows (~30 sites/window at window_size=3000)."""
    rng = np.random.RandomState(seed)
    hap = rng.randint(0, 2, (n, nv)).astype(np.int8)
    for j in range(0, nv, 7):
        hap[:, j] = rng.randint(0, 3, n)
    for j in range(0, nv, 17):
        hap[:, j] = rng.randint(0, 4, n)
    pos = np.arange(nv) * 100
    return HaplotypeMatrix(hap, pos, 0, nv * 100), hap, pos


class TestLocalPCAMultiallelic:
    """local_pca / lostruct / jackknife route each window through
    _prepare_centered (the all-allele centered standardization) and _window_gram,
    so per-window results are per-allele-correct."""

    def _direct_eigvals(self, hap_sub, pos_sub, s, e, k):
        sub = HaplotypeMatrix(hap_sub, pos_sub, int(s), int(e))
        X, _ = _prepare_centered(sub, missing_data='include', need_segregating=False)
        C, _ = _window_gram(X, X.shape[1])
        return np.sort(cp.asnumpy(cp.linalg.eigh(C)[0]))[::-1][:k]

    def test_single_window_matches_direct(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=1)
        seqlen = hap.shape[1] * 100
        res = local_pca(hm, k=3, window_type='bp', window_size=seqlen + 1000)
        direct = self._direct_eigvals(hap, pos, 0, seqlen + 1000, 3)
        np.testing.assert_allclose(res.eigvals[0], direct, rtol=1e-8, atol=1e-8)

    def test_per_window_matches_direct(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=2)
        res = local_pca(hm, k=2, window_type='bp', window_size=3000,
                        step_size=3000)
        valid = res.windows['n_variants'].values >= 2
        assert not np.isnan(res.eigvals[valid]).any()
        for w, (s, e, nvar) in enumerate(zip(res.windows['start'],
                                             res.windows['end'],
                                             res.windows['n_variants'])):
            if nvar < 2:
                continue
            m = (pos >= s) & (pos < e)
            direct = self._direct_eigvals(hap[:, m], pos[m], s, e, 2)
            np.testing.assert_allclose(res.eigvals[w], direct, rtol=1e-7, atol=1e-7)

    def test_lostruct_runs_finite(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=3)
        lo = lostruct(hm, k=2, window_type='bp', window_size=3000, step_size=3000)
        assert np.isfinite(lo.distance).all()
        assert np.isfinite(lo.mds).all()
        np.testing.assert_allclose(lo.distance, lo.distance.T, atol=1e-10)

    def test_jackknife_se_finite(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=4)
        res = local_pca(hm, k=2, window_type='bp', window_size=3000,
                        step_size=3000)
        se = local_pca_jackknife(hm, k=2, n_blocks=3, window_type='bp',
                                 window_size=3000, step_size=3000)
        valid = res.windows['n_variants'].values >= 6
        assert np.isfinite(se[valid]).all()
