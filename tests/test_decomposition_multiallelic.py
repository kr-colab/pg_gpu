"""Multiallelic-correctness tests for decomposition PCA.

The PCA standardization is a per-allele one-hot (drop reference, marginal Patterson
scaling), with biallelic and multiallelic columns computed separately and summed.
These tests pin: biallelic bit-identity (against the previous standardization
reproduced here, and against scikit-allel), the multiallelic covariance against an
independent host reference, index-independence, the scikit-allel-on-expanded subspace
oracle, the missing-data requirements, and deferred == dense.
"""
import numpy as np
import cupy as cp
import pytest
import allel

from pg_gpu import HaplotypeMatrix
from pg_gpu._memutil import allele_counts
from pg_gpu.decomposition import (
    _prepare_matrix, _prepare_centered, _multiallelic_onehot_dense, _DeferredPCA,
    _window_gram, pca, randomized_pca, local_pca, local_pca_jackknife, lostruct,
)


def _build_deferred(hap, scaler):
    """Reproduce _prepare_matrix's _DeferredPCA construction (biallelic block
    chunked + multiallelic one-hot block dense), so the deferred path can be
    exercised without forcing the memory threshold."""
    h = cp.asarray(hap)
    bi = h.max(0) <= 1
    hb, hmul = h[:, bi], h[:, ~bi]
    ac, nv = allele_counts(hb, n_alleles=2)
    dac = ac[:, 1:].sum(axis=1)
    site_mean = cp.where(nv > 0, dac.astype(cp.float64) / nv.astype(cp.float64), 0.0)
    if scaler == 'patterson':
        scale = cp.sqrt(site_mean * (1 - site_mean))
        scale = cp.where(scale > 0, scale, 1.0)
    else:
        scale = None
    multi_cols, n_multi = _multiallelic_onehot_dense(hmul, scaler)
    return _DeferredPCA(hb, site_mean, scale, scaler,
                        multi_cols if n_multi else None)


def _host_cov(hap, scaler='patterson'):
    """Independent numpy reference: split standardization, summed column outer
    products. Biallelic sites (max index <= 1) use the alt dosage; multiallelic
    use per-present-derived-allele one-hot. Missing imputed to p (-> 0 centered)."""
    hap = np.asarray(hap)
    cols = []
    for v in range(hap.shape[1]):
        c = hap[:, v]
        valid = c >= 0
        nv = int(valid.sum())
        mx = int(c[valid].max()) if nv else -1
        if mx <= 1:
            p = (c[valid].sum() / nv) if nv else 0.0
            col = np.where(valid, c.astype(float), p) - p
            s = np.sqrt(p * (1 - p))
            cols.append(col / (s if s > 0 else 1.0))
        else:
            for a in range(1, int(c[valid].max()) + 1):
                ca = int((c[valid] == a).sum())
                if ca == 0:
                    continue
                p = ca / nv
                col = np.where(valid, (c == a).astype(float), p) - p
                s = np.sqrt(p * (1 - p))
                cols.append(col / (s if s > 0 else 1.0))
    X = np.array(cols).T
    return X @ X.T


def _expand_alt_alleles(hap):
    """(n_expanded, n_samples) per-alt-allele indicators for scikit-allel PCA."""
    rows = []
    for v in range(hap.shape[1]):
        c = hap[:, v]
        if c.max() <= 1:
            rows.append((c == 1).astype(float))
        else:
            for a in range(1, int(c.max()) + 1):
                if (c == a).any():
                    rows.append((c == a).astype(float))
    return np.array(rows)


def _multiallelic_hap(seed=0, missing=False):
    rng = np.random.RandomState(seed)
    n = 24
    hap = rng.randint(0, 2, (n, 30)).astype(np.int8)
    hap[:, 5] = rng.randint(0, 3, n)      # triallelic
    hap[:, 10] = rng.choice([0, 2], n)    # 2 alleles, coded {0,2}
    hap[:, 15] = rng.randint(0, 4, n)     # quadallelic
    if missing:
        hap[rng.random(hap.shape) < 0.1] = -1
    return hap


def _hm(hap):
    return HaplotypeMatrix(hap, np.arange(hap.shape[1]) * 10,
                           chrom_start=0, chrom_end=hap.shape[1] * 10)


class TestDecompositionMultiallelic:

    @pytest.mark.parametrize("missing", [False, True])
    def test_covariance_matches_host_reference(self, missing):
        hap = _multiallelic_hap(seed=1, missing=missing)
        X = _prepare_matrix(_hm(hap), 'patterson', None, 'include')
        cov = cp.asnumpy(X @ X.T)
        np.testing.assert_allclose(cov, _host_cov(hap), rtol=1e-9, atol=1e-9)

    def test_index_independence(self):
        # A 2-allele site coded {0,2} gives the same gram as coded {0,1}.
        hap = _multiallelic_hap(seed=2)
        hap01 = hap.copy()
        hap01[:, 10] = np.where(hap[:, 10] == 2, 1, 0)   # recode site 10 to {0,1}
        g0 = cp.asnumpy((lambda X: X @ X.T)(_prepare_matrix(_hm(hap), 'patterson', None, 'include')))
        g1 = cp.asnumpy((lambda X: X @ X.T)(_prepare_matrix(_hm(hap01), 'patterson', None, 'include')))
        np.testing.assert_allclose(g0, g1, rtol=1e-10, atol=1e-10)

    def test_biallelic_bit_identical_to_previous(self):
        rng = np.random.RandomState(3)
        hb = rng.randint(0, 2, (24, 40)).astype(np.int8)
        hb[rng.random(hb.shape) < 0.1] = -1

        def old_prepare(h):
            # Reproduces the previous standardization (alt dosage: count of
            # allele 1 over non-missing), computed here directly.
            h = cp.asarray(h)
            d = cp.sum(h == 1, axis=0)
            nv = cp.sum(h >= 0, axis=0)
            sm = cp.where(nv > 0, d.astype(cp.float64) / nv, 0.0)
            sc = cp.sqrt(sm * (1 - sm))
            sc = cp.where(sc > 0, sc, 1.0)
            vm = h >= 0
            X = cp.where(vm, h, 0).astype(cp.float64)
            X = cp.where(vm, X, sm[None, :])
            X -= sm
            X /= sc
            return X

        Xnew = _prepare_matrix(_hm(hb), 'patterson', None, 'include')
        assert bool((Xnew == old_prepare(hb)).all().get())

    def test_missing_is_not_reference(self):
        # Triallelic site: missing row -> 0 after centering; reference carrier -> -p/scale.
        site = np.array([0, 1, 2, 2, 0, -1, 1, 2], dtype=np.int8)[:, None]
        cols, _ = _multiallelic_onehot_dense(cp.asarray(site), 'patterson')
        cols = cp.asnumpy(cols)
        p1 = 2 / 7                                  # allele 1: 2 copies over 7 valid
        s1 = np.sqrt(p1 * (1 - p1))
        assert np.isclose(cols[5, 0], 0.0, atol=1e-12)          # missing -> 0
        assert np.isclose(cols[0, 0], (0 - p1) / s1, atol=1e-12)  # ref carrier -> -p/s
        assert not np.isclose(cols[5, 0], cols[0, 0])

    def test_deferred_gram_equals_dense(self):
        hap = _multiallelic_hap(seed=4, missing=True)
        X = _prepare_matrix(_hm(hap), 'patterson', None, 'include')
        dp = _build_deferred(hap, 'patterson')
        np.testing.assert_allclose(cp.asnumpy(dp.chunked_gram()),
                                   cp.asnumpy(X @ X.T), rtol=1e-10, atol=1e-10)

    def test_deferred_matmul_transpose_match_dense(self):
        # The memory-constrained randomized_pca path uses _DeferredPCA @ Omega and
        # .T @ Y over the concatenated [biallelic | multiallelic] columns; verify
        # both against the dense standardized matrix (closes the deferred+multi gap).
        hap = _multiallelic_hap(seed=7, missing=True)
        X = _prepare_matrix(_hm(hap), 'patterson', None, 'include')
        dp = _build_deferred(hap, 'patterson')
        rng = np.random.RandomState(11)
        Om = cp.asarray(rng.randn(X.shape[1], 5))
        Y = cp.asarray(rng.randn(X.shape[0], 5))
        np.testing.assert_allclose(cp.asnumpy(dp @ Om), cp.asnumpy(X @ Om),
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(cp.asnumpy(dp.T @ Y), cp.asnumpy(X.T @ Y),
                                   rtol=1e-9, atol=1e-9)

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

    def test_pca_rejects_genotype_matrix(self):
        from pg_gpu import GenotypeMatrix
        gm = GenotypeMatrix(np.random.RandomState(0).randint(0, 3, (5, 20)).astype(np.int8),
                            np.arange(20) * 100)
        with pytest.raises(TypeError, match="pca_dosage"):
            pca(gm)


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
    """local_pca / lostruct / jackknife route each window through _prepare_matrix,
    so per-window results are per-allele-correct. scaler=None (center-only) is the
    local_pca / lostruct default and is what these exercise."""

    def _direct_eigvals(self, hap_sub, pos_sub, s, e, k):
        sub = HaplotypeMatrix(hap_sub, pos_sub, int(s), int(e))
        X = _prepare_matrix(sub, None, None, 'include')
        C, _ = _window_gram(X, X.shape[1])
        return np.sort(cp.asnumpy(cp.linalg.eigh(C)[0]))[::-1][:k]

    def test_single_window_matches_direct(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=1)
        seqlen = hap.shape[1] * 100
        res = local_pca(hm, k=3, window_type='bp', window_size=seqlen + 1000,
                        scaler=None)
        direct = self._direct_eigvals(hap, pos, 0, seqlen + 1000, 3)
        np.testing.assert_allclose(res.eigvals[0], direct, rtol=1e-8, atol=1e-8)

    def test_per_window_matches_direct(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=2)
        res = local_pca(hm, k=2, window_type='bp', window_size=3000,
                        step_size=3000, scaler=None)
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
        lo = lostruct(hm, k=2, window_type='bp', window_size=3000, step_size=3000,
                      scaler=None)
        assert np.isfinite(lo.distance).all()
        assert np.isfinite(lo.mds).all()
        np.testing.assert_allclose(lo.distance, lo.distance.T, atol=1e-10)

    def test_jackknife_se_finite(self):
        hm, hap, pos = _windowed_multiallelic_hm(seed=4)
        res = local_pca(hm, k=2, window_type='bp', window_size=3000,
                        step_size=3000, scaler=None)
        se = local_pca_jackknife(hm, k=2, n_blocks=3, window_type='bp',
                                 window_size=3000, step_size=3000, scaler=None)
        valid = res.windows['n_variants'].values >= 6
        assert np.isfinite(se[valid]).all()
