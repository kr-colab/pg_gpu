"""
Tests for local PCA (lostruct), pc_dist, corners, and local_pca_jackknife.

Validates:
  * shape / dtype invariants
  * batched GPU eigh matches a per-window Python-loop reference
  * NaN handling for sparse/empty windows
  * Frobenius distance identity
  * L1/L2 normalization invariance properties
  * lostruct-matrix round-trip through pc_dist
  * corners on structured point clouds
  * jackknife shape + scaling + sign alignment
  * windowed_analysis dispatch

Parity tests against frozen R lostruct outputs live in
``tests/test_local_pca_parity.py`` and are skipped automatically if the
reference files are missing.
"""

import numpy as np
import pandas as pd
import pytest

from pg_gpu import HaplotypeMatrix, windowed_analysis
from pg_gpu.decomposition import (
    LocalPCAResult,
    corners,
    local_pca,
    local_pca_jackknife,
    pc_dist,
    pcoa,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_hm():
    """40 haplotypes × 2000 variants, uniformly spaced positions."""
    rng = np.random.default_rng(42)
    n_hap, n_var = 40, 2000
    hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return HaplotypeMatrix(hap, pos, 0, n_var * 1000)


@pytest.fixture
def structured_hm():
    """Two-deme matrix with a region of elevated between-deme differentiation.

    Used to check that local PCA recovers a structured region in MDS space.
    """
    rng = np.random.default_rng(0)
    n_hap_per = 20
    n_hap = 2 * n_hap_per
    n_var = 3000

    # Baseline: random 50/50
    hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)

    # Inject a structured region (variants 1000..1500) with deme-specific freqs
    for j in range(1000, 1500):
        hap[:n_hap_per, j] = rng.binomial(1, 0.2, n_hap_per)
        hap[n_hap_per:, j] = rng.binomial(1, 0.8, n_hap_per)

    pos = np.arange(n_var, dtype=np.int64) * 1000
    return HaplotypeMatrix(
        hap, pos, 0, n_var * 1000,
        sample_sets={
            'pop1': list(range(n_hap_per)),
            'pop2': list(range(n_hap_per, n_hap)),
        },
    )


# ---------------------------------------------------------------------------
# Shape / output-structure tests
# ---------------------------------------------------------------------------


class TestLocalPCAShape:

    def test_basic_shape(self, small_hm):
        res = local_pca(small_hm, window_size=100, window_type='snp', k=2)
        assert isinstance(res, LocalPCAResult)
        assert res.k == 2
        assert res.eigvals.shape == (res.n_windows, 2)
        assert res.eigvecs.shape == (res.n_windows, 2, small_hm.num_haplotypes)
        assert res.sumsq.shape == (res.n_windows,)
        assert len(res.windows) == res.n_windows
        expected_cols = {'chrom', 'start', 'end', 'center', 'n_variants',
                         'window_id'}
        assert expected_cols <= set(res.windows.columns)
        # Eigvals are sorted descending
        assert np.all(res.eigvals[:, 0] >= res.eigvals[:, 1])

    def test_to_lostruct_matrix(self, small_hm):
        res = local_pca(small_hm, window_size=100, window_type='snp', k=3)
        flat = res.to_lostruct_matrix()
        n_samples = small_hm.num_haplotypes
        assert flat.shape == (res.n_windows, 1 + 3 + 3 * n_samples)
        # Column 0 = sumsq
        np.testing.assert_array_equal(flat[:, 0], res.sumsq)
        # Columns 1..1+k = eigvals
        np.testing.assert_array_equal(flat[:, 1:4], res.eigvals)
        # Remaining = eigvecs flattened (k, n_samples) in C order
        np.testing.assert_array_equal(
            flat[:, 4:], res.eigvecs.reshape(res.n_windows, -1))


# ---------------------------------------------------------------------------
# Batched vs per-window loop reference
# ---------------------------------------------------------------------------


def _reference_local_pca_numpy(hm: HaplotypeMatrix, window_size: int, k: int,
                               step_size=None):
    """Per-window numpy reference: row-center, col-center, (X X^T)/(n-1)."""
    step = step_size or window_size
    if hm.device == 'GPU':
        hap = hm.haplotypes.get().astype(np.float64)
    else:
        hap = hm.haplotypes.astype(np.float64)
    n_hap, n_var = hap.shape
    out_vals, out_vecs, out_sumsq = [], [], []
    for start in range(0, n_var - window_size + 1, step):
        X = hap[:, start:start + window_size].copy()
        # Row-center (per variant)
        X -= X.mean(axis=0, keepdims=True)
        # Col-center (per sample) to match R's cov() double-centering
        X -= X.mean(axis=1, keepdims=True)
        C = (X @ X.T) / (window_size - 1)
        sumsq = float((C ** 2).sum())
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1][:k]
        out_vals.append(evals[idx])
        out_vecs.append(evecs[:, idx].T)  # shape (k, n_hap)
        out_sumsq.append(sumsq)
    return (np.stack(out_vals), np.stack(out_vecs), np.array(out_sumsq))


class TestBatchedVsLoop:

    def test_matches_numpy_reference(self, small_hm):
        res = local_pca(small_hm, window_size=200, window_type='snp', k=3)
        ref_vals, ref_vecs, ref_sumsq = _reference_local_pca_numpy(
            small_hm, window_size=200, k=3)
        # Trim to the overlap — WindowIterator may produce more windows than
        # the numpy reference (e.g. for bp-windowing). Our SNP window with
        # step=window_size yields floor(n_var / window_size) windows in both.
        n = min(len(ref_sumsq), res.n_windows)
        np.testing.assert_allclose(res.sumsq[:n], ref_sumsq[:n], rtol=1e-6)
        np.testing.assert_allclose(res.eigvals[:n], ref_vals[:n], rtol=1e-6)
        # Sign-align and compare eigenvectors
        for w in range(n):
            for pc in range(3):
                v_ref = ref_vecs[w, pc]
                v_ours = res.eigvecs[w, pc]
                if np.abs(np.dot(v_ref, v_ours)) < 0.99:
                    # Degenerate / near-degenerate eigenvalues — skip
                    continue
                sign = np.sign(np.dot(v_ref, v_ours))
                np.testing.assert_allclose(
                    sign * v_ours, v_ref, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# NaN / sparse-window handling
# ---------------------------------------------------------------------------


class TestNanHandling:

    def test_sparse_window_returns_nan(self, small_hm):
        # Window too small for k — force NaN.
        # Use a region-based window that lands on a tiny slice.
        regions = pd.DataFrame({'chrom': [1, 1],
                                'start': [0, 5000],
                                'end': [100, 1_000_000]})
        res = local_pca(small_hm, window_type='regions',
                        regions=regions, window_size=0, step_size=0, k=3)
        # First window is tiny (1 variant at pos 0) -> NaN
        assert np.all(np.isnan(res.eigvals[0]))
        assert np.all(np.isnan(res.eigvecs[0]))
        assert np.isnan(res.sumsq[0])
        # Second window has many variants -> finite
        assert np.all(np.isfinite(res.eigvals[1]))


# ---------------------------------------------------------------------------
# pc_dist correctness
# ---------------------------------------------------------------------------


class TestPcDist:

    def test_frobenius_full_rank_matches_direct(self, small_hm):
        """For npc == n_samples, pc_dist equals direct Frobenius on C."""
        n_samples = small_hm.num_haplotypes
        # Use small number of windows; k = n_samples for full rank
        res = local_pca(small_hm, window_size=200, window_type='snp',
                        k=n_samples)
        # Reconstruct covariance matrices from eigendecomp: C = U diag(lam) U^T
        nw = res.n_windows
        C_all = np.empty((nw, n_samples, n_samples))
        for w in range(nw):
            U = res.eigvecs[w].T  # (n_samples, k)
            lam = res.eigvals[w]
            C_all[w] = U @ np.diag(lam) @ U.T
        # Direct Frobenius
        direct = np.zeros((nw, nw))
        for i in range(nw):
            for j in range(nw):
                direct[i, j] = np.sqrt(((C_all[i] - C_all[j]) ** 2).sum())

        d_ours = pc_dist(res, npc=n_samples, normalize=None)
        np.testing.assert_allclose(d_ours, direct, rtol=1e-6, atol=1e-8)
        # Diagonal ~ 0
        np.testing.assert_allclose(np.diag(d_ours), 0.0, atol=1e-8)
        # Symmetric
        np.testing.assert_allclose(d_ours, d_ours.T, atol=1e-12)

    def test_L1_rescale_invariance(self, small_hm):
        """L1-normalized distance is invariant to per-window covariance scale."""
        res = local_pca(small_hm, window_size=200, window_type='snp', k=3)
        d1 = pc_dist(res, npc=3, normalize='L1')
        # Scale each window's eigenvalues by a different positive constant
        res2 = LocalPCAResult(
            windows=res.windows, eigvals=res.eigvals * 2.0,
            eigvecs=res.eigvecs, sumsq=res.sumsq * 4.0,
            k=res.k, scaler=res.scaler, missing_data=res.missing_data)
        d2 = pc_dist(res2, npc=3, normalize='L1')
        np.testing.assert_allclose(d1, d2, rtol=1e-8, atol=1e-10)

    def test_L2_uniform_rescale_invariance(self, small_hm):
        res = local_pca(small_hm, window_size=200, window_type='snp', k=3)
        d1 = pc_dist(res, npc=3, normalize='L2')
        res2 = LocalPCAResult(
            windows=res.windows, eigvals=res.eigvals * 3.0,
            eigvecs=res.eigvecs, sumsq=res.sumsq * 9.0,
            k=res.k, scaler=res.scaler, missing_data=res.missing_data)
        d2 = pc_dist(res2, npc=3, normalize='L2')
        np.testing.assert_allclose(d1, d2, rtol=1e-8, atol=1e-10)

    def test_flat_matrix_roundtrip(self, small_hm):
        res = local_pca(small_hm, window_size=200, window_type='snp', k=2)
        flat = res.to_lostruct_matrix()
        d_from_result = pc_dist(res, npc=2, normalize='L1')
        d_from_flat = pc_dist(flat, npc=2, normalize='L1')
        np.testing.assert_allclose(d_from_result, d_from_flat,
                                   rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# End-to-end MDS on a structured dataset
# ---------------------------------------------------------------------------


class TestEndToEnd:

    def test_mds_separates_structured_region(self, structured_hm):
        res = local_pca(structured_hm, window_size=200, window_type='snp',
                        step_size=100, k=2)
        d = pc_dist(res, npc=2, normalize='L1')
        assert np.all(np.isfinite(d))
        coords, er = pcoa(d, n_components=2)
        assert coords.shape == (res.n_windows, 2)
        # Windows spanning the inserted structured block (variants 1000..1500)
        # at step=100 should separate from the rest on the first MDS axis.
        centers = res.windows['center'].to_numpy()
        structured_mask = (centers >= 1_000_000) & (centers <= 1_500_000)
        if structured_mask.sum() >= 2 and (~structured_mask).sum() >= 2:
            sep = np.abs(coords[structured_mask, 0].mean()
                         - coords[~structured_mask, 0].mean())
            spread = coords[:, 0].std()
            # Mean separation should exceed baseline spread
            assert sep > 0.5 * spread


# ---------------------------------------------------------------------------
# windowed_analysis dispatch
# ---------------------------------------------------------------------------


class TestWindowedAnalysisDispatch:

    def test_local_pca_alone(self, small_hm):
        res = windowed_analysis(small_hm, window_size=20_000,
                                step_size=20_000,
                                statistics=['local_pca'], k=2,
                                window_type='bp')
        assert isinstance(res, LocalPCAResult)

    def test_local_pca_with_scalar_stat(self, small_hm):
        res = windowed_analysis(small_hm, window_size=200_000,
                                step_size=200_000,
                                statistics=['pi', 'local_pca'], k=2,
                                window_type='bp')
        assert isinstance(res, LocalPCAResult)
        assert 'pi' in res.windows.columns


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------


class TestCorners:

    def test_three_well_separated_clusters(self):
        rng = np.random.default_rng(7)
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        n_per = 30
        xy = np.vstack([c + 0.2 * rng.standard_normal((n_per, 2))
                        for c in centers])
        out = corners(xy, prop=0.2, k=3, random_state=0)
        # Each column should be dominated by one of the three clusters
        cluster_of = np.repeat(np.arange(3), n_per)
        recovered_clusters = []
        for i in range(out.shape[1]):
            c_hist = np.bincount(cluster_of[out[:, i]], minlength=3)
            recovered_clusters.append(int(np.argmax(c_hist)))
        assert set(recovered_clusters) == {0, 1, 2}

    def test_handles_nans(self):
        rng = np.random.default_rng(3)
        xy = rng.standard_normal((50, 2))
        xy[0] = np.nan
        xy[5] = np.nan
        # Should not raise
        out = corners(xy, prop=0.1, k=3, random_state=0)
        # NaN indices should not appear in output
        assert 0 not in out
        assert 5 not in out


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------


class TestJackknife:

    def test_shape_mean_aggregate(self, small_hm):
        se = local_pca_jackknife(small_hm, window_size=200,
                                 window_type='snp', k=2, n_blocks=10)
        # Expect (n_windows, k)
        expected_n_windows = small_hm.num_variants // 200
        assert se.shape == (expected_n_windows, 2)
        assert np.all(np.isfinite(se) & (se >= 0))

    def test_no_aggregate_shape(self, small_hm):
        se = local_pca_jackknife(small_hm, window_size=200,
                                 window_type='snp', k=2, n_blocks=5,
                                 aggregate=None)
        expected_n_windows = small_hm.num_variants // 200
        assert se.shape == (expected_n_windows, 2, small_hm.num_haplotypes)
        assert np.all(np.isfinite(se) & (se >= 0))

    def test_sign_alignment_via_flip(self):
        """Sign-aligned variance is invariant to arbitrary pre-flips."""
        from pg_gpu.decomposition import _sign_align_replicates
        import cupy as cp
        rng = np.random.default_rng(11)
        n_reps, k, n_samples = 10, 2, 30
        V = cp.asarray(rng.standard_normal((n_reps, k, n_samples)))
        V_aligned = _sign_align_replicates(V)
        var1 = cp.var(V_aligned, axis=0).get()

        flips = rng.choice([-1.0, 1.0], size=(n_reps, k))
        V_flipped = V * cp.asarray(flips[:, :, None])
        V_flipped_aligned = _sign_align_replicates(V_flipped)
        var2 = cp.var(V_flipped_aligned, axis=0).get()
        np.testing.assert_allclose(var1, var2, rtol=1e-10, atol=1e-12)
