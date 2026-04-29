"""
Failing tests for the streaming + (optional) randomized local-PCA engine.

This file is the Step 1 / Step 3 contract artefact for the
``feat/local-pca-streaming-randomized`` branch (see
``docs/local_pca_streaming_plan.md``). The tests are written
*before* the implementation lands so each passing test pins down a
piece of the new contract:

Per-window kernel (``_local_pca_window_dense``,
``_local_pca_window_rsvd``)
  * top-k eigvals + eigvecs match exact ``cp.linalg.eigh`` on the
    Gram (dense) or within a randomized-method tolerance (rsvd)
  * ``sumsq`` matches ``(C ** 2).sum()`` where C is the window
    Gram, exactly for the dense path and within an explicit
    rtol budget for the rsvd path
  * thin-window edge cases (n_var < k + oversample) fall back to
    deterministic eigh

Streaming dispatcher (``_streaming_local_pca``)
  * eigvals / sumsq match the existing ``local_pca`` output for
    the dense-eigh engine within numerical roundoff
  * results are invariant to ``tile_size`` (equivalently,
    streaming + tile_size=N == today's all-in-memory path for the
    same data)
  * subspace alignment matches existing ``local_pca`` output via
    Procrustes (handles sign / rotation freedom)
  * downstream ``pc_dist`` and ``corners`` are unchanged

End-to-end
  * ``lostruct(..., engine='streaming-dense')`` recovers the same
    sweep windows as the legacy engine on a small simulated
    sweep dataset

These tests intentionally fail at branch HEAD; they should
all pass after Step 5.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from pg_gpu import HaplotypeMatrix  # noqa: E402
from pg_gpu.decomposition import (  # noqa: E402
    LocalPCAResult,
    local_pca,
    lostruct,
)

# These names DO NOT exist yet at HEAD; importing them will fail until
# Step 2 / Step 3 land. Intentional.
streaming_helpers = pytest.importorskip(
    "pg_gpu.decomposition",
    reason="streaming helpers not yet implemented",
)


def _maybe_attr(obj, name):
    """Return obj.name or raise pytest.skip with a clear message.

    Lets the parts of this file that depend on later-step helpers skip
    cleanly while earlier-step tests still run as soon as their helper
    lands.
    """
    try:
        return getattr(obj, name)
    except AttributeError:
        pytest.skip(
            f"{name} not yet implemented (see "
            "docs/local_pca_streaming_plan.md)")


# ---------------------------------------------------------------------------
# Subspace / sign helpers
# ---------------------------------------------------------------------------


def _subspace_misalignment(U, V):
    """Sin theta between two row-stacked orthonormal frames.

    Each of U, V has shape (k, n). Returns ||I - U V^T (V U^T)||_F /
    sqrt(k), which is 0 when the two frames span the same subspace.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    M = U @ V.T
    # Singular values of M are cos(theta_i); 1 - sigma^2 = sin^2(theta).
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return float(np.sqrt(np.maximum(0.0, 1.0 - sigma ** 2).sum()))


def _exact_topk_from_gram(C_cpu, k):
    """Reference top-k eigenpairs of a small Gram matrix on CPU."""
    evals, evecs = np.linalg.eigh(C_cpu)
    order = np.argsort(evals)[::-1][:k]
    return evals[order], evecs[:, order].T  # (k,), (k, n)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_psd_block():
    """Centred GPU haplotype block with a known low-rank-plus-noise Gram."""
    rng = np.random.default_rng(42)
    n_hap, n_var = 50, 200
    # Two strong directions + noise -- gives a clear top-2 spectrum.
    L = rng.standard_normal((n_hap, 2)).astype(np.float64)
    R = rng.standard_normal((2, n_var)).astype(np.float64)
    noise = 0.05 * rng.standard_normal((n_hap, n_var))
    X = L @ R + noise
    X = X - X.mean(axis=1, keepdims=True)
    return cupy.asarray(X), n_var


@pytest.fixture
def thin_block():
    """Window with fewer variants than k+oversample -> exact-eigh fallback."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))   # n_var_w = 5 < 2 + 8
    X = X - X.mean(axis=1, keepdims=True)
    return cupy.asarray(X), 5


@pytest.fixture
def small_hm():
    """40 haplotypes x 2000 variants for streaming/parity tests."""
    rng = np.random.default_rng(42)
    n_hap, n_var = 40, 2000
    hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return HaplotypeMatrix(hap, pos, 0, n_var * 1000)


@pytest.fixture
def structured_hm():
    """Two-deme matrix with a structured region; exercises lostruct corners."""
    rng = np.random.default_rng(0)
    n_hap_per = 20
    n_hap = 2 * n_hap_per
    n_var = 3000
    hap = rng.integers(0, 2, (n_hap, n_var), dtype=np.int8)
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
# Step 2.A: per-window dense kernel (`_local_pca_window_dense`)
# ---------------------------------------------------------------------------


class TestWindowDenseKernel:
    """Per-window helper that materialises the Gram transiently and returns
    exact top-k eigenpairs plus the Frobenius-squared sumsq."""

    def test_eigvals_match_exact_eigh(self, random_psd_block):
        """Top-k eigvals match a reference CPU eigh on the same Gram."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_dense")
        X, n_var = random_psd_block
        eigvals_gpu, eigvecs_gpu, sumsq_gpu = helper(X, n_var, k=2)

        # Reference: form Gram on CPU and take exact top-k.
        X_cpu = cupy.asnumpy(X)
        C = X_cpu @ X_cpu.T / max(n_var - 1, 1)
        ref_vals, ref_vecs = _exact_topk_from_gram(C, k=2)

        np.testing.assert_allclose(cupy.asnumpy(eigvals_gpu), ref_vals,
                                    rtol=1e-10, atol=1e-12)
        # Subspace match. Tolerance is 1e-6 rather than 1e-8 because the
        # GPU eigh and the CPU reference take FP64 sums in different
        # orders, so individual eigenvector components disagree at the
        # ~1e-8 level even when the subspaces are numerically identical.
        misalign = _subspace_misalignment(cupy.asnumpy(eigvecs_gpu), ref_vecs)
        assert misalign < 1e-6

    def test_sumsq_matches_frobenius_squared(self, random_psd_block):
        """Returned sumsq equals (C ** 2).sum() bit-for-bit (single FP path)."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_dense")
        X, n_var = random_psd_block
        _, _, sumsq_gpu = helper(X, n_var, k=2)

        X_cpu = cupy.asnumpy(X)
        C = X_cpu @ X_cpu.T / max(n_var - 1, 1)
        ref_sumsq = float((C ** 2).sum())

        np.testing.assert_allclose(float(cupy.asnumpy(sumsq_gpu)), ref_sumsq,
                                    rtol=1e-10, atol=1e-12)

    def test_thin_window_returns_valid_result(self, thin_block):
        """When n_var < k+oversample, the dense path still works exactly."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_dense")
        X, n_var = thin_block
        eigvals, eigvecs, sumsq = helper(X, n_var, k=2)
        assert cupy.asnumpy(eigvals).shape == (2,)
        # eigvals descending; non-NaN
        vals = cupy.asnumpy(eigvals)
        assert np.all(np.isfinite(vals))
        assert vals[0] >= vals[1]

    def test_returns_eigvecs_in_rows(self, random_psd_block):
        """Eigvecs come out shape (k, n_hap), matching _batched_top_k_eigh."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_dense")
        X, n_var = random_psd_block
        eigvals, eigvecs, _ = helper(X, n_var, k=2)
        n_hap = X.shape[0]
        assert cupy.asnumpy(eigvecs).shape == (2, n_hap)


# ---------------------------------------------------------------------------
# Step 2.B: per-window randomized kernel (`_local_pca_window_rsvd`)
# ---------------------------------------------------------------------------


class TestWindowRsvdKernel:
    """Randomized SVD on the centred X_w. Approximate sumsq, top-k accurate
    to randomized-method tolerance with sufficient oversampling."""

    def _seeded_rng(self):
        return cupy.random.RandomState(seed=12345)

    def test_eigvals_match_exact_within_rsvd_tolerance(self, random_psd_block):
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_rsvd")
        X, n_var = random_psd_block
        rng = self._seeded_rng()
        eigvals_gpu, eigvecs_gpu, _ = helper(
            X, n_var, k=2, oversample=8, n_iter=1, rng=rng)

        X_cpu = cupy.asnumpy(X)
        C = X_cpu @ X_cpu.T / max(n_var - 1, 1)
        ref_vals, _ = _exact_topk_from_gram(C, k=2)

        np.testing.assert_allclose(cupy.asnumpy(eigvals_gpu), ref_vals,
                                    rtol=5e-3, atol=1e-6)

    def test_subspace_alignment_within_tolerance(self, random_psd_block):
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_rsvd")
        X, n_var = random_psd_block
        rng = self._seeded_rng()
        _, eigvecs_gpu, _ = helper(
            X, n_var, k=2, oversample=8, n_iter=1, rng=rng)

        X_cpu = cupy.asnumpy(X)
        C = X_cpu @ X_cpu.T / max(n_var - 1, 1)
        _, ref_vecs = _exact_topk_from_gram(C, k=2)

        misalign = _subspace_misalignment(cupy.asnumpy(eigvecs_gpu), ref_vecs)
        assert misalign < 5e-3

    def test_recovers_planted_low_rank_structure(self):
        """A rank-2 ground truth with small noise -> rsvd recovers both directions."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_rsvd")
        rng_np = np.random.default_rng(7)
        n_hap, n_var = 80, 500
        L = rng_np.standard_normal((n_hap, 2)).astype(np.float64)
        # Planted directions: orthogonalise so they're recoverable separately.
        Lq, _ = np.linalg.qr(L)
        scales = np.array([5.0, 3.0])
        L_planted = Lq * scales
        R = rng_np.standard_normal((2, n_var)).astype(np.float64)
        noise = 0.01 * rng_np.standard_normal((n_hap, n_var))
        X = L_planted @ R + noise
        X = X - X.mean(axis=1, keepdims=True)
        X_gpu = cupy.asarray(X)

        rng = self._seeded_rng()
        eigvals, eigvecs, _ = helper(X_gpu, n_var, k=2,
                                     oversample=8, n_iter=1, rng=rng)

        # Top-2 eigvecs of X X^T should align with span(Lq).
        misalign = _subspace_misalignment(cupy.asnumpy(eigvecs), Lq.T)
        assert misalign < 1e-2

    def test_thin_window_falls_back_to_dense(self, thin_block):
        """n_var < k+oversample -> rsvd falls back to deterministic eigh."""
        rsvd = _maybe_attr(streaming_helpers, "_local_pca_window_rsvd")
        dense = _maybe_attr(streaming_helpers, "_local_pca_window_dense")
        X, n_var = thin_block
        rng = cupy.random.RandomState(seed=12345)
        rsvd_vals, rsvd_vecs, rsvd_sumsq = rsvd(
            X, n_var, k=2, oversample=8, n_iter=1, rng=rng)
        dense_vals, dense_vecs, dense_sumsq = dense(X, n_var, k=2)

        np.testing.assert_allclose(cupy.asnumpy(rsvd_vals),
                                    cupy.asnumpy(dense_vals),
                                    rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(float(cupy.asnumpy(rsvd_sumsq)),
                                    float(cupy.asnumpy(dense_sumsq)),
                                    rtol=1e-10, atol=1e-12)

    def test_sumsq_within_rsvd_tolerance(self, random_psd_block):
        """rsvd's sumsq estimator is within rtol=5e-3 of the exact Frobenius²."""
        helper = _maybe_attr(streaming_helpers, "_local_pca_window_rsvd")
        X, n_var = random_psd_block
        rng = self._seeded_rng()
        _, _, sumsq_gpu = helper(
            X, n_var, k=2, oversample=20, n_iter=1, rng=rng)

        X_cpu = cupy.asnumpy(X)
        C = X_cpu @ X_cpu.T / max(n_var - 1, 1)
        ref_sumsq = float((C ** 2).sum())

        np.testing.assert_allclose(float(cupy.asnumpy(sumsq_gpu)), ref_sumsq,
                                    rtol=5e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Step 3: streaming dispatcher (`engine='streaming-dense'` and
# `engine='streaming-rsvd'`).
# ---------------------------------------------------------------------------


class TestStreamingDispatcher:
    """``local_pca(... engine='streaming-dense')`` should be numerically
    equivalent to today's path; the rsvd engine matches under tolerance."""

    def test_streaming_dense_matches_legacy_eigvals(self, structured_hm):
        legacy = local_pca(structured_hm, window_size=300,
                            window_type='snp', k=2)
        try:
            new = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")
        np.testing.assert_allclose(new.eigvals, legacy.eigvals,
                                    rtol=1e-8, atol=1e-10)

    def test_streaming_dense_matches_legacy_sumsq(self, structured_hm):
        legacy = local_pca(structured_hm, window_size=300,
                            window_type='snp', k=2)
        try:
            new = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")
        np.testing.assert_allclose(new.sumsq, legacy.sumsq,
                                    rtol=1e-8, atol=1e-10)

    def test_streaming_dense_subspace_alignment_matches_legacy(self,
                                                                structured_hm):
        legacy = local_pca(structured_hm, window_size=300,
                            window_type='snp', k=2)
        try:
            new = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")
        for w in range(legacy.n_windows):
            if not np.all(np.isfinite(legacy.eigvecs[w])):
                continue
            misalign = _subspace_misalignment(
                new.eigvecs[w], legacy.eigvecs[w])
            assert misalign < 1e-6, f"window {w}: {misalign}"

    def test_tile_size_invariance(self, structured_hm):
        """Streaming dispatcher results don't depend on the tile size."""
        try:
            small_tile = local_pca(structured_hm, window_size=300,
                                    window_type='snp', k=2,
                                    engine='streaming-dense', tile_size=4)
            big_tile = local_pca(structured_hm, window_size=300,
                                  window_type='snp', k=2,
                                  engine='streaming-dense', tile_size=64)
        except TypeError:
            pytest.skip("engine= / tile_size= kwargs not yet implemented")
        np.testing.assert_allclose(small_tile.eigvals, big_tile.eigvals,
                                    rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(small_tile.sumsq, big_tile.sumsq,
                                    rtol=1e-10, atol=1e-12)

    def test_pc_dist_unchanged_under_streaming_dense(self, structured_hm):
        from pg_gpu.decomposition import pc_dist
        legacy = local_pca(structured_hm, window_size=300,
                            window_type='snp', k=2)
        try:
            new = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")
        d_legacy = pc_dist(legacy)
        d_new = pc_dist(new)
        # atol=1e-6 absorbs FP-roundoff differences on the
        # mathematically-zero diagonal entries (per-window eigh and
        # batched eigh accumulate roundoff in different orders).
        np.testing.assert_allclose(d_new, d_legacy, rtol=1e-6, atol=1e-6)

    def test_streaming_rsvd_subspace_within_tolerance(self, structured_hm):
        """Streaming-rsvd matches legacy subspace per window under
        Procrustes/sin-theta tolerance, but only on windows that have
        a genuine spectral gap above the noise floor.

        Pure-noise windows (where the top eigenvalues are within
        roundoff of each other) have arbitrary eigenvectors, so any
        algorithm-to-algorithm comparison is meaningless on them.
        Restrict the assertion to windows where the legacy top-1
        eigenvalue exceeds the bulk by a clear factor.
        """
        legacy = local_pca(structured_hm, window_size=300,
                            window_type='snp', k=2)
        try:
            new = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-rsvd', oversample=10)
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")

        # Only test windows whose leading eigenvalue is well-isolated
        # from the bulk. structured_hm has a planted frequency
        # gradient between variants 1000 and 1500; windows overlapping
        # that region are the only ones with an unambiguous top-1
        # direction. For windows where eigval[1] is itself in a
        # near-degenerate band with the rest of the spectrum the
        # second eigenvector is mathematically ill-defined and rsvd
        # vs. dense will pick different (equally valid) representatives,
        # so we restrict the rsvd parity check to the leading direction.
        finite = np.all(np.isfinite(legacy.eigvecs.reshape(legacy.n_windows, -1)),
                         axis=1)
        ratio = legacy.eigvals[:, 0] / np.maximum(legacy.eigvals[:, 1], 1e-30)
        structural = finite & (ratio > 2.0)
        assert structural.any(), "no test windows have a >2x leading-eigval gap"
        for w in np.where(structural)[0]:
            leading = abs(float(legacy.eigvecs[w, 0] @ new.eigvecs[w, 0]))
            assert leading > 0.99, f"window {w}: leading dot = {leading}"


# ---------------------------------------------------------------------------
# End-to-end / lostruct
# ---------------------------------------------------------------------------


class TestLostructStreamingEndToEnd:

    def test_streaming_dense_corners_match_legacy(self, structured_hm):
        """Lostruct corner indices should be set-equal between engines."""
        legacy = lostruct(structured_hm, window_size=300, window_type='snp',
                           k=2, n_corners=2, corner_prop=0.1, random_state=0)
        try:
            new = lostruct(structured_hm, window_size=300, window_type='snp',
                            k=2, n_corners=2, corner_prop=0.1,
                            random_state=0, engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")
        legacy_set = set(map(tuple, np.sort(legacy.corner_indices, axis=0).T))
        new_set = set(map(tuple, np.sort(new.corner_indices, axis=0).T))
        assert legacy_set == new_set


# ---------------------------------------------------------------------------
# Issue #91 regression: per-window output host syncs must not return
# ---------------------------------------------------------------------------


class TestStreamingHostSyncCount:
    """The streaming dispatcher must NOT issue per-window host transfers
    for its three outputs (eigvals / eigvecs / sumsq). Issue #91:
    per-window ``cp.asnumpy(...)`` calls saturated host CPU on whole-
    genome scans. The fix routes per-window outputs into GPU-resident
    buffers and does a single bulk ``.get()`` after the loop; this test
    pins that invariant.

    The dispatcher path also includes pre-existing infrastructure host
    syncs that are NOT part of issue #91 -- ``WindowIterator.get_subset``
    does an ``.all()`` bounds check (2 transfers/window) and
    ``_prepare_matrix`` does a has-missing check (1 transfer/window).
    Plus 1 fixed transfer from iterator init and 3 fixed bulk transfers
    at the end of the dispatcher. So the post-fix expected rate is
    3 transfers/window + ~4 fixed = ~3.4 per window for the
    ``structured_hm`` fixture (10 windows).

    A regression that re-introduces even ONE of the three per-window
    output syncs would push the rate to >=4 per window; restoring all
    three would push it to ~6.4 per window. Either case fails the
    3.5-per-window assertion below.

    If the unrelated infrastructure syncs in ``get_subset`` /
    ``_prepare_matrix`` are later optimized away, the post-fix rate
    drops further and this test still passes.
    """

    def test_dispatcher_per_window_output_transfers_stay_constant(
            self, structured_hm, monkeypatch):
        import cupy as cp

        n_get = {"calls": 0}
        n_asnumpy = {"calls": 0}

        real_get = cp.ndarray.get
        real_asnumpy = cp.asnumpy

        def counted_get(self, *a, **kw):
            n_get["calls"] += 1
            return real_get(self, *a, **kw)

        def counted_asnumpy(arr, *a, **kw):
            n_asnumpy["calls"] += 1
            return real_asnumpy(arr, *a, **kw)

        monkeypatch.setattr(cp.ndarray, "get", counted_get)
        monkeypatch.setattr(cp, "asnumpy", counted_asnumpy)

        try:
            res = local_pca(structured_hm, window_size=300,
                             window_type='snp', k=2,
                             engine='streaming-dense')
        except TypeError:
            pytest.skip("engine= kwarg not yet implemented")

        assert res.n_windows >= 5, (
            "fixture must have >=5 windows for this ratio test to be meaningful")

        total = n_get["calls"] + n_asnumpy["calls"]
        per_window = total / res.n_windows

        # Pre-fix would be ~6.4/window; partial regression (one output
        # sync re-introduced) is ~4.4/window. Current post-fix is
        # ~3.4/window. Set the ceiling at 3.5 to catch both partial
        # and full regressions while leaving room for the infrastructure
        # syncs the dispatcher legitimately performs.
        assert per_window <= 3.5, (
            f"streaming dispatcher made {total} host transfers "
            f"({n_get['calls']} .get(), {n_asnumpy['calls']} cp.asnumpy) "
            f"over {res.n_windows} windows -- {per_window:.2f}/window "
            f"is above the 3.5/window ceiling. Regression of issue #91? "
            f"Per-window output syncs (eigvals/eigvecs/sumsq) should be "
            f"O(1) total, not O(n_windows).")
