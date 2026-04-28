"""Parity tests for the diploid DD fused CUDA kernels.

Verify that ``_DD_GENO_SINGLE_KERN`` and ``_DD_GENO_BETWEEN_KERN``
in ``pg_gpu.genotype_kernels`` produce the same per-pair output
as the reference polynomial implementations
``ld_statistics_genotype.dd_geno_single`` and ``dd_geno_between``,
to floating-point precision (rtol=1e-12).
"""

import cupy as cp
import numpy as np
import pytest

from pg_gpu.genotype_kernels import (
    _DD_GENO_SINGLE_KERN,
    _DD_GENO_BETWEEN_KERN,
    _PopDataGeno,
    _launch,
    compute_all_dd_geno,
)
from pg_gpu.ld_pipeline import compute_genotype_counts_for_pairs
from pg_gpu import GenotypeMatrix
from pg_gpu import ld_statistics_genotype as ldg


def _random_gm(n_indiv, n_var, seed=0):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.95, size=n_var)
    a1 = rng.binomial(1, af[None, :], size=(n_indiv, n_var))
    a2 = rng.binomial(1, af[None, :], size=(n_indiv, n_var))
    geno = (a1 + a2).astype(np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return GenotypeMatrix(geno, pos, 0, n_var * 1000)


def _build_pop_data(gm, idx_i=None, idx_j=None):
    """Build a single _PopDataGeno from all-pairs upper-triangle counts."""
    if gm.device != 'GPU':
        gm.transfer_to_gpu()
    g = gm.genotypes
    n_var = g.shape[1]
    if idx_i is None or idx_j is None:
        i, j = cp.triu_indices(n_var, k=1)
        idx_i = i.astype(cp.int32)
        idx_j = j.astype(cp.int32)
    counts, n_valid = compute_genotype_counts_for_pairs(g, idx_i, idx_j)
    return _PopDataGeno(counts, n_valid)


def _kernel_dd_single(p):
    """Run _DD_GENO_SINGLE_KERN over a single _PopDataGeno."""
    g = cp.ascontiguousarray(cp.stack(
        [p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9], axis=-1))
    N = g.shape[0]
    out = cp.empty(N, dtype=cp.float64)
    _launch(_DD_GENO_SINGLE_KERN, (g, out, N), N)
    return out


def _kernel_dd_between(pi, pj):
    """Run _DD_GENO_BETWEEN_KERN over two _PopDataGeno objects."""
    D = cp.ascontiguousarray(cp.concatenate([pi.D_geno, pj.D_geno]))
    nn = cp.ascontiguousarray(cp.concatenate([pi.n, pj.n]))
    N = pi.D_geno.shape[0]
    I = cp.arange(N, dtype=cp.int32)
    J = cp.arange(N, 2 * N, dtype=cp.int32)
    out = cp.empty(N, dtype=cp.float64)
    _launch(_DD_GENO_BETWEEN_KERN, (D, nn, I, J, out, N), N)
    return out


# ---------------------------------------------------------------------------
# Single-pop parity
# ---------------------------------------------------------------------------


class TestSinglePopParity:

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_random_panels(self, seed):
        gm = _random_gm(40, 30, seed=seed)
        p = _build_pop_data(gm)
        kern = _kernel_dd_single(p).get()
        poly = ldg.dd_geno_single(p).get()
        np.testing.assert_allclose(kern, poly, rtol=1e-12, atol=1e-12)

    def test_small_sample_size(self):
        # n=4 is the smallest valid case; verify both paths agree there.
        gm = _random_gm(4, 15, seed=99)
        p = _build_pop_data(gm)
        kern = _kernel_dd_single(p).get()
        poly = ldg.dd_geno_single(p).get()
        np.testing.assert_allclose(kern, poly, rtol=1e-12, atol=1e-12)

    def test_larger_panel(self):
        gm = _random_gm(100, 60, seed=2026)
        p = _build_pop_data(gm)
        kern = _kernel_dd_single(p).get()
        poly = ldg.dd_geno_single(p).get()
        np.testing.assert_allclose(kern, poly, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Between-pop parity
# ---------------------------------------------------------------------------


class TestBetweenPopParity:

    @pytest.mark.parametrize("seed", [3, 11, 100])
    def test_random_panels(self, seed):
        gm1 = _random_gm(30, 25, seed=seed)
        gm2 = _random_gm(30, 25, seed=seed + 1000)
        p1 = _build_pop_data(gm1)
        p2 = _build_pop_data(gm2)
        kern = _kernel_dd_between(p1, p2).get()
        poly = ldg.dd_geno_between(p1, p2).get()
        np.testing.assert_allclose(kern, poly, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# End-to-end: compute_all_dd_geno dispatch
# ---------------------------------------------------------------------------


class TestComputeAllDispatch:

    def test_mixed_calls_match_polynomial(self):
        """compute_all_dd_geno (now kernel-backed) must agree with
        the prior polynomial dispatch on the same inputs."""
        gm1 = _random_gm(30, 20, seed=4)
        gm2 = _random_gm(30, 20, seed=5)
        gm3 = _random_gm(30, 20, seed=6)
        # Each pop needs its own pair-counts so all three _PopDataGeno
        # objects share the same N (number of pairs).
        ps = [_build_pop_data(gm) for gm in (gm1, gm2, gm3)]

        dd_calls = [(0, 0), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2)]
        kernel_results = compute_all_dd_geno(ps, dd_calls)

        for k, (i, j) in enumerate(dd_calls):
            if i == j:
                expected = ldg.dd_geno_single(ps[i]).get()
            else:
                expected = ldg.dd_geno_between(ps[i], ps[j]).get()
            np.testing.assert_allclose(
                kernel_results[k].get(), expected,
                rtol=1e-12, atol=1e-12,
                err_msg=f"call {k} (i={i}, j={j}) disagrees")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_n_below_4_returns_zero(self):
        """Both kernel and polynomial return 0 when n<4 (denominator
        guard). Build a small fixture where the per-pair valid sample
        count is below 4 to exercise the early-return branch."""
        n_indiv, n_var = 3, 4
        rng = np.random.default_rng(0)
        geno = rng.integers(0, 3, (n_indiv, n_var)).astype(np.int8)
        pos = np.arange(n_var) * 100
        gm = GenotypeMatrix(geno, pos, 0, n_var * 100)
        p = _build_pop_data(gm)
        kern = _kernel_dd_single(p).get()
        poly = ldg.dd_geno_single(p).get()
        assert np.all(kern == 0.0)
        assert np.all(poly == 0.0)
