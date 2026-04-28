"""Parity tests for haplotype count-based r, r^2, D' fused CUDA kernels.

The polynomial elementwise expressions previously living inline in
``pg_gpu.ld_statistics.r``, ``r_squared``, and ``d_prime`` were replaced
by single-launch RawKernels (``_R_KERN``, ``_R_SQUARED_KERN``,
``_D_PRIME_KERN``) in ``pg_gpu.haplotype_kernels``. These tests verify
that the new kernel-backed entry points produce the same output as the
old polynomial expressions on the same inputs.
"""

import cupy as cp
import numpy as np
import pytest

from pg_gpu import ld_statistics


def _polynomial_r(counts, n_valid=None):
    """Old elementwise-polynomial path, kept here as the parity reference."""
    c11, c10, c01, c00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = (n_valid.astype(cp.float64) if n_valid is not None
         else cp.sum(counts, axis=1).astype(cp.float64))
    p_A = (c11 + c10) / n
    p_B = (c11 + c01) / n
    D = (c11 * c00 - c10 * c01).astype(cp.float64) / (n * n)
    denom = p_A * (1 - p_A) * p_B * (1 - p_B)
    out = cp.full(n.shape[0], cp.nan, dtype=cp.float64)
    valid = denom > 0
    out[valid] = D[valid] / cp.sqrt(denom[valid])
    return out


def _polynomial_r_squared(counts, n_valid=None):
    return _polynomial_r(counts, n_valid) ** 2


def _polynomial_d_prime(counts, n_valid=None):
    c11, c10, c01, c00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = (n_valid.astype(cp.float64) if n_valid is not None
         else cp.sum(counts, axis=1).astype(cp.float64))
    p_A = (c11 + c10) / n
    q_A = 1.0 - p_A
    p_B = (c11 + c01) / n
    q_B = 1.0 - p_B
    D = (c11 * c00 - c10 * c01).astype(cp.float64) / (n * n)
    Dmax = cp.where(D >= 0,
                    cp.minimum(p_A * q_B, q_A * p_B),
                    cp.minimum(p_A * p_B, q_A * q_B))
    out = cp.full(n.shape[0], cp.nan, dtype=cp.float64)
    valid = Dmax > 0
    out[valid] = D[valid] / Dmax[valid]
    return out


def _random_counts(n_pairs, n_total=200, seed=0):
    """Random multinomial-ish 4-way haplotype counts."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([1, 1, 1, 1], size=n_pairs)
    counts = np.round(raw * n_total).astype(np.int64)
    return cp.asarray(counts.astype(np.float64))


def _check_finite_match(kernel_out, ref):
    kern = kernel_out.get()
    poly = ref.get()
    finite = np.isfinite(kern) & np.isfinite(poly)
    assert np.array_equal(np.isnan(kern), np.isnan(poly))
    np.testing.assert_allclose(
        kern[finite], poly[finite], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
class TestRParity:

    def test_r_random_panels(self, seed):
        counts = _random_counts(200, seed=seed)
        _check_finite_match(ld_statistics.r(counts), _polynomial_r(counts))

    def test_r_squared_random_panels(self, seed):
        counts = _random_counts(200, seed=seed)
        _check_finite_match(
            ld_statistics.r_squared(counts), _polynomial_r_squared(counts))

    def test_d_prime_random_panels(self, seed):
        counts = _random_counts(200, seed=seed)
        _check_finite_match(
            ld_statistics.d_prime(counts), _polynomial_d_prime(counts))


class TestEdgeCases:

    def test_monomorphic_pair_yields_nan(self):
        # n10 = n11 = 0 -> p_A = 0 -> denom = 0 -> NaN.
        counts = cp.asarray([[0.0, 0.0, 50.0, 50.0]])
        assert cp.all(cp.isnan(ld_statistics.r(counts)))
        assert cp.all(cp.isnan(ld_statistics.r_squared(counts)))
        assert cp.all(cp.isnan(ld_statistics.d_prime(counts)))

    def test_perfect_positive_correlation(self):
        # Only (1,1) and (0,0) -> r=1, r^2=1, D'=1.
        counts = cp.asarray([[50.0, 0.0, 0.0, 50.0]])
        np.testing.assert_allclose(ld_statistics.r(counts).get(), [1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(ld_statistics.r_squared(counts).get(),
                                   [1.0], atol=1e-12)
        np.testing.assert_allclose(ld_statistics.d_prime(counts).get(),
                                   [1.0], atol=1e-12)

    def test_perfect_negative_correlation(self):
        # Only (1,0) and (0,1) -> r=-1, r^2=1, D'=-1.
        counts = cp.asarray([[0.0, 50.0, 50.0, 0.0]])
        np.testing.assert_allclose(ld_statistics.r(counts).get(), [-1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(ld_statistics.r_squared(counts).get(),
                                   [1.0], atol=1e-12)
        np.testing.assert_allclose(ld_statistics.d_prime(counts).get(),
                                   [-1.0], atol=1e-12)

    def test_with_n_valid_override(self):
        counts = _random_counts(50, seed=99)
        n_valid = cp.sum(counts, axis=1)  # happens to equal the default
        kern = ld_statistics.r_squared(counts, n_valid).get()
        ref = _polynomial_r_squared(counts, n_valid).get()
        finite = np.isfinite(kern) & np.isfinite(ref)
        np.testing.assert_allclose(
            kern[finite], ref[finite], rtol=1e-12, atol=1e-12)
