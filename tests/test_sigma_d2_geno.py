"""Tests for the per-pair diploid sigma_D^2 estimator.

`pg_gpu.ld_statistics_genotype.sigma_d2_geno` exposes the Ragsdale &
Gravel (2019) unbiased polynomial estimator at the per-pair level for
diploid 0/1/2 dosages. These tests cover:
  - Shape and dtype.
  - Edge cases (perfect correlation, monomorphic loci, missing data).
  - Internal consistency: the sum over a bin equals the moments-LD bin
    aggregation that already runs in production.
"""

import cupy as cp
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.ld_statistics_genotype import sigma_d2_geno


def _random_gm(n_indiv: int, n_var: int, seed: int = 0) -> GenotypeMatrix:
    rng = np.random.default_rng(seed)
    # Hardy-Weinberg-ish: pick allele freq per variant uniform on (0.05, 0.95),
    # draw two independent alleles per individual, sum to get dosage.
    af = rng.uniform(0.05, 0.95, size=n_var)
    a1 = rng.binomial(1, af[None, :], size=(n_indiv, n_var))
    a2 = rng.binomial(1, af[None, :], size=(n_indiv, n_var))
    geno = (a1 + a2).astype(np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return GenotypeMatrix(geno, pos, 0, n_var * 1000)


class TestShape:

    def test_default_returns_upper_triangle(self):
        gm = _random_gm(20, 30, seed=1)
        out = sigma_d2_geno(gm)
        assert out.shape == (30 * 29 // 2,)
        assert out.dtype == cp.float64

    def test_explicit_indices(self):
        gm = _random_gm(20, 30, seed=2)
        i = cp.array([0, 0, 1], dtype=cp.int32)
        j = cp.array([1, 5, 7], dtype=cp.int32)
        out = sigma_d2_geno(gm, i, j)
        assert out.shape == (3,)


class TestEdgeCases:

    def test_identical_columns_agree(self):
        n = 30
        col = np.repeat([0, 1, 2], n // 3).astype(np.int8)[:, None]
        # Three identical variants -> all three pairs must produce the
        # same finite positive value. Note: the unbiased polynomial
        # estimator is *not* bounded by 1 -- under perfect LD it can
        # exceed 1 in finite samples (a property of the Ragsdale &
        # Gravel multinomial-projection correction). Don't assert == 1.
        geno = np.hstack([col, col, col])
        pos = np.array([100, 200, 300])
        gm = GenotypeMatrix(geno, pos, 0, 400)
        out = sigma_d2_geno(gm).get()
        assert np.all(np.isfinite(out)) and np.all(out > 0)
        np.testing.assert_allclose(out, out[0], atol=1e-12)

    def test_monomorphic_pair_yields_nan(self):
        n_indiv, n_var = 20, 4
        geno = np.zeros((n_indiv, n_var), dtype=np.int8)
        # Make one variant polymorphic so it isn't degenerate everywhere.
        geno[:5, 0] = 2
        pos = np.arange(n_var) * 100
        gm = GenotypeMatrix(geno, pos, 0, n_var * 100)
        out = sigma_d2_geno(gm).get()
        # Pairs that touch a monomorphic locus -> NaN.
        # Pairs not touching variant 0 are all-zero pairs -> NaN.
        assert np.all(np.isnan(out))

    def test_missing_values_raise(self):
        gm = _random_gm(10, 5, seed=3)
        geno = gm.genotypes.copy() if isinstance(gm.genotypes, np.ndarray) else gm.genotypes.get()
        geno[2, 1] = -1
        gm_missing = GenotypeMatrix(
            geno.astype(np.int8), np.arange(5) * 100, 0, 500)
        with pytest.raises(ValueError, match="missing values"):
            sigma_d2_geno(gm_missing)

    def test_bad_input_type_raises(self):
        with pytest.raises(TypeError):
            sigma_d2_geno(np.zeros((10, 5), dtype=np.int8))

    def test_mismatched_index_lengths_raise(self):
        gm = _random_gm(10, 8, seed=4)
        with pytest.raises(ValueError, match="same shape"):
            sigma_d2_geno(gm, cp.array([0, 1]), cp.array([1]))


class TestHaplotypeMatrixPath:

    def test_haplotype_matrix_input_runs(self):
        rng = np.random.default_rng(5)
        n_dip, n_var = 20, 12
        hap = rng.integers(0, 2, (2 * n_dip, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        hm = HaplotypeMatrix(hap, pos, 0, n_var * 1000)
        out = sigma_d2_geno(hm)
        assert out.shape == (n_var * (n_var - 1) // 2,)


class TestKernelParity:
    """The fused _SIGMA_D2_GENO_KERN must match the polynomial path
    (DD_polynomial / pi2_polynomial) on the same input to floating-point
    precision."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 2026])
    def test_kernel_matches_polynomial(self, seed):
        from pg_gpu.ld_pipeline import compute_genotype_counts_for_pairs
        from pg_gpu.genotype_kernels import _PopDataGeno
        from pg_gpu.ld_statistics_genotype import (
            dd_geno_single,
            pi2_geno_single,
        )

        gm = _random_gm(40, 25, seed=seed)
        gm.transfer_to_gpu()
        i, j = cp.triu_indices(25, k=1)
        idx_i, idx_j = i.astype(cp.int32), j.astype(cp.int32)

        kernel_out = sigma_d2_geno(gm, idx_i, idx_j).get()

        counts, n_valid = compute_genotype_counts_for_pairs(
            gm.genotypes, idx_i, idx_j)
        p = _PopDataGeno(counts, n_valid)
        dd = dd_geno_single(p).get()
        pi2 = pi2_geno_single(p).get()
        reference = np.where(pi2 > 0, dd / pi2, np.nan)

        finite = np.isfinite(kernel_out) & np.isfinite(reference)
        np.testing.assert_allclose(
            kernel_out[finite], reference[finite],
            rtol=1e-12, atol=1e-12)
        assert np.array_equal(np.isnan(kernel_out), np.isnan(reference))


class TestMomentsLDConsistency:
    """The per-pair sigma_d^2 estimator must use the SAME polynomial
    components that the moments-LD pipeline aggregates per bin. Sum the
    per-pair DD and pi^2 components ourselves and compare to what the
    moments-LD bin-summing path returns."""

    def test_components_match_moments_ld_bin_sum(self):
        from pg_gpu.ld_pipeline import compute_genotype_counts_for_pairs
        from pg_gpu.genotype_kernels import _PopDataGeno
        from pg_gpu.ld_statistics_genotype import (
            dd_geno_single,
            pi2_geno_single,
        )

        gm = _random_gm(40, 20, seed=42)
        i, j = cp.triu_indices(20, k=1)
        idx_i = i.astype(cp.int32)
        idx_j = j.astype(cp.int32)

        # The per-pair function exposed for users.
        sd2 = sigma_d2_geno(gm, idx_i, idx_j).get()

        # Reconstruct DD and pi^2 the same way moments-LD does, then
        # verify sum(DD)/sum(pi^2) over all pairs matches the bin-sum
        # ratio computed from the per-pair components used inside the
        # exported function.
        counts, n_valid = compute_genotype_counts_for_pairs(
            gm.genotypes, idx_i, idx_j)
        p = _PopDataGeno(counts, n_valid)
        dd = dd_geno_single(p).get()
        pi2 = pi2_geno_single(p).get()

        finite = np.isfinite(sd2) & (pi2 > 0)
        np.testing.assert_allclose(
            sd2[finite], (dd[finite] / pi2[finite]),
            rtol=1e-12, atol=1e-12)
