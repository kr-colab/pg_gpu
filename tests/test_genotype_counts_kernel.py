"""Parity tests for the fused 9-way diploid genotype-counts kernel.

``_GENO_COUNTS_KERN`` in ``pg_gpu.ld_pipeline`` replaced the previous
CuPy fancy-index + 9-way mask-and-sum path. These tests verify the
new kernel produces identical output (counts and n_valid) to the old
elementwise path on the same inputs, including with missing data and
population subsetting.
"""

import cupy as cp
import numpy as np
import pytest

from pg_gpu.ld_pipeline import compute_genotype_counts_for_pairs


def _polynomial_counts(genotypes, idx_i, idx_j, pop_indices=None):
    """The pre-kernel implementation, kept here as the parity reference."""
    if pop_indices is not None:
        if isinstance(pop_indices, list):
            pop_indices = cp.array(pop_indices, dtype=cp.int32)
        genotypes = genotypes[pop_indices, :]

    geno_i = genotypes[:, idx_i]
    geno_j = genotypes[:, idx_j]

    has_missing = cp.any(geno_i < 0) or cp.any(geno_j < 0)

    if has_missing:
        valid_mask = (geno_i >= 0) & (geno_j >= 0)
        n_valid = cp.sum(valid_mask, axis=0, dtype=cp.int32)
        gi = cp.where(valid_mask, geno_i, 0)
        gj = cp.where(valid_mask, geno_j, 0)
    else:
        n_valid = None
        valid_mask = None
        gi = geno_i
        gj = geno_j

    combo = gi * 3 + gj
    cols = []
    for k in range(9):
        mask = combo == k
        if valid_mask is not None:
            mask = mask & valid_mask
        cols.append(cp.sum(mask, axis=0, dtype=cp.int32))

    return cp.stack(cols, axis=1), n_valid


def _random_genotypes(n_indiv, n_var, seed=0, missing_rate=0.0):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.95, size=n_var)
    g = (rng.binomial(1, af[None], (n_indiv, n_var))
         + rng.binomial(1, af[None], (n_indiv, n_var))).astype(np.int8)
    if missing_rate > 0:
        miss = rng.random((n_indiv, n_var)) < missing_rate
        g[miss] = -1
    return cp.asarray(g)


def _check_match(kernel_counts, kernel_nv, ref_counts, ref_nv):
    np.testing.assert_array_equal(kernel_counts.get(), ref_counts.get())
    if ref_nv is None:
        assert kernel_nv is None
    else:
        np.testing.assert_array_equal(kernel_nv.get(), ref_nv.get())


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
class TestRandomPanels:

    def test_no_missing(self, seed):
        g = _random_genotypes(50, 40, seed=seed)
        i, j = cp.triu_indices(40, k=1)
        ki, knv = compute_genotype_counts_for_pairs(
            g, i.astype(cp.int32), j.astype(cp.int32))
        ri, rnv = _polynomial_counts(
            g, i.astype(cp.int32), j.astype(cp.int32))
        _check_match(ki, knv, ri, rnv)

    def test_with_missing(self, seed):
        g = _random_genotypes(50, 40, seed=seed, missing_rate=0.05)
        i, j = cp.triu_indices(40, k=1)
        ki, knv = compute_genotype_counts_for_pairs(
            g, i.astype(cp.int32), j.astype(cp.int32))
        ri, rnv = _polynomial_counts(
            g, i.astype(cp.int32), j.astype(cp.int32))
        _check_match(ki, knv, ri, rnv)

    def test_with_pop_indices(self, seed):
        g = _random_genotypes(80, 30, seed=seed)
        i, j = cp.triu_indices(30, k=1)
        # Subsample half the individuals.
        pop = list(range(0, 80, 2))
        ki, knv = compute_genotype_counts_for_pairs(
            g, i.astype(cp.int32), j.astype(cp.int32), pop_indices=pop)
        ri, rnv = _polynomial_counts(
            g, i.astype(cp.int32), j.astype(cp.int32), pop_indices=pop)
        _check_match(ki, knv, ri, rnv)


class TestEdgeCases:

    def test_single_pair(self):
        g = _random_genotypes(20, 10, seed=99)
        i = cp.array([2], dtype=cp.int32)
        j = cp.array([7], dtype=cp.int32)
        ki, knv = compute_genotype_counts_for_pairs(g, i, j)
        ri, rnv = _polynomial_counts(g, i, j)
        _check_match(ki, knv, ri, rnv)

    def test_large_panel(self):
        # Stress test with realistic dimensions.
        g = _random_genotypes(100, 200, seed=2026)
        i, j = cp.triu_indices(200, k=1)
        ki, knv = compute_genotype_counts_for_pairs(
            g, i.astype(cp.int32), j.astype(cp.int32))
        ri, rnv = _polynomial_counts(
            g, i.astype(cp.int32), j.astype(cp.int32))
        _check_match(ki, knv, ri, rnv)

    def test_n_valid_returned_only_when_missing(self):
        g_no_missing = _random_genotypes(20, 15, seed=3, missing_rate=0)
        g_missing = _random_genotypes(20, 15, seed=4, missing_rate=0.1)
        i, j = cp.triu_indices(15, k=1)

        _, nv_clean = compute_genotype_counts_for_pairs(
            g_no_missing, i.astype(cp.int32), j.astype(cp.int32))
        assert nv_clean is None

        _, nv_dirty = compute_genotype_counts_for_pairs(
            g_missing, i.astype(cp.int32), j.astype(cp.int32))
        assert nv_dirty is not None
        assert nv_dirty.shape == (i.shape[0],)
