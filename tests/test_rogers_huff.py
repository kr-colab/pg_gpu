"""Rogers-Huff r / r² estimator: parity with scikit-allel + edge cases.

Tests the new ``pg_gpu.ld_statistics.rogers_huff_r`` and
``rogers_huff_r_squared`` functions, plus their integration with the
``estimator`` kwarg on ``HaplotypeMatrix`` / ``GenotypeMatrix``
methods.
"""

import allel
import cupy as cp
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.ld_statistics import (
    _resolve_ld_estimator,
    rogers_huff_r,
    rogers_huff_r_squared,
)


def _random_hm(n_diploids: int, n_var: int, seed: int = 42) -> HaplotypeMatrix:
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, (2 * n_diploids, n_var), dtype=np.int8)
    pos = np.arange(n_var, dtype=np.int64) * 1000
    return HaplotypeMatrix(hap, pos, 0, n_var * 1000)


def _allel_r(hm: HaplotypeMatrix) -> np.ndarray:
    """Reference scikit-allel Rogers-Huff r on the same dosages."""
    hap = hm.haplotypes
    if hasattr(hap, "get"):
        hap = hap.get()
    gn = (hap[0::2] + hap[1::2]).T.astype(np.int8)
    return allel.rogers_huff_r(gn)


# ---------------------------------------------------------------------------
# Parity with scikit-allel
# ---------------------------------------------------------------------------


class TestParityAgainstAllel:

    @pytest.mark.parametrize("seed", [0, 1, 42, 2026])
    def test_random_panels(self, seed):
        hm = _random_hm(n_diploids=50, n_var=80, seed=seed)
        r_pg = rogers_huff_r(hm).get()
        r_allel = _allel_r(hm).astype(np.float64)
        finite = np.isfinite(r_allel) & np.isfinite(r_pg)
        # allel uses float32 internally; pg_gpu uses float64. Tolerance
        # reflects the float32 precision floor.
        np.testing.assert_allclose(
            r_pg[finite], r_allel[finite], rtol=1e-5, atol=1e-5)

    def test_genotype_matrix_path(self):
        """rogers_huff_r on a GenotypeMatrix matches allel directly."""
        hm = _random_hm(n_diploids=40, n_var=60, seed=7)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        r_pg = rogers_huff_r(gm).get()
        r_allel = _allel_r(hm).astype(np.float64)
        finite = np.isfinite(r_allel) & np.isfinite(r_pg)
        np.testing.assert_allclose(
            r_pg[finite], r_allel[finite], rtol=1e-5, atol=1e-5)

    def test_haplotype_and_genotype_paths_agree(self):
        """rogers_huff_r is invariant to whether dosages are derived from
        a HaplotypeMatrix (auto-pair) or a GenotypeMatrix (direct)."""
        hm = _random_hm(n_diploids=30, n_var=40, seed=99)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        r_hm = rogers_huff_r(hm).get()
        r_gm = rogers_huff_r(gm).get()
        np.testing.assert_array_equal(r_hm, r_gm)

    def test_r_squared_matches_squared_r(self):
        hm = _random_hm(n_diploids=20, n_var=30, seed=3)
        r = rogers_huff_r(hm).get()
        r2 = rogers_huff_r_squared(hm).get()
        # Both paths compute the same float64 r and then square; tiny
        # rounding from GPU op order is fine.
        np.testing.assert_allclose(r2, r ** 2, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Output shape / ordering
# ---------------------------------------------------------------------------


class TestOutputShape:

    def test_condensed_shape_matches_allel(self):
        hm = _random_hm(n_diploids=10, n_var=25, seed=11)
        r_pg = rogers_huff_r(hm)
        assert r_pg.shape == (25 * 24 // 2,)

    def test_pair_ordering_matches_allel(self):
        """Both libraries lay pairs out as the upper triangle scanned
        row-major: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)."""
        hm = _random_hm(n_diploids=10, n_var=12, seed=13)
        r_pg = rogers_huff_r(hm).get()
        r_allel = _allel_r(hm).astype(np.float64)
        # If ordering matched, the per-position diff should be small everywhere.
        # If it didn't, the diff would be large at most positions.
        finite = np.isfinite(r_pg) & np.isfinite(r_allel)
        assert finite.sum() > 50, "need enough finite pairs to detect ordering"
        np.testing.assert_allclose(
            r_pg[finite], r_allel[finite], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_monomorphic_variants_yield_nan(self):
        """A column of all 0s (or all 2s) has zero variance; r involving it
        is NaN, matching allel's behavior."""
        n_hap, n_var = 20, 5
        hap = np.zeros((n_hap, n_var), dtype=np.int8)  # all monomorphic
        pos = np.arange(n_var) * 100
        hm = HaplotypeMatrix(hap, pos, 0, n_var * 100)
        r_pg = rogers_huff_r(hm).get()
        assert np.all(np.isnan(r_pg))

    def test_perfectly_correlated_pair(self):
        """Two identical SNP columns give |r| = 1 and r² = 1."""
        n_hap = 20
        col = np.repeat([0, 1], n_hap // 2).astype(np.int8)[:, None]
        hap = np.hstack([col, col, col])  # 3 identical variants
        pos = np.array([100, 200, 300])
        hm = HaplotypeMatrix(hap, pos, 0, 400)
        r_pg = rogers_huff_r(hm).get()
        # Three pairs, all identical -> r = +1
        np.testing.assert_allclose(r_pg, [1.0, 1.0, 1.0], atol=1e-12)

    def test_perfectly_anticorrelated_pair(self):
        n_hap = 20
        col_a = np.repeat([0, 1], n_hap // 2).astype(np.int8)[:, None]
        col_b = 1 - col_a
        hap = np.hstack([col_a, col_b])
        hm = HaplotypeMatrix(hap, np.array([100, 200]), 0, 300)
        r_pg = rogers_huff_r(hm).get()
        np.testing.assert_allclose(r_pg, [-1.0], atol=1e-12)

    def test_missing_in_haplotype_matrix_raises(self):
        n_hap, n_var = 10, 5
        hap = np.zeros((n_hap, n_var), dtype=np.int8)
        hap[3, 2] = -1
        pos = np.arange(n_var) * 100
        hm = HaplotypeMatrix(hap, pos, 0, n_var * 100)
        with pytest.raises(ValueError, match="missing values"):
            rogers_huff_r(hm)

    def test_missing_in_genotype_matrix_raises(self):
        rng = np.random.default_rng(0)
        geno = rng.integers(0, 3, (10, 5), dtype=np.int8)
        geno[3, 2] = -1
        pos = np.arange(5) * 100
        gm = GenotypeMatrix(geno, pos, 0, 500)
        with pytest.raises(ValueError, match="missing values"):
            rogers_huff_r(gm)

    def test_odd_haplotype_count_raises(self):
        hap = np.zeros((11, 5), dtype=np.int8)  # odd -> can't pair
        pos = np.arange(5) * 100
        hm = HaplotypeMatrix(hap, pos, 0, 500)
        with pytest.raises(ValueError, match="odd number"):
            rogers_huff_r(hm)

    def test_unsupported_input_type_raises(self):
        with pytest.raises(TypeError):
            rogers_huff_r(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# Estimator resolver
# ---------------------------------------------------------------------------


class TestEstimatorResolver:

    def test_auto_haplotype_resolves_to_sigma_d2(self):
        assert _resolve_ld_estimator('auto', is_hap_matrix=True) == 'sigma_d2'

    def test_auto_genotype_resolves_to_rogers_huff(self):
        # New default for non-HaplotypeMatrix inputs: rogers_huff (was r2).
        assert _resolve_ld_estimator(
            'auto', is_hap_matrix=False) == 'rogers_huff'

    def test_explicit_rogers_huff_passes_through(self):
        assert _resolve_ld_estimator(
            'rogers_huff', is_hap_matrix=True) == 'rogers_huff'
        assert _resolve_ld_estimator(
            'rogers_huff', is_hap_matrix=False) == 'rogers_huff'

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            _resolve_ld_estimator('bogus', is_hap_matrix=True)
