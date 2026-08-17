"""Rogers-Huff r / r² estimator: parity with scikit-allel + edge cases.

Tests ``pg_gpu.ld_statistics.rogers_huff_r`` / ``rogers_huff_r_squared``, which
are dosage-native and require a ``GenotypeMatrix``, plus the ``estimator``
resolver.
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


def _gm(hm: HaplotypeMatrix) -> GenotypeMatrix:
    return GenotypeMatrix.from_haplotype_matrix(hm)


def _allel_r(hm: HaplotypeMatrix) -> np.ndarray:
    """Reference scikit-allel Rogers-Huff r on the same 0/1/2 dosages."""
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
        r_pg = rogers_huff_r(_gm(hm)).get()
        r_allel = _allel_r(hm).astype(np.float64)
        finite = np.isfinite(r_allel) & np.isfinite(r_pg)
        # allel uses float32 internally; pg_gpu uses float64. Tolerance
        # reflects the float32 precision floor.
        np.testing.assert_allclose(
            r_pg[finite], r_allel[finite], rtol=1e-5, atol=1e-5)

    def test_r_squared_matches_squared_r(self):
        gm = _gm(_random_hm(n_diploids=20, n_var=30, seed=3))
        r = rogers_huff_r(gm).get()
        r2 = rogers_huff_r_squared(gm).get()
        np.testing.assert_allclose(r2, r ** 2, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Output shape / ordering
# ---------------------------------------------------------------------------


class TestOutputShape:

    def test_condensed_shape_matches_allel(self):
        r_pg = rogers_huff_r(_gm(_random_hm(n_diploids=10, n_var=25, seed=11)))
        assert r_pg.shape == (25 * 24 // 2,)

    def test_pair_ordering_matches_allel(self):
        """Both libraries lay pairs out as the upper triangle scanned
        row-major: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)."""
        hm = _random_hm(n_diploids=10, n_var=12, seed=13)
        r_pg = rogers_huff_r(_gm(hm)).get()
        r_allel = _allel_r(hm).astype(np.float64)
        finite = np.isfinite(r_pg) & np.isfinite(r_allel)
        assert finite.sum() > 50, "need enough finite pairs to detect ordering"
        np.testing.assert_allclose(
            r_pg[finite], r_allel[finite], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_monomorphic_variants_yield_nan(self):
        """A column of all 0s has zero variance; r involving it is NaN,
        matching allel's behavior."""
        geno = np.zeros((10, 5), dtype=np.int8)
        gm = GenotypeMatrix(geno, np.arange(5) * 100, 0, 500)
        r_pg = rogers_huff_r(gm).get()
        assert np.all(np.isnan(r_pg))

    def test_perfectly_correlated_pair(self):
        """Three identical dosage columns give r = +1 for every pair."""
        col = np.repeat([0, 1, 2], 4).astype(np.int8)[:, None]
        geno = np.hstack([col, col, col])
        gm = GenotypeMatrix(geno, np.array([100, 200, 300]), 0, 400)
        r_pg = rogers_huff_r(gm).get()
        np.testing.assert_allclose(r_pg, [1.0, 1.0, 1.0], atol=1e-12)

    def test_perfectly_anticorrelated_pair(self):
        col_a = np.repeat([0, 1, 2], 4).astype(np.int8)[:, None]
        col_b = 2 - col_a
        geno = np.hstack([col_a, col_b])
        gm = GenotypeMatrix(geno, np.array([100, 200]), 0, 300)
        r_pg = rogers_huff_r(gm).get()
        np.testing.assert_allclose(r_pg, [-1.0], atol=1e-12)

    def test_haplotype_matrix_input_raises(self):
        """rogers_huff is dosage-native: a HaplotypeMatrix is rejected with a
        message pointing at the conversion."""
        hm = _random_hm(n_diploids=5, n_var=5, seed=0)
        with pytest.raises(TypeError, match="GenotypeMatrix"):
            rogers_huff_r(hm)

    def test_missing_in_genotype_matrix_raises(self):
        rng = np.random.default_rng(0)
        geno = rng.integers(0, 3, (10, 5), dtype=np.int8)
        geno[3, 2] = -1
        gm = GenotypeMatrix(geno, np.arange(5) * 100, 0, 500)
        with pytest.raises(ValueError, match="missing values"):
            rogers_huff_r(gm)

    def test_unsupported_input_type_raises(self):
        with pytest.raises(TypeError, match="GenotypeMatrix"):
            rogers_huff_r(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# Estimator resolver
# ---------------------------------------------------------------------------


class TestEstimatorResolver:

    def test_auto_haplotype_resolves_to_sigma_d2(self):
        assert _resolve_ld_estimator('auto', is_hap_matrix=True) == 'sigma_d2'

    def test_auto_genotype_resolves_to_rogers_huff(self):
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
