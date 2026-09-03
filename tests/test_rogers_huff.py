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
    """_resolve_ld_estimator maps 'auto' per input kind and raises for
    estimators a kind cannot compute."""

    def test_auto_policy_per_kind(self):
        assert _resolve_ld_estimator('auto', 'haplotype') == 'sigma_d2'
        assert _resolve_ld_estimator('auto', 'genotype') == 'rogers_huff'
        assert _resolve_ld_estimator('auto', 'array') == 'r2'

    def test_computable_names_pass_through(self):
        assert _resolve_ld_estimator('rogers_huff', 'haplotype') == 'rogers_huff'
        assert _resolve_ld_estimator('rogers_huff', 'genotype') == 'rogers_huff'
        assert _resolve_ld_estimator('sigma_d2', 'haplotype') == 'sigma_d2'
        for kind in ('haplotype', 'genotype', 'array'):
            assert _resolve_ld_estimator('r2', kind) == 'r2'

    def test_incomputable_combinations_raise(self):
        with pytest.raises(ValueError, match="requires a HaplotypeMatrix"):
            _resolve_ld_estimator('sigma_d2', 'genotype')
        with pytest.raises(ValueError, match="pre-computed"):
            _resolve_ld_estimator('sigma_d2', 'array')
        with pytest.raises(ValueError, match="pre-computed"):
            _resolve_ld_estimator('rogers_huff', 'array')

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            _resolve_ld_estimator('bogus', 'haplotype')


# ---------------------------------------------------------------------------
# HaplotypeMatrix methods that route through the dosage estimator
# ---------------------------------------------------------------------------


class TestHaplotypeMatrixEstimator:

    def test_auto_estimator_accepted_and_equals_r2(self):
        # 'auto' is the default estimator name elsewhere; the matrix methods
        # must accept it (meaning naive r2 here), not reject it.
        hm = _random_hm(n_diploids=25, n_var=40, seed=3)
        np.testing.assert_array_equal(
            np.nan_to_num(hm.pairwise_r2(estimator='auto').get()),
            np.nan_to_num(hm.pairwise_r2(estimator='r2').get()))
        va, ca = hm.windowed_r_squared([0, 20_000, 40_000], estimator='auto')
        vr, cr = hm.windowed_r_squared([0, 20_000, 40_000], estimator='r2')
        np.testing.assert_array_equal(np.nan_to_num(va), np.nan_to_num(vr))
        np.testing.assert_array_equal(ca, cr)

    @pytest.mark.parametrize("bad", ["sigma_d2", "bogus"])
    def test_unavailable_estimator_raises_unknown(self, bad):
        # sigma_d2 is a real estimator but not available on these methods;
        # both it and a nonsense name raise the shared "Unknown estimator".
        hm = _random_hm(n_diploids=25, n_var=40, seed=3)
        with pytest.raises(ValueError, match="Unknown estimator"):
            hm.pairwise_r2(estimator=bad)
        with pytest.raises(ValueError, match="Unknown estimator"):
            hm.windowed_r_squared([0, 40_000], estimator=bad)

    def test_pairwise_r2_matches_module_function(self):
        hm = _random_hm(n_diploids=30, n_var=40, seed=11)
        r2_hm = hm.pairwise_r2(estimator='rogers_huff').get()
        import scipy.spatial.distance as ssd
        r2_gm = ssd.squareform(rogers_huff_r_squared(_gm(hm)).get())
        finite = np.isfinite(r2_gm)
        assert np.isfinite(r2_hm).sum() == finite.sum()
        np.testing.assert_allclose(r2_hm[finite], r2_gm[finite], rtol=1e-12)

    def test_pairwise_r2_matches_allel(self):
        hm = _random_hm(n_diploids=40, n_var=50, seed=5)
        r2_hm = hm.pairwise_r2(estimator='rogers_huff').get()
        import scipy.spatial.distance as ssd
        r_allel = ssd.squareform(_allel_r(hm).astype(np.float64))
        r2_allel = r_allel ** 2
        finite = np.isfinite(r2_allel) & np.isfinite(r2_hm)
        np.fill_diagonal(finite, False)
        np.testing.assert_allclose(
            r2_hm[finite], r2_allel[finite], rtol=1e-5, atol=1e-5)

    def test_pairwise_r2_multiallelic_site_is_nan(self):
        hm = _random_hm(n_diploids=20, n_var=12, seed=7)
        hap = np.array(cp.asnumpy(hm.haplotypes))
        carriers = np.where(hap[:, 4] == 1)[0]
        hap[carriers[: carriers.size // 2], 4] = 2
        hm3 = HaplotypeMatrix(hap, cp.asnumpy(hm.positions), 0, 12 * 1000)
        with pytest.warns(UserWarning, match="biallelic"):
            r2 = hm3.pairwise_r2(estimator='rogers_huff').get()
        assert r2.shape == (12, 12)
        off = np.ones(12, dtype=bool)
        off[4] = False
        assert np.isnan(r2[4, off]).all() and np.isnan(r2[off, 4]).all()
        assert r2[4, 4] == 0
        kept = np.ix_(off, off)
        assert np.isfinite(r2[kept]).any()

    def test_pairwise_r2_monomorphic_site_is_nan(self):
        hm = _random_hm(n_diploids=20, n_var=8, seed=9)
        hap = np.array(cp.asnumpy(hm.haplotypes))
        hap[:, 3] = 0
        hm0 = HaplotypeMatrix(hap, cp.asnumpy(hm.positions), 0, 8 * 1000)
        r2 = hm0.pairwise_r2(estimator='rogers_huff').get()
        off = np.ones(8, dtype=bool)
        off[3] = False
        assert np.isnan(r2[3, off]).all()
        assert r2[3, 3] == 0

    def test_windowed_r_squared_runs_and_matches_manual(self):
        hm = _random_hm(n_diploids=30, n_var=40, seed=13)
        bins = [0, 10000, 20000, 40000]
        res, counts = hm.windowed_r_squared(bins, percentile=50,
                                            estimator='rogers_huff')
        r = _allel_r(hm).astype(np.float64)
        import scipy.spatial.distance as ssd
        r2_full = ssd.squareform(r) ** 2
        pos = cp.asnumpy(hm.positions)
        ii, jj = np.triu_indices(40, k=1)
        d = pos[jj] - pos[ii]
        vals = r2_full[ii, jj]
        for b in range(3):
            sel = (d >= bins[b]) & (d < bins[b + 1]) & np.isfinite(vals)
            assert counts[b] == sel.sum()
            if sel.any():
                np.testing.assert_allclose(res[b], np.percentile(vals[sel], 50),
                                           rtol=1e-5, atol=1e-5)

    def test_windowed_r_squared_drops_multiallelic_pairs(self):
        hm = _random_hm(n_diploids=20, n_var=10, seed=15)
        hap = np.array(cp.asnumpy(hm.haplotypes))
        carriers = np.where(hap[:, 2] == 1)[0]
        hap[carriers[: carriers.size // 2], 2] = 2
        hm3 = HaplotypeMatrix(hap, cp.asnumpy(hm.positions), 0, 10 * 1000)
        bins = [0, 10000]
        with pytest.warns(UserWarning, match="biallelic"):
            _, counts3 = hm3.windowed_r_squared(bins,
                                                estimator='rogers_huff')
        _, counts2 = hm.windowed_r_squared(bins, estimator='rogers_huff')
        assert counts3[0] < counts2[0]

    def test_windowed_r_squared_pop_raises(self):
        hm = _random_hm(n_diploids=10, n_var=10, seed=17)
        hm.sample_sets = {"p": list(range(10))}
        with pytest.raises(NotImplementedError):
            hm.windowed_r_squared([0, 10000], pop="p",
                                  estimator='rogers_huff')

    def test_missing_values_raise(self):
        hm = _random_hm(n_diploids=10, n_var=10, seed=19)
        hap = np.array(cp.asnumpy(hm.haplotypes))
        hap[0, 0] = -1
        hmm = HaplotypeMatrix(hap, cp.asnumpy(hm.positions), 0, 10 * 1000)
        with pytest.raises(ValueError, match="missing"):
            hmm.pairwise_r2(estimator='rogers_huff')

    def test_odd_haplotype_count_raises(self):
        rng = np.random.default_rng(21)
        hap = rng.integers(0, 2, (9, 10), dtype=np.int8)
        pos = np.arange(10, dtype=np.int64) * 1000
        hm = HaplotypeMatrix(hap, pos, 0, 10 * 1000)
        with pytest.raises(ValueError, match="even number"):
            hm.pairwise_r2(estimator='rogers_huff')
