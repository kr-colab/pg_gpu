"""
Validation tests comparing pg_gpu selection statistics against scikit-allel.

These tests ensure numerical agreement between the GPU implementation
and the reference scikit-allel implementation.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix, selection

from .conftest import founder_haplotypes


@pytest.fixture
def simulated_data():
    """Generate deterministic haplotype data for validation."""
    np.random.seed(42)
    n_haplotypes = 40
    n_variants = 200
    hap_allel = np.random.randint(0, 2, size=(n_variants, n_haplotypes)).astype('i1')
    pos = np.arange(1, n_variants + 1) * 100
    return hap_allel, pos


@pytest.fixture
def two_pop_data():
    """Generate two-population data for cross-population tests."""
    np.random.seed(123)
    n_variants = 200
    n_hap1 = 20
    n_hap2 = 20
    h1 = np.random.randint(0, 2, size=(n_variants, n_hap1)).astype('i1')
    h2 = np.random.randint(0, 2, size=(n_variants, n_hap2)).astype('i1')
    pos = np.arange(1, n_variants + 1) * 100
    return h1, h2, pos


class TestNSLComparison:
    """Validate nSL against scikit-allel."""

    def test_nsl_correspondence(self, simulated_data):
        hap_allel, pos = simulated_data

        # scikit-allel
        score_allel = allel.nsl(hap_allel, use_threads=False)

        # pg_gpu (needs transposed format)
        hap_pg = hap_allel.T.copy()
        matrix = HaplotypeMatrix(hap_pg, pos, 0, pos[-1] + 1)
        score_pg = selection.nsl(matrix)

        # compare non-NaN values
        both_valid = ~np.isnan(score_allel) & ~np.isnan(score_pg)
        assert np.sum(both_valid) > 0, "No valid scores to compare"
        np.testing.assert_allclose(
            score_pg[both_valid], score_allel[both_valid],
            rtol=1e-5,
            err_msg="nSL scores do not match allel"
        )

        # NaN positions should mostly agree
        nan_allel = np.isnan(score_allel)
        nan_pg = np.isnan(score_pg)
        agreement = np.mean(nan_allel == nan_pg)
        assert agreement > 0.95, \
            f"NaN positions agree only {agreement:.1%}"


class TestXPNSLComparison:
    """Validate XP-nSL against scikit-allel."""

    def test_xpnsl_correspondence(self, two_pop_data):
        h1_allel, h2_allel, pos = two_pop_data

        # scikit-allel
        score_allel = allel.xpnsl(h1_allel, h2_allel, use_threads=False)

        # pg_gpu
        n1 = h1_allel.shape[1]
        n2 = h2_allel.shape[1]
        combined_allel = np.hstack([h1_allel, h2_allel])
        hap_pg = combined_allel.T.copy()
        matrix = HaplotypeMatrix(
            hap_pg, pos, 0, pos[-1] + 1,
            sample_sets={'pop1': list(range(n1)),
                         'pop2': list(range(n1, n1 + n2))}
        )
        score_pg = selection.xpnsl(matrix, 'pop1', 'pop2')

        both_valid = ~np.isnan(score_allel) & ~np.isnan(score_pg)
        assert np.sum(both_valid) > 0
        np.testing.assert_allclose(
            score_pg[both_valid], score_allel[both_valid],
            rtol=1e-5,
            err_msg="XP-nSL scores do not match allel"
        )


class TestIHSComparison:
    """Validate iHS against scikit-allel."""

    def test_ihs_correspondence(self, simulated_data):
        hap_allel, pos = simulated_data

        for include_edges in [True, False]:
            # scikit-allel
            score_allel = allel.ihs(
                hap_allel, pos,
                include_edges=include_edges,
                use_threads=False
            )

            # pg_gpu
            hap_pg = hap_allel.T.copy()
            matrix = HaplotypeMatrix(hap_pg, pos, 0, pos[-1] + 1)
            score_pg = selection.ihs(matrix, include_edges=include_edges)

            both_valid = ~np.isnan(score_allel) & ~np.isnan(score_pg)
            if np.sum(both_valid) > 0:
                np.testing.assert_allclose(
                    score_pg[both_valid], score_allel[both_valid],
                    rtol=1e-5,
                    err_msg=f"iHS scores differ (include_edges={include_edges})"
                )

    def test_ihs_min_maf(self, simulated_data):
        hap_allel, pos = simulated_data

        for min_maf in [0.0, 0.05, 0.1]:
            score_allel = allel.ihs(
                hap_allel, pos, min_maf=min_maf,
                include_edges=True, use_threads=False
            )
            hap_pg = hap_allel.T.copy()
            matrix = HaplotypeMatrix(hap_pg, pos, 0, pos[-1] + 1)
            score_pg = selection.ihs(
                matrix, min_maf=min_maf, include_edges=True
            )

            both_valid = ~np.isnan(score_allel) & ~np.isnan(score_pg)
            if np.sum(both_valid) > 0:
                np.testing.assert_allclose(
                    score_pg[both_valid], score_allel[both_valid],
                    rtol=1e-5,
                    err_msg=f"iHS differs at min_maf={min_maf}"
                )


class TestXPEHHComparison:
    """Validate XP-EHH against scikit-allel."""

    def test_xpehh_correspondence(self, two_pop_data):
        h1_allel, h2_allel, pos = two_pop_data

        for include_edges in [True, False]:
            # scikit-allel
            score_allel = allel.xpehh(
                h1_allel, h2_allel, pos,
                include_edges=include_edges,
                use_threads=False
            )

            # pg_gpu
            n1 = h1_allel.shape[1]
            n2 = h2_allel.shape[1]
            combined = np.hstack([h1_allel, h2_allel])
            hap_pg = combined.T.copy()
            matrix = HaplotypeMatrix(
                hap_pg, pos, 0, pos[-1] + 1,
                sample_sets={'pop1': list(range(n1)),
                             'pop2': list(range(n1, n1 + n2))}
            )
            score_pg = selection.xpehh(
                matrix, 'pop1', 'pop2',
                include_edges=include_edges
            )

            both_valid = ~np.isnan(score_allel) & ~np.isnan(score_pg)
            if np.sum(both_valid) > 0:
                np.testing.assert_allclose(
                    score_pg[both_valid], score_allel[both_valid],
                    rtol=1e-5,
                    err_msg=f"XP-EHH differs (include_edges={include_edges})"
                )


class TestEHHDecayComparison:
    """Validate EHH decay against scikit-allel."""

    def test_ehh_decay_correspondence(self):
        h_allel = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0]], dtype='i1')

        ehh_allel = allel.ehh_decay(h_allel)

        hap_pg = h_allel.T.copy()
        pos = np.arange(h_allel.shape[0]) * 1000
        matrix = HaplotypeMatrix(hap_pg, pos, 0, h_allel.shape[0] * 1000)
        ehh_pg = selection.ehh_decay(matrix)

        np.testing.assert_array_almost_equal(ehh_pg, ehh_allel)

    def test_ehh_decay_random(self):
        np.random.seed(99)
        n_var = 50
        n_hap = 10
        h_allel = np.random.randint(0, 2, size=(n_var, n_hap)).astype('i1')

        ehh_allel = allel.ehh_decay(h_allel)

        hap_pg = h_allel.T.copy()
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap_pg, pos, 0, n_var * 1000)
        ehh_pg = selection.ehh_decay(matrix)

        np.testing.assert_array_almost_equal(ehh_pg, ehh_allel)


class TestGarudHComparison:
    """Validate Garud's H against scikit-allel."""

    def test_garud_h_correspondence(self):
        np.random.seed(42)
        for _ in range(5):
            n_var = np.random.randint(5, 30)
            n_hap = np.random.randint(4, 20)
            h_allel = np.random.randint(0, 2, size=(n_var, n_hap)).astype('i1')

            h1_a, h12_a, h123_a, h2h1_a = allel.garud_h(h_allel)

            hap_pg = h_allel.T.copy()
            pos = np.arange(n_var) * 1000
            matrix = HaplotypeMatrix(hap_pg, pos, 0, n_var * 1000)
            h1_p, h12_p, h123_p, h2h1_p = selection.garud_h(matrix)

            assert np.isclose(h1_p, h1_a, rtol=1e-10), \
                f"H1: pg={h1_p}, allel={h1_a}"
            assert np.isclose(h12_p, h12_a, rtol=1e-10), \
                f"H12: pg={h12_p}, allel={h12_a}"
            assert np.isclose(h123_p, h123_a, rtol=1e-10), \
                f"H123: pg={h123_p}, allel={h123_a}"
            assert np.isclose(h2h1_p, h2h1_a, rtol=1e-10), \
                f"H2/H1: pg={h2h1_p}, allel={h2h1_a}"

    @pytest.mark.parametrize("n_hap,step", [(40, None), (40, 7), (2200, 7)])
    def test_moving_garud_h_correspondence(self, n_hap, step):
        n_var = 200
        hap = founder_haplotypes(np.random.default_rng(5), n_hap, n_var, 25)
        expected = allel.moving_garud_h(hap.T, 20, step=step)

        matrix = HaplotypeMatrix(hap, np.arange(n_var) * 1000, 0, n_var * 1000)
        got = selection.moving_garud_h(matrix, size=20, step=step)
        for name, e, g in zip(("H1", "H12", "H123", "H2/H1"), expected, got):
            np.testing.assert_allclose(g, e, rtol=1e-10, err_msg=name)


class TestStandardizeComparison:
    """Validate standardization against scikit-allel."""

    def test_standardize_correspondence(self):
        np.random.seed(42)
        score = np.random.randn(100)
        score[::10] = np.nan

        result_allel = allel.standardize(score)
        result_pg = selection.standardize(score)

        both_valid = ~np.isnan(result_allel) & ~np.isnan(result_pg)
        np.testing.assert_allclose(
            result_pg[both_valid], result_allel[both_valid],
            rtol=1e-10
        )

    def test_standardize_by_allele_count_correspondence(self):
        np.random.seed(42)
        score = np.random.randn(200)
        score[::15] = np.nan
        aac = np.random.randint(1, 30, size=200)

        bins = np.array([1, 5, 10, 15, 20, 30])

        result_allel, _ = allel.standardize_by_allele_count(
            score, aac, bins=bins, diagnostics=False
        )
        result_pg, _ = selection.standardize_by_allele_count(
            score, aac, bins=bins
        )

        both_valid = ~np.isnan(result_allel) & ~np.isnan(result_pg)
        np.testing.assert_allclose(
            result_pg[both_valid], result_allel[both_valid],
            rtol=1e-10
        )
