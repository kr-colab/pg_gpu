"""
Tests for heterozygosity and inbreeding coefficient functions.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity


class TestHeterozygosityExpected:
    """Test expected heterozygosity (gene diversity)."""

    def test_he_monomorphic(self):
        """Monomorphic sites: He = 0."""
        hap = np.zeros((10, 5), dtype=np.int8)
        pos = np.arange(5) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 5000)
        he = diversity.heterozygosity_expected(matrix)
        np.testing.assert_array_almost_equal(he, 0.0)

    def test_he_balanced(self):
        """p = 0.5: He = 2*0.5*0.5 = 0.5."""
        hap = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                        dtype=np.int8)
        pos = np.array([0, 1000])
        matrix = HaplotypeMatrix(hap, pos, 0, 2000)
        he = diversity.heterozygosity_expected(matrix)
        np.testing.assert_allclose(he, [0.5, 0.5])

    def test_he_vs_allel(self):
        """Compare He against allel's heterozygosity_expected."""
        np.random.seed(42)
        n_ind = 20
        n_var = 50
        hap = np.random.randint(0, 2, (n_ind * 2, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)

        he_pg = diversity.heterozygosity_expected(matrix)

        # allel: needs allele frequencies
        n_hap = n_ind * 2
        dac = np.sum(hap, axis=0)
        af = np.column_stack([1 - dac / n_hap, dac / n_hap])
        he_allel = allel.heterozygosity_expected(af, ploidy=2)

        np.testing.assert_allclose(he_pg, he_allel, rtol=1e-10)


class TestHeterozygosityObserved:
    """Test observed heterozygosity."""

    def test_ho_all_homozygous(self):
        """All individuals homozygous: Ho = 0."""
        # 4 diploid individuals, all hom ref
        hap = np.zeros((8, 3), dtype=np.int8)
        pos = np.arange(3) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 3000)
        ho = diversity.heterozygosity_observed(matrix)
        np.testing.assert_array_almost_equal(ho, 0.0)

    def test_ho_all_heterozygous(self):
        """All individuals heterozygous: Ho = 1."""
        # 4 diploid individuals, all het (0|1)
        hap = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=np.int8)
        pos = np.arange(3) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, 3000)
        ho = diversity.heterozygosity_observed(matrix)
        np.testing.assert_array_almost_equal(ho, 1.0)

    def test_ho_mixed(self):
        """Mix of het and hom individuals."""
        # 4 diploid individuals at 1 variant:
        # ind0: 0|0 (hom), ind1: 0|1 (het), ind2: 1|1 (hom), ind3: 0|1 (het)
        hap = np.array([[0], [0], [0], [1], [1], [1], [0], [1]], dtype=np.int8)
        pos = np.array([0])
        matrix = HaplotypeMatrix(hap, pos, 0, 1)
        ho = diversity.heterozygosity_observed(matrix)
        assert np.isclose(ho[0], 0.5)  # 2/4 individuals are het

    def test_ho_vs_allel(self):
        """Compare Ho against allel's heterozygosity_observed."""
        np.random.seed(42)
        n_ind = 20
        n_var = 50
        hap = np.random.randint(0, 2, (n_ind * 2, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)

        ho_pg = diversity.heterozygosity_observed(matrix)

        # allel: construct GenotypeArray
        # shape: (n_variants, n_individuals, 2)
        g = np.zeros((n_var, n_ind, 2), dtype=np.int8)
        for i in range(n_ind):
            g[:, i, 0] = hap[2 * i]
            g[:, i, 1] = hap[2 * i + 1]
        ga = allel.GenotypeArray(g)
        ho_allel = allel.heterozygosity_observed(ga)

        np.testing.assert_allclose(ho_pg, ho_allel, rtol=1e-10)

    @pytest.mark.parametrize("ploidy", [3, 4])
    def test_polyploid_matches_allel(self, ploidy):
        """Polyploid Ho (an individual is het when its ploidy haplotypes are
        not all identical) matches scikit-allel for ploidy 3 and 4."""
        rng = np.random.default_rng(7)
        n_ind, n_var = 6, 30
        hap = rng.integers(0, 3, (n_ind * ploidy, n_var)).astype(np.int8)
        hm = HaplotypeMatrix(hap, np.arange(n_var) * 1000, 0, n_var * 1000)
        ho_pg = diversity.heterozygosity_observed(hm, ploidy=ploidy)

        g = np.empty((n_var, n_ind, ploidy), dtype=np.int8)
        for i in range(n_ind):
            for k in range(ploidy):
                g[:, i, k] = hap[i * ploidy + k]
        ho_allel = allel.heterozygosity_observed(allel.GenotypeArray(g))
        np.testing.assert_allclose(ho_pg, ho_allel, rtol=1e-10)

    def test_polyploid_skips_missing_individuals(self):
        """Under the default 'include', an individual with any missing
        haplotype is skipped; a site with no fully-called individual is NaN."""
        # ploidy 3, 2 individuals, 2 sites.
        hap = np.array([[0, 0],    # ind0 hap0
                        [-1, -1],  # ind0 hap1 (missing at both sites)
                        [1, 0],    # ind0 hap2
                        [0, 1],    # ind1 hap0
                        [0, -1],   # ind1 hap1
                        [1, 1]],   # ind1 hap2
                       dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.array([0, 1000]), 0, 2000)
        ho = diversity.heterozygosity_observed(hm, ploidy=3)
        # site 0: ind0 missing (skipped); ind1 = [0, 0, 1] is het -> 1/1
        np.testing.assert_allclose(ho[0], 1.0)
        # site 1: both individuals missing -> no valid individual -> NaN
        assert np.isnan(ho[1])

    def test_ploidy_not_dividing_n_haplotypes_raises(self):
        hap = np.zeros((6, 2), dtype=np.int8)  # 6 not divisible by 4
        hm = HaplotypeMatrix(hap, np.array([0, 1000]), 0, 2000)
        with pytest.raises(ValueError, match="not divisible by ploidy"):
            diversity.heterozygosity_observed(hm, ploidy=4)


class TestInbreedingCoefficient:
    """Test Wright's inbreeding coefficient."""

    def test_f_all_het(self):
        """All het at p=0.5: F = 1 - 1.0/0.5 = -1.0."""
        hap = np.array([[0, 0], [1, 1], [0, 0], [1, 1],
                         [0, 0], [1, 1], [0, 0], [1, 1]], dtype=np.int8)
        pos = np.array([0, 1000])
        matrix = HaplotypeMatrix(hap, pos, 0, 2000)
        f = diversity.inbreeding_coefficient(matrix)
        np.testing.assert_allclose(f, -1.0)

    def test_f_all_hom(self):
        """All hom: Ho=0, F=1."""
        hap = np.array([[0, 0], [0, 0], [1, 1], [1, 1],
                         [0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int8)
        pos = np.array([0, 1000])
        matrix = HaplotypeMatrix(hap, pos, 0, 2000)
        f = diversity.inbreeding_coefficient(matrix)
        np.testing.assert_allclose(f, 1.0)

    def test_f_vs_allel(self):
        """Compare F against allel's inbreeding_coefficient."""
        np.random.seed(42)
        n_ind = 20
        n_var = 50
        hap = np.random.randint(0, 2, (n_ind * 2, n_var), dtype=np.int8)
        pos = np.arange(n_var) * 1000
        matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)

        f_pg = diversity.inbreeding_coefficient(matrix)

        # allel
        g = np.zeros((n_var, n_ind, 2), dtype=np.int8)
        for i in range(n_ind):
            g[:, i, 0] = hap[2 * i]
            g[:, i, 1] = hap[2 * i + 1]
        ga = allel.GenotypeArray(g)
        f_allel = allel.inbreeding_coefficient(ga)

        # compare where both are valid
        both_valid = ~np.isnan(f_pg) & ~np.isnan(f_allel)
        np.testing.assert_allclose(
            f_pg[both_valid], f_allel[both_valid], rtol=1e-10)
