"""Tests for the 'pairwise' missing data mode (pixy-style comparison counting)."""

import numpy as np
import cupy as cp
import pytest
import warnings

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix
from pg_gpu import diversity, divergence
from pg_gpu.diversity import PairwiseResult, _pairwise_pi_components


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_matrix():
    """6 haplotypes, 4 variant sites, no missing data."""
    haps = np.array([
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.int8)
    pos = np.array([100, 200, 300, 400])
    return HaplotypeMatrix(haps, pos, chrom_start=0, chrom_end=500)


@pytest.fixture
def missing_matrix():
    """6 haplotypes, 4 sites, some missing data (-1)."""
    haps = np.array([
        [ 0,  1, -1,  1],
        [ 0,  0,  1,  0],
        [ 1, -1,  0,  1],
        [ 0,  1,  1, -1],
        [ 1,  1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.int8)
    pos = np.array([100, 200, 300, 400])
    return HaplotypeMatrix(haps, pos, chrom_start=0, chrom_end=500)


@pytest.fixture
def two_pop_matrix():
    """Matrix with two populations, some missing data."""
    haps = np.array([
        # pop1 (indices 0-3)
        [ 0,  1,  0,  1],
        [ 0,  0,  1,  0],
        [ 1, -1,  0,  1],
        [ 0,  1,  1, -1],
        # pop2 (indices 4-7)
        [ 1,  1,  0,  0],
        [ 0,  0,  0,  1],
        [ 1,  0, -1,  0],
        [ 0,  1,  0,  1],
    ], dtype=np.int8)
    pos = np.array([100, 200, 300, 400])
    sample_sets = {'pop1': [0, 1, 2, 3], 'pop2': [4, 5, 6, 7]}
    return HaplotypeMatrix(haps, pos, chrom_start=0, chrom_end=500,
                           sample_sets=sample_sets)


# ---------------------------------------------------------------------------
# Matrix infrastructure tests
# ---------------------------------------------------------------------------

class TestMatrixInvariantSupport:

    def test_n_total_sites_default_none(self, simple_matrix):
        assert simple_matrix.n_total_sites is None
        assert not simple_matrix.has_invariant_info
        assert simple_matrix.n_invariant_sites is None

    def test_n_total_sites_set(self, simple_matrix):
        simple_matrix.n_total_sites = 500
        assert simple_matrix.has_invariant_info
        n_inv = simple_matrix.n_invariant_sites
        assert n_inv is not None
        # 4 variant sites in simple_matrix -> 500 - 4 = 496 invariant
        assert n_inv == 496

    def test_genotype_matrix_propagation(self):
        haps = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8)
        pos = np.array([100, 200])
        hm = HaplotypeMatrix(haps, pos, n_total_sites=1000)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        assert gm.n_total_sites == 1000
        hm2 = gm.to_haplotype_matrix()
        assert hm2.n_total_sites == 1000


# ---------------------------------------------------------------------------
# _pairwise_pi_components tests
# ---------------------------------------------------------------------------

class TestPairwisePiComponents:

    def test_no_missing(self, simple_matrix):
        simple_matrix.transfer_to_gpu()
        diffs, comps, missing, n_sites = _pairwise_pi_components(
            simple_matrix.haplotypes)
        # 6 haplotypes, C(6,2) = 15 comparisons per site, 4 sites
        assert comps == 15 * 4
        assert n_sites == 4
        assert missing == 0
        assert diffs > 0

    def test_with_missing(self, missing_matrix):
        missing_matrix.transfer_to_gpu()
        diffs, comps, missing, n_sites = _pairwise_pi_components(
            missing_matrix.haplotypes)
        # Sites have variable n_valid: comps < 15*4
        assert comps < 15 * 4
        assert n_sites == 4
        assert diffs > 0

    def test_invariant_sites_increase_denominator(self, simple_matrix):
        simple_matrix.transfer_to_gpu()
        d1, c1, _, _ = _pairwise_pi_components(simple_matrix.haplotypes)
        d2, c2, _, _ = _pairwise_pi_components(
            simple_matrix.haplotypes, n_total_sites=100,
            n_haplotypes_full=6)
        assert d1 == d2  # same diffs
        assert c2 > c1   # more comparisons from invariant sites
        # pi should be lower with invariant sites
        pi_without = d1 / c1
        pi_with = d2 / c2
        assert pi_with < pi_without

    def test_hand_computed(self):
        """Verify against hand computation.

        Site 1: [0, 0, 1] -> derived=1, ancestral=2, diffs=1*2=2, C(3,2)=3
        Site 2: [1, 1, 0] -> derived=2, ancestral=1, diffs=2*1=2, C(3,2)=3
        pi = 4/6 = 0.6667
        """
        haps = cp.array([[0, 1], [0, 1], [1, 0]], dtype=cp.int8)
        diffs, comps, _, _ = _pairwise_pi_components(haps)
        assert diffs == pytest.approx(4.0)
        assert comps == pytest.approx(6.0)
        assert diffs / comps == pytest.approx(2 / 3, rel=1e-10)


# ---------------------------------------------------------------------------
# Pairwise pi tests
# ---------------------------------------------------------------------------

class TestPairwisePi:

    def test_returns_float(self, simple_matrix):
        result = diversity.pi(simple_matrix, missing_data='pairwise',
                              span_normalize=False)
        assert isinstance(result, float)
        assert result > 0

    def test_return_components(self, simple_matrix):
        result = diversity.pi(simple_matrix, missing_data='pairwise',
                              span_normalize=False, return_components=True)
        assert isinstance(result, PairwiseResult)
        assert result.total_diffs > 0
        assert result.total_comps > 0
        assert result.value == pytest.approx(
            result.total_diffs / result.total_comps)

    def test_warns_without_invariant_info(self, simple_matrix):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diversity.pi(simple_matrix, missing_data='pairwise',
                         span_normalize=False)
            assert any("invariant" in str(x.message).lower() for x in w)

    def test_no_warning_with_invariant_info(self, simple_matrix):
        simple_matrix.n_total_sites = 500
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diversity.pi(simple_matrix, missing_data='pairwise',
                         span_normalize=False)
            assert not any("invariant" in str(x.message).lower() for x in w)

    def test_invariant_sites_lower_pi(self, simple_matrix):
        pi_no_inv = diversity.pi(simple_matrix, missing_data='pairwise',
                                 span_normalize=False)
        simple_matrix.n_total_sites = 100
        pi_with_inv = diversity.pi(simple_matrix, missing_data='pairwise',
                                   span_normalize=False)
        assert pi_with_inv < pi_no_inv

    def test_with_missing_data(self, missing_matrix):
        result = diversity.pi(missing_matrix, missing_data='pairwise',
                              span_normalize=False)
        assert isinstance(result, float)
        assert result > 0

    def test_no_missing_agrees_with_include(self, simple_matrix):
        """Without missing data, pairwise pi equals include pi / n_sites.

        'include' with span_normalize=False returns sum of per-site pi.
        'pairwise' returns sum(diffs)/sum(comps) = per-comparison average.
        With constant n, these differ by a factor of n_sites.
        """
        pi_pw = diversity.pi(simple_matrix, missing_data='pairwise',
                             span_normalize=False)
        pi_inc = diversity.pi(simple_matrix, missing_data='include',
                              span_normalize=False)
        n_sites = simple_matrix.num_variants
        assert pi_pw == pytest.approx(pi_inc / n_sites, rel=1e-10)


# ---------------------------------------------------------------------------
# Pairwise dxy tests
# ---------------------------------------------------------------------------

class TestPairwiseDxy:

    def test_returns_float(self, two_pop_matrix):
        result = divergence.dxy(two_pop_matrix, 'pop1', 'pop2',
                                missing_data='pairwise')
        assert isinstance(result, float)
        assert result > 0

    def test_return_components(self, two_pop_matrix):
        result = divergence.dxy(two_pop_matrix, 'pop1', 'pop2',
                                missing_data='pairwise',
                                return_components=True)
        assert isinstance(result, PairwiseResult)
        assert result.value == pytest.approx(
            result.total_diffs / result.total_comps)

    def test_hand_computed(self):
        """Verify against hand computation.

        Pop1: [0, 1]  Pop2: [1, 0]
        Site 1: pop1_d=0, pop1_a=1, pop2_d=1, pop2_a=0
            diffs = 0*0 + 1*1 = 1, comps = 1*1 = 1
        Site 2: pop1_d=1, pop1_a=0, pop2_d=0, pop2_a=1
            diffs = 1*1 + 0*0 = 1, comps = 1*1 = 1
        dxy = 2/2 = 1.0
        """
        haps = np.array([[0, 1], [1, 0]], dtype=np.int8)
        pos = np.array([100, 200])
        m = HaplotypeMatrix(haps, pos, sample_sets={'a': [0], 'b': [1]})
        result = divergence.dxy(m, 'a', 'b', missing_data='pairwise')
        assert result == pytest.approx(1.0)

    def test_invariant_sites_lower_dxy(self, two_pop_matrix):
        dxy_no = divergence.dxy(two_pop_matrix, 'pop1', 'pop2',
                                missing_data='pairwise')
        two_pop_matrix.n_total_sites = 100
        dxy_with = divergence.dxy(two_pop_matrix, 'pop1', 'pop2',
                                  missing_data='pairwise')
        assert dxy_with < dxy_no


# ---------------------------------------------------------------------------
# Pairwise FST tests
# ---------------------------------------------------------------------------

class TestPairwiseFst:

    def test_returns_float(self, two_pop_matrix):
        result = divergence.fst_hudson(two_pop_matrix, 'pop1', 'pop2',
                                       missing_data='pairwise')
        assert isinstance(result, float)
        assert result <= 1

    def test_fst_dispatcher(self, two_pop_matrix):
        result = divergence.fst(two_pop_matrix, 'pop1', 'pop2',
                                method='hudson', missing_data='pairwise')
        assert isinstance(result, float)
        assert result <= 1


# ---------------------------------------------------------------------------
# Pairwise theta_w tests
# ---------------------------------------------------------------------------

class TestPairwiseThetaW:

    def test_returns_float(self, simple_matrix):
        result = diversity.theta_w(simple_matrix, missing_data='pairwise',
                                   span_normalize=False)
        assert isinstance(result, float)
        assert result > 0

    def test_return_components(self, simple_matrix):
        result = diversity.theta_w(simple_matrix, missing_data='pairwise',
                                   span_normalize=False,
                                   return_components=True)
        assert isinstance(result, PairwiseResult)
        # total_diffs is raw_theta, total_comps is n_sites
        assert result.total_diffs > 0

    def test_invariant_sites_lower_theta(self, simple_matrix):
        tw_no = diversity.theta_w(simple_matrix, missing_data='pairwise',
                                  span_normalize=False)
        simple_matrix.n_total_sites = 100
        tw_with = diversity.theta_w(simple_matrix, missing_data='pairwise',
                                    span_normalize=False)
        assert tw_with < tw_no


# ---------------------------------------------------------------------------
# Pairwise Tajima's D tests
# ---------------------------------------------------------------------------

class TestPairwiseTajimasD:

    def test_returns_float(self, simple_matrix):
        result = diversity.tajimas_d(simple_matrix, missing_data='pairwise')
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_with_missing(self, missing_matrix):
        result = diversity.tajimas_d(missing_matrix, missing_data='pairwise')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Segregating sites and singletons pass-through
# ---------------------------------------------------------------------------

class TestPairwisePassthrough:

    def test_segregating_sites(self, missing_matrix):
        s_pw = diversity.segregating_sites(missing_matrix,
                                           missing_data='pairwise')
        s_inc = diversity.segregating_sites(missing_matrix,
                                            missing_data='include')
        assert s_pw == s_inc

    def test_singleton_count(self, missing_matrix):
        s_pw = diversity.singleton_count(missing_matrix,
                                         missing_data='pairwise')
        s_inc = diversity.singleton_count(missing_matrix,
                                          missing_data='include')
        assert s_pw == s_inc


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_missing_one_site(self):
        """Site with all missing should contribute nothing."""
        haps = np.array([
            [ 0, -1],
            [ 1, -1],
            [ 0, -1],
        ], dtype=np.int8)
        pos = np.array([100, 200])
        m = HaplotypeMatrix(haps, pos)
        result = diversity.pi(m, missing_data='pairwise',
                              span_normalize=False, return_components=True)
        # Only site 1 contributes: derived=1, ancestral=2, diffs=2, comps=3
        assert result.total_diffs == pytest.approx(2.0)
        assert result.total_comps == pytest.approx(3.0)

    def test_monomorphic_sites_zero_diffs(self):
        """All-ref sites contribute 0 diffs but positive comps."""
        haps = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        pos = np.array([100, 200, 300])
        m = HaplotypeMatrix(haps, pos)
        result = diversity.pi(m, missing_data='pairwise',
                              span_normalize=False, return_components=True)
        # Sites 1&2 are monomorphic: 0 diffs, C(3,2)=3 comps each
        # Site 3: derived=1, ancestral=2, diffs=2, comps=3
        assert result.total_diffs == pytest.approx(2.0)
        assert result.total_comps == pytest.approx(9.0)

    def test_single_sample_excluded(self):
        """Sites with only 1 valid sample should be excluded."""
        haps = np.array([
            [ 0, -1],
            [-1, -1],
            [-1, -1],
        ], dtype=np.int8)
        pos = np.array([100, 200])
        m = HaplotypeMatrix(haps, pos)
        result = diversity.pi(m, missing_data='pairwise',
                              span_normalize=False)
        # No site has >= 2 valid samples
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# theta_h tests
# ---------------------------------------------------------------------------

class TestPairwiseThetaH:

    def test_returns_float(self, simple_matrix):
        result = diversity.theta_h(simple_matrix, span_normalize=False,
                                   missing_data='pairwise')
        assert isinstance(result, float)
        assert result > 0

    def test_no_missing_agrees_with_include(self, simple_matrix):
        th_pw = diversity.theta_h(simple_matrix, span_normalize=False,
                                  missing_data='pairwise')
        th_inc = diversity.theta_h(simple_matrix, span_normalize=False,
                                  missing_data='include')
        assert th_pw == pytest.approx(th_inc, rel=1e-10)

    def test_with_missing(self, missing_matrix):
        result = diversity.theta_h(missing_matrix, span_normalize=False,
                                   missing_data='pairwise')
        assert isinstance(result, float)

    def test_exclude_mode(self, missing_matrix):
        result = diversity.theta_h(missing_matrix, span_normalize=False,
                                   missing_data='exclude')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# theta_l tests
# ---------------------------------------------------------------------------

class TestPairwiseThetaL:

    def test_returns_float(self, simple_matrix):
        result = diversity.theta_l(simple_matrix, span_normalize=False,
                                   missing_data='pairwise')
        assert isinstance(result, float)
        assert result > 0

    def test_no_missing_agrees_with_include(self, simple_matrix):
        tl_pw = diversity.theta_l(simple_matrix, span_normalize=False,
                                  missing_data='pairwise')
        tl_inc = diversity.theta_l(simple_matrix, span_normalize=False,
                                  missing_data='include')
        assert tl_pw == pytest.approx(tl_inc, rel=1e-10)

    def test_with_missing(self, missing_matrix):
        result = diversity.theta_l(missing_matrix, span_normalize=False,
                                   missing_data='pairwise')
        assert isinstance(result, float)

    def test_hand_computed(self):
        """theta_L = sum(d_i / (n_i - 1)) per site.

        Site 1: [0, 1, 0] -> d=1, n=3, contrib = 1/2 = 0.5
        Site 2: [1, 1, 0] -> d=2, n=3, contrib = 2/2 = 1.0
        theta_L = 1.5
        """
        haps = cp.array([[0, 1], [1, 1], [0, 0]], dtype=cp.int8)
        pos = cp.array([100, 200])
        m = HaplotypeMatrix(haps, pos)
        result = diversity.theta_l(m, span_normalize=False,
                                   missing_data='pairwise')
        assert result == pytest.approx(1.5, rel=1e-10)


# ---------------------------------------------------------------------------
# normalized_fay_wus_h tests
# ---------------------------------------------------------------------------

class TestPairwiseNormalizedH:

    def test_returns_float(self, simple_matrix):
        result = diversity.normalized_fay_wus_h(simple_matrix,
                                                missing_data='pairwise')
        assert isinstance(result, float)

    def test_no_missing_agrees_with_include(self, simple_matrix):
        h_pw = diversity.normalized_fay_wus_h(simple_matrix,
                                              missing_data='pairwise')
        h_inc = diversity.normalized_fay_wus_h(simple_matrix,
                                              missing_data='include')
        assert h_pw == pytest.approx(h_inc, rel=1e-6)

    def test_with_missing(self, missing_matrix):
        result = diversity.normalized_fay_wus_h(missing_matrix,
                                                missing_data='pairwise')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# fay_wus_h pairwise tests
# ---------------------------------------------------------------------------

class TestPairwiseFayWuH:

    def test_returns_float(self, simple_matrix):
        result = diversity.fay_wus_h(simple_matrix, missing_data='pairwise')
        assert isinstance(result, float)

    def test_no_missing_agrees_with_include(self, simple_matrix):
        h_pw = diversity.fay_wus_h(simple_matrix, missing_data='pairwise')
        h_inc = diversity.fay_wus_h(simple_matrix, missing_data='include')
        # Without missing data these should be very close
        assert h_pw == pytest.approx(h_inc, rel=1e-6)

    def test_with_missing(self, missing_matrix):
        result = diversity.fay_wus_h(missing_matrix, missing_data='pairwise')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# zeng_e tests
# ---------------------------------------------------------------------------

class TestPairwiseZengE:

    def test_returns_float(self, simple_matrix):
        result = diversity.zeng_e(simple_matrix, missing_data='pairwise')
        assert isinstance(result, float)

    def test_no_missing_agrees_with_include(self, simple_matrix):
        e_pw = diversity.zeng_e(simple_matrix, missing_data='pairwise')
        e_inc = diversity.zeng_e(simple_matrix, missing_data='include')
        assert e_pw == pytest.approx(e_inc, rel=1e-6)

    def test_with_missing(self, missing_matrix):
        result = diversity.zeng_e(missing_matrix, missing_data='pairwise')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# zeng_dh tests
# ---------------------------------------------------------------------------

class TestPairwiseZengDH:

    def test_returns_float(self, simple_matrix):
        result = diversity.zeng_dh(simple_matrix, missing_data='pairwise')
        assert isinstance(result, float)

    def test_with_missing(self, missing_matrix):
        result = diversity.zeng_dh(missing_matrix, missing_data='pairwise')
        assert isinstance(result, float)
