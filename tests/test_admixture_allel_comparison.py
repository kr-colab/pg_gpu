"""
Validation tests comparing pg_gpu admixture/F-statistics against scikit-allel.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import admixture


def _make_matrix(hap_dict, n_var=100):
    """Build a HaplotypeMatrix from a dict of pop_name -> haplotype array."""
    parts = []
    sample_sets = {}
    offset = 0
    for name, hap in hap_dict.items():
        parts.append(hap)
        n = hap.shape[0]
        sample_sets[name] = list(range(offset, offset + n))
        offset += n

    combined = np.vstack(parts)
    pos = np.arange(combined.shape[1]) * 1000
    return HaplotypeMatrix(
        combined, pos, 0, combined.shape[1] * 1000,
        sample_sets=sample_sets
    )


def _allele_counts(hap):
    """Allele counts (n_variants, 2); ``-1`` counts as a missing call."""
    return np.column_stack([np.sum(hap == 0, axis=0), np.sum(hap == 1, axis=0)])


@pytest.fixture
def four_pop_data():
    """Four-population data for Patterson D tests."""
    np.random.seed(42)
    n_var = 200
    pops = {}
    for name in ['A', 'B', 'C', 'D']:
        pops[name] = np.random.randint(0, 2, (8, n_var), dtype=np.int8)
    matrix = _make_matrix(pops, n_var)
    return matrix, pops


@pytest.fixture
def three_pop_data():
    """Three-population data for Patterson F3 tests."""
    np.random.seed(123)
    n_var = 200
    pops = {}
    for name in ['C', 'A', 'B']:
        pops[name] = np.random.randint(0, 2, (10, n_var), dtype=np.int8)
    matrix = _make_matrix(pops, n_var)
    return matrix, pops


@pytest.fixture
def two_pop_data():
    """Two-population data for Patterson F2 tests."""
    np.random.seed(456)
    n_var = 200
    pops = {}
    for name in ['A', 'B']:
        pops[name] = np.random.randint(0, 2, (10, n_var), dtype=np.int8)
    matrix = _make_matrix(pops, n_var)
    return matrix, pops


class TestPattersonF2Comparison:
    """Validate Patterson F2 against scikit-allel."""

    def test_f2_correspondence(self, two_pop_data):
        matrix, pops = two_pop_data
        result = admixture.patterson_f2(matrix, 'A', 'B')

        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        expected = allel.patterson_f2(aca, acb)

        both_valid = ~np.isnan(result) & ~np.isnan(expected)
        np.testing.assert_allclose(
            result[both_valid], expected[both_valid],
            rtol=1e-10, err_msg="F2 does not match allel"
        )


class TestPattersonF3Comparison:
    """Validate Patterson F3 against scikit-allel."""

    def test_f3_correspondence(self, three_pop_data):
        matrix, pops = three_pop_data
        T_pg, B_pg = admixture.patterson_f3(matrix, 'C', 'A', 'B')

        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        T_allel, B_allel = allel.patterson_f3(acc, aca, acb)

        both_valid = ~np.isnan(T_pg) & ~np.isnan(T_allel)
        np.testing.assert_allclose(
            T_pg[both_valid], T_allel[both_valid], rtol=1e-10,
            err_msg="F3 numerator does not match"
        )
        np.testing.assert_allclose(
            B_pg[both_valid], B_allel[both_valid], rtol=1e-10,
            err_msg="F3 denominator does not match"
        )

    def test_f3_normalized(self, three_pop_data):
        matrix, pops = three_pop_data
        T_pg, B_pg = admixture.patterson_f3(matrix, 'C', 'A', 'B')
        f3_pg = np.nansum(T_pg) / np.nansum(B_pg)

        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        T_a, B_a = allel.patterson_f3(acc, aca, acb)
        f3_allel = np.nansum(T_a) / np.nansum(B_a)

        np.testing.assert_allclose(f3_pg, f3_allel, rtol=1e-10)


class TestPattersonDComparison:
    """Validate Patterson D against scikit-allel."""

    def test_d_correspondence(self, four_pop_data):
        matrix, pops = four_pop_data
        num_pg, den_pg = admixture.patterson_d(matrix, 'A', 'B', 'C', 'D')

        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        acc = _allele_counts(pops['C'])
        acd = _allele_counts(pops['D'])
        num_allel, den_allel = allel.patterson_d(aca, acb, acc, acd)

        both_valid = ~np.isnan(num_pg) & ~np.isnan(num_allel)
        np.testing.assert_allclose(
            num_pg[both_valid], num_allel[both_valid], rtol=1e-10,
            err_msg="D numerator does not match"
        )
        np.testing.assert_allclose(
            den_pg[both_valid], den_allel[both_valid], rtol=1e-10,
            err_msg="D denominator does not match"
        )

    def test_d_normalized(self, four_pop_data):
        matrix, pops = four_pop_data
        num_pg, den_pg = admixture.patterson_d(matrix, 'A', 'B', 'C', 'D')
        d_pg = np.nansum(num_pg) / np.nansum(den_pg)

        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        acc = _allele_counts(pops['C'])
        acd = _allele_counts(pops['D'])
        num_a, den_a = allel.patterson_d(aca, acb, acc, acd)
        d_allel = np.nansum(num_a) / np.nansum(den_a)

        np.testing.assert_allclose(d_pg, d_allel, rtol=1e-10)


class TestAveragePattersonComparison:
    """Validate block-jackknife Patterson statistics."""

    def test_average_f3(self, three_pop_data):
        matrix, pops = three_pop_data
        f3_pg, se_pg, z_pg, _, _ = admixture.average_patterson_f3(
            matrix, 'C', 'A', 'B', blen=20)

        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        f3_allel, se_allel, z_allel, _, _ = allel.average_patterson_f3(
            acc, aca, acb, blen=20)

        np.testing.assert_allclose(f3_pg, f3_allel, rtol=1e-10)
        np.testing.assert_allclose(se_pg, se_allel, rtol=1e-5)

    def test_average_d(self, four_pop_data):
        matrix, pops = four_pop_data
        d_pg, se_pg, z_pg, _, _ = admixture.average_patterson_d(
            matrix, 'A', 'B', 'C', 'D', blen=20)

        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        acc = _allele_counts(pops['C'])
        acd = _allele_counts(pops['D'])
        d_allel, se_allel, z_allel, _, _ = allel.average_patterson_d(
            aca, acb, acc, acd, blen=20)

        np.testing.assert_allclose(d_pg, d_allel, rtol=1e-10)
        np.testing.assert_allclose(se_pg, se_allel, rtol=1e-5)


class TestKnownValues:
    """Test Patterson stats with known expected values (from allel tests)."""

    def test_patterson_f2_known(self):
        """Reproduce allel's test_patterson_f2 expected values."""
        # 4 variants, 2 populations with 2 haplotypes each
        hap_a = np.array([[0, 0, 1, 0],
                          [0, 1, 0, 0]], dtype=np.int8)
        hap_b = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int8)
        matrix = _make_matrix({'A': hap_a, 'B': hap_b}, n_var=4)

        f2 = admixture.patterson_f2(matrix, 'A', 'B')
        # Expected: variant 0 (both 0): f2=0; variant 1 (freq diff 0.5): f2>0
        # With n=2 per pop, h_hat = n0*n1/(n*(n-1)) = adjustments matter
        assert f2.shape == (4,)
        assert np.isclose(f2[0], 0.0)  # no difference

    def test_patterson_d_known(self):
        """Basic D-statistic sanity check."""
        np.random.seed(999)
        n_var = 50
        # identical A and B -> D should be ~0
        hap_same = np.random.randint(0, 2, (6, n_var), dtype=np.int8)
        hap_c = np.random.randint(0, 2, (6, n_var), dtype=np.int8)
        hap_d = np.random.randint(0, 2, (6, n_var), dtype=np.int8)
        matrix = _make_matrix({
            'A': hap_same, 'B': hap_same.copy(),
            'C': hap_c, 'D': hap_d
        }, n_var=n_var)

        num, den = admixture.patterson_d(matrix, 'A', 'B', 'C', 'D')
        # numerator should be exactly 0 since a == b
        np.testing.assert_array_almost_equal(num, 0.0)


class TestPattersonDZeroCallSites:
    """Sites where one population has no called gametes leave D entirely."""

    @pytest.fixture
    def zero_call_data(self, four_pop_data):
        """four_pop_data with all calls missing in one pop at some sites."""
        _, base = four_pop_data
        pops = {name: hap.copy() for name, hap in base.items()}
        missing = np.zeros(pops['A'].shape[1], dtype=bool)
        pops['A'][:, 10:20] = -1
        pops['C'][:, 50:55] = -1
        missing[10:20] = True
        missing[50:55] = True
        return _make_matrix(pops), pops, missing

    def test_zero_call_sites_match_allel(self, zero_call_data):
        matrix, pops, missing = zero_call_data
        num_pg, den_pg = admixture.patterson_d(matrix, 'A', 'B', 'C', 'D')
        acs = [_allele_counts(pops[p]) for p in 'ABCD']
        num_a, den_a = allel.patterson_d(*acs)

        # allel's nan sites are exactly our zeroed sites.
        undefined = np.isnan(num_a) | np.isnan(den_a)
        np.testing.assert_array_equal(undefined, missing)
        np.testing.assert_array_equal(num_pg[undefined], 0.0)
        np.testing.assert_array_equal(den_pg[undefined], 0.0)
        # The defined sites agree.
        np.testing.assert_allclose(num_pg[~undefined], num_a[~undefined],
                                   rtol=1e-10)
        np.testing.assert_allclose(den_pg[~undefined], den_a[~undefined],
                                   rtol=1e-10)

    def test_zero_call_site_contributes_nothing(self):
        """The aggregate D over [zero-call site, informative site] is the
        informative site's own ratio; a frequency-0 stand-in would have let
        the zero-call site move it."""
        pops = {
            'A': np.array([[-1, 1], [-1, 1]], dtype=np.int8),
            'B': np.array([[1, 1], [1, 0]], dtype=np.int8),
            'C': np.array([[1, 1], [1, 0]], dtype=np.int8),
            'D': np.array([[0, 0], [0, 0]], dtype=np.int8),
        }
        num, den = admixture.patterson_d(_make_matrix(pops), 'A', 'B', 'C', 'D')
        assert num[0] == 0.0 and den[0] == 0.0
        np.testing.assert_allclose(np.nansum(num) / np.nansum(den),
                                   num[1] / den[1], rtol=1e-12)

    def test_moving_d_matches_allel_with_zero_call_sites(self, zero_call_data):
        """size=5 puts three windows entirely inside the missing blocks, so
        this also pins the all-undefined-window -> NaN convention."""
        matrix, pops, _ = zero_call_data
        d_pg = admixture.moving_patterson_d(matrix, 'A', 'B', 'C', 'D', size=5)
        acs = [_allele_counts(pops[p]) for p in 'ABCD']
        with np.errstate(invalid='ignore'):  # allel reaches NaN through 0/0
            d_a = allel.moving_patterson_d(*acs, size=5)
        np.testing.assert_array_equal(np.isnan(d_pg), np.isnan(d_a))
        valid = ~np.isnan(d_a)
        np.testing.assert_allclose(d_pg[valid], d_a[valid], rtol=1e-10)

    def test_average_d_matches_allel_with_zero_call_sites(self, zero_call_data):
        matrix, pops, _ = zero_call_data
        d_pg, se_pg, z_pg, _, _ = admixture.average_patterson_d(
            matrix, 'A', 'B', 'C', 'D', blen=20)
        acs = [_allele_counts(pops[p]) for p in 'ABCD']
        d_a, se_a, z_a, _, _ = allel.average_patterson_d(*acs, blen=20)
        np.testing.assert_allclose(d_pg, d_a, rtol=1e-10)
        np.testing.assert_allclose(se_pg, se_a, rtol=1e-5)

    def test_exclude_mode_drops_only_the_missing_sites(self, zero_call_data):
        matrix, _, missing = zero_call_data
        num, den = admixture.patterson_d(matrix, 'A', 'B', 'C', 'D',
                                         missing_data='exclude')
        # The base data has no other missing calls, so exclude keeps
        # exactly the fully-called sites.
        assert num.shape == den.shape == ((~missing).sum(),)
        assert np.isfinite(np.nansum(num) / np.nansum(den))


class TestPattersonF2ZeroCallSites:
    """A site with fewer than 2 calls in a pop is non-estimable: NaN, not 0."""

    @pytest.fixture
    def zero_call_data(self, two_pop_data):
        """two_pop_data with all calls missing in A at some sites."""
        _, base = two_pop_data
        pops = {name: hap.copy() for name, hap in base.items()}
        undefined = np.zeros(pops['A'].shape[1], dtype=bool)
        pops['A'][:, 10:20] = -1
        undefined[10:20] = True
        return _make_matrix(pops), pops, undefined

    def test_f2_zero_call_sites_match_allel(self, zero_call_data):
        matrix, pops, undefined = zero_call_data
        f2_pg = admixture.patterson_f2(matrix, 'A', 'B')
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        f2_a = allel.patterson_f2(aca, acb)

        np.testing.assert_array_equal(np.isnan(f2_a), undefined)
        np.testing.assert_array_equal(np.isnan(f2_pg), undefined)
        np.testing.assert_allclose(f2_pg[~undefined], f2_a[~undefined],
                                   rtol=1e-10)


class TestPattersonF3ZeroCallSites:
    """A site where C has fewer than 2 calls, or a source has none, leaves
    T non-estimable: NaN, not 0. These sites previously biased the
    normed=False mean of average_patterson_f3/moving_patterson_f3 toward 0."""

    @pytest.fixture
    def zero_call_data(self, three_pop_data):
        """three_pop_data with all calls missing in C at some sites, and in
        source A at others."""
        _, base = three_pop_data
        pops = {name: hap.copy() for name, hap in base.items()}
        undefined = np.zeros(pops['C'].shape[1], dtype=bool)
        pops['C'][:, 10:20] = -1
        pops['A'][:, 50:55] = -1
        undefined[10:20] = True
        undefined[50:55] = True
        return _make_matrix(pops), pops, undefined

    def test_f3_zero_call_sites_match_allel(self, zero_call_data):
        matrix, pops, undefined = zero_call_data
        T_pg, B_pg = admixture.patterson_f3(matrix, 'C', 'A', 'B')
        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        T_a, _ = allel.patterson_f3(acc, aca, acb)

        np.testing.assert_array_equal(np.isnan(T_a), undefined)
        np.testing.assert_array_equal(np.isnan(T_pg), undefined)
        np.testing.assert_allclose(T_pg[~undefined], T_a[~undefined],
                                   rtol=1e-10)

    def test_average_f3_unnormed_matches_allel_with_zero_call_sites(
            self, zero_call_data):
        """Every block here still has at least one defined site (the
        missing calls only partially cover two blocks), so no block is
        dropped and this matches allel's own unweighted mean of block
        means exactly, same as with no missing data at all."""
        matrix, pops, _ = zero_call_data
        f3_pg, se_pg, z_pg, _, _ = admixture.average_patterson_f3(
            matrix, 'C', 'A', 'B', blen=20, normed=False)
        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        f3_a, se_a, z_a, _, _ = allel.average_patterson_f3(
            acc, aca, acb, blen=20, normed=False)
        np.testing.assert_allclose(f3_pg, f3_a, rtol=1e-10)
        np.testing.assert_allclose(se_pg, se_a, rtol=1e-5)

    def test_moving_f3_unnormed_matches_allel_with_zero_call_sites(
            self, zero_call_data):
        matrix, pops, _ = zero_call_data
        f3_pg = admixture.moving_patterson_f3(
            matrix, 'C', 'A', 'B', size=20, normed=False)
        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        with np.errstate(invalid='ignore'):
            f3_a = allel.moving_patterson_f3(acc, aca, acb, size=20,
                                             normed=False)
        np.testing.assert_array_equal(np.isnan(f3_pg), np.isnan(f3_a))
        valid = ~np.isnan(f3_a)
        np.testing.assert_allclose(f3_pg[valid], f3_a[valid], rtol=1e-10)


class TestAverageF3UnnormedWholeBlockUndefined:
    """A block with no estimable site used to poison average_patterson_f3's
    normed=False jackknife entirely: every leave-one-out estimate except
    the one dropping that block kept its NaN per-block mean, so se/z came
    out NaN even with only one uninformative block among many. That block
    now drops out of the resampling instead; every other block still
    contributes its own unweighted mean, exactly as scikit-allel would
    compute it with that one block already removed."""

    def test_whole_block_undefined_does_not_poison_se(self):
        np.random.seed(123)
        n_var, blen = 200, 20
        pops = {name: np.random.randint(0, 2, (10, n_var)).astype(np.int8)
               for name in ['C', 'A', 'B']}
        pops['C'][:, 20:40] = -1  # block 1 entirely non-estimable
        matrix = _make_matrix(pops, n_var)

        f3, se, z, vb, vj = admixture.average_patterson_f3(
            matrix, 'C', 'A', 'B', blen=blen, normed=False)

        assert np.isnan(vb[1])  # the bad block's own value is still NaN
        assert np.isfinite(f3)
        assert np.isfinite(se)
        assert np.isfinite(z)
        assert len(vj) == len(vb)  # vj stays aligned with vb
        assert np.isnan(vj[1])  # the dropped block has no leave-one-out estimate
        assert np.isfinite(np.delete(vj, 1)).all()  # no other estimate is poisoned

        # Independent check: drop the bad block, then take the plain
        # (unweighted) delete-1 jackknife over what's left -- scikit-allel's
        # own formula, applied to the n-1 defined blocks.
        defined = vb[np.isfinite(vb)]
        n = len(defined)
        keep = np.ones(n, dtype=bool)
        per_iter = np.empty(n)
        for i in range(n):
            keep[i] = False
            per_iter[i] = defined[keep].mean()
            keep[i] = True
        expected_se = np.sqrt(((n - 1) / n)
                              * np.sum((per_iter - per_iter.mean()) ** 2))
        np.testing.assert_allclose(se, expected_se, rtol=1e-10)
        np.testing.assert_allclose(vj[np.isfinite(vb)], per_iter, rtol=1e-10)

    def test_every_block_undefined_is_nan_not_a_crash(self):
        """With no block left to jackknife over, se/z are NaN (as when
        missing_data='exclude' leaves no site at all), not a
        ZeroDivisionError from an empty block set."""
        np.random.seed(1)
        n_var, blen = 50, 10
        pops = {name: np.random.randint(0, 2, (10, n_var)).astype(np.int8)
               for name in ['C', 'A', 'B']}
        pops['C'][:] = -1  # every site non-estimable in C
        matrix = _make_matrix(pops, n_var)

        f3, se, z, vb, vj = admixture.average_patterson_f3(
            matrix, 'C', 'A', 'B', blen=blen, normed=False)

        assert np.isnan(f3) and np.isnan(se) and np.isnan(z)
        assert np.isnan(vb).all()
        assert len(vj) == len(vb)
        assert np.isnan(vj).all()


class TestPattersonF4ZeroCallSites:
    """A site where any pop has no call is non-estimable: NaN, not 0.
    scikit-allel has no patterson_f4 to compare against, so this checks
    pg_gpu's own convention directly (mirroring
    TestPattersonDZeroCallSites.test_zero_call_site_contributes_nothing)."""

    def test_zero_call_site_is_nan(self):
        pops = {
            'A': np.array([[-1, 1], [-1, 1]], dtype=np.int8),
            'B': np.array([[1, 1], [1, 0]], dtype=np.int8),
            'C': np.array([[1, 1], [1, 0]], dtype=np.int8),
            'D': np.array([[0, 0], [0, 0]], dtype=np.int8),
        }
        f4 = admixture.patterson_f4(_make_matrix(pops), 'A', 'B', 'C', 'D')
        assert np.isnan(f4[0])
        assert np.isfinite(f4[1])


class TestUnnormedF3Comparison:
    """The normed=False F3 paths (per-window/aggregate mean of the numerator)
    against scikit-allel."""

    def test_moving_f3_unnormed(self, three_pop_data):
        matrix, pops = three_pop_data
        f3_pg = admixture.moving_patterson_f3(
            matrix, 'C', 'A', 'B', size=20, normed=False)
        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        f3_a = allel.moving_patterson_f3(acc, aca, acb, size=20, normed=False)
        np.testing.assert_allclose(f3_pg, f3_a, rtol=1e-10)

    def test_average_f3_unnormed(self, three_pop_data):
        matrix, pops = three_pop_data
        f3_pg, se_pg, z_pg, _, _ = admixture.average_patterson_f3(
            matrix, 'C', 'A', 'B', blen=20, normed=False)
        acc = _allele_counts(pops['C'])
        aca = _allele_counts(pops['A'])
        acb = _allele_counts(pops['B'])
        f3_a, se_a, z_a, _, _ = allel.average_patterson_f3(
            acc, aca, acb, blen=20, normed=False)
        np.testing.assert_allclose(f3_pg, f3_a, rtol=1e-10)
        np.testing.assert_allclose(se_pg, se_a, rtol=1e-5)
