"""
Tests for GPU-native windowed statistics.

Validates windowed_analysis() against scikit-allel's windowed functions
and against pg_gpu's own per-variant functions for consistency.

These checks previously targeted the removed ``windowed_statistics`` scatter
engine; they now run against the public ``windowed_analysis`` dispatcher. The
windows are built the same way on both sides -- non-overlapping windows of
``size`` bp over ``[positions[0], positions[-1]]`` -- so the per-window values
line up for an exact comparison.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis


def _allele_counts(hap):
    """Allele counts (n_variants, 2) from haplotype array."""
    n = hap.shape[0]
    dac = np.sum(hap, axis=0)
    return np.column_stack([n - dac, dac])


@pytest.fixture
def sim_data():
    """Simulated haplotype data for windowed tests."""
    np.random.seed(42)
    n_hap = 40
    n_var = 500
    hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
    # positions spread across 100kb
    pos = np.sort(np.random.randint(1, 100001, n_var))
    # ensure unique positions
    pos = np.unique(pos)
    n_var = len(pos)
    hap = hap[:, :n_var]
    matrix = HaplotypeMatrix(hap, pos, int(pos[0]), int(pos[-1]))
    return matrix, hap, pos


@pytest.fixture
def two_pop_data():
    """Two-population data for FST/Dxy windowed tests."""
    np.random.seed(123)
    n_var = 500
    n1, n2 = 20, 20
    hap = np.random.randint(0, 2, (n1 + n2, n_var), dtype=np.int8)
    pos = np.sort(np.random.randint(1, 100001, n_var))
    pos = np.unique(pos)
    n_var = len(pos)
    hap = hap[:, :n_var]
    matrix = HaplotypeMatrix(
        hap, pos, int(pos[0]), int(pos[-1]),
        sample_sets={'pop1': list(range(n1)), 'pop2': list(range(n1, n1 + n2))}
    )
    return matrix, hap, pos, n1, n2


class TestWindowedPi:
    """Validate windowed pi against allel."""

    def test_pi_vs_allel(self, sim_data):
        matrix, hap, pos = sim_data
        size = 10000
        start, stop = int(pos[0]), int(pos[-1])

        # pg_gpu
        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['pi'])
        pi_pg = result['pi'].to_numpy()

        # allel
        ac = _allele_counts(hap)
        mpd = allel.mean_pairwise_difference(ac, fill=0)
        mpd_sum, windows, counts = allel.windowed_statistic(
            pos, mpd, statistic=np.sum, size=size, start=start, stop=stop)
        pi_allel, _ = allel.per_base(mpd_sum, windows)

        # compare
        both_valid = ~np.isnan(pi_pg) & ~np.isnan(pi_allel)
        assert np.sum(both_valid) > 0
        np.testing.assert_allclose(
            pi_pg[both_valid], pi_allel[both_valid],
            rtol=1e-3, err_msg="windowed pi does not match allel")

    def test_pi_no_normalize(self, sim_data):
        matrix, hap, pos = sim_data
        size = 10000

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['pi'], span_normalize=False)
        pi_raw = result['pi'].to_numpy()
        assert not np.all(np.isnan(pi_raw))


class TestWindowedThetaW:
    """Validate windowed Watterson's theta against allel."""

    def test_theta_w_vs_allel(self, sim_data):
        matrix, hap, pos = sim_data
        size = 10000
        start, stop = int(pos[0]), int(pos[-1])

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['theta_w'])
        theta_pg = result['theta_w'].to_numpy()

        # allel
        ac = _allele_counts(hap)
        is_seg = (ac[:, 0] > 0) & (ac[:, 1] > 0)
        S, windows, counts = allel.windowed_statistic(
            pos, is_seg, statistic=np.sum, size=size, start=start, stop=stop)
        n = hap.shape[0]
        a1 = np.sum(1.0 / np.arange(1, n))
        theta_abs = S / a1
        theta_allel, _ = allel.per_base(theta_abs, windows)

        both_valid = ~np.isnan(theta_pg) & ~np.isnan(theta_allel)
        assert np.sum(both_valid) > 0
        np.testing.assert_allclose(
            theta_pg[both_valid], theta_allel[both_valid],
            rtol=1e-3, err_msg="windowed theta_w does not match allel")


class TestWindowedTajimasD:
    """Validate windowed Tajima's D."""

    def test_tajimas_d_shape(self, sim_data):
        matrix, hap, pos = sim_data
        size = 20000

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['tajimas_d'])
        tajd = result['tajimas_d'].to_numpy()
        n_windows = int(np.ceil((pos[-1] - pos[0]) / size))
        assert tajd.shape[0] == n_windows

    def test_tajimas_d_finite(self, sim_data):
        """Tajima's D should produce finite values for windows with data."""
        matrix, hap, pos = sim_data
        size = 25000

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['tajimas_d'])
        tajd = result['tajimas_d'].to_numpy()
        valid = ~np.isnan(tajd)
        assert np.sum(valid) > 0
        assert np.all(np.isfinite(tajd[valid]))


class TestWindowedSegSites:
    """Validate segregating sites count."""

    def test_seg_sites_vs_allel(self, sim_data):
        matrix, hap, pos = sim_data
        size = 10000
        start, stop = int(pos[0]), int(pos[-1])

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['segregating_sites'])
        seg_pg = result['segregating_sites'].to_numpy()

        # allel
        ac = _allele_counts(hap)
        is_seg = (ac[:, 0] > 0) & (ac[:, 1] > 0)
        S, _, _ = allel.windowed_statistic(
            pos, is_seg, statistic=np.sum, size=size, start=start, stop=stop)

        np.testing.assert_array_equal(seg_pg.astype(int), S.astype(int))


class TestWindowedHetExpected:
    """Validate windowed expected heterozygosity against allel.

    het_expected is the per-window mean of per-site expected heterozygosity
    (``1 - sum_a p_a^2``) over the sites with a defined value.
    """

    def test_het_expected_vs_allel(self, sim_data):
        matrix, hap, pos = sim_data
        size = 10000
        start, stop = int(pos[0]), int(pos[-1])

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['het_expected'])
        he_pg = result['het_expected'].to_numpy()

        # allel reference: per-site He = 1 - sum_a p_a^2, averaged per window
        ac = _allele_counts(hap)
        p = ac / hap.shape[0]
        he_site = 1.0 - (p ** 2).sum(axis=1)
        he_allel, windows, counts = allel.windowed_statistic(
            pos, he_site, statistic=np.mean, size=size, start=start, stop=stop)

        both_valid = ~np.isnan(he_pg) & ~np.isnan(he_allel)
        assert np.sum(both_valid) > 0
        np.testing.assert_allclose(
            he_pg[both_valid], he_allel[both_valid],
            rtol=1e-6, err_msg="windowed het_expected does not match allel")


class TestWindowedFST:
    """Validate windowed Hudson FST."""

    def test_fst_vs_allel(self, two_pop_data):
        matrix, hap, pos, n1, n2 = two_pop_data
        size = 20000
        start, stop = int(pos[0]), int(pos[-1])

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['fst'],
                                   populations=['pop1', 'pop2'])
        fst_pg = result['fst'].to_numpy()

        # allel
        ac1 = _allele_counts(hap[:n1])
        ac2 = _allele_counts(hap[n1:])
        num, den = allel.hudson_fst(ac1, ac2)

        def avg_fst(wn, wd):
            d = np.nansum(wd)
            return np.nansum(wn) / d if d != 0 else np.nan

        fst_allel, _, _ = allel.windowed_statistic(
            pos, values=(num, den), statistic=avg_fst,
            size=size, start=start, stop=stop)

        both_valid = ~np.isnan(fst_pg) & ~np.isnan(fst_allel)
        if np.sum(both_valid) > 0:
            np.testing.assert_allclose(
                fst_pg[both_valid], fst_allel[both_valid],
                rtol=1e-3, err_msg="windowed FST does not match allel")


class TestWindowedDxy:
    """Validate windowed Dxy."""

    def test_dxy_vs_allel(self, two_pop_data):
        matrix, hap, pos, n1, n2 = two_pop_data
        size = 20000
        start, stop = int(pos[0]), int(pos[-1])

        result = windowed_analysis(matrix, window_size=size, step_size=size,
                                   statistics=['dxy'],
                                   populations=['pop1', 'pop2'])
        dxy_pg = result['dxy'].to_numpy()

        # allel
        ac1 = _allele_counts(hap[:n1])
        ac2 = _allele_counts(hap[n1:])
        mpd_b = allel.mean_pairwise_difference_between(ac1, ac2, fill=0)
        dxy_sum, windows, _ = allel.windowed_statistic(
            pos, mpd_b, statistic=np.sum, size=size, start=start, stop=stop)
        dxy_allel, _ = allel.per_base(dxy_sum, windows)

        both_valid = ~np.isnan(dxy_pg) & ~np.isnan(dxy_allel)
        if np.sum(both_valid) > 0:
            np.testing.assert_allclose(
                dxy_pg[both_valid], dxy_allel[both_valid],
                rtol=1e-3, err_msg="windowed Dxy does not match allel")


class TestMultipleStats:
    """Test computing multiple statistics at once."""

    def test_all_single_pop(self, sim_data):
        matrix, _, pos = sim_data
        size = 20000

        result = windowed_analysis(
            matrix, window_size=size, step_size=size,
            statistics=['pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                        'singletons', 'het_expected'])

        for key in ('pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                    'singletons', 'het_expected'):
            assert key in result.columns
        n_windows = int(np.ceil((pos[-1] - pos[0]) / size))
        for key in ('pi', 'theta_w', 'tajimas_d'):
            assert result[key].to_numpy().shape == (n_windows,)

    def test_mixed_stats(self, two_pop_data):
        matrix, _, _, _, _ = two_pop_data
        size = 25000

        result = windowed_analysis(
            matrix, window_size=size, step_size=size,
            statistics=['pi', 'fst', 'dxy'],
            populations=['pop1', 'pop2'])

        assert 'pi' in result.columns
        assert 'fst' in result.columns
        assert 'dxy' in result.columns
