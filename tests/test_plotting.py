"""
Smoke tests for plotting functions.

Verifies each plot function runs without error and returns axes/figures.
Uses matplotlib Agg backend (no display needed).
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pg_gpu import HaplotypeMatrix, sfs, plotting
from pg_gpu.windowed_analysis import windowed_statistics


@pytest.fixture(autouse=True)
def close_plots():
    """Close all plots after each test."""
    yield
    plt.close('all')


@pytest.fixture
def sim_data():
    np.random.seed(42)
    n_hap, n_var = 20, 100
    hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)
    return matrix, hap, pos


class TestSFSPlots:

    def test_plot_sfs(self, sim_data):
        matrix, _, _ = sim_data
        s = sfs.sfs(matrix)
        ax = plotting.plot_sfs(s)
        assert ax is not None

    def test_plot_sfs_folded(self, sim_data):
        matrix, _, _ = sim_data
        s = sfs.sfs_folded(matrix)
        ax = plotting.plot_sfs(s, folded=True)
        assert ax is not None

    def test_plot_joint_sfs(self, sim_data):
        matrix, _, _ = sim_data
        matrix.sample_sets = {'p1': list(range(10)), 'p2': list(range(10, 20))}
        j = sfs.joint_sfs(matrix, 'p1', 'p2')
        ax = plotting.plot_joint_sfs(j)
        assert ax is not None


class TestLDPlots:

    def test_plot_pairwise_ld(self, sim_data):
        matrix, _, pos = sim_data
        r2 = matrix.pairwise_r2()
        if hasattr(r2, 'get'):
            r2 = r2.get()
        ax = plotting.plot_pairwise_ld(r2, positions=pos)
        assert ax is not None

    def test_plot_ld_decay(self):
        np.random.seed(42)
        distances = np.random.uniform(0, 50000, 1000)
        r2 = np.random.uniform(0, 1, 1000) * np.exp(-distances / 10000)
        ax = plotting.plot_ld_decay(distances, r2)
        assert ax is not None


class TestPCAPlots:

    def test_plot_pca_no_labels(self, sim_data):
        from pg_gpu import decomposition
        matrix, _, _ = sim_data
        coords, ev = decomposition.pca(matrix, n_components=3)
        ax = plotting.plot_pca(coords, explained_variance=ev)
        assert ax is not None

    def test_plot_pca_with_labels(self, sim_data):
        from pg_gpu import decomposition
        matrix, _, _ = sim_data
        coords, ev = decomposition.pca(matrix, n_components=3)
        labels = np.array(['A'] * 10 + ['B'] * 10)
        ax = plotting.plot_pca(coords, explained_variance=ev, labels=labels)
        assert ax is not None


class TestDistancePlots:

    def test_plot_pairwise_distance(self, sim_data):
        from pg_gpu import decomposition
        matrix, _, _ = sim_data
        dist = decomposition.pairwise_distance(matrix)
        ax = plotting.plot_pairwise_distance(dist)
        assert ax is not None


class TestWindowedPlots:

    def test_plot_windowed(self, sim_data):
        matrix, _, _ = sim_data
        bp_bins = np.arange(0, 100001, 10000)
        result = windowed_statistics(matrix, bp_bins, statistics=('pi',))
        ax = plotting.plot_windowed(result['window_start'], result['pi'],
                                    stat_name='pi')
        assert ax is not None

    def test_plot_windowed_panel(self, sim_data):
        matrix, _, _ = sim_data
        bp_bins = np.arange(0, 100001, 20000)
        result = windowed_statistics(
            matrix, bp_bins,
            statistics=('pi', 'theta_w', 'segregating_sites'))
        fig, axes = plotting.plot_windowed_panel(result)
        assert fig is not None
        assert len(axes) == 3


class TestHaplotypePlots:

    def test_plot_haplotype_frequencies(self):
        freqs = np.array([0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05,
                          0.025, 0.025, 0.025, 0.025])
        ax = plotting.plot_haplotype_frequencies(freqs)
        assert ax is not None

    def test_plot_variant_locator(self, sim_data):
        _, _, pos = sim_data
        ax = plotting.plot_variant_locator(pos)
        assert ax is not None
