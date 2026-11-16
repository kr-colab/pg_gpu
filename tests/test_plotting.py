"""
Smoke tests for plotting functions.

Verifies each plot function runs without error and returns axes/figures.
Uses matplotlib Agg backend (no display needed).
"""

import pytest
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pg_gpu import HaplotypeMatrix, sfs, plotting
from pg_gpu.windowed_analysis import windowed_analysis


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


    def test_joint_sfs_with_no_positive_cell_draws(self):
        rng = np.random.default_rng(0)
        hap = np.zeros((8, 5), np.int8)
        hap[4:] = rng.integers(0, 2, (4, 5))   # the first population is monomorphic
        hm = HaplotypeMatrix(hap, np.arange(5) * 100 + 1, 0, 1000)
        joint = sfs.joint_sfs(hm, [0, 1, 2, 3], [4, 5, 6, 7])
        assert joint.sum() > 0 and not joint[1:, 1:].any()
        ax = plotting.plot_joint_sfs(joint)
        # LogNorm fails at draw time, not when the artist is made.
        ax.figure.canvas.draw()

    @pytest.mark.parametrize("fill", [0.0, 4.0], ids=["only-nan", "nan-and-counts"])
    def test_joint_sfs_leaves_a_nan_cell_blank(self, fill):
        joint = np.full((4, 4), fill)
        joint[2, 2] = np.nan
        ax = plotting.plot_joint_sfs(joint)
        ax.figure.canvas.draw()
        assert ax.collections[0].get_array().mask[1, 1]


class TestLDPlots:

    def test_plot_pairwise_ld(self, sim_data):
        # pairwise_r2 returns a CuPy array, and the docstring names it as
        # the input.
        matrix, _, _ = sim_data
        r2 = matrix.pairwise_r2()
        assert isinstance(r2, cp.ndarray)
        ax = plotting.plot_pairwise_ld(r2, positions=cp.asarray(matrix.positions))
        np.testing.assert_allclose(ax.images[0].get_array(), cp.asnumpy(r2), equal_nan=True)

    def test_plot_ld_decay(self):
        np.random.seed(42)
        distances = np.random.uniform(0, 50000, 1000)
        r2 = np.random.uniform(0, 1, 1000) * np.exp(-distances / 10000)
        ax = plotting.plot_ld_decay(distances, r2)
        assert ax is not None

    @pytest.mark.parametrize("bins", [5, np.int64(5)], ids=["int", "numpy-int"])
    def test_ld_decay_counts_the_pair_at_the_largest_distance(self, bins):
        distances = np.array([0., 10, 20, 30, 100])
        r2 = np.array([.9, .7, .5, .4, .05])
        centers, values = plotting.plot_ld_decay(distances, r2, bins=bins).lines[0].get_data()
        np.testing.assert_allclose(centers, [10, 30, 50, 70, 90])
        np.testing.assert_allclose(values, [.8, .45, np.nan, np.nan, .05], equal_nan=True)

    DISTANCES = np.array([5., 15, 35, 45, 65, 95, 105])
    R2 = np.array([.95, .85, .6, .5, .3, .2, .02])

    def test_ld_decay_bins_hold_the_pairs_np_histogram_counts(self):
        values = plotting.plot_ld_decay(self.DISTANCES, self.R2, bins=4).lines[0].get_data()[1]
        assert np.histogram(self.DISTANCES, 4)[0].tolist() == [2, 2, 1, 2]
        np.testing.assert_allclose(values, [.9, .55, .3, .11])

    def test_ld_decay_explicit_edges_drop_pairs_past_the_last_edge(self):
        to_105, to_100 = (
            plotting.plot_ld_decay(self.DISTANCES, self.R2, bins=edges).lines[0].get_data()[1]
            for edges in ([0, 50, 105], [0, 50, 100]))
        np.testing.assert_allclose(to_105, [.725, .2])
        np.testing.assert_allclose(to_100, [.725, .25])

    def test_ld_decay_drops_a_pair_with_a_nan_distance(self):
        distances = np.append(self.DISTANCES, np.nan)
        r2 = np.append(self.R2, .5)
        values = plotting.plot_ld_decay(distances, r2, bins=4).lines[0].get_data()[1]
        np.testing.assert_allclose(values, [.9, .55, .3, .11])

    @pytest.mark.parametrize("empty", [False, True], ids=["all-nan", "empty"])
    def test_ld_decay_with_no_defined_r2_raises(self, empty):
        # The r2 of monomorphic sites is NaN for every pair.
        mono = HaplotypeMatrix(np.zeros((10, 4), np.int8), np.array([10, 20, 30, 40]), 0, 100)
        r2 = cp.asnumpy(mono.pairwise_r2())[np.triu_indices(4, k=1)]
        distances = np.array([10, 20, 30, 10, 20, 10])
        assert np.isnan(r2).all()
        if empty:
            distances, r2 = distances[:0], r2[:0]
        with pytest.raises(ValueError, match="no pair with a defined r-squared"):
            plotting.plot_ld_decay(distances, r2)


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
        result = windowed_analysis(matrix, window_size=10000, step_size=10000,
                                   statistics=['pi'])
        ax = plotting.plot_windowed(result['start'], result['pi'],
                                    stat_name='pi')
        assert ax is not None

    def test_plot_windowed_panel(self, sim_data):
        matrix, _, _ = sim_data
        result = windowed_analysis(
            matrix, window_size=20000, step_size=20000,
            statistics=['pi', 'theta_w', 'segregating_sites'])
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


class TestPlotEdgeInputs:
    """Input shapes and ranges that select the less-common drawing branches."""

    def test_sfs_without_endpoint_clipping(self):
        s = np.arange(1, 22, dtype=float)
        assert plotting.plot_sfs(s, clip_endpoints=False) is not None

    def test_pca_without_explained_variance(self):
        coords = np.random.RandomState(0).randn(20, 3)
        assert plotting.plot_pca(coords) is not None

    def test_pairwise_distance_square_matrix_with_labels(self):
        a = np.random.RandomState(0).rand(6, 6)
        dist = (a + a.T) / 2
        np.fill_diagonal(dist, 0.0)
        ax = plotting.plot_pairwise_distance(
            dist, labels=[f"s{i}" for i in range(6)])
        assert ax is not None

    def test_windowed_axis_in_mb(self):
        starts = np.arange(0, 5_000_000, 500_000)  # largest start 4.5 Mb
        ax = plotting.plot_windowed(starts, np.random.rand(len(starts)))
        assert ax.get_xlabel() == "Position (Mb)"

    def test_windowed_axis_in_bp(self):
        starts = np.arange(0, 1000, 100)  # largest start 900 bp
        ax = plotting.plot_windowed(starts, np.random.rand(len(starts)))
        assert ax.get_xlabel() == "Position (bp)"

    def test_windowed_panel_single_statistic(self, sim_data):
        matrix, _, _ = sim_data
        result = windowed_analysis(matrix, window_size=20000, step_size=20000,
                                   statistics=['pi'])
        _, axes = plotting.plot_windowed_panel(result)
        assert len(axes) == 1

    def test_variant_locator_with_step(self, sim_data):
        _, _, pos = sim_data
        assert plotting.plot_variant_locator(pos, step=2) is not None


# Each plotting function with its array arguments as CuPy arrays, the form in
# which the statistics return them.
CUPY_CALLS = {
    "plot_sfs": lambda: plotting.plot_sfs(cp.arange(9.0)),
    "plot_joint_sfs": lambda: plotting.plot_joint_sfs(cp.arange(1, 26).reshape(5, 5)),
    "plot_ld_decay": lambda: plotting.plot_ld_decay(
        cp.linspace(0, 100, 30), cp.linspace(1, 0, 30), bins=cp.linspace(0, 100, 6)),
    "plot_pca": lambda: plotting.plot_pca(
        cp.arange(30.0).reshape(10, 3) % 7, explained_variance=cp.asarray([.5, .3, .2])),
    "plot_pairwise_distance": lambda: plotting.plot_pairwise_distance(cp.linspace(.1, 1, 10)),
    "plot_windowed": lambda: plotting.plot_windowed(cp.arange(5) * 1000, cp.linspace(.1, .5, 5)),
    "plot_haplotype_frequencies": lambda: plotting.plot_haplotype_frequencies(
        cp.asarray([.5, .3, .2])),
    "plot_variant_locator": lambda: plotting.plot_variant_locator(cp.arange(50) * 10),
}


@pytest.mark.parametrize("name", CUPY_CALLS)
def test_plot_functions_take_cupy_arrays(name):
    ax = CUPY_CALLS[name]()
    ax.figure.canvas.draw()
