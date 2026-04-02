"""
Tests for pg_gpu.moments_ld integration layer.

Validates that pg_gpu produces the same LD and heterozygosity statistics
as moments for a two-population IM model dataset.

Requires the 'moments' pixi environment: pixi run -e moments pytest tests/test_moments_ld.py
"""

import pytest
import numpy as np

# Skip entire module if moments LD is not available
try:
    import moments.LD
except (ImportError, AttributeError):
    pytest.skip("moments.LD not available (use pixi -e moments)",
                allow_module_level=True)

from pg_gpu.moments_ld import (
    compute_ld_statistics,
    _compute_heterozygosity,
    _interpolate_genetic_distances,
    _LD_NAMES,
    _HET_NAMES,
)
from pg_gpu.haplotype_matrix import HaplotypeMatrix


VCF = "examples/data/im-parsing-example.vcf"
POP_FILE = "examples/data/im_pop.txt"
POPS = ["deme0", "deme1"]
BP_BINS = np.logspace(2, 6, 6)


@pytest.fixture(scope="module")
def moments_stats():
    """Compute moments reference stats once for the module."""
    return moments.LD.Parsing.compute_ld_statistics(
        VCF, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=False, report=False,
    )


@pytest.fixture(scope="module")
def gpu_stats():
    """Compute pg_gpu stats once for the module."""
    return compute_ld_statistics(
        VCF, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, report=False,
    )


class TestOutputFormat:
    """Verify the output dict has the correct structure."""

    def test_keys(self, gpu_stats):
        assert set(gpu_stats.keys()) == {'bins', 'sums', 'stats', 'pops'}

    def test_pops(self, gpu_stats):
        assert gpu_stats['pops'] == POPS

    def test_stats_names(self, gpu_stats):
        ld_names, het_names = gpu_stats['stats']
        assert ld_names == _LD_NAMES
        assert het_names == _HET_NAMES

    def test_bins_count(self, gpu_stats):
        assert len(gpu_stats['bins']) == len(BP_BINS) - 1

    def test_sums_count(self, gpu_stats):
        # One array per LD bin + one for heterozygosity
        assert len(gpu_stats['sums']) == len(BP_BINS) - 1 + 1

    def test_ld_sums_shape(self, gpu_stats):
        for i in range(len(BP_BINS) - 1):
            assert gpu_stats['sums'][i].shape == (15,)

    def test_het_sums_shape(self, gpu_stats):
        assert gpu_stats['sums'][-1].shape == (3,)


class TestLDStatistics:
    """Verify LD statistics match moments at machine precision."""

    def test_ld_bins_match(self, moments_stats, gpu_stats):
        for m_bin, g_bin in zip(moments_stats['bins'], gpu_stats['bins']):
            assert np.isclose(m_bin[0], g_bin[0])
            assert np.isclose(m_bin[1], g_bin[1])

    def test_ld_sums_match(self, moments_stats, gpu_stats):
        for i in range(len(moments_stats['bins'])):
            m = moments_stats['sums'][i]
            g = gpu_stats['sums'][i]
            np.testing.assert_allclose(g, m, rtol=1e-6,
                err_msg=f"LD sums mismatch in bin {i}")

    def test_het_sums_match(self, moments_stats, gpu_stats):
        m = moments_stats['sums'][-1]
        g = gpu_stats['sums'][-1]
        np.testing.assert_allclose(g, m, rtol=1e-6,
            err_msg="Heterozygosity sums mismatch")


class TestHeterozygosity:
    """Verify heterozygosity computation independently."""

    def test_within_pop_positive(self, gpu_stats):
        het = gpu_stats['sums'][-1]
        assert het[0] > 0  # H_0_0
        assert het[2] > 0  # H_1_1

    def test_cross_pop_positive(self, gpu_stats):
        het = gpu_stats['sums'][-1]
        assert het[1] > 0  # H_0_1

    def test_cross_between_within(self, gpu_stats):
        """Cross-pop het should be between within-pop values for diverged pops."""
        H_0_0, H_0_1, H_1_1 = gpu_stats['sums'][-1]
        assert H_0_1 >= min(H_0_0, H_1_1)


class TestMomentsCompatibility:
    """Verify output can be fed into moments downstream functions."""

    def test_means_from_region_data(self, gpu_stats):
        """moments.LD.Parsing.means_from_region_data should accept our output."""
        all_data = {0: gpu_stats}
        means = moments.LD.Parsing.means_from_region_data(
            all_data, gpu_stats['stats'])
        assert len(means) == len(gpu_stats['bins']) + 1
        for m in means:
            assert isinstance(m, np.ndarray)
            assert np.all(np.isfinite(m))

    def test_means_match_moments(self, moments_stats, gpu_stats):
        """Normalized means should match between moments and pg_gpu."""
        means_m = moments.LD.Parsing.means_from_region_data(
            {0: moments_stats}, moments_stats['stats'])
        means_g = moments.LD.Parsing.means_from_region_data(
            {0: gpu_stats}, gpu_stats['stats'])
        for mm, mg in zip(means_m, means_g):
            np.testing.assert_allclose(mg, mm, rtol=1e-6)
