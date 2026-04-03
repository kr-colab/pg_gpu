"""
Tests for pg_gpu.moments_ld integration layer.

Validates that pg_gpu produces the same LD and heterozygosity statistics
as moments for a two-population IM model dataset.

Requires the 'moments' pixi environment: pixi run -e moments pytest tests/test_moments_ld.py
"""

import os
import tempfile

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
)
from pg_gpu.haplotype_matrix import HaplotypeMatrix, _ld_names, _het_names


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
        assert ld_names == _ld_names(2)
        assert het_names == _het_names(2)

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


# ---------------------------------------------------------------------------
# Multi-population integration tests (3-pop, 4-pop)
# ---------------------------------------------------------------------------

def _simulate_multipop_vcf(n_pops, n_samples=8, seq_len=30_000, seed=42):
    """Simulate a multi-population VCF and pop file using msprime."""
    import msprime

    demography = msprime.Demography()
    for i in range(n_pops):
        demography.add_population(name=f"pop{i}", initial_size=1000)
    # Chain of splits: pop{n-1} splits from pop{n-2} at time 500*(n-1-i)
    if n_pops >= 2:
        demography.add_population(name="anc01", initial_size=2000)
        demography.add_population_split(
            time=500, derived=["pop0", "pop1"], ancestral="anc01")
    if n_pops >= 3:
        demography.add_population(name="anc012", initial_size=2000)
        demography.add_population_split(
            time=1000, derived=["anc01", "pop2"], ancestral="anc012")
    if n_pops >= 4:
        demography.add_population(name="anc0123", initial_size=2000)
        demography.add_population_split(
            time=1500, derived=["anc012", "pop3"], ancestral="anc0123")

    samples = {}
    for i in range(n_pops):
        samples[f"pop{i}"] = n_samples

    ts = msprime.sim_ancestry(
        samples=samples, demography=demography,
        sequence_length=seq_len, recombination_rate=1e-8,
        random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=seed)

    vcf_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.vcf', delete=False)
    pop_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False)

    # Write VCF
    with open(vcf_file.name, 'w') as f:
        ts.write_vcf(f)

    # Write pop file
    pops_list = [f"pop{i}" for i in range(n_pops)]
    with open(pop_file.name, 'w') as f:
        f.write("sample\tpop\n")
        for ind in ts.individuals():
            pop_name = ts.population(ind.population).metadata.get(
                'name', f"pop{ind.population}")
            f.write(f"tsk_{ind.id}\t{pop_name}\n")

    return vcf_file.name, pop_file.name, pops_list


@pytest.fixture(scope="module")
def three_pop_data():
    """Simulate 3-population data and compute both moments and pg_gpu stats."""
    vcf, pop_file, pops = _simulate_multipop_vcf(3)
    bp_bins = np.array([0, 1000, 5000, 15000, 30000], dtype=np.float64)
    try:
        m_stats = moments.LD.Parsing.compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, use_genotypes=False, report=False)
        g_stats = compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, report=False)
        yield m_stats, g_stats, pops
    finally:
        os.unlink(vcf)
        os.unlink(pop_file)


@pytest.fixture(scope="module")
def four_pop_data():
    """Simulate 4-population data and compute both moments and pg_gpu stats."""
    vcf, pop_file, pops = _simulate_multipop_vcf(4)
    bp_bins = np.array([0, 1000, 5000, 15000, 30000], dtype=np.float64)
    try:
        m_stats = moments.LD.Parsing.compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, use_genotypes=False, report=False)
        g_stats = compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, report=False)
        yield m_stats, g_stats, pops
    finally:
        os.unlink(vcf)
        os.unlink(pop_file)


class TestThreePopLD:
    """Verify 3-population LD statistics match moments."""

    def test_output_format(self, three_pop_data):
        _, g, pops = three_pop_data
        assert g['pops'] == pops
        ld_names, het_names = g['stats']
        assert len(ld_names) == 45
        assert len(het_names) == 6
        assert ld_names == _ld_names(3)
        assert het_names == _het_names(3)

    def test_ld_sums_match(self, three_pop_data):
        m, g, _ = three_pop_data
        for i in range(len(m['bins'])):
            np.testing.assert_allclose(
                g['sums'][i], m['sums'][i], rtol=1e-6,
                err_msg=f"3-pop LD sums mismatch in bin {i}")

    def test_het_sums_match(self, three_pop_data):
        m, g, _ = three_pop_data
        np.testing.assert_allclose(
            g['sums'][-1], m['sums'][-1], rtol=1e-6,
            err_msg="3-pop heterozygosity mismatch")

    def test_moments_compatibility(self, three_pop_data):
        _, g, _ = three_pop_data
        means = moments.LD.Parsing.means_from_region_data(
            {0: g}, g['stats'])
        assert len(means) == len(g['bins']) + 1
        for m in means:
            assert np.all(np.isfinite(m))


class TestFourPopLD:
    """Verify 4-population LD statistics match moments."""

    def test_output_format(self, four_pop_data):
        _, g, pops = four_pop_data
        assert g['pops'] == pops
        ld_names, het_names = g['stats']
        assert len(ld_names) == 105
        assert len(het_names) == 10
        assert ld_names == _ld_names(4)
        assert het_names == _het_names(4)

    def test_ld_sums_match(self, four_pop_data):
        m, g, _ = four_pop_data
        for i in range(len(m['bins'])):
            np.testing.assert_allclose(
                g['sums'][i], m['sums'][i], rtol=1e-6,
                err_msg=f"4-pop LD sums mismatch in bin {i}")

    def test_het_sums_match(self, four_pop_data):
        m, g, _ = four_pop_data
        np.testing.assert_allclose(
            g['sums'][-1], m['sums'][-1], rtol=1e-6,
            err_msg="4-pop heterozygosity mismatch")

    def test_moments_compatibility(self, four_pop_data):
        _, g, _ = four_pop_data
        means = moments.LD.Parsing.means_from_region_data(
            {0: g}, g['stats'])
        assert len(means) == len(g['bins']) + 1
        for m in means:
            assert np.all(np.isfinite(m))
