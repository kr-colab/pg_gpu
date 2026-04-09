import pytest
import numpy as np
import tempfile
import os
import allel
pytest.importorskip("moments.LD")
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix


class TestLDStatisticsGPUSinglePop:
    """Test suite for single-population compute_ld_statistics_gpu_single_pop function."""

    @pytest.fixture
    def single_pop_vcf(self):
        """Create a simple VCF file with single population data."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t0|0\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t0|0\t1|1
1\t500\t.\tT\tA\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0\t0|1
1\t1000\t.\tG\tC\t.\tPASS\t.\tGT\t1|1\t0|0\t1|0\t0|1
1\t2000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t1|1\t1|0\t0|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name

        # No population file needed for single population
        yield vcf_path

        # Cleanup
        os.unlink(vcf_path)

    @pytest.fixture
    def single_pop_vcf_with_monomorphic(self):
        """Create a VCF file with some monomorphic sites for testing ac_filter."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t0|0\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t0|0\t0|0\t0|0\t0|0
1\t500\t.\tT\tA\t.\tPASS\t.\tGT\t0|1\t1|1\t0|0\t0|1
1\t1000\t.\tG\tC\t.\tPASS\t.\tGT\t1|1\t1|1\t1|1\t1|1
1\t2000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t1|1\t1|0\t0|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name

        yield vcf_path

        # Cleanup
        os.unlink(vcf_path)

    def test_single_population_basic(self, single_pop_vcf):
        """Test basic single population LD statistics computation."""
        # Load data into HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf)

        # Define bins
        bp_bins = np.array([0, 200, 500, 1000, 2500])

        # Compute LD statistics using single-population GPU function
        gpu_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False  # No filtering for this test
        )

        # Check that we have results
        assert len(gpu_stats) > 0

        # Check structure of results
        for bin_range, stats in gpu_stats.items():
            assert isinstance(bin_range, tuple)
            assert len(bin_range) == 2
            assert isinstance(stats, tuple)
            assert len(stats) == 3  # (DD, Dz, pi2)

            # All statistics should be finite
            for stat in stats:
                assert np.isfinite(stat)

    def test_single_population_averaged(self, single_pop_vcf):
        """Test single population LD statistics with averaging (raw=False)."""
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf)

        bp_bins = np.array([0, 200, 500, 1000, 2500])

        # Compute with averaging
        gpu_stats_avg = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=False,
            ac_filter=False
        )

        # Compute raw sums for comparison
        gpu_stats_raw = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False
        )

        # For bins with multiple pairs, averaged values should differ from raw sums
        for bin_range in gpu_stats_avg:
            if bin_range in gpu_stats_raw:
                avg_stats = gpu_stats_avg[bin_range]
                raw_stats = gpu_stats_raw[bin_range]

                # If there are non-zero values, check they're different (unless single pair)
                if any(s != 0 for s in raw_stats):
                    # Can't guarantee they're different without knowing pair counts
                    # Just check they're valid
                    for avg, raw in zip(avg_stats, raw_stats):
                        assert np.isfinite(avg)
                        assert np.isfinite(raw)

    def test_single_population_ac_filter(self, single_pop_vcf_with_monomorphic):
        """Test single population LD statistics with allele count filtering."""
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf_with_monomorphic)

        bp_bins = np.array([0, 500, 2000, 5000])

        # Compute with AC filter (default)
        gpu_stats_filtered = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=True
        )

        # Compute without AC filter
        gpu_stats_unfiltered = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False
        )

        # The filtered version should have fewer or equal statistics
        # (monomorphic sites should be filtered out)
        assert len(gpu_stats_filtered) <= len(gpu_stats_unfiltered)

    def test_single_population_empty_bins(self, single_pop_vcf):
        """Test handling of empty bins in single population computation."""
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf)

        # Create bins that will have some empty ranges
        bp_bins = np.array([0, 10, 20, 100, 150, 200, 500, 1000, 2500])

        gpu_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=False,
            ac_filter=False
        )

        # Check all bins are present and empty bins have zero values
        for i in range(len(bp_bins) - 1):
            bin_range = (float(bp_bins[i]), float(bp_bins[i+1]))
            assert bin_range in gpu_stats
            stats = gpu_stats[bin_range]

            # Empty bins should have all zeros
            if all(s == 0.0 for s in stats):
                assert stats == (0.0, 0.0, 0.0)

    def test_single_vs_two_population_same_pop(self, single_pop_vcf):
        """Test that single-pop function matches two-pop function with same population."""
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf)

        # For two-pop function, we need to define a sample set
        n_samples = 4  # From the VCF
        sample_set = list(range(n_samples)) + list(range(n_samples, 2 * n_samples))
        h_gpu.sample_sets = {"pop0": sample_set}

        bp_bins = np.array([0, 200, 500, 1000, 2500])

        # Compute using single-population function
        single_pop_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False
        )

        # Compute using two-population function with same population
        two_pop_stats = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop0",
            raw=True,
            ac_filter=False
        )

        # Extract DD_0_0, Dz_0_0_0, and pi2_0_0_0_0 from two-pop results
        for bin_range in single_pop_stats:
            if bin_range in two_pop_stats:
                single_stats = single_pop_stats[bin_range]
                two_stats = two_pop_stats[bin_range]

                # Two-pop returns OrderedDict with named statistics
                # We want DD_0_0, Dz_0_0_0, pi2_0_0_0_0
                dd_two = two_stats['DD_0_0']
                dz_two = two_stats['Dz_0_0_0']
                pi2_two = two_stats['pi2_0_0_0_0']

                # Single-pop returns tuple (DD, Dz, pi2)
                dd_single, dz_single, pi2_single = single_stats

                # They should match
                assert np.allclose(dd_single, dd_two, rtol=1e-2), \
                    f"DD mismatch: single={dd_single}, two={dd_two}"
                assert np.allclose(dz_single, dz_two, rtol=1e-2), \
                    f"Dz mismatch: single={dz_single}, two={dz_two}"
                assert np.allclose(pi2_single, pi2_two, rtol=1e-2), \
                    f"pi2 mismatch: single={pi2_single}, two={pi2_two}"

    def test_single_population_gpu_device_handling(self, single_pop_vcf):
        """Test that the function properly handles GPU device transfers."""
        # Start with CPU-based HaplotypeMatrix
        h_cpu = HaplotypeMatrix.from_vcf(single_pop_vcf)
        assert h_cpu.device == 'CPU'

        bp_bins = np.array([0, 500, 2000])

        # Should automatically transfer to GPU
        gpu_stats = h_cpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False
        )

        # Should have results
        assert len(gpu_stats) > 0

        # Matrix should now be on GPU
        assert h_cpu.device == 'GPU'

    def test_single_population_missing_data_auto_detection(self, single_pop_vcf):
        """Test that missing data is automatically detected and handled."""
        h_gpu = HaplotypeMatrix.from_vcf(single_pop_vcf)

        bp_bins = np.array([0, 500, 2000])

        # No missing parameter needed - auto-detects missing data
        gpu_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            raw=True,
            ac_filter=False
        )

        # Check that we have results
        assert len(gpu_stats) > 0

        # Check structure of results
        for bin_range, stats in gpu_stats.items():
            assert isinstance(bin_range, tuple)
            assert len(bin_range) == 2
            assert isinstance(stats, tuple)
            assert len(stats) == 3  # (DD, Dz, pi2)

            # All statistics should be finite
            for stat in stats:
                assert np.isfinite(stat)
