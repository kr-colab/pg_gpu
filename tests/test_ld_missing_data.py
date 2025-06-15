"""
Tests for LD statistics computation with missing data.

This module tests the handling of missing data in LD statistics calculations,
comparing pg_gpu implementations against moments reference implementations.
"""

import pytest
import numpy as np
import tempfile
import os
import allel
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix
from collections import OrderedDict


class TestLDMissingData:
    """Test suite for LD statistics with missing data."""
    
    @pytest.fixture
    def vcf_with_missing_data(self):
        """Create a VCF file with missing genotypes."""
        # Note: VCF uses . for missing genotypes, which gets converted to -1 in parsing
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t.|.\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t0|0\t.|.
1\t500\t.\tT\tA\t.\tPASS\t.\tGT\t.|1\t1|1\t0|0\t0|1
1\t1000\t.\tG\tC\t.\tPASS\t.\tGT\t1|1\t.|0\t1|0\t0|1
1\t2000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t1|1\t1|.\t0|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        yield vcf_path
        
        # Cleanup
        os.unlink(vcf_path)
    
    @pytest.fixture
    def vcf_with_missing_data_two_pops(self):
        """Create a VCF file with missing genotypes for two populations."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\tind2\tind3
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t.|.\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t.|1\t0|0\t1|1
1\t500\t.\tT\tA\t.\tPASS\t.\tGT\t0|1\t1|1\t.|0\t1|0
1\t1000\t.\tG\tC\t.\tPASS\t.\tGT\t1|.\t0|0\t1|0\t0|.
1\t2000\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t1|1\t1|0\t.|1
"""
        
        pop_content = """sample\tpop
ind0\tpop0
ind1\tpop0
ind2\tpop1
ind3\tpop1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(pop_content)
            pop_path = f.name
        
        yield vcf_path, pop_path
        
        # Cleanup
        os.unlink(vcf_path)
        os.unlink(pop_path)
    
    @pytest.fixture
    def vcf_extreme_missing(self):
        """Create a VCF with extreme missing data patterns."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t.|.\t.|.\t.|.\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t0|0\t1|1
1\t500\t.\tT\tA\t.\tPASS\t.\tGT\t.|.\t.|.\t.|.\t.|.
1\t1000\t.\tG\tC\t.\tPASS\t.\tGT\t1|1\t0|0\t1|0\t0|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        yield vcf_path
        
        # Cleanup
        os.unlink(vcf_path)
    
    def test_load_missing_data_vcf(self, vcf_with_missing_data):
        """Test that we can load VCF files with missing data."""
        # Load with allel to check how missing data is represented
        vcf_data = allel.read_vcf(vcf_with_missing_data)
        genotypes = vcf_data['calldata/GT']
        
        # Check that missing genotypes are represented as -1
        assert np.any(genotypes == -1), "Missing genotypes should be represented as -1"
        
        # Count missing genotypes
        n_missing = np.sum(genotypes == -1)
        assert n_missing > 0, "Should have some missing genotypes"
        
        # Verify specific missing patterns from our test data
        # sample3 at position 100 should be missing (both alleles)
        assert genotypes[0, 2, 0] == -1 and genotypes[0, 2, 1] == -1
        # sample4 at position 200 should be missing
        assert genotypes[1, 3, 0] == -1 and genotypes[1, 3, 1] == -1
    
    def test_moments_missing_data_single_pop(self, vcf_with_missing_data):
        """Test that moments can compute LD statistics with missing data."""
        bp_bins = np.array([0, 500, 2000, 5000])
        
        # Compute with moments (it should handle missing data)
        moments_stats = mParsing.compute_ld_statistics(
            vcf_with_missing_data,
            pop_file=None,  # Single population
            pops=None,
            bp_bins=bp_bins,
            use_genotypes=False,  # Use haplotypes
            report=False
        )
        
        # Check that we got results
        assert moments_stats is not None
        assert 'sums' in moments_stats
        assert 'stats' in moments_stats
        # moments returns bins including the heterozygosity bin
        assert len(moments_stats['sums']) == len(bp_bins)
        
        # Check that statistics are computed (not all NaN)
        # Skip the last bin which is heterozygosity
        for bin_sums in moments_stats['sums'][:-1]:
            assert not all(np.isnan(bin_sums))
    
    def test_moments_missing_data_two_pops(self, vcf_with_missing_data_two_pops):
        """Test moments computation with missing data for two populations."""
        vcf_path, pop_path = vcf_with_missing_data_two_pops
        bp_bins = np.array([0, 500, 2000, 5000])
        
        # Compute with moments
        moments_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_path,
            pops=["pop0", "pop1"],
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Verify we got all expected statistics
        assert moments_stats is not None
        assert len(moments_stats['stats'][0]) == 15  # All two-pop statistics
        
        # Check that we have non-zero statistics
        for bin_sums in moments_stats['sums']:
            assert any(s != 0 for s in bin_sums)
    
    def test_pg_gpu_missing_data_single_pop(self, vcf_with_missing_data):
        """Test pg_gpu single population LD with missing data."""
        h_gpu = HaplotypeMatrix.from_vcf(vcf_with_missing_data)
        bp_bins = np.array([0, 500, 2000, 5000])
        
        # This should work once missing data is implemented
        gpu_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            missing=True,  # Enable missing data handling
            raw=True,
            ac_filter=False
        )
        
        # Check basic structure
        assert len(gpu_stats) > 0
        for bin_range, stats in gpu_stats.items():
            assert len(stats) == 3  # DD, Dz, pi2
    
    def test_pg_gpu_missing_data_two_pops(self, vcf_with_missing_data_two_pops):
        """Test pg_gpu two population LD with missing data."""
        vcf_path, pop_path = vcf_with_missing_data_two_pops
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
        
        # Set up population structure
        vcf_data = allel.read_vcf(vcf_path)
        n_samples = vcf_data['samples'].shape[0]
        
        # Read population assignments
        pop_assignments = {}
        with open(pop_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop
        
        # Create sample sets
        pop_sets = {"pop0": [], "pop1": []}
        for i, sample_name in enumerate(vcf_data['samples']):
            pop = pop_assignments.get(sample_name)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)  # Add haplotype indices
        
        h_gpu.sample_sets = pop_sets
        
        bp_bins = np.array([0, 500, 2000, 5000])
        
        # This should work once missing data is implemented
        # Note: ac_filter=False is required when using missing=True
        gpu_stats = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
            missing=True,
            raw=True,
            ac_filter=False
        )
        
        # Check we have all statistics
        for bin_range, stats in gpu_stats.items():
            assert len(stats) == 15
    
    def test_missing_data_correspondence(self, vcf_with_missing_data):
        """Test that pg_gpu matches moments for missing data."""
        bp_bins = np.array([0, 500, 2000, 5000])
        
        # Compute with moments
        moments_stats = mParsing.compute_ld_statistics(
            vcf_with_missing_data,
            pop_file=None,
            pops=None,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compute with pg_gpu
        h_gpu = HaplotypeMatrix.from_vcf(vcf_with_missing_data)
        gpu_stats = h_gpu.compute_ld_statistics_gpu_single_pop(
            bp_bins=bp_bins,
            missing=True,
            raw=True,
            ac_filter=False
        )
        
        # Compare results
        stat_names = moments_stats['stats'][0]
        for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_stats['bins'], moments_stats['sums'])):
            gpu_bin = gpu_stats[(float(bin_range[0]), float(bin_range[1]))]
            
            # Single pop: check DD, Dz, pi2
            dd_idx = stat_names.index('DD_0_0')
            dz_idx = stat_names.index('Dz_0_0_0')
            pi2_idx = stat_names.index('pi2_0_0_0_0')
            
            gpu_dd, gpu_dz, gpu_pi2 = gpu_bin
            
            assert np.allclose(gpu_dd, moments_sums[dd_idx], rtol=1e-10)
            assert np.allclose(gpu_dz, moments_sums[dz_idx], rtol=1e-10)
            assert np.allclose(gpu_pi2, moments_sums[pi2_idx], rtol=1e-10)
    
    def test_extreme_missing_data_patterns(self, vcf_extreme_missing):
        """Test handling of extreme missing data cases."""
        # Test with moments to establish expected behavior
        bp_bins = np.array([0, 500, 2000, 5000])
        
        moments_stats = mParsing.compute_ld_statistics(
            vcf_extreme_missing,
            pop_file=None,
            pops=None,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Check that moments handles the extreme cases
        # Variant at position 100 has only one non-missing sample
        # Variant at position 500 is entirely missing
        assert moments_stats is not None
        
        # Some bins might have no computable statistics due to missing data
        for bin_sums in moments_stats['sums']:
            # Check if any statistics could be computed
            if not all(s == 0 for s in bin_sums):
                assert any(not np.isnan(s) for s in bin_sums)
    
    def test_missing_data_affects_counts(self, vcf_with_missing_data):
        """Test that missing data correctly affects haplotype counts."""
        # Load the VCF and examine the genotype matrix
        vcf_data = allel.read_vcf(vcf_with_missing_data)
        genotypes = vcf_data['calldata/GT']
        
        # Convert to haplotype matrix (phased data)
        haplotypes = genotypes.reshape(genotypes.shape[0], -1)
        
        # Count valid (non-missing) haplotypes per variant
        valid_counts = []
        for i in range(haplotypes.shape[0]):
            n_valid = np.sum(haplotypes[i] != -1)
            valid_counts.append(n_valid)
        
        # Verify that different variants have different numbers of valid haplotypes
        assert len(set(valid_counts)) > 1, "Should have varying numbers of valid haplotypes"
        
        # The effective sample size for LD calculations should vary by variant pair
        # depending on how many samples have valid data for both variants