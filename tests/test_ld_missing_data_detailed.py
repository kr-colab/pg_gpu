"""
Detailed tests to understand how moments handles missing data in LD calculations.
"""

import pytest
import numpy as np
import tempfile
import os
import allel
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix


class TestLDMissingDataDetailed:
    """Detailed tests to understand missing data behavior."""
    
    @pytest.fixture
    def simple_missing_vcf(self):
        """Create a simple VCF with known missing patterns."""
        # Two variants, 4 samples (8 haplotypes)
        # This allows us to manually calculate expected values
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t0|0\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t.|.\t1|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        yield vcf_path
        os.unlink(vcf_path)
    
    @pytest.fixture
    def no_missing_vcf(self):
        """Create the same VCF without missing data for comparison."""
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1,length=10000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\tsample3\tsample4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0\t0|0\t1|1
1\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1|0\t0|1\t0|0\t1|1
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.vcf') as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        yield vcf_path
        os.unlink(vcf_path)
    
    def test_manual_haplotype_counts(self, simple_missing_vcf, no_missing_vcf):
        """Manually compute haplotype counts to understand missing data handling."""
        # First, load both VCFs
        vcf_missing = allel.read_vcf(simple_missing_vcf)
        vcf_no_missing = allel.read_vcf(no_missing_vcf)
        
        # Get haplotype matrices
        gt_missing = vcf_missing['calldata/GT']
        gt_no_missing = vcf_no_missing['calldata/GT']
        
        # Reshape to haplotype format
        hap_missing = gt_missing.reshape(gt_missing.shape[0], -1)
        hap_no_missing = gt_no_missing.reshape(gt_no_missing.shape[0], -1)
        
        print("\nHaplotype matrix WITH missing data:")
        print(hap_missing)
        print("\nHaplotype matrix WITHOUT missing data:")
        print(hap_no_missing)
        
        # For the pair of variants (100, 200), count haplotypes manually
        # Without missing: all 8 haplotypes
        # With missing: only 6 haplotypes (sample3's are missing)
        
        # Count valid haplotypes for variant pair
        valid_mask = (hap_missing[0] != -1) & (hap_missing[1] != -1)
        n_valid = np.sum(valid_mask)
        print(f"\nNumber of valid haplotypes for pair: {n_valid}")
        
        # Expected: 6 (8 total - 2 missing from sample3)
        assert n_valid == 6
    
    def test_moments_counts_with_missing(self, simple_missing_vcf):
        """Test how moments computes haplotype counts with missing data."""
        bp_bins = np.array([0, 1000])
        
        # Use moments internal functions to get counts
        from moments.LD import Parsing
        
        # Load genotypes using allel instead
        import allel
        vcf_data = allel.read_vcf(simple_missing_vcf)
        genotypes = vcf_data['calldata/GT']
        positions = vcf_data['variants/POS']
        
        # Convert to haplotypes
        haplotypes = genotypes[:, :, 0].T  # First haplotype of each individual
        haplotypes2 = genotypes[:, :, 1].T  # Second haplotype
        
        # Stack haplotypes
        all_haps = np.vstack([haplotypes, haplotypes2])
        
        print("\nAll haplotypes shape:", all_haps.shape)
        print("Haplotypes:\n", all_haps)
        
        # Count missing
        n_missing_var1 = np.sum(all_haps[:, 0] == -1)
        n_missing_var2 = np.sum(all_haps[:, 1] == -1)
        print(f"\nMissing in variant 1: {n_missing_var1}")
        print(f"Missing in variant 2: {n_missing_var2}")
    
    def test_compare_missing_vs_no_missing(self, simple_missing_vcf, no_missing_vcf):
        """Compare LD statistics with and without missing data."""
        bp_bins = np.array([0, 1000])
        
        # Compute with missing data
        stats_missing = mParsing.compute_ld_statistics(
            simple_missing_vcf,
            pop_file=None,
            pops=None,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compute without missing data  
        stats_no_missing = mParsing.compute_ld_statistics(
            no_missing_vcf,
            pop_file=None,
            pops=None,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        print("\nStats WITH missing data:")
        print("Stats names:", stats_missing['stats'][0])
        print("Sums:", stats_missing['sums'][0])
        
        print("\nStats WITHOUT missing data:")
        print("Sums:", stats_no_missing['sums'][0])
        
        # The statistics should be different due to reduced sample size
        # with missing data
        assert not np.allclose(stats_missing['sums'][0], stats_no_missing['sums'][0])
    
    def test_missing_data_haplotype_counting(self):
        """Test the specific haplotype counting logic with missing data."""
        # Create a simple haplotype matrix with missing data
        # 2 variants, 6 haplotypes
        # Variant 1: [0, 1, 0, 1, 1, -1]
        # Variant 2: [1, 0, -1, 1, 1, 0]
        
        hap_matrix = np.array([
            [0, 1, 0, 1, 1, -1],
            [1, 0, -1, 1, 1, 0]
        ])
        
        # Count valid haplotype combinations
        # Only count pairs where both variants are non-missing
        valid_mask = (hap_matrix[0] != -1) & (hap_matrix[1] != -1)
        valid_haps = hap_matrix[:, valid_mask]
        
        print("\nValid haplotypes:")
        print(valid_haps)
        
        # Count haplotype combinations
        n11 = np.sum((valid_haps[0] == 1) & (valid_haps[1] == 1))
        n10 = np.sum((valid_haps[0] == 1) & (valid_haps[1] == 0))
        n01 = np.sum((valid_haps[0] == 0) & (valid_haps[1] == 1))
        n00 = np.sum((valid_haps[0] == 0) & (valid_haps[1] == 0))
        
        print(f"\nHaplotype counts:")
        print(f"n11={n11}, n10={n10}, n01={n01}, n00={n00}")
        print(f"Total valid: {n11 + n10 + n01 + n00}")
        
        # Check that we correctly excluded missing data
        assert n11 + n10 + n01 + n00 == 4  # 6 total - 2 with missing = 4 valid
    
    def test_pg_gpu_haplotype_matrix_with_missing(self, simple_missing_vcf):
        """Test how pg_gpu loads haplotype matrices with missing data."""
        # Load with pg_gpu
        h_matrix = HaplotypeMatrix.from_vcf(simple_missing_vcf)
        
        # Check the haplotype matrix
        hap_data = h_matrix.haplotypes
        if hasattr(hap_data, 'get'):  # If on GPU
            hap_data = hap_data.get()
        
        print("\npg_gpu haplotype matrix shape:", hap_data.shape)
        print("Haplotype data:\n", hap_data)
        
        # Check for -1 values (missing data)
        has_missing = np.any(hap_data == -1)
        print(f"\nContains missing data (-1 values): {has_missing}")
        
        # Note: The current implementation might not preserve -1 values
        # This test helps us understand current behavior