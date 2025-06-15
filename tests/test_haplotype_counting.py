#!/usr/bin/env python3
"""
Tests for haplotype counting functionality in pg_gpu.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import pytest
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import tempfile
import allel
import cupy as cp


class TestHaplotypeCounting:
    """Test haplotype counting functionality."""
    
    def test_within_population_counting(self):
        """Test haplotype counting within a single population."""
        # Create simple test case with known counts
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t1|0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            vcf_path = f.name
            f.write(vcf_content)
        
        try:
            # Load data
            h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
            
            # Count haplotypes for variant pair (0,1)
            # Expected: ind0_hap0=(1,1), ind0_hap1=(1,0), ind1_hap0=(0,1), ind1_hap1=(1,0)
            # Counts: n11=1, n10=2, n01=1, n00=0
            counts, n_valid = h_gpu.tally_gpu_haplotypes()
            
            assert counts.shape == (1, 4)  # 1 pair, 4 count types
            counts_cpu = counts[0].get()
            
            expected = np.array([1, 2, 1, 0])
            np.testing.assert_array_equal(counts_cpu, expected)
            
        finally:
            os.unlink(vcf_path)
    
    def test_two_population_counting(self):
        """Test haplotype counting for two populations."""
        # Create test data with 2 populations
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\tind2\tind3
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1\t1|0\t0|0
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t1|0\t1|1\t0|0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            vcf_path = f.name
            f.write(vcf_content)
        
        try:
            h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
            
            # Define populations
            pop_sets = {
                "pop0": [0, 1, 4, 5],  # ind0 and ind1 haplotypes
                "pop1": [2, 3, 6, 7]   # ind2 and ind3 haplotypes
            }
            h_gpu.sample_sets = pop_sets
            
            # Test within-population counts
            counts_pop0, _ = h_gpu.tally_gpu_haplotypes(pop="pop0")
            counts_pop1, _ = h_gpu.tally_gpu_haplotypes(pop="pop1")
            counts_pop0 = counts_pop0[0].get()
            counts_pop1 = counts_pop1[0].get()
            
            # Pop0: ind0=(1,1)|(1,0), ind1=(0,1)|(1,0)
            # Haplotypes: (1,1), (1,0), (0,1), (1,0)
            # Expected: n11=1, n10=2, n01=1, n00=0
            np.testing.assert_array_equal(counts_pop0, [1, 2, 1, 0])
            
            # Pop1: ind2=(1,1)|(0,0), ind3=(0,0)|(0,0)  
            # Haplotypes: (1,1), (0,1), (0,0), (0,0)
            # Expected: n11=1, n10=0, n01=1, n00=2
            np.testing.assert_array_equal(counts_pop1, [1, 0, 1, 2])
            
            # Test between-population counts
            counts_between, _, _ = h_gpu.tally_gpu_haplotypes_two_pops("pop0", "pop1")
            counts_between = counts_between[0].get()
            
            # Should contain 8 values: [counts_pop0, counts_pop1]
            expected_between = np.array([1, 2, 1, 0, 1, 0, 1, 2])
            np.testing.assert_array_equal(counts_between, expected_between)
            
        finally:
            os.unlink(vcf_path)
    
    def test_multiple_variant_pairs(self):
        """Test counting for multiple variant pairs."""
        # Create data with 3 variants -> 3 pairs
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|0
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t0|1
1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            vcf_path = f.name
            f.write(vcf_content)
        
        try:
            h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
            counts, n_valid = h_gpu.tally_gpu_haplotypes()
            
            # Should have 3 pairs: (0,1), (0,2), (1,2)
            assert counts.shape == (3, 4)
            
            # Verify each pair has valid counts (sum to number of haplotypes)
            for i in range(3):
                pair_counts = counts[i].get()
                assert np.sum(pair_counts) == 4  # 2 individuals * 2 haplotypes
                
        finally:
            os.unlink(vcf_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])