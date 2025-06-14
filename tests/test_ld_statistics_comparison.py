#!/usr/bin/env python3
"""
Unit tests for comparing LD statistics between pg_gpu and moments implementations.

Each test uses a reference dataset and computes a specific LD statistic using both
implementations, checking that they agree within numerical tolerance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'moments'))

import numpy as np
import pytest
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import moments.LD.stats_from_haplotype_counts as moments_shc
import tempfile
import allel


class TestLDStatisticsComparison:
    """Test suite for comparing LD statistics between pg_gpu and moments."""
    
    @classmethod
    def setup_class(cls):
        """Create reference dataset for all tests."""
        # Create a simple test dataset with known structure
        # 2 populations, 10 samples each (20 haplotypes per population)
        # 5 variants
        np.random.seed(42)
        
        # Create haplotype data
        n_variants = 5
        n_samples_per_pop = 10
        n_pops = 2
        
        # Generate haplotype matrix (variants x haplotypes)
        # Make it somewhat structured to get meaningful LD
        haplotypes = np.zeros((n_variants, n_samples_per_pop * n_pops * 2), dtype=np.int8)
        
        # Create some LD structure
        # Population 0: correlated variants 0-1 and 2-3
        for i in range(n_samples_per_pop * 2):
            if np.random.rand() < 0.7:
                haplotypes[0, i] = 1
                haplotypes[1, i] = 1
            if np.random.rand() < 0.6:
                haplotypes[2, i] = 1
                haplotypes[3, i] = 1
            haplotypes[4, i] = np.random.randint(2)
        
        # Population 1: different LD structure
        offset = n_samples_per_pop * 2
        for i in range(n_samples_per_pop * 2):
            if np.random.rand() < 0.5:
                haplotypes[0, offset + i] = 1
                haplotypes[2, offset + i] = 1
            if np.random.rand() < 0.8:
                haplotypes[1, offset + i] = 1
                haplotypes[3, offset + i] = 1
            haplotypes[4, offset + i] = np.random.randint(2)
        
        # Create positions
        positions = np.array([100, 200, 500, 1000, 2000])
        
        # Create sample names
        samples = [f"ind{i}" for i in range(n_samples_per_pop * n_pops)]
        
        # Create genotype array from haplotypes (samples x variants x 2)
        genotypes = np.zeros((n_samples_per_pop * n_pops, n_variants, 2), dtype=np.int8)
        for i in range(n_samples_per_pop * n_pops):
            genotypes[i, :, 0] = haplotypes[:, i]
            genotypes[i, :, 1] = haplotypes[:, i + n_samples_per_pop * n_pops]
        
        # Create VCF for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            cls.vcf_path = f.name
            
            # Write VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##contig=<ID=1>\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT')
            for sample in samples:
                f.write(f'\t{sample}')
            f.write('\n')
            
            # Write variants
            for v in range(n_variants):
                f.write(f'1\t{positions[v]}\t.\tA\tT\t.\tPASS\t.\tGT')
                for s in range(len(samples)):
                    f.write(f'\t{genotypes[s, v, 0]}|{genotypes[s, v, 1]}')
                f.write('\n')
        
        # Create population file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            cls.pop_file = f.name
            f.write("sample\tpop\n")
            for i, sample in enumerate(samples):
                if i < n_samples_per_pop:
                    f.write(f"{sample}\tpop0\n")
                else:
                    f.write(f"{sample}\tpop1\n")
        
        # Store test data
        cls.haplotypes = haplotypes
        cls.positions = positions
        cls.samples = samples
        cls.n_samples_per_pop = n_samples_per_pop
        cls.n_pops = n_pops
        
    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        if os.path.exists(cls.vcf_path):
            os.remove(cls.vcf_path)
        if os.path.exists(cls.pop_file):
            os.remove(cls.pop_file)
    
    def get_haplotype_counts_for_pair(self, i, j, pop_indices):
        """Get haplotype counts for a variant pair for specified population(s)."""
        haps = self.haplotypes[[i, j], :][:, pop_indices]
        
        # Count haplotypes: 00, 01, 10, 11
        counts = np.zeros(4, dtype=np.int32)
        for h in range(haps.shape[1]):
            if haps[0, h] == 0 and haps[1, h] == 0:
                counts[3] += 1  # 00
            elif haps[0, h] == 0 and haps[1, h] == 1:
                counts[2] += 1  # 01
            elif haps[0, h] == 1 and haps[1, h] == 0:
                counts[1] += 1  # 10
            else:  # 11
                counts[0] += 1  # 11
        
        return counts
    
    def test_DD_single_population(self):
        """Test DD statistics within single populations (DD_0_0, DD_1_1)."""
        # Setup HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(self.vcf_path)
        
        # Read population file to set up sample sets
        pop_assignments = {}
        with open(self.pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop
        
        # Create sample sets
        vcf = allel.read_vcf(self.vcf_path)
        n_samples = len(vcf['samples'])
        pop_sets = {"pop0": [], "pop1": []}
        
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name)
            if pop in pop_sets:
                pop_sets[pop].extend([i, i + n_samples])
        
        h_gpu.sample_sets = pop_sets
        
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Population 0
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                
                # Compute DD using moments
                dd_moments = moments_shc.DD([counts_pop0], [0, 0])
                
                # Compute DD using pg_gpu (simplified direct calculation)
                c1, c2, c3, c4 = counts_pop0
                n = sum(counts_pop0)
                numer = (c1 * (c1 - 1) * c4 * (c4 - 1) +
                        c2 * (c2 - 1) * c3 * (c3 - 1) -
                        2 * c1 * c2 * c3 * c4)
                dd_gpu = numer / (n * (n - 1) * (n - 2) * (n - 3))
                
                assert np.allclose(dd_gpu, dd_moments, rtol=1e-10), \
                    f"DD_0_0 mismatch for pair ({i},{j}): GPU={dd_gpu}, moments={dd_moments}"
                
                # Population 1
                pop1_indices = list(range(self.n_samples_per_pop * 2, self.n_samples_per_pop * 4))
                counts_pop1 = self.get_haplotype_counts_for_pair(i, j, pop1_indices)
                
                dd_moments = moments_shc.DD([counts_pop1], [0, 0])
                
                c1, c2, c3, c4 = counts_pop1
                n = sum(counts_pop1)
                numer = (c1 * (c1 - 1) * c4 * (c4 - 1) +
                        c2 * (c2 - 1) * c3 * (c3 - 1) -
                        2 * c1 * c2 * c3 * c4)
                dd_gpu = numer / (n * (n - 1) * (n - 2) * (n - 3))
                
                assert np.allclose(dd_gpu, dd_moments, rtol=1e-10), \
                    f"DD_1_1 mismatch for pair ({i},{j}): GPU={dd_gpu}, moments={dd_moments}"
    
    def test_DD_between_populations(self):
        """Test DD statistics between populations (DD_0_1)."""
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Get counts for each population
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                pop1_indices = list(range(self.n_samples_per_pop * 2, self.n_samples_per_pop * 4))
                
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                counts_pop1 = self.get_haplotype_counts_for_pair(i, j, pop1_indices)
                
                # Compute DD using moments
                dd_moments = moments_shc.DD([counts_pop0, counts_pop1], [0, 1])
                
                # Compute DD using pg_gpu formula
                c11, c12, c13, c14 = counts_pop0
                c21, c22, c23, c24 = counts_pop1
                n1 = sum(counts_pop0)
                n2 = sum(counts_pop1)
                
                numer = (c12 * c13 - c11 * c14) * (c22 * c23 - c21 * c24)
                dd_gpu = numer / (n1 * (n1 - 1) * n2 * (n2 - 1))
                
                assert np.allclose(dd_gpu, dd_moments, rtol=1e-10), \
                    f"DD_0_1 mismatch for pair ({i},{j}): GPU={dd_gpu}, moments={dd_moments}"
    
    def test_Dz_single_population(self):
        """Test Dz statistics within single populations (Dz_0_0_0, Dz_1_1_1)."""
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Population 0
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                
                # Compute Dz using moments
                dz_moments = moments_shc.Dz([counts_pop0], [0, 0, 0])
                
                # Compute Dz using pg_gpu formula
                c1, c2, c3, c4 = counts_pop0
                n = sum(counts_pop0)
                numer = (
                    (c1 * c4 - c2 * c3) * (c3 + c4 - c1 - c2) * (c2 + c4 - c1 - c3)
                    + (c1 * c4 - c2 * c3) * (c2 + c3 - c1 - c4)
                    + 2 * (c2 * c3 + c1 * c4)
                )
                dz_gpu = numer / (n * (n - 1) * (n - 2) * (n - 3))
                
                assert np.allclose(dz_gpu, dz_moments, rtol=1e-10), \
                    f"Dz_0_0_0 mismatch for pair ({i},{j}): GPU={dz_gpu}, moments={dz_moments}"
    
    def test_Dz_mixed_populations(self):
        """Test Dz statistics with mixed populations (Dz_0_0_1, Dz_0_1_1, etc)."""
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Get counts for each population
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                pop1_indices = list(range(self.n_samples_per_pop * 2, self.n_samples_per_pop * 4))
                
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                counts_pop1 = self.get_haplotype_counts_for_pair(i, j, pop1_indices)
                counts_list = [counts_pop0, counts_pop1]
                
                # Test Dz_0_0_1: Dz(i,i,j)
                dz_moments = moments_shc.Dz(counts_list, [0, 0, 1])
                
                c11, c12, c13, c14 = counts_pop0
                c21, c22, c23, c24 = counts_pop1
                n1 = sum(counts_pop0)
                n2 = sum(counts_pop1)
                numer = (
                    (-c11 - c12 + c13 + c14)
                    * (-(c12 * c13) + c11 * c14)
                    * (-c21 + c22 - c23 + c24)
                )
                dz_gpu = numer / (n2 * n1 * (n1 - 1) * (n1 - 2))
                
                assert np.allclose(dz_gpu, dz_moments, rtol=1e-10), \
                    f"Dz_0_0_1 mismatch for pair ({i},{j}): GPU={dz_gpu}, moments={dz_moments}"
                
                # Test Dz_0_1_1: Dz(i,j,j)
                dz_moments = moments_shc.Dz(counts_list, [0, 1, 1])
                
                numer = (-(c12 * c13) + c11 * c14) * (-c21 + c22 + c23 - c24) + (
                    -(c12 * c13) + c11 * c14
                ) * (-c21 + c22 - c23 + c24) * (-c21 - c22 + c23 + c24)
                dz_gpu = numer / (n1 * (n1 - 1) * n2 * (n2 - 1))
                
                assert np.allclose(dz_gpu, dz_moments, rtol=1e-10), \
                    f"Dz_0_1_1 mismatch for pair ({i},{j}): GPU={dz_gpu}, moments={dz_moments}"
    
    def test_pi2_single_population(self):
        """Test pi2 statistics within single populations (pi2_0_0_0_0, pi2_1_1_1_1)."""
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Population 0
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                
                # Compute pi2 using moments
                pi2_moments = moments_shc.pi2([counts_pop0], [0, 0, 0, 0])
                
                # Compute pi2 using pg_gpu formula
                c1, c2, c3, c4 = counts_pop0
                n = sum(counts_pop0)
                numer = (
                    (c1 + c2) * (c1 + c3) * (c2 + c4) * (c3 + c4)
                    - c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
                    - c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4)
                )
                pi2_gpu = numer / (n * (n - 1) * (n - 2) * (n - 3))
                
                assert np.allclose(pi2_gpu, pi2_moments, rtol=1e-10), \
                    f"pi2_0_0_0_0 mismatch for pair ({i},{j}): GPU={pi2_gpu}, moments={pi2_moments}"
    
    def test_pi2_mixed_populations(self):
        """Test pi2 statistics with mixed populations (pi2_0_0_0_1, pi2_0_0_1_1, etc)."""
        # Test for each variant pair
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                # Get counts for each population
                pop0_indices = list(range(self.n_samples_per_pop * 2))
                pop1_indices = list(range(self.n_samples_per_pop * 2, self.n_samples_per_pop * 4))
                
                counts_pop0 = self.get_haplotype_counts_for_pair(i, j, pop0_indices)
                counts_pop1 = self.get_haplotype_counts_for_pair(i, j, pop1_indices)
                counts_list = [counts_pop0, counts_pop1]
                
                # Test pi2_0_0_1_1: pi2(i,i,j,j)
                pi2_moments = moments_shc.pi2(counts_list, [0, 0, 1, 1])
                
                c11, c12, c13, c14 = counts_pop0
                c21, c22, c23, c24 = counts_pop1
                n1 = sum(counts_pop0)
                n2 = sum(counts_pop1)
                numer = (c11 + c12) * (c13 + c14) * (c21 + c23) * (c22 + c24)
                pi2_gpu = numer / (n1 * (n1 - 1) * n2 * (n2 - 1))
                
                assert np.allclose(pi2_gpu, pi2_moments, rtol=1e-10), \
                    f"pi2_0_0_1_1 mismatch for pair ({i},{j}): GPU={pi2_gpu}, moments={pi2_moments}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])