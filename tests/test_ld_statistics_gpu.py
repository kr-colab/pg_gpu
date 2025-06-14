#!/usr/bin/env python3
"""
Unit tests for comparing LD statistics between pg_gpu GPU implementation and moments.

This test directly uses the pg_gpu GPU code through HaplotypeMatrix.compute_ld_statistics_gpu_two_pops
and compares against moments implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import pytest
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import moments.LD.Parsing as mParsing
import tempfile
import allel
import pickle
from pathlib import Path


class TestLDStatisticsGPU:
    """Test suite for comparing GPU LD statistics implementation against moments."""
    
    @classmethod
    def setup_class(cls):
        """Create reference dataset for all tests."""
        # Create a test dataset with known structure
        # 2 populations, 10 samples each (20 haplotypes per population)
        # 10 variants for more statistical power
        np.random.seed(42)
        
        # Create haplotype data
        n_variants = 10
        n_samples_per_pop = 10
        n_pops = 2
        
        # Generate haplotype matrix (variants x haplotypes)
        haplotypes = np.zeros((n_variants, n_samples_per_pop * n_pops * 2), dtype=np.int8)
        
        # Create LD structure
        # Population 0: strong LD between adjacent variants
        for i in range(n_samples_per_pop * 2):
            # Create blocks of correlated variants
            if np.random.rand() < 0.7:
                haplotypes[0:2, i] = 1
            if np.random.rand() < 0.6:
                haplotypes[2:4, i] = 1
            if np.random.rand() < 0.5:
                haplotypes[4:6, i] = 1
            # Random for remaining
            haplotypes[6:, i] = np.random.randint(2, size=n_variants-6)
        
        # Population 1: different LD pattern
        offset = n_samples_per_pop * 2
        for i in range(n_samples_per_pop * 2):
            # Different correlation pattern
            if np.random.rand() < 0.5:
                haplotypes[0, offset + i] = 1
                haplotypes[3, offset + i] = 1
            if np.random.rand() < 0.8:
                haplotypes[1, offset + i] = 1
                haplotypes[2, offset + i] = 1
            if np.random.rand() < 0.6:
                haplotypes[4, offset + i] = 1
                haplotypes[5, offset + i] = 1
                haplotypes[6, offset + i] = 1
            # Random for remaining
            haplotypes[7:, offset + i] = np.random.randint(2, size=n_variants-7)
        
        # Create positions with varying distances
        positions = np.array([100, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000])
        
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
            f.write("sample\tpop\n")  # Header required by moments
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
        cls.n_variants = n_variants
        
    @classmethod
    def teardown_class(cls):
        """Clean up test files."""
        if os.path.exists(cls.vcf_path):
            os.remove(cls.vcf_path)
        if os.path.exists(cls.pop_file):
            os.remove(cls.pop_file)
    
    def test_gpu_ld_statistics_within_population(self):
        """Test GPU LD statistics computation within populations."""
        # Setup HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(self.vcf_path)
        
        # Read population file to set up sample sets
        pop_assignments = {}
        with open(self.pop_file, 'r') as f:
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
                # Add both haplotypes for this sample
                pop_sets[pop].extend([i, i + n_samples])
        
        h_gpu.sample_sets = pop_sets
        
        # Define bins based on positions
        bp_bins = np.array([0, 500, 2000, 5000, 25000])
        
        # Compute LD statistics using GPU for pop0
        gpu_stats_pop0 = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop0",
            missing=False,
            raw=True
        )
        
        # Compute using moments for comparison
        pops = ["pop0"]
        
        # Create cache directory
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Use moments to compute LD statistics
        moments_stats = mParsing.compute_ld_statistics(
            self.vcf_path,
            pop_file=self.pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compare results for each bin
        for i, (bin_start, bin_end) in enumerate(moments_stats['bins']):
            gpu_values = gpu_stats_pop0[(float(bin_start), float(bin_end))]
            moments_values = moments_stats['sums'][i]
            
            # The GPU now returns an OrderedDict with named statistics
            # For single population, we only have DD_0_0, Dz_0_0_0, pi2_0_0_0_0
            # In moments output for single pop: DD_0_0, Dz_0_0_0, pi2_0_0_0_0
            
            # Compare DD_0_0
            assert np.allclose(gpu_values['DD_0_0'], moments_values[0], rtol=1e-8), \
                f"DD_0_0 mismatch in bin {i}: GPU={gpu_values['DD_0_0']}, moments={moments_values[0]}"
            
            # Compare Dz_0_0_0
            assert np.allclose(gpu_values['Dz_0_0_0'], moments_values[1], rtol=1e-8), \
                f"Dz_0_0_0 mismatch in bin {i}: GPU={gpu_values['Dz_0_0_0']}, moments={moments_values[1]}"
            
            # Compare pi2_0_0_0_0
            assert np.allclose(gpu_values['pi2_0_0_0_0'], moments_values[2], rtol=1e-8), \
                f"pi2_0_0_0_0 mismatch in bin {i}: GPU={gpu_values['pi2_0_0_0_0']}, moments={moments_values[2]}"
    
    def test_gpu_ld_statistics_between_populations(self):
        """Test GPU LD statistics computation between populations."""
        # Setup HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(self.vcf_path)
        
        # Read population file to set up sample sets
        pop_assignments = {}
        with open(self.pop_file, 'r') as f:
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
        
        # Define bins
        bp_bins = np.array([0, 500, 2000, 5000, 25000])
        
        # Compute LD statistics using GPU between populations
        gpu_stats = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
            missing=False,
            raw=True
        )
        
        # Compute using moments for two populations
        pops = ["pop0", "pop1"]
        
        moments_stats = mParsing.compute_ld_statistics(
            self.vcf_path,
            pop_file=self.pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compare results for each bin
        for i, (bin_start, bin_end) in enumerate(moments_stats['bins']):
            gpu_values = gpu_stats[(float(bin_start), float(bin_end))]
            moments_values = moments_stats['sums'][i]
            
            # For two populations, moments returns 15 statistics
            # Compare all 15 statistics
            stat_names = moments_stats['stats'][0]
            
            for j, stat_name in enumerate(stat_names):
                assert np.allclose(gpu_values[stat_name], moments_values[j], rtol=1e-8), \
                    f"{stat_name} mismatch in bin {i}: GPU={gpu_values[stat_name]}, moments={moments_values[j]}"
    
    def test_gpu_ld_statistics_averaged(self):
        """Test GPU LD statistics computation with averaging (raw=False)."""
        # Setup HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(self.vcf_path)
        
        # Read population file to set up sample sets
        pop_assignments = {}
        with open(self.pop_file, 'r') as f:
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
        
        # Define bins
        bp_bins = np.array([0, 2000, 25000])
        
        # Compute raw sums
        gpu_stats_raw = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
            missing=False,
            raw=True
        )
        
        # Compute averages
        gpu_stats_avg = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
            missing=False,
            raw=False
        )
        
        # For each bin, verify that average = sum / num_pairs
        for bin_range in gpu_stats_raw:
            raw_dict = gpu_stats_raw[bin_range]
            avg_dict = gpu_stats_avg[bin_range]
            
            # Count number of pairs in this bin
            bin_start, bin_end = bin_range
            pair_count = 0
            for i in range(self.n_variants):
                for j in range(i + 1, self.n_variants):
                    dist = self.positions[j] - self.positions[i]
                    if bin_start <= dist < bin_end:
                        pair_count += 1
            
            if pair_count > 0:
                # Check each statistic
                for stat_name in raw_dict.keys():
                    expected_avg = raw_dict[stat_name] / pair_count
                    assert np.allclose(avg_dict[stat_name], expected_avg, rtol=1e-10), \
                        f"{stat_name} average calculation mismatch in bin {bin_range}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])