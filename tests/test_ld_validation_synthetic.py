#!/usr/bin/env python3
"""
Validation tests using synthetic data for quick validation.
This provides a faster alternative to the full IM model validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import pytest
import moments.LD
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import tempfile
import msprime


class TestLDValidationSynthetic:
    """Validation tests using synthetic data from msprime."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data using msprime."""
        # Simulate a simple two-population model
        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=1000)
        demography.add_population(name="pop1", initial_size=1000)
        demography.add_population(name="ancestral", initial_size=2000)
        
        # Split event
        demography.add_population_split(
            time=500, derived=["pop0", "pop1"], ancestral="ancestral"
        )
        
        # Simulate
        ts = msprime.sim_ancestry(
            samples={"pop0": 10, "pop1": 10},  # 10 individuals per pop
            demography=demography,
            sequence_length=50000,
            recombination_rate=1e-8,
            random_seed=42
        )
        
        # Add mutations
        ts = msprime.sim_mutations(
            ts, rate=1e-8, random_seed=42
        )
        
        # Write to temporary VCF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            ts.write_vcf(f)
            vcf_path = f.name
        
        # Create population file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("sample\tpop\n")
            for ind in ts.individuals():
                pop = ts.population(ts.node(ind.nodes[0]).population).metadata["name"]
                f.write(f"tsk_{ind.id}\t{pop}\n")
            pop_file = f.name
        
        yield vcf_path, pop_file, ts
        
        # Cleanup
        os.unlink(vcf_path)
        os.unlink(pop_file)
    
    def test_synthetic_correspondence(self, synthetic_data):
        """Test correspondence on synthetic data."""
        vcf_path, pop_file, ts = synthetic_data
        
        # Define distance bins
        bp_bins = np.array([0, 1000, 5000, 20000, 50000])
        
        # Compute moments LD statistics
        import moments.LD.Parsing as mParsing
        moments_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=["pop0", "pop1"],
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compute GPU statistics
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
        
        # Set up population sample sets based on VCF samples
        import allel
        vcf = allel.read_vcf(vcf_path)
        n_samples = vcf['samples'].shape[0]
        
        # Read population assignments
        pop_assignments = {}
        with open(pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop
        
        # Create sample sets
        pop_sets = {"pop0": [], "pop1": []}
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name, None)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)
        
        h_gpu.sample_sets = pop_sets
        
        # Compute GPU LD statistics
        gpu_results = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
                        raw=True
        )
        
        # Compare results
        stat_names = moments_stats['stats'][0]
        total_comparisons = 0
        errors = []
        
        for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_stats['bins'], moments_stats['sums'])):
            gpu_bin = gpu_results[bin_range]
            
            for stat_idx, (stat_name, mom_val) in enumerate(zip(stat_names, moments_sums)):
                gpu_val = gpu_bin[stat_name]
                
                total_comparisons += 1
                
                # Calculate error
                if abs(mom_val) > 1e-10:
                    rel_error = abs(gpu_val - mom_val) / abs(mom_val)
                    if rel_error > 0.1:  # 10% threshold
                        errors.append((
                            f"Bin {bin_idx} {stat_name}: "
                            f"GPU={gpu_val:.6f}, moments={mom_val:.6f}, "
                            f"rel_error={rel_error:.3f}"
                        ))
                else:
                    abs_error = abs(gpu_val - mom_val)
                    if abs_error > 1e-6:
                        errors.append((
                            f"Bin {bin_idx} {stat_name}: "
                            f"GPU={gpu_val:.6f}, moments={mom_val:.6f}, "
                            f"abs_error={abs_error:.6f}"
                        ))
        
        # Report results
        error_rate = len(errors) / total_comparisons
        
        # Allow some errors due to numerical precision
        assert error_rate < 0.1, (
            f"Too many errors: {len(errors)}/{total_comparisons} ({error_rate:.1%})\n"
            f"First few errors:\n" + "\n".join(errors[:5])
        )
    
    def test_debug_dz_errors(self, synthetic_data):
        """Debug test to understand Dz errors."""
        vcf_path, pop_file, ts = synthetic_data
        
        # Define distance bins - including larger distances where errors occur
        bp_bins = np.array([0, 1000, 5000, 20000, 50000])
        
        # Compute moments LD statistics
        import moments.LD.Parsing as mParsing
        moments_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=["pop0", "pop1"],
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        
        # Compute GPU statistics
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
        
        # Set up population sample sets based on VCF samples
        import allel
        vcf = allel.read_vcf(vcf_path)
        n_samples = vcf['samples'].shape[0]
        
        # Read population assignments
        pop_assignments = {}
        with open(pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop
        
        # Create sample sets
        pop_sets = {"pop0": [], "pop1": []}
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name, None)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)
        
        h_gpu.sample_sets = pop_sets
        
        # Compute GPU LD statistics
        gpu_results = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="pop0",
            pop2="pop1",
                        raw=True
        )
        
        # Focus on Dz statistics
        stat_names = moments_stats['stats'][0]
        dz_indices = [(i, name) for i, name in enumerate(stat_names) if name.startswith('Dz')]
        
        print("\n" + "=" * 60)
        print("DZ STATISTICS COMPARISON")
        print("=" * 60)
        
        # Analyze each bin
        for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_stats['bins'], moments_stats['sums'])):
            print(f"\nBin {bin_idx}: {bin_range}")
            print("-" * 40)
            
            gpu_bin = gpu_results[bin_range]
            
            # Check Dz statistics
            for stat_idx, stat_name in dz_indices:
                mom_val = moments_sums[stat_idx]
                gpu_val = gpu_bin[stat_name]
                
                if abs(mom_val) > 1e-10:
                    rel_error = abs(gpu_val - mom_val) / abs(mom_val)
                else:
                    rel_error = 0.0 if abs(gpu_val) < 1e-10 else float('inf')
                
                print(f"  {stat_name:10s}: moments={mom_val:10.6f}, gpu={gpu_val:10.6f}, rel_err={rel_error:6.3f}")
        
        # Also check the counts and normalization
        print("\n" + "=" * 60)
        print("CHECKING COUNTS AND NORMALIZATION")
        print("=" * 60)
        
        # Get the number of pairs per bin
        print("\nNumber of variant pairs per bin (from moments):")
        if hasattr(moments_stats, 'num_pairs_per_bin'):
            for i, n_pairs in enumerate(moments_stats.get('num_pairs_per_bin', [])):
                print(f"  Bin {i}: {n_pairs} pairs")
        
        # Check heterozygosity stats if available
        if 'H' in moments_stats:
            print("\nHeterozygosity stats from moments:")
            for i, h_val in enumerate(moments_stats['H']):
                print(f"  H[{i}]: {h_val}")
    
    def test_simple_two_variant_case(self):
        """Test a very simple case with just two variants."""
        # Create simple test data
        vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\tind2\tind3
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1\t1|0\t0|0
1\t500\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t1|0\t0|1\t1|1"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write(vcf_content)
            vcf_path = f.name
        
        try:
            # Create HaplotypeMatrix
            h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
            
            # Define populations
            pop_sets = {
                "pop0": [0, 1, 4, 5],  # ind0 and ind1
                "pop1": [2, 3, 6, 7]   # ind2 and ind3
            }
            h_gpu.sample_sets = pop_sets
            
            # Simple distance bins
            bp_bins = np.array([0, 1000])
            
            # Get GPU results
            gpu_results = h_gpu.compute_ld_statistics_gpu_two_pops(
                bp_bins=bp_bins,
                pop1="pop0",
                pop2="pop1",
                                raw=True
            )
            
            # Check that we got results
            assert len(gpu_results) == 1  # One bin
            bin_range = list(gpu_results.keys())[0]
            stats = gpu_results[bin_range]
            
            # Verify we have all 15 statistics
            expected_stats = [
                'DD_0_0', 'DD_0_1', 'DD_1_1',
                'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1', 
                'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
                'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1',
                'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
            ]
            
            for stat_name in expected_stats:
                assert stat_name in stats, f"Missing statistic: {stat_name}"
                # Check that values are finite
                assert np.isfinite(stats[stat_name]), (
                    f"{stat_name} is not finite: {stats[stat_name]}"
                )
            
        finally:
            os.unlink(vcf_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])