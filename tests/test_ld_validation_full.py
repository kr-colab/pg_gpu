#!/usr/bin/env python3
"""
Validation tests comparing pg_gpu against moments for two-population LD statistics.
This is a test-suite version of the validate_two_pop_ld.py script.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import pytest
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import allel
import tempfile
from pathlib import Path
import pickle


class TestLDValidationFull:
    """Full validation tests against moments package."""
    
    @pytest.fixture(scope="class")
    def im_model_data(self):
        """Fixture for IM model test data."""
        # Check if test data exists
        vcf_path = "data/im-parsing-example.vcf"
        pop_file = "data/im_pop.txt"
        
        if not os.path.exists(vcf_path) or not os.path.exists(pop_file):
            pytest.skip("IM model test data not available")
        
        return vcf_path, pop_file
    
    @pytest.fixture(scope="class") 
    def moments_results(self, im_model_data):
        """Compute or load cached moments results."""
        vcf_path, pop_file = im_model_data
        
        # Use fewer bins for faster testing
        bp_bins = np.logspace(2, 5, 4)  # Only 4 bins instead of 6
        pops = ["deme0", "deme1"]
        
        # Create cache for test results
        cache_dir = Path("tests/cache")
        cache_dir.mkdir(exist_ok=True)
        
        cache_key = f"test_im_4bins_{'-'.join(pops)}"
        cache_file = cache_dir / f"moments_ld_{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Compute moments LD statistics
        ld_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False  # Quiet for tests
        )
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(ld_stats, f)
        
        return ld_stats
    
    @pytest.fixture(scope="class")
    def gpu_results(self, im_model_data, moments_results):
        """Compute GPU results."""
        vcf_path, pop_file = im_model_data
        
        # Get bins from moments results
        bp_bins = [b[1] for b in moments_results['bins']]
        bp_bins = [moments_results['bins'][0][0]] + bp_bins  # Add start point
        
        # Setup GPU computation
        vcf = allel.read_vcf(vcf_path)
        n_samples = vcf['samples'].shape[0]
        
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
        
        # Read population assignments
        pop_assignments = {}
        with open(pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop
        
        # Create sample sets
        pop_sets = {"deme0": [], "deme1": []}
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name, None)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)
        
        h_gpu.sample_sets = pop_sets
        
        # Compute LD statistics
        return h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="deme0",
            pop2="deme1",
            missing=False,
            raw=True
        )
    
    @pytest.mark.parametrize("stat_idx,stat_name", [
        (0, "DD_0_0"),
        (1, "DD_0_1"), 
        (2, "DD_1_1"),
        (3, "Dz_0_0_0"),
        (4, "Dz_0_0_1"),
        (5, "Dz_0_1_1"),
        (6, "Dz_1_0_0"),
        (7, "Dz_1_0_1"),
        (8, "Dz_1_1_1"),
        (9, "pi2_0_0_0_0"),
        (10, "pi2_0_0_0_1"),
        (11, "pi2_0_0_1_1"),
        (12, "pi2_0_1_0_1"),
        (13, "pi2_0_1_1_1"),
        (14, "pi2_1_1_1_1")
    ])
    def test_individual_statistics(self, moments_results, gpu_results, stat_idx, stat_name):
        """Test each statistic individually across all bins."""
        # Tolerance for comparison
        # Dz statistics can have higher errors at large distances due to small absolute values
        if stat_name.startswith("Dz"):
            rtol = 0.20  # 20% relative tolerance for Dz
        else:
            rtol = 0.05  # 5% relative tolerance for DD and pi2
        atol = 1e-6  # Small absolute tolerance for near-zero values
        
        # Compare across all bins
        for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_results['bins'], moments_results['sums'])):
            moments_val = moments_sums[stat_idx]
            gpu_val = gpu_results[bin_range][stat_name]
            
            # Check if values are close
            if abs(moments_val) > atol:  # For non-zero values
                rel_error = abs(gpu_val - moments_val) / abs(moments_val)
                assert rel_error < rtol, (
                    f"{stat_name} in bin {bin_idx} {bin_range}: "
                    f"GPU={gpu_val:.6f}, moments={moments_val:.6f}, "
                    f"rel_error={rel_error:.6f}"
                )
            else:  # For near-zero values
                assert abs(gpu_val - moments_val) < atol, (
                    f"{stat_name} in bin {bin_idx} {bin_range}: "
                    f"GPU={gpu_val:.6f}, moments={moments_val:.6f}"
                )
    
    def test_overall_correlation(self, moments_results, gpu_results):
        """Test overall correlation between moments and GPU results."""
        moments_vals = []
        gpu_vals = []
        
        stat_names = moments_results['stats'][0]
        
        for bin_range, moments_sums in zip(moments_results['bins'], moments_results['sums']):
            gpu_bin_results = gpu_results[bin_range]
            
            for stat_name, mom_val in zip(stat_names, moments_sums):
                moments_vals.append(mom_val)
                gpu_vals.append(gpu_bin_results[stat_name])
        
        moments_vals = np.array(moments_vals)
        gpu_vals = np.array(gpu_vals)
        
        # Calculate correlation
        correlation = np.corrcoef(moments_vals, gpu_vals)[0, 1]
        
        # Should have very high correlation
        assert correlation > 0.999, f"Correlation too low: {correlation}"
    
    def test_mean_relative_error(self, moments_results, gpu_results):
        """Test that mean relative error is acceptably low."""
        relative_errors = []
        
        stat_names = moments_results['stats'][0]
        
        for bin_range, moments_sums in zip(moments_results['bins'], moments_results['sums']):
            gpu_bin_results = gpu_results[bin_range]
            
            for stat_name, mom_val in zip(stat_names, moments_sums):
                gpu_val = gpu_bin_results[stat_name]
                
                if abs(mom_val) > 1e-10:  # Avoid division by zero
                    rel_error = abs(gpu_val - mom_val) / abs(mom_val)
                    relative_errors.append(rel_error)
        
        mean_rel_error = np.mean(relative_errors)
        
        # Mean relative error should be low
        assert mean_rel_error < 0.05, f"Mean relative error too high: {mean_rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])