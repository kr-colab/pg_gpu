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
pytest.importorskip("moments.LD")
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
        vcf_path = "examples/data/im-parsing-example.vcf"
        pop_file = "examples/data/im_pop.txt"

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
        rtol = 0.01  # 1% relative tolerance for all statistics
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

    def test_debug_dz_discrepancies(self, moments_results, gpu_results):
        """Debug test to understand Dz discrepancies with IM model data."""
        print("\n" + "=" * 60)
        print("DEBUGGING DZ DISCREPANCIES - IM MODEL")
        print("=" * 60)

        stat_names = moments_results['stats'][0]
        dz_indices = [(i, name) for i, name in enumerate(stat_names) if name.startswith('Dz')]

        # Track errors by bin
        errors_by_bin = {}

        for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_results['bins'], moments_results['sums'])):
            print(f"\nBin {bin_idx}: {bin_range}")
            print("-" * 50)

            gpu_bin = gpu_results[bin_range]
            errors_by_bin[bin_idx] = []

            # Check all statistics but focus on Dz
            for stat_idx, stat_name in dz_indices:
                mom_val = moments_sums[stat_idx]
                gpu_val = gpu_bin[stat_name]

                if abs(mom_val) > 1e-10:
                    rel_error = abs(gpu_val - mom_val) / abs(mom_val)
                else:
                    rel_error = 0.0 if abs(gpu_val) < 1e-10 else float('inf')

                errors_by_bin[bin_idx].append((stat_name, rel_error))

                # Print details for high errors
                if rel_error > 0.05:  # 5% threshold
                    print(f"  {stat_name:10s}: moments={mom_val:10.6f}, gpu={gpu_val:10.6f}, rel_err={rel_error:6.3f} ***")
                else:
                    print(f"  {stat_name:10s}: moments={mom_val:10.6f}, gpu={gpu_val:10.6f}, rel_err={rel_error:6.3f}")

        # Summary of errors
        print("\n" + "=" * 60)
        print("ERROR SUMMARY")
        print("=" * 60)

        for bin_idx, errors in errors_by_bin.items():
            high_errors = [(name, err) for name, err in errors if err > 0.05]
            if high_errors:
                print(f"\nBin {bin_idx} - High error statistics:")
                for name, err in high_errors:
                    print(f"  {name}: {err:.3f}")

        # Check if errors are systematic
        print("\n" + "=" * 60)
        print("CHECKING FOR SYSTEMATIC PATTERNS")
        print("=" * 60)

        # Group errors by statistic type
        errors_by_stat = {}
        for bin_idx, errors in errors_by_bin.items():
            for stat_name, err in errors:
                if stat_name not in errors_by_stat:
                    errors_by_stat[stat_name] = []
                errors_by_stat[stat_name].append((bin_idx, err))

        # Print patterns
        for stat_name, bin_errors in errors_by_stat.items():
            max_err = max(err for _, err in bin_errors)
            if max_err > 0.05:
                print(f"\n{stat_name}:")
                for bin_idx, err in bin_errors:
                    print(f"  Bin {bin_idx}: {err:.3f}")

        # Additional debugging info
        print("\n" + "=" * 60)
        print("ADDITIONAL INFO")
        print("=" * 60)

        # Check if we have counts info
        if 'counts' in moments_results:
            print("\nPair counts per bin from moments:")
            for i, count in enumerate(moments_results.get('counts', [])):
                print(f"  Bin {i}: {count}")

        # Check heterozygosity
        if len(moments_results['sums']) > 0 and len(moments_results['stats']) > 1:
            print(f"\nHeterozygosity stats: {moments_results['stats'][1]}")
            for i, h_vals in enumerate(moments_results['sums']):
                if i < len(moments_results['bins']):
                    print(f"  Bin {i}: H values = {h_vals[-3:] if len(h_vals) > 15 else 'N/A'}")

    def test_debug_pair_counts(self, im_model_data, moments_results):
        """Debug test to check pair counts per bin."""
        vcf_path, pop_file = im_model_data

        print("\n" + "=" * 60)
        print("DEBUGGING PAIR COUNTS PER BIN")
        print("=" * 60)

        # Get bins from moments
        bp_bins = [b[1] for b in moments_results['bins']]
        bp_bins = [moments_results['bins'][0][0]] + bp_bins

        # Load VCF data
        import allel
        vcf = allel.read_vcf(vcf_path)
        positions = vcf['variants/POS']
        n_variants = len(positions)

        print(f"\nTotal variants: {n_variants}")
        print(f"Total possible pairs: {n_variants * (n_variants - 1) // 2}")
        print(f"\nBins: {bp_bins}")

        # Count pairs per bin manually
        import numpy as np
        pair_counts = np.zeros(len(bp_bins) - 1)

        for i in range(n_variants):
            for j in range(i + 1, n_variants):
                dist = positions[j] - positions[i]
                bin_idx = np.digitize(dist, bp_bins) - 1
                if 0 <= bin_idx < len(pair_counts):
                    pair_counts[bin_idx] += 1

        print("\nPair counts per bin (manual calculation):")
        for i, count in enumerate(pair_counts):
            print(f"  Bin {i} {moments_results['bins'][i]}: {int(count)} pairs")

        # Check if moments provides pair counts
        print("\nChecking moments internal data...")
        for key in moments_results.keys():
            if 'count' in key.lower() or 'pair' in key.lower():
                print(f"  Found key: {key} = {moments_results[key]}")

    def test_debug_population_assignment(self, im_model_data, gpu_results):
        """Debug test to check population assignments."""
        vcf_path, pop_file = im_model_data

        print("\n" + "=" * 60)
        print("DEBUGGING POPULATION ASSIGNMENTS")
        print("=" * 60)

        # Load VCF data
        import allel
        vcf = allel.read_vcf(vcf_path)
        n_samples = vcf['samples'].shape[0]

        # Read population file
        pop_assignments = {}
        with open(pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop

        # Count samples per population
        pop_counts = {}
        for sample in vcf['samples']:
            pop = pop_assignments.get(sample, 'unknown')
            pop_counts[pop] = pop_counts.get(pop, 0) + 1

        print(f"\nTotal samples: {n_samples}")
        print("\nPopulation counts:")
        for pop, count in sorted(pop_counts.items()):
            print(f"  {pop}: {count} samples ({count * 2} haplotypes)")

        # Check GPU haplotype matrix setup
        from pg_gpu.haplotype_matrix import HaplotypeMatrix
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)

        # Set up populations as in the main test
        pop_sets = {"deme0": [], "deme1": []}
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name, None)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)

        h_gpu.sample_sets = pop_sets

        print("\nGPU population setup:")
        for pop, indices in pop_sets.items():
            print(f"  {pop}: {len(indices)} haplotype indices")
            print(f"    Sample indices: {sorted(set(i % n_samples for i in indices))}")

        # Test with a simple case - just count haplotypes
        print("\nTesting haplotype counts for first few variants...")
        for var_idx in range(min(3, h_gpu.num_variants)):
            print(f"\nVariant {var_idx}:")
            for pop in ["deme0", "deme1"]:
                counts = h_gpu.tally_gpu_haplotypes(pop=pop)
                if var_idx < len(counts):
                    print(f"  {pop}: counts = {counts[var_idx].get() if hasattr(counts[var_idx], 'get') else counts[var_idx]}")

    def test_debug_specific_dz_calculation(self, im_model_data):
        """Debug a specific Dz calculation to understand the discrepancy."""
        vcf_path, pop_file = im_model_data

        print("\n" + "=" * 60)
        print("DEBUGGING SPECIFIC DZ CALCULATION")
        print("=" * 60)

        # Import what we need
        import numpy as np
        from pg_gpu.haplotype_matrix import HaplotypeMatrix
        import allel

        # Load data
        vcf = allel.read_vcf(vcf_path)
        h_gpu = HaplotypeMatrix.from_vcf(vcf_path)

        # Set up populations
        pop_assignments = {}
        with open(pop_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                sample, pop = line.strip().split()
                pop_assignments[sample] = pop

        pop_sets = {"deme0": [], "deme1": []}
        n_samples = vcf['samples'].shape[0]
        for i, sample_name in enumerate(vcf['samples']):
            pop = pop_assignments.get(sample_name, None)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)

        h_gpu.sample_sets = pop_sets

        print("\nPopulation setup complete:")
        print(f"  deme0: {len(pop_sets['deme0'])} haplotypes")
        print(f"  deme1: {len(pop_sets['deme1'])} haplotypes")
        print(f"  Total variants: {h_gpu.num_variants}")

        # Test basic computation to ensure setup is working
        print("\nTesting basic LD computation...")
        bp_bins = [100, 1000, 10000, 100000]
        result = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1="deme0",
            pop2="deme1",
            raw=True
        )

        print(f"  Computation successful, got {len(result)} bins")
        print(f"  First bin has {len(result[(100, 1000)])} statistics")

        # Check one statistic as a sanity check
        first_bin = result[(100, 1000)]
        print(f"  Example - DD_0_0 in first bin: {first_bin['DD_0_0']:.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
