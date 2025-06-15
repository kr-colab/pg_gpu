#!/usr/bin/env python3
"""
Minimal Reproducible Example for validating two-population LD statistics
between pg_gpu and moments packages.

This script compares the LD statistics computed by both packages for a 
two-population scenario to identify discrepancies.
"""

import sys
import os
# Add moments to path if it's not installed system-wide
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'moments'))

import numpy as np
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import allel
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def setup_test_data():
    """Setup test data and parameters."""
    # Use the existing test VCF with IM model
    vcf_path = "data/im-parsing-example.vcf"
    pop_file = "data/im_pop.txt"
    
    # Define base-pair bins for LD calculation
    bp_bins = np.logspace(2, 6, 6)  # [100, 631, 3981, 25119, 158489, 1000000]
    
    # Population names as defined in the pop file
    pops = ["deme0", "deme1"]
    
    return vcf_path, pop_file, bp_bins, pops


def run_moments_ld(vcf_path, pop_file, bp_bins, pops, use_cache=True):
    """Run moments LD calculation with caching."""
    # Create cache filename based on input parameters
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a unique cache key based on input parameters
    cache_key = f"{Path(vcf_path).stem}_{Path(pop_file).stem}_{len(bp_bins)}bins_{'-'.join(pops)}"
    cache_file = cache_dir / f"moments_ld_{cache_key}.pkl"
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        print("=" * 60)
        print("LOADING MOMENTS LD FROM CACHE")
        print("=" * 60)
        print(f"Loading from: {cache_file}")
        
        with open(cache_file, 'rb') as f:
            ld_stats = pickle.load(f)
            
        print("\nLoaded moments output structure:")
        print(f"  Keys: {list(ld_stats.keys())}")
        print(f"  Number of bins: {len(ld_stats['bins'])}")
        print(f"  Statistics computed: {ld_stats['stats']}")
        print(f"  Populations: {ld_stats['pops']}")
    else:
        print("=" * 60)
        print("COMPUTING MOMENTS LD (THIS MAY TAKE A WHILE)")
        print("=" * 60)
        
        ld_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=True
        )
        
        # Save to cache
        print(f"\nSaving to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(ld_stats, f)
        
        print("\nMoments output structure:")
        print(f"  Keys: {list(ld_stats.keys())}")
        print(f"  Number of bins: {len(ld_stats['bins'])}")
        print(f"  Statistics computed: {ld_stats['stats']}")
        print(f"  Populations: {ld_stats['pops']}")
    
    # Display the raw sums for each bin
    print("\nMoments raw sums per bin:")
    for i, (bin_range, sums) in enumerate(zip(ld_stats['bins'], ld_stats['sums'])):
        print(f"\n  Bin {i} {bin_range}:")
        stat_names = ld_stats['stats'][0]  # LD statistics names
        for j, (stat_name, value) in enumerate(zip(stat_names, sums)):
            print(f"    {stat_name}: {value:.6f}")
    
    return ld_stats


def setup_gpu_populations(vcf_path):
    """Setup population assignments for GPU calculation."""
    # Read VCF to understand structure
    vcf = allel.read_vcf(vcf_path)
    n_samples = vcf['samples'].shape[0]
    
    # Create HaplotypeMatrix
    h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
    
    # Read population file to understand assignments
    pop_assignments = {}
    with open("data/im_pop.txt", 'r') as f:
        for line in f:
            sample, pop = line.strip().split()
            pop_assignments[sample] = pop
    
    # Create sample sets for GPU
    # Each sample contributes 2 haplotypes (diploid)
    pop_sets = {"deme0": [], "deme1": []}
    
    for i, sample_name in enumerate(vcf['samples']):
        pop = pop_assignments.get(sample_name, None)
        if pop in pop_sets:
            # Add both haplotypes for this sample
            pop_sets[pop].append(i)  # First haplotype
            pop_sets[pop].append(i + n_samples)  # Second haplotype
    
    h_gpu.sample_sets = pop_sets
    
    print(f"\nGPU population setup:")
    print(f"  Total samples: {n_samples}")
    print(f"  Total haplotypes: {h_gpu.num_haplotypes}")
    print(f"  deme0 haplotypes: {len(pop_sets['deme0'])}")
    print(f"  deme1 haplotypes: {len(pop_sets['deme1'])}")
    
    return h_gpu


def run_gpu_ld(h_gpu, bp_bins):
    """Run GPU LD calculation."""
    print("\n" + "=" * 60)
    print("GPU LD CALCULATION")
    print("=" * 60)
    
    # Compute LD statistics for all population pairs
    ld_stats_gpu = {}
    
    # Within population statistics
    for pop in ["deme0", "deme1"]:
        key = (pop, pop)
        ld_stats_gpu[key] = h_gpu.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop,
            pop2=pop,
            raw=True
        )
    
    # Between population statistics - both orderings
    ld_stats_gpu[("deme0", "deme1")] = h_gpu.compute_ld_statistics_gpu_two_pops(
        bp_bins=bp_bins,
        pop1="deme0",
        pop2="deme1",
        raw=True
    )
    
    # Also compute swapped ordering for correct Dz_1_0_0 and Dz_1_0_1
    ld_stats_gpu[("deme1", "deme0")] = h_gpu.compute_ld_statistics_gpu_two_pops(
        bp_bins=bp_bins,
        pop1="deme1",
        pop2="deme0",
        raw=True
    )
    
    print("\nGPU output structure:")
    print(f"  Population pairs: {list(ld_stats_gpu.keys())}")
    
    # Display statistics for each population pair
    for pop_pair, pop_stats in ld_stats_gpu.items():
        print(f"\n  Population pair {pop_pair}:")
        for bin_range, stats in pop_stats.items():
            print(f"    Bin {bin_range}: {stats}")
    
    return ld_stats_gpu


def compare_results(moments_stats, gpu_stats):
    """Compare results between moments and GPU implementations."""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    # Extract population pairs from GPU results
    gpu_pop_pairs = list(gpu_stats.keys())
    print(f"\nGPU population pairs: {gpu_pop_pairs}")
    
    # Moments statistics structure
    moments_stat_names = moments_stats['stats'][0]
    print(f"\nMoments statistics (total {len(moments_stat_names)}): {moments_stat_names}")
    
    # Try to understand the mapping between moments and GPU statistics
    print("\n\nDetailed comparison by bin:")
    print("-" * 40)
    
    for i, (bin_range, moments_sums) in enumerate(zip(moments_stats['bins'], moments_stats['sums'])):
        print(f"\nBin {i}: {bin_range}")
        
        # Show moments values
        print("  Moments values:")
        for stat_name, value in zip(moments_stat_names, moments_sums):
            print(f"    {stat_name}: {value:.6f}")
        
        # Show corresponding GPU values for each population pair
        print("  GPU values:")
        for pop_pair in gpu_pop_pairs:
            if bin_range in gpu_stats[pop_pair]:
                gpu_values = gpu_stats[pop_pair][bin_range]
                print(f"    {pop_pair}: {gpu_values}")


def plot_correspondence(moments_stats, gpu_stats):
    """Plot correspondence between moments and GPU statistics."""
    print("\n" + "=" * 60)
    print("PLOTTING CORRESPONDENCE")
    print("=" * 60)
    
    # Extract statistics names and prepare data
    stat_names = moments_stats['stats'][0]  # LD statistics
    n_stats = len(stat_names)
    n_bins = len(moments_stats['bins'])
    
    # Prepare GPU data - need to aggregate across population pairs
    # The GPU returns statistics for (pop0,pop0), (pop1,pop1), and (pop0,pop1)
    # Moments combines these in a specific way
    
    # Initialize arrays for plotting
    moments_values = []
    gpu_values = []
    stat_labels = []
    colors = []
    
    # Color map for different statistics
    color_map = {
        'DD': 'blue',
        'Dz': 'green',
        'pi2': 'red'
    }
    
    # Process each bin
    for bin_idx, (bin_range, moments_sums) in enumerate(zip(moments_stats['bins'], moments_stats['sums'])):
        # Get GPU values for this bin
        gpu_pop00 = gpu_stats[('deme0', 'deme0')][bin_range]
        gpu_pop11 = gpu_stats[('deme1', 'deme1')][bin_range]
        gpu_pop01 = gpu_stats[('deme0', 'deme1')][bin_range]
        
        # Match moments ordering of statistics
        # The GPU now returns an OrderedDict with statistics
        # Extract values in the same order as moments expects
        
        # For two-population case, we need the (deme0, deme1) result
        # The GPU returns an OrderedDict, so we need to extract values in moments order
        
        gpu_ordered = [gpu_pop01[stat_name] for stat_name in stat_names]
        
        # Add to plotting arrays
        for i, (stat_name, mom_val, gpu_val) in enumerate(zip(stat_names, moments_sums, gpu_ordered)):
            moments_values.append(mom_val)
            gpu_values.append(gpu_val)
            stat_labels.append(f"{stat_name}_bin{bin_idx}")
            
            # Determine color based on statistic type
            if stat_name.startswith('DD'):
                colors.append(color_map['DD'])
            elif stat_name.startswith('Dz'):
                colors.append(color_map['Dz'])
            else:  # pi2
                colors.append(color_map['pi2'])
    
    # Convert to numpy arrays
    moments_values = np.array(moments_values)
    gpu_values = np.array(gpu_values)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(moments_values, gpu_values, c=colors, alpha=0.6, s=50)
    
    # Add diagonal line for perfect correspondence
    min_val = min(moments_values.min(), gpu_values.min())
    max_val = max(moments_values.max(), gpu_values.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correspondence')
    
    ax1.set_xlabel('Moments Statistics')
    ax1.set_ylabel('GPU Statistics')
    ax1.set_title('Correspondence between Moments and GPU LD Statistics')
    ax1.set_xscale('symlog')
    ax1.set_yscale('symlog')
    ax1.grid(True, alpha=0.3)
    
    # Create legend
    dd_patch = mpatches.Patch(color='blue', label='DD statistics')
    dz_patch = mpatches.Patch(color='green', label='Dz statistics')
    pi2_patch = mpatches.Patch(color='red', label='pi2 statistics')
    ax1.legend(handles=[dd_patch, dz_patch, pi2_patch], loc='upper left')
    
    # Relative error plot
    relative_errors = np.abs(gpu_values - moments_values) / (np.abs(moments_values) + 1e-10)
    
    # Group by statistic type
    dd_mask = [s.startswith('DD') for s in stat_names * n_bins]
    dz_mask = [s.startswith('Dz') for s in stat_names * n_bins]
    pi2_mask = [s.startswith('pi2') for s in stat_names * n_bins]
    
    x_pos = np.arange(len(stat_names))
    width = 0.25
    
    # Calculate mean relative errors by bin for each stat type
    for bin_idx in range(n_bins):
        start_idx = bin_idx * n_stats
        end_idx = (bin_idx + 1) * n_stats
        bin_errors = relative_errors[start_idx:end_idx]
        
        dd_errors = bin_errors[[i for i, name in enumerate(stat_names) if name.startswith('DD')]]
        dz_errors = bin_errors[[i for i, name in enumerate(stat_names) if name.startswith('Dz')]]
        pi2_errors = bin_errors[[i for i, name in enumerate(stat_names) if name.startswith('pi2')]]
        
        offset = (bin_idx - n_bins/2) * width / n_bins
        
        ax2.bar([0 + offset], [dd_errors.mean()], width/n_bins, label=f'Bin {bin_idx}' if bin_idx == 0 else '', color=f'C{bin_idx}')
        ax2.bar([1 + offset], [dz_errors.mean()], width/n_bins, color=f'C{bin_idx}')
        ax2.bar([2 + offset], [pi2_errors.mean()], width/n_bins, color=f'C{bin_idx}')
    
    ax2.set_ylabel('Mean Relative Error')
    ax2.set_title('Relative Error by Statistic Type and Distance Bin')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['DD', 'Dz', 'pi2'])
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'validation_correspondence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Overall correlation: {np.corrcoef(moments_values, gpu_values)[0, 1]:.6f}")
    print(f"  Mean relative error: {relative_errors.mean():.6f}")
    print(f"  Max relative error: {relative_errors.max():.6f}")
    print(f"  Median relative error: {np.median(relative_errors):.6f}")
    
    # Close plot
    plt.close()


def main():
    """Main execution function."""
    # Setup test data
    vcf_path, pop_file, bp_bins, pops = setup_test_data()
    
    # Run moments calculation
    moments_stats = run_moments_ld(vcf_path, pop_file, bp_bins, pops)
    
    # Setup GPU populations
    h_gpu = setup_gpu_populations(vcf_path)
    
    # Run GPU calculation
    gpu_stats = run_gpu_ld(h_gpu, bp_bins)
    
    # Compare results
    compare_results(moments_stats, gpu_stats)
    
    # Additional diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    
    # Check if we're computing the same number of variant pairs
    print("\nChecking variant counts and pairs:")
    vcf = allel.read_vcf(vcf_path)
    n_variants = vcf['variants/POS'].shape[0]
    n_pairs = n_variants * (n_variants - 1) // 2
    print(f"  Number of variants: {n_variants}")
    print(f"  Number of variant pairs: {n_pairs}")
    
    # Check heterozygosity statistics
    if len(moments_stats['sums']) > 5:
        print(f"\nMoments heterozygosity stats: {moments_stats['sums'][5]}")
    
    # Plot correspondence
    plot_correspondence(moments_stats, gpu_stats)


if __name__ == "__main__":
    main()