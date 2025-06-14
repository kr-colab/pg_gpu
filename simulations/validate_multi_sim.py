#!/usr/bin/env python3
"""
Validate two-population LD statistics between pg_gpu and moments packages
using multiple simulations with different random seeds.

This script:
1. Runs 3 simulations in parallel with different seeds
2. Computes LD statistics using both moments and pg_gpu
3. Creates a figure with 3 scatter plots comparing the results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moments'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

# Import the required modules
import moments.LD.Parsing as mParsing
from pg_gpu.haplotype_matrix import HaplotypeMatrix
import allel
import msprime
import demes


def simulate_im_model(seed, output_dir):
    """
    Simulate IM model with given seed.
    Returns paths to VCF and population files.
    """
    print(f"Simulating with seed {seed}...")
    
    # Simulation parameters (matching simulate_im_vcf.py)
    L = 1e6
    u = r = 1.5e-8
    n = 10
    
    # Load demographic model
    g = demes.load("data/demes_mod.yaml")
    demog = msprime.Demography.from_demes(g)
    
    # Run simulation
    trees = msprime.sim_ancestry(
        {"deme0": n, "deme1": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        random_seed=seed,
    )
    
    trees = msprime.sim_mutations(trees, rate=u, random_seed=seed + 1000)
    
    # Write VCF
    vcf_path = os.path.join(output_dir, f"sim_seed{seed}.vcf")
    with open(vcf_path, "w") as fout:
        trees.write_vcf(fout)
    
    # Create population file
    pop_file = os.path.join(output_dir, f"sim_seed{seed}_pops.txt")
    with open(pop_file, "w") as f:
        f.write("sample\tpop\n")
        for i in range(n):
            f.write(f"tsk_{i}\tdeme0\n")
        for i in range(n, 2*n):
            f.write(f"tsk_{i}\tdeme1\n")
    
    print(f"Simulation with seed {seed} complete")
    return vcf_path, pop_file, trees


def run_validation_for_seed(seed, output_dir, bp_bins, use_cache=True):
    """
    Run validation for a single simulation.
    Returns moments and GPU statistics.
    """
    # Run simulation
    vcf_path, pop_file, trees = simulate_im_model(seed, output_dir)
    
    # Define populations
    pops = ["deme0", "deme1"]
    
    # Cache setup
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = f"seed{seed}_{len(bp_bins)}bins"
    cache_file = cache_dir / f"moments_ld_{cache_key}.pkl"
    
    # Run moments LD calculation
    if use_cache and cache_file.exists():
        print(f"Loading cached moments results for seed {seed}")
        with open(cache_file, 'rb') as f:
            moments_stats = pickle.load(f)
    else:
        print(f"Computing moments LD for seed {seed}...")
        moments_stats = mParsing.compute_ld_statistics(
            vcf_path,
            pop_file=pop_file,
            pops=pops,
            bp_bins=bp_bins,
            use_genotypes=False,
            report=False
        )
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(moments_stats, f)
    
    # Run GPU LD calculation
    print(f"Computing GPU LD for seed {seed}...")
    h_gpu = HaplotypeMatrix.from_vcf(vcf_path)
    
    # Set up population assignments
    vcf = allel.read_vcf(vcf_path)
    n_samples = vcf['samples'].shape[0]
    
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
    
    # Compute GPU statistics
    gpu_stats = h_gpu.compute_ld_statistics_gpu_two_pops(
        bp_bins=bp_bins,
        pop1="deme0",
        pop2="deme1",
        missing=False,
        raw=True
    )
    
    return seed, moments_stats, gpu_stats


def extract_statistics_for_plotting(moments_stats, gpu_stats):
    """
    Extract statistics from moments and GPU results for plotting.
    Returns two arrays of corresponding values.
    """
    stat_names = moments_stats['stats'][0]
    
    moments_values = []
    gpu_values = []
    
    for bin_range, moments_sums in zip(moments_stats['bins'], moments_stats['sums']):
        gpu_bin = gpu_stats[bin_range]
        
        for stat_name, mom_val in zip(stat_names, moments_sums):
            gpu_val = gpu_bin[stat_name]
            moments_values.append(mom_val)
            gpu_values.append(gpu_val)
    
    return np.array(moments_values), np.array(gpu_values)


def create_comparison_figure(results, output_file):
    """
    Create figure with 3 scatter plots comparing moments vs GPU statistics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color map for different statistics
    color_map = {
        'DD': 'blue',
        'Dz': 'green',
        'pi2': 'red'
    }
    
    for idx, (seed, moments_stats, gpu_stats) in enumerate(results):
        ax = axes[idx]
        
        # Extract values
        moments_vals, gpu_vals = extract_statistics_for_plotting(moments_stats, gpu_stats)
        
        # Get statistic names for coloring
        stat_names = moments_stats['stats'][0]
        n_stats = len(stat_names)
        n_bins = len(moments_stats['bins'])
        
        # Create colors array
        colors = []
        for bin_idx in range(n_bins):
            for stat_name in stat_names:
                if stat_name.startswith('DD'):
                    colors.append(color_map['DD'])
                elif stat_name.startswith('Dz'):
                    colors.append(color_map['Dz'])
                else:  # pi2
                    colors.append(color_map['pi2'])
        
        # Create scatter plot
        ax.scatter(moments_vals, gpu_vals, c=colors, alpha=0.6, s=30)
        
        # Add diagonal line
        min_val = min(moments_vals.min(), gpu_vals.min())
        max_val = max(moments_vals.max(), gpu_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Set scales and labels
        ax.set_xscale('symlog')
        ax.set_yscale('symlog')
        ax.set_xlabel('Moments Statistics')
        if idx == 0:
            ax.set_ylabel('GPU Statistics')
        ax.set_title(f'Seed {seed}')
        ax.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        correlation = np.corrcoef(moments_vals, gpu_vals)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend to the last subplot
    dd_patch = mpatches.Patch(color='blue', label='DD statistics')
    dz_patch = mpatches.Patch(color='green', label='Dz statistics')
    pi2_patch = mpatches.Patch(color='red', label='π₂ statistics')
    axes[-1].legend(handles=[dd_patch, dz_patch, pi2_patch], 
                    loc='lower right', framealpha=0.9)
    
    plt.suptitle('Correspondence between Moments and GPU LD Statistics\nIM Model with Different Random Seeds', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_file}")
    plt.close()


def main():
    """Main execution function."""
    # Create output directory
    output_dir = "simulations/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameters
    seeds = [12345, 67890, 11111]  # Three different random seeds
    bp_bins = np.logspace(2, 6, 6)  # [100, 631, 3981, 25119, 158489, 1000000]
    
    print("=" * 60)
    print("MULTI-SIMULATION VALIDATION")
    print("=" * 60)
    print(f"Running {len(seeds)} simulations with different random seeds")
    print(f"Seeds: {seeds}")
    print(f"Distance bins: {bp_bins}")
    
    # Run validations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(run_validation_for_seed, seed, output_dir, bp_bins): seed 
            for seed in seeds
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed validation for seed {seed}")
            except Exception as e:
                print(f"Validation for seed {seed} failed with error: {e}")
    
    # Sort results by seed to maintain consistent ordering
    results.sort(key=lambda x: x[0])
    
    # Create comparison figure
    output_file = "simulations/multi_sim_validation.png"
    create_comparison_figure(results, output_file)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for seed, moments_stats, gpu_stats in results:
        moments_vals, gpu_vals = extract_statistics_for_plotting(moments_stats, gpu_stats)
        
        # Calculate errors
        relative_errors = np.abs(gpu_vals - moments_vals) / (np.abs(moments_vals) + 1e-10)
        
        print(f"\nSeed {seed}:")
        print(f"  Overall correlation: {np.corrcoef(moments_vals, gpu_vals)[0, 1]:.6f}")
        print(f"  Mean relative error: {relative_errors.mean():.6f}")
        print(f"  Max relative error: {relative_errors.max():.6f}")
        print(f"  Median relative error: {np.median(relative_errors):.6f}")
    
    print("\nValidation complete!")


if __name__ == "__main__":
    main()