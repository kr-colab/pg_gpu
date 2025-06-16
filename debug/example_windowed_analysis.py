#!/usr/bin/env python
"""
Example usage of the windowed_analysis module.
"""

import numpy as np
import pandas as pd
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import WindowedAnalyzer, windowed_analysis


def main():
    print("=== Windowed Analysis Example ===\n")
    
    # 1. Create synthetic data for demonstration
    print("1. Creating synthetic haplotype data...")
    n_variants = 10000
    n_samples = 100
    
    # Create synthetic haplotypes with some structure
    np.random.seed(42)
    haplotypes = np.random.binomial(1, 0.3, size=(n_samples, n_variants))
    
    # Create positions (1 variant per 1000 bp on average)
    positions = np.sort(np.random.randint(1, 10_000_000, size=n_variants))
    
    # Create HaplotypeMatrix
    hap_matrix = HaplotypeMatrix(
        haplotypes, 
        positions,
        chrom_start=positions[0],
        chrom_end=positions[-1]
    )
    
    # Define populations
    hap_matrix.sample_sets = {
        'pop1': list(range(0, 50)),
        'pop2': list(range(50, 100))
    }
    
    print(f"Created matrix with {n_variants} variants and {n_samples} samples")
    print(f"Genomic span: {positions[0]:,} - {positions[-1]:,} bp\n")
    
    # 2. Simple windowed analysis
    print("2. Running simple windowed analysis (100kb windows, 50kb step)...")
    results = windowed_analysis(
        hap_matrix,
        window_size=100_000,
        step_size=50_000,
        statistics=['pi', 'tajimas_d', 'n_variants'],
        populations=['pop1', 'pop2']
    )
    
    print(f"Computed statistics for {len(results)} windows")
    print("\nFirst 5 windows:")
    print(results.head())
    
    # 3. Advanced analysis with custom configuration
    print("\n3. Advanced analysis with FST and LD decay...")
    analyzer = WindowedAnalyzer(
        window_type='bp',
        window_size=200_000,
        step_size=100_000,
        statistics=['pi', 'fst', 'ld_decay'],
        populations=['pop1', 'pop2'],
        ld_bins=[0, 1000, 5000, 10000, 50000, 100000],
        progress_bar=True
    )
    
    adv_results = analyzer.compute(hap_matrix)
    
    # Show FST results
    print(f"\nWindows with highest FST:")
    fst_cols = [col for col in adv_results.columns if 'fst' in col]
    if fst_cols:
        top_fst = adv_results.nlargest(3, fst_cols[0])[['start', 'end'] + fst_cols]
        print(top_fst)
    
    # 4. SNP-based windows
    print("\n4. Using SNP-based windows (100 SNPs per window)...")
    snp_analyzer = WindowedAnalyzer(
        window_type='snp',
        window_size=100,
        step_size=50,
        statistics=['pi', 'n_singletons'],
        populations=['pop1', 'pop2']
    )
    
    snp_results = snp_analyzer.compute(hap_matrix)
    print(f"Computed {len(snp_results)} SNP-based windows")
    print("\nFirst 3 windows:")
    print(snp_results.head(3))
    
    # 5. Custom regions
    print("\n5. Analyzing custom regions...")
    # Define some "genes" as regions
    regions = pd.DataFrame({
        'chrom': [1, 1, 1],
        'start': [1_000_000, 3_000_000, 5_000_000],
        'end': [1_500_000, 3_200_000, 5_800_000],
        'name': ['gene1', 'gene2', 'gene3']
    })
    
    region_analyzer = WindowedAnalyzer(
        window_type='regions',
        regions=regions,
        statistics=['pi', 'tajimas_d'],
        populations=['pop1', 'pop2']
    )
    
    region_results = region_analyzer.compute(hap_matrix)
    print(f"Analyzed {len(region_results)} regions")
    print(region_results)
    
    # 6. Streaming for large datasets
    print("\n6. Demonstrating streaming computation...")
    stream_analyzer = WindowedAnalyzer(
        window_size=500_000,
        statistics=['pi'],
        progress_bar=False
    )
    
    batch_num = 0
    for batch in stream_analyzer.compute_streaming(hap_matrix, batch_size=5):
        batch_num += 1
        print(f"Batch {batch_num}: {len(batch)} windows, mean pi = {batch['pi'].mean():.4f}")
    
    # 7. Custom statistic
    print("\n7. Using custom statistics...")
    
    def segregating_sites_ratio(window, min_freq=0.05):
        """Custom statistic: ratio of common variants (MAF > min_freq)."""
        matrix = window.matrix
        if matrix.device == 'GPU':
            import cupy as cp
            af = cp.sum(matrix.haplotypes, axis=0) / matrix.num_haplotypes
            maf = cp.minimum(af, 1 - af)
            ratio = cp.sum(maf > min_freq) / matrix.num_variants
            return float(ratio.get())
        else:
            af = np.sum(matrix.haplotypes, axis=0) / matrix.num_haplotypes
            maf = np.minimum(af, 1 - af)
            return np.sum(maf > min_freq) / matrix.num_variants
    
    custom_analyzer = WindowedAnalyzer(
        window_size=100_000,
        statistics=['pi', segregating_sites_ratio],
        custom_stat_kwargs={'segregating_sites_ratio': {'min_freq': 0.1}}
    )
    
    custom_results = custom_analyzer.compute(hap_matrix)
    print(f"Windows with high proportion of common variants:")
    print(custom_results.nlargest(3, 'segregating_sites_ratio')[
        ['start', 'end', 'pi', 'segregating_sites_ratio']
    ])
    
    # 8. Save results
    print("\n8. Saving results...")
    output_file = 'windowed_stats_example.csv'
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()