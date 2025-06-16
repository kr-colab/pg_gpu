#!/usr/bin/env python
"""Debug windowed analysis comparison."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis

# Create test data
np.random.seed(42)
n_haplotypes = 50
n_variants = 1000

haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)

# Add variation with different patterns across the genome
for i in range(0, n_variants, 100):
    end_i = min(i + 100, n_variants)
    # Vary allele frequency across regions
    p = 0.1 + 0.4 * (i / n_variants)  # Frequency gradient
    haplotypes[:, i:end_i] = np.random.choice([0, 1], size=(n_haplotypes, end_i - i), p=[1-p, p])

# Positions with 1kb spacing
positions = np.arange(n_variants) * 1000 + 10000
start, end = positions[0], positions[-1]

window_size = 50000  # 50kb windows

print(f"Data: {n_haplotypes} haplotypes, {n_variants} variants")
print(f"Positions: {start} to {end}")
print(f"Total span: {end - start + 1}")
print(f"Window size: {window_size}")

# pg_gpu windowed analysis
matrix = HaplotypeMatrix(haplotypes, positions, start, end)
pg_results = windowed_analysis(
    matrix,
    window_size=window_size,
    statistics=['pi'],
    progress_bar=False
)

print(f"\npg_gpu results:")
print(f"  Number of windows: {len(pg_results)}")
print(f"  Pi values: {pg_results['pi'].tolist()}")
print(f"  Pi range: {pg_results['pi'].min():.6f} to {pg_results['pi'].max():.6f}")

# scikit-allel windowed diversity
h = allel.HaplotypeArray(haplotypes.T)
ac = h.count_alleles()
allel_pi, allel_windows, allel_n_bases, allel_counts = allel.windowed_diversity(
    positions, ac, size=window_size, start=start, stop=end
)

print(f"\nscikit-allel results:")
print(f"  Number of windows: {len(allel_pi)}")
print(f"  Pi values: {allel_pi}")
print(f"  Pi range: {np.nanmin(allel_pi):.6f} to {np.nanmax(allel_pi):.6f}")

print(f"\nWindow details:")
for i in range(min(5, len(pg_results), len(allel_pi))):
    pg_val = pg_results['pi'].iloc[i]
    allel_val = allel_pi[i]
    print(f"  Window {i}: pg_gpu={pg_val:.6f}, allel={allel_val:.6f}, ratio={pg_val/allel_val:.1f}")

# Test single window calculation to understand the difference
single_window_start = allel_windows[0][0]
single_window_end = allel_windows[0][1]
print(f"\nSingle window analysis:")
print(f"  Window: {single_window_start} to {single_window_end}")

# Find variants in this window
in_window = (positions >= single_window_start) & (positions <= single_window_end)
window_positions = positions[in_window]
window_haplotypes = haplotypes[:, in_window]
n_vars_in_window = np.sum(in_window)

print(f"  Variants in window: {n_vars_in_window}")

if n_vars_in_window > 0:
    # Manual calculation
    window_matrix = HaplotypeMatrix(window_haplotypes, window_positions, 
                                  single_window_start, single_window_end)
    
    # Raw diversity (no span normalization)
    from pg_gpu import diversity
    pi_raw = diversity.pi(window_matrix, span_normalize=False)
    pi_span = diversity.pi(window_matrix, span_normalize=True)
    
    print(f"  pg_gpu pi (raw): {pi_raw:.6f}")
    print(f"  pg_gpu pi (span): {pi_span:.6f}")
    
    # scikit-allel for same window
    h_window = allel.HaplotypeArray(window_haplotypes.T)
    ac_window = h_window.count_alleles()
    allel_pi_window = allel.sequence_diversity(window_positions, ac_window, 
                                             start=single_window_start, 
                                             stop=single_window_end)
    print(f"  allel pi: {allel_pi_window:.6f}")
    
    # Check span calculation
    window_span = single_window_end - single_window_start + 1
    print(f"  Window span: {window_span}")
    print(f"  Number of sites: {n_vars_in_window}")
    print(f"  Raw * sites / span: {pi_raw * n_vars_in_window / window_span:.6f}")