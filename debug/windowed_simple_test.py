#!/usr/bin/env python
"""Simple windowed test to check span normalization."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis

# Create simple test data
np.random.seed(42)
haplotypes = np.random.choice([0, 1], size=(20, 100), p=[0.7, 0.3])
positions = np.arange(100) * 1000 + 1000
start, end = positions[0], positions[-1]

window_size = 25000  # 25kb windows

# pg_gpu windowed analysis
matrix = HaplotypeMatrix(haplotypes, positions, start, end)
pg_results = windowed_analysis(
    matrix,
    window_size=window_size,
    statistics=['pi'],
    progress_bar=False
)

# scikit-allel windowed diversity
h = allel.HaplotypeArray(haplotypes.T)
ac = h.count_alleles()
allel_pi, allel_windows, allel_n_bases, allel_counts = allel.windowed_diversity(
    positions, ac, size=window_size, start=start, stop=end
)

print(f"Window size: {window_size}")
print(f"pg_gpu pi values: {pg_results['pi'].tolist()}")
print(f"allel pi values: {allel_pi}")

print("\nComparison:")
for i in range(len(pg_results)):
    pg_val = pg_results['pi'].iloc[i]
    allel_val = allel_pi[i]
    print(f"Window {i}: pg_gpu={pg_val:.8f}, allel={allel_val:.8f}, diff={abs(pg_val - allel_val):.8f}")