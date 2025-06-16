#!/usr/bin/env python
"""Debug diversity calculations."""

import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix, diversity

# Test case: all samples identical (all 1s)
n_variants = 50
n_samples = 20
haplotypes = np.ones((n_samples, n_variants), dtype=int)
positions = np.arange(n_variants) * 1000

matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
matrix.transfer_to_gpu()

# Check AFS
# For n_samples=20, we expect bins for 0,1,2,...,20 (21 bins total)
expected_bins = cp.arange(n_samples + 1)
print(f"Expected histogram bins: {expected_bins.get()}")

# Count alleles manually
allele_counts = cp.sum(matrix.haplotypes, axis=0)
print(f"Allele counts at each site (first 5): {allele_counts.get()[:5]}")
print(f"All counts are {allele_counts[0].get()} because all sites fixed for 1")

afs = diversity.allele_frequency_spectrum(matrix)
print("\nAFS:", afs.get() if hasattr(afs, 'get') else afs)
print("AFS shape:", afs.shape)
print(f"AFS has {len(afs)} bins for {n_samples} haplotypes")

# The issue: all sites have frequency n_samples (20)
# So afs[20] = 50 (all variants fixed for allele 1)

# Let's manually compute pi to debug
i = cp.arange(1, n_samples, dtype=cp.float64)
weight = (2 * i * (n_samples - i)) / (n_samples * (n_samples - 1))
print(f"Weight array shape: {weight.shape}")
print(f"Weight array: {weight.get()[:5]}... (first 5)")

# The issue is we're taking afs[1:n_haplotypes] which is afs[1:20]
# But afs has 21 elements (0 to 20)
# afs[20] contains the fixed sites count
print(f"\nafs[1:n_samples] = {afs[1:n_samples].get()}")
print(f"Last element afs[{n_samples}] = {afs[n_samples].get()}")

# So the multiplication is:
# weight * afs[1:20] where afs[19] = 0 but afs[20] = 50
# We need to make sure we're not including the last bin

# Check pi calculation
pi_value = diversity.pi(matrix, span_normalize=False)
print(f"\nPi value: {pi_value}")

# For fixed sites, pi should be 0
# Let's check with all 0s
haplotypes_zeros = np.zeros((n_samples, n_variants), dtype=int)
matrix_zeros = HaplotypeMatrix(haplotypes_zeros, positions, 0, positions[-1])
pi_zeros = diversity.pi(matrix_zeros, span_normalize=False)
print(f"Pi for all zeros: {pi_zeros}")