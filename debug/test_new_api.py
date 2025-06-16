#!/usr/bin/env python
"""
Test the new unified LD statistics API.
"""

import numpy as np
import cupy as cp
from pg_gpu import ld_statistics
from pg_gpu import stats_from_haplotype_counts_gpu_unified as old_stats

# Test single population
print("Testing single population...")
counts = cp.array([[10, 5, 3, 2], [8, 7, 6, 4], [15, 2, 8, 5]])
n_valid = cp.array([20, 25, 30])

# New API
dd_new = ld_statistics.dd(counts, n_valid=n_valid)
dz_new = ld_statistics.dz(counts, n_valid=n_valid)
pi2_new = ld_statistics.pi2(counts, n_valid=n_valid)

# Old API
dd_old = old_stats.DD(counts, n_valid)
dz_old = old_stats.Dz(counts, n_valid)
pi2_old = old_stats.pi2(counts, n_valid)

print(f"DD match: {cp.allclose(dd_new, dd_old)}")
print(f"Dz match: {cp.allclose(dz_new, dz_old)}")
print(f"pi2 match: {cp.allclose(pi2_new, pi2_old)}")

# Test two populations
print("\nTesting two populations...")
counts_2pop = cp.array([[10, 5, 3, 2, 8, 4, 2, 1], 
                        [12, 6, 4, 3, 9, 5, 3, 2]])
n_valid_2pop = (cp.array([20, 25]), cp.array([15, 19]))

# New API - DD between populations
dd_between_new = ld_statistics.dd(counts_2pop, populations=(0, 1), n_valid=n_valid_2pop)

# Old API
dd_between_old = old_stats.DD_two_pops(counts_2pop, 0, 1, n_valid_2pop[0], n_valid_2pop[1])

print(f"DD between match: {cp.allclose(dd_between_new, dd_between_old)}")

# Test Dz with different population configs
print("\nTesting Dz configurations...")
dz_001_new = ld_statistics.dz(counts_2pop, populations=(0, 0, 1), n_valid=n_valid_2pop)
dz_001_old = old_stats.Dz_two_pops(counts_2pop, (0, 0, 1), n_valid_2pop[0], n_valid_2pop[1])
print(f"Dz(0,0,1) match: {cp.allclose(dz_001_new, dz_001_old)}")

# Test pi2 with different population configs
print("\nTesting pi2 configurations...")
pi2_0011_new = ld_statistics.pi2(counts_2pop, populations=(0, 0, 1, 1), n_valid=n_valid_2pop)
pi2_0011_old = old_stats.pi2_two_pops(counts_2pop, (0, 0, 1, 1), n_valid_2pop[0], n_valid_2pop[1])
print(f"pi2(0,0,1,1) match: {cp.allclose(pi2_0011_new, pi2_0011_old)}")

# Test batch computation
print("\nTesting batch computation...")
results = ld_statistics.compute_ld_statistics(
    counts,
    statistics=['dd', 'dz', 'pi2'],
    n_valid=n_valid
)
print(f"Batch computation keys: {list(results.keys())}")
print(f"Batch DD match: {cp.allclose(results['dd'], dd_new)}")
print(f"Batch Dz match: {cp.allclose(results['dz'], dz_new)}")
print(f"Batch pi2 match: {cp.allclose(results['pi2'], pi2_new)}")

# Test with missing data
print("\nTesting with missing data...")
counts_missing = cp.array([[10, -1, 3, 2], [8, 7, -1, 4], [15, 2, 8, -1]])
n_valid_missing = cp.array([15, 19, 25])

dd_missing = ld_statistics.dd(counts_missing, n_valid=n_valid_missing)
print(f"DD with missing data computed: {dd_missing}")

print("\nAll tests completed!")