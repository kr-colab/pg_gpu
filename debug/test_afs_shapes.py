#!/usr/bin/env python
"""Debug AFS shapes between pg_gpu and scikit-allel."""

import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity

# Create simple test data
np.random.seed(42)
n_haplotypes = 50
n_variants = 100

haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)

# Different frequency classes
haplotypes[:5, :20] = 1    # 20 sites with 5/50 = 10% frequency
haplotypes[:15, 20:40] = 1 # 20 sites with 15/50 = 30% frequency  
haplotypes[:25, 40:60] = 1 # 20 sites with 25/50 = 50% frequency
haplotypes[:40, 60:80] = 1 # 20 sites with 40/50 = 80% frequency

# 20 singletons
for i in range(20):
    haplotypes[i, 80 + i] = 1

positions = np.arange(n_variants) * 1000 + 1000

print(f"Haplotype shape: {haplotypes.shape}")
print(f"n_haplotypes: {n_haplotypes}")

# pg_gpu
matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
afs_pg = diversity.allele_frequency_spectrum(matrix)
if hasattr(afs_pg, 'get'):
    afs_pg = afs_pg.get()

print(f"\npg_gpu AFS:")
print(f"  Shape: {afs_pg.shape}")
print(f"  AFS: {afs_pg}")
print(f"  Sum: {np.sum(afs_pg)}")

# scikit-allel
h = allel.HaplotypeArray(haplotypes.T)
ac = h.count_alleles()
sfs_allel = allel.sfs(ac[:, 1])

print(f"\nscikit-allel SFS:")
print(f"  Shape: {sfs_allel.shape}")
print(f"  SFS: {sfs_allel}")
print(f"  Sum: {np.sum(sfs_allel)}")

print(f"\nComparing:")
print(f"  pg_gpu[0]: {afs_pg[0]} (sites fixed for ancestral)")
print(f"  pg_gpu[50]: {afs_pg[50]} (sites fixed for derived)")
print(f"  allel doesn't have fixed sites in SFS")

print(f"\n  pg_gpu[1:50]: {afs_pg[1:50]}")
print(f"  allel[1:]: {sfs_allel[1:]}")
print(f"  Shapes: {afs_pg[1:50].shape} vs {sfs_allel[1:].shape}")

# Check ac shape
print(f"\nAllele counts shape: {ac.shape}")
print(f"ac[:, 1] shape: {ac[:, 1].shape}")
print(f"Unique values in ac[:, 1]: {np.unique(ac[:, 1])}")