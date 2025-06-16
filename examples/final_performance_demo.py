#!/usr/bin/env python3
"""
Final performance demonstration after fixes.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import allel
from pg_gpu import HaplotypeMatrix, diversity, ld_statistics

def create_test_data(n_haplotypes=100, n_variants=5000, seed=42):
    """Create realistic test data."""
    np.random.seed(seed)
    
    # Create realistic allele frequencies
    allele_freqs = np.random.beta(0.5, 2.0, n_variants)  
    haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
    
    for i in range(n_variants):
        n_derived = int(allele_freqs[i] * n_haplotypes)
        if n_derived > 0:
            derived_indices = np.random.choice(n_haplotypes, n_derived, replace=False)
            haplotypes[derived_indices, i] = 1
    
    positions = np.arange(n_variants) * 100 + 10000
    return haplotypes, positions

def test_ld_performance(haplotypes, positions, label):
    """Test LD statistics performance between scikit-allel and pg_gpu."""
    
    # For LD testing, use smaller subset to avoid excessive computation time
    if haplotypes.shape[1] > 2000:
        # Use first 2000 variants for LD testing to keep it reasonable
        haplotypes_ld = haplotypes[:, :2000]
        positions_ld = positions[:2000]
        print(f"  [LD testing using first 2000 variants for performance]")
    else:
        haplotypes_ld = haplotypes
        positions_ld = positions
    
    n_variants_ld = haplotypes_ld.shape[1]
    n_pairs = n_variants_ld * (n_variants_ld - 1) // 2
    
    # Test scikit-allel LD (Rogers-Huff r)
    print(f"  Testing scikit-allel LD ({n_pairs:,} pairs)...")
    start = time.time()
    
    # Convert to diploid genotypes for scikit-allel
    # Combine pairs of haplotypes into diploid genotypes
    n_samples = haplotypes_ld.shape[0] // 2
    genotypes = np.zeros((n_variants_ld, n_samples), dtype=np.int8)
    for i in range(n_samples):
        genotypes[:, i] = haplotypes_ld[2*i, :] + haplotypes_ld[2*i+1, :]
    
    # Compute r values
    r_values = allel.rogers_huff_r(genotypes)
    r_squared_mean = np.mean(r_values ** 2)
    allel_ld_time = time.time() - start
    
    # Test pg_gpu LD
    print(f"  Testing pg_gpu LD...")
    start = time.time()
    
    h = HaplotypeMatrix(haplotypes_ld, positions_ld)
    h.transfer_to_gpu()
    
    # Compute haplotype counts for LD statistics
    counts = h.tally_gpu_haplotypes()
    if isinstance(counts, tuple):
        counts, n_valid = counts
    else:
        n_valid = None
    
    # Compute DD statistic (similar to r^2)
    dd_values = ld_statistics.dd(counts, n_valid=n_valid)
    dd_mean = float(dd_values.mean().get() if hasattr(dd_values.mean(), 'get') else dd_values.mean())
    gpu_ld_time = time.time() - start
    
    print(f"    scikit-allel r² mean: {r_squared_mean:.6f} in {allel_ld_time:.3f}s")
    print(f"    pg_gpu DD mean: {dd_mean:.6f} in {gpu_ld_time:.3f}s")
    
    ld_speedup = allel_ld_time / gpu_ld_time if gpu_ld_time > 0 else float('inf')
    if ld_speedup > 1:
        print(f"    🚀 pg_gpu LD is {ld_speedup:.1f}x faster!")
    elif ld_speedup > 0.5:
        print(f"    ⚡ pg_gpu LD is competitive ({ld_speedup:.1f}x)")
    else:
        print(f"    ⚠️  GPU overhead dominates ({ld_speedup:.1f}x)")
    
    return allel_ld_time, gpu_ld_time, ld_speedup

def main():
    print("Final Performance Demo: pg_gpu vs scikit-allel")
    print("=" * 60)
    
    # Test different sizes
    sizes = [
        (50, 1000, "Small"),
        (100, 5000, "Medium"),
        (200, 10000, "Large"),
        (500, 25000, "Very Large"),
        (1000, 50000, "Huge"),
        (2000, 100000, "Massive"),
    ]
    
    for n_haplotypes, n_variants, label in sizes:
        print(f"\n{label} dataset: {n_haplotypes} haplotypes, {n_variants:,} variants")
        print("-" * 60)
        
        # Create data
        haplotypes, positions = create_test_data(n_haplotypes, n_variants)
        
        # Test scikit-allel
        start = time.time()
        h_allel = allel.HaplotypeArray(haplotypes.T)
        ac = h_allel.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac)
        theta_allel = allel.watterson_theta(positions, ac)
        tajd_allel = allel.tajima_d(ac, pos=positions)
        allel_time = time.time() - start
        
        # Test pg_gpu (now with fast defaults)
        h = HaplotypeMatrix(haplotypes, positions)
        h.transfer_to_gpu()
        
        start = time.time()
        pi_gpu = diversity.pi(h, span_normalize=True)
        theta_gpu = diversity.theta_w(h, span_normalize=True)
        tajd_gpu = diversity.tajimas_d(h)
        gpu_time = time.time() - start
        
        # Results
        print(f"Results validation:")
        print(f"  Pi: {pi_allel:.6f} vs {pi_gpu:.6f} (diff: {abs(pi_allel-pi_gpu):.8f})")
        print(f"  Theta: {theta_allel:.6f} vs {theta_gpu:.6f} (diff: {abs(theta_allel-theta_gpu):.8f})")
        print(f"  Tajima's D: {tajd_allel:.6f} vs {tajd_gpu:.6f} (diff: {abs(tajd_allel-tajd_gpu):.8f})")
        
        # Performance
        speedup = allel_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"\nDiversity Performance:")
        print(f"  scikit-allel: {allel_time:.3f}s")
        print(f"  pg_gpu: {gpu_time:.3f}s")
        
        if speedup > 1:
            print(f"  🚀 pg_gpu is {speedup:.1f}x faster!")
        elif speedup > 0.5:
            print(f"  ⚡ pg_gpu is competitive ({speedup:.1f}x)")
        else:
            print(f"  ⚠️  GPU overhead dominates ({speedup:.1f}x)")
        
        # Test LD performance for datasets that aren't too large
        if n_variants <= 25000:  # Skip LD for very large datasets to save time
            print(f"\nLD Performance:")
            try:
                allel_ld_time, gpu_ld_time, ld_speedup = test_ld_performance(haplotypes, positions, label)
            except Exception as e:
                print(f"  LD testing failed: {e}")
        else:
            print(f"\nLD Performance: Skipped for {label} dataset (too large for demo)")

if __name__ == '__main__':
    main()