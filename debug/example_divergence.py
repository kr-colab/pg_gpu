#!/usr/bin/env python
"""
Example usage of the divergence module for population genetics analysis.
"""

import numpy as np
from pg_gpu import HaplotypeMatrix, divergence


def main():
    print("=== Divergence Statistics Example ===\n")
    
    # 1. Create synthetic data with population structure
    print("1. Creating synthetic population data...")
    n_variants = 1000
    n_samples = 120
    
    # Create three populations with different allele frequencies
    haplotypes = np.zeros((n_samples, n_variants), dtype=int)
    
    # Population 1 (samples 0-39): Low frequency of alternate allele
    haplotypes[:40, :] = np.random.choice([0, 1], size=(40, n_variants), p=[0.8, 0.2])
    
    # Population 2 (samples 40-79): High frequency of alternate allele
    haplotypes[40:80, :] = np.random.choice([0, 1], size=(40, n_variants), p=[0.3, 0.7])
    
    # Population 3 (samples 80-119): Intermediate frequency
    haplotypes[80:, :] = np.random.choice([0, 1], size=(40, n_variants), p=[0.5, 0.5])
    
    # Create positions
    positions = np.arange(n_variants) * 1000
    
    # Create HaplotypeMatrix
    matrix = HaplotypeMatrix(haplotypes, positions)
    matrix.sample_sets = {
        'Europe': list(range(40)),
        'Africa': list(range(40, 80)),
        'Asia': list(range(80, 120))
    }
    
    print(f"Created {n_variants} variants across {n_samples} samples")
    print(f"Populations: Europe (n=40), Africa (n=40), Asia (n=40)\n")
    
    # 2. Calculate FST between populations
    print("2. Calculating FST between populations...")
    
    # Europe vs Africa (should be high)
    fst_eur_afr = divergence.fst(matrix, 'Europe', 'Africa', method='hudson')
    print(f"FST (Europe-Africa): {fst_eur_afr:.4f}")
    
    # Europe vs Asia (should be moderate)
    fst_eur_asia = divergence.fst(matrix, 'Europe', 'Asia', method='hudson')
    print(f"FST (Europe-Asia): {fst_eur_asia:.4f}")
    
    # Africa vs Asia (should be moderate)
    fst_afr_asia = divergence.fst(matrix, 'Africa', 'Asia', method='hudson')
    print(f"FST (Africa-Asia): {fst_afr_asia:.4f}")
    
    # 3. Compare different FST estimators
    print("\n3. Comparing FST estimators (Europe vs Africa)...")
    fst_hudson = divergence.fst_hudson(matrix, 'Europe', 'Africa')
    fst_wc = divergence.fst_weir_cockerham(matrix, 'Europe', 'Africa')
    fst_nei = divergence.fst_nei(matrix, 'Europe', 'Africa')
    
    print(f"Hudson's FST: {fst_hudson:.4f}")
    print(f"Weir & Cockerham's FST: {fst_wc:.4f}")
    print(f"Nei's GST: {fst_nei:.4f}")
    
    # 4. Calculate absolute divergence (Dxy)
    print("\n4. Calculating absolute divergence (Dxy)...")
    dxy_eur_afr = divergence.dxy(matrix, 'Europe', 'Africa')
    dxy_eur_asia = divergence.dxy(matrix, 'Europe', 'Asia')
    dxy_afr_asia = divergence.dxy(matrix, 'Africa', 'Asia')
    
    print(f"Dxy (Europe-Africa): {dxy_eur_afr:.4f}")
    print(f"Dxy (Europe-Asia): {dxy_eur_asia:.4f}")
    print(f"Dxy (Africa-Asia): {dxy_afr_asia:.4f}")
    
    # 5. Calculate net divergence (Da)
    print("\n5. Calculating net divergence (Da)...")
    da_eur_afr = divergence.da(matrix, 'Europe', 'Africa')
    da_eur_asia = divergence.da(matrix, 'Europe', 'Asia')
    da_afr_asia = divergence.da(matrix, 'Africa', 'Asia')
    
    print(f"Da (Europe-Africa): {da_eur_afr:.4f}")
    print(f"Da (Europe-Asia): {da_eur_asia:.4f}")
    print(f"Da (Africa-Asia): {da_afr_asia:.4f}")
    
    # 6. Within-population diversity
    print("\n6. Calculating within-population diversity (π)...")
    pi_eur = divergence.pi_within_population(matrix, 'Europe')
    pi_afr = divergence.pi_within_population(matrix, 'Africa')
    pi_asia = divergence.pi_within_population(matrix, 'Asia')
    
    print(f"π (Europe): {pi_eur:.4f}")
    print(f"π (Africa): {pi_afr:.4f}")
    print(f"π (Asia): {pi_asia:.4f}")
    
    # 7. All statistics at once
    print("\n7. Computing all divergence statistics at once...")
    all_stats = divergence.divergence_stats(
        matrix, 'Europe', 'Africa',
        statistics=['fst', 'fst_wc', 'dxy', 'da', 'pi1', 'pi2']
    )
    
    print("Europe vs Africa summary:")
    for stat, value in all_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    # 8. Pairwise FST matrix
    print("\n8. Computing pairwise FST matrix...")
    fst_matrix, pop_names = divergence.pairwise_fst(matrix)
    
    # Convert to numpy if on GPU
    if hasattr(fst_matrix, 'get'):
        fst_matrix = fst_matrix.get()
    
    print("\nPairwise FST matrix:")
    print("       ", "  ".join(f"{pop:>8}" for pop in pop_names))
    for i, pop1 in enumerate(pop_names):
        row = f"{pop1:>8}"
        for j, pop2 in enumerate(pop_names):
            row += f"  {fst_matrix[i, j]:8.4f}"
        print(row)
    
    # 9. GPU acceleration (if available)
    try:
        import cupy as cp
        if cp.cuda.is_available():
            print("\n9. Testing GPU acceleration...")
            matrix.transfer_to_gpu()
            
            import time
            
            # Time CPU calculation
            matrix.transfer_to_cpu()
            start = time.time()
            fst_cpu = divergence.fst(matrix, 'Europe', 'Africa')
            cpu_time = time.time() - start
            
            # Time GPU calculation
            matrix.transfer_to_gpu()
            start = time.time()
            fst_gpu = divergence.fst(matrix, 'Europe', 'Africa')
            gpu_time = time.time() - start
            
            print(f"CPU time: {cpu_time*1000:.2f} ms")
            print(f"GPU time: {gpu_time*1000:.2f} ms")
            print(f"Speedup: {cpu_time/gpu_time:.1f}x")
            print(f"Results match: {abs(fst_cpu - fst_gpu) < 1e-6}")
    except ImportError:
        print("\n9. GPU not available, skipping acceleration test")
    
    print("\n=== Analysis Complete ===")


if __name__ == '__main__':
    main()