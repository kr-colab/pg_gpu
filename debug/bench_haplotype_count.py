"""Benchmark haplotype_count: before (CPU string hashing) vs after (GPU dot-product hashing)."""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu.diversity import haplotype_count, haplotype_diversity


def make_data(n_haps, n_snps, seed=42):
    rng = np.random.default_rng(seed)
    founders = rng.integers(0, 2, size=(5, n_snps), dtype=np.int8)
    assignments = rng.integers(0, 5, size=n_haps)
    haps = founders[assignments].copy()
    mutations = rng.random(size=(n_haps, n_snps)) < 0.02
    haps ^= mutations.astype(np.int8)
    positions = np.arange(n_snps) * 100
    hm = HaplotypeMatrix(haps, positions, positions[0], positions[-1])
    hm.transfer_to_gpu()
    return hm


def bench(fn, hm, n_reps=5):
    fn(hm)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        fn(hm)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    configs = [
        (100, 5000),
        (100, 10000),
        (100, 50000),
        (200, 10000),
        (200, 50000),
    ]

    print("Correctness check:")
    for n_haps, n_snps in [(50, 500), (100, 1000), (200, 5000)]:
        hm = make_data(n_haps, n_snps)
        count = haplotype_count(hm)
        print(f"  {n_haps} haps x {n_snps} snps: {count} unique haplotypes")

    print(f"\n{'n_haps':>7} {'n_snps':>8} | {'time (ms)':>10}")
    print("-" * 35)
    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)
        t = bench(haplotype_count, hm)
        print(f"{n_haps:>7} {n_snps:>8} | {t:>10.2f}")


if __name__ == "__main__":
    main()
