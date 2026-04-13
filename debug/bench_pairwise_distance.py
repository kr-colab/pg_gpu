"""Benchmark pairwise_distance: Gram matrix trick vs batched pairwise."""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import pairwise_distance


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


def bench(fn, n_reps=5):
    fn()
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    # Correctness: compare euclidean distances with scipy reference
    print("Correctness check:")
    from scipy.spatial.distance import pdist
    for n_haps, n_snps in [(20, 200), (100, 2000)]:
        hm = make_data(n_haps, n_snps)
        d_gpu = pairwise_distance(hm, metric='euclidean')
        hap_cpu = hm.haplotypes.get().astype(np.float64)
        d_ref = pdist(hap_cpu, metric='euclidean')
        max_diff = np.max(np.abs(d_gpu - d_ref))
        rel_err = max_diff / np.max(np.abs(d_ref))
        status = "PASS" if rel_err < 1e-10 else "FAIL"
        print(f"  {n_haps} haps x {n_snps} snps: max_rel_err={rel_err:.2e} {status}")

    # Speed
    configs = [
        (100, 10000),
        (100, 50000),
        (200, 10000),
        (200, 50000),
        (200, 100000),
    ]

    print(f"\n{'n_haps':>7} {'n_snps':>8} | {'euclidean (ms)':>14} {'sqeuclidean (ms)':>16}")
    print("-" * 55)
    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)
        t_euc = bench(lambda: pairwise_distance(hm, metric='euclidean'))
        t_sqe = bench(lambda: pairwise_distance(hm, metric='sqeuclidean'))
        print(f"{n_haps:>7} {n_snps:>8} | {t_euc:>14.2f} {t_sqe:>16.2f}")


if __name__ == "__main__":
    main()
