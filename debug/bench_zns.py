"""Benchmark ZnS: Gram matrix trick vs tiled approach."""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu.ld_statistics import zns, _zns_tiled_impl, _prepare_segregating


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


def bench(fn, n_reps=3):
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
    # Correctness: compare Gram vs tiled for small data
    print("Correctness check:")
    for n_haps, n_snps in [(50, 500), (100, 2000), (200, 5000)]:
        hm = make_data(n_haps, n_snps)
        zns_new = zns(hm)
        # Force tiled path
        hap_clean, valid_mask, m = _prepare_segregating(hm)
        zns_old = _zns_tiled_impl(hap_clean, valid_mask, m, 'include')
        diff = abs(zns_new - zns_old)
        rel = diff / max(abs(zns_old), 1e-15)
        status = "PASS" if rel < 1e-6 else "FAIL"
        print(f"  {n_haps} haps x {n_snps} snps: "
              f"gram={zns_new:.8f} tiled={zns_old:.8f} rel_err={rel:.2e} {status}")

    # Speed comparison
    configs = [
        (100, 5000),
        (100, 10000),
        (100, 50000),
        (200, 10000),
        (200, 50000),
    ]

    print(f"\n{'n_haps':>7} {'n_snps':>8} | {'gram (ms)':>10} {'tiled (ms)':>11} {'speedup':>8}")
    print("-" * 55)

    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)

        t_gram = bench(lambda: zns(hm))

        hap_clean, valid_mask, m = _prepare_segregating(hm)
        t_tiled = bench(lambda: _zns_tiled_impl(hap_clean, valid_mask, m, 'include'))

        speedup = t_tiled / t_gram
        print(f"{n_haps:>7} {n_snps:>8} | {t_gram:>10.2f} {t_tiled:>11.2f} {speedup:>7.1f}x")


if __name__ == "__main__":
    main()
