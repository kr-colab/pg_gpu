"""Benchmark ZnS: chunked Gram matrix path vs tiled."""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu.ld_statistics import (zns, _zns_from_precomputed,
                                  _prepare_segregating)


def make_data(n_haps, n_snps, miss_rate=0.0, seed=42):
    rng = np.random.default_rng(seed)
    founders = rng.integers(0, 2, size=(5, n_snps), dtype=np.int8)
    assignments = rng.integers(0, 5, size=n_haps)
    haps = founders[assignments].copy()
    mutations = rng.random(size=(n_haps, n_snps)) < 0.02
    haps ^= mutations.astype(np.int8)
    if miss_rate > 0:
        missing = rng.random(size=(n_haps, n_snps)) < miss_rate
        haps[missing] = -1
    positions = np.arange(n_snps) * 100
    hm = HaplotypeMatrix(haps, positions, positions[0], positions[-1])
    hm.transfer_to_gpu()
    return hm


def tiled_reference(hm, missing_data='include'):
    """Compute ZnS via the tiled path for reference."""
    hap_clean, valid_mask, m = _prepare_segregating(hm, missing_data)
    if m < 2:
        return 0.0
    return _zns_from_precomputed(hap_clean, valid_mask, 0, m)


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
    # Correctness: Gram vs tiled (no missing data)
    print("Correctness (no missing data):")
    for n_haps, n_snps in [(50, 500), (100, 2000), (200, 5000)]:
        hm = make_data(n_haps, n_snps)
        zns_gram = zns(hm)
        zns_ref = tiled_reference(hm)
        diff = abs(zns_gram - zns_ref)
        rel = diff / max(abs(zns_ref), 1e-15)
        status = "PASS" if rel < 1e-6 else "FAIL"
        print(f"  {n_haps} haps x {n_snps} snps: "
              f"gram={zns_gram:.8f} tiled={zns_ref:.8f} "
              f"rel_err={rel:.2e} {status}")

    # Correctness: Gram with missing data at various rates
    for miss_rate in [0.01, 0.05, 0.10]:
        print(f"\nCorrectness ({int(miss_rate*100)}% missing data, corrected mean imputation):")
        for n_haps, n_snps in [(50, 500), (100, 2000), (200, 5000)]:
            hm = make_data(n_haps, n_snps, miss_rate=miss_rate)
            zns_gram = zns(hm)
            zns_ref = tiled_reference(hm)
            diff = abs(zns_gram - zns_ref)
            rel = diff / max(abs(zns_ref), 1e-15)
            status = "PASS" if rel < 0.05 else "WARN"
            print(f"  {n_haps} haps x {n_snps} snps: "
                  f"gram={zns_gram:.8f} tiled={zns_ref:.8f} "
                  f"rel_err={rel:.2e} {status}")

    # Speed comparison
    configs = [
        (100, 5000),
        (100, 10000),
        (100, 50000),
        (200, 10000),
        (200, 50000),
    ]

    print(f"\n{'n_haps':>7} {'n_snps':>8} | "
          f"{'gram (ms)':>10} {'tiled (ms)':>11} {'speedup':>8}")
    print("-" * 58)

    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)
        t_gram = bench(lambda: zns(hm))
        t_tiled = bench(lambda: tiled_reference(hm))
        speedup = t_tiled / t_gram
        print(f"{n_haps:>7} {n_snps:>8} | "
              f"{t_gram:>10.2f} {t_tiled:>11.2f} {speedup:>7.1f}x")


if __name__ == "__main__":
    main()
