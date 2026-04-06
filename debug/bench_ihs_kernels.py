"""Benchmark iHS kernel paths: fused vs histogram for various haplotype counts.

The fused kernel uses one block per variant with __syncthreads() per scan step.
The histogram kernel uses one thread per pair with no sync, building histograms.
Currently the threshold is n_haplotypes > 256 for histogram. This tests whether
the histogram path is actually faster for smaller counts too.
"""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu.selection import (
    _ihh01_scan_gpu, _ihh01_scan_hist_gpu, _get_pair_indices,
    _compute_gaps, ihs
)


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


def bench_kernel(scan_fn, hap_t, gaps_gpu, n_reps=3):
    """Benchmark a single scan direction."""
    scan_fn(hap_t, gaps_gpu, 0.05, 0.05, False)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        scan_fn(hap_t, gaps_gpu, 0.05, 0.05, False)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def bench_full_ihs(hm, n_reps=3):
    """Benchmark the full ihs() call (fwd + rev + log)."""
    ihs(hm)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        ihs(hm)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def correctness_check(hm):
    """Verify fused and histogram give same results."""
    hap_t = hm.haplotypes.T
    pos = hm.positions
    if hasattr(pos, 'get'):
        pos = pos.get()
    gaps = _compute_gaps(pos)
    gaps_gpu = cp.asarray(gaps)

    ihh0_f, ihh1_f = _ihh01_scan_gpu(hap_t, gaps_gpu, 0.05, 0.05, False)
    ihh0_h, ihh1_h = _ihh01_scan_hist_gpu(hap_t, gaps_gpu, 0.05, 0.05, False)

    # Compare where both are non-NaN
    mask0 = ~np.isnan(ihh0_f) & ~np.isnan(ihh0_h) & (ihh0_f > 0) & (ihh0_h > 0)
    mask1 = ~np.isnan(ihh1_f) & ~np.isnan(ihh1_h) & (ihh1_f > 0) & (ihh1_h > 0)

    if mask0.sum() > 0:
        rel0 = np.max(np.abs(ihh0_f[mask0] - ihh0_h[mask0]) / ihh0_f[mask0])
    else:
        rel0 = 0.0
    if mask1.sum() > 0:
        rel1 = np.max(np.abs(ihh1_f[mask1] - ihh1_h[mask1]) / ihh1_f[mask1])
    else:
        rel1 = 0.0

    return rel0, rel1, mask0.sum(), mask1.sum()


def main():
    configs = [
        (100, 10000),
        (100, 50000),
        (200, 10000),
        (200, 50000),
    ]

    # Correctness
    print("Correctness: fused vs histogram")
    for n_haps, n_snps in [(100, 5000), (200, 5000)]:
        hm = make_data(n_haps, n_snps)
        r0, r1, n0, n1 = correctness_check(hm)
        print(f"  {n_haps} haps x {n_snps} snps: "
              f"ihh0 rel_err={r0:.2e} ({n0} sites), "
              f"ihh1 rel_err={r1:.2e} ({n1} sites)")

    # Speed: single direction scan
    print(f"\nSingle-direction scan timing:")
    print(f"{'n_haps':>7} {'n_snps':>8} | {'fused (ms)':>11} {'histogram (ms)':>15} {'ratio':>7}")
    print("-" * 58)

    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)
        hap_t = hm.haplotypes.T
        pos = hm.positions
        if hasattr(pos, 'get'):
            pos = pos.get()
        gaps = _compute_gaps(pos)
        gaps_gpu = cp.asarray(gaps)

        t_fused = bench_kernel(_ihh01_scan_gpu, hap_t, gaps_gpu)
        t_hist = bench_kernel(_ihh01_scan_hist_gpu, hap_t, gaps_gpu)
        ratio = t_fused / t_hist

        print(f"{n_haps:>7} {n_snps:>8} | {t_fused:>11.2f} {t_hist:>15.2f} {ratio:>6.2f}x")

    # Full iHS timing
    print(f"\nFull ihs() timing (fwd + rev + log):")
    print(f"{'n_haps':>7} {'n_snps':>8} | {'ihs (ms)':>10}")
    print("-" * 30)
    for n_haps, n_snps in configs:
        hm = make_data(n_haps, n_snps)
        t = bench_full_ihs(hm)
        print(f"{n_haps:>7} {n_snps:>8} | {t:>10.2f}")


if __name__ == "__main__":
    main()
