#!/usr/bin/env python
"""Benchmark for issue #91: GPU-resident accumulation in
``_local_pca_streaming``.

Times ``local_pca(..., engine='streaming-dense')`` on a synthetic
haplotype matrix sized to exercise the per-window host-sync chain
that issue #91 targets. Prints a median wall-clock over 3 timed
runs (1 warmup) so the result is stable enough to compare across
the before/after of the change. Also reports the GPU memory cost
of the proposed GPU-resident output buffers.

Run before and after the implementation change:

    pixi run python debug/bench_local_pca_streaming.py
"""

import statistics
import time
import numpy as np
import cupy as cp

from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import local_pca

N_HAP = 200
N_VAR = 1_200_000     # 6000 windows * 200 SNPs/window
WINDOW_SIZE = 200
K = 2
SEED = 0
N_WARMUP = 1
N_TIMED = 3


def build_matrix(seed: int) -> HaplotypeMatrix:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.5, size=N_VAR).astype(np.float32)
    haps = (rng.random((N_HAP, N_VAR), dtype=np.float32) < p).astype(np.int8)
    positions = np.arange(N_VAR, dtype=np.int64)
    return HaplotypeMatrix(haps, positions, 0, N_VAR)


def time_once(hm: HaplotypeMatrix) -> float:
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    res = local_pca(hm, window_size=WINDOW_SIZE, window_type='snp',
                    k=K, engine='streaming-dense')
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    n_win = res.n_windows
    return elapsed, n_win


def main() -> None:
    print(f"Building synthetic HaplotypeMatrix: "
          f"{N_HAP} haps x {N_VAR} variants, window={WINDOW_SIZE} SNPs")
    hm = build_matrix(SEED)
    hm.transfer_to_gpu()

    print(f"Warmup ({N_WARMUP} run)...")
    for _ in range(N_WARMUP):
        time_once(hm)

    print(f"Timing ({N_TIMED} runs)...")
    times = []
    n_win = None
    for i in range(N_TIMED):
        t, n_win = time_once(hm)
        print(f"  run {i+1}: {t:.3f} s ({n_win} windows)")
        times.append(t)

    median = statistics.median(times)
    per_window_us = 1e6 * median / n_win
    print()
    print(f"streaming-dense median wall-clock: {median:.3f} s")
    print(f"  per-window: {per_window_us:.1f} us  (n_windows = {n_win})")

    # Cost estimate of GPU-resident buffers proposed in issue #91:
    eigvals_bytes = n_win * K * 8
    eigvecs_bytes = n_win * K * N_HAP * 8
    sumsq_bytes = n_win * 8
    total_mb = (eigvals_bytes + eigvecs_bytes + sumsq_bytes) / 1024 / 1024
    print(f"GPU output-buffer footprint at this size: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
