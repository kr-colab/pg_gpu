"""Microbench for issue #94: ehh_decay output-assembly cost.

Times ``selection.ehh_decay`` on a synthetic 200-hap x 10M-variant matrix,
matching the issue's repro. Reports median wall-clock + result shape +
the largest nonzero index (to confirm the trailing-zero region the fix
targets is, in fact, dominant).

Run before and after the implementation change:

    pixi run python debug/bench_ehh_decay.py
"""

import statistics
import time
import numpy as np
import cupy as cp

from pg_gpu import HaplotypeMatrix, selection

N_HAP = 200
N_VAR = 10_000_000
N_WARMUP = 1
N_TIMED = 3
SEED = 0


def main() -> None:
    print(f"Synthetic: {N_HAP} haps x {N_VAR} variants")
    rng = np.random.default_rng(SEED)
    hap = rng.integers(0, 2, (N_HAP, N_VAR), dtype=np.int8)
    pos = np.arange(N_VAR, dtype=np.int64)
    hm = HaplotypeMatrix(hap, pos, 0, N_VAR)
    hm.transfer_to_gpu()

    print(f"Warmup ({N_WARMUP} run)...")
    for _ in range(N_WARMUP):
        selection.ehh_decay(hm)
        cp.cuda.Stream.null.synchronize()

    print(f"Timing ({N_TIMED} runs)...")
    times = []
    last_ehh = None
    for i in range(N_TIMED):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        ehh = selection.ehh_decay(hm)
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        last_ehh = ehh
        max_idx = int(np.nonzero(ehh)[0].max()) if np.any(ehh) else 0
        print(f"  run {i+1}: {t*1000:.1f} ms, shape={ehh.shape}, "
              f"max nonzero idx={max_idx}")

    median = statistics.median(times)
    print()
    print(f"ehh_decay median wall-clock: {median*1000:.1f} ms")
    print(f"  shape={last_ehh.shape}, dtype={last_ehh.dtype}")


if __name__ == "__main__":
    main()
