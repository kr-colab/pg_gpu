#!/usr/bin/env python
"""
Large-scale ZnS validation on real Ag1000G 3L data.

Tests the chunked Gram path at chromosome scale (2940 haplotypes x ~8M
variants) to verify no OOM and measure performance.  Also validates
correctness against the tiled reference on a 50K-variant subset.

Usage:
    pixi run python debug/test_large_scale.py
"""

import time
import numpy as np
import cupy as cp
import zarr

from pg_gpu import HaplotypeMatrix
from pg_gpu.ld_statistics import (zns, _zns_from_precomputed,
                                  _prepare_segregating)

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
CHROM = "3L"


def load_3L():
    """Load full Ag1000G 3L arm as HaplotypeMatrix on GPU."""
    print(f"Loading {CHROM} from {ZARR_PATH}...", flush=True)
    t0 = time.time()
    store = zarr.open(ZARR_PATH, mode='r')
    chrom = store[CHROM]
    positions = np.array(chrom['variants/POS'])
    gt = np.array(chrom['calldata/GT'])
    n_v, n_s, _ = gt.shape

    hap = np.empty((n_v, 2 * n_s), dtype=gt.dtype)
    hap[:, :n_s] = gt[:, :, 0]
    hap[:, n_s:] = gt[:, :, 1]
    hap = hap.T
    del gt

    hm = HaplotypeMatrix(hap, positions, int(positions[0]), int(positions[-1]))
    n_hap, n_var = hm.num_haplotypes, hm.num_variants
    print(f"  {n_hap} haplotypes x {n_var:,} variants ({time.time()-t0:.0f}s)",
          flush=True)

    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  Transferred to GPU", flush=True)
    return hm


def tiled_ref(hm):
    """Compute ZnS via the O(m^2) tiled path for reference."""
    hap_clean, valid_mask, m = _prepare_segregating(hm)
    if m < 2:
        return 0.0
    return _zns_from_precomputed(hap_clean, valid_mask, 0, m)


def main():
    hm = load_3L()
    n_hap = hm.num_haplotypes
    n_var = hm.num_variants

    # --- Part 1: Correctness on 50K-variant subset ----------------------
    # Tiled reference is O(m^2), only feasible for small m.
    SUBSET = 50_000
    print(f"\n{'='*65}", flush=True)
    print(f"Correctness: {n_hap} haps x {SUBSET:,} variants (tiled reference)",
          flush=True)
    print(f"{'='*65}", flush=True)

    # Take first SUBSET variants
    subset_idx = cp.arange(min(SUBSET, n_var))
    hm_sub = hm.get_subset(subset_idx)

    gram_val = zns(hm_sub)
    tiled_val = tiled_ref(hm_sub)
    rel = abs(gram_val - tiled_val) / max(abs(tiled_val), 1e-15)
    print(f"  Gram:  {gram_val:.10f}", flush=True)
    print(f"  Tiled: {tiled_val:.10f}", flush=True)
    print(f"  Relative error: {rel:.2e}  {'PASS' if rel < 1e-6 else 'FAIL'}",
          flush=True)

    del hm_sub
    cp.get_default_memory_pool().free_all_blocks()

    # --- Part 2: Full chromosome ZnS ------------------------------------
    print(f"\n{'='*65}", flush=True)
    print(f"Full scale: {n_hap} haps x {n_var:,} variants", flush=True)
    print(f"{'='*65}", flush=True)

    mem_before = cp.cuda.Device(0).mem_info
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    z_full = zns(hm)
    cp.cuda.Device(0).synchronize()
    t1 = time.perf_counter()
    mem_after = cp.cuda.Device(0).mem_info

    print(f"  ZnS = {z_full:.10f}", flush=True)
    print(f"  Time = {t1 - t0:.2f}s", flush=True)
    print(f"  GPU memory: {mem_before[0]/1e9:.1f} GB free before, "
          f"{mem_after[0]/1e9:.1f} GB free after "
          f"(of {mem_before[1]/1e9:.0f} GB total)", flush=True)
    print(f"  No OOM -- success!", flush=True)


if __name__ == "__main__":
    main()
