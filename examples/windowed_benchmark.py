#!/usr/bin/env python
"""
Benchmark: pg_gpu fused windowed statistics vs scikit-allel.

Compares wall-clock time for computing windowed population genetics
statistics on real Ag1000G data across multiple window sizes.

Usage:
    pixi run python examples/windowed_benchmark.py

Requires the example VCF:
    examples/data/gamb.X.8-12Mb.n100.derived.vcf.gz
"""

import time
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix, windowed_analysis


VCF_PATH = "examples/data/gamb.X.8-12Mb.n100.derived.vcf.gz"
POP1_IDX = list(range(100))
POP2_IDX = list(range(100, 200))


def load_data():
    """Load data and build both pg_gpu and allel objects."""
    print(f"Loading {VCF_PATH} ...")
    t0 = time.time()
    hm = HaplotypeMatrix.from_vcf(VCF_PATH)
    hm.sample_sets = {"pop1": POP1_IDX, "pop2": POP2_IDX}
    hm.transfer_to_gpu()
    load_time = time.time() - t0

    hap_np = hm.haplotypes.get().T
    positions = hm.positions.get() if hasattr(hm.positions, 'get') else np.asarray(hm.positions)
    h_allel = allel.HaplotypeArray(hap_np)
    ac_all = h_allel.count_alleles()
    ac1 = h_allel.count_alleles(subpop=POP1_IDX)
    ac2 = h_allel.count_alleles(subpop=POP2_IDX)
    pos_allel = allel.SortedIndex(positions)

    print(f"  {hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants "
          f"(loaded in {load_time:.1f}s)\n")
    return hm, positions, pos_allel, ac_all, ac1, ac2


def bench_allel_single(pos, ac, win_size):
    """Time allel's single-pop windowed stats."""
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_diversity(pos, ac, size=win_size, start=start, stop=stop)
    allel.windowed_watterson_theta(pos, ac, size=win_size, start=start, stop=stop)
    allel.windowed_tajima_d(pos, ac, size=win_size, start=start, stop=stop)
    return time.time() - t0


def bench_allel_twopop(pos, ac1, ac2, win_size):
    """Time allel's two-pop windowed stats."""
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_hudson_fst(pos, ac1, ac2, size=win_size, start=start, stop=stop)
    allel.windowed_divergence(pos, ac1, ac2, size=win_size, start=start, stop=stop)
    return time.time() - t0


def bench_allel_wc(pos, hap_np, win_size):
    """Time allel's Weir-Cockerham windowed FST."""
    start, stop = int(pos[0]), int(pos[-1])
    h_allel = allel.HaplotypeArray(hap_np)
    g = h_allel.to_genotypes(ploidy=2)
    dip_pop1 = list(range(len(POP1_IDX) // 2))
    dip_pop2 = list(range(len(POP1_IDX) // 2, (len(POP1_IDX) + len(POP2_IDX)) // 2))
    t0 = time.time()
    allel.windowed_weir_cockerham_fst(pos, g, [dip_pop1, dip_pop2],
                                       size=win_size, start=start, stop=stop)
    return time.time() - t0


def bench_pg_single(hm, win_size):
    """Time pg_gpu fused single-pop windowed stats."""
    t0 = time.time()
    windowed_analysis(hm, window_size=win_size,
                      statistics=["pi", "theta_w", "tajimas_d"])
    return time.time() - t0


def bench_pg_twopop(hm, win_size):
    """Time pg_gpu fused two-pop windowed stats."""
    t0 = time.time()
    windowed_analysis(hm, window_size=win_size,
                      statistics=["fst", "dxy"],
                      populations=["pop1", "pop2"])
    return time.time() - t0


def bench_pg_wc(hm, win_size):
    """Time pg_gpu fused Weir-Cockerham windowed FST."""
    t0 = time.time()
    windowed_analysis(hm, window_size=win_size,
                      statistics=["fst_wc"],
                      populations=["pop1", "pop2"])
    return time.time() - t0


def bench_pg_all(hm, win_size):
    """Time pg_gpu: ALL stats in a single call."""
    t0 = time.time()
    windowed_analysis(hm, window_size=win_size,
                      statistics=["pi", "theta_w", "tajimas_d",
                                  "fst", "fst_wc", "dxy", "da"],
                      populations=["pop1", "pop2"])
    return time.time() - t0


def main():
    hm, positions, pos_allel, ac_all, ac1, ac2 = load_data()
    hap_np = hm.haplotypes.get().T

    # Warmup pg_gpu kernels
    _ = windowed_analysis(hm, window_size=100_000, statistics=["pi"])
    _ = windowed_analysis(hm, window_size=100_000, statistics=["fst"],
                          populations=["pop1", "pop2"])

    window_sizes = [50_000, 100_000, 200_000, 500_000]

    print("=" * 90)
    print(f"{'':30s} {'allel (s)':>10s} {'pg_gpu (s)':>10s} "
          f"{'speedup':>8s} {'windows':>8s}")
    print("=" * 90)

    for win_size in window_sizes:
        n_win = (int(positions[-1]) - int(positions[0])) // win_size
        print(f"\n--- {win_size // 1000}kb windows ({n_win} windows) ---")

        # Single-pop: pi + theta_w + tajima_d
        ta = bench_allel_single(pos_allel, ac_all, win_size)
        tp = bench_pg_single(hm, win_size)
        print(f"{'pi + theta_w + tajimas_d':<30s} {ta:>10.3f} {tp:>10.3f} "
              f"{ta/tp:>7.1f}x {n_win:>8d}")

        # Two-pop: fst + dxy
        ta = bench_allel_twopop(pos_allel, ac1, ac2, win_size)
        tp = bench_pg_twopop(hm, win_size)
        print(f"{'fst_hudson + dxy':<30s} {ta:>10.3f} {tp:>10.3f} "
              f"{ta/tp:>7.1f}x {n_win:>8d}")

        # Weir-Cockerham FST
        ta = bench_allel_wc(pos_allel, hap_np, win_size)
        tp = bench_pg_wc(hm, win_size)
        print(f"{'fst_weir_cockerham':<30s} {ta:>10.3f} {tp:>10.3f} "
              f"{ta/tp:>7.1f}x {n_win:>8d}")

        # pg_gpu ALL stats (no allel equivalent -- single call)
        tp_all = bench_pg_all(hm, win_size)
        print(f"{'ALL 7 stats (pg_gpu only)':<30s} {'---':>10s} {tp_all:>10.3f} "
              f"{'---':>8s} {n_win:>8d}")

    print("\n" + "=" * 90)
    print("Notes:")
    print("  - pg_gpu computes all stats in a single fused CUDA kernel launch")
    print("  - allel requires separate function calls for each statistic")
    print("  - 'ALL 7 stats' = pi + theta_w + tajimas_d + fst + fst_wc + dxy + da")
    print("  - Weir-Cockerham is especially expensive in allel (full genotype ANOVA)")


if __name__ == "__main__":
    main()
