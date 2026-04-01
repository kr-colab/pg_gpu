#!/usr/bin/env python
"""
Benchmark: pg_gpu fused windowed statistics vs scikit-allel.

Compares wall-clock time for computing windowed population genetics
statistics on real Ag1000G data across multiple window sizes. Produces
a summary figure showing speedups.

Usage:
    pixi run python examples/windowed_benchmark.py

Requires the example VCF:
    examples/data/gamb.X.8-12Mb.n100.derived.vcf.gz
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    return hm, hap_np, positions, pos_allel, ac_all, ac1, ac2


# ── allel benchmark functions ─────────────────────────────────────────────

def bench_allel_pi(pos, ac, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_diversity(pos, ac, size=win_size, start=start, stop=stop)
    return time.time() - t0

def bench_allel_theta_w(pos, ac, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_watterson_theta(pos, ac, size=win_size, start=start, stop=stop)
    return time.time() - t0

def bench_allel_tajimas_d(pos, ac, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_tajima_d(pos, ac, size=win_size, start=start, stop=stop)
    return time.time() - t0

def bench_allel_fst_hudson(pos, ac1, ac2, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_hudson_fst(pos, ac1, ac2, size=win_size, start=start, stop=stop)
    return time.time() - t0

def bench_allel_dxy(pos, ac1, ac2, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    t0 = time.time()
    allel.windowed_divergence(pos, ac1, ac2, size=win_size, start=start, stop=stop)
    return time.time() - t0

def bench_allel_fst_wc(pos, hap_np, win_size):
    start, stop = int(pos[0]), int(pos[-1])
    h_allel = allel.HaplotypeArray(hap_np)
    g = h_allel.to_genotypes(ploidy=2)
    n_dip = hap_np.shape[1] // 2
    dip1 = list(range(n_dip // 2))
    dip2 = list(range(n_dip // 2, n_dip))
    t0 = time.time()
    allel.windowed_weir_cockerham_fst(pos, g, [dip1, dip2],
                                       size=win_size, start=start, stop=stop)
    return time.time() - t0


# ── pg_gpu benchmark functions ────────────────────────────────────────────

def bench_pg(hm, win_size, statistics, populations=None):
    t0 = time.time()
    windowed_analysis(hm, window_size=win_size,
                      statistics=statistics,
                      populations=populations)
    return time.time() - t0


# ── main ──────────────────────────────────────────────────────────────────

def main():
    hm, hap_np, positions, pos_allel, ac_all, ac1, ac2 = load_data()

    # Warmup pg_gpu kernels
    _ = windowed_analysis(hm, window_size=100_000, statistics=["pi"])
    _ = windowed_analysis(hm, window_size=100_000, statistics=["fst"],
                          populations=["pop1", "pop2"])

    window_sizes = [50_000, 100_000, 200_000, 500_000]

    # Define the stat groups to benchmark
    stat_groups = [
        ("pi",           lambda ws: bench_allel_pi(pos_allel, ac_all, ws),
                         lambda ws: bench_pg(hm, ws, ["pi"])),
        ("theta_w",      lambda ws: bench_allel_theta_w(pos_allel, ac_all, ws),
                         lambda ws: bench_pg(hm, ws, ["theta_w"])),
        ("tajimas_d",    lambda ws: bench_allel_tajimas_d(pos_allel, ac_all, ws),
                         lambda ws: bench_pg(hm, ws, ["tajimas_d"])),
        ("fst_hudson",   lambda ws: bench_allel_fst_hudson(pos_allel, ac1, ac2, ws),
                         lambda ws: bench_pg(hm, ws, ["fst"], ["pop1", "pop2"])),
        ("dxy",          lambda ws: bench_allel_dxy(pos_allel, ac1, ac2, ws),
                         lambda ws: bench_pg(hm, ws, ["dxy"], ["pop1", "pop2"])),
        ("fst_wc",       lambda ws: bench_allel_fst_wc(pos_allel, hap_np, ws),
                         lambda ws: bench_pg(hm, ws, ["fst_wc"], ["pop1", "pop2"])),
        ("all 7 stats",  None,  # no single allel equivalent
                         lambda ws: bench_pg(hm, ws,
                             ["pi", "theta_w", "tajimas_d", "fst", "fst_wc", "dxy", "da"],
                             ["pop1", "pop2"])),
    ]

    # Collect results
    rows = []
    for win_size in window_sizes:
        n_win = (int(positions[-1]) - int(positions[0])) // win_size
        win_label = f"{win_size // 1000}kb"
        print(f"Benchmarking {win_label} windows ({n_win} windows) ...")

        for name, allel_fn, pg_fn in stat_groups:
            tp = pg_fn(win_size)
            if allel_fn is not None:
                ta = allel_fn(win_size)
                speedup = ta / tp
                rows.append(dict(statistic=name, window=win_label,
                                 allel=ta, pg_gpu=tp, speedup=speedup,
                                 n_windows=n_win))
            else:
                rows.append(dict(statistic=name, window=win_label,
                                 allel=np.nan, pg_gpu=tp, speedup=np.nan,
                                 n_windows=n_win))

    df = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Statistic':<16s} {'Window':>8s} {'allel (s)':>10s} "
          f"{'pg_gpu (s)':>10s} {'Speedup':>8s}")
    print("-" * 80)
    for _, r in df.iterrows():
        allel_s = f"{r['allel']:.3f}" if np.isfinite(r['allel']) else "---"
        speedup_s = f"{r['speedup']:.1f}x" if np.isfinite(r['speedup']) else "---"
        print(f"{r['statistic']:<16s} {r['window']:>8s} {allel_s:>10s} "
              f"{r['pg_gpu']:>10.3f} {speedup_s:>8s}")
    print("=" * 80)

    # ── Figure ────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="talk", palette="Set2")

    df_plot = df[df['speedup'].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [3, 2]})

    # Panel A: speedup by statistic and window size
    ax = axes[0]
    sns.barplot(data=df_plot, x="statistic", y="speedup", hue="window",
                ax=ax, edgecolor="0.3", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Speedup (pg_gpu / allel)")
    ax.set_xlabel("")
    ax.axhline(1, color="0.4", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_title("GPU speedup by statistic")
    ax.legend(title="Window size", fontsize=9, title_fontsize=10)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # Panel B: absolute time -- pg_gpu "all 7 stats" vs allel individual stats summed
    ax2 = axes[1]
    # For each window size, sum allel times for the 6 individual stats and compare
    # to pg_gpu's single "all 7 stats" call
    bar_rows = []
    for win_label in df['window'].unique():
        wdf = df[df['window'] == win_label]
        allel_total = wdf[wdf['statistic'] != 'all 7 stats']['allel'].sum()
        pg_all = wdf[wdf['statistic'] == 'all 7 stats']['pg_gpu'].values[0]
        bar_rows.append(dict(window=win_label, tool="allel (6 calls)", time=allel_total))
        bar_rows.append(dict(window=win_label, tool="pg_gpu (1 call)", time=pg_all))
    bar_df = pd.DataFrame(bar_rows)

    sns.barplot(data=bar_df, x="window", y="time", hue="tool", ax=ax2,
                edgecolor="0.3", linewidth=0.5,
                palette=["#e74c3c", "#2ecc71"])
    ax2.set_ylabel("Wall time (s)")
    ax2.set_xlabel("Window size")
    ax2.set_title("Total time: 7 statistics")
    ax2.legend(fontsize=9)

    fig.suptitle(f"pg_gpu vs scikit-allel windowed statistics\n"
                 f"({hm.num_variants:,} variants, {hm.num_haplotypes} haplotypes)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig("examples/windowed_benchmark.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to examples/windowed_benchmark.png")
    plt.show()


if __name__ == "__main__":
    main()
