#!/usr/bin/env python
"""
Benchmark: pg_gpu two-population LD statistics vs moments.

Validates that pg_gpu produces the same LD statistics (DD, Dz, pi2)
as moments for a two-population IM model, and compares wall-clock time.

Usage:
    pixi run python examples/two_pop_ld_benchmark.py

Requires:
    data/im-parsing-example.vcf  (IM model simulation)
    data/im_pop.txt              (population assignments)
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

import moments.LD.Parsing as mParsing
import allel
from pg_gpu.haplotype_matrix import HaplotypeMatrix


VCF_PATH = "data/im-parsing-example.vcf"
POP_FILE = "data/im_pop.txt"
CACHE_DIR = Path("cache")
POPS = ["deme0", "deme1"]
BP_BINS = np.logspace(2, 6, 6)  # [100, 631, 3981, 25119, 158489, 1000000]


def load_pg_gpu():
    """Load VCF into a HaplotypeMatrix with population assignments."""
    hm = HaplotypeMatrix.from_vcf(VCF_PATH)
    vcf = allel.read_vcf(VCF_PATH)
    n_samples = vcf['samples'].shape[0]

    pop_map = {}
    with open(POP_FILE) as f:
        for line in f:
            sample, pop = line.strip().split()
            pop_map[sample] = pop

    pop_sets = {"deme0": [], "deme1": []}
    for i, name in enumerate(vcf['samples']):
        pop = pop_map.get(name)
        if pop in pop_sets:
            pop_sets[pop].append(i)
            pop_sets[pop].append(i + n_samples)

    hm.sample_sets = pop_sets
    return hm


def run_moments(use_cache=True):
    """Run moments LD computation (with disk cache)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "moments_ld_im.pkl"

    if use_cache and cache_file.exists():
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        print(f"  moments: loaded from cache")
        return result['stats'], result['time']

    t0 = time.time()
    ld_stats = mParsing.compute_ld_statistics(
        VCF_PATH, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=False, report=False)
    elapsed = time.time() - t0

    with open(cache_file, 'wb') as f:
        pickle.dump({'stats': ld_stats, 'time': elapsed}, f)

    print(f"  moments: computed in {elapsed:.1f}s (cached for next run)")
    return ld_stats, elapsed


def run_pg_gpu(hm):
    """Run pg_gpu LD computation (already returns moments-compatible format)."""
    hm.transfer_to_gpu()

    # Warmup
    _ = hm.compute_ld_statistics_gpu_two_pops(
        BP_BINS, "deme0", "deme1", raw=True)

    t0 = time.time()
    gpu_stats = hm.compute_ld_statistics_gpu_two_pops(
        BP_BINS, "deme0", "deme1", raw=True)
    elapsed = time.time() - t0

    print(f"  pg_gpu: computed in {elapsed:.3f}s")
    return gpu_stats, elapsed


def compare(moments_stats, gpu_stats):
    """Compare moments and pg_gpu results, return (moments_vals, gpu_vals, stat_names)."""
    stat_names = moments_stats['stats'][0]
    n_stats = len(stat_names)

    all_moments = []
    all_gpu = []
    all_labels = []

    for bin_idx, (bin_range, mom_sums) in enumerate(
            zip(moments_stats['bins'], moments_stats['sums'])):
        if bin_range not in gpu_stats:
            continue
        gpu_vals = gpu_stats[bin_range]
        gpu_ordered = [gpu_vals[s] for s in stat_names]

        for s, mv, gv in zip(stat_names, mom_sums, gpu_ordered):
            all_moments.append(mv)
            all_gpu.append(gv)
            all_labels.append(s)

    mom = np.array(all_moments)
    gpu = np.array(all_gpu)
    rel_err = np.abs(gpu - mom) / (np.abs(mom) + 1e-10)

    print(f"\n  {len(mom)} statistic-bin pairs compared")
    print(f"  Correlation: {np.corrcoef(mom, gpu)[0,1]:.8f}")
    print(f"  Mean relative error: {rel_err.mean():.2e}")
    print(f"  Max relative error:  {rel_err.max():.2e}")
    print(f"  Pairs with <1% error: {(rel_err < 0.01).sum()}/{len(rel_err)}")

    return mom, gpu, all_labels, stat_names


def plot_results(mom, gpu, labels, stat_names, t_moments, t_pg):
    """Create a two-panel figure: correspondence plot + timing comparison."""
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={'width_ratios': [3, 2]})

    # Panel A: correspondence
    ax = axes[0]
    colors = []
    for lab in labels:
        if lab.startswith('DD'):
            colors.append('#3498db')
        elif lab.startswith('Dz'):
            colors.append('#2ecc71')
        else:
            colors.append('#e74c3c')

    ax.scatter(mom, gpu, c=colors, alpha=0.7, s=40, edgecolors='0.3', linewidths=0.3)
    lo = min(mom.min(), gpu.min())
    hi = max(mom.max(), gpu.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel("moments")
    ax.set_ylabel("pg_gpu")
    ax.set_title("Two-population LD statistics")
    ax.set_xscale('symlog', linthresh=1e-6)
    ax.set_yscale('symlog', linthresh=1e-6)
    ticks = [-1e-2, -1e-4, 0, 1e-4, 1e-2, 1, 1e2]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#3498db', label='DD'),
                       Patch(color='#2ecc71', label='Dz'),
                       Patch(color='#e74c3c', label='pi2')],
              loc='upper left', fontsize=10)

    # Panel B: timing
    ax2 = axes[1]
    tools = ['moments', 'pg_gpu']
    times = [t_moments, t_pg]
    bars = ax2.bar(tools, times, color=['#e74c3c', '#2ecc71'],
                   edgecolor='0.3', linewidth=0.5)
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title(f"Speedup: {t_moments/t_pg:.0f}x")
    ax2.set_yscale('log')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, t * 1.2,
                 f"{t:.2f}s", ha='center', va='bottom', fontsize=11)

    fig.suptitle("pg_gpu vs moments: two-population LD (DD, Dz, pi2)\n"
                 f"IM model, {len(stat_names)} statistics x {len(mom)//len(stat_names)} distance bins",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig("examples/two_pop_ld_benchmark.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to examples/two_pop_ld_benchmark.png")


def main():
    print("Two-population LD benchmark: pg_gpu vs moments")
    print("=" * 55)

    print(f"\nData: {VCF_PATH}")
    hm = load_pg_gpu()
    print(f"  {hm.num_haplotypes} haplotypes, {hm.num_variants:,} variants")
    print(f"  Populations: {list(hm.sample_sets.keys())}")
    print(f"  Distance bins: {len(BP_BINS)-1} bins from "
          f"{BP_BINS[0]:.0f} to {BP_BINS[-1]:.0f} bp")

    print(f"\nComputing LD statistics ...")
    moments_stats, t_moments = run_moments()
    gpu_stats, t_pg = run_pg_gpu(hm)

    print(f"\nValidation:")
    mom, gpu, labels, stat_names = compare(moments_stats, gpu_stats)

    print(f"\nTiming:")
    print(f"  moments: {t_moments:.2f}s")
    print(f"  pg_gpu:  {t_pg:.3f}s")
    print(f"  Speedup: {t_moments/t_pg:.0f}x")

    plot_results(mom, gpu, labels, stat_names, t_moments, t_pg)


if __name__ == "__main__":
    main()
