#!/usr/bin/env python
"""
Stress test and benchmark: pg_gpu vs scikit-allel on Ag1000G Phase 3.

Loads the full chromosome arm and benchmarks every public statistic
against scikit-allel where an equivalent exists. Builds allel objects
lazily to avoid upfront memory explosion. Streams results as they
complete.

Usage:
    pixi run python debug/stress_test_ag1000g.py
"""

import sys
import time
import traceback
import numpy as np
import cupy as cp
import allel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pg_gpu import (
    HaplotypeMatrix,
    diversity,
    divergence,
    selection,
    sfs,
    admixture,
    decomposition,
    relatedness,
    windowed_analysis,
    ld_statistics,
    distance_stats,
)
from pg_gpu.genotype_matrix import GenotypeMatrix

# ── Config ───────────────────────────────────────────────────────────────
ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
ACC_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz"
CHROM = "3R"
REGION_START = None  # None = full arm
REGION_END = None
N_DIP_PER_POP = 100
N_ITER = 1  # single run for long-running benchmarks


# ── Lazy allel object cache ──────────────────────────────────────────────
_allel_cache = {}

def _get_allel(gt, positions, key):
    """Lazily build and cache allel objects."""
    if key in _allel_cache:
        return _allel_cache[key]

    pop1_dip = list(range(N_DIP_PER_POP))
    pop2_dip = list(range(N_DIP_PER_POP, 2 * N_DIP_PER_POP))
    pop3_dip = list(range(2 * N_DIP_PER_POP, 3 * N_DIP_PER_POP))
    pop1_hap = list(range(0, 2 * N_DIP_PER_POP))
    pop2_hap = list(range(2 * N_DIP_PER_POP, 4 * N_DIP_PER_POP))

    if key == 'pos':
        _allel_cache[key] = allel.SortedIndex(positions)
    elif key == 'ac':
        g = _get_allel(gt, positions, 'g')
        _allel_cache[key] = g.count_alleles()
    elif key == 'ac1':
        g = _get_allel(gt, positions, 'g')
        _allel_cache[key] = g.count_alleles(subpop=pop1_dip)
    elif key == 'ac2':
        g = _get_allel(gt, positions, 'g')
        _allel_cache[key] = g.count_alleles(subpop=pop2_dip)
    elif key == 'ac3':
        g = _get_allel(gt, positions, 'g')
        _allel_cache[key] = g.count_alleles(subpop=pop3_dip)
    elif key == 'g':
        _allel_cache[key] = allel.GenotypeArray(gt)
    elif key == 'h1':
        g = _get_allel(gt, positions, 'g')
        h = g.to_haplotypes()
        _allel_cache[key] = h.subset(sel1=pop1_hap)
        _allel_cache['h2'] = h.subset(sel1=pop2_hap)
        del h
    elif key == 'h2':
        _get_allel(gt, positions, 'h1')  # builds both
    return _allel_cache[key]


def load_data():
    """Load region from Ag1000G zarr."""
    import zarr

    region_str = (f"{CHROM}:{REGION_START:,}-{REGION_END:,}"
                  if REGION_START is not None else f"{CHROM} (full arm)")
    print(f"Loading {region_str} from Ag1000G...", flush=True)
    t0 = time.time()

    store = zarr.open(ZARR_PATH, mode='r')
    chrom_grp = store[CHROM]
    positions = np.array(chrom_grp['variants/POS'])

    if REGION_START is not None and REGION_END is not None:
        mask = (positions >= REGION_START) & (positions < REGION_END)
        positions = positions[mask]
        gt = np.array(chrom_grp['calldata/GT'][np.where(mask)[0], :, :])
    else:
        gt = np.array(chrom_grp['calldata/GT'])

    samples = np.array(chrom_grp['samples'])
    n_variants, n_samples, ploidy = gt.shape
    assert ploidy == 2

    haplotypes = np.empty((n_variants, 2 * n_samples), dtype=gt.dtype)
    haplotypes[:, :n_samples] = gt[:, :, 0]
    haplotypes[:, n_samples:] = gt[:, :, 1]
    haplotypes = haplotypes.T

    chrom_start = REGION_START if REGION_START is not None else int(positions[0])
    chrom_end = REGION_END if REGION_END is not None else int(positions[-1])
    hm = HaplotypeMatrix(
        haplotypes, positions,
        chrom_start=chrom_start, chrom_end=chrom_end,
        samples=list(samples),
    )

    n_hap = 2 * N_DIP_PER_POP
    hm.sample_sets = {
        "pop1": list(range(0, n_hap)),
        "pop2": list(range(n_hap, 2 * n_hap)),
        "pop3": list(range(2 * n_hap, 3 * n_hap)),
    }

    print(f"  {hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants ({time.time()-t0:.0f}s)",
          flush=True)
    return hm, gt, positions


def bench(fn, n_iter=N_ITER, sync_gpu=True):
    """Run fn n_iter times, return median wall time."""
    times = []
    for _ in range(n_iter):
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    hm, gt, positions = load_data()
    gm = GenotypeMatrix.from_haplotype_matrix(hm)
    pos = positions

    # Transfer full matrix to GPU upfront
    print("Transferring to GPU...", flush=True)
    t0 = time.time()
    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  Transfer: {time.time()-t0:.1f}s", flush=True)

    # Warmup kernels
    _ = diversity.pi(hm, population="pop1")
    cp.cuda.Stream.null.synchronize()

    # LD subset
    hm_ld = hm.get_subset(np.arange(min(1000, hm.num_variants)))
    hm_ld.transfer_to_gpu()
    counts_1pop, nv_1pop = hm_ld.tally_gpu_haplotypes(pop="pop1")

    win_start = int(pos[0])
    win_stop = int(pos[-1])

    # ── Benchmarks ───────────────────────────────────────────────────────
    # (name, pg_gpu_fn, allel_fn_or_None)
    # allel fns use lazy cache to avoid upfront memory explosion

    def allel_fn(key_deps, fn):
        """Wrapper that builds allel objects on demand."""
        def wrapped():
            for k in key_deps:
                _get_allel(gt, pos, k)
            return fn()
        return wrapped

    benchmarks = []

    # --- Diversity ---
    # Diploid sample indices for allel (fair: include allele counting in timing)
    pop1_dip = list(range(N_DIP_PER_POP))
    pop2_dip = list(range(N_DIP_PER_POP, 2 * N_DIP_PER_POP))
    pop3_dip = list(range(2 * N_DIP_PER_POP, 3 * N_DIP_PER_POP))

    benchmarks.append(("diversity.pi",
        lambda: diversity.pi(hm, population="pop1"),
        allel_fn(['g'], lambda: allel.mean_pairwise_difference(
            _allel_cache['g'].count_alleles(subpop=pop1_dip)))))
    benchmarks.append(("diversity.theta_w",
        lambda: diversity.theta_w(hm, population="pop1"),
        allel_fn(['pos', 'g'], lambda: allel.watterson_theta(
            _allel_cache['pos'], _allel_cache['g'].count_alleles(subpop=pop1_dip)))))
    benchmarks.append(("diversity.tajimas_d",
        lambda: diversity.tajimas_d(hm, population="pop1"),
        allel_fn(['g'], lambda: allel.tajima_d(
            _allel_cache['g'].count_alleles(subpop=pop1_dip)))))
    benchmarks.append(("diversity.haplotype_diversity",
        lambda: diversity.haplotype_diversity(hm, population="pop1"),
        allel_fn(['h1'], lambda: allel.haplotype_diversity(_allel_cache['h1']))))
    benchmarks.append(("diversity.heterozygosity_expected",
        lambda: diversity.heterozygosity_expected(hm, population="pop1"),
        allel_fn(['g'], lambda: np.mean(allel.heterozygosity_expected(
            _allel_cache['g'].count_alleles(subpop=pop1_dip).to_frequencies(), ploidy=2)))))
    benchmarks.append(("diversity.heterozygosity_observed",
        lambda: diversity.heterozygosity_observed(hm, population="pop1"),
        allel_fn(['g'], lambda: np.mean(allel.heterozygosity_observed(_allel_cache['g'].subset(sel1=list(range(N_DIP_PER_POP))))))))
    benchmarks.append(("diversity.inbreeding_coefficient",
        lambda: diversity.inbreeding_coefficient(hm, population="pop1"),
        allel_fn(['g'], lambda: np.nanmean(allel.inbreeding_coefficient(_allel_cache['g'].subset(sel1=list(range(N_DIP_PER_POP))))))))
    benchmarks.append(("diversity.allele_freq_spectrum",
        lambda: diversity.allele_frequency_spectrum(hm, population="pop1"),
        allel_fn(['g'], lambda: allel.sfs(_allel_cache['g'].count_alleles(subpop=pop1_dip)[:, 1]))))
    benchmarks.append(("diversity.segregating_sites",
        lambda: diversity.segregating_sites(hm, population="pop1"),
        None))
    benchmarks.append(("diversity.singleton_count",
        lambda: diversity.singleton_count(hm, population="pop1"),
        None))
    benchmarks.append(("diversity.theta_h",
        lambda: diversity.theta_h(hm, population="pop1"), None))
    benchmarks.append(("diversity.theta_l",
        lambda: diversity.theta_l(hm, population="pop1"), None))
    benchmarks.append(("diversity.fay_wus_h",
        lambda: diversity.fay_wus_h(hm, population="pop1"), None))
    benchmarks.append(("diversity.zeng_e",
        lambda: diversity.zeng_e(hm, population="pop1"), None))
    benchmarks.append(("diversity.max_daf",
        lambda: diversity.max_daf(hm, population="pop1"), None))
    benchmarks.append(("diversity.mu_var",
        lambda: diversity.mu_var(hm, population="pop1"), None))
    benchmarks.append(("diversity.mu_sfs",
        lambda: diversity.mu_sfs(hm, population="pop1"), None))
    benchmarks.append(("diversity.daf_histogram",
        lambda: diversity.daf_histogram(hm, population="pop1"), None))

    # --- Divergence ---
    benchmarks.append(("divergence.fst_hudson",
        lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
        allel_fn(['g'], lambda: allel.hudson_fst(
            _allel_cache['g'].count_alleles(subpop=pop1_dip),
            _allel_cache['g'].count_alleles(subpop=pop2_dip)))))
    benchmarks.append(("divergence.fst_weir_cockerham",
        lambda: divergence.fst_weir_cockerham(hm, "pop1", "pop2"),
        allel_fn(['g'], lambda: allel.weir_cockerham_fst(_allel_cache['g'],
            [pop1_dip, pop2_dip]))))
    benchmarks.append(("divergence.dxy",
        lambda: divergence.dxy(hm, "pop1", "pop2"),
        allel_fn(['pos', 'g'], lambda: allel.sequence_divergence(
            _allel_cache['pos'],
            _allel_cache['g'].count_alleles(subpop=pop1_dip),
            _allel_cache['g'].count_alleles(subpop=pop2_dip)))))
    benchmarks.append(("divergence.fst_nei",
        lambda: divergence.fst_nei(hm, "pop1", "pop2"), None))
    benchmarks.append(("divergence.da",
        lambda: divergence.da(hm, "pop1", "pop2"), None))
    benchmarks.append(("divergence.pbs",
        lambda: divergence.pbs(hm, "pop1", "pop2", "pop3", window_size=50_000), None))

    # --- SFS ---
    benchmarks.append(("sfs.sfs",
        lambda: sfs.sfs(hm, population="pop1"),
        allel_fn(['g'], lambda: allel.sfs(_allel_cache['g'].count_alleles(subpop=pop1_dip)[:, 1]))))
    benchmarks.append(("sfs.sfs_folded",
        lambda: sfs.sfs_folded(hm, population="pop1"),
        allel_fn(['g'], lambda: allel.sfs_folded(_allel_cache['g'].count_alleles(subpop=pop1_dip)))))
    benchmarks.append(("sfs.joint_sfs",
        lambda: sfs.joint_sfs(hm, pop1="pop1", pop2="pop2"),
        allel_fn(['g'], lambda: allel.joint_sfs(
            _allel_cache['g'].count_alleles(subpop=pop1_dip)[:, 1],
            _allel_cache['g'].count_alleles(subpop=pop2_dip)[:, 1]))))

    # --- Selection ---
    benchmarks.append(("selection.garud_h",
        lambda: selection.garud_h(hm, population="pop1"),
        allel_fn(['h1'], lambda: allel.garud_h(_allel_cache['h1']))))
    benchmarks.append(("selection.nsl",
        lambda: selection.nsl(hm, population="pop1"),
        allel_fn(['h1'], lambda: allel.nsl(_allel_cache['h1']))))
    benchmarks.append(("selection.ihs",
        lambda: selection.ihs(hm, population="pop1"),
        allel_fn(['h1', 'pos'], lambda: allel.ihs(_allel_cache['h1'], _allel_cache['pos'], include_edges=True))))
    benchmarks.append(("selection.xpehh",
        lambda: selection.xpehh(hm, "pop1", "pop2"),
        allel_fn(['h1', 'h2', 'pos'], lambda: allel.xpehh(_allel_cache['h1'], _allel_cache['h2'], _allel_cache['pos'], include_edges=True))))
    benchmarks.append(("selection.ehh_decay",
        lambda: selection.ehh_decay(hm, population="pop1"),
        allel_fn(['h1'], lambda: allel.ehh_decay(_allel_cache['h1']))))

    # --- Admixture ---
    benchmarks.append(("admixture.patterson_f2",
        lambda: admixture.patterson_f2(hm, "pop1", "pop2"),
        allel_fn(['g'], lambda: allel.patterson_f2(
            _allel_cache['g'].count_alleles(subpop=pop1_dip),
            _allel_cache['g'].count_alleles(subpop=pop2_dip)))))
    benchmarks.append(("admixture.patterson_f3",
        lambda: admixture.patterson_f3(hm, "pop1", "pop2", "pop3"),
        allel_fn(['g'], lambda: allel.patterson_f3(
            _allel_cache['g'].count_alleles(subpop=pop1_dip),
            _allel_cache['g'].count_alleles(subpop=pop2_dip),
            _allel_cache['g'].count_alleles(subpop=pop3_dip)))))
    benchmarks.append(("admixture.patterson_d",
        lambda: admixture.patterson_d(hm, "pop1", "pop2", "pop3", "pop1"),
        allel_fn(['g'], lambda: allel.patterson_d(
            _allel_cache['g'].count_alleles(subpop=pop1_dip),
            _allel_cache['g'].count_alleles(subpop=pop2_dip),
            _allel_cache['g'].count_alleles(subpop=pop3_dip),
            _allel_cache['g'].count_alleles(subpop=pop1_dip)))))

    # --- Decomposition ---
    # randomized_pca needs O(n_hap * n_var * 8) intermediates; OOM at full-arm scale
    benchmarks.append(("decomposition.randomized_pca",
        lambda: decomposition.randomized_pca(hm, n_components=4, scaler="patterson"),
        None))

    # --- Relatedness ---
    # grm/ibs need GenotypeMatrix which OOMs at full-arm (16GB boolean intermediate)
    benchmarks.append(("relatedness.grm",
        lambda: relatedness.grm(gm), None))
    benchmarks.append(("relatedness.ibs",
        lambda: relatedness.ibs(gm), None))

    # --- LD (small subset) ---
    benchmarks.append(("ld_statistics.zns",
        lambda: ld_statistics.zns(hm_ld), None))
    benchmarks.append(("ld_statistics.omega",
        lambda: ld_statistics.omega(hm_ld), None))

    # --- Distance stats ---
    benchmarks.append(("distance_stats.pairwise_diffs",
        lambda: distance_stats.pairwise_diffs(hm, population="pop1"), None))
    benchmarks.append(("distance_stats.dist_var",
        lambda: distance_stats.dist_var(hm, population="pop1"), None))

    # --- Windowed analysis ---
    benchmarks.append(("windowed pi+tw+td (50kb)",
        lambda: windowed_analysis(hm, window_size=50_000,
            statistics=["pi", "theta_w", "tajimas_d"]),
        allel_fn(['pos', 'g'], lambda: (
            allel.windowed_diversity(_allel_cache['pos'], _allel_cache['g'].count_alleles(), size=50_000, start=win_start, stop=win_stop),
            allel.windowed_watterson_theta(_allel_cache['pos'], _allel_cache['g'].count_alleles(), size=50_000, start=win_start, stop=win_stop),
            allel.windowed_tajima_d(_allel_cache['pos'], _allel_cache['g'].count_alleles(), size=50_000, start=win_start, stop=win_stop)))))
    benchmarks.append(("windowed fst+dxy (50kb)",
        lambda: windowed_analysis(hm, window_size=50_000,
            statistics=["fst", "dxy"], populations=["pop1", "pop2"]),
        allel_fn(['pos', 'g'], lambda: (
            allel.windowed_hudson_fst(_allel_cache['pos'],
                _allel_cache['g'].count_alleles(subpop=pop1_dip),
                _allel_cache['g'].count_alleles(subpop=pop2_dip),
                size=50_000, start=win_start, stop=win_stop),
            allel.windowed_divergence(_allel_cache['pos'],
                _allel_cache['g'].count_alleles(subpop=pop1_dip),
                _allel_cache['g'].count_alleles(subpop=pop2_dip),
                size=50_000, start=win_start, stop=win_stop)))))
    benchmarks.append(("windowed all 7 (100kb)",
        lambda: windowed_analysis(hm, window_size=100_000,
            statistics=["pi", "theta_w", "tajimas_d", "fst", "fst_wc", "dxy", "da"],
            populations=["pop1", "pop2"]),
        None))

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\nRunning {len(benchmarks)} benchmarks...\n", flush=True)
    print(f"{'Statistic':<42s} {'pg_gpu':>9s} {'allel':>9s} {'speedup':>9s}", flush=True)
    print("-" * 72, flush=True)

    results = []
    for name, pg_fn, allel_fn_wrapped in benchmarks:
        t_pg = bench(pg_fn, sync_gpu=True)
        t_al = bench(allel_fn_wrapped, sync_gpu=False) if allel_fn_wrapped else None

        pg_str = f"{t_pg:.3f}s" if t_pg is not None else "FAIL"
        al_str = f"{t_al:.3f}s" if t_al is not None else "---"

        if t_pg is not None and t_al is not None and t_pg > 0:
            speedup = t_al / t_pg
            sp_str = f"{speedup:.1f}x"
        else:
            speedup = None
            sp_str = "---"

        print(f"{name:<42s} {pg_str:>9s} {al_str:>9s} {sp_str:>9s}", flush=True)
        results.append({"name": name, "pg_gpu": t_pg, "allel": t_al, "speedup": speedup})

    # ── Summary ──────────────────────────────────────────────────────────
    print(flush=True)
    n_pass = sum(1 for r in results if r["pg_gpu"] is not None)
    n_fail = len(results) - n_pass
    speedups = [r["speedup"] for r in results if r["speedup"] is not None]
    print(f"pg_gpu: {n_pass}/{len(results)} passed", flush=True)
    if speedups:
        print(f"Speedups ({len(speedups)} compared): "
              f"median {np.median(speedups):.1f}x, "
              f"range {min(speedups):.1f}x - {max(speedups):.1f}x", flush=True)

    # ── Figure ───────────────────────────────────────────────────────────
    compared = [r for r in results if r["speedup"] is not None]
    if not compared:
        return

    sns.set_theme(style="whitegrid", context="talk", palette="Set2")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [2, 1]})

    names = [r["name"] for r in compared]
    speedups_arr = np.array([r["speedup"] for r in compared])
    colors = ["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups_arr]

    ax = axes[0]
    bars = ax.barh(range(len(names)), speedups_arr, color=colors,
                   edgecolor="0.3", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xscale("log")
    ax.axvline(1, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Speedup (pg_gpu vs scikit-allel)")
    ax.set_title(f"pg_gpu speedups on Ag1000G {CHROM} "
                 f"({hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants)")
    ax.invert_yaxis()
    for bar, sp in zip(bars, speedups_arr):
        ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                f"{sp:.1f}x", va="center", fontsize=9)

    ax2 = axes[1]
    x = np.arange(len(names))
    width = 0.35
    ax2.bar(x - width/2, [r["pg_gpu"] for r in compared], width,
            label="pg_gpu", color="#2ecc71", edgecolor="0.3", linewidth=0.5)
    ax2.bar(x + width/2, [r["allel"] for r in compared], width,
            label="scikit-allel", color="#e74c3c", edgecolor="0.3", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.legend()
    ax2.set_title("Absolute wall-clock time")

    plt.tight_layout()
    outpath = "debug/stress_test_ag1000g.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {outpath}", flush=True)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
