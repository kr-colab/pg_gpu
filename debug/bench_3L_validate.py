#!/usr/bin/env python
"""
Benchmark + validation on Ag1000G 3L arm.

Benchmarks pg_gpu vs scikit-allel with fair timing (includes allele counting),
AND spot-checks that values match between implementations.

Usage:
    pixi run python debug/bench_3L_validate.py
"""

import time
import numpy as np
import cupy as cp
import allel
import zarr

from pg_gpu import (
    HaplotypeMatrix, diversity, divergence, selection, sfs, admixture,
    windowed_analysis,
)

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
CHROM = "3L"
N_DIP_PER_POP = 100


def load():
    print(f"Loading {CHROM} full arm...", flush=True)
    t0 = time.time()
    store = zarr.open(ZARR_PATH, mode='r')
    chrom = store[CHROM]
    positions = np.array(chrom['variants/POS'])
    gt = np.array(chrom['calldata/GT'])
    samples = np.array(chrom['samples'])
    n_v, n_s, _ = gt.shape

    hap = np.empty((n_v, 2 * n_s), dtype=gt.dtype)
    hap[:, :n_s] = gt[:, :, 0]
    hap[:, n_s:] = gt[:, :, 1]
    hap = hap.T

    hm = HaplotypeMatrix(hap, positions, int(positions[0]), int(positions[-1]),
                          samples=list(samples))
    # Map diploid sample indices to haplotype indices:
    # diploid i -> haplotype i (allele 1) and haplotype i + n_samples (allele 2)
    def dip_to_hap(dip_indices):
        hap = []
        for i in dip_indices:
            hap.append(i)
            hap.append(i + n_s)
        return hap

    hm.sample_sets = {
        "pop1": dip_to_hap(range(0, N_DIP_PER_POP)),
        "pop2": dip_to_hap(range(N_DIP_PER_POP, 2 * N_DIP_PER_POP)),
        "pop3": dip_to_hap(range(2 * N_DIP_PER_POP, 3 * N_DIP_PER_POP)),
    }
    print(f"  {hm.num_haplotypes} haplotypes x {hm.num_variants:,} variants ({time.time()-t0:.0f}s)",
          flush=True)

    hm.transfer_to_gpu()
    cp.cuda.Stream.null.synchronize()
    print(f"  On GPU", flush=True)
    return hm, gt, positions, n_s


def bench(fn, sync=True):
    if sync:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if sync:
        cp.cuda.Stream.null.synchronize()
    return result, time.perf_counter() - t0


def validate_scalar(name, pg_val, allel_val, rtol=0.05):
    """Check scalar values match within tolerance."""
    pg_val = float(np.asarray(pg_val).flat[0]) if hasattr(pg_val, '__len__') else float(pg_val)
    allel_val = float(np.asarray(allel_val).flat[0]) if hasattr(allel_val, '__len__') else float(allel_val)
    if np.isnan(pg_val) and np.isnan(allel_val):
        return True, ""
    if abs(allel_val) < 1e-15:
        ok = abs(pg_val) < 1e-10
    else:
        ok = abs(pg_val - allel_val) / abs(allel_val) < rtol
    msg = f"pg={pg_val:.6g} allel={allel_val:.6g}" if not ok else ""
    return ok, msg


def validate_array(name, pg_arr, allel_arr, rtol=0.05):
    """Check array values match (correlation + max relative error)."""
    both = np.isfinite(pg_arr) & np.isfinite(allel_arr) & (np.abs(allel_arr) > 1e-12)
    if np.sum(both) == 0:
        return True, "no valid comparisons"
    corr = np.corrcoef(pg_arr[both], allel_arr[both])[0, 1]
    rel_err = np.median(np.abs(pg_arr[both] - allel_arr[both]) / np.abs(allel_arr[both]))
    ok = corr > 0.99 and rel_err < rtol
    msg = f"corr={corr:.4f} med_rel_err={rel_err:.4f}" if not ok else f"corr={corr:.4f}"
    return ok, msg


def main():
    hm, gt, positions, n_samples = load()

    pop1_dip = list(range(N_DIP_PER_POP))
    pop2_dip = list(range(N_DIP_PER_POP, 2 * N_DIP_PER_POP))

    # Build allel genotype array (lazy-ish)
    print("Building allel GenotypeArray...", flush=True)
    g = allel.GenotypeArray(gt)
    pos_allel = allel.SortedIndex(positions)
    del gt

    # Build allel haplotype subsets for selection scans
    # allel haplotype layout: diploid i -> hap 2*i (allele 1) and 2*i+1 (allele 2)
    h_allel = g.to_haplotypes()
    pop1_hap = []
    for i in range(N_DIP_PER_POP):
        pop1_hap.extend([2*i, 2*i+1])
    pop2_hap = []
    for i in range(N_DIP_PER_POP, 2*N_DIP_PER_POP):
        pop2_hap.extend([2*i, 2*i+1])
    h1_allel = h_allel.subset(sel1=pop1_hap)
    h2_allel = h_allel.subset(sel1=pop2_hap)
    del h_allel

    print(f"\n{'Statistic':<40s} {'pg_gpu':>8s} {'allel':>8s} {'speedup':>8s} {'match':>8s}",
          flush=True)
    print("-" * 78, flush=True)

    results = []

    def run(name, pg_fn, allel_fn, validate_fn):
        pg_result, t_pg = bench(pg_fn, sync=True)
        allel_result, t_al = bench(allel_fn, sync=False)
        speedup = t_al / t_pg if t_pg > 0 else float('inf')
        ok, detail = validate_fn(pg_result, allel_result)
        status = "OK" if ok else "MISMATCH"
        print(f"{name:<40s} {t_pg:>7.3f}s {t_al:>7.3f}s {speedup:>7.1f}x {status:>8s}  {detail}",
              flush=True)
        results.append({"name": name, "pg_gpu": t_pg, "allel": t_al,
                        "speedup": speedup, "ok": ok})

    # ── Diversity (scalar) ───────────────────────────────────────────────
    # Both span-normalized for comparison
    run("diversity.pi",
        lambda: diversity.pi(hm, population="pop1"),
        lambda: np.nansum(allel.mean_pairwise_difference(
            g.count_alleles(subpop=pop1_dip))) / (positions[-1] - positions[0]),
        lambda pg, al: validate_scalar("pi", pg, al))

    run("diversity.theta_w",
        lambda: diversity.theta_w(hm, population="pop1"),
        lambda: allel.watterson_theta(pos_allel, g.count_alleles(subpop=pop1_dip)),
        lambda pg, al: validate_scalar("theta_w", pg, al))

    run("diversity.tajimas_d",
        lambda: diversity.tajimas_d(hm, population="pop1"),
        lambda: allel.tajima_d(g.count_alleles(subpop=pop1_dip)),
        lambda pg, al: validate_scalar("tajimas_d", pg, al))

    # ── Divergence ───────────────────────────────────────────────────────
    # allel.hudson_fst returns (num_array, den_array); global FST = sum(num)/sum(den)
    def _allel_hudson_fst():
        ac1 = g.count_alleles(subpop=pop1_dip)
        ac2 = g.count_alleles(subpop=pop2_dip)
        num, den = allel.hudson_fst(ac1, ac2)
        return np.nansum(num) / np.nansum(den)

    run("divergence.fst_hudson",
        lambda: divergence.fst_hudson(hm, "pop1", "pop2"),
        _allel_hudson_fst,
        lambda pg, al: validate_scalar("fst_hudson", pg, al))

    # allel.weir_cockerham_fst returns (a, b, c) per-variant; global = sum(a)/sum(a+b+c)
    def _allel_wc_fst():
        a, b, c = allel.weir_cockerham_fst(g, [pop1_dip, pop2_dip])
        return np.nansum(a) / np.nansum(a + b + c)

    run("divergence.fst_weir_cockerham",
        lambda: divergence.fst_weir_cockerham(hm, "pop1", "pop2"),
        _allel_wc_fst,
        lambda pg, al: validate_scalar("fst_wc", pg, al))

    # allel.sequence_divergence normalizes by genomic span (bp); compare unnormalized
    def _allel_dxy():
        ac1 = g.count_alleles(subpop=pop1_dip)
        ac2 = g.count_alleles(subpop=pop2_dip)
        # sequence_divergence = sum(per_site_dxy) / (stop - start)
        # We want un-normalized: just mean per-site dxy
        an1 = ac1.sum(axis=1)
        an2 = ac2.sum(axis=1)
        d = allel.mean_pairwise_difference_between(ac1, ac2, an1, an2)
        return np.nanmean(d)

    run("divergence.dxy",
        lambda: divergence.dxy(hm, "pop1", "pop2"),
        _allel_dxy,
        lambda pg, al: validate_scalar("dxy", pg, al))

    # ── SFS ──────────────────────────────────────────────────────────────
    run("sfs.sfs",
        lambda: sfs.sfs(hm, population="pop1"),
        lambda: allel.sfs(g.count_alleles(subpop=pop1_dip)[:, 1]),
        lambda pg, al: validate_array("sfs", pg.astype(float), al.astype(float)))

    run("sfs.joint_sfs",
        lambda: sfs.joint_sfs(hm, pop1="pop1", pop2="pop2"),
        lambda: allel.joint_sfs(g.count_alleles(subpop=pop1_dip)[:, 1],
                                 g.count_alleles(subpop=pop2_dip)[:, 1]),
        lambda pg, al: validate_array("joint_sfs", pg.flatten().astype(float),
                                       al.flatten().astype(float)))

    # ── Admixture ────────────────────────────────────────────────────────
    # allel.patterson_f2 returns (num, den)
    def _allel_f2():
        ac1 = g.count_alleles(subpop=pop1_dip)
        ac2 = g.count_alleles(subpop=pop2_dip)
        result = allel.patterson_f2(ac1, ac2)
        # Returns (num, den, n_pairs) or just mean depending on version
        if isinstance(result, tuple):
            return np.nansum(result[0]) / np.nansum(result[1])
        return np.nanmean(result)

    run("admixture.patterson_f2",
        lambda: np.nanmean(admixture.patterson_f2(hm, "pop1", "pop2")),
        _allel_f2,
        lambda pg, al: validate_scalar("f2", pg, al))

    # ── Selection scans ──────────────────────────────────────────────────
    run("selection.garud_h",
        lambda: selection.garud_h(hm, population="pop1"),
        lambda: allel.garud_h(h1_allel),
        lambda pg, al: validate_scalar("garud_h1", pg[0], al[0]))

    run("selection.nsl",
        lambda: selection.nsl(hm, population="pop1"),
        lambda: allel.nsl(h1_allel),
        lambda pg, al: validate_array("nsl", pg, al))

    run("selection.ehh_decay",
        lambda: selection.ehh_decay(hm, population="pop1"),
        lambda: allel.ehh_decay(h1_allel),
        lambda pg, al: validate_array("ehh_decay", pg, al))

    # ── Windowed ─────────────────────────────────────────────────────────
    ws, we = int(positions[0]), int(positions[-1])

    def _pg_windowed():
        return windowed_analysis(hm, window_size=50_000,
                                 statistics=["pi", "theta_w", "tajimas_d"])

    def _allel_windowed():
        ac = g.count_alleles()
        result_pi = allel.windowed_diversity(pos_allel, ac, size=50_000, start=ws, stop=we)
        result_tw = allel.windowed_watterson_theta(pos_allel, ac, size=50_000, start=ws, stop=we)
        return result_pi[0], result_tw[0]

    run("windowed pi+tw+td (50kb)",
        _pg_windowed,
        _allel_windowed,
        lambda pg, al: validate_array("windowed_pi",
            pg["pi"].values, al[0]))

    def _pg_windowed_fst():
        return windowed_analysis(hm, window_size=50_000,
                                 statistics=["fst", "dxy"],
                                 populations=["pop1", "pop2"])

    def _allel_windowed_fst():
        ac1 = g.count_alleles(subpop=pop1_dip)
        ac2 = g.count_alleles(subpop=pop2_dip)
        result = allel.windowed_hudson_fst(pos_allel, ac1, ac2, size=50_000, start=ws, stop=we)
        return result[0]

    run("windowed fst+dxy (50kb)",
        _pg_windowed_fst,
        _allel_windowed_fst,
        lambda pg, al: validate_array("windowed_fst",
            pg["fst"].values, al))

    # ── Summary ──────────────────────────────────────────────────────────
    print(flush=True)
    n_total = len(results)
    n_ok = sum(r["ok"] for r in results)
    speedups = [r["speedup"] for r in results]
    print(f"Validation: {n_ok}/{n_total} matched", flush=True)
    print(f"Speedups: median {np.median(speedups):.1f}x, "
          f"range {min(speedups):.1f}x - {max(speedups):.1f}x", flush=True)

    if n_ok < n_total:
        print("\nMismatches:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['name']}")


if __name__ == "__main__":
    main()
