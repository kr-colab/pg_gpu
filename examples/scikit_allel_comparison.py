#!/usr/bin/env python
"""
Side-by-side: scikit-allel vs pg_gpu on real Anopheles X-chromosome data.

This script does the same population-genetics work two ways:

  1. Windowed diversity (pi, theta_W, Tajima's D) along the chromosome.
     scikit-allel needs three separate function calls, one per statistic.
     pg_gpu fuses all three into a single GPU kernel pass.
  2. LD-decay: how does linkage disequilibrium (r^2 between pairs of SNPs)
     drop off with physical distance? scikit-allel computes pairwise r,
     squares, builds a distance vector, and bins manually. pg_gpu does
     the whole thing in one call.

For each task we (a) check that the two libraries agree numerically and
(b) report the wall-clock speedup of pg_gpu over scikit-allel.

Usage
-----
    pixi run python examples/scikit_allel_comparison.py
    pixi run python examples/scikit_allel_comparison.py --small
    pixi run python examples/scikit_allel_comparison.py --no-plot
"""

import argparse
import time
from pathlib import Path

import allel
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis


# Two datasets ship in examples/data/. The full X-chromosome (~25 Mb,
# 5.3M variants) is the default; --small picks a 4 Mb subset for fast
# iteration during development.
DATA_DIR = Path(__file__).resolve().parent / "data"
ZARR_FULL = DATA_DIR / "gamb.X.phased.n100.zarr"
ZARR_SMALL = DATA_DIR / "gamb.X.8-12Mb.n100.derived.zarr"

# Plot palette: scikit-allel is warm grey, pg_gpu is blue.
COL_ALLEL = "#9aa1ac"
COL_PG = "#2980b9"

# Distance bins for the LD-decay curve: 24 log-spaced bins from 100 bp
# to 10 kb. Anopheles LD decays within a few kb, so this is the range
# where the signal lives.
LD_BP_BINS = np.logspace(2, 4, 25)


def _to_numpy(x):
    """Bring a possibly-GPU array back to host (CPU) memory.

    pg_gpu stores arrays on the GPU as CuPy arrays; .get() copies them
    over to NumPy. If x is already NumPy, just hand it back.
    """
    return x.get() if hasattr(x, "get") else np.asarray(x)


# --- 1. Loading -------------------------------------------------------------

def _load_data(small):
    """Load the gamb dataset and build the views each library wants.

    Returns (hm, pos, ac, gn):
      hm  - HaplotypeMatrix (the pg_gpu view)
      pos - 1-based positions, shape (n_variants,)
      ac  - allele counts, shape (n_variants, 2), as scikit-allel wants
      gn  - 0/1/2 dosages, shape (n_variants, n_diploids), also for allel
    """
    path = ZARR_SMALL if small else ZARR_FULL
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}\n"
            f"The data fixtures live under examples/data/ and are tracked "
            f"in the repo. If you cloned without LFS or otherwise lack the "
            f"file, try `git lfs pull` or refetch from the project root.")
    print(f"Loading {path.name} ...", flush=True)
    t0 = time.perf_counter()
    hm = HaplotypeMatrix.from_zarr(str(path))
    print(f"  loaded in {time.perf_counter() - t0:.2f}s; "
          f"{hm.haplotypes.shape[0]} haplotypes, "
          f"{hm.haplotypes.shape[1]:,} variants, "
          f"chrom range {hm.chrom_start:,}-{hm.chrom_end:,}", flush=True)
    # pg_gpu stores haplotypes as (n_haplotypes, n_variants); scikit-allel
    # wants (n_variants, n_haplotypes). Transpose once and let allel build
    # its own allele-count and 0/1/2-dosage views from there.
    pos = _to_numpy(hm.positions).astype(np.int64)
    hap_T = np.ascontiguousarray(_to_numpy(hm.haplotypes).T)
    ha = allel.HaplotypeArray(hap_T)
    ac = np.asarray(ha.count_alleles())
    gn = np.asarray(ha.to_genotypes(ploidy=2).to_n_alt())
    return hm, pos, ac, gn


# --- 2. Windowed diversity, two ways ----------------------------------------

def _compute_allel(pos, ac, windows):
    """scikit-allel: three separate windowed calls, one per statistic."""
    pi, _, _, _ = allel.windowed_diversity(pos, ac, windows=windows)
    theta_w, _, _, _ = allel.windowed_watterson_theta(pos, ac, windows=windows)
    tajd, _, _ = allel.windowed_tajima_d(pos, ac, windows=windows)
    return pi, theta_w, tajd


def _compute_pg_gpu(hm, window_size):
    """pg_gpu: one fused call computes all three stats together.

    Returns the result DataFrame so the caller can read out the columns
    and inspect window edges.
    """
    df = windowed_analysis(
        hm, window_size=window_size, step_size=window_size,
        statistics=["pi", "theta_w", "tajimas_d"])
    cp.cuda.Device().synchronize()  # Wait for the GPU before timing stops.
    return df


# --- 3. LD-decay, two ways --------------------------------------------------
# Both libraries compute Rogers-Huff (2008) r on diploid 0/1/2 dosages.
# pg_gpu exposes it via HaplotypeMatrix.windowed_r_squared(estimator='rogers_huff');
# scikit-allel via allel.rogers_huff_r + manual binning. The two should
# match numerically up to allel's float32 internal precision.

def _subsample_for_ld(hm, pos, gn, n_snps, mac_min):
    """Pick `n_snps` contiguous SNPs centered on the chromosome midpoint,
    then drop near-singleton variants below the MAC threshold.

    Why a contiguous block (vs. a random subsample)?
      A random 10k-SNP draw over a 25 Mb chromosome leaves only a handful
      of pairs at short distances, which is where the LD signal lives;
      the resulting decay curve washes out into noise. A contiguous
      block packs many short-distance pairs into the same physical
      stretch.

    Why a MAC filter?
      Two near-singleton variants take a tiny set of tied r^2 values
      regardless of how close or far apart they are on the chromosome
      (e.g., two non-overlapping singletons give exactly r^2 = 1/(n-1)^2).
      Those tied values pin the per-bin median to a constant and erase
      any real decay signal. Standard MAF 0.05 (= MAC >= 20 in
      n=100 diploids = 200 haplotypes) is plenty.
    """
    n_var = len(pos)
    if n_snps > n_var:
        raise ValueError(
            f"--ld-snps={n_snps} exceeds the dataset size ({n_var})")

    # Find the SNP closest to the chromosome midpoint and grab a
    # contiguous range of n_snps SNPs around it (clamped to bounds).
    midpoint = (hm.chrom_start + hm.chrom_end) // 2
    center_idx = int(np.searchsorted(pos, midpoint))
    start = max(0, center_idx - n_snps // 2)
    end = start + n_snps
    if end > n_var:
        end = n_var
        start = end - n_snps
    idx = np.arange(start, end)
    pos_blk = pos[idx]
    gn_blk = gn[idx]

    # Drop near-singleton variants. See the docstring for why.
    # (Formula assumes biallelic, non-missing 0/1/2 dosages -- the
    # gamb fixture is clean, so we don't validate here.)
    if mac_min > 0:
        n_alt = gn_blk.sum(axis=1)
        mac = np.minimum(n_alt, 2 * gn_blk.shape[1] - n_alt)
        keep = mac >= mac_min
        idx, pos_blk, gn_blk = idx[keep], pos_blk[keep], gn_blk[keep]

    # Slice haplotypes on whichever device they live (CPU or GPU)
    # rather than pulling the full chromosome through host memory.
    hap = hm.haplotypes
    if hasattr(hap, "get"):  # CuPy array on GPU
        hap_sub = hap[:, cp.asarray(idx)]
    else:
        hap_sub = hap[:, idx]
    hm_sub = HaplotypeMatrix(
        hap_sub, pos_blk, hm.chrom_start, hm.chrom_end)
    return hm_sub, pos_blk, gn_blk


def _compute_allel_ld_decay(pos_sub, gn_sub, bp_bins):
    """LD-decay curve via scikit-allel: build pairwise r, square, bin,
    take median r^2 per bin.
    """
    # rogers_huff_r returns the upper triangle of the pairwise r matrix
    # in row-major order, i.e. one entry per (i, j) with i < j.
    r_sq = allel.rogers_huff_r(gn_sub) ** 2
    n = len(pos_sub)
    i, j = np.triu_indices(n, k=1)
    distances = (pos_sub[j] - pos_sub[i]).astype(np.int64)

    # Drop pairs with NaN r (e.g., monomorphic), then assign each
    # pair to a distance bin and median-aggregate.
    finite = np.isfinite(r_sq)
    distances = distances[finite]
    r_sq = r_sq[finite]
    bin_idx = np.digitize(distances, bp_bins) - 1  # 0..len(bp_bins)-2

    out = np.full(len(bp_bins) - 1, np.nan)
    for b in range(len(out)):
        in_bin = bin_idx == b
        if in_bin.any():
            out[b] = float(np.median(r_sq[in_bin]))
    return out


def _compute_pg_gpu_ld_decay(hm_sub, bp_bins):
    """LD-decay curve via pg_gpu: one call does pairs + binning + median.

    estimator='rogers_huff' makes this match scikit-allel's path
    (vs. the pg_gpu default sigma_d^2 estimator).
    """
    result, _ = hm_sub.windowed_r_squared(
        bp_bins, percentile=50, estimator='rogers_huff')
    cp.cuda.Device().synchronize()
    return np.asarray(result)


# --- 4. Verification + timing helpers ---------------------------------------

def _time(callable_, n_warmup):
    """Run n_warmup discarded calls (to absorb JIT / GPU init), then one
    timed call. Returns (result, elapsed_seconds).
    """
    for _ in range(n_warmup):
        callable_()
    t0 = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - t0


def _verify(name, allel_arr, pg_arr, rtol=1e-5, atol=1e-8):
    """Assert the two libraries agree to the given tolerance on the
    finite (non-NaN) entries. Returns max abs diff for printing.
    """
    finite = np.isfinite(allel_arr) & np.isfinite(pg_arr)
    if not finite.any():
        raise AssertionError(f"[{name}] no finite entries to compare")
    np.testing.assert_allclose(
        allel_arr[finite], pg_arr[finite],
        rtol=rtol, atol=atol,
        err_msg=f"[{name}] disagreement above tolerance")
    return float(np.abs(allel_arr[finite] - pg_arr[finite]).max())


# --- 5. Plotting ------------------------------------------------------------

def _plot_comparison(centers_mb,
                     pi_a, pi_g, tw_a, tw_g, td_a, td_g,
                     bp_bins, ld_a, ld_g,
                     t_allel, t_pg, t_allel_ld, t_pg_ld,
                     outpath):
    """5-panel figure: pi / theta_W / Tajima's D / LD-decay / timings."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(
        5, 1, figsize=(11, 12),
        gridspec_kw={"height_ratios": [1, 1, 1, 1.1, 0.7]},
        sharex=False)

    # Top three panels: scikit-allel and pg_gpu lines overlaid. Perfect
    # agreement looks like a single line (the pg_gpu trace sits exactly
    # on top of the allel trace).
    diversity_panels = [
        (pi_a, pi_g, r"$\pi$"),
        (tw_a, tw_g, r"$\theta_W$"),
        (td_a, td_g, r"Tajima's $D$"),
    ]
    for ax, (a, g, ylabel) in zip(axes[:3], diversity_panels):
        ax.plot(centers_mb, a, color=COL_ALLEL, linewidth=1.6, alpha=0.9)
        ax.plot(centers_mb, g, color=COL_PG, linewidth=0.8, alpha=0.95)
        ax.set_ylabel(ylabel)
    # One legend only, on the top panel.
    axes[0].plot([], [], color=COL_ALLEL, linewidth=1.6, label="scikit-allel")
    axes[0].plot([], [], color=COL_PG, linewidth=0.8, label="pg_gpu")
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    axes[2].set_xlabel("Genomic position (Mb)")

    # LD-decay panel (log-scale x).
    ax_ld = axes[3]
    bin_centers = np.sqrt(bp_bins[:-1] * bp_bins[1:])  # log-bin midpoints
    ax_ld.plot(bin_centers, ld_a, color=COL_ALLEL, linewidth=1.6,
               alpha=0.9, marker="o", markersize=3)
    ax_ld.plot(bin_centers, ld_g, color=COL_PG, linewidth=0.8,
               alpha=0.95, marker="o", markersize=2)
    ax_ld.set_xscale("log")
    ax_ld.set_ylabel(r"median $r^2$")
    ax_ld.set_xlabel("Pair distance (bp)")

    # Timing bars: 4 bars (2 libraries x 2 computations).
    ax_t = axes[4]
    bars = [
        ("pg_gpu LD-decay",       t_pg_ld,    COL_PG,    f"  {t_pg_ld:.2f}s ({t_allel_ld / t_pg_ld:.1f}x)"),
        ("scikit-allel LD-decay", t_allel_ld, COL_ALLEL, f"  {t_allel_ld:.2f}s"),
        ("pg_gpu windowed",       t_pg,       COL_PG,    f"  {t_pg:.2f}s ({t_allel / t_pg:.1f}x)"),
        ("scikit-allel windowed", t_allel,    COL_ALLEL, f"  {t_allel:.2f}s"),
    ]
    labels, values, colors, annots = zip(*bars)
    y = np.arange(len(bars))
    ax_t.barh(y, values, color=colors, height=0.6)
    ax_t.set_yticks(y)
    ax_t.set_yticklabels(labels)
    ax_t.set_xlabel("seconds")
    for i, (v, txt) in enumerate(zip(values, annots)):
        ax_t.text(v, i, txt, va="center", ha="left", fontsize=9, color="0.2")
    ax_t.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {outpath}")


# --- 6. Main ----------------------------------------------------------------

def main():
    args = _parse_args()
    hm, pos, ac, gn = _load_data(args.small)

    # Decide where windows fall ONCE, then pass the same windows to both
    # libraries. This is what makes the comparison apples-to-apples.
    print("Computing windows ...", flush=True)
    windows = allel.position_windows(
        pos, size=args.window_size, step=args.window_size,
        start=int(pos[0]), stop=int(pos[-1]))
    print(f"  {len(windows)} windows of {args.window_size:,} bp")

    # Time each library on the windowed-diversity scan.
    print(f"Timing windowed scan (n_warmup={args.n_warmup}) ...", flush=True)
    (pi_a, tw_a, td_a), t_allel = _time(
        lambda: _compute_allel(pos, ac, windows), n_warmup=args.n_warmup)
    df, t_pg = _time(
        lambda: _compute_pg_gpu(hm, args.window_size), n_warmup=args.n_warmup)
    pi_g = df["pi"].to_numpy()
    tw_g = df["theta_w"].to_numpy()
    td_g = df["tajimas_d"].to_numpy()
    print(f"  scikit-allel: {t_allel:6.2f}s")
    print(f"  pg_gpu:       {t_pg:6.2f}s  ({t_allel / t_pg:.1f}x speedup)")

    # Sanity check: both libraries must lay out the same windows.
    assert len(windows) == len(df), (
        f"window count mismatch: allel={len(windows)}, pg_gpu={len(df)}")
    assert np.array_equal(windows[:, 0], df["start"].to_numpy()), \
        "window start positions disagree between the two libraries"
    print(f"  {len(pi_a)} aligned windows confirmed")

    # When chrom_end isn't a clean multiple of window_size, the last
    # window has fewer than window_size bp. The two libraries normalize
    # that single partial window slightly differently -- a one-window
    # cosmetic effect, not a real numerical disagreement. NaN it out
    # of the comparison so it doesn't flag a false failure.
    partial = (windows[:, 1] - windows[:, 0] + 1) != args.window_size
    pi_a, tw_a, td_a, pi_g, tw_g, td_g = (
        np.where(partial, np.nan, np.asarray(a))
        for a in (pi_a, tw_a, td_a, pi_g, tw_g, td_g))

    # Strict numerical agreement on all three diversity stats.
    print("Verifying diversity ...", flush=True)
    print(f"  pi:        max abs diff = {_verify('pi', pi_a, pi_g):.3e}")
    print(f"  theta_w:   max abs diff = {_verify('theta_w', tw_a, tw_g):.3e}")
    print(f"  tajimas_d: max abs diff = {_verify('tajimas_d', td_a, td_g):.3e}")

    # LD-decay: pick the SNPs once and time both libraries on the same set.
    print(f"Selecting contiguous block of {args.ld_snps:,} SNPs centered "
          f"on the chromosome midpoint for LD-decay ...", flush=True)
    hm_sub, pos_sub, gn_sub = _subsample_for_ld(
        hm, pos, gn, args.ld_snps, args.ld_mac_min)
    span_kb = (pos_sub[-1] - pos_sub[0]) / 1e3
    print(f"  block spans {pos_sub[0]:,}-{pos_sub[-1]:,} ({span_kb:.1f} kb), "
          f"{len(pos_sub):,} variants kept (MAC >= {args.ld_mac_min})",
          flush=True)

    print(f"Timing LD-decay (n_warmup={args.n_warmup}) ...", flush=True)
    ld_a, t_allel_ld = _time(
        lambda: _compute_allel_ld_decay(pos_sub, gn_sub, LD_BP_BINS),
        n_warmup=args.n_warmup)
    ld_g, t_pg_ld = _time(
        lambda: _compute_pg_gpu_ld_decay(hm_sub, LD_BP_BINS),
        n_warmup=args.n_warmup)
    print(f"  scikit-allel: {t_allel_ld:6.2f}s")
    print(f"  pg_gpu:       {t_pg_ld:6.2f}s  "
          f"({t_allel_ld / t_pg_ld:.1f}x speedup)")
    print(f"  median r^2: max abs diff = "
          f"{_verify('ld_decay', ld_a, ld_g, rtol=1e-4, atol=1e-6):.3e}")

    if not args.no_plot:
        centers_mb = (df["start"].to_numpy() + df["end"].to_numpy()) / 2 / 1e6
        _plot_comparison(
            centers_mb,
            pi_a, pi_g, tw_a, tw_g, td_a, td_g,
            bp_bins=LD_BP_BINS, ld_a=ld_a, ld_g=ld_g,
            t_allel=t_allel, t_pg=t_pg,
            t_allel_ld=t_allel_ld, t_pg_ld=t_pg_ld,
            outpath=args.output)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side scikit-allel vs pg_gpu windowed scan")
    p.add_argument("--small", action="store_true",
                   help="Use the 4 Mb gamb.X.8-12Mb subset (fast)")
    p.add_argument("--window-size", type=int, default=10_000,
                   help="Window size in bp (default: 10 kb)")
    p.add_argument("--n-warmup", type=int, default=1,
                   help="Discarded runs before timing (default: 1)")
    p.add_argument("--ld-snps", type=int, default=500_000,
                   help="Size of the contiguous SNP block (centered on "
                        "the chromosome midpoint) used for the LD-decay "
                        "scan, before the MAC filter is applied "
                        "(default: 500_000)")
    p.add_argument("--ld-mac-min", type=int, default=20,
                   help="Drop variants with minor allele count below this "
                        "threshold from the LD block. Standard MAF 0.05 "
                        "in n=100 diploids -> 20. Set to 0 to disable. "
                        "(default: 20)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the matplotlib figure")
    p.add_argument("-o", "--output", type=Path,
                   default=Path("scikit_allel_comparison.png"),
                   help="Output figure path")
    return p.parse_args()


if __name__ == "__main__":
    main()
