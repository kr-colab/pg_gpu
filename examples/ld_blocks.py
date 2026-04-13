#!/usr/bin/env python
"""
LD block partitioning via a bridging-score scan.

Simulates a chromosome with two recombination hotspots, computes the full
pairwise r^2 matrix on the GPU via pg_gpu, then partitions SNPs into LD
blocks by scanning a bridging score along the chromosome and locating
its dips with scipy.signal.find_peaks.

For each candidate boundary k (between SNPs k-1 and k) we compute

    bridge(k) = mean r^2 over (left_set, right_set) pairs
                left_set  = SNPs at indices [k-W, k)
                right_set = SNPs at indices [k, k+W)

Inside a block, both halves of the window live in the same LD region and
the bridging score is high; at a recombination hotspot, LD does not span
the boundary and the score dips. Block boundaries are dips in this trace.


Usage
-----
    pixi run python examples/ld_blocks.py
    pixi run python examples/ld_blocks.py --window 40 --prominence 0.05
    pixi run python examples/ld_blocks.py --no-plot
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks

from pg_gpu import HaplotypeMatrix
from pg_gpu.plotting import plot_pairwise_ld


# ── simulation ──────────────────────────────────────────────────────────────
# Three flat-rate stretches of ~333 kb separated by two narrow hotspots.

SEQ_LEN = 1_000_000
HOTSPOT_1 = (320_000, 340_000)
HOTSPOT_2 = (660_000, 680_000)
RECOMB_LOW = 1e-9
RECOMB_HIGH = 1e-6
MUTATION_RATE = 2e-8
NE = 10_000


def build_recomb_map() -> msprime.RateMap:
    """Piecewise-constant recombination map with two hotspots."""
    return msprime.RateMap(
        position=[0, HOTSPOT_1[0], HOTSPOT_1[1], HOTSPOT_2[0],
                  HOTSPOT_2[1], SEQ_LEN],
        rate=[RECOMB_LOW, RECOMB_HIGH, RECOMB_LOW, RECOMB_HIGH, RECOMB_LOW],
    )


def simulate(n_diploids: int, seed: int):
    ts = msprime.sim_ancestry(
        samples=n_diploids,
        sequence_length=SEQ_LEN,
        recombination_rate=build_recomb_map(),
        population_size=NE,
        ploidy=2,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=seed)


# ── partitioning ────────────────────────────────────────────────────────────

def bridging_score(r2, window):
    """Per-boundary mean r^2 across a sliding (left, right) window pair.

    Returns an array of length n_variants where index k holds the mean r^2
    over pairs (i, j) with i in [k-W, k) and j in [k, k+W). Indices in
    [0, W) and [n-W, n) get NaN (window doesn't fit).
    """
    n = r2.shape[0]
    score = np.full(n, np.nan)
    for k in range(window, n - window):
        score[k] = r2[k - window:k, k:k + window].mean()
    return score


def find_block_boundaries(score, max_score, min_separation):
    """Find breakpoints as local minima whose value is below max_score.

    Hotspots produce dips with bridging score near zero (no LD spans the
    boundary), while within-block fluctuations rarely dip below ~0.05
    even for substantial prominence. Filtering on an absolute height
    threshold separates them cleanly.

    Parameters
    ----------
    score : (n,) ndarray, possibly with NaNs at the edges.
    max_score : float
        Maximum bridging score for a valid breakpoint dip.
    min_separation : int
        Minimum SNP-index distance between detected breakpoints.
    """
    finite = np.isfinite(score)
    # find_peaks wants peaks; invert the signal so dips become peaks.
    inverted = -score.copy()
    inverted[~finite] = -np.inf
    peaks, _ = find_peaks(inverted, height=-max_score,
                          distance=min_separation)
    return peaks


def boundaries_to_blocks(boundary_idx, positions):
    """Turn breakpoint SNP indices into (start_bp, end_bp, n_snps) blocks."""
    n = len(positions)
    edges = np.concatenate([[0], boundary_idx, [n]])
    blocks = []
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            continue
        blocks.append((int(positions[lo]), int(positions[hi - 1]), hi - lo))
    return blocks


# ── plotting ────────────────────────────────────────────────────────────────

def plot_blocks(r2, positions, blocks, score, boundary_idx, outpath):
    sns.set_theme(style="white", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Top: r^2 heatmap with detected boundaries.
    plot_pairwise_ld(r2, ax=axes[0], cmap="Greys", positions=positions)
    for bi in boundary_idx:
        axes[0].axvline(bi, color="#c0392b", linewidth=1.0, linestyle="--")
        axes[0].axhline(bi, color="#c0392b", linewidth=1.0, linestyle="--")
    axes[0].set_title(f"Pairwise $r^2$ with {len(blocks)} detected blocks",
                      fontweight="bold", fontsize=11, loc="left")

    # Middle: bridging score trace (SNP index on x).
    axes[1].plot(np.arange(len(score)), score, color="#2980b9", linewidth=1.0)
    for bi in boundary_idx:
        axes[1].axvline(bi, color="#c0392b", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Variant index")
    axes[1].set_ylabel("Bridging score")
    axes[1].set_title("Mean cross-window $r^2$ (dips = block boundaries)",
                      fontweight="bold", fontsize=11, loc="left")
    axes[1].set_xlim(0, len(score) - 1)

    # Bottom: recombination rate map (Mb on x).
    rmap = build_recomb_map()
    pos_mb = np.asarray(rmap.position) / 1e6
    rates = np.concatenate([rmap.rate, rmap.rate[-1:]])
    axes[2].step(pos_mb, rates, where="post", color="#16a085", linewidth=1.5)
    axes[2].set_yscale("log")
    axes[2].set_xlim(0, SEQ_LEN / 1e6)
    axes[2].set_xlabel("Genomic position (Mb)")
    axes[2].set_ylabel("Recombination rate")
    axes[2].set_title("Recombination map (true block boundaries at hotspots)",
                      fontweight="bold", fontsize=11, loc="left")
    for bi in boundary_idx:
        # boundary SNP index -> bp position
        axes[2].axvline(positions[bi] / 1e6, color="#c0392b", linewidth=1.0,
                        linestyle="--")

    fig.suptitle("LD blocks via bridging-score scan",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {outpath}")


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LD block partitioning demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=30,
                   help="diploid sample size (default: 30)")
    p.add_argument("--window", type=int, default=150,
                   help="bridging-window half-width in SNPs (default: 150)")
    p.add_argument("--max-score", type=float, default=0.03,
                   help="max bridging score for a valid breakpoint dip (default: 0.03)")
    p.add_argument("--min-separation", type=int, default=200,
                   help="min SNP gap between breakpoints (default: 200)")
    p.add_argument("-o", "--output", default="ld_blocks.pdf",
                   help="output figure path (default: ld_blocks.pdf)")
    p.add_argument("--no-plot", action="store_true",
                   help="print results without writing a figure")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Simulating {SEQ_LEN/1e6:.1f} Mb chromosome with hotspots at "
          f"{HOTSPOT_1[0]/1e6:.2f} Mb and {HOTSPOT_2[0]/1e6:.2f} Mb...")
    ts = simulate(args.samples, args.seed)
    hm = HaplotypeMatrix.from_ts(ts)
    positions = np.asarray(hm.positions)
    print(f"  {hm.num_variants:,} variants, {hm.num_haplotypes} haplotypes")

    # GPU pairwise r^2 — this is the part the example is here to showcase.
    print("Computing pairwise r^2 on GPU...")
    r2 = hm.pairwise_r2().get()

    score = bridging_score(r2, args.window)
    boundary_idx = find_block_boundaries(score, args.max_score,
                                         args.min_separation)
    blocks = boundaries_to_blocks(boundary_idx, positions)

    print(f"\nDetected {len(blocks)} blocks "
          f"(window={args.window}, max_score={args.max_score}):")
    for i, (s, e, sz) in enumerate(blocks, 1):
        print(f"  block {i}: [{s/1e6:.2f}, {e/1e6:.2f}] Mb  "
              f"({(e-s)/1e3:.0f} kb, {sz} SNPs)")

    if len(boundary_idx) > 0:
        true_boundaries = [(HOTSPOT_1[0] + HOTSPOT_1[1]) / 2,
                           (HOTSPOT_2[0] + HOTSPOT_2[1]) / 2]
        print("\nBoundary accuracy (detected vs nearest hotspot midpoint):")
        for bi in boundary_idx:
            d = positions[bi]
            nearest = min(true_boundaries, key=lambda t: abs(t - d))
            print(f"  detected {d/1e6:.3f} Mb  ->  hotspot {nearest/1e6:.2f} "
                  f"Mb  (off by {abs(d - nearest)/1e3:.0f} kb)")

    if not args.no_plot:
        plot_blocks(r2, positions, blocks, score, boundary_idx, args.output)


if __name__ == "__main__":
    main()
