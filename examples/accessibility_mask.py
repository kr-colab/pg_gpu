#!/usr/bin/env python
"""
Accessibility masks + sliding-window diversity.

Simulates a 1 Mb chromosome with a 200 kb 'exon'-like block of 100x
lower mutation rate, then computes sliding-window pi with and without
an accessibility mask that excludes that block.

The unmasked trace shows a misleading dip where mutation is low; the
masked trace drops those windows entirely (NaN -> visual gap) or
correctly normalizes surrounding windows by accessible bases only. The
comparison makes the role of the mask obvious without a paragraph of
explanation.

Usage
-----
    pixi run python examples/accessibility_mask.py
    pixi run python examples/accessibility_mask.py --seed 7 --window 200_000
    pixi run python examples/accessibility_mask.py --no-plot
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import msprime
import numpy as np
import seaborn as sns

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis


# ── simulation ──────────────────────────────────────────────────────────────
# A single 10 Mb chromosome with a central low-mutation region acting as
# a stand-in for a low-callability exon: density of segregating sites
# drops there even though the underlying genealogy is identical to the
# flanking neutral sequence.

SEQ_LEN = 1_000_000
EXON_START = 400_000
EXON_END = 600_000
MU_HIGH = 1e-8
MU_LOW = 1e-10  # two orders of magnitude lower inside the "exon"
NE = 10_000
RECOMB_RATE = 1e-8


def build_rate_map() -> msprime.RateMap:
    """Piecewise-constant mutation map with a low-rate central block."""
    # RateMap takes N+1 positions and N rates: high / low / high across
    # the three intervals defined by [0, EXON_START, EXON_END, SEQ_LEN].
    return msprime.RateMap(
        position=[0, EXON_START, EXON_END, SEQ_LEN],
        rate=[MU_HIGH, MU_LOW, MU_HIGH],
    )


def simulate(n_diploids: int, seed: int):
    """Panmictic ancestry + spatially-varying mutation map, return ts."""
    ts = msprime.sim_ancestry(
        samples=n_diploids,
        sequence_length=SEQ_LEN,
        recombination_rate=RECOMB_RATE,
        population_size=NE,
        ploidy=2,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=build_rate_map(), random_seed=seed)


def build_mask() -> np.ndarray:
    """Boolean accessibility mask: True = accessible, False = excluded exon."""
    mask = np.ones(SEQ_LEN, dtype=bool)
    mask[EXON_START:EXON_END] = False
    return mask


# ── windowed pi ─────────────────────────────────────────────────────────────

def run_windowed(hm, label, window, step):
    """Compute sliding-window pi and print a one-line summary."""
    # span_normalize=True (the default) divides each window's sum of
    # per-site pi by its accessible-base count; fully-masked windows
    # come back as NaN, which renders as a gap in the plot below.
    df = windowed_analysis(
        hm, window_size=window, step_size=step,
        statistics=["pi"], span_normalize=True,
    )
    pi = df["pi"].to_numpy()
    finite = np.isfinite(pi)
    mean_pi = pi[finite].mean() if finite.any() else float("nan")
    min_pi = pi[finite].min() if finite.any() else float("nan")
    min_pos_mb = (df["start"].to_numpy()[finite][pi[finite].argmin()] / 1e6
                  if finite.any() else float("nan"))
    n_nan = int((~finite).sum())

    print(f"\n=== {label} ===")
    print(f"  n windows       = {len(df)}  (NaN = {n_nan})")
    print(f"  mean pi         = {mean_pi:.3e}")
    print(f"  min pi          = {min_pi:.3e}  (window at {min_pos_mb:.2f} Mb)")
    return df


# ── plotting ────────────────────────────────────────────────────────────────

def plot_comparison(df_unmasked, df_masked, outpath):
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # The expected neutral per-bp pi at baseline mutation rate — a visual
    # anchor for the flanking regions.
    expected_pi = 4 * NE * MU_HIGH

    for ax, df, title in [
        (axes[0], df_unmasked, "Unmasked — raw sliding-window π"),
        (axes[1], df_masked, "Masked — sliding-window π (exon excluded)"),
    ]:
        centers_mb = 0.5 * (df["start"] + df["stop"]) / 1e6
        ax.plot(centers_mb, df["pi"], color="#2980b9",
                linewidth=1.0, alpha=0.9)
        ax.axhline(expected_pi, color="0.4", linestyle="--", linewidth=0.7,
                   label=f"4 Ne μ = {expected_pi:.1e}")
        # Shade the low-mutation region on both panels.
        ax.add_patch(mpatches.Rectangle(
            (EXON_START / 1e6, 0), (EXON_END - EXON_START) / 1e6, 1.0,
            transform=ax.get_xaxis_transform(), color="#e67e22", alpha=0.15,
            zorder=0,
        ))
        ax.set_ylabel(r"$\pi$ per bp")
        ax.set_title(title, fontweight="bold", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right", frameon=True)

    axes[1].set_xlabel("Genomic position (Mb)")
    axes[0].text(
        (EXON_START + EXON_END) / 2 / 1e6,
        axes[0].get_ylim()[1] * 0.95,
        "low-mutation region\n(excluded by mask)",
        ha="center", va="top", fontsize=8, color="#a04000",
    )

    fig.suptitle("Accessibility masks correct a spurious diversity dip",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {outpath}")


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Accessibility mask + windowed π demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=50,
                   help="diploid sample size (default: 50)")
    p.add_argument("--window", type=int, default=10_000,
                   help="window size in bp (default: 10 kb)")
    p.add_argument("--step", type=int, default=None,
                   help="step size in bp (default: window_size, non-overlapping)")
    p.add_argument("-o", "--output", default="accessibility_mask.pdf",
                   help="output figure path (default: accessibility_mask.pdf)")
    p.add_argument("--no-plot", action="store_true",
                   help="print results without writing a figure")
    return p.parse_args()


def main():
    args = parse_args()
    # Non-overlapping windows: pg_gpu's fused scatter kernel has a known
    # issue with overlapping windows (tracked upstream as issue #64) that
    # undercounts variants in every window. step == window avoids it.
    step = args.step if args.step is not None else args.window

    print(f"Simulating {SEQ_LEN/1e6:.1f} Mb chromosome with low-mutation "
          f"region at {EXON_START/1e3:.0f}–{EXON_END/1e3:.0f} kb "
          f"(μ_low / μ_high = {MU_LOW/MU_HIGH:.0e})...")
    ts = simulate(args.samples, args.seed)
    print(f"  {ts.num_sites:,} variants, {2*args.samples} haplotypes")

    # Two handles onto the same tree sequence: one unmasked, one with
    # the exon region flagged inaccessible via an in-memory bool array.
    hm_unmasked = HaplotypeMatrix.from_ts(ts)
    hm_masked = HaplotypeMatrix.from_ts(ts).set_accessible_mask(build_mask())

    df_unmasked = run_windowed(hm_unmasked, "Unmasked", args.window, step)
    df_masked = run_windowed(hm_masked, "Masked (exon excluded)",
                             args.window, step)

    if not args.no_plot:
        plot_comparison(df_unmasked, df_masked, args.output)


if __name__ == "__main__":
    main()
