#!/usr/bin/env python
"""
Genome scan workflow with pg_gpu.

Loads genotype data from a VCF or zarr store, optionally assigns
populations from a tab-delimited file, computes windowed and scalar
statistics on the GPU, and produces a multi-panel genome scan figure.

Usage
-----
    # from VCF, single population
    python genome_scan.py data.vcf.gz --region 3R

    # from zarr, two populations
    python genome_scan.py data.zarr --region 3R --pop-file pops.tsv

    # with accessible mask and custom windows
    python genome_scan.py data.vcf.gz --region 3R --pop-file pops.tsv \
        --accessible-bed mask.bed --window-size 50000 --step-size 10000

The population file should be tab-delimited with columns: sample, pop.
When two populations are present, divergence statistics (Fst, Dxy) and
joint SFS are included automatically.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pg_gpu import HaplotypeMatrix, diversity, divergence, sfs, windowed_analysis


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="pg_gpu genome scan workflow")
    p.add_argument("input", help="path to VCF (.vcf, .vcf.gz, .bcf) or zarr store")
    p.add_argument("--region", help="genomic region, e.g. '3R' or '3R:1000000-5000000'")
    p.add_argument("--pop-file", help="tab-delimited file (sample, pop)")
    p.add_argument("--accessible-bed", help="BED file of accessible regions")
    p.add_argument("--window-size", type=int, default=100_000)
    p.add_argument("--step-size", type=int, default=10_000)
    p.add_argument("-o", "--output", default="genome_scan.pdf",
                   help="output figure path (default: genome_scan.pdf)")
    return p.parse_args()


def load_data(args):
    """Load VCF or zarr, optionally assign populations."""
    path = args.input
    if path.endswith((".vcf", ".vcf.gz", ".bcf")):
        hm = HaplotypeMatrix.from_vcf(
            path, region=args.region, accessible_bed=args.accessible_bed,
        )
    else:
        hm = HaplotypeMatrix.from_zarr(
            path, region=args.region, accessible_bed=args.accessible_bed,
        )
    if args.pop_file:
        hm.load_pop_file(args.pop_file)
    print(f"Loaded {hm.num_haplotypes} haplotypes × {hm.num_variants:,} variants")
    return hm


def population_names(hm):
    """Return list of population names (or [None] when no pops assigned)."""
    if hm.sample_sets is None:
        return [None]
    return sorted(hm.sample_sets.keys())


def compute_windowed(hm, args):
    """Compute windowed statistics and return dict of DataFrames."""
    pops = population_names(hm)
    two_pop = len(pops) >= 2

    # single-population stats (use first pop, or all samples if no pops)
    pop1 = pops[0]
    pop_arg = [pop1] if pop1 else None

    results = {}

    print("Computing single-population windowed statistics...")
    results["diversity"] = windowed_analysis(
        hm, window_size=args.window_size, step_size=args.step_size,
        statistics=["pi", "theta_w", "tajimas_d", "segregating_sites",
                     "fay_wu_h", "normalized_fay_wu_h"],
        populations=pop_arg,
    )

    print("Computing Garud's H statistics...")
    results["garud"] = windowed_analysis(
        hm, window_size=args.window_size, step_size=args.step_size,
        statistics=["garud_h1", "garud_h12", "garud_h123", "garud_h2h1"],
        populations=pop_arg,
    )

    if two_pop:
        print("Computing two-population divergence statistics...")
        results["divergence"] = windowed_analysis(
            hm, window_size=args.window_size, step_size=args.step_size,
            statistics=["fst", "dxy", "da"],
            populations=pops[:2],
        )

    return results


def compute_scalar(hm):
    """Compute scalar summary statistics."""
    pops = population_names(hm)
    two_pop = len(pops) >= 2

    scalars = {}
    for pop in pops:
        label = pop or "all"
        scalars[f"pi_{label}"] = diversity.pi(hm, population=pop)
        scalars[f"tajimas_d_{label}"] = diversity.tajimas_d(hm, population=pop)

    if two_pop:
        scalars["fst"] = divergence.fst_hudson(hm, pops[0], pops[1])
        scalars["dxy"] = divergence.dxy(hm, pops[0], pops[1])

    return scalars


def compute_sfs(hm):
    """Compute SFS (and joint SFS when two pops are present)."""
    pops = population_names(hm)
    two_pop = len(pops) >= 2

    sfs_results = {}
    sfs_results["sfs"] = sfs.sfs(hm, population=pops[0])
    if two_pop:
        sfs_results["joint_sfs"] = sfs.joint_sfs(hm, pop1=pops[0], pop2=pops[1])
    return sfs_results


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_genome_scan(windowed, sfs_results, hm, args):
    """Multi-panel genome scan figure."""
    pops = population_names(hm)
    two_pop = len(pops) >= 2

    df_div = windowed["diversity"]
    df_gar = windowed["garud"]
    pos_mb = df_div["start"].values / 1e6

    # determine panels
    panels = [
        (df_div["pi"].values,                   "Nucleotide diversity",     r"$\pi$",       "#2980b9"),
        (df_div["theta_w"].values,              "Watterson's theta",        r"$\theta_W$",  "#27ae60"),
        (df_div["tajimas_d"].values,            "Tajima's D",               "D",            "#8e44ad"),
        (df_div["normalized_fay_wu_h"].values,  "Fay & Wu's H*",           "H*",           "#d35400"),
        (df_gar["garud_h12"].values,            "Garud's H12",             "H12",          "#e67e22"),
        (df_div["segregating_sites"].values,    "Segregating sites",        "S",            "#7f8c8d"),
    ]
    if two_pop:
        df_dv2 = windowed["divergence"]
        panels.insert(4, (df_dv2["fst"].values, r"Hudson $F_{ST}$", r"$F_{ST}$", "#c0392b"))
        panels.insert(5, (df_dv2["dxy"].values, r"$D_{xy}$",        r"$D_{xy}$", "#16a085"))

    n_panels = len(panels)
    has_jsfs = "joint_sfs" in sfs_results
    n_right = 2 if has_jsfs else 1

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    fig = plt.figure(figsize=(14, 2.0 * n_panels))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_panels, 2, figure=fig, hspace=0.45, wspace=0.3,
                  width_ratios=[3, 1])

    region_label = args.region or "genomic"
    zero_line_stats = {"D", "H*"}

    for i, (y, title, ylabel, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(pos_mb, y, color=color, alpha=0.6, linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7)
        if ylabel in zero_line_stats:
            ax.axhline(0, color="0.4", linewidth=0.5, linestyle="--")
        if i < n_panels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"{region_label} position (Mb)", fontsize=8)

    # right column: SFS
    sfs_arr = sfs_results["sfs"]
    sfs_arr = sfs_arr[1:-1]  # exclude fixed classes
    rows_sfs = n_panels // 2

    ax_sfs = fig.add_subplot(gs[:rows_sfs, 1])
    ax_sfs.bar(range(1, len(sfs_arr) + 1), sfs_arr, color="#2980b9",
               edgecolor="0.3", linewidth=0.3, width=1.0)
    ax_sfs.set_xlabel("Derived allele count", fontsize=8)
    ax_sfs.set_ylabel("Count", fontsize=8)
    ax_sfs.set_title(f"SFS ({pops[0] or 'all'})", fontsize=9, fontweight="bold")
    ax_sfs.set_xlim(0, min(50, len(sfs_arr)))
    ax_sfs.tick_params(labelsize=7)

    if has_jsfs:
        ax_jsfs = fig.add_subplot(gs[rows_sfs:, 1])
        jsfs_plot = np.log10(sfs_results["joint_sfs"] + 1)
        n1 = min(50, jsfs_plot.shape[0])
        n2 = min(50, jsfs_plot.shape[1])
        im = ax_jsfs.imshow(jsfs_plot[:n1, :n2].T, origin="lower", aspect="auto",
                            cmap="viridis")
        ax_jsfs.set_xlabel(f"{pops[0]} DAC", fontsize=8)
        ax_jsfs.set_ylabel(f"{pops[1]} DAC", fontsize=8)
        ax_jsfs.set_title("Joint SFS (log10)", fontsize=9, fontweight="bold")
        ax_jsfs.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax_jsfs, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"pg_gpu genome scan: {region_label}\n"
        f"{hm.num_haplotypes} haplotypes, {hm.num_variants:,} variants, "
        f"{args.window_size // 1000}kb windows",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Figure saved to {args.output}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    hm = load_data(args)

    windowed = compute_windowed(hm, args)
    scalars = compute_scalar(hm)
    sfs_results = compute_sfs(hm)

    print("\nScalar statistics:")
    for k, v in scalars.items():
        print(f"  {k}: {v:.6f}")

    plot_genome_scan(windowed, sfs_results, hm, args)


if __name__ == "__main__":
    main()
