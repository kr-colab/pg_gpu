#!/usr/bin/env python
"""
Block-bootstrap confidence intervals for Tajima's D under a completed sweep.

Pipeline:
    1. Simulate a 10 Mb chromosome with msprime using
       `SweepGenicSelection` at the midpoint, targeting a final sweep
       allele frequency of 1.0 (i.e. a completed sweep). The sweep
        drives Tajima's D sharply negative in the local region.
    2. `windowed_analysis` with `statistic='tajimas_d'` in bp windows along
       the chromosome produces per-window Tajima's D values.
    3. Partition windows into
         * sweep-local  (|center - focal| <= local_radius)
         * distal       (|center - focal| >  distal_radius)
    4. `block_bootstrap` on each regime separately for the 95% CI of its
       mean Tajima's D. Under a completed sweep the sweep-local CI excludes
       zero while the distal CI brackets zero — the calibrated CIs are what
       make that statement defensible.

Usage
-----
    pixi run python examples/sweep_tajimas_d_bootstrap.py
    pixi run python examples/sweep_tajimas_d_bootstrap.py --seed 7
    pixi run python examples/sweep_tajimas_d_bootstrap.py --no-plot
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np

# The three pg_gpu entry points used in this example:
#   HaplotypeMatrix    — the core phased-data container; everything else
#                        takes one of these as input.
#   windowed_analysis  — GPU-fused per-window statistics, returns a pandas
#                        DataFrame with one row per window.
#   block_bootstrap    — generic block-resampling CIs on any scalar stat
#                        computed over pre-binned per-block values.
from pg_gpu import HaplotypeMatrix, block_bootstrap, windowed_analysis


SEQ_LEN = 10_000_000
SWEEP_POS = 5_000_000
NE = 10_000
MU = 1e-8
RECOMB_RATE = 1e-8


def _simulate(n_diploids: int, s: float, seed: int):
    """Completed genic-selection sweep at the chromosome midpoint.

    msprime's ``SweepGenicSelection`` requires ``end_frequency < 1``; using
    ``1 - 1/(2 Ne)`` corresponds to the final copy reaching fixation.
    """
    sweep = msprime.SweepGenicSelection(
        position=SWEEP_POS,
        start_frequency=1.0 / (2 * NE),
        end_frequency=1.0 - 1.0 / (2 * NE),
        s=s,
        dt=1e-6,
    )
    ts = msprime.sim_ancestry(
        samples=n_diploids,
        sequence_length=SEQ_LEN,
        recombination_rate=RECOMB_RATE,
        population_size=NE,
        model=[sweep, msprime.StandardCoalescent()],
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    hap = ts.genotype_matrix().T.astype(np.int8)
    positions = ts.tables.sites.position.astype(np.int64)
    # Msprime can simulate multiple mutations at the same position, so here we are enforcing unique positions by shifting any non-ascending positions to the right.
    # In real data, this would be a multi-allelic site, and the VCF format allows representing it as a single site with multiple alternate alleles.
    # Though the HaplotypeMatrix can also represent multi-allelic sites, for simplicity we are treating them as separate biallelic sites here.
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = positions[i - 1] + 1
    return hap, positions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--window", type=int, default=50_000,
                   help="Window size (bp)")
    p.add_argument("--step", type=int, default=25_000,
                   help="Window step (bp)")
    p.add_argument("--local-radius", type=int, default=500_000,
                   help="Half-width of sweep-local region around focal site")
    p.add_argument("--distal-radius", type=int, default=2_000_000,
                   help="Minimum distance from focal site for distal set")
    p.add_argument("--n-diploids", type=int, default=50)
    p.add_argument("--s", type=float, default=0.1,
                   help="Selection coefficient for the sweep")
    p.add_argument("--n-replicates", type=int, default=2000)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(__file__).with_suffix(".png"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print(f"Simulating completed sweep at {SWEEP_POS:,} bp "
          f"(seed={args.seed}, n_diploids={args.n_diploids}, "
          f"s={args.s}) ...")
    hap, positions = _simulate(args.n_diploids, args.s, args.seed)
    print(f"  n_haplotypes={hap.shape[0]}  n_variants={hap.shape[1]}")

    # Build a HaplotypeMatrix from raw numpy. The constructor takes the
    # (n_haplotypes, n_variants) matrix, variant positions, and the genomic
    # region bounds [start, stop). For real data, prefer the loader class
    # methods: HaplotypeMatrix.from_vcf / from_zarr / from_ts.
    hm = HaplotypeMatrix(hap, positions, 0, SEQ_LEN)

    # windowed_analysis runs one fused GPU kernel across all windows and
    # returns a pandas DataFrame with one row per window. Columns include
    # 'start', 'stop', 'center', plus one column per requested statistic
    # (here just 'tajimas_d'). window_type='bp' uses base-pair windows;
    # pass 'snp' for fixed-SNP-count windows instead. `step_size` less than
    # `window_size` produces overlapping windows.
    print(f"Windowed Tajima's D (window={args.window:,} bp, "
          f"step={args.step:,} bp) ...")
    df = windowed_analysis(
        hm, window_size=args.window, step_size=args.step,
        statistics=['tajimas_d'], window_type='bp',
    )
    centers = df['center'].to_numpy()
    tajd = df['tajimas_d'].to_numpy()
    # Tajima's D is undefined on windows with <2 segregating sites, which
    # show up as NaN. Drop them before downstream analysis.
    finite = np.isfinite(tajd)
    centers, tajd = centers[finite], tajd[finite]
    print(f"  n_windows={len(tajd)}  (finite)")

    dist = np.abs(centers - SWEEP_POS)
    sweep_mask = dist <= args.local_radius
    distal_mask = dist > args.distal_radius
    d_sweep = tajd[sweep_mask]
    d_distal = tajd[distal_mask]
    print(f"  sweep-local windows: {sweep_mask.sum()}  "
          f"distal windows: {distal_mask.sum()}")

    # block_bootstrap takes pre-binned per-block values (here one Tajima's D
    # value per genomic window) and a callable `statistic`, draws
    # `n_replicates` with-replacement resamples of the block indices, and
    # returns the plug-in point estimate, SE of the replicates, and the raw
    # replicate distribution. Passing `rng=<int>` makes the draw reproducible.
    # Extract percentile CIs from the replicate array directly.
    print(f"Block bootstrap (n_replicates={args.n_replicates}) ...")
    est_s, se_s, reps_s = block_bootstrap(
        d_sweep, statistic=np.mean,
        n_replicates=args.n_replicates, rng=args.seed,
    )
    est_d, se_d, reps_d = block_bootstrap(
        d_distal, statistic=np.mean,
        n_replicates=args.n_replicates, rng=args.seed + 1,
    )
    lo_s, hi_s = np.quantile(reps_s, [0.025, 0.975])
    lo_d, hi_d = np.quantile(reps_d, [0.025, 0.975])

    def fmt(est, se, lo, hi):
        return f"{est:+.3f} ± {se:.3f}  (95% CI [{lo:+.3f}, {hi:+.3f}])"

    print(f"  sweep-local  mean D = {fmt(est_s, se_s, lo_s, hi_s)}")
    print(f"  distal       mean D = {fmt(est_d, se_d, lo_d, hi_d)}")

    if args.no_plot:
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6),
        gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.35},
    )

    ax_top.plot(centers / 1e6, tajd, color='steelblue', lw=1.2,
                label="Tajima's D (per window)")
    ax_top.axhline(0, color='gray', lw=0.8, alpha=0.6)
    ax_top.axvline(SWEEP_POS / 1e6, color='orange', linestyle='--',
                   alpha=0.7, label='sweep focal site')
    ax_top.axvspan((SWEEP_POS - args.local_radius) / 1e6,
                   (SWEEP_POS + args.local_radius) / 1e6,
                   color='#D94E4E', alpha=0.12, label='sweep-local region')
    ax_top.axvspan(0, (SWEEP_POS - args.distal_radius) / 1e6,
                   color='#4C9AFF', alpha=0.10, label='distal region')
    ax_top.axvspan((SWEEP_POS + args.distal_radius) / 1e6, SEQ_LEN / 1e6,
                   color='#4C9AFF', alpha=0.10)
    ax_top.set_xlabel("chromosome position (Mb)")
    ax_top.set_ylabel("Tajima's D")
    ax_top.set_xlim(0, SEQ_LEN / 1e6)
    ax_top.set_title("Windowed Tajima's D along a chromosome with a completed sweep")
    ax_top.legend(loc='best', fontsize=8)

    labels = ['sweep-local', 'distal']
    ests = [est_s, est_d]
    los = [lo_s, lo_d]
    his = [hi_s, hi_d]
    colors = ['#D94E4E', '#4C9AFF']
    y = np.arange(len(labels))
    for i, (e, lo, hi, c) in enumerate(zip(ests, los, his, colors)):
        ax_bot.plot([lo, hi], [y[i], y[i]], color=c, lw=3,
                    solid_capstyle='round')
        ax_bot.plot(e, y[i], 'o', color=c, markersize=7,
                    markeredgecolor='white')
    ax_bot.axvline(0, color='gray', lw=0.8, alpha=0.6)
    ax_bot.set_yticks(y)
    ax_bot.set_yticklabels(labels)
    ax_bot.set_xlabel("mean Tajima's D  (95% block-bootstrap CI)")
    ax_bot.set_title(f"Per-regime mean  (n_replicates={args.n_replicates})")
    ax_bot.invert_yaxis()

    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
