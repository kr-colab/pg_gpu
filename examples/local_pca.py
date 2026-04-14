#!/usr/bin/env python
"""
Local PCA (lostruct) along a chromosome shaped by a partial selective sweep.

Pipeline:
    1. Simulate a 10 Mb chromosome with msprime using
       `SweepGenicSelection` at the midpoint, targeting a final sweep
       allele frequency of 0.5 (i.e. a partial / incomplete sweep).
    2. `local_pca` in SNP windows → per-window top-k eigendecomposition of
       the sample-sample covariance matrix.
    3. `pc_dist` → Frobenius distance matrix between windows.
    4. `pcoa` on the distance matrix → 2D MDS embedding.
    5. `corners` → highlight the windows sitting at the extremes of MDS,
       which should coincide with the sweep region (where the partial-sweep
       haplotype split creates local population structure).
    6. For comparison, also scan Garud H12 and H2/H1 along the chromosome
       (canonical haplotype-frequency sweep statistics).

Usage
-----
    pixi run python examples/local_pca.py
    pixi run python examples/local_pca.py --seed 7 --window 150
    pixi run python examples/local_pca.py --no-plot
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np

from pg_gpu import HaplotypeMatrix, windowed_analysis
from pg_gpu.decomposition import corners, pc_dist, pcoa


SEQ_LEN = 10_000_000
SWEEP_POS = 5_000_000
NE = 10_000
MU = 1e-8
RECOMB_RATE = 1e-8


def _simulate(n_diploids: int, s: float, end_freq: float, seed: int):
    """Partial genic-selection sweep at the chromosome midpoint."""
    sweep = msprime.SweepGenicSelection(
        position=SWEEP_POS,
        start_frequency=1.0 / (2 * NE),
        end_frequency=end_freq,
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
    hap = ts.genotype_matrix().T.astype(np.int8)   # (n_haplotypes, n_variants)
    positions = ts.tables.sites.position.astype(np.int64)
    # msprime can leave duplicated positions after dedup; make strictly
    # increasing so HaplotypeMatrix / WindowIterator behave.
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = positions[i - 1] + 1
    return hap, positions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--window", type=int, default=500,
                   help="SNP-count window size")
    p.add_argument("--k", type=int, default=2,
                   help="Number of PCs per window")
    p.add_argument("--n-diploids", type=int, default=50,
                   help="Number of diploid samples")
    p.add_argument("--s", type=float, default=0.1,
                   help="Selection coefficient for the sweep")
    p.add_argument("--end-freq", type=float, default=0.5,
                   help="Final frequency of the sweep allele")
    p.add_argument("--prop", type=float, default=0.05,
                   help="Proportion of windows per corner")
    p.add_argument("--n-corners", type=int, default=3)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(__file__).with_suffix(".png"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    print(f"Simulating partial sweep at {SWEEP_POS:,} bp "
          f"(seed={args.seed}, n_diploids={args.n_diploids}, "
          f"s={args.s}, end_freq={args.end_freq}) ...")
    hap, positions = _simulate(args.n_diploids, args.s, args.end_freq,
                               args.seed)
    n_hap, n_var = hap.shape
    print(f"  n_haplotypes={n_hap}  n_variants={n_var}")

    hm = HaplotypeMatrix(hap, positions, 0, SEQ_LEN)

    print(f"Running local_pca (window={args.window} SNPs, k={args.k}) ...")
    result = windowed_analysis(
        hm, window_size=args.window, step_size=args.window // 2,
        statistics=['local_pca'], window_type='snp', k=args.k)
    print(f"  n_windows={result.n_windows}")

    print("Computing pc_dist (L1) ...")
    d = pc_dist(result, npc=args.k, normalize='L1')
    print(f"  dist matrix shape={d.shape}")

    print("Computing MDS via pcoa ...")
    mds, er = pcoa(d, n_components=2)
    print(f"  variance explained MDS1/MDS2: {er[0]:.3f} / {er[1]:.3f}")

    print(f"Identifying {args.n_corners} corner clusters ...")
    corner_idx = corners(mds, prop=args.prop, k=args.n_corners,
                         random_state=args.seed)
    print(f"  corner_idx shape={corner_idx.shape}")

    centers = result.windows['center'].to_numpy()
    # Define a 200 kb "sweep region" around the focal site for highlighting.
    sweep_half_width = 100_000
    in_sweep = (centers >= SWEEP_POS - sweep_half_width) & \
               (centers <= SWEEP_POS + sweep_half_width)
    print(f"  windows within ±{sweep_half_width:,} bp of sweep: "
          f"{in_sweep.sum()} / {len(centers)}")

    # Scalar summary statistics along the chromosome (bp windows, for the
    # right-hand panels). Garud H12 is a classic sweep detector that peaks
    # where a few haplotypes dominate.
    scalar_window = 100_000
    scalar_step = 50_000
    print(f"Computing windowed Garud H12 "
          f"(bp window={scalar_window:,}, step={scalar_step:,}) ...")
    scalar_df = windowed_analysis(
        hm, window_size=scalar_window, step_size=scalar_step,
        statistics=['garud_h12'], window_type='bp')
    # Column names vary between pipelines; pick whichever start/stop pair exists.
    start_col = 'window_start' if 'window_start' in scalar_df.columns else 'start'
    stop_col = 'window_stop' if 'window_stop' in scalar_df.columns else 'stop'
    scalar_centers = ((scalar_df[start_col] + scalar_df[stop_col]) / 2).to_numpy()
    print(f"  n_scalar_windows={len(scalar_df)}")

    if args.no_plot:
        return

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.2],
        height_ratios=[1.0, 1.0],
        hspace=0.08, wspace=0.22,
    )
    ax_mds = fig.add_subplot(gs[:, 0])
    ax_mds1 = fig.add_subplot(gs[0, 1])
    ax_h12 = fig.add_subplot(gs[1, 1], sharex=ax_mds1)

    # Left panel: MDS scatter
    ax_mds.scatter(mds[:, 0], mds[:, 1], c='lightgray', s=30,
                   edgecolors='white', linewidths=0.3,
                   label='other windows')
    colors = plt.get_cmap("tab10")(range(args.n_corners))
    for i in range(args.n_corners):
        ax_mds.scatter(mds[corner_idx[:, i], 0], mds[corner_idx[:, i], 1],
                       c=[colors[i]], s=60, edgecolors='black',
                       linewidths=0.5, label=f'corner {i+1}')
    ax_mds.scatter(mds[in_sweep, 0], mds[in_sweep, 1],
                   facecolors='none', edgecolors='red', s=120,
                   linewidths=1.5, label='sweep-proximal windows')
    ax_mds.set_xlabel("MDS 1")
    ax_mds.set_ylabel("MDS 2")
    ax_mds.set_title("Local-PCA MDS of window distances")
    ax_mds.legend(loc='best', fontsize=8)

    # Top-right: MDS1 along the chromosome
    ax_mds1.scatter(centers / 1e6, mds[:, 0], c='gray', s=20,
                    label='all windows')
    ax_mds1.scatter(centers[in_sweep] / 1e6, mds[in_sweep, 0],
                    c='red', s=30, label='sweep-proximal windows')
    for i in range(args.n_corners):
        ax_mds1.scatter(centers[corner_idx[:, i]] / 1e6,
                        mds[corner_idx[:, i], 0],
                        c=[colors[i]], s=40, edgecolors='black',
                        linewidths=0.4)
    ax_mds1.axvline(SWEEP_POS / 1e6, color='orange', linestyle='--',
                    alpha=0.7, label='sweep focal site')
    ax_mds1.set_ylabel("MDS 1")
    ax_mds1.set_title(f"MDS1 and Garud H12 along the chromosome "
                      f"(end_freq={args.end_freq})")
    ax_mds1.set_xlim(0, SEQ_LEN / 1e6)
    ax_mds1.legend(loc='best', fontsize=8)
    plt.setp(ax_mds1.get_xticklabels(), visible=False)

    # Bottom-right: Garud H12 along the chromosome (shares x with MDS1)
    ax_h12.plot(scalar_centers / 1e6,
                scalar_df['garud_h12'].to_numpy(),
                color='steelblue', lw=1.5, label='H12')
    ax_h12.axvline(SWEEP_POS / 1e6, color='orange', linestyle='--',
                   alpha=0.7)
    ax_h12.set_xlabel("chromosome position (Mb)")
    ax_h12.set_ylabel("Garud H12")
    ax_h12.set_xlim(0, SEQ_LEN / 1e6)
    ax_h12.legend(loc='best', fontsize=8)

    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
