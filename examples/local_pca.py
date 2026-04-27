#!/usr/bin/env python
"""
Local PCA (lostruct) along a chromosome shaped by a partial selective sweep.

Pipeline:
    1. Simulate a 10 Mb chromosome with msprime using
       `SweepGenicSelection` at the midpoint, targeting a final sweep
       allele frequency of 0.5 (i.e. a partial / incomplete sweep).
    2. `lostruct(...)` runs the full four-step Li & Ralph pipeline in
       one call: per-window local PCA, Frobenius distance between
       windows' covariance reps, MDS, and corner detection in MDS
       space.
    3. 1D k-means (k=3) on MDS1 values to partition windows into
       neutral / linked / sweep regimes based on how far each window's
       MDS1 sits from the chromosome-wide baseline.
    4. For comparison, scan Garud H12 along the chromosome (a canonical
       haplotype-frequency sweep statistic).

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
from scipy.cluster.vq import kmeans2

from pg_gpu import HaplotypeMatrix, lostruct, windowed_analysis


REGIME_NAMES = ("neutral", "linked", "sweep")
REGIME_COLORS = {"neutral": "#4C9AFF", "linked": "#F2B84B", "sweep": "#D94E4E"}


def _cluster_mds1(mds1: np.ndarray, seed: int):
    """1D k-means (k=3) on MDS1, relabeled as neutral / linked / sweep.

    Clusters are ordered by distance of their centroid from the
    chromosome-wide median: closest → neutral, middle → linked,
    farthest → sweep. This works regardless of the sign of the sweep
    signal in MDS space.
    """
    centroids, labels = kmeans2(mds1.astype(np.float64), k=3,
                                minit='++', seed=seed)
    dist = np.abs(centroids - np.median(mds1))
    rank = np.argsort(dist)
    remap = np.empty(3, dtype=np.int64)
    for rank_idx, cluster_idx in enumerate(rank):
        remap[cluster_idx] = rank_idx
    regime = np.array([REGIME_NAMES[remap[l]] for l in labels])
    return regime, centroids[rank]


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
    # Li & Ralph's lostruct pipeline: local PCA, window distance, MDS, corners.
    # this is the core of the example, and the only call to lostruct. All other
    # analyses are downstream of this.

    print(f"Running lostruct pipeline (window={args.window} SNPs, "
          f"k={args.k}, n_corners={args.n_corners}) ...")
    res = lostruct(hm, window_size=args.window,
                   step_size=args.window // 2,
                   window_type='snp', k=args.k,
                   corner_prop=args.prop, n_corners=args.n_corners,
                   random_state=args.seed)
    mds = res.mds
    corner_idx = res.corner_indices
    print(f"  n_windows={res.n_windows}  "
          f"variance explained MDS1/MDS2: "
          f"{res.explained_variance_ratio[0]:.3f} / "
          f"{res.explained_variance_ratio[1]:.3f}")

    print("Clustering MDS1 into neutral / linked / sweep regimes (k=3) ...")
    regime, regime_centroids = _cluster_mds1(mds[:, 0], seed=args.seed)
    for name, centroid in zip(REGIME_NAMES, regime_centroids):
        print(f"  {name:8s}  MDS1 centroid={centroid:+.3f}  "
              f"n_windows={(regime == name).sum()}")

    centers = res.windows['center'].to_numpy()

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
    scalar_centers = scalar_df['center'].to_numpy()
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

    # Left panel: MDS scatter, colored by k-means regime
    for name in REGIME_NAMES:
        mask = regime == name
        ax_mds.scatter(mds[mask, 0], mds[mask, 1],
                       c=REGIME_COLORS[name], s=35,
                       edgecolors='white', linewidths=0.3,
                       label=f"{name} (n={mask.sum()})")
    corner_edge_colors = plt.get_cmap("tab10")(range(args.n_corners))
    for i in range(args.n_corners):
        ax_mds.scatter(mds[corner_idx[:, i], 0], mds[corner_idx[:, i], 1],
                       facecolors='none',
                       edgecolors=[corner_edge_colors[i]], s=140,
                       linewidths=1.4, label=f'corner {i+1}')
    ax_mds.set_xlabel("MDS 1")
    ax_mds.set_ylabel("MDS 2")
    ax_mds.set_title("Local-PCA MDS, colored by k-means regime")
    ax_mds.legend(loc='best', fontsize=8)

    # Top-right: MDS1 along the chromosome, colored by regime
    for name in REGIME_NAMES:
        mask = regime == name
        ax_mds1.scatter(centers[mask] / 1e6, mds[mask, 0],
                        c=REGIME_COLORS[name], s=25,
                        edgecolors='white', linewidths=0.2,
                        label=name)
    for i in range(args.n_corners):
        ax_mds1.scatter(centers[corner_idx[:, i]] / 1e6,
                        mds[corner_idx[:, i], 0],
                        facecolors='none',
                        edgecolors=[corner_edge_colors[i]], s=70,
                        linewidths=1.0)
    ax_mds1.axvline(SWEEP_POS / 1e6, color='orange', linestyle='--',
                    alpha=0.7, label='sweep focal site')
    ax_mds1.set_ylabel("MDS 1")
    ax_mds1.set_title(f"MDS1 (k-means regime) and Garud H12 along the "
                      f"chromosome (end_freq={args.end_freq})")
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
