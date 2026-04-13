#!/usr/bin/env python
"""
Admixture detection with Patterson's D and block-jackknife CIs.

Simulates a 4-population msprime tree sequence under two demographic
scenarios (null and with a C -> B admixture pulse), computes Patterson's
D(A, B; C, D) on the GPU via pg_gpu, and reports a block-jackknife 95%
confidence interval for each scenario.

Topology: (((A, B), C), D_outgroup)
    A-B split at 2,000 generations ago
    (A,B) splits from C at 10,000 generations ago
    (A,B,C) splits from outgroup D at 50,000 generations ago
    Admixed scenario: 10% pulse from C into B at 500 generations ago

pg_gpu defines D via num = (f_A - f_B)(f_C - f_D), so a C -> B pulse
drives D *negative* under this convention. Expected: null D consistent
with 0 (|Z| < ~2), admixed D significantly non-zero (|Z| > ~5), so the
CI excludes 0 under admixture.

Usage
-----
    pixi run python examples/admixture_detection.py
    pixi run python examples/admixture_detection.py --seed 123 --length 20_000_000
    pixi run python examples/admixture_detection.py --no-plot
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import seaborn as sns

from pg_gpu import HaplotypeMatrix, admixture


# ── simulation ──────────────────────────────────────────────────────────────
# We build a standard ABBA-BABA quartet with D as the outgroup, simulate under
# two scenarios, and then run Patterson's D on each.

POPS = ("A", "B", "C", "D")
NE = 10_000
SPLIT_AB = 2_000
SPLIT_ABC = 10_000
SPLIT_ABCD = 50_000
PULSE_TIME = 500
PULSE_FRAC = 0.10
MUTATION_RATE = 1e-8
RECOMB_RATE = 1e-8


def build_demography(with_pulse: bool) -> msprime.Demography:
    """Four-pop demography with optional C -> B admixture pulse."""
    # Four extant pops (A, B, C, D) plus three ancestors on the
    # (((A, B), C), D) backbone.
    d = msprime.Demography()
    for name in POPS:
        d.add_population(name=name, initial_size=NE)
    d.add_population(name="AB", initial_size=NE)
    d.add_population(name="ABC", initial_size=NE)
    d.add_population(name="ABCD", initial_size=NE)

    # Going backwards in time: A and B merge first, then (A,B) merges with C,
    # then finally with the outgroup D.
    d.add_population_split(time=SPLIT_AB, derived=["A", "B"], ancestral="AB")
    d.add_population_split(time=SPLIT_ABC, derived=["AB", "C"], ancestral="ABC")
    d.add_population_split(time=SPLIT_ABCD, derived=["ABC", "D"], ancestral="ABCD")

    if with_pulse:
        # source/dest are backwards-in-time: lineages in B move to C going
        # back, i.e. forward-in-time gene flow from C into B.
        d.add_mass_migration(time=PULSE_TIME, source="B", dest="C",
                             proportion=PULSE_FRAC)

    d.sort_events()
    return d


def simulate(demography, length, n_diploids, seed):
    """Run ancestry + mutation simulation, return TreeSequence."""
    samples = {p: n_diploids for p in POPS}
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=length,
        recombination_rate=RECOMB_RATE,
        ploidy=2,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=seed)
    return ts


# ── D test ──────────────────────────────────────────────────────────────────

def run_d_test(hm, scenario, label, blen=None):
    """Compute D(A, B; C, D) with block-jackknife SE and 95% CI."""
    # Aim for ~100 blocks: large enough that the jackknife SE is stable,
    # small enough that each block is roughly linkage-independent.
    if blen is None:
        blen = max(500, hm.num_variants // 100)

    # Block-jackknife: splits the genome into `blen`-variant chunks, drops
    # each in turn, and recomputes D. Returns the overall estimate, its SE,
    # the Z-score d/se, per-block D values, and per-iteration estimates.
    d, se, z, vb, _ = admixture.average_patterson_d(
        hm, "A", "B", "C", "D", blen=blen,
    )
    # Normal-theory 95% CI from the jackknife SE.
    lo, hi = d - 1.96 * se, d + 1.96 * se

    print(f"\n=== {label} ===")
    print(f"  n_variants    = {hm.num_variants:,}  (block size = {blen:,} variants)")
    print(f"  D(A, B; C, D) = {d:+.4f}")
    print(f"  SE            = {se:.4f}")
    print(f"  Z             = {z:+.2f}")
    print(f"  95% CI        = [{lo:+.4f}, {hi:+.4f}]")
    verdict = "rejects D=0" if lo > 0 or hi < 0 else "consistent with D=0"
    print(f"  -> {verdict}")

    return {"scenario": scenario, "label": label, "d": d, "se": se,
            "ci": (lo, hi), "per_block": vb[np.isfinite(vb)]}


# ── plotting ────────────────────────────────────────────────────────────────

def plot_result(results, outpath):
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = {"null": "#2980b9", "admixed": "#c0392b"}

    # Left: per-block D distributions
    ax = axes[0]
    for r in results:
        ax.hist(r["per_block"], bins=40, alpha=0.55,
                color=colors[r["scenario"]], label=r["label"],
                edgecolor="0.3", linewidth=0.3)
    ax.axvline(0, color="0.3", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Per-block D(A, B; C, D)")
    ax.set_ylabel("Block count")
    ax.set_title("Block-level D distribution", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, frameon=True)

    # Right: point estimate + 95% CI
    ax = axes[1]
    labels = [r["label"] for r in results]
    ds = [r["d"] for r in results]
    errs = [[r["d"] - r["ci"][0] for r in results],
            [r["ci"][1] - r["d"] for r in results]]
    bar_colors = [colors[r["scenario"]] for r in results]
    ax.errorbar(range(len(results)), ds, yerr=errs,
                fmt="o", capsize=6, capthick=1.2, linewidth=1.5,
                ecolor="0.3", markersize=8,
                mfc="white", mec="0.3")
    for i, (d, color) in enumerate(zip(ds, bar_colors)):
        ax.scatter([i], [d], s=80, color=color, zorder=5, edgecolor="0.3")
    ax.axhline(0, color="0.3", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("D(A, B; C, D)")
    ax.set_title("Point estimate with 95% block-jackknife CI",
                 fontweight="bold", fontsize=10)

    fig.suptitle("Patterson's D: detecting a C -> B admixture pulse",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {outpath}")


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Admixture detection demo with Patterson's D")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (null uses seed, admixed uses seed+1)")
    p.add_argument("--length", type=int, default=10_000_000,
                   help="sequence length in bp (default: 10 Mb)")
    p.add_argument("--samples", type=int, default=10,
                   help="diploid samples per population (default: 10)")
    p.add_argument("--blen", type=int, default=None,
                   help="jackknife block size in variants (default: n_variants // 100)")
    p.add_argument("-o", "--output", default="admixture_detection.pdf",
                   help="output figure path (default: admixture_detection.pdf)")
    p.add_argument("--no-plot", action="store_true",
                   help="print results without writing a figure")
    return p.parse_args()


def main():
    args = parse_args()

    # Two back-to-back sims with the same demography except for the
    # admixture pulse. Different seeds so the two trees aren't identical.
    scenarios = [
        ("null", "null (no admixture)", args.seed),
        ("admixed",
         f"admixed ({int(PULSE_FRAC*100)}% C->B at t={PULSE_TIME})",
         args.seed + 1),
    ]

    results = []
    for scenario, label, seed in scenarios:
        with_pulse = scenario == "admixed"
        print(f"\nSimulating {label!r} (seed={seed})...")
        ts = simulate(build_demography(with_pulse), args.length, args.samples, seed)
        # from_ts auto-populates hm.sample_sets from the ts population metadata,
        # so the A/B/C/D labels flow through without an extra loop here.
        hm = HaplotypeMatrix.from_ts(ts)
        print(f"  {ts.num_sites:,} variants, {hm.num_haplotypes} haplotypes across "
              f"{len(hm.sample_sets)} populations")
        results.append(run_d_test(hm, scenario, label, blen=args.blen))

    if not args.no_plot:
        plot_result(results, args.output)


if __name__ == "__main__":
    main()
