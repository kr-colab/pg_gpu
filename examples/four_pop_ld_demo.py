#!/usr/bin/env python
"""
Four-population LD parsing validation: pg_gpu vs moments.

Simulates replicate 1Mb regions under a balanced-tree isolation model
((pop0,pop1),(pop2,pop3)), computes 4-population LD statistics with
both pg_gpu and moments, and produces a two-panel figure:
  - Left: scatter plot of pg_gpu vs moments values (colored by stat type)
  - Right: bar plot of total wall time

The demographic model:
  - Ancestral pop (N=10,000) splits at T_deep=2000 gen into two clades
  - Each clade splits at T_shallow=500 gen (pure isolation, no migration)
  - Pop sizes: pop0=5k, pop1=15k, pop2=3k, pop3=20k

Usage:
    pixi run -e moments python examples/four_pop_ld_demo.py

Requires the 'moments' pixi environment (pixi install -e moments).
"""

import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import demes
import moments
import moments.LD

from pg_gpu.moments_ld import compute_ld_statistics


# -- Parameters ---------------------------------------------------------------

NUM_REPS = 10
SEQ_LEN = 1_000_000
MUT_RATE = 1.5e-8
REC_RATE = 1.5e-8
SAMPLE_SIZE = 10  # diploid individuals per population
DATA_DIR = "examples/data/four_pop_demo"
POPS = ["pop0", "pop1", "pop2", "pop3"]
R_BINS = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
                    1e-4, 2e-4, 5e-4, 1e-3])
SIM_WORKERS = 10


# -- Demographic model -------------------------------------------------------

def demographic_model():
    """Balanced tree: ((pop0,pop1),(pop2,pop3)), pure isolation."""
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=2000)])
    b.add_deme("AB", ancestors=["anc"],
               epochs=[dict(start_size=10000, end_time=500)])
    b.add_deme("CD", ancestors=["anc"],
               epochs=[dict(start_size=10000, end_time=500)])
    b.add_deme("pop0", ancestors=["AB"], epochs=[dict(start_size=5000)])
    b.add_deme("pop1", ancestors=["AB"], epochs=[dict(start_size=15000)])
    b.add_deme("pop2", ancestors=["CD"], epochs=[dict(start_size=3000)])
    b.add_deme("pop3", ancestors=["CD"], epochs=[dict(start_size=20000)])
    return b.resolve()


# -- Simulation ---------------------------------------------------------------

def _simulate_one(args):
    """Worker for parallel simulation."""
    ii, demog = args
    ts = msprime.sim_ancestry(
        {p: SAMPLE_SIZE for p in POPS},
        demography=demog,
        sequence_length=SEQ_LEN,
        recombination_rate=REC_RATE,
        random_seed=123 + ii,
    )
    ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=ii + 100)
    vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
    with open(vcf_path, "w") as f:
        ts.write_vcf(f, allow_position_zero=True)
    return ii


def simulate_data():
    """Simulate replicate regions with msprime."""
    from multiprocessing import Pool
    os.makedirs(DATA_DIR, exist_ok=True)

    g = demographic_model()
    demog = msprime.Demography.from_demes(g)

    print(f"Simulating {NUM_REPS} x {SEQ_LEN/1e6:.0f}Mb regions "
          f"(4 pops, {SIM_WORKERS} workers) ...")
    work = [(ii, demog) for ii in range(NUM_REPS)]
    with Pool(SIM_WORKERS) as pool:
        list(pool.map(_simulate_one, work))

    # Write samples file
    with open(os.path.join(DATA_DIR, "samples.txt"), "w") as f:
        f.write("sample\tpop\n")
        for pop_idx, pop_name in enumerate(POPS):
            for ind in range(SAMPLE_SIZE):
                sample_id = pop_idx * SAMPLE_SIZE + ind
                f.write(f"tsk_{sample_id}\t{pop_name}\n")

    # Write flat recombination map
    with open(os.path.join(DATA_DIR, "flat_map.txt"), "w") as f:
        f.write("pos\tMap(cM)\n")
        f.write("0\t0\n")
        f.write(f"{SEQ_LEN}\t{REC_RATE * SEQ_LEN * 100}\n")


# -- Parsing ------------------------------------------------------------------


def parse_all(n_reps):
    """Parse LD statistics with both pg_gpu and moments (serial).

    Returns per-replicate timings for both parsers.
    """
    pop_file = os.path.join(DATA_DIR, "samples.txt")
    rec_map = os.path.join(DATA_DIR, "flat_map.txt")

    # pg_gpu (serial)
    gpu_stats = {}
    gpu_times = []
    for ii in range(n_reps):
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        t0 = time.time()
        gpu_stats[ii] = compute_ld_statistics(
            vcf_path, rec_map_file=rec_map, pop_file=pop_file,
            pops=POPS, r_bins=R_BINS, report=False,
        )
        gpu_times.append(time.time() - t0)
    t_gpu = sum(gpu_times)
    print(f"  pg_gpu:  {n_reps} reps in {t_gpu:.1f}s "
          f"({np.mean(gpu_times):.2f} +/- {np.std(gpu_times):.2f}s per rep)")

    # moments (serial)
    mom_stats = {}
    mom_times = []
    for ii in range(n_reps):
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        t0 = time.time()
        mom_stats[ii] = moments.LD.Parsing.compute_ld_statistics(
            vcf_path, rec_map_file=rec_map, pop_file=pop_file,
            pops=POPS, r_bins=R_BINS, report=False,
            use_genotypes=False,
        )
        mom_times.append(time.time() - t0)
        print(f"    rep {ii}: {mom_times[-1]:.1f}s", flush=True)
    t_mom = sum(mom_times)
    print(f"  moments: {n_reps} reps in {t_mom:.1f}s "
          f"({np.mean(mom_times):.1f} +/- {np.std(mom_times):.1f}s per rep)")

    return gpu_stats, mom_stats, gpu_times, mom_times


# -- Plotting -----------------------------------------------------------------

def plot_validation(gpu_stats, mom_stats, n_reps, gpu_times, mom_times,
                    output):
    """Two-panel figure: scatter of values + timing bar plot."""
    ld_names = gpu_stats[0]['stats'][0]
    het_names = gpu_stats[0]['stats'][1]
    n_sums = len(gpu_stats[0]['sums'])

    # Collect all values by stat type
    dd_gpu, dd_mom = [], []
    dz_gpu, dz_mom = [], []
    pi2_gpu, pi2_mom = [], []
    h_gpu, h_mom = [], []

    n_ld = len(ld_names)
    for rep in range(n_reps):
        for b in range(n_sums):
            gv = np.array(gpu_stats[rep]['sums'][b])
            mv = np.array(mom_stats[rep]['sums'][b])
            for j in range(len(gv)):
                if j < n_ld:
                    name = ld_names[j]
                    if name.startswith("DD"):
                        dd_gpu.append(gv[j])
                        dd_mom.append(mv[j])
                    elif name.startswith("Dz"):
                        dz_gpu.append(gv[j])
                        dz_mom.append(mv[j])
                    elif name.startswith("pi2"):
                        pi2_gpu.append(gv[j])
                        pi2_mom.append(mv[j])
                else:
                    h_gpu.append(gv[j])
                    h_mom.append(mv[j])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left panel: scatter
    ms = 3
    alpha = 0.5
    ax1.scatter(dd_mom, dd_gpu, s=ms, alpha=alpha, label='DD', color='C0')
    ax1.scatter(dz_mom, dz_gpu, s=ms, alpha=alpha, label='Dz', color='C1')
    ax1.scatter(pi2_mom, pi2_gpu, s=ms, alpha=alpha, label=r'$\pi_2$',
                color='C2')
    ax1.scatter(h_mom, h_gpu, s=ms, alpha=alpha, label='H', color='C3')

    # Identity line
    all_vals = dd_mom + dz_mom + pi2_mom + h_mom
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.05
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
             'k--', lw=0.8, alpha=0.5)
    ax1.set_xlabel('moments', fontsize=11)
    ax1.set_ylabel('pg_gpu', fontsize=11)
    ax1.set_title('LD statistic values (raw sums)', fontsize=11)
    ax1.legend(fontsize=9, markerscale=3)

    # Compute max relative error for annotation
    all_gpu = np.array(dd_gpu + dz_gpu + pi2_gpu + h_gpu)
    all_mom = np.array(dd_mom + dz_mom + pi2_mom + h_mom)
    nz = np.abs(all_mom) > 1e-15
    max_rel = np.max(np.abs(all_gpu[nz] - all_mom[nz]) / np.abs(all_mom[nz]))
    ax1.text(0.05, 0.92, f'max relative error: {max_rel:.1e}',
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right panel: per-replicate timing with error bars
    gpu_arr = np.array(gpu_times)
    mom_arr = np.array(mom_times)
    means = [np.mean(gpu_arr), np.mean(mom_arr)]
    sems = [np.std(gpu_arr) / np.sqrt(len(gpu_arr)),
            np.std(mom_arr) / np.sqrt(len(mom_arr))]

    bars = ax2.bar(['pg_gpu', 'moments'], means, yerr=sems,
                   color=['C0', 'C1'], width=0.5, capsize=5)
    ax2.set_ylabel('Time per replicate (s)', fontsize=11)
    ax2.set_title(f'Parsing time ({n_reps} replicates)', fontsize=11)
    for bar, m, s in zip(bars, means, sems):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + s + 0.5,
                 f'{m:.1f}s', ha='center', va='bottom', fontsize=10)

    # Per-replicate speedup with error bars
    per_rep_speedup = mom_arr / gpu_arr
    mean_speedup = np.mean(per_rep_speedup)
    std_speedup = np.std(per_rep_speedup)
    ax2.text(0.5, 0.85,
             f'{mean_speedup:.0f}x $\\pm$ {std_speedup:.0f}x speedup',
             transform=ax2.transAxes, ha='center', fontsize=11,
             fontweight='bold')

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output}")
    plt.close(fig)


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 66)
    print("Four-Population LD Parsing Validation: pg_gpu vs moments")
    print("=" * 66)

    cache_file = os.path.join(DATA_DIR, "ld_stats_cache.pkl")

    if not os.path.exists(cache_file):
        simulate_data()

        print(f"\nParsing LD statistics ({len(R_BINS)-1} bins, "
              f"{len(POPS)} pops) ...")
        gpu_stats, mom_stats, gpu_times, mom_times = parse_all(NUM_REPS)

        with open(cache_file, "wb") as f:
            pickle.dump({
                'gpu': gpu_stats, 'moments': mom_stats,
                'gpu_times': gpu_times, 'mom_times': mom_times,
            }, f)
    else:
        print(f"\nLoading cached LD statistics from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        gpu_stats = cache['gpu']
        mom_stats = cache['moments']
        gpu_times = cache['gpu_times']
        mom_times = cache['mom_times']

    # Print comparison summary
    ld_names = gpu_stats[0]['stats'][0]
    n_sums = len(gpu_stats[0]['sums'])
    max_rel_err = 0.0
    for rep in range(NUM_REPS):
        for b in range(n_sums):
            gv = np.array(gpu_stats[rep]['sums'][b])
            mv = np.array(mom_stats[rep]['sums'][b])
            nz = np.abs(mv) > 1e-15
            if nz.any():
                rel = np.max(np.abs(gv[nz] - mv[nz]) / np.abs(mv[nz]))
                max_rel_err = max(max_rel_err, rel)

    gpu_arr = np.array(gpu_times)
    mom_arr = np.array(mom_times)
    per_rep_speedup = mom_arr / gpu_arr

    n_bins = len(gpu_stats[0]['bins'])
    print(f"\n  {NUM_REPS} replicates, {n_bins} bins, "
          f"{len(ld_names)} LD stats")
    print(f"  Max relative error: {max_rel_err:.2e}")
    if max_rel_err < 1e-6:
        print("  PASS: pg_gpu and moments produce identical LD statistics")

    print(f"\n  pg_gpu:  {np.mean(gpu_arr):.2f} +/- "
          f"{np.std(gpu_arr):.2f}s per rep")
    print(f"  moments: {np.mean(mom_arr):.1f} +/- "
          f"{np.std(mom_arr):.1f}s per rep")
    print(f"  Speedup: {np.mean(per_rep_speedup):.0f}x +/- "
          f"{np.std(per_rep_speedup):.0f}x")

    # Plot
    print("\nPlotting ...")
    outfile = os.path.join(DATA_DIR, "four_pop_ld_validation.pdf")
    plot_validation(gpu_stats, mom_stats, NUM_REPS, gpu_times, mom_times,
                    outfile)


if __name__ == "__main__":
    main()
