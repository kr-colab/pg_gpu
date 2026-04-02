#!/usr/bin/env python
"""
Demographic inference from LD statistics using pg_gpu + moments.

Simulates 50 replicate 1Mb regions under an isolation-with-migration model
using msprime, computes two-population LD statistics with pg_gpu (GPU-
accelerated), then fits the demographic model using moments' inference
engine. Compares timing of the LD parsing step: pg_gpu vs moments.

The demographic model:
  - Ancestral population of size 10,000
  - Split 1,500 generations ago into:
    - deme0: size 2,000
    - deme1: size 20,000
  - Symmetric migration rate: 1e-4

Usage:
    pixi run -e moments python examples/ld_inference_demo.py

Requires the 'moments' pixi environment (pixi install -e moments).
"""

import os
import time
import pickle
import numpy as np
import msprime
import demes
import moments
import moments.LD

from pg_gpu.moments_ld import compute_ld_statistics


# ── Simulation parameters ─────────────────────────────────────────────────

NUM_REPS = 20
COMPARE_MOMENTS = False  # Set True to also time moments (slow: ~20s per rep)
SEQ_LEN = 1_000_000
MUT_RATE = 1.5e-8
REC_RATE = 1.5e-8
SAMPLE_SIZE = 10  # diploid individuals per population
DATA_DIR = "examples/data/ld_inference_demo"
R_BINS = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])


def demographic_model():
    """Define the true demographic model using demes."""
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=1500)])
    b.add_deme("deme0", ancestors=["anc"], epochs=[dict(start_size=2000)])
    b.add_deme("deme1", ancestors=["anc"], epochs=[dict(start_size=20000)])
    b.add_migration(demes=["deme0", "deme1"], rate=1e-4)
    return b.resolve()


def simulate_data():
    """Simulate replicate regions with msprime and write VCFs."""
    os.makedirs(DATA_DIR, exist_ok=True)

    g = demographic_model()
    demog = msprime.Demography.from_demes(g)

    print(f"Simulating {NUM_REPS} x {SEQ_LEN/1e6:.0f}Mb regions ...")
    tree_sequences = msprime.sim_ancestry(
        {"deme0": SAMPLE_SIZE, "deme1": SAMPLE_SIZE},
        demography=demog,
        sequence_length=SEQ_LEN,
        recombination_rate=REC_RATE,
        num_replicates=NUM_REPS,
        random_seed=42,
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=ii + 1)
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        with open(vcf_path, "w") as f:
            ts.write_vcf(f)

    # Write samples file
    with open(os.path.join(DATA_DIR, "samples.txt"), "w") as f:
        f.write("sample\tpop\n")
        for pop_idx in range(2):
            for ind in range(SAMPLE_SIZE):
                f.write(f"tsk_{pop_idx * SAMPLE_SIZE + ind}\tdeme{pop_idx}\n")

    # Write flat recombination map
    with open(os.path.join(DATA_DIR, "flat_map.txt"), "w") as f:
        f.write("pos\tMap(cM)\n")
        f.write("0\t0\n")
        f.write(f"{SEQ_LEN}\t{REC_RATE * SEQ_LEN * 100}\n")


def parse_ld_gpu():
    """Parse LD statistics from all replicates using pg_gpu."""
    pop_file = os.path.join(DATA_DIR, "samples.txt")
    rec_map = os.path.join(DATA_DIR, "flat_map.txt")

    ld_stats = {}
    t0 = time.time()
    for ii in range(NUM_REPS):
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        ld_stats[ii] = compute_ld_statistics(
            vcf_path,
            rec_map_file=rec_map,
            pop_file=pop_file,
            pops=["deme0", "deme1"],
            r_bins=R_BINS,
            report=False,
        )
    elapsed = time.time() - t0
    print(f"  pg_gpu: {NUM_REPS} replicates in {elapsed:.1f}s "
          f"({elapsed/NUM_REPS:.2f}s per rep)")
    return ld_stats, elapsed


def parse_ld_moments():
    """Parse LD statistics from all replicates using moments (for timing comparison)."""
    pop_file = os.path.join(DATA_DIR, "samples.txt")
    rec_map = os.path.join(DATA_DIR, "flat_map.txt")

    ld_stats = {}
    t0 = time.time()
    for ii in range(NUM_REPS):
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        ld_stats[ii] = moments.LD.Parsing.compute_ld_statistics(
            vcf_path,
            rec_map_file=rec_map,
            pop_file=pop_file,
            pops=["deme0", "deme1"],
            r_bins=R_BINS,
            report=False,
        )
    elapsed = time.time() - t0
    print(f"  moments: {NUM_REPS} replicates in {elapsed:.1f}s "
          f"({elapsed/NUM_REPS:.2f}s per rep)")
    return ld_stats, elapsed


def main():
    print("=" * 65)
    print("Demographic Inference from LD: pg_gpu + moments")
    print("=" * 65)

    # Step 1: Simulate data
    cache_file = os.path.join(DATA_DIR, "ld_stats_cache.pkl")
    if not os.path.exists(cache_file):
        simulate_data()

        # Step 2: Parse LD statistics with pg_gpu
        print(f"\nParsing LD statistics ({len(R_BINS)-1} recombination bins) ...")
        ld_stats_gpu, t_gpu = parse_ld_gpu()

        # Optional moments comparison
        t_moments = None
        if COMPARE_MOMENTS:
            _, t_moments = parse_ld_moments()
            print(f"  Speedup: {t_moments/t_gpu:.1f}x")

        with open(cache_file, "wb") as f:
            pickle.dump({'gpu': ld_stats_gpu, 't_gpu': t_gpu,
                         't_moments': t_moments}, f)
    else:
        print(f"\nLoading cached LD statistics from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        ld_stats_gpu = cache['gpu']
        t_gpu = cache['t_gpu']
        t_moments = cache.get('t_moments')

    # Step 3: Bootstrap means and variance-covariance
    print(f"\nBootstrapping {NUM_REPS} replicates ...")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats_gpu)
    all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats_gpu)

    # Step 4: Run inference
    print("\nRunning demographic inference ...")
    demo_func = moments.LD.Demographics2D.split_mig

    # Initial guess: [nu0, nu1, T, m, Ne]
    p_guess = [0.1, 2, 0.075, 2, 10000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    t0 = time.time()
    opt_params, LL = moments.LD.Inference.optimize_log_lbfgsb(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=R_BINS,
    )
    t_inference = time.time() - t0
    print(f"  Inference completed in {t_inference:.1f}s (LL = {LL:.2f})")

    # Convert to physical units
    physical = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "m", "Ne"])

    # Step 5: Confidence intervals
    print("\nEstimating confidence intervals ...")
    try:
        uncerts = moments.LD.Godambe.GIM_uncert(
            demo_func, all_boot, opt_params,
            mv["means"], mv["varcovs"], r_edges=R_BINS,
        )
        lower = moments.LD.Util.rescale_params(
            opt_params - 1.96 * uncerts, ["nu", "nu", "T", "m", "Ne"])
        upper = moments.LD.Util.rescale_params(
            opt_params + 1.96 * uncerts, ["nu", "nu", "T", "m", "Ne"])
        has_ci = True
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  Could not estimate CIs: {e}")
        lower = [np.nan] * 5
        upper = [np.nan] * 5
        has_ci = False

    # Step 6: Print results
    g = demographic_model()
    true_vals = [
        g.demes[1].epochs[0].start_size,   # N(deme0)
        g.demes[2].epochs[0].start_size,   # N(deme1)
        g.demes[1].epochs[0].start_time,   # Div time
        g.migrations[0].rate,               # Migration
        g.demes[0].epochs[0].start_size,   # N(ancestral)
    ]
    names = ["N(deme0)", "N(deme1)", "Div. time (gen)", "Migration rate", "N(ancestral)"]

    print("\n" + "=" * 65)
    if has_ci:
        print(f"{'Parameter':<20s} {'True':>10s} {'Inferred':>10s} {'95% CI':>22s}")
    else:
        print(f"{'Parameter':<20s} {'True':>10s} {'Inferred':>10s}")
    print("-" * 65)
    for name, true, est, lo, hi in zip(names, true_vals, physical, lower, upper):
        if name == "Migration rate":
            ci = f"[{lo:.6f}, {hi:.6f}]" if has_ci else ""
            print(f"{name:<20s} {true:>10.6f} {est:>10.6f} {ci}")
        else:
            ci = f"[{lo:.0f}, {hi:.0f}]" if has_ci else ""
            print(f"{name:<20s} {true:>10.0f} {est:>10.0f} {ci}")
    print("=" * 65)

    print(f"\nTiming summary:")
    print(f"  LD parsing (pg_gpu):  {t_gpu:>8.1f}s  ({NUM_REPS} replicates)")
    if t_moments is not None:
        print(f"  LD parsing (moments): {t_moments:>8.1f}s  ({t_moments/t_gpu:.0f}x slower)")
    print(f"  Inference:            {t_inference:>8.1f}s")
    print(f"  Total (pg_gpu):       {t_gpu + t_inference:>8.1f}s")
    if t_moments is not None:
        print(f"  Total (moments):      {t_moments + t_inference:>8.1f}s")


if __name__ == "__main__":
    main()
