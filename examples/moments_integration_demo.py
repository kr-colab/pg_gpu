#!/usr/bin/env python
"""
Demographic inference from LD statistics using pg_gpu + moments.

Simulates 50 replicate 1Mb regions under an isolation-with-migration model
using msprime, computes two-population LD statistics with pg_gpu,
then fits the demographic model using moments' inference engine.
Compares timing of the LD parsing step: pg_gpu vs moments.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import demes
import moments
import moments.LD

from pg_gpu.moments_ld import compute_ld_statistics


# ── Simulation parameters ─────────────────────────────────────────────────

NUM_REPS = 20
COMPARE_MOMENTS = True  # Set True to also time moments (slow: ~12s per rep)
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


def parse_ld_replicates(parser_func, label):
    """Parse LD statistics from all replicates using a given parser."""
    pop_file = os.path.join(DATA_DIR, "samples.txt")
    rec_map = os.path.join(DATA_DIR, "flat_map.txt")
    pops = ["deme0", "deme1"]

    ld_stats = {}
    t0 = time.time()
    for ii in range(NUM_REPS):
        vcf_path = os.path.join(DATA_DIR, f"rep_{ii}.vcf")
        ld_stats[ii] = parser_func(
            vcf_path, rec_map_file=rec_map, pop_file=pop_file,
            pops=pops, r_bins=R_BINS, report=False,
        )
    elapsed = time.time() - t0
    print(f"  {label}: {NUM_REPS} replicates in {elapsed:.1f}s "
          f"({elapsed/NUM_REPS:.2f}s per rep)")
    return ld_stats, elapsed


def _expected_ld(demo_func, params, r_bins, Ne):
    """Compute expected LD curves from a demographic model."""
    rho = 4 * Ne * r_bins
    y = moments.LD.Inference.bin_stats(demo_func, params[:-1], rho=rho)
    return moments.LD.Inference.sigmaD2(y, normalization=0)


def plot_ld_comparison(mv, opt_params, demo_func, r_bins, true_model,
                       opt_params_moments=None,
                       output="ld_comparison.pdf"):
    """Plot empirical LD statistics against theoretical expectations.

    Shows the inferred model fit (solid lines) and the true model (dashed)
    overlaid on empirical means with bootstrap standard errors. When
    opt_params_moments is provided, also plots the moments-native inference.
    """
    Ne_inferred = opt_params[-1]
    Ne_true = true_model.demes[0].epochs[0].start_size

    # True model parameters: [nu1, nu2, T, m] in coalescent units
    true_nu1 = true_model.demes[1].epochs[0].start_size / Ne_true
    true_nu2 = true_model.demes[2].epochs[0].start_size / Ne_true
    true_T = true_model.demes[1].epochs[0].start_time / (2 * Ne_true)
    true_m = true_model.migrations[0].rate * 2 * Ne_true
    true_params = [true_nu1, true_nu2, true_T, true_m, Ne_true]

    y_inferred = _expected_ld(demo_func, opt_params, r_bins, Ne_inferred)
    y_true = _expected_ld(demo_func, true_params, r_bins, Ne_true)
    y_moments = None
    if opt_params_moments is not None:
        Ne_moments = opt_params_moments[-1]
        y_moments = _expected_ld(demo_func, opt_params_moments, r_bins, Ne_moments)

    ld_names, het_names = y_inferred.names()
    r_mids = (r_bins[:-1] + r_bins[1:]) / 2

    # Statistics to plot: 3x3 grid of DD, Dz, pi2
    panels = [
        ("DD_0_0", r"$D_0^2$"),
        ("DD_0_1", r"$D_0 D_1$"),
        ("DD_1_1", r"$D_1^2$"),
        ("Dz_0_0_0", r"$Dz_{0,0,0}$"),
        ("Dz_0_1_1", r"$Dz_{0,1,1}$"),
        ("Dz_1_1_1", r"$Dz_{1,1,1}$"),
        ("pi2_0_0_1_1", r"$\pi_{2;0,0,1,1}$"),
        ("pi2_0_1_0_1", r"$\pi_{2;0,1,0,1}$"),
        ("pi2_1_1_1_1", r"$\pi_{2;1,1,1,1}$"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    empirical_means = mv["means"]
    empirical_vcs = mv["varcovs"]

    for ax, (stat_name, label) in zip(axes, panels):
        idx = ld_names.index(stat_name)

        # Empirical data with standard errors
        emp_vals = np.array([empirical_means[b][idx] for b in range(len(r_mids))])
        emp_se = np.array([np.sqrt(empirical_vcs[b][idx, idx])
                           for b in range(len(r_mids))])

        # Theoretical curves
        inf_vals = np.array([y_inferred[b][idx] for b in range(len(r_mids))])
        true_vals = np.array([y_true[b][idx] for b in range(len(r_mids))])

        ax.errorbar(r_mids, emp_vals, yerr=emp_se, fmt='o', ms=4,
                    color='k', capsize=2, label='empirical', zorder=3)
        ax.plot(r_mids, inf_vals, '-', color='C0', lw=2,
                label='inferred (pg_gpu)')
        if y_moments is not None:
            mom_vals = np.array([y_moments[b][idx] for b in range(len(r_mids))])
            ax.plot(r_mids, mom_vals, '-', color='C2', lw=2, alpha=0.7,
                    label='inferred (moments)')
        ax.plot(r_mids, true_vals, '--', color='C1', lw=1.5, label='true')

        ax.set_xscale('log')
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    axes[0].legend(fontsize=7, loc='best')
    for ax in axes[6:]:
        ax.set_xlabel('r (recombination rate)', fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n  Saved LD comparison plot to {output}")
    plt.close(fig)


def main():
    print("=" * 66)
    print("Demographic Inference from LD: pg_gpu + moments")
    print("=" * 66)

    # Step 1: Simulate data
    cache_file = os.path.join(DATA_DIR, "ld_stats_cache.pkl")
    if not os.path.exists(cache_file):
        simulate_data()

        # Step 2: Parse LD statistics with pg_gpu
        print(f"\nParsing LD statistics ({len(R_BINS)-1} recombination bins) ...")
        ld_stats_gpu, t_gpu = parse_ld_replicates(compute_ld_statistics, "pg_gpu")

        ld_stats_moments = None
        t_moments = None
        if COMPARE_MOMENTS:
            def _moments_parse_haploid(*args, **kwargs):
                return moments.LD.Parsing.compute_ld_statistics(
                    *args, use_genotypes=False, **kwargs)
            ld_stats_moments, t_moments = parse_ld_replicates(
                _moments_parse_haploid, "moments")
            print(f"  Speedup: {t_moments/t_gpu:.1f}x")

        with open(cache_file, "wb") as f:
            pickle.dump({'gpu': ld_stats_gpu, 't_gpu': t_gpu,
                         'moments': ld_stats_moments,
                         't_moments': t_moments}, f)
    else:
        print(f"\nLoading cached LD statistics from {cache_file}")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        ld_stats_gpu = cache['gpu']
        t_gpu = cache['t_gpu']
        ld_stats_moments = cache.get('moments')
        t_moments = cache.get('t_moments')

    # Step 3: Bootstrap means and variance-covariance
    print(f"\nBootstrapping {NUM_REPS} replicates ...")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats_gpu)
    all_boot = moments.LD.Parsing.get_bootstrap_sets(
        ld_stats_gpu, remove_norm_stats=False)

    # Step 4: Run inference on pg_gpu-parsed data
    print("\nRunning demographic inference (pg_gpu data) ...")
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

    # Run inference on moments-parsed data if available
    opt_params_moments = None
    if ld_stats_moments is not None:
        print("\nRunning demographic inference (moments data) ...")
        mv_mom = moments.LD.Parsing.bootstrap_data(ld_stats_moments)
        p_guess_mom = moments.LD.Util.perturb_params(
            [0.1, 2, 0.075, 2, 10000], fold=0.1)
        t0 = time.time()
        opt_params_moments, LL_mom = moments.LD.Inference.optimize_log_lbfgsb(
            p_guess_mom, [mv_mom["means"], mv_mom["varcovs"]],
            [demo_func], rs=R_BINS,
        )
        t_inf_mom = time.time() - t0
        print(f"  Inference completed in {t_inf_mom:.1f}s (LL = {LL_mom:.2f})")

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

    physical_moments = None
    if opt_params_moments is not None:
        physical_moments = moments.LD.Util.rescale_params(
            opt_params_moments, ["nu", "nu", "T", "m", "Ne"])

    print("\n" + "=" * 80)
    if physical_moments is not None:
        print(f"{'Parameter':<20s} {'True':>10s} {'pg_gpu':>10s} {'moments':>10s} {'95% CI':>22s}")
    elif has_ci:
        print(f"{'Parameter':<20s} {'True':>10s} {'Inferred':>10s} {'95% CI':>22s}")
    else:
        print(f"{'Parameter':<20s} {'True':>10s} {'Inferred':>10s}")
    print("-" * 80)
    for i, (name, true, est, lo, hi) in enumerate(
            zip(names, true_vals, physical, lower, upper)):
        mom_str = ""
        if physical_moments is not None:
            m = physical_moments[i]
            mom_str = f" {m:>10.6f}" if name == "Migration rate" else f" {m:>10.0f}"
        ci = ""
        if has_ci:
            ci = (f" [{lo:.6f}, {hi:.6f}]" if name == "Migration rate"
                  else f" [{lo:.0f}, {hi:.0f}]")
        if name == "Migration rate":
            print(f"{name:<20s} {true:>10.6f} {est:>10.6f}{mom_str}{ci}")
        else:
            print(f"{name:<20s} {true:>10.0f} {est:>10.0f}{mom_str}{ci}")
    print("=" * 80)

    # Step 7: Plot empirical vs theoretical LD curves
    print("\nPlotting LD comparison ...")
    g = demographic_model()
    plot_ld_comparison(mv, opt_params, demo_func, R_BINS, g,
                       opt_params_moments=opt_params_moments,
                       output=os.path.join(DATA_DIR, "ld_comparison.pdf"))

    print("\nTiming summary:")
    print(f"  LD parsing (pg_gpu):  {t_gpu:>8.1f}s  ({NUM_REPS} replicates)")
    if t_moments is not None:
        print(f"  LD parsing (moments): {t_moments:>8.1f}s  ({t_moments/t_gpu:.0f}x slower)")
    print(f"  Inference:            {t_inference:>8.1f}s")
    print(f"  Total (pg_gpu):       {t_gpu + t_inference:>8.1f}s")
    if t_moments is not None:
        print(f"  Total (moments):      {t_moments + t_inference:>8.1f}s")


if __name__ == "__main__":
    main()
