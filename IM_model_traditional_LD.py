"""
This script uses msprime to simulate under an isolation with migration model,
writing the outputs to VCF. We'll simulate a small dataset: 100 x 1Mb regions,
each with recombination and mutation rates of 1.5e-8. We'll then use moments
to compute LD statistics from each of the 100 replicates to compute statistic
means and variances/covariances. These are then used to refit the simulated
model using moments.LD, and then we use bootstrapped datasets to estimate
confidence intervals.

The demographic model is a population of size 10,000 that splits into a
population of size 2,000 and a population of size 20,000. The split occurs
1,500 generations ago followed by symmetric migration at rate 1e-4.
"""

import os
import time
import csv
import pickle
import numpy as np
import msprime
import moments
import demes

assert msprime.__version__ >= "1"

# --- output dirs
DATA_DIR = "./data"
LD_DIR = "./MomentsLD/LD_stats"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LD_DIR, exist_ok=True)

# clean old intermediates (ignore missing)
os.system("rm -f ./data/*.vcf.gz")
os.system("rm -f ./data/*.h5")


def demographic_model():
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=1500)])
    b.add_deme("deme0", ancestors=["anc"], epochs=[dict(start_size=2000)])
    b.add_deme("deme1", ancestors=["anc"], epochs=[dict(start_size=20000)])
    b.add_migration(demes=["deme0", "deme1"], rate=1e-4)
    g = b.resolve()
    return g


def run_msprime_replicates(num_reps=100, L=5000000, u=1.5e-8, r=1.5e-8, n=10):
    g = demographic_model()
    demog = msprime.Demography.from_demes(g)
    tree_sequences = msprime.sim_ancestry(
        {"deme0": n, "deme1": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        num_replicates=num_reps,
        random_seed=42,
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=u, random_seed=ii + 1)
        vcf_name = f"{DATA_DIR}/split_mig.{ii}.vcf"
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip -f {vcf_name}")  # force overwrite if exists


def write_samples_and_rec_map(L=5000000, r=1.5e-8, n=10):
    # samples file
    with open(f"{DATA_DIR}/samples.txt", "w+") as fout:
        fout.write("sample\tpop\n")
        for jj in range(2):
            for ii in range(n):
                fout.write(f"tsk_{jj * n + ii}\tdeme{jj}\n")
    # recombination map
    with open(f"{DATA_DIR}/flat_map.txt", "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{L}\t{r * L * 100}\n")


def get_LD_stats(rep_ii, r_bins):
    """Compute moments LD stats for one replicate, returning (stats_dict, seconds)."""
    vcf_file = f"{DATA_DIR}/split_mig.{rep_ii}.vcf.gz"
    t0 = time.perf_counter()
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file=f"{DATA_DIR}/flat_map.txt",
        pop_file=f"{DATA_DIR}/samples.txt",
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )
    dt = time.perf_counter() - t0
    print(f"  finished rep {rep_ii} in {dt:.2f} s")
    return ld_stats, dt


if __name__ == "__main__":
    num_reps = 100
    # define the bin edges (genetic distance)
    r_bins = np.concatenate(([0], np.logspace(-6, -3, 16)))

    try:
        print("loading data if pre-computed")
        with open(f"{DATA_DIR}/means.varcovs.split_mig.{num_reps}_reps.bp", "rb") as fin:
            mv = pickle.load(fin)
        with open(f"{DATA_DIR}/bootstrap_sets.split_mig.{num_reps}_reps.bp", "rb") as fin:
            all_boot = pickle.load(fin)
    except IOError:
        print("running msprime and writing vcfs")
        run_msprime_replicates(num_reps=num_reps)

        print("writing samples and recombination map")
        write_samples_and_rec_map()

        print("parsing LD statistics (and saving per-replicate PKLs)")
        ld_stats = {}
        per_rep_times = []

        overall_start = time.perf_counter()
        for ii in range(num_reps):
            ld, dt = get_LD_stats(ii, r_bins)
            ld_stats[ii] = ld
            per_rep_times.append((ii, dt))

            # Save each replicate to MomentsLD/LD_stats/LD_stats_window_<ii>.pkl
            out_pkl = f"{LD_DIR}/LD_stats_window_{ii}.pkl"
            with open(out_pkl, "wb") as fout:
                pickle.dump(ld, fout, protocol=pickle.HIGHEST_PROTOCOL)

        overall_elapsed = time.perf_counter() - overall_start

        # ---- Timing summaries ----
        total_rep_sum = sum(dt for _, dt in per_rep_times)
        mean_rep = total_rep_sum / len(per_rep_times) if per_rep_times else 0.0
        print("\n=== LD timing summary ===")
        print(f"  per-replicate mean: {mean_rep:.2f} s")
        print(f"  per-replicate sum : {total_rep_sum:.2f} s")
        print(f"  overall wall time : {overall_elapsed:.2f} s (includes overhead)")
        print("=========================\n")

        # Save timings to CSV and PKL for easy inspection later
        timing_csv = f"{DATA_DIR}/ld_timing.csv"
        with open(timing_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["replicate", "seconds"])
            for rep, dt in per_rep_times:
                w.writerow([rep, f"{dt:.6f}"])
            w.writerow([])
            w.writerow(["mean_per_rep_seconds", f"{mean_rep:.6f}"])
            w.writerow(["sum_per_rep_seconds", f"{total_rep_sum:.6f}"])
            w.writerow(["overall_wall_seconds", f"{overall_elapsed:.6f}"])

        timing_pkl = f"{DATA_DIR}/ld_timing.pkl"
        with open(timing_pkl, "wb") as f:
            pickle.dump(
                {
                    "per_rep": per_rep_times,  # list of (rep, seconds)
                    "mean_per_rep_seconds": mean_rep,
                    "sum_per_rep_seconds": total_rep_sum,
                    "overall_wall_seconds": overall_elapsed,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"[timing] wrote {timing_csv}")
        print(f"[timing] wrote {timing_pkl}")

        print("computing mean and varcov matrix from LD statistics sums")
        mv = moments.LD.Parsing.bootstrap_data(ld_stats)
        with open(f"{DATA_DIR}/means.varcovs.split_mig.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(mv, fout)

        print("computing bootstrap replicates of mean statistics (for confidence intervals)")
        all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats)
        with open(f"{DATA_DIR}/bootstrap_sets.split_mig.{num_reps}_reps.bp", "wb+") as fout:
            pickle.dump(all_boot, fout)

        # cleanup big intermediates
        os.system("rm -f ./data/*.vcf.gz")
        os.system("rm -f ./data/*.h5")

    print("computing expectations under the model")
    g = demographic_model()
    y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_bins)
    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # plot simulated data vs expectations under the model
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=[
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ],
        labels=[
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ],
        rows=3,
        plot_vcs=True,
        show=False,
        fig_size=(6, 4),
        output="split_mig_comparison.pdf",
    )

    print("running inference")
    # Run inference using the parsed data
    demo_func = moments.LD.Demographics2D.split_mig
    # initial guess (nu0, nu1, T, m, Ne)
    p_guess = [0.1, 2, 0.075, 2, 10000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins
    )

    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "m", "Ne"]
    )

    print("Simulated parameters:")
    print(f"  N(deme0)         :  {g.demes[1].epochs[0].start_size:.1f}")
    print(f"  N(deme1)         :  {g.demes[2].epochs[0].start_size:.1f}")
    print(f"  Div. time (gen)  :  {g.demes[1].epochs[0].start_time:.1f}")
    print(f"  Migration rate   :  {g.migrations[0].rate:.6f}")
    print(f"  N(ancestral)     :  {g.demes[0].epochs[0].start_size:.1f}")

    print("best fit parameters:")
    print(f"  N(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N(deme1)         :  {physical_units[1]:.1f}")
    print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
    print(f"  Migration rate   :  {physical_units[3]:.6f}")
    print(f"  N(ancestral)     :  {physical_units[4]:.1f}")
