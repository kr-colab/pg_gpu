#!/usr/bin/env python3
"""
End-to-end IM model pipeline:

1) Simulate IM model and save .trees (+ VCFs) to IM_model_simulations/
2) Traditional moments LD (from VCFs) — PARALLEL with Ray:
   - per-replicate LD PKLs → traditional/LD_stats/
   - bootstrap means/varcovs + boot sets + plot + inference → traditional/MomentsLD_results/
   - per-rep and rolled-up timing saved in the results dir
3) GPU LD (from .trees):
   - per-replicate LD PKLs (moments-compatible structure) → GPU/LD_stats/
   - bootstrap means/varcovs + boot sets + plot + inference → GPU/MomentsLD_results/
   - per-rep and rolled-up timing saved in the results dir
"""

import os
import re
import csv
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import msprime
import tskit
import demes
import moments

# Optional: comment out if you don't have GPU path
from pg_gpu.haplotype_matrix import HaplotypeMatrix

# ---------- optional Ray (parallel for traditional LD) ----------
try:
    import ray  # noqa: F401
    _HAVE_RAY = True
except Exception:
    _HAVE_RAY = False


# --------------------------- Paths/Dirs helpers -------------------------------
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def clean_glob(pattern: str) -> None:
    os.system(f"rm -f {pattern}")


# --------------------------- Model & Simulation -------------------------------
def demographic_model():
    """
    IM model:
      Ancestor: Ne=10000
      Deme0:    Ne=2000
      Deme1:    Ne=20000
      Split:    1500 generations
      Migration: 1e-4 symmetric
    """
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=1500)])
    b.add_deme("deme0", ancestors=["anc"], epochs=[dict(start_size=2000)])
    b.add_deme("deme1", ancestors=["anc"], epochs=[dict(start_size=20000)])
    b.add_migration(demes=["deme0", "deme1"], rate=1e-4)
    return b.resolve()


def simulate_replicates(sim_dir: str, num_reps: int, L: int, mu: float, r_per_bp: float, n_per_pop: int):
    """
    Writes:
      - IM_model_simulations/window_<i>.trees
      - IM_model_simulations/split_mig.<i>.vcf.gz
    """
    ensure_dir(sim_dir)
    demog = msprime.Demography.from_demes(demographic_model())

    tree_sequences = msprime.sim_ancestry(
        {"deme0": n_per_pop, "deme1": n_per_pop},
        demography=demog,
        sequence_length=L,
        recombination_rate=r_per_bp,
        num_replicates=num_reps,
        random_seed=42,
    )
    for ii, ts in enumerate(tree_sequences):
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=ii + 1)

        ts_file = f"{sim_dir}/window_{ii}.trees"
        ts.dump(ts_file)

        vcf_name = f"{sim_dir}/split_mig.{ii}.vcf"
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip -f {vcf_name}")  # produces split_mig.<i>.vcf.gz


def write_samples_and_rec_map(sim_dir: str, L: int, r_per_bp: float, n_per_pop: int):
    # samples file
    with open(f"{sim_dir}/samples.txt", "w+") as fout:
        fout.write("sample\tpop\n")
        for jj in range(2):
            for ii in range(n_per_pop):
                fout.write(f"tsk_{jj * n_per_pop + ii}\tdeme{jj}\n")
    # recombination map, flat (cM)
    with open(f"{sim_dir}/flat_map.txt", "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{L}\t{r_per_bp * L * 100}\n")


# --------------------------- Traditional (moments) LD -------------------------
def compute_traditional_ld_for_rep(sim_dir: str, rep: int, r_bins) -> dict:
    """moments LD stats from VCF for one replicate."""
    vcf_gz = f"{sim_dir}/split_mig.{rep}.vcf.gz"
    ld = moments.LD.Parsing.compute_ld_statistics(
        vcf_gz,
        rec_map_file=f"{sim_dir}/flat_map.txt",
        pop_file=f"{sim_dir}/samples.txt",
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )
    return ld


def list_ld_pkls(ld_dir: str):
    pat = re.compile(r"LD_stats_window_(\d+)\.pkl$")
    pairs = []
    for name in os.listdir(ld_dir):
        m = pat.search(name)
        if m:
            i = int(m.group(1))
            pairs.append((i, os.path.join(ld_dir, name)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def bootstrap_and_plot(ld_stats_dict_of_dicts: dict, r_bins, results_dir: str, tag: str):
    """
    ld_stats_dict_of_dicts: {rep: {'bins','sums','stats','pops'}}
    Saves:
      - means.varcovs.<tag>.<N>_reps.bp
      - bootstrap_sets.<tag>.<N>_reps.bp
      - comparison_<tag>.pdf
      - prints inference results
    """
    ensure_dir(results_dir)

    # Bootstrap
    print(f"[{tag}] computing mean and varcov matrix")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats_dict_of_dicts)
    mv_path = f"{results_dir}/means.varcovs.{tag}.{len(ld_stats_dict_of_dicts)}_reps.bp"
    with open(mv_path, "wb") as fout:
        pickle.dump(mv, fout)

    print(f"[{tag}] computing bootstrap replicate sets")
    all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats_dict_of_dicts)
    boot_path = f"{results_dir}/bootstrap_sets.{tag}.{len(ld_stats_dict_of_dicts)}_reps.bp"
    with open(boot_path, "wb") as fout:
        pickle.dump(all_boot, fout)

    # Expectations
    print(f"[{tag}] computing model expectations and plotting")
    g = demographic_model()
    y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_bins)
    y = moments.LD.LDstats(
        [(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    plot_path = f"{results_dir}/comparison_{tag}.pdf"
    moments.LD.Plotting.plot_ld_curves_comp(
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
        output=plot_path,
    )

    # Inference
    print(f"[{tag}] running inference")
    demo_func = moments.LD.Demographics2D.split_mig
    p_guess = [0.1, 2, 0.075, 2, 10000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose=10
    )
    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "m", "Ne"]
    )

    print(f"[{tag}] best fit parameters:")
    print(f"  N(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N(deme1)         :  {physical_units[1]:.1f}")
    print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
    print(f"  Migration rate   :  {physical_units[3]:.6f}")
    print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

    return dict(mv_path=mv_path, boot_path=boot_path, plot_path=plot_path)


# --------------------------- GPU LD helpers ----------------------------------
LD_STAT_NAMES = [
    'DD_0_0', 'DD_0_1', 'DD_1_1',
    'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1',
    'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
    'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1',
    'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
]
H_STAT_NAMES = ['H_0_0', 'H_0_1', 'H_1_1']


def genetic_to_bp_bins(r_bins, r_per_bp: float):
    r_bins = np.asarray(r_bins, dtype=float)
    if r_per_bp <= 0:
        raise ValueError("r_per_bp must be > 0")
    bp_bins = r_bins / float(r_per_bp)
    if bp_bins[0] != 0 or not np.all(np.diff(bp_bins) > 0):
        raise ValueError("r_bins must start at 0 and be strictly increasing")
    return bp_bins


def build_sample_sets(ts: tskit.TreeSequence):
    # Use population names if present; else pick first two non-empty
    pop_names = {}
    for pid in range(ts.num_populations):
        pop = ts.population(pid)
        name = None
        if hasattr(pop, "name") and pop.name:
            name = pop.name
        elif hasattr(pop, "metadata") and isinstance(pop.metadata, dict):
            name = pop.metadata.get("name")
        pop_names[pid] = name

    samples_by_pid = {pid: [int(x) for x in ts.samples(population=pid)]
                      for pid in range(ts.num_populations)}

    pid_d0 = next((pid for pid, nm in pop_names.items() if nm == "deme0"), None)
    pid_d1 = next((pid for pid, nm in pop_names.items() if nm == "deme1"), None)
    if pid_d0 is None or pid_d1 is None:
        nonempty = [(pid, s) for pid, s in samples_by_pid.items() if len(s) > 0]
        nonempty.sort(key=lambda x: x[0])
        if len(nonempty) < 2:
            raise ValueError("Need two non-empty pops in the tree sequence.")
        pid_d0, pid_d1 = nonempty[0][0], nonempty[1][0]

    ss = {"deme0": samples_by_pid[pid_d0], "deme1": samples_by_pid[pid_d1]}
    if len(ss["deme0"]) == 0 or len(ss["deme1"]) == 0:
        raise ValueError("Empty sample set(s)")
    return ss


def attach_sample_sets(h: HaplotypeMatrix, sample_sets: dict):
    normalized = {k: [int(x) for x in v] for k, v in sample_sets.items()}
    if hasattr(h, "set_sample_sets") and callable(getattr(h, "set_sample_sets")):
        h.set_sample_sets(normalized)
    else:
        h.sample_sets = normalized
    return h


def gpu_ld_from_trees(ts_path: str, r_bins, r_per_bp: float, pop1="deme0", pop2="deme1", raw=True) -> dict:
    """
    Returns a moments-compatible dict:
      {'bins': [(r0,r1),...],
       'sums': [np(15) per bin, np(3) H],
       'stats': ([15 names], ['H_0_0','H_0_1','H_1_1']),
       'pops':  [pop1, pop2]}
    """
    bp_bins = genetic_to_bp_bins(r_bins, r_per_bp)
    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)
    h = HaplotypeMatrix.from_ts(ts)
    attach_sample_sets(h, sample_sets)

    try:
        stats_by_bin = h.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw, ac_filter=True
        )
    except ValueError as e:
        if "sample_sets must be defined" in str(e):
            filtered = h.apply_biallelic_filter()
            attach_sample_sets(filtered, sample_sets)
            stats_by_bin = filtered.compute_ld_statistics_gpu_two_pops(
                bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw, ac_filter=False
            )
        else:
            raise

    # Assemble sums in fixed order
    sums = []
    for (b0, b1) in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key)
        vec = np.array([od[name] for name in LD_STAT_NAMES], dtype=float)
        sums.append(vec)

    # H terms as sums over sites
    S = ts.num_sites
    H00_mean = ts.diversity(sample_sets=sample_sets['deme0'])
    H11_mean = ts.diversity(sample_sets=sample_sets['deme1'])
    H01_mean = ts.divergence([sample_sets['deme0'], sample_sets['deme1']])
    sums.append(np.array([H00_mean * S, H01_mean * S, H11_mean * S], dtype=float))

    bins_gen = [(np.float64(r_bins[i]), np.float64(r_bins[i+1])) for i in range(len(r_bins)-1)]
    return {
        "bins": bins_gen,
        "sums": sums,
        "stats": (LD_STAT_NAMES, H_STAT_NAMES),
        "pops": [pop1, pop2],
    }


# --------------------------- Timing helpers ----------------------------------
def write_timing(results_dir: str, label: str, per_rep_times: list, overall_seconds: float):
    """
    per_rep_times: list[(rep, seconds)]
    """
    ensure_dir(results_dir)
    total_rep_sum = sum(dt for _, dt in per_rep_times)
    mean_rep = total_rep_sum / len(per_rep_times) if per_rep_times else 0.0

    # print summary
    print(f"\n== {label} timing summary ==")
    print(f"  per-replicate mean: {mean_rep:.2f} s")
    print(f"  per-replicate sum : {total_rep_sum:.2f} s")
    print(f"  overall wall time : {overall_seconds:.2f} s (includes overhead)")
    print("================================\n")

    # CSV
    timing_csv = f"{results_dir}/{label}_timing.csv"
    with open(timing_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "seconds"])
        for rep, dt in per_rep_times:
            w.writerow([rep, f"{dt:.6f}"])
        w.writerow([])
        w.writerow(["mean_per_rep_seconds", f"{mean_rep:.6f}"])
        w.writerow(["sum_per_rep_seconds", f"{total_rep_sum:.6f}"])
        w.writerow(["overall_wall_seconds", f"{overall_seconds:.6f}"])

    # PKL
    timing_pkl = f"{results_dir}/{label}_timing.pkl"
    with open(timing_pkl, "wb") as f:
        pickle.dump(
            {
                "per_rep": per_rep_times,
                "mean_per_rep_seconds": mean_rep,
                "sum_per_rep_seconds": total_rep_sum,
                "overall_wall_seconds": overall_seconds,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"[timing] wrote {timing_csv}")
    print(f"[timing] wrote {timing_pkl}")


# --------------------------- Pipeline runners --------------------------------
# ---- RAY-PARALLEL traditional ----
def run_traditional(sim_dir: str, out_ld_dir: str, results_dir: str, num_reps: int, r_bins,
                    backend: str = "ray", ray_address: str = None, ray_num_cpus: int = None,
                    ray_max_inflight: int = 64):
    """
    If backend == 'ray', compute per-rep LD in parallel. Else, fall back to serial.
    """
    ensure_dir(out_ld_dir)
    ensure_dir(results_dir)

    if backend != "ray":
        # Serial fallback
        ld_stats_dict = {}
        per_times = []
        t0_global = time.perf_counter()
        for ii in range(num_reps):
            t0 = time.perf_counter()
            ld = compute_traditional_ld_for_rep(sim_dir, ii, r_bins)
            dt = time.perf_counter() - t0

            ld_stats_dict[ii] = ld
            per_times.append((ii, dt))

            out_pkl = f"{out_ld_dir}/LD_stats_window_{ii}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[traditional] wrote {out_pkl} in {dt:.2f} s")
        overall = time.perf_counter() - t0_global

        write_timing(results_dir, "traditional", per_times, overall)
        return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="traditional")

    if backend == "ray" and not _HAVE_RAY:
        raise RuntimeError("Ray is not installed but --traditional-backend ray was requested.")

    # --- RAY path ---
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(num_cpus=ray_num_cpus or None, ignore_reinit_error=True)

    @ray.remote(num_cpus=1)
    def traditional_worker(rep: int, sim_dir_: str, r_bins_local):
        import time as _t
        import moments as _moments

        t0 = _t.perf_counter()
        vcf_gz = f"{sim_dir_}/split_mig.{rep}.vcf.gz"
        ld = _moments.LD.Parsing.compute_ld_statistics(
            vcf_gz,
            rec_map_file=f"{sim_dir_}/flat_map.txt",
            pop_file=f"{sim_dir_}/samples.txt",
            pops=["deme0", "deme1"],
            r_bins=r_bins_local,   # r_bins already dereferenced
            report=False,
        )
        dt = _t.perf_counter() - t0
        return rep, ld, dt

    ld_stats_dict = {}
    per_times = []
    t0_global = time.perf_counter()

    # Put shared r_bins once (optional; also fine to pass raw array)
    r_bins_ref = ray.put(np.asarray(r_bins, dtype=float))

    # Submit with backpressure
    pending = []
    next_rep = 0
    completed = 0

    def submit_one(rep_idx):
        return traditional_worker.remote(rep_idx, sim_dir, r_bins_ref)

    # prime the pump
    while next_rep < num_reps and len(pending) < ray_max_inflight:
        pending.append(submit_one(next_rep))
        next_rep += 1

    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        rep, ld, dt = ray.get(done[0])

        # save to dict and PKL
        ld_stats_dict[rep] = ld
        per_times.append((rep, dt))
        out_pkl = f"{out_ld_dir}/LD_stats_window_{rep}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
        completed += 1
        print(f"[traditional][{completed}/{num_reps}] wrote {out_pkl} in {dt:.2f} s")

        # top up queue
        if next_rep < num_reps:
            pending.append(submit_one(next_rep))
            next_rep += 1

    overall = time.perf_counter() - t0_global
    per_times.sort(key=lambda x: x[0])

    write_timing(results_dir, "traditional", per_times, overall)

    # be tidy
    try:
        ray.shutdown()
    except Exception:
        pass

    # bootstrap + plot + inference
    return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="traditional")


def run_gpu(sim_dir: str, out_ld_dir: str, results_dir: str, num_reps: int, r_bins, r_per_bp: float):
    ensure_dir(out_ld_dir)
    ensure_dir(results_dir)

    ld_stats_dict = {}
    per_times = []
    t0_global = time.perf_counter()
    for ii in range(num_reps):
        ts_path = f"{sim_dir}/window_{ii}.trees"
        t0 = time.perf_counter()
        ld = gpu_ld_from_trees(ts_path, r_bins, r_per_bp, pop1="deme0", pop2="deme1", raw=True)
        dt = time.perf_counter() - t0

        ld_stats_dict[ii] = ld
        per_times.append((ii, dt))

        out_pkl = f"{out_ld_dir}/LD_stats_window_{ii}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[GPU] wrote {out_pkl} in {dt:.2f} s")
    overall = time.perf_counter() - t0_global

    write_timing(results_dir, "gpu", per_times, overall)

    # bootstrap + plot + inference
    return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="gpu")


# --------------------------- Main --------------------------------------------
def main():
    p = argparse.ArgumentParser(description="IM model: sims → traditional & GPU LD + MomentsLD analysis, with timing")
    p.add_argument("--num-reps", type=int, default=100)
    p.add_argument("--L", type=int, default=5_000_000)
    p.add_argument("--mu", type=float, default=1.5e-8)
    p.add_argument("--r-per-bp", type=float, default=1.5e-8)
    p.add_argument("--n-per-pop", type=int, default=10)

    # directories
    p.add_argument("--sim-dir", type=str, default="IM_model_simulations")
    p.add_argument("--traditional-ld-dir", type=str, default="traditional/LD_stats")
    p.add_argument("--traditional-results-dir", type=str, default="traditional/MomentsLD_results")
    p.add_argument("--gpu-ld-dir", type=str, default="GPU/LD_stats")
    p.add_argument("--gpu-results-dir", type=str, default="GPU/MomentsLD_results")

    # binning
    p.add_argument("--num-rbins", type=int, default=16, help="logspace bins between 1e-6 and 1e-3; total bins = this value")

    # parallelization for traditional
    p.add_argument("--traditional-backend", choices=["ray", "none"], default="ray",
                   help="Use Ray to parallelize traditional moments LD parsing.")
    p.add_argument("--ray-address", type=str, default=None,
                   help="Ray address (e.g., 'auto') to connect to a running cluster; default starts local ray.")
    p.add_argument("--ray-num-cpus", type=int, default=None,
                   help="If starting local ray, cap the number of CPUs/workers.")
    p.add_argument("--ray-max-inflight", type=int, default=32,
                   help="Max number of in-flight Ray tasks for traditional LD.")

    args = p.parse_args()

    # create dirs
    ensure_dir(args.sim_dir)
    ensure_dir(args.traditional_ld_dir)
    ensure_dir(args.traditional_results_dir)
    ensure_dir(args.gpu_ld_dir)
    ensure_dir(args.gpu_results_dir)

    # r bins (genetic distance): [0, logspace(1e-6..1e-3, num_rbins)]
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, args.num_rbins)))

    # (1) simulate + write rec map & samples into the simulations dir
    print("[SIM] simulating tree sequences + VCFs")
    # clear old intermediates in sim dir
    clean_glob(f"{args.sim_dir}/*.vcf.gz")
    clean_glob(f"{args.sim_dir}/*.h5")
    clean_glob(f"{args.sim_dir}/*.trees")

    simulate_replicates(args.sim_dir, args.num_reps, args.L, args.mu, args.r_per_bp, args.n_per_pop)
    write_samples_and_rec_map(args.sim_dir, args.L, args.r_per_bp, args.n_per_pop)

    # (2) TRADITIONAL moments LD from VCFs (Ray or serial)
    print("[TRADITIONAL] computing LD from VCFs via moments")
    traditional_artifacts = run_traditional(
        sim_dir=args.sim_dir,
        out_ld_dir=args.traditional_ld_dir,
        results_dir=args.traditional_results_dir,
        num_reps=args.num_reps,
        r_bins=r_bins,
        backend=args.traditional_backend,
        ray_address=args.ray_address,
        ray_num_cpus=args.ray_num_cpus,
        ray_max_inflight=args.ray_max_inflight,
    )

    # (3) GPU LD from .trees
    print("[GPU] computing LD from .trees via HaplotypeMatrix (GPU)")
    gpu_artifacts = run_gpu(
        sim_dir=args.sim_dir,
        out_ld_dir=args.gpu_ld_dir,
        results_dir=args.gpu_results_dir,
        num_reps=args.num_reps,
        r_bins=r_bins,
        r_per_bp=args.r_per_bp,
    )

    print("\nAll done.")
    print("Traditional artifacts:", traditional_artifacts)
    print("GPU artifacts:", gpu_artifacts)


if __name__ == "__main__":
    main()
