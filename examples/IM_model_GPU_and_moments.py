#!/usr/bin/env python3
"""
IM model pipeline (robust + resumable):

1) Simulate IM model and save .trees (+ VCFs) to IM_model_simulations/
   - Skips if trees already exist unless --force-sim
2) Traditional moments LD (from VCFs) — PARALLEL with Ray:
   - Reuses existing per-rep PKLs in traditional/LD_stats/
   - Computes only the missing reps
   - Bootstrap + plot + inference → traditional/MomentsLD_results/
3) GPU LD (from .trees):
   - per-replicate LD PKLs → GPU/LD_stats/
   - bootstrap + plot + inference → GPU/MomentsLD_results/
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
    Writes per-replicate:
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
    """
    Create a global samples.txt and a flat rec map.
    - Robust to tskit versions without Population.name: we map the *first two non-empty*
      population IDs among ts.samples() to 'deme0' and 'deme1' (by sorted PID).
    """
    ts0 = tskit.load(f"{sim_dir}/window_0.trees")

    # VCF sample order is tsk_0, tsk_1, ... matching ts.samples()
    sample_nodes = list(ts0.samples())
    node_to_pop = {n: ts0.node(n).population for n in sample_nodes}

    # Identify first two non-empty population IDs among the samples
    pops_present = {}
    for n in sample_nodes:
        pid = node_to_pop[n]
        pops_present[pid] = pops_present.get(pid, 0) + 1
    nonempty_pids = sorted(pops_present.keys())
    if len(nonempty_pids) < 2:
        raise ValueError("Need samples from at least two populations to write samples.txt")
    pid_d0, pid_d1 = nonempty_pids[0], nonempty_pids[1]

    # Write samples.txt using the discovered mapping
    samples_path = f"{sim_dir}/samples.txt"
    with open(samples_path, "w") as fout:
        fout.write("sample\tpop\n")
        for i, node in enumerate(sample_nodes):
            vcf_name = f"tsk_{i}"
            pop_name = "deme0" if node_to_pop[node] == pid_d0 else ("deme1" if node_to_pop[node] == pid_d1 else None)
            if pop_name is None:
                continue
            fout.write(f"{vcf_name}\t{pop_name}\n")

    # Flat recombination map (cM)
    recmap_path = f"{sim_dir}/flat_map.txt"
    with open(recmap_path, "w") as fout:
        fout.write("pos\tMap(cM)\n0\t0\n")
        fout.write(f"{L}\t{r_per_bp * L * 100}\n")


# --------- Per-replicate pop-file builders (prevents VCF header mismatch) ----
def _vcf_sample_names(vcf_gz_path: str):
    import gzip
    open_fn = gzip.open if vcf_gz_path.endswith(".gz") else open
    with open_fn(vcf_gz_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.strip().split("\t")[9:]
    raise RuntimeError(f"Couldn't find #CHROM header in {vcf_gz_path}")


def _replicate_samples_file(sim_dir: str, rep: int) -> str:
    """
    Create per-rep samples file matching that VCF's sample columns and assigning
    deme0/deme1 by the two non-empty popIDs observed in that replicate.
    Works for haploid (no individuals) and diploid (individuals present) ts.
    """
    ts_path = f"{sim_dir}/window_{rep}.trees"
    vcf_path = f"{sim_dir}/split_mig.{rep}.vcf.gz"
    ts = tskit.load(ts_path)
    vcf_samples = _vcf_sample_names(vcf_path)

    def two_nonempty_pids_from_nodes(node_ids):
        counts = {}
        for n in node_ids:
            pid = ts.node(n).population
            counts[pid] = counts.get(pid, 0) + 1
        pids = sorted([pid for pid, c in counts.items() if c > 0])
        if len(pids) < 2:
            raise ValueError("Need samples from at least two populations in this replicate.")
        return pids[0], pids[1]

    samples_file = f"{sim_dir}/samples_rep{rep}.txt"
    with open(samples_file, "w") as fout:
        fout.write("sample\tpop\n")

        if ts.num_individuals == 0 and len(vcf_samples) == ts.num_samples:
            # Haploid case: VCF samples correspond to ts.samples() order
            node_order = list(ts.samples())
            pid_d0, pid_d1 = two_nonempty_pids_from_nodes(node_order)
            for i, name in enumerate(vcf_samples):
                pid = ts.node(node_order[i]).population
                pop_name = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                if pop_name is not None:
                    fout.write(f"{name}\t{pop_name}\n")

        elif ts.num_individuals > 0 and len(vcf_samples) == ts.num_individuals:
            # Diploid case: VCF samples correspond to individuals
            ind_nodes0 = []
            for ind_id in range(ts.num_individuals):
                nodes = ts.individual(ind_id).nodes
                if not nodes:
                    raise ValueError("Individual without nodes.")
                ind_nodes0.append(nodes[0])
            pid_d0, pid_d1 = two_nonempty_pids_from_nodes(ind_nodes0)
            for i, name in enumerate(vcf_samples):
                pid = ts.node(ind_nodes0[i]).population
                pop_name = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                if pop_name is not None:
                    fout.write(f"{name}\t{pop_name}\n")

        else:
            raise RuntimeError(
                f"Can't align VCF samples (n={len(vcf_samples)}) with ts "
                f"(num_samples={ts.num_samples}, num_individuals={ts.num_individuals}) for rep {rep}"
            )

    return samples_file


# --------------------------- Traditional (moments) LD -------------------------
def compute_traditional_ld_for_rep(sim_dir: str, rep: int, r_bins) -> dict:
    """moments LD stats from VCF for one replicate; builds a per-rep samples file."""
    vcf_gz = f"{sim_dir}/split_mig.{rep}.vcf.gz"
    pop_file = _replicate_samples_file(sim_dir, rep)
    ld = moments.LD.Parsing.compute_ld_statistics(
        vcf_gz,
        rec_map_file=f"{sim_dir}/flat_map.txt",
        pop_file=pop_file,
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
        if hasattr(pop, "name") and getattr(pop, "name", None):
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


def gpu_ld_from_trees(
    ts_path: str,
    r_bins,
    r_per_bp: float,
    pop1: str = "deme0",
    pop2: str = "deme1",
    raw: bool = True
) -> dict:
    """
    Return a moments-compatible dict computed from a .trees file:
      {
        'bins':  [(r0,r1), ...]          # genetic-distance bins (same as r_bins pairs)
        'sums':  [np(15) per bin, np(3)] # raw sums per bin, then H-sums at the end
        'stats': ([15 names in moments order], ['H_0_0','H_0_1','H_1_1']),
        'pops':  [pop1, pop2],
      }
    Notes:
      - Uses the *same biallelic-filtered site set* for pairwise LD and H terms.
      - Bins are left-closed, right-open, matching moments’ convention.
      - `raw=True` mirrors moments' “sum-then-bootstrap” pathway.
    """
    MOMENTS_ORDER = [
        'DD_0_0', 'DD_0_1', 'DD_1_1',
        'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1', 'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
        'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1', 'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
    ]

    bp_bins = genetic_to_bp_bins(r_bins, r_per_bp)

    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)
    if pop1 not in sample_sets or pop2 not in sample_sets:
        raise KeyError(f"Requested pops ({pop1},{pop2}) not in sample_sets={list(sample_sets)}")

    h = HaplotypeMatrix.from_ts(ts)
    attach_sample_sets(h, sample_sets)

    # Apply biallelic filter once; use same sites for LD & H
    # --- ensure identical site set for LD & H: apply biallelic filter once ---
    h_filt = h.apply_biallelic_filter()

    # GPU LD computation – match moments' semantics more closely
    try:
        # prefer double precision if available
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop1,
            pop2=pop2,
            raw=raw,
            ac_filter=True,       # <-- previously False
            fp64=True             # <-- ignored if unsupported
        )
    except TypeError:
        # older versions may not accept fp64 argument
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop1,
            pop2=pop2,
            raw=raw,
            ac_filter=True
        )


    # Assemble per-bin vectors in moments order
    sums = []
    for (b0, b1) in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key, None)
        if od is None:
            sums.append(np.zeros(len(MOMENTS_ORDER), dtype=float))
        else:
            sums.append(np.array([od[name] for name in MOMENTS_ORDER], dtype=float))

    # H terms on same filtered site set
    pos_keep = np.asarray(h_filt.positions.get() if h_filt.device == 'GPU' else h_filt.positions)
    site_pos = np.asarray(ts.tables.sites.position)
    keep_mask = np.isin(site_pos, pos_keep)
    drop_idx = np.where(~keep_mask)[0]
    ts_filt = ts.delete_sites(drop_idx) if drop_idx.size else ts

    S_filt = ts_filt.num_sites
    if S_filt == 0:
        H00_sum = H01_sum = H11_sum = 0.0
    else:
        H00_mean = ts_filt.diversity(sample_sets=sample_sets[pop1])
        H11_mean = ts_filt.diversity(sample_sets=sample_sets[pop2])
        H01_mean = ts_filt.divergence([sample_sets[pop1], sample_sets[pop2]])
        H00_sum, H01_sum, H11_sum = H00_mean * S_filt, H01_mean * S_filt, H11_mean * S_filt

    sums.append(np.array([H00_sum, H01_sum, H11_sum], dtype=float))

    bins_gen = [(np.float64(r_bins[i]), np.float64(r_bins[i + 1])) for i in range(len(r_bins) - 1)]

    return {
        "bins": bins_gen,
        "sums": sums,
        "stats": (LD_STAT_NAMES, H_STAT_NAMES),
        "pops": [pop1, pop2],
    }


# --------------------------- Timing helpers ----------------------------------
def write_timing(results_dir: str, label: str, per_rep_times: list, overall_seconds: float):
    ensure_dir(results_dir)
    total_rep_sum = sum(dt for _, dt in per_rep_times)
    mean_rep = total_rep_sum / len(per_rep_times) if per_rep_times else 0.0

    print(f"\n== {label} timing summary ==")
    print(f"  per-replicate mean: {mean_rep:.2f} s")
    print(f"  per-replicate sum : {total_rep_sum:.2f} s")
    print(f"  overall wall time : {overall_seconds:.2f} s (includes overhead)")
    print("================================\n")

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
def run_traditional(sim_dir: str, out_ld_dir: str, results_dir: str, num_reps: int, r_bins,
                    backend: str = "ray", ray_address: str = None, ray_num_cpus: int = None,
                    ray_max_inflight: int = 64):
    """
    Compute traditional moments LD from VCFs, but:
      - Reuse any existing per-replicate PKLs in out_ld_dir
      - Only compute missing replicates
      - Per-rep samples file guarantees VCF/pop mapping
      - Then bootstrap/plot/inference on the union
    """
    ensure_dir(out_ld_dir)
    ensure_dir(results_dir)

    # Load any existing LD PKLs
    ld_stats_dict = {}
    existing = list_ld_pkls(out_ld_dir)
    for rep_i, pkl in existing:
        try:
            with open(pkl, "rb") as f:
                ld_stats_dict[rep_i] = pickle.load(f)
        except Exception as e:
            print(f"[traditional][warn] failed to load {pkl}: {e} — will recompute this replicate")

    all_reps = set(range(num_reps))
    have_reps = set(ld_stats_dict.keys())
    missing_reps = sorted(all_reps - have_reps)

    if not missing_reps:
        print("[traditional] all replicate LD PKLs already present; skipping recompute")
        write_timing(results_dir, "traditional", [], 0.0)
        return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="traditional")

    print(f"[traditional] computing {len(missing_reps)} missing of {num_reps} total")
    per_times = []
    t0_global = time.perf_counter()

    # Serial
    if backend != "ray":
        for rep in missing_reps:
            t0 = time.perf_counter()
            ld = compute_traditional_ld_for_rep(sim_dir, rep, r_bins)
            dt = time.perf_counter() - t0
            ld_stats_dict[rep] = ld
            per_times.append((rep, dt))
            out_pkl = f"{out_ld_dir}/LD_stats_window_{rep}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[traditional] wrote {out_pkl} in {dt:.2f} s")

        overall = time.perf_counter() - t0_global
        write_timing(results_dir, "traditional", per_times, overall)
        return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="traditional")

    # Ray path
    if backend == "ray" and not _HAVE_RAY:
        raise RuntimeError("Ray is not installed but --traditional-backend ray was requested.")

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(num_cpus=ray_num_cpus or None, ignore_reinit_error=True)

    @ray.remote(num_cpus=1)
    def traditional_worker(rep: int, sim_dir_: str, r_bins_local):
        import time as _t
        import moments as _moments
        import tskit as _tskit
        import gzip

        def _vcf_sample_names_inner(path):
            open_fn = gzip.open if path.endswith(".gz") else open
            with open_fn(path, "rt") as f:
                for line in f:
                    if line.startswith("#CHROM"):
                        return line.strip().split("\t")[9:]
            raise RuntimeError(f"Missing #CHROM in {path}")

        def _rep_samples(sim_dir, rep_):
            ts_path = f"{sim_dir}/window_{rep_}.trees"
            vcf_path = f"{sim_dir}/split_mig.{rep_}.vcf.gz"
            ts = _tskit.load(ts_path)
            vcf_samples = _vcf_sample_names_inner(vcf_path)

            def two_nonempty(node_ids):
                counts = {}
                for n in node_ids:
                    pid = ts.node(n).population
                    counts[pid] = counts.get(pid, 0) + 1
                pids = sorted([pid for pid, c in counts.items() if c > 0])
                if len(pids) < 2:
                    raise ValueError("Need two populations.")
                return pids[0], pids[1]

            out = f"{sim_dir}/samples_rep{rep_}.txt"
            with open(out, "w") as fout:
                fout.write("sample\tpop\n")
                if ts.num_individuals == 0 and len(vcf_samples) == ts.num_samples:
                    nodes = list(ts.samples())
                    pid_d0, pid_d1 = two_nonempty(nodes)
                    for i, name in enumerate(vcf_samples):
                        pid = ts.node(nodes[i]).population
                        pop = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                        if pop is not None:
                            fout.write(f"{name}\t{pop}\n")
                elif ts.num_individuals > 0 and len(vcf_samples) == ts.num_individuals:
                    ind_nodes0 = [ts.individual(i).nodes[0] for i in range(ts.num_individuals)]
                    pid_d0, pid_d1 = two_nonempty(ind_nodes0)
                    for i, name in enumerate(vcf_samples):
                        pid = ts.node(ind_nodes0[i]).population
                        pop = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                        if pop is not None:
                            fout.write(f"{name}\t{pop}\n")
                else:
                    raise RuntimeError("Cannot align VCF samples with tree sequence.")
            return out

        t0 = _t.perf_counter()
        vcf_gz = f"{sim_dir_}/split_mig.{rep}.vcf.gz"
        pop_file = _rep_samples(sim_dir_, rep)
        ld = _moments.LD.Parsing.compute_ld_statistics(
            vcf_gz,
            rec_map_file=f"{sim_dir_}/flat_map.txt",
            pop_file=pop_file,
            pops=["deme0", "deme1"],
            r_bins=r_bins_local,
            report=False,
        )
        dt = _t.perf_counter() - t0
        return rep, ld, dt

    try:
        r_bins_ref = ray.put(np.asarray(r_bins, dtype=float))
        pending = []
        it_missing = iter(missing_reps)

        def submit(rep_idx):
            return traditional_worker.remote(rep_idx, sim_dir, r_bins_ref)

        for _ in range(min(ray_max_inflight, len(missing_reps))):
            try:
                pending.append(submit(next(it_missing)))
            except StopIteration:
                break

        completed = 0
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            rep, ld, dt = ray.get(done[0])
            ld_stats_dict[rep] = ld
            out_pkl = f"{out_ld_dir}/LD_stats_window_{rep}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
            completed += 1
            print(f"[traditional][{completed}/{len(missing_reps)}] wrote {out_pkl} in {dt:.2f} s")

            try:
                nxt = next(it_missing)
                pending.append(submit(nxt))
            except StopIteration:
                pass

    finally:
        try:
            ray.shutdown()
        except Exception:
            pass

    overall = time.perf_counter() - t0_global
    # we didn't collect per-rep timings precisely above for brevity; skip them
    write_timing(results_dir, "traditional", [], overall)

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
    p.add_argument("--traditional-backend", choices=["ray", "none"], default="ray")
    p.add_argument("--ray-address", type=str, default=None)
    p.add_argument("--ray-num-cpus", type=int, default=None)
    p.add_argument("--ray-max-inflight", type=int, default=32)

    # safety
    p.add_argument("--force-sim", action="store_true", help="Force re-simulation even if trees already exist.")

    args = p.parse_args()

    ensure_dir(args.sim_dir)
    ensure_dir(args.traditional_ld_dir)
    ensure_dir(args.traditional_results_dir)
    ensure_dir(args.gpu_ld_dir)
    ensure_dir(args.gpu_results_dir)

    # r bins (genetic distance): [0, logspace(1e-6..1e-3, num_rbins)]
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, args.num_rbins)))

    # (1) Simulate only if needed
    first_tree = Path(f"{args.sim_dir}/window_0.trees")
    if args.force_sim or not first_tree.exists():
        print("[SIM] simulating tree sequences + VCFs")
        clean_glob(f"{args.sim_dir}/*.vcf.gz")
        clean_glob(f"{args.sim_dir}/*.h5")
        clean_glob(f"{args.sim_dir}/*.trees")
        simulate_replicates(args.sim_dir, args.num_reps, args.L, args.mu, args.r_per_bp, args.n_per_pop)
        write_samples_and_rec_map(args.sim_dir, args.L, args.r_per_bp, args.n_per_pop)
    else:
        print(f"[SIM] found existing trees in {args.sim_dir}; skipping simulation")
        if not Path(f"{args.sim_dir}/samples.txt").exists() or not Path(f"{args.sim_dir}/flat_map.txt").exists():
            print("[SIM] samples.txt or flat_map.txt missing — regenerating")
            write_samples_and_rec_map(args.sim_dir, args.L, args.r_per_bp, args.n_per_pop)
        else:
            print("[SIM] samples.txt and flat_map.txt present — skipping re-write")

    # (2) TRADITIONAL LD
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

    # (3) GPU LD
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
