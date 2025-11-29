#!/usr/bin/env python3
"""
IM model pipeline (robust + resumable) with IDENTICAL SITE SETS across methods.

Flow:
1) Simulate IM model and save .trees (+ VCFs) to IM_model_simulations/
2) Build ONE site set per replicate by applying a single biallelic filter to .trees
   -> materialize filtered tree sequence (ts_filt) and write filtered VCF
3) Traditional LD (moments) reads the filtered VCF
4) GPU LD reads the filtered tree sequence
=> Both consume identical sites. We verify with S_filt and SHA1 of positions.

This version also:
- Logs GPU memory via `nvidia-smi` around GPU LD calls.
- Prints detailed exception info if GPU steps fail.
"""

import os
import re
import csv
import time
import pickle
import argparse
import hashlib
import subprocess
import textwrap
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


# --------------------------- GPU memory helpers -------------------------------
def _gpu_memory_snapshot(label: str = "") -> None:
    """
    Log GPU memory usage via nvidia-smi.
    Safe to call even if nvidia-smi is missing (will just print a warning).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[GPU MEM]{f'[{label}]' if label else ''} could not query nvidia-smi: {type(e).__name__}: {e}")
        return

    print(f"[GPU MEM]{f'[{label}]' if label else ''}")
    for line in out.strip().splitlines():
        idx, name, total, used, free = [x.strip() for x in line.split(",")]
        print(
            textwrap.dedent(
                f"""\
                - GPU {idx} ({name})
                  total = {total} MiB
                  used  = {used} MiB
                  free  = {free} MiB"""
            )
        )


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
    - Robust across tskit versions: infer the first two non-empty population IDs.
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


# --------- VCF header reader (gz/flat) ---------------------------------------
def _vcf_sample_names(vcf_path: str):
    import gzip
    open_fn = gzip.open if vcf_path.endswith(".gz") else open
    with open_fn(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.strip().split("\t")[9:]
    raise RuntimeError(f"Couldn't find #CHROM header in {vcf_path}")


# ======================= SINGLE-SOURCE SITE SET BUILDER =======================
def _hash_positions(positions: np.ndarray) -> str:
    """SHA1 of float64 positions, little-endian bytes."""
    buf = positions.astype(np.float64).tobytes(order="C")
    return hashlib.sha1(buf).hexdigest()


def _build_filtered_ts_and_cache(
    sim_dir: str,
    rep: int,
    max_sites: int | None = None,
    rng_seed: int | None = None,
):
    """
    Load window_<rep>.trees, apply ONE biallelic filter via HaplotypeMatrix,
    optionally THIN the kept sites to at most `max_sites`, and persist:

      - filtered trees:   window_<rep>.filtered.trees
      - filtered VCF:     split_mig.filtered.<rep>.vcf.gz
      - kept site list:   kept_sites_rep<rep>.txt  (tab-separated index,position)
      - kept site sha1:   kept_sites_rep<rep>.sha1 (first line: N <tab> sha1)

    All downstream consumers (traditional & GPU) use these filtered artifacts.
    If max_sites is None, no thinning is performed beyond biallelic filtering.
    """
    ts_path = f"{sim_dir}/window_{rep}.trees"
    ts = tskit.load(ts_path)

    # Original site positions (before any filtering)
    site_pos = np.asarray(ts.tables.sites.position)
    n_sites_start = int(site_pos.size)

    # Apply biallelic filter once via HaplotypeMatrix
    h = HaplotypeMatrix.from_ts(ts)
    h_filt = h.apply_biallelic_filter()

    # Positions kept by the biallelic filter (on host)
    if getattr(h_filt, "device", None) == "GPU":
        pos_keep = np.asarray(h_filt.positions.get())
    else:
        pos_keep = np.asarray(h_filt.positions)

    n_sites_bi = int(pos_keep.size)
    print(f"[FILTER][rep={rep}] start={n_sites_start} sites, biallelic_keep={n_sites_bi}")

    # Optional thinning: reduce to at most max_sites (if requested)
    if max_sites is not None and n_sites_bi > max_sites:
        seed = rng_seed if rng_seed is not None else (1337 + int(rep))
        rng = np.random.default_rng(seed)
        # choose a subset of the biallelic-kept positions
        keep_idx = np.sort(rng.choice(n_sites_bi, size=max_sites, replace=False))
        pos_keep = pos_keep[keep_idx]
        print(
            f"[FILTER][rep={rep}] thinning biallelic sites from {n_sites_bi} "
            f"to {max_sites} (deterministic with seed={seed})"
        )

    # Now drop all sites in ts whose position is NOT in pos_keep
    # (so ts_filt has exactly the biallelic + thinned site set)
    keep_mask = np.isin(site_pos, pos_keep)
    drop_idx = np.where(~keep_mask)[0]

    if drop_idx.size:
        ts_filt = ts.delete_sites(drop_idx)
    else:
        ts_filt = ts

    S_filt = int(ts_filt.num_sites)
    sha1 = _hash_positions(np.asarray(ts_filt.tables.sites.position))
    print(f"[sites][rep={rep}] S_filt={S_filt} sha1={sha1}")

    # Persist filtered trees
    ts_filt_path = f"{sim_dir}/window_{rep}.filtered.trees"
    ts_filt.dump(ts_filt_path)

    # Persist filtered VCF (Traditional + GPU must use this same site set)
    vcf_filt = f"{sim_dir}/split_mig.filtered.{rep}.vcf"
    with open(vcf_filt, "w") as fout:
        ts_filt.write_vcf(fout, allow_position_zero=True)
    os.system(f"gzip -f {vcf_filt}")  # -> .vcf.gz
    vcf_filt_gz = f"{vcf_filt}.gz"

    # Persist kept sites (index, position) for debugging/QC
    kept_txt = f"{sim_dir}/kept_sites_rep{rep}.txt"
    with open(kept_txt, "w") as f:
        f.write("idx\tposition\n")
        for i, p in enumerate(ts_filt.tables.sites.position):
            f.write(f"{i}\t{p:.0f}\n")  # integer bp positions from msprime

    # Persist sha1 summary
    sha_path = f"{sim_dir}/kept_sites_rep{rep}.sha1"
    with open(sha_path, "w") as f:
        f.write(f"{S_filt}\t{sha1}\n")

    print(
        f"[FILTER][rep={rep}] ts_filt={ts_filt_path} "
        f"vcf={vcf_filt_gz} S_filt={S_filt} sha1={sha1}"
    )

    return dict(
        ts_filt_path=ts_filt_path,
        vcf_filt_gz=vcf_filt_gz,
        S_filt=S_filt,
        sha1=sha1,
    )


# --------- Per-replicate pop-file builder (FROM FILTERED VCF HEADER) ---------
def _replicate_samples_file(sim_dir: str, rep: int, vcf_path: str) -> str:
    """
    Create per-rep samples file matching *FILTERED* VCF's sample columns and assigning
    deme0/deme1 by the two non-empty popIDs observed in that replicate.
    Works for haploid (no individuals) and diploid (individuals present) ts.
    """
    ts_path = f"{sim_dir}/window_{rep}.trees"
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

        # Diploid: VCF samples correspond to individuals
        if ts.num_individuals > 0 and len(vcf_samples) == ts.num_individuals:
            ind_nodes0 = []
            for ind_id in range(ts.num_individuals):
                nodes = ts.individual(ind_id).nodes
                # nodes is a NumPy array; check length explicitly
                if len(nodes) == 0:
                    raise ValueError("Individual without nodes.")
                ind_nodes0.append(int(nodes[0]))
            pid_d0, pid_d1 = two_nonempty_pids_from_nodes(ind_nodes0)
            for i, name in enumerate(vcf_samples):
                pid = ts.node(ind_nodes0[i]).population
                pop_name = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                if pop_name is not None:
                    fout.write(f"{name}\t{pop_name}\n")


        # Haploid: VCF samples correspond to ts.samples() order
        elif ts.num_individuals == 0 and len(vcf_samples) == ts.num_samples:
            node_order = list(ts.samples())
            pid_d0, pid_d1 = two_nonempty_pids_from_nodes(node_order)
            for i, name in enumerate(vcf_samples):
                pid = ts.node(node_order[i]).population
                pop_name = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                if pop_name is not None:
                    fout.write(f"{name}\t{pop_name}\n")

        else:
            raise RuntimeError(
                f"Can't align filtered VCF samples (n={len(vcf_samples)}) with ts "
                f"(num_samples={ts.num_samples}, num_individuals={ts.num_individuals}) for rep {rep}"
            )

    return samples_file


# --------------------------- Traditional (moments) LD -------------------------
def compute_traditional_ld_for_rep(sim_dir: str, rep: int, r_bins) -> dict:
    """
    moments LD stats from *FILTERED* VCF for one replicate; builds a per-rep samples file
    from the same filtered VCF header (guaranteed alignment).
    """
    built = _build_filtered_ts_and_cache(sim_dir, rep)  # ensures filtered artifacts exist
    vcf_gz = built["vcf_filt_gz"]
    pop_file = _replicate_samples_file(sim_dir, rep, vcf_gz)

    # Quick sanity: ensure every sample in pop_file is present in filtered VCF header
    header_set = set(_vcf_sample_names(vcf_gz))
    with open(pop_file) as f:
        next(f)
        names = [ln.split()[0] for ln in f if ln.strip()]
    missing = [nm for nm in names if nm not in header_set]
    if missing:
        raise RuntimeError(f"Pop-file sample(s) missing in filtered VCF header: {missing[:5]}{'...' if len(missing)>5 else ''}")

    ld = moments.LD.Parsing.compute_ld_statistics(
        vcf_gz,
        rec_map_file=f"{sim_dir}/flat_map.txt",
        pop_file=pop_file,
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )
    ld["_sitecheck"] = dict(S_filt=built["S_filt"], sha1=built["sha1"])
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


def _bin_midpoints(r_bins: np.ndarray) -> np.ndarray:
    r_edges = np.asarray(r_bins, dtype=float)
    if r_edges.ndim != 1 or r_edges.size < 2:
        raise ValueError("r_bins must be 1D with at least two edges")
    return np.sqrt(r_edges[:-1] * r_edges[1:])


def bootstrap_and_plot(ld_stats_dict_of_dicts: dict, r_bins, results_dir: str, tag: str):
    """
    ld_stats_dict_of_dicts: {rep: {'bins','sums','stats','pops', '_sitecheck':{S_filt,sha1}}}
    Saves:
      - means.varcovs.<tag>.<N>_reps.bp
      - bootstrap_sets.<tag>.<N>_reps.bp
      - comparison_<tag>.pdf
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

    # Expectations at bin midpoints (better vs. edge-averaging)
    print(f"[{tag}] computing model expectations and plotting")
    g = demographic_model()
    r_mids = _bin_midpoints(np.asarray(r_bins, dtype=float))
    y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_mids)
    y = moments.LD.Inference.sigmaD2(y)

    plot_path = f"{results_dir}/comparison_{tag}.pdf"
    moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],  # Exclude H statistics for plotting
        mv["varcovs"][:-1],  # Exclude H statistics for plotting
        rs=r_mids,
        binned_data=False,  # Explicitly set to False to avoid auto-detection issues
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

    try:
        opt_params, LL = moments.LD.Inference.optimize_log_fmin(
            p_guess, [mv["means"][:-1], mv["varcovs"][:-1]], [demo_func], rs=r_mids, verbose=10
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

    except np.linalg.LinAlgError as e:
        print(f"[{tag}] Optimization failed due to singular matrix: {e}")
        print(f"[{tag}] This can happen with insufficient data variation or perfect correlations")
        print(f"[{tag}] Bootstrap and plot files have been saved successfully")
        physical_units = [np.nan] * 5
        LL = np.nan

    except Exception as e:
        print(f"[{tag}] Optimization failed with error: {e}")
        print(f"[{tag}] Bootstrap and plot files have been saved successfully")
        physical_units = [np.nan] * 5
        LL = np.nan

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


def _H_sums_from_filtered_ts(ts: tskit.TreeSequence, sample_sets: dict) -> tuple[float, float, float]:
    """
    Compute H_00, H_01, H_11 *sums* over the exact (filtered, biallelic) site set in ts.
    Works for haploid or diploid: we operate on sample NODES (haploid chromosomes).
    Missing genotypes (-1) are ignored site-wise.
    """
    # Map node -> index in ts.samples() to index into variant.genotypes
    samples_vec = np.array(list(ts.samples()), dtype=np.int64)
    node_to_idx = {int(n): i for i, n in enumerate(samples_vec)}

    # Indices of samples for each pop in the genotypes vector
    idx_A = np.array([node_to_idx[n] for n in sample_sets["deme0"] if n in node_to_idx], dtype=np.int64)
    idx_B = np.array([node_to_idx[n] for n in sample_sets["deme1"] if n in node_to_idx], dtype=np.int64)

    if idx_A.size == 0 or idx_B.size == 0:
        raise ValueError("Empty index set while building H sums.")

    H00 = 0.0
    H11 = 0.0
    H01 = 0.0

    # Iterate over the *filtered* site set
    for var in ts.variants(samples=samples_vec, alleles=None, impute_missing_data=False):
        g = var.genotypes  # 0/1 (biallelic), -1 for missing
        # Mask missing per population
        gA = g[idx_A]
        gB = g[idx_B]
        valA = gA[gA >= 0]
        valB = gB[gB >= 0]
        if valA.size == 0 or valB.size == 0:
            # Skip this site if one pop is entirely missing
            continue

        # Allele frequency per haplotype (chromosome) in each pop
        pA = float(valA.mean())  # mean of {0,1}
        pB = float(valB.mean())

        # Per-site contributions
        H00 += 2.0 * pA * (1.0 - pA)
        H11 += 2.0 * pB * (1.0 - pB)
        H01 += pA * (1.0 - pB) + (1.0 - pA) * pB

    return H00, H01, H11


def gpu_ld_from_trees(
    ts_path: str,
    r_bins,
    r_per_bp: float,
    pop1: str = "deme0",
    pop2: str = "deme1",
    raw: bool = True,
) -> dict:
    """
    Return a moments-compatible dict computed from a *FILTERED* .trees file.
    Also includes _sitecheck {S_filt, sha1} for verification.

    Uses HaplotypeMatrix.compute_ld_statistics_gpu_two_pops without fp64.
    Logs GPU memory before/after to help debug OOM issues.
    """
    MOMENTS_ORDER = [
        "DD_0_0",
        "DD_0_1",
        "DD_1_1",
        "Dz_0_0_0",
        "Dz_0_0_1",
        "Dz_0_1_1",
        "Dz_1_0_0",
        "Dz_1_0_1",
        "Dz_1_1_1",
        "pi2_0_0_0_0",
        "pi2_0_0_0_1",
        "pi2_0_0_1_1",
        "pi2_0_1_0_1",
        "pi2_0_1_1_1",
        "pi2_1_1_1_1",
    ]

    # convert r-bins (genetic) -> bp-bins for GPU code
    bp_bins = genetic_to_bp_bins(r_bins, r_per_bp)

    # load filtered tree sequence
    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)
    if pop1 not in sample_sets or pop2 not in sample_sets:
        raise KeyError(
            f"Requested pops ({pop1},{pop2}) not in sample_sets={list(sample_sets)}"
        )

    # ---- GPU memory BEFORE building haplotype matrix ----
    _gpu_memory_snapshot(label=f"before HaplotypeMatrix (ts={ts_path})")

    h = HaplotypeMatrix.from_ts(ts)
    attach_sample_sets(h, sample_sets)

    # ts_path is already filtered; biallelic filter should be a no-op
    h_filt = h.apply_biallelic_filter()

    # ---- GPU memory BEFORE LD stats ----
    _gpu_memory_snapshot(label=f"before LD stats (ts={ts_path})")

    try:
        # single, clean call – no fp64 kwarg
        stats_by_bin = h_filt.compute_ld_statistics_gpu_two_pops(
            bp_bins=bp_bins,
            pop1=pop1,
            pop2=pop2,
            raw=raw,
            ac_filter=True,
        )
    except Exception as e:
        # Detailed logging on GPU failure, then re-raise so Snakemake sees the error
        print(f"[GPU ERROR] compute_ld_statistics_gpu_two_pops failed on {ts_path}")
        print(f"[GPU ERROR] Exception type: {type(e).__name__}")
        print(f"[GPU ERROR] Message       : {e}")
        _gpu_memory_snapshot(label=f"after failure (ts={ts_path})")
        raise

    # ---- GPU memory AFTER LD stats ----
    _gpu_memory_snapshot(label=f"after LD stats (ts={ts_path})")

    # Assemble per-bin vectors in moments order
    sums = []
    for (b0, b1) in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0), float(b1))
        od = stats_by_bin.get(key, None)
        if od is None:
            sums.append(np.zeros(len(MOMENTS_ORDER), dtype=float))
        else:
            sums.append(
                np.array([od[name] for name in MOMENTS_ORDER], dtype=float)
            )

    # H terms: sums over sites on the same filtered TS
    H00_sum, H01_sum, H11_sum = _H_sums_from_filtered_ts(ts, sample_sets)
    sums.append(np.array([H00_sum, H01_sum, H11_sum], dtype=float))

    # bins in genetic units (for moments)
    bins_gen = [
        (np.float64(r_bins[i]), np.float64(r_bins[i + 1]))
        for i in range(len(r_bins) - 1)
    ]

    # Verification tag (site count + SHA1 of positions)
    positions = np.asarray(ts.tables.sites.position)
    sha1 = _hash_positions(positions)
    S_filt = int(ts.num_sites)

    return {
        "bins": bins_gen,
        "sums": sums,
        "stats": (LD_STAT_NAMES, H_STAT_NAMES),
        "pops": [pop1, pop2],
        "_sitecheck": {"S_filt": S_filt, "sha1": sha1},
    }


# --------------------------- Timing helpers ----------------------------------
def write_timing(results_dir: str, label: str, per_rep_times: list, overall_seconds: float):
    ensure_dir(results_dir)
    total_rep_sum = sum(dt for _, dt in per_rep_times) if per_rep_times else 0.0
    mean_rep = (total_rep_sum / len(per_rep_times)) if per_rep_times else 0.0

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
    Compute traditional moments LD from FILTERED VCFs, but:
      - Reuse any existing per-replicate PKLs in out_ld_dir
      - Only compute missing replicates
      - Per-rep samples file is built from the filtered VCF header (exact column match)
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

    # Serial path
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
            print(f"[traditional] wrote {out_pkl} in {dt:.2f} s  | sitecheck: S={ld['_sitecheck']['S_filt']} sha1={ld['_sitecheck']['sha1']}")

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
        import numpy as _np
        import tskit as _tskit
        from hashlib import sha1 as _sha1
        from pg_gpu.haplotype_matrix import HaplotypeMatrix as _HM
        import gzip as _gzip
        import os as _os

        def _hash_pos(arr):
            return _sha1(_np.asarray(arr, dtype=_np.float64).tobytes(order="C")).hexdigest()

        # Build (or reuse) filtered artifacts inside worker
        ts_path = f"{sim_dir_}/window_{rep}.trees"
        ts = _tskit.load(ts_path)
        h = _HM.from_ts(ts)
        h_filt = h.apply_biallelic_filter()
        pos_keep = _np.asarray(h_filt.positions.get() if getattr(h_filt, "device", None) == "GPU" else h_filt.positions)
        site_pos = _np.asarray(ts.tables.sites.position)
        drop_idx = _np.where(~_np.isin(site_pos, pos_keep))[0]
        ts_filt = ts.delete_sites(drop_idx) if drop_idx.size else ts

        # write (idempotent) filtered VCF for moments
        vcf_filt = f"{sim_dir_}/split_mig.filtered.{rep}.vcf"
        with open(vcf_filt, "w") as fout:
            ts_filt.write_vcf(fout, allow_position_zero=True)
        _os.system(f"gzip -f {vcf_filt}")
        vcf_gz = f"{vcf_filt}.gz"

        # pop file from FILTERED VCF header
        def _vcf_names(path):
            ofn = _gzip.open if path.endswith(".gz") else open
            with ofn(path, "rt") as f:
                for line in f:
                    if line.startswith("#CHROM"):
                        return line.strip().split("\t")[9:]
            raise RuntimeError("Missing #CHROM")
        vcf_samples = _vcf_names(vcf_gz)

        pop_file = f"{sim_dir_}/samples_rep{rep}.txt"
        with open(pop_file, "w") as fout:
            fout.write("sample\tpop\n")
            if ts.num_individuals > 0 and len(vcf_samples) == ts.num_individuals:
                ind_nodes0 = [ts.individual(i).nodes[0] for i in range(ts.num_individuals)]
                counts = {}
                for n in ind_nodes0:
                    pid = ts.node(n).population
                    counts[pid] = counts.get(pid, 0) + 1
                pids = sorted([pid for pid, c in counts.items() if c > 0])
                pid_d0, pid_d1 = pids[0], pids[1]
                for i, name in enumerate(vcf_samples):
                    pid = ts.node(ind_nodes0[i]).population
                    pop = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                    if pop is not None:
                        fout.write(f"{name}\t{pop}\n")
            else:
                nodes = list(ts.samples())
                counts = {}
                for n in nodes:
                    pid = ts.node(n).population
                    counts[pid] = counts.get(pid, 0) + 1
                pids = sorted([pid for pid, c in counts.items() if c > 0])
                pid_d0, pid_d1 = pids[0], pids[1]
                # Here vcf_samples length must equal ts.num_samples
                for i, name in enumerate(vcf_samples):
                    pid = ts.node(nodes[i]).population
                    pop = "deme0" if pid == pid_d0 else ("deme1" if pid == pid_d1 else None)
                    if pop is not None:
                        fout.write(f"{name}\t{pop}\n")

        # quick sanity: ensure pop_file names exist in filtered VCF header
        header_set = set(vcf_samples)
        with open(pop_file) as f:
            next(f)
            names = [ln.split()[0] for ln in f if ln.strip()]
        missing = [nm for nm in names if nm not in header_set]
        if missing:
            raise RuntimeError(f"Pop-file sample(s) missing in filtered VCF header: {missing[:5]}{'...' if len(missing)>5 else ''}")

        t0 = _t.perf_counter()
        ld = _moments.LD.Parsing.compute_ld_statistics(
            vcf_gz,
            rec_map_file=f"{sim_dir_}/flat_map.txt",
            pop_file=pop_file,
            pops=["deme0", "deme1"],
            r_bins=r_bins_local,
            report=False,
        )
        dt = _t.perf_counter() - t0
        ld["_sitecheck"] = dict(S_filt=int(ts_filt.num_sites), sha1=_hash_pos(ts_filt.tables.sites.position))
        return rep, ld, dt

    per_times = []
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
            per_times.append((rep, dt))
            out_pkl = f"{out_ld_dir}/LD_stats_window_{rep}.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
            completed += 1
            print(f"[traditional][{completed}/{len(missing_reps)}] wrote {out_pkl} in {dt:.2f} s | sitecheck: S={ld['_sitecheck']['S_filt']} sha1={ld['_sitecheck']['sha1']}")

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
    write_timing(results_dir, "traditional", per_times, overall)

    return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="traditional")


def run_gpu(sim_dir: str, out_ld_dir: str, results_dir: str, num_reps: int, r_bins, r_per_bp: float):
    """
    Run GPU LD on all replicates, logging timing and GPU memory.

    If a replicate fails, prints detailed info (exception type/message + memory snapshot)
    and re-raises the exception.
    """
    ensure_dir(out_ld_dir)
    ensure_dir(results_dir)

    ld_stats_dict = {}
    per_times = []
    t0_global = time.perf_counter()
    for ii in range(num_reps):
        # Always use the pre-built FILTERED trees
        built = _build_filtered_ts_and_cache(sim_dir, ii)  # idempotent
        ts_path = built["ts_filt_path"]

        try:
            t0 = time.perf_counter()
            ld = gpu_ld_from_trees(ts_path, r_bins, r_per_bp, pop1="deme0", pop2="deme1", raw=True)
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"[GPU][rep={ii}] FAILED on {ts_path}")
            print(f"[GPU][rep={ii}] Exception type: {type(e).__name__}")
            print(f"[GPU][rep={ii}] Message       : {e}")
            _gpu_memory_snapshot(label=f"failure in run_gpu rep={ii}")
            # re-raise so caller / Snakemake sees a hard error
            raise

        # quick cross-check print
        print(f"[sites][GPU rep={ii}] S_filt={ld['_sitecheck']['S_filt']} sha1={ld['_sitecheck']['sha1']}")

        ld_stats_dict[ii] = ld
        per_times.append((ii, dt))

        out_pkl = f"{out_ld_dir}/LD_stats_window_{ii}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[GPU] wrote {out_pkl} in {dt:.2f} s")
    overall = time.perf_counter() - t0_global

    write_timing(results_dir, "gpu", per_times, overall)
    return bootstrap_and_plot(ld_stats_dict, r_bins, results_dir, tag="gpu")
