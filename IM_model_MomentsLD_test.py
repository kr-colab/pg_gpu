#!/usr/bin/env python3
"""
IM model → per-replicate LD stats (GPU) saved as pkl matching MomentsLD format.

Writes, for each replicate i:
  MomentsLD/LD_stats/LD_stats_window_<i>.pkl  with keys: bins, sums, stats, pops
"""

import os
import argparse
from pathlib import Path
import pickle
import numpy as np
import msprime
import demes
import tskit
from pg_gpu.haplotype_matrix import HaplotypeMatrix
from tqdm import tqdm


# --------------------------- Utilities ---------------------------------------
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def genetic_to_bp_bins(r_bins, r_per_bp):
    """
    Convert genetic-distance bin edges (recomb fraction per bp) into base-pair bins.
    We *save* bins in genetic units, but the GPU function bins by base-pairs.
    """
    r_bins = np.asarray(r_bins, dtype=float)
    if r_per_bp <= 0:
        raise ValueError("r_per_bp must be > 0")
    bp_bins = r_bins / float(r_per_bp)
    if bp_bins[0] != 0:
        raise ValueError("First bin edge must be 0.")
    if not np.all(np.diff(bp_bins) > 0):
        raise ValueError("Bin edges must be strictly increasing.")
    return bp_bins


def build_sample_sets(ts: tskit.TreeSequence):
    """
    Return {'deme0': [int,...], 'deme1': [int,...]} as plain Python lists.
    Uses population names when available; otherwise picks first two non-empty pops.
    """
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

    pid_deme0 = next((pid for pid, nm in pop_names.items() if nm == "deme0"), None)
    pid_deme1 = next((pid for pid, nm in pop_names.items() if nm == "deme1"), None)

    if pid_deme0 is None or pid_deme1 is None:
        nonempty = [(pid, s) for pid, s in samples_by_pid.items() if len(s) > 0]
        nonempty.sort(key=lambda x: x[0])
        if len(nonempty) < 2:
            raise ValueError("Need two non-empty populations in the tree sequence.")
        pid_deme0 = nonempty[0][0]
        pid_deme1 = nonempty[1][0]

    ss = {
        "deme0": samples_by_pid[pid_deme0],
        "deme1": samples_by_pid[pid_deme1],
    }
    if len(ss["deme0"]) == 0 or len(ss["deme1"]) == 0:
        raise ValueError("Empty sample set(s) after detection.")
    return ss


def attach_sample_sets(h: HaplotypeMatrix, sample_sets: dict) -> HaplotypeMatrix:
    """Normalize to lists of ints and attach to HaplotypeMatrix."""
    normalized = {k: [int(x) for x in v] for k, v in sample_sets.items()}
    if hasattr(h, "set_sample_sets") and callable(getattr(h, "set_sample_sets")):
        h.set_sample_sets(normalized)
    else:
        # property enforces list type
        h.sample_sets = normalized
    return h


# --------------------------- Simulation model --------------------------------
def demographic_model():
    """
    IM model:
      Ancestor: Ne=10000
      Deme0:    Ne=2000
      Deme1:    Ne=20000
      Split:    1500 generations
      Migration rate: 1e-4 (symmetric)
    """
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=10000, end_time=1500)])
    b.add_deme("deme0", ancestors=["anc"], epochs=[dict(start_size=2000)])
    b.add_deme("deme1", ancestors=["anc"], epochs=[dict(start_size=20000)])
    b.add_migration(demes=["deme0", "deme1"], rate=1e-4)
    return b.resolve()


def run_msprime_replicates(out_dir="split_isolation_sims", data_dir="split_isolation_sims",
                           num_reps=1, L=5_000_000, u=1.5e-8, r=1.5e-8, n=10):
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
        ts_file = f'{out_dir}/window_{ii}.trees'
        ts.dump(ts_file)

        vcf_name = f"{data_dir}/split_mig.{ii}.vcf"
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout, allow_position_zero=True)
        os.system(f"gzip -f {vcf_name}")  # overwrite if exists


def write_samples_and_rec_map(L=5_000_000, r=1.5e-8, n=10, data_dir="data"):
    # samples file
    with open(f"{data_dir}/samples.txt", "w+") as fout:
        fout.write("sample\tpop\n")
        for jj in range(2):
            for ii in range(n):
                fout.write(f"tsk_{jj * n + ii}\tdeme{jj}\n")
    # flat recombination map (cM)
    with open(f"{data_dir}/flat_map.txt", "w+") as fout:
        fout.write("pos\tMap(cM)\n")
        fout.write("0\t0\n")
        fout.write(f"{L}\t{r * L * 100}\n")


# --------------------------- LD stats → PKL ----------------------------------
LD_STAT_NAMES = [
    'DD_0_0', 'DD_0_1', 'DD_1_1',
    'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1',
    'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
    'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1',
    'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1'
]
H_STAT_NAMES = ['H_0_0', 'H_0_1', 'H_1_1']


def compute_and_dump_ld_pkl(ts_path: str, r_bins: np.ndarray, r_per_bp: float,
                            out_pkl: str, pop1='deme0', pop2='deme1', raw=True):
    """
    Compute GPU LD stats per bin (raw sums) and dump a PKL:
      {'bins': list[(r0,r1)], 'sums': [15-vectors per bin, then 3-vector H], 'stats': (LD names, H names), 'pops': [pop1,pop2]}
    """
    # Convert r-bins (genetic) to bp-bins for computation; we save bins back in genetic units
    bp_bins = genetic_to_bp_bins(r_bins, r_per_bp)

    # Load tree sequence and build sample sets
    ts = tskit.load(ts_path)
    sample_sets = build_sample_sets(ts)

    # Build matrix and attach sets
    h = HaplotypeMatrix.from_ts(ts)
    attach_sample_sets(h, sample_sets)

    # Compute per-bin LD sums on GPU (ensure biallelic filter parity)
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

    # Assemble per-bin sums in the fixed order
    sums = []
    for (b0_bp, b1_bp) in zip(bp_bins[:-1], bp_bins[1:]):
        key = (float(b0_bp), float(b1_bp))
        od = stats_by_bin.get(key)
        # If no pairs landed in this bin, od may be zeros already; still ensure vector ordering.
        vec = np.array([od[name] for name in LD_STAT_NAMES], dtype=float)
        sums.append(vec)

    # Heterozygosity/divergence "sums" (sum over sites): derive from tskit means × #sites
    S = ts.num_sites
    # within-pop heterozygosity (mean per site)
    H00_mean = ts.diversity(sample_sets=sample_sets['deme0'])  # mean per site
    H11_mean = ts.diversity(sample_sets=sample_sets['deme1'])
    # between-pop divergence (mean per site)
    H01_mean = ts.divergence([sample_sets['deme0'], sample_sets['deme1']])

    H_vec = np.array([H00_mean * S, H01_mean * S, H11_mean * S], dtype=float)
    sums.append(H_vec)

    # Bins saved in genetic units (recomb fraction), matching your example
    bins_gen = [(np.float64(r_bins[i]), np.float64(r_bins[i+1])) for i in range(len(r_bins)-1)]

    payload = {
        'bins': bins_gen,
        'sums': sums,
        'stats': (LD_STAT_NAMES, H_STAT_NAMES),
        'pops': [pop1, pop2],
    }

    ensure_dir(os.path.dirname(out_pkl))
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------- Main --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="IM model → per-replicate LD stats PKL (GPU).")
    ap.add_argument("--num-reps", type=int, default=100)
    ap.add_argument("--L", type=int, default=5_000_000)
    ap.add_argument("--mu", type=float, default=1.5e-8)
    ap.add_argument("--r-per-bp", type=float, default=1.5e-8)
    ap.add_argument("--n-per-pop", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="split_isolation_sims")
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--ld-out-dir", type=str, default="MomentsLD/LD_stats")
    ap.add_argument("--pop1", type=str, default="deme0")
    ap.add_argument("--pop2", type=str, default="deme1")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.data_dir)
    ensure_dir(args.ld_out_dir)

    # Example genetic-distance bins (0, 1e-6 .. 1e-3) — saved in PKL as genetic units
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, 10)))  # 10 bins like your example

    # Simulate & write inputs
    run_msprime_replicates(out_dir=args.out_dir, data_dir=args.data_dir,
                           num_reps=args.num_reps, L=args.L, u=args.mu,
                           r=args.r_per_bp, n=args.n_per_pop)
    write_samples_and_rec_map(L=args.L, r=args.r_per_bp, n=args.n_per_pop, data_dir=args.data_dir)

    # Compute & dump PKLs per replicate
    for ii in range(args.num_reps):
        ts_path = f"{args.out_dir}/window_{ii}.trees"
        out_pkl = f"{args.ld_out_dir}/LD_stats_window_{ii}.pkl"
        compute_and_dump_ld_pkl(
            ts_path=ts_path,
            r_bins=r_bins,
            r_per_bp=args.r_per_bp,
            out_pkl=out_pkl,
            pop1=args.pop1,
            pop2=args.pop2,
            raw=True  # sums to match your 'sums' arrays
        )
        print(f"[done] Wrote {out_pkl}")

if __name__ == "__main__":
    main()
