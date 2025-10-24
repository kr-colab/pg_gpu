#!/usr/bin/env python3
"""
Post-process precomputed LD PKLs into moments-compatible structures, bootstrap,
and plot expectations vs simulated means.

Inputs (already created):
  MomentsLD/LD_stats/LD_stats_window_<i>.pkl, each with:
    {'bins': [(r0,r1),...], 'sums': [np(15) per bin, ..., np(3) H], 'stats': (...), 'pops': [...]}

Outputs:
  ./data/means.varcovs.split_mig.<N>_reps.bp         (pickle)
  ./data/bootstrap_sets.split_mig.<N>_reps.bp        (pickle)
  ./split_mig_comparison.pdf                         (plot)
"""

import os
import re
import pickle
import numpy as np
import demes
import moments  # make sure the 'moments' package is available
from pathlib import Path
from tqdm import tqdm

LD_DIR_DEFAULT = "MomentsLD/LD_stats"
DATA_DIR_DEFAULT = "data"
PLOT_PATH_DEFAULT = "split_mig_comparison.pdf"


# --------------------------- Utilities ---------------------------------------
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


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


def list_ld_pkls(ld_dir: str):
    """Return sorted list of (rep_index, pkl_path)."""
    pat = re.compile(r"LD_stats_window_(\d+)\.pkl$")
    pairs = []
    for name in os.listdir(ld_dir):
        m = pat.search(name)
        if m:
            i = int(m.group(1))
            pairs.append((i, os.path.join(ld_dir, name)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def read_ld_pkl(pkl_path: str):
    """Load one PKL and return the full dict {'bins','sums','stats','pops'}."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    # basic sanity
    for k in ("bins", "sums", "stats", "pops"):
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {pkl_path}")
    return d


# --------------------------- Main --------------------------------------------
def main(ld_dir=LD_DIR_DEFAULT, data_dir=DATA_DIR_DEFAULT, plot_path=PLOT_PATH_DEFAULT):
    ensure_dir(data_dir)

    pairs = list_ld_pkls(ld_dir)
    if not pairs:
        raise FileNotFoundError(f"No PKLs found in {ld_dir}")

    # Read first file to get bins → r_bins (edges)
    first = read_ld_pkl(pairs[0][1])
    first_bins = first["bins"]  # list of (r0, r1) in genetic units (recomb fraction)
    r_bins = np.array([first_bins[0][0]] + [b[1] for b in first_bins], dtype=float)

    # Build dict-of-dicts for moments
    ld_stats = {}
    for i, pkl_path in tqdm(pairs, desc="Parsing LD PKLs"):
        d = read_ld_pkl(pkl_path)

        # ensure bins consistent across replicates
        bins_i = d["bins"]
        if len(bins_i) != len(first_bins) or any(
            (float(a0) != float(b0) or float(a1) != float(b1))
            for (a0, a1), (b0, b1) in zip(bins_i, first_bins)
        ):
            raise ValueError(f"Bins in {pkl_path} do not match the first PKL's bins.")

        ld_stats[i] = d  # keep full replicate dict (has "sums", which moments expects)

    # Bootstrap means/varcov from ld_stats
    print("computing mean and varcov matrix from LD statistics sums")
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    mv_path = f"{data_dir}/means.varcovs.split_mig.{len(ld_stats)}_reps.bp"
    with open(mv_path, "wb") as fout:
        pickle.dump(mv, fout)

    print("computing bootstrap replicates of mean statistics (for confidence intervals)")
    all_boot = moments.LD.Parsing.get_bootstrap_sets(ld_stats)
    boot_path = f"{data_dir}/bootstrap_sets.split_mig.{len(ld_stats)}_reps.bp"
    with open(boot_path, "wb") as fout:
        pickle.dump(all_boot, fout)

    # Compute model expectations at r-bin midpoints (match your earlier snippet)
    print("computing expectations under the model")
    g = demographic_model()
    # rho = 4 * Nref * r ; use Nref=10000 as in your example
    y = moments.Demes.LD(g, sampled_demes=["deme0", "deme1"], rho=4 * 10000 * r_bins)
    y = moments.LD.LDstats(
        [(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # Plot simulated (means with varcovs) vs expectations
    print("plotting comparison")
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


    print("running inference")
    # Run inference using the parsed data
    demo_func = moments.LD.Demographics2D.split_mig
    # Set up the initial guess
    # The split_mig function takes four parameters (nu0, nu1, T, m), and we append
    # the last parameter to fit Ne, which doesn't get passed to the function but
    # scales recombination rates so can be simultaneously fit
    p_guess = [0.1, 2, 0.075, 2, 10000]
    p_guess = moments.LD.Util.perturb_params(p_guess, fold=0.1)

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [demo_func], rs=r_bins, verbose=10
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


    print(f"[done]\n  {mv_path}\n  {boot_path}\n  {plot_path}")


if __name__ == "__main__":
    # You can tweak paths here or make them CLI args if you want
    main(ld_dir=LD_DIR_DEFAULT, data_dir=DATA_DIR_DEFAULT, plot_path=PLOT_PATH_DEFAULT)
