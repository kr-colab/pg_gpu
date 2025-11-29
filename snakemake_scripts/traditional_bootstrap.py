#!/usr/bin/env python3
import argparse
import sys
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import im_model_core as core  # type: ignore


def main():
    p = argparse.ArgumentParser("Bootstrap + plot traditional LD across all replicates")
    p.add_argument("--trad-ld-dir", required=True, type=str)
    p.add_argument("--trad-results-dir", required=True, type=str)
    p.add_argument("--num-reps", required=True, type=int)
    p.add_argument("--num-rbins", required=True, type=int)
    args = p.parse_args()

    ld_dir = Path(args.trad_ld_dir)
    res_dir = Path(args.trad_results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # r_bins as in original
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, args.num_rbins)))

    ld_stats = {}
    for rep in range(args.num_reps):
        pkl = ld_dir / f"LD_stats_window_{rep}.pkl"
        with open(pkl, "rb") as f:
            ld_stats[rep] = pickle.load(f)

    print("[TRAD] bootstrapping + plotting")
    core.bootstrap_and_plot(
        ld_stats_dict_of_dicts=ld_stats,
        r_bins=r_bins,
        results_dir=str(res_dir),
        tag="traditional",
    )


if __name__ == "__main__":
    main()
