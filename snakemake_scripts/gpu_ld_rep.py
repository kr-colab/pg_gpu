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
    p = argparse.ArgumentParser("Compute GPU LD for one replicate")
    p.add_argument("--sim-dir", required=True, type=str)
    p.add_argument("--gpu-ld-dir", required=True, type=str)
    p.add_argument("--rep", required=True, type=int)
    p.add_argument("--r-per-bp", required=True, type=float)
    p.add_argument("--num-rbins", required=True, type=int)
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.gpu_ld_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = args.rep

    ts_filt_path = sim_dir / f"window_{rep}.filtered.trees"

    # r_bins: same as in original script
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, args.num_rbins)))

    print(f"[GPU] computing LD for rep={rep}")
    ld = core.gpu_ld_from_trees(
        ts_path=str(ts_filt_path),
        r_bins=r_bins,
        r_per_bp=args.r_per_bp,
        pop1="deme0",
        pop2="deme1",
        raw=True,
    )

    S_filt = ld["_sitecheck"]["S_filt"]
    sha1 = ld["_sitecheck"]["sha1"]

    out_pkl = out_dir / f"LD_stats_window_{rep}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[GPU] wrote {out_pkl} | S_filt={S_filt} sha1={sha1}")


if __name__ == "__main__":
    main()
