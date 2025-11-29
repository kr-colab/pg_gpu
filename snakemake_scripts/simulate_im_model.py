#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import im_model_core as core  # type: ignore


def main():
    p = argparse.ArgumentParser("Simulate IM model replicates and write flat_map.txt")
    p.add_argument("--sim-dir", required=True, type=str)
    p.add_argument("--num-reps", required=True, type=int)
    p.add_argument("--L", required=True, type=int)
    p.add_argument("--mu", required=True, type=float)
    p.add_argument("--r-per-bp", required=True, type=float)
    p.add_argument("--n-per-pop", required=True, type=int)
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    core.ensure_dir(sim_dir)

    first_tree = sim_dir / "window_0.trees"
    if not first_tree.exists():
        print("[SIM] running msprime simulation for all replicates")
        core.clean_glob(str(sim_dir / "*.vcf.gz"))
        core.clean_glob(str(sim_dir / "*.h5"))
        core.clean_glob(str(sim_dir / "*.trees"))
        core.simulate_replicates(
            sim_dir=str(sim_dir),
            num_reps=args.num_reps,
            L=args.L,
            mu=args.mu,
            r_per_bp=args.r_per_bp,
            n_per_pop=args.n_per_pop,
        )
    else:
        print("[SIM] found existing trees; skipping resimulation")

    print("[SIM] writing samples.txt and flat_map.txt")
    core.write_samples_and_rec_map(
        sim_dir=str(sim_dir),
        L=args.L,
        r_per_bp=args.r_per_bp,
        n_per_pop=args.n_per_pop,
    )


if __name__ == "__main__":
    main()
