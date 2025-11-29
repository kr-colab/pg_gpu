#!/usr/bin/env python3
"""
Wrapper script for Snakemake rule `filter_sites`.

For a given replicate:
  - loads IM_model_simulations/window_<rep>.trees
  - applies biallelic filter via HaplotypeMatrix
  - optionally thins to at most `max_sites` biallelic sites
  - writes:
      IM_model_simulations/window_<rep>.filtered.trees
      IM_model_simulations/split_mig.filtered.<rep>.vcf.gz
      IM_model_simulations/kept_sites_rep<rep>.txt
      IM_model_simulations/kept_sites_rep<rep>.sha1
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import im_model_core as core  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build filtered/THINNED site sets for one replicate.")
    ap.add_argument("--sim-dir", required=True, help="Simulation directory (e.g. IM_model_simulations)")
    ap.add_argument("--rep", type=int, required=True, help="Replicate index")
    ap.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Cap on number of biallelic sites per replicate (after thinning). "
             "If omitted, use all biallelic sites.",
    )
    ap.add_argument(
    "--filter-mode",
    choices=["biallelic", "none"],
    default="biallelic",
    help="How to build the site set: "
         "'biallelic' (apply biallelic filter + optional thinning) or "
         "'none' (no filtering/thinning; use all simulated sites).",
    )
    args = ap.parse_args()

    out = core._build_filtered_ts_and_cache(
        sim_dir=args.sim_dir,
        rep=args.rep,
        max_sites=args.max_sites,
        rng_seed=None,  # uses deterministic 1337 + rep inside if None
        filter_mode=args.filter_mode,

    )

    # Just a friendly log; Snakemake checks files via outputs.
    print(
        f"[filter_sites.py] rep={args.rep} "
        f"ts_filt={out['ts_filt_path']} "
        f"vcf={out['vcf_filt_gz']} "
        f"S_filt={out['S_filt']} sha1={out['sha1']}"
    )


if __name__ == "__main__":
    main()
