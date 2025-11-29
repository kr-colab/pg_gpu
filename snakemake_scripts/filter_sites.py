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
    p = argparse.ArgumentParser("Build filtered ts + VCF + kept sites for one replicate")
    p.add_argument("--sim-dir", required=True, type=str)
    p.add_argument("--rep", required=True, type=int)
    args = p.parse_args()

    built = core._build_filtered_ts_and_cache(args.sim_dir, args.rep)

    # Just print for sanity; Snakemake checks existence of files
    print(
        f"[FILTER] rep={args.rep} "
        f"ts_filt={built['ts_filt_path']} vcf={built['vcf_filt_gz']} "
        f"S_filt={built['S_filt']} sha1={built['sha1']}"
    )


if __name__ == "__main__":
    main()
