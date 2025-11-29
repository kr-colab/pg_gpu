#!/usr/bin/env python3
import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import moments  # type: ignore
import tskit    # type: ignore

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import im_model_core as core  # type: ignore


def main():
    p = argparse.ArgumentParser("Compute traditional (moments) LD for one replicate")
    p.add_argument("--sim-dir", required=True, type=str)
    p.add_argument("--trad-ld-dir", required=True, type=str)
    p.add_argument("--rep", required=True, type=int)
    p.add_argument("--num-rbins", required=True, type=int)
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    out_dir = Path(args.trad_ld_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = args.rep

    ts_filt_path = sim_dir / f"window_{rep}.filtered.trees"
    vcf_filt_gz = sim_dir / f"split_mig.filtered.{rep}.vcf.gz"
    recmap_path = sim_dir / "flat_map.txt"

    # r_bins: same as in original script
    r_bins = np.concatenate(([0.0], np.logspace(-6, -3, args.num_rbins)))

    print(f"[TRAD] computing LD for rep={rep}")

    # Build per-rep samples file from FILTERED VCF header
    pop_file = core._replicate_samples_file(
        sim_dir=str(sim_dir),
        rep=rep,
        vcf_path=str(vcf_filt_gz),
    )

    # Sanity: all samples in pop file should be in VCF header
    header_set = set(core._vcf_sample_names(str(vcf_filt_gz)))
    with open(pop_file) as f:
        next(f)
        names = [ln.split()[0] for ln in f if ln.strip()]
    missing = [nm for nm in names if nm not in header_set]
    if missing:
        raise RuntimeError(
            f"Pop-file sample(s) missing in filtered VCF header for rep={rep}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Compute LD stats using moments
    ld = moments.LD.Parsing.compute_ld_statistics(
        str(vcf_filt_gz),
        rec_map_file=str(recmap_path),
        pop_file=pop_file,
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )

    # Attach sitecheck information from filtered ts
    ts_filt = tskit.load(str(ts_filt_path))
    positions = np.asarray(ts_filt.tables.sites.position)
    sha1 = core._hash_positions(positions)
    ld["_sitecheck"] = dict(S_filt=int(ts_filt.num_sites), sha1=sha1)

    out_pkl = out_dir / f"LD_stats_window_{rep}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(ld, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[TRAD] wrote {out_pkl} | S_filt={ts_filt.num_sites} sha1={sha1}")


if __name__ == "__main__":
    main()
