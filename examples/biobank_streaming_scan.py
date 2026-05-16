"""End-to-end biobank-scale scan of a single chromosome from a VCZ store.

Streams the chromosome through the GPU one genomic chunk at a time
and accumulates, in a single pass over ``stream.iter_gpu_chunks()``:

* Per-window diversity / divergence at three scales (10 kb, 100 kb,
  1 Mb).
* Marginal SFS per population.
* Joint SFS projected from the full panel to a small display grid
  via per-variant hypergeometric sampling -- every variant from
  every haplotype contributes, no subsampling.
* Garud's H per population (uses a 1,000-haplotype subsample because
  the fused Garud kernel caps near 1024 haplotypes).

Then -- after the streaming walk -- materializes a 1 Mb sub-region
with a 5,000-haplotype subsample so the pairwise-r^2 heatmap can
see every variant simultaneously.

Output:

    {OUT_DIR}/chr{CHROM}_pi_100kb.csv
    {OUT_DIR}/chr{CHROM}_sfs_{pop}.csv
    {OUT_DIR}/chr{CHROM}_joint_sfs.npy
    {OUT_DIR}/chr{CHROM}_garud.csv
    {OUT_DIR}/chr{CHROM}_r2_heatmap.npy

A larger version of this scan (with multiscale plots, moments-LD
decay across probe regions, and figure output) lives at
``pg_gpu-paper-analysis/06_simulated_genome_scan/scripts/genome_scan_ooa.py``.

Run inside the pg_gpu pixi environment with a free GPU:

    CUDA_VISIBLE_DEVICES=0 pixi run python examples/biobank_streaming_scan.py \\
        --vcz path/to/chr15.vcz --out-dir scan_out
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pg_gpu import HaplotypeMatrix, sfs, windowed_analysis
# Pop names are whatever the store / pop_file declares.
POPS = ("AFR", "EUR")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vcz", required=True, help="path to a chr*.vcz store")
    p.add_argument("--pop-file", default=None,
                   help="optional path/dict/zarr-key for sample -> pop "
                        "assignments. Default: auto-load <vcz>.pops.tsv")
    p.add_argument("--out-dir", default="scan_out", help="output directory")
    p.add_argument("--chunk-bp", type=int, default=500_000,
                   help="streaming chunk size in bp (default 500 kb)")
    p.add_argument("--heatmap-region", default=None,
                   help="region 'start-end' bp to materialize for the "
                        "pairwise-r^2 heatmap (default: 1 Mb at chrom "
                        "midpoint)")
    p.add_argument("--heatmap-subsample", type=int, default=5_000,
                   help="haplotypes drawn from one population for the "
                        "pairwise-r^2 heatmap (default 5,000)")
    p.add_argument("--joint-sfs-target", type=int, default=200,
                   help="hypergeometric projection target for the joint "
                        "SFS, in haplotypes per population")
    p.add_argument("--garud-subsample", type=int, default=1_000,
                   help="haplotypes drawn per population for Garud's H "
                        "(kernel caps near 1024)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = HaplotypeMatrix.from_zarr(
        args.vcz, streaming="always",
        chunk_bp=args.chunk_bp,
        pop_file=args.pop_file,  # None -> autoload <vcz>.pops.tsv
    )
    chrom = stream.chrom
    chrom_len = stream.chrom_end - stream.chrom_start

    full_pop = {p: [int(i) for i in stream.sample_sets[p]] for p in POPS}
    n_haps = {p: len(full_pop[p]) for p in POPS}
    sub_g = {p: full_pop[p][:args.garud_subsample] for p in POPS}

    # Containers populated by the streaming walk.
    win_parts = {"10kb": [], "100kb": [], "1mb": []}
    garud_parts = []
    marginal = {p: None for p in POPS}
    joint = None

    t_scan = time.perf_counter()
    n_chunks = len(stream._chunks)
    for ci, (left, right, chunk) in enumerate(stream.iter_gpu_chunks()):
        t_chunk = time.perf_counter()

        # Garud uses a registered subsample per chunk; the rest of
        # the kernels see every haplotype in the chunk.
        chunk.sample_sets = {**full_pop,
                              **{f"{p}_g": sub_g[p] for p in POPS}}

        # Per-window diversity + divergence at three scales.
        for label, bp in (("10kb", 10_000), ("100kb", 100_000),
                          ("1mb", 1_000_000)):
            div = windowed_analysis(chunk, window_size=bp, step_size=bp,
                                     statistics=["pi", "theta_w",
                                                  "tajimas_d", "fst",
                                                  "dxy"],
                                     populations=list(POPS))
            if not div.empty:
                win_parts[label].append(div.reset_index(drop=True))

        # Garud's H per pop in 10 kb windows, on the small subsample.
        g_parts = [windowed_analysis(chunk, window_size=10_000,
                                      step_size=10_000,
                                      statistics=["garud_h12",
                                                   "haplotype_count"],
                                      populations=[f"{p}_g"])
                   for p in POPS]
        gdf = g_parts[0][["chrom", "start", "end", "center"]].copy()
        for i, p in enumerate(POPS):
            gdf[f"garud_h12_{p}"] = g_parts[i]["garud_h12"].values
        garud_parts.append(gdf.reset_index(drop=True))

        # Marginal SFS per pop (one bincount per chunk, summed at the
        # end).
        for p in POPS:
            s = np.asarray(sfs.sfs(chunk, population=p))
            marginal[p] = s if marginal[p] is None else marginal[p] + s

        # Joint SFS projected to the small display grid as we go.
        # The full (n_hap+1, n_hap+1) histogram would be ~80 GB at
        # 100k haps per pop and would not fit on the GPU.
        j = np.asarray(sfs.project_joint_sfs(
            chunk, pop1=POPS[0], pop2=POPS[1],
            target_n1=args.joint_sfs_target,
            target_n2=args.joint_sfs_target,
        ))
        joint = j if joint is None else joint + j

        print(f"  chunk {ci + 1}/{n_chunks} "
              f"[{left / 1e6:.1f}-{right / 1e6:.1f} Mb] in "
              f"{time.perf_counter() - t_chunk:.1f}s", flush=True)

    print(f"\nStreaming walk: {time.perf_counter() - t_scan:.1f}s total")

    # Concat per-scale window frames, dump tables / arrays.
    for label, parts in win_parts.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_csv(out_dir / f"chr{chrom}_{label}.csv", index=False)
    pd.concat(garud_parts, ignore_index=True).to_csv(
        out_dir / f"chr{chrom}_garud.csv", index=False)
    for p in POPS:
        np.savetxt(out_dir / f"chr{chrom}_sfs_{p}.csv",
                   marginal[p], delimiter=",", fmt="%d")
    np.save(out_dir / f"chr{chrom}_joint_sfs.npy", joint)

    # Pairwise-r^2 heatmap of a 1 Mb sub-region: needs every variant
    # simultaneously, so we materialize it eagerly with a haplotype
    # subsample from one pop.
    if args.heatmap_region:
        lo, hi = (int(x) for x in args.heatmap_region.split("-"))
    else:
        mid = stream.chrom_start + chrom_len // 2
        lo, hi = mid - 500_000, mid + 500_000
    sub = list(stream.sample_sets[POPS[0]][:args.heatmap_subsample])
    region_hm = stream.materialize(region=(lo, hi), sample_subset=sub)
    r2 = region_hm.pairwise_r2()
    np.save(out_dir / f"chr{chrom}_r2_heatmap.npy", np.asarray(r2))

    # One-line summary.
    summary = {
        "chromosome": chrom,
        "chromosome_length_bp": int(chrom_len),
        "n_variants": int(stream.num_variants),
        "haplotypes_per_pop": n_haps,
        "joint_sfs_target": args.joint_sfs_target,
        "garud_subsample": args.garud_subsample,
        "heatmap_region": [lo, hi],
        "heatmap_subsample": args.heatmap_subsample,
    }
    (out_dir / f"chr{chrom}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
