#!/usr/bin/env python
"""
Load VCF/VCZ FORMAT/INFO quality metrics, filter on them, write a clean VCZ.

Walks through the four pieces of the ``fields=`` / ``filter()`` /
``to_zarr`` workflow that pg_gpu_paper#108 (the GQ / DP / MQ feature)
added:

1. Build a VCZ store from a small msprime simulation and inject
   synthetic ``variant_MQ`` and ``call_GQ`` / ``call_DP`` arrays. In a
   real workflow these come straight from bio2zarr converting a VCF
   that already has ``FMT/GQ``, ``FMT/DP``, and ``INFO/MQ``; the
   injection here just keeps the example fully self-contained.
2. Reopen the store with ``fields=['MQ', 'GQ', 'DP']`` so ``hm.fields``
   is populated.
3. Summarize each field with one ``np.percentile`` call -- the
   matrices come back as plain numpy arrays so any matplotlib /
   seaborn snippet works.
4. Build per-variant and per-genotype boolean masks from those arrays
   and feed them to ``hm.filter`` to get a clean matrix.
5. Save the clean matrix back out as a VCZ. The output round-trips the
   surviving QC arrays.

Usage
-----
    pixi run python examples/vcf_qc_filter.py
    pixi run python examples/vcf_qc_filter.py --seed 7 --min-gq 25
    pixi run python examples/vcf_qc_filter.py --out /tmp/clean.vcz
"""

import argparse
import shutil
from pathlib import Path

import msprime
import numpy as np
import zarr

from pg_gpu import HaplotypeMatrix


def _simulate(n_samples: int, seq_len: int, seed: int) -> HaplotypeMatrix:
    """Small msprime fixture so the script needs no external data."""
    ts = msprime.sim_ancestry(
        samples=n_samples,
        sequence_length=seq_len,
        recombination_rate=1e-7,
        random_seed=seed,
        ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


def _write_vcz_with_synthetic_qc(hm: HaplotypeMatrix, path: str, seed: int):
    """Write a VCZ store and stamp in synthetic MQ / GQ / DP arrays.

    The MQ values are correlated with variant position so the variant
    filter has something interesting to remove; GQ and DP are random
    so the per-genotype filter masks a realistic fraction of calls.
    """
    if Path(path).exists():
        shutil.rmtree(path)
    hm.to_zarr(path, format="vcz", contig_name="chr1")
    store = zarr.open_group(path, mode="r+")
    rng = np.random.default_rng(seed)
    n_var = int(hm.haplotypes.shape[1])
    n_sam = int(hm.haplotypes.shape[0] // 2)
    # MQ trends from low (left) to high (right) plus a touch of noise
    # so a threshold filter has a clear effect.
    mq = np.linspace(20.0, 60.0, n_var, dtype=np.float32)
    mq += rng.normal(0.0, 5.0, size=n_var).astype(np.float32)
    gq = rng.integers(0, 99, size=(n_var, n_sam), dtype=np.int16)
    dp = rng.integers(1, 40, size=(n_var, n_sam), dtype=np.int16)
    store.create_array("variant_MQ", data=mq)
    store.create_array("call_GQ", data=gq)
    store.create_array("call_DP", data=dp)


def _summarize_field(name: str, arr: np.ndarray):
    """Print median + IQR for a field. Flat for INFO, flattened for FORMAT."""
    values = arr.ravel()
    q = np.percentile(values, [5, 25, 50, 75, 95])
    print(f"  {name:>4}: shape={arr.shape}, "
          f"5/25/50/75/95% = {q[0]:.2f} / {q[1]:.2f} / "
          f"{q[2]:.2f} / {q[3]:.2f} / {q[4]:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=30,
                   help="Number of diploid individuals (default: 30)")
    p.add_argument("--seq-len", type=int, default=1_000_000,
                   help="Simulated chromosome length in bp (default: 1 Mb)")
    p.add_argument("--min-mq", type=float, default=35.0,
                   help="Variant-level MQ threshold (default: 35.0)")
    p.add_argument("--min-gq", type=int, default=20,
                   help="Per-genotype GQ threshold (default: 20)")
    p.add_argument("--min-dp", type=int, default=10,
                   help="Per-genotype DP threshold (default: 10)")
    p.add_argument("--src", type=str, default="/tmp/vcf_qc_filter.src.vcz",
                   help="Where to write the source (pre-filter) VCZ.")
    p.add_argument("--out", type=str, default="/tmp/vcf_qc_filter.clean.vcz",
                   help="Where to write the filtered (clean) VCZ.")
    args = p.parse_args()

    print(f"Simulating ({args.n_samples} diploids, {args.seq_len:,} bp) ...")
    hm_src = _simulate(args.n_samples, args.seq_len, args.seed)
    n_var = int(hm_src.haplotypes.shape[1])
    print(f"  {n_var:,} segregating sites")

    print(f"Writing VCZ with synthetic MQ / GQ / DP to {args.src} ...")
    _write_vcz_with_synthetic_qc(hm_src, args.src, args.seed)

    print("Reopening with fields=['MQ', 'GQ', 'DP'] ...")
    hm = HaplotypeMatrix.from_zarr(
        args.src, fields=["MQ", "GQ", "DP"], streaming="never",
    )
    print(f"  hm.fields keys: {sorted(hm.fields)}")
    print("Field summaries (5 / 25 / 50 / 75 / 95 percentiles):")
    for name in ("MQ", "GQ", "DP"):
        _summarize_field(name, hm.fields[name])

    print("Building masks ...")
    variants = hm.fields["MQ"] >= args.min_mq
    genotypes = (hm.fields["GQ"] >= args.min_gq) & (hm.fields["DP"] >= args.min_dp)
    n_var_keep = int(variants.sum())
    n_gt_drop = int((~genotypes).sum())
    n_gt_total = genotypes.size
    print(f"  variants kept (MQ >= {args.min_mq}): "
          f"{n_var_keep:,} / {n_var:,} "
          f"({100.0 * n_var_keep / n_var:.1f} %)")
    print(f"  genotypes masked (GQ >= {args.min_gq} & DP >= {args.min_dp}): "
          f"{n_gt_drop:,} / {n_gt_total:,} "
          f"({100.0 * n_gt_drop / n_gt_total:.1f} %)")

    print("Filtering ...")
    hm_clean = hm.filter(
        variants=variants,
        genotypes=genotypes,
        drop_all_missing=True,
    )
    n_clean = int(hm_clean.haplotypes.shape[1])
    print(f"  {n_clean:,} variants survive both filters "
          f"(drop_all_missing rolled in)")

    print(f"Writing clean VCZ to {args.out} ...")
    if Path(args.out).exists():
        shutil.rmtree(args.out)
    hm_clean.to_zarr(args.out, format="vcz", contig_name="chr1")

    # Reload to confirm the round-trip; the surviving QC arrays
    # are present on the new matrix, byte-equal to the in-memory
    # filtered version.
    rt = HaplotypeMatrix.from_zarr(
        args.out, fields=["MQ", "GQ", "DP"], streaming="never",
    )
    np.testing.assert_array_equal(rt.fields["MQ"], hm_clean.fields["MQ"])
    np.testing.assert_array_equal(rt.fields["GQ"], hm_clean.fields["GQ"])
    np.testing.assert_array_equal(rt.fields["DP"], hm_clean.fields["DP"])
    print("Round-trip OK -- reloaded fields match the filtered matrix.")


if __name__ == "__main__":
    main()
