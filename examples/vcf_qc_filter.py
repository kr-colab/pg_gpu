#!/usr/bin/env python
"""
Quality-aware filtering on a real VCZ: load FORMAT / INFO arrays, mask,
write a clean VCZ.

Demonstrates the ``fields=`` / ``filter()`` / ``to_zarr`` workflow added
to pg_gpu against the empirical Anopheles X-chromosome VCZ that ships
under ``examples/data/``. The store carries every field bio2zarr
preserved during VCF encoding -- ``variant_AC`` / ``variant_AF`` /
``variant_AAProb`` / ``call_PQ`` / ... -- so a real "tune thresholds
read-side" demo runs without external data.

The script:

1. Opens a 4 Mb window of ``examples/data/gamb.X.phased.n100.vcz`` with
   ``fields=['AC', 'AAProb', 'PQ']``. ``AC`` is per-variant
   ``(n_var,)``, ``AAProb`` is per-variant ``(n_var, 1)`` (bio2zarr
   shape for INFO with ``Number=A``), and ``PQ`` is per-genotype
   ``(n_var, n_samples)``. pg_gpu auto-resolves which kind each is.
2. Summarizes each loaded field so the chosen thresholds are visible.
3. Builds a real variant filter -- ``AC >= 4`` to drop singletons and
   doubletons, ``AAProb >= 0.9`` to keep variants with a confident
   ancestral-allele call.
4. Builds a per-genotype mask placeholder (``PQ >= 0``). The gamb
   fixture carries ``-1`` everywhere in ``call_PQ`` because the source
   VCF didn't populate phasing quality, so this mask is True for every
   call. On a VCZ derived from a VCF that DID carry ``FMT/GQ`` or
   ``FMT/DP`` the same line would read ``(hm.fields['GQ'] >= 20) &
   (hm.fields['DP'] >= 10)``.
5. Runs ``hm.filter`` and writes the survivor to a fresh VCZ. The
   surviving ``variant_*`` / ``call_*`` arrays are preserved in the
   clean store and reload with the same ``fields=`` set.

Usage
-----
    pixi run python examples/vcf_qc_filter.py
    pixi run python examples/vcf_qc_filter.py --min-ac 8 --min-aa-prob 0.95
    pixi run python examples/vcf_qc_filter.py --region X:8000000-12000000
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from pg_gpu import HaplotypeMatrix


DEFAULT_SRC = "examples/data/gamb.X.phased.n100.vcz"


def _summarize(name, arr):
    """Median + IQR; works on 1D, 2D INFO (n_var, n_alt), or FORMAT."""
    flat = arr.ravel()
    q = np.percentile(flat, [5, 25, 50, 75, 95])
    print(f"  {name:>8}: shape={arr.shape}, "
          f"5/25/50/75/95% = {q[0]:.3f} / {q[1]:.3f} / "
          f"{q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--src", type=str, default=DEFAULT_SRC,
                   help=f"Source VCZ (default: {DEFAULT_SRC}). "
                        "Any VCZ from bio2zarr will work.")
    p.add_argument("--region", type=str, default="X:1000000-5000000",
                   help="chrom:start-end window of the source to load. "
                        "Keep it under ~5 Mb to fit comfortably in memory; "
                        "the full chromosome is 25 Mb / 5.3M variants.")
    p.add_argument("--min-ac", type=int, default=4,
                   help="Drop variants whose allele count is below this. "
                        "AC=1 are singletons, AC=2 doubletons; default 4 "
                        "trims everything <0.02 MAF.")
    p.add_argument("--min-aa-prob", type=float, default=0.9,
                   help="Drop variants where the ancestral-allele "
                        "probability is below this (only the most "
                        "confident polarizations survive). Default 0.9.")
    p.add_argument("--out", type=str, default="/tmp/gamb.X.clean.vcz",
                   help="Where to write the filtered VCZ.")
    args = p.parse_args()

    if not Path(args.src).exists():
        raise SystemExit(
            f"Source VCZ not found: {args.src}. The default ships with "
            f"the repo under examples/data/; if you cloned without LFS "
            f"you may need to fetch the data fixtures.")

    print(f"Opening {args.src}")
    print(f"  region: {args.region}")
    print("  fields: ['AC', 'AAProb', 'PQ']")
    hm = HaplotypeMatrix.from_zarr(
        args.src, region=args.region,
        fields=["AC", "AAProb", "PQ"],
        streaming="never",
    )
    n_var = int(hm.haplotypes.shape[1])
    n_sam = int(hm.haplotypes.shape[0] // 2)
    print(f"  loaded {n_var:,} variants x {n_sam} diploids")
    print()
    print("Field summaries (5 / 25 / 50 / 75 / 95 percentiles):")
    _summarize("AC", hm.fields["AC"])
    _summarize("AAProb", hm.fields["AAProb"])
    _summarize("PQ", hm.fields["PQ"])
    print()
    print(f"Filtering: variants kept where AC >= {args.min_ac} and "
          f"AAProb >= {args.min_aa_prob}")
    # variant_AAProb arrives as (n_var, 1); .ravel() peels off the
    # trailing axis so the mask is the (n_var,) bool that filter()
    # expects.
    aa_prob = hm.fields["AAProb"].ravel()
    variants = (hm.fields["AC"] >= args.min_ac) & (aa_prob >= args.min_aa_prob)
    # In this store call_PQ wasn't populated by the source VCF (all -1);
    # the per-genotype mask is universally True, so it has no effect
    # here. On a store derived from a VCF that DID carry FMT/GQ + FMT/DP
    # the natural line would be:
    #     genotypes = (hm.fields['GQ'] >= 20) & (hm.fields['DP'] >= 10)
    genotypes = hm.fields["PQ"] >= -1
    print(f"  variants kept: {int(variants.sum()):,} / {n_var:,} "
          f"({100.0 * float(variants.mean()):.1f} %)")

    hm_clean = hm.filter(
        variants=variants,
        genotypes=genotypes,
        drop_all_missing=True,
    )
    n_clean = int(hm_clean.haplotypes.shape[1])
    print(f"  after filter: {n_clean:,} variants")
    print()
    print(f"Writing clean VCZ to {args.out}")
    if Path(args.out).exists():
        shutil.rmtree(args.out)
    hm_clean.to_zarr(args.out, format="vcz", contig_name="X")

    # Reload and assert the round-trip is byte-exact.
    rt = HaplotypeMatrix.from_zarr(
        args.out, fields=["AC", "AAProb", "PQ"], streaming="never",
    )
    np.testing.assert_array_equal(rt.fields["AC"], hm_clean.fields["AC"])
    np.testing.assert_array_equal(rt.fields["AAProb"],
                                   hm_clean.fields["AAProb"])
    np.testing.assert_array_equal(rt.fields["PQ"], hm_clean.fields["PQ"])
    print("Round-trip OK -- reloaded fields match the filtered matrix.")


if __name__ == "__main__":
    main()
