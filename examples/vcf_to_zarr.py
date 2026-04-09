#!/usr/bin/env python
"""
Convert VCF to VCZ-format zarr using pg_gpu / bio2zarr.

Usage
-----
    python vcf_to_zarr.py input.vcf.gz output.zarr
    python vcf_to_zarr.py input.vcf.gz output.zarr --workers 4
"""

import argparse

from pg_gpu import HaplotypeMatrix


def main():
    p = argparse.ArgumentParser(description="Convert VCF to VCZ zarr store")
    p.add_argument("vcf", help="path to bgzipped, indexed VCF (.vcf.gz) or BCF")
    p.add_argument("zarr", help="output zarr store path")
    p.add_argument("--workers", type=int, default=None,
                   help="number of worker processes (default: all CPUs)")
    args = p.parse_args()

    HaplotypeMatrix.vcf_to_zarr(args.vcf, args.zarr,
                                worker_processes=args.workers)
    print(f"Done: {args.zarr}")


if __name__ == "__main__":
    main()
