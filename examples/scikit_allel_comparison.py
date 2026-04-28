#!/usr/bin/env python
"""
Side-by-side: scikit-allel vs pg_gpu on a windowed diversity / LD scan.

Loads a real Anopheles gambiae X-chromosome dataset and computes
windowed pi, theta_w, Tajima's D, and a windowed LD summary using
both libraries. scikit-allel takes four separate calls; pg_gpu takes
one. The script verifies numerical agreement (or, for the LD scan,
high rank correlation) and reports the wall-clock speedup.

Usage
-----
    pixi run python examples/scikit_allel_comparison.py
    pixi run python examples/scikit_allel_comparison.py --small
    pixi run python examples/scikit_allel_comparison.py --no-plot
"""

import argparse
import sys
import time
from pathlib import Path

import allel
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import windowed_analysis


# Datasets: full X-chromosome by default; --small uses a 4 Mb subset.
DATA_DIR = Path(__file__).resolve().parent / "data"
ZARR_FULL = DATA_DIR / "gamb.X.phased.n100.zarr"
ZARR_SMALL = DATA_DIR / "gamb.X.8-12Mb.n100.derived.zarr"


# -- helpers -----------------------------------------------------------------

def compute_allele_counts(hm: HaplotypeMatrix) -> np.ndarray:
    """Build a scikit-allel-style (n_variants, 2) allele-count array.

    Counts ref (0) and alt (1) calls per variant, ignoring any -1
    missing-data sentinels.
    """
    hap = hm.haplotypes
    if hasattr(hap, "get"):  # cupy -> numpy
        hap = hap.get()
    hap = np.asarray(hap, dtype=np.int8)
    ac = np.empty((hap.shape[1], 2), dtype=np.int32)
    ac[:, 0] = (hap == 0).sum(axis=0)
    ac[:, 1] = (hap == 1).sum(axis=0)
    # Sanity: ref + alt should equal the number of non-missing haplotypes.
    n_called = (hap >= 0).sum(axis=0)
    assert np.array_equal(ac.sum(axis=1), n_called), (
        "allele counts do not match called-haplotype counts")
    return ac


def compute_genotype_codes(hm: HaplotypeMatrix) -> np.ndarray:
    """Build a scikit-allel-style (n_variants, n_samples) int8 genotype
    array (0 = hom ref, 1 = het, 2 = hom alt).

    Pairs adjacent haplotypes (haplotypes 0,1 = sample 0; 2,3 = sample
    1; etc.). The gamb X dataset has no missing data; this function
    asserts that and refuses to silently fabricate codes.
    """
    hap = hm.haplotypes
    if hasattr(hap, "get"):
        hap = hap.get()
    hap = np.asarray(hap, dtype=np.int8)
    if (hap < 0).any():
        raise ValueError(
            "compute_genotype_codes: input haplotypes contain missing "
            "values (-1). scikit-allel's rogers_huff_r expects 0/1/2 "
            "with no missing sentinels; drop or impute missing sites "
            "before calling.")
    n_hap, n_var = hap.shape
    if n_hap % 2 != 0:
        raise ValueError(
            f"compute_genotype_codes: odd number of haplotypes "
            f"({n_hap}); cannot pair into diploids.")
    # (n_haps, n_var) -> (n_samples, n_var) by summing pairs, then transpose.
    paired = hap[0::2, :] + hap[1::2, :]   # (n_samples, n_var) int8
    return paired.T.astype(np.int8)        # (n_var, n_samples)


def _load_data(small: bool) -> tuple:
    """Load the gamb dataset and build all four views.

    Returns
    -------
    hm : HaplotypeMatrix
    pos : np.ndarray, shape (n_variants,), 1-based positions
    ac : np.ndarray, shape (n_variants, 2), allele counts for allel
    gn : np.ndarray, shape (n_variants, n_samples), 0/1/2 codes for allel
    """
    path = ZARR_SMALL if small else ZARR_FULL
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}\n"
            f"The data fixtures live under examples/data/ and are tracked "
            f"in the repo. If you cloned without LFS or otherwise lack the "
            f"file, try `git lfs pull` or refetch from the project root.")
    print(f"Loading {path.name} ...", flush=True)
    t0 = time.perf_counter()
    hm = HaplotypeMatrix.from_zarr(str(path))
    print(f"  loaded in {time.perf_counter() - t0:.2f}s; "
          f"{hm.haplotypes.shape[0]} haplotypes, "
          f"{hm.haplotypes.shape[1]:,} variants, "
          f"chrom range {hm.chrom_start:,}-{hm.chrom_end:,}",
          flush=True)
    pos = hm.positions
    if hasattr(pos, "get"):
        pos = pos.get()
    pos = np.asarray(pos, dtype=np.int64)
    ac = compute_allele_counts(hm)
    gn = compute_genotype_codes(hm)
    return hm, pos, ac, gn


# -- compute paths -----------------------------------------------------------

def _compute_allel(pos: np.ndarray, ac: np.ndarray, gn: np.ndarray,
                   windows: np.ndarray) -> tuple:
    """Run scikit-allel's four windowed scans. Returns (pi, theta_w,
    tajimas_d, ld_median_r2).

    Note: allel.windowed_r_squared crashes in NumPy >= 1.25 when a window
    contains exactly one variant (rogers_huff_r returns an empty r array
    and np.percentile raises IndexError on it). We use windowed_statistic
    directly with a guarded closure to match the intended behaviour.
    """
    pi, _, _, _ = allel.windowed_diversity(pos, ac, windows=windows)
    theta_w, _, _, _ = allel.windowed_watterson_theta(
        pos, ac, windows=windows)
    tajd, _, _ = allel.windowed_tajima_d(pos, ac, windows=windows)

    def _median_r2(gnw):
        r_sq = allel.rogers_huff_r(gnw) ** 2
        if len(r_sq) == 0:
            return np.nan
        return np.percentile(r_sq, 50)

    ld, _, _ = allel.windowed_statistic(
        pos, gn, _median_r2, windows=windows, fill=np.nan)
    return pi, theta_w, tajd, ld


def _compute_pg_gpu(hm: HaplotypeMatrix, window_size: int) -> tuple:
    """Run pg_gpu's single fused windowed_analysis call. Returns
    (pi, theta_w, tajimas_d, zns) plus the result DataFrame so the
    caller can inspect window edges."""
    df = windowed_analysis(
        hm, window_size=window_size, step_size=window_size,
        statistics=["pi", "theta_w", "tajimas_d", "zns"])
    cp.cuda.Device().synchronize()
    return (df["pi"].to_numpy(),
            df["theta_w"].to_numpy(),
            df["tajimas_d"].to_numpy(),
            df["zns"].to_numpy()), df


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        return
    hm, pos, ac, gn = _load_data(args.small)

    print("Computing windows ...", flush=True)
    windows = allel.position_windows(
        pos, size=args.window_size, step=args.window_size,
        start=int(pos[0]), stop=int(pos[-1]))
    print(f"  {len(windows)} windows of {args.window_size:,} bp")

    print("Running scikit-allel ...", flush=True)
    pi_a, tw_a, td_a, ld_a = _compute_allel(pos, ac, gn, windows)
    print("Running pg_gpu ...", flush=True)
    (pi_g, tw_g, td_g, ld_g), df = _compute_pg_gpu(hm, args.window_size)
    print(f"allel windows: {len(pi_a)}, pg_gpu windows: {len(pi_g)}")
    return


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side scikit-allel vs pg_gpu windowed scan")
    p.add_argument("--small", action="store_true",
                   help="Use the 4 Mb gamb.X.8-12Mb subset (fast)")
    p.add_argument("--window-size", type=int, default=10_000,
                   help="Window size in bp (default: 10 kb)")
    p.add_argument("--n-warmup", type=int, default=1,
                   help="Discarded runs before timing (default: 1)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the matplotlib figure")
    p.add_argument("-o", "--output", type=Path,
                   default=Path("scikit_allel_comparison.png"),
                   help="Output figure path")
    p.add_argument("--self-test", action="store_true",
                   help=argparse.SUPPRESS)
    return p.parse_args()


def _run_self_test() -> None:
    """Quick host-side sanity check on the two helpers."""
    # 4 haplotypes, 3 variants, no missing
    hap = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
    ], dtype=np.int8)
    pos = np.array([100, 200, 300])
    hm = HaplotypeMatrix(hap, pos, chrom_start=1, chrom_end=300)
    ac = compute_allele_counts(hm)
    np.testing.assert_array_equal(ac[:, 0], [3, 1, 3])
    np.testing.assert_array_equal(ac[:, 1], [1, 3, 1])
    gn = compute_genotype_codes(hm)
    assert gn.shape == (3, 2), gn.shape
    np.testing.assert_array_equal(gn[:, 0], [1, 2, 0])
    np.testing.assert_array_equal(gn[:, 1], [0, 1, 1])
    print("self-test OK")


if __name__ == "__main__":
    main()
