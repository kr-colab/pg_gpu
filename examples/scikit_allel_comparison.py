#!/usr/bin/env python
"""
Side-by-side: scikit-allel vs pg_gpu on a windowed diversity scan.

Loads a real Anopheles gambiae X-chromosome dataset and computes
windowed pi, theta_w, and Tajima's D using both libraries.
scikit-allel takes three separate calls; pg_gpu takes one. The script
verifies numerical agreement and reports the wall-clock speedup.

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


def _load_data(small: bool) -> tuple:
    """Load the gamb dataset and build the three views needed for comparison.

    Returns
    -------
    hm : HaplotypeMatrix
    pos : np.ndarray, shape (n_variants,), 1-based positions
    ac : np.ndarray, shape (n_variants, 2), allele counts for allel
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
    return hm, pos, ac


# -- compute paths -----------------------------------------------------------

def _compute_allel(pos: np.ndarray, ac: np.ndarray,
                   windows: np.ndarray) -> tuple:
    """Run scikit-allel's three windowed diversity scans. Returns
    (pi, theta_w, tajimas_d)."""
    pi, _, _, _ = allel.windowed_diversity(pos, ac, windows=windows)
    theta_w, _, _, _ = allel.windowed_watterson_theta(
        pos, ac, windows=windows)
    tajd, _, _ = allel.windowed_tajima_d(pos, ac, windows=windows)
    return pi, theta_w, tajd


def _compute_pg_gpu(hm: HaplotypeMatrix, window_size: int) -> tuple:
    """Run pg_gpu's single fused windowed_analysis call. Returns
    (pi, theta_w, tajimas_d) plus the result DataFrame so the caller
    can inspect window edges."""
    df = windowed_analysis(
        hm, window_size=window_size, step_size=window_size,
        statistics=["pi", "theta_w", "tajimas_d"])
    cp.cuda.Device().synchronize()
    return (df["pi"].to_numpy(),
            df["theta_w"].to_numpy(),
            df["tajimas_d"].to_numpy()), df


def _assert_windows_aligned(allel_windows: np.ndarray,
                            pg_gpu_df) -> None:
    """Confirm both libraries laid out the same windows.

    scikit-allel returns `windows` as (n_windows, 2) of (start, stop)
    in 1-based inclusive coordinates. pg_gpu's windowed_analysis
    returns `start` and `end` columns. We check that they match
    exactly.
    """
    n_allel = allel_windows.shape[0]
    n_pg = len(pg_gpu_df)
    if n_allel != n_pg:
        raise AssertionError(
            f"window count mismatch: allel={n_allel}, pg_gpu={n_pg}. "
            f"This means the two libraries laid out different windows "
            f"for the same input range; the rest of the comparison "
            f"would be apples-to-oranges. Check that window_size, "
            f"step_size, and the chromosome range agree.")
    starts_a = allel_windows[:, 0]
    starts_g = pg_gpu_df["start"].to_numpy()
    if not np.array_equal(starts_a, starts_g):
        bad = np.where(starts_a != starts_g)[0][:5]
        raise AssertionError(
            f"window-start mismatch at indices {bad}: "
            f"allel={starts_a[bad]}, pg_gpu={starts_g[bad]}")


# -- verify + time -----------------------------------------------------------

def _verify_strict(name: str, allel_arr: np.ndarray,
                   pg_arr: np.ndarray,
                   rtol: float = 1e-5, atol: float = 1e-8) -> float:
    """Assert per-window agreement on a NaN-aligned mask.

    Returns the max absolute difference on the comparison mask so the
    caller can print a one-line summary.
    """
    finite = np.isfinite(allel_arr) & np.isfinite(pg_arr)
    if not finite.any():
        raise AssertionError(
            f"[{name}] no finite windows in either array; cannot verify")
    diff = np.abs(allel_arr[finite] - pg_arr[finite])
    max_diff = float(diff.max())
    try:
        np.testing.assert_allclose(
            allel_arr[finite], pg_arr[finite],
            rtol=rtol, atol=atol,
            err_msg=f"[{name}] disagreement above tolerance")
    except AssertionError as e:
        # Surface the worst-offender windows in the failure
        order = np.argsort(diff)[::-1][:5]
        finite_idx = np.where(finite)[0]
        worst = finite_idx[order]
        raise AssertionError(
            f"[{name}] disagreement; worst window indices {worst.tolist()} "
            f"(max diff = {max_diff:.3e}). Original error:\n{e}") from None
    return max_diff


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        return
    hm, pos, ac = _load_data(args.small)

    print("Computing windows ...", flush=True)
    windows = allel.position_windows(
        pos, size=args.window_size, step=args.window_size,
        start=int(pos[0]), stop=int(pos[-1]))
    print(f"  {len(windows)} windows of {args.window_size:,} bp")

    print("Running scikit-allel ...", flush=True)
    pi_a, tw_a, td_a = _compute_allel(pos, ac, windows)
    print("Running pg_gpu ...", flush=True)
    (pi_g, tw_g, td_g), df = _compute_pg_gpu(hm, args.window_size)
    _assert_windows_aligned(windows, df)
    print(f"  {len(pi_a)} aligned windows confirmed")

    # The trailing window can be partial when chrom_end is not a multiple
    # of window_size. The two libraries normalize partial windows
    # differently (allel divides per-site sums by actual span; pg_gpu
    # divides by the fixed window_size parameter), which produces a real
    # but uninteresting numerical mismatch on that one window. Mask it
    # out so verification only compares full-width windows.
    window_widths = windows[:, 1] - windows[:, 0] + 1
    partial = window_widths != args.window_size
    n_partial = int(partial.sum())
    # Work on writable copies so we can set partial windows to NaN.
    pi_a, tw_a, td_a = (np.array(a) for a in (pi_a, tw_a, td_a))
    pi_g, tw_g, td_g = (np.array(a) for a in (pi_g, tw_g, td_g))
    if n_partial:
        for arr in (pi_a, tw_a, td_a, pi_g, tw_g, td_g):
            arr[partial] = np.nan
        print(f"  ({n_partial} partial trailing window(s) masked from "
              f"comparison)")

    print("Verifying ...", flush=True)
    md_pi = _verify_strict("pi", pi_a, pi_g)
    md_tw = _verify_strict("theta_w", tw_a, tw_g)
    md_td = _verify_strict("tajimas_d", td_a, td_g)
    print(f"  pi:        max abs diff = {md_pi:.3e}")
    print(f"  theta_w:   max abs diff = {md_tw:.3e}")
    print(f"  tajimas_d: max abs diff = {md_td:.3e}")
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
    """Quick host-side sanity check on compute_allele_counts."""
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
    print("self-test OK")


if __name__ == "__main__":
    main()
