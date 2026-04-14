#!/usr/bin/env python
"""
Benchmark: pg_gpu.local_pca vs R lostruct::eigen_windows + pc_dist.

**Developer-only.** Requires a local R install with the ``lostruct`` and
``jsonlite`` packages. Not intended for end users; not referenced from
the user docs.

Sweeps dataset size (total variants) at fixed window size, and for each
size times:

    pg_gpu:   local_pca + pc_dist (single GPU call chain)
    lostruct: eigen_windows + pc_dist (via examples/local_pca_benchmark.R)

Produces a matplotlib PNG with total wall-clock time and speedup ratio
side by side.

Usage
-----
    pixi run python examples/local_pca_benchmark.py
    pixi run python examples/local_pca_benchmark.py --sizes 5000 20000 80000
    pixi run python examples/local_pca_benchmark.py --n-haps 200 --window 200
    pixi run python examples/local_pca_benchmark.py --skip-r        # GPU-only
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import local_pca, pc_dist


HERE = pathlib.Path(__file__).resolve().parent
R_SCRIPT = HERE / "local_pca_benchmark.R"


def _simulate(n_haps: int, n_variants: int, seed: int):
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, size=(n_haps, n_variants), dtype=np.int8)
    positions = np.arange(n_variants, dtype=np.int64)
    return hap, positions


def _time_pg_gpu(hm: HaplotypeMatrix, window_size: int, k: int):
    import cupy as cp

    # Warm up: first call pays one-time kernel compile / cuBLAS setup costs.
    _ = local_pca(hm, window_size=window_size, window_type='snp', k=k)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    result = local_pca(hm, window_size=window_size, window_type='snp', k=k)
    cp.cuda.Stream.null.synchronize()
    t_eigen = time.perf_counter() - t0

    t0 = time.perf_counter()
    dist = pc_dist(result, npc=k, normalize='L1')
    cp.cuda.Stream.null.synchronize()
    t_dist = time.perf_counter() - t0

    return {
        "n_windows": result.n_windows,
        "eigen_windows_s": t_eigen,
        "pc_dist_s": t_dist,
        "total_s": t_eigen + t_dist,
        "dist_shape": dist.shape,
    }


def _time_lostruct(hap: np.ndarray, window_size: int, k: int,
                   rscript: str) -> Optional[dict]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        in_path = tmp / "in.json"
        out_path = tmp / "out.json"
        with open(in_path, "w") as fh:
            json.dump({
                "hap": hap.astype(int).tolist(),
                "window_size": int(window_size),
                "k": int(k),
            }, fh)
        try:
            subprocess.run(
                [rscript, str(R_SCRIPT), str(in_path), str(out_path)],
                check=True, capture_output=True, timeout=1200, text=True)
        except FileNotFoundError:
            print(f"  skipped — {rscript} not on PATH", file=sys.stderr)
            return None
        except subprocess.CalledProcessError as e:
            print(f"  skipped — R error: {e.stderr[-300:]}", file=sys.stderr)
            return None
        with open(out_path) as fh:
            return json.load(fh)


def _plot(sizes, rows, out_path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pg_total = np.array([r["pg"]["total_s"] for r in rows])
    r_total = np.array([r["r"]["total_s"] if r["r"] is not None else np.nan
                        for r in rows])
    n_windows = np.array([r["pg"]["n_windows"] for r in rows])

    ax = axes[0]
    ax.loglog(sizes, pg_total, 'o-', color='steelblue', lw=1.8,
              markersize=7, label='pg_gpu (GPU)')
    if np.any(np.isfinite(r_total)):
        ax.loglog(sizes, r_total, 's--', color='darkorange', lw=1.8,
                  markersize=7, label='lostruct (R, CPU)')
    ax.set_xlabel("n_variants")
    ax.set_ylabel("wall-clock seconds")
    ax.set_title("local_pca + pc_dist total time")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')

    ax = axes[1]
    valid = np.isfinite(r_total) & (pg_total > 0)
    if valid.any():
        speedup = r_total[valid] / pg_total[valid]
        ax.semilogx(np.asarray(sizes)[valid], speedup, 'o-',
                    color='darkgreen', lw=1.8, markersize=7)
        ax.axhline(1, color='gray', lw=0.5, alpha=0.5)
        for x, y, n in zip(np.asarray(sizes)[valid], speedup,
                           n_windows[valid]):
            ax.annotate(f"{n}w", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color='dimgray')
        ax.set_xlabel("n_variants")
        ax.set_ylabel("speedup (lostruct / pg_gpu)")
        ax.set_title("pg_gpu speedup vs lostruct")
        ax.grid(True, which='both', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "R timings unavailable\n(--skip-r or R not on PATH)",
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sizes", type=int, nargs='+',
                   default=[5_000, 10_000, 25_000, 50_000, 100_000],
                   help="Total variants per benchmark run")
    p.add_argument("--n-haps", type=int, default=100,
                   help="Number of haplotypes (samples)")
    p.add_argument("--window", type=int, default=200,
                   help="SNP-count window size")
    p.add_argument("--k", type=int, default=2, help="Number of PCs per window")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rscript", default="Rscript",
                   help="Path to Rscript executable")
    p.add_argument("--skip-r", action="store_true",
                   help="Skip the lostruct (R) side; only time pg_gpu")
    p.add_argument("--out", type=pathlib.Path,
                   default=HERE / "local_pca_benchmark.png")
    args = p.parse_args()

    print(f"{'n_var':>8}  {'n_win':>6}  {'pg_eigen':>9}  {'pg_pcd':>8}  "
          f"{'pg_tot':>8}  {'R_eigen':>9}  {'R_pcd':>8}  {'R_tot':>8}  "
          f"{'speedup':>7}")
    print("-" * 92)

    rows = []
    for n_var in args.sizes:
        hap, pos = _simulate(args.n_haps, n_var, args.seed)
        hm = HaplotypeMatrix(hap, pos, 0, n_var)

        pg = _time_pg_gpu(hm, args.window, args.k)
        r = None if args.skip_r else _time_lostruct(
            hap, args.window, args.k, args.rscript)
        rows.append({"pg": pg, "r": r})

        if r is None:
            print(f"{n_var:>8}  {pg['n_windows']:>6}  "
                  f"{pg['eigen_windows_s']:>9.3f}  {pg['pc_dist_s']:>8.3f}  "
                  f"{pg['total_s']:>8.3f}  {'skip':>9}  {'skip':>8}  "
                  f"{'skip':>8}  {'--':>7}")
        else:
            speedup = r['total_s'] / pg['total_s']
            print(f"{n_var:>8}  {pg['n_windows']:>6}  "
                  f"{pg['eigen_windows_s']:>9.3f}  {pg['pc_dist_s']:>8.3f}  "
                  f"{pg['total_s']:>8.3f}  {r['eigen_windows_s']:>9.3f}  "
                  f"{r['pc_dist_s']:>8.3f}  {r['total_s']:>8.3f}  "
                  f"{speedup:>6.1f}x")

    _plot(args.sizes, rows, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
