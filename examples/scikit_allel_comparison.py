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


def main() -> None:
    raise NotImplementedError("main not yet wired up")


if __name__ == "__main__":
    main()
