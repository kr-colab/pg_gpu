"""Smoke tests that run example scripts end-to-end on small fixtures."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scikit_allel_comparison_runs_small():
    """examples/scikit_allel_comparison.py --small --no-plot must exit 0
    and verify the three diversity statistics + the LD-decay curve on
    the 4 Mb gamb subset.

    Uses a smaller --ld-snps to keep the smoke test under a minute on
    CI-class hardware (the default 10_000 takes ~20s on the
    scikit-allel side, mostly in rogers_huff_r). Disables the MAC
    filter so the small block keeps enough variants for parity to be
    meaningful."""
    script = REPO_ROOT / "examples" / "scikit_allel_comparison.py"
    result = subprocess.run(
        [sys.executable, str(script), "--small", "--no-plot",
         "--ld-snps", "2000", "--ld-mac-min", "0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n")
    # Sanity: both verification blocks must have printed.
    assert "tajimas_d: max abs diff" in result.stdout, result.stdout
    assert "median r" in result.stdout, result.stdout
