"""Smoke tests that run example scripts end-to-end on small fixtures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The script takes about 10 s; the limit only stops a hung run. It is far
# above the run time because the run is mostly process start-up and
# single-threaded CPU work, which slow down with the host's load.
TIMEOUT_S = 600


def test_scikit_allel_comparison_runs_small():
    """examples/scikit_allel_comparison.py --small --no-plot must exit 0
    and verify the three diversity statistics + the LD-decay curve on
    the 4 Mb gamb subset.

    Uses a small --ld-snps to keep the scikit-allel side short (its
    rogers_huff_r is single-threaded and grows with the square of the
    SNP count). Disables the MAC filter so the small block keeps enough
    variants for parity to be meaningful.

    A timeout says nothing about correctness, so it is reported apart
    from a failed run, with the host's load."""
    script = REPO_ROOT / "examples" / "scikit_allel_comparison.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--small", "--no-plot",
             "--ld-snps", "2000", "--ld-mac-min", "0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        load = os.getloadavg()[0]
        pytest.fail(
            f"TIMEOUT, not a wrong result: the script did not finish in "
            f"{TIMEOUT_S} s (host load average {load:.0f} on "
            f"{os.cpu_count()} CPUs). Run this test again on a quiet host "
            f"before you read it as a regression.\n"
            f"--- stdout so far ---\n{exc.stdout or ''}\n"
            f"--- stderr so far ---\n{exc.stderr or ''}\n",
            pytrace=False)
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n")
    # Sanity: both verification blocks must have printed.
    assert "tajimas_d: max abs diff" in result.stdout, result.stdout
    assert "median r" in result.stdout, result.stdout
