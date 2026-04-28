"""Smoke tests that run example scripts end-to-end on small fixtures."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scikit_allel_comparison_runs_small():
    """examples/scikit_allel_comparison.py --small --no-plot must exit 0
    and verify all three diversity statistics on the 4 Mb gamb subset."""
    script = REPO_ROOT / "examples" / "scikit_allel_comparison.py"
    result = subprocess.run(
        [sys.executable, str(script), "--small", "--no-plot"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"script failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n")
    # Sanity: the script must have actually run the verification.
    assert "max abs diff" in result.stdout, result.stdout
