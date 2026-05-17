"""Smoke test for examples/biobank_streaming_scan.py.

The example script doubles as the worked-example link in the
biobank-scale streaming tutorial. Run it end-to-end against a small
msprime-derived VCZ fixture so that an unintentional API drift, a
silently-empty window scale, or a forgotten ``cupy -> numpy``
conversion is caught here instead of by a reader trying to follow
the docs.

Skipped when the example script is not present (e.g. trimmed source
distributions)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import msprime
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix

EXAMPLE = (Path(__file__).resolve().parent.parent /
           "examples" / "biobank_streaming_scan.py")


@pytest.fixture(scope="module")
def two_pop_vcz(tmp_path_factory):
    """Build a 2 Mb / 2-population (AFR, EUR) VCZ store with a
    companion pop file. 2 Mb is the smallest chromosome length that
    lets the example's 1 Mb window scale produce multiple windows."""
    if not EXAMPLE.exists():
        pytest.skip(f"missing {EXAMPLE}")
    tmp = tmp_path_factory.mktemp("biobank_example")
    demo = msprime.Demography.island_model([10_000, 10_000],
                                            migration_rate=1e-4)
    demo.populations[0].name = "AFR"
    demo.populations[1].name = "EUR"
    ts = msprime.sim_ancestry(
        samples={"AFR": 25, "EUR": 25}, sequence_length=2_000_000,
        recombination_rate=1e-8, random_seed=11,
        demography=demo,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=11)
    hm = HaplotypeMatrix.from_ts(ts)
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    vcz = str(tmp / "chr1.vcz")
    hm.to_zarr(vcz, format="vcz", contig_name="1")
    with open(vcz + ".pops.tsv", "w") as f:
        f.write("sample\tpop\n")
        for i in range(25):
            f.write(f"s{i}\tAFR\n")
        for i in range(25, 50):
            f.write(f"s{i}\tEUR\n")
    return vcz, tmp


def test_example_runs_end_to_end(two_pop_vcz, tmp_path):
    vcz, _ = two_pop_vcz
    out_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parent.parent

    proc = subprocess.run(
        [sys.executable, str(EXAMPLE),
         "--vcz", vcz, "--out-dir", str(out_dir),
         "--chunk-bp", "1000000",
         "--heatmap-subsample", "10",
         "--garud-subsample", "20",
         "--joint-sfs-target", "10"],
        cwd=str(repo_root),
        env={**os.environ},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"example failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # All declared outputs must exist and be non-trivial.
    summary = json.loads((out_dir / "chr1_summary.json").read_text())
    assert summary["chromosome"] == "1"
    assert summary["haplotypes_per_pop"] == {"AFR": 50, "EUR": 50}

    # Per-scale window counts: 2 Mb chromosome / scale = number of
    # windows. If any scale silently emits zero or chunk-clipped rows
    # (the bug that prompted this test), we catch it here.
    for label, bp in (("10kb", 10_000), ("100kb", 100_000),
                      ("1mb", 1_000_000)):
        path = out_dir / f"chr1_{label}.csv"
        assert path.exists(), f"missing {path}"
        lines = path.read_text().splitlines()
        # 1 header + (chromosome_length / window_size) data rows.
        assert len(lines) - 1 == 2_000_000 // bp, (
            f"{label}: expected {2_000_000 // bp} windows, got "
            f"{len(lines) - 1}")

    # Joint SFS shape matches the projection target.
    joint = np.load(out_dir / "chr1_joint_sfs.npy")
    assert joint.shape == (11, 11)  # target_n1 + 1 by target_n2 + 1

    # Pairwise r^2 heatmap exists and is a square matrix.
    r2 = np.load(out_dir / "chr1_r2_heatmap.npy")
    assert r2.ndim == 2 and r2.shape[0] == r2.shape[1]


def test_example_errors_on_bad_chunk_bp(two_pop_vcz, tmp_path):
    # chunk_bp=500_000 doesn't divide the 1 Mb window scale; the
    # example must refuse rather than silently emit chunk-clipped
    # rows.
    vcz, _ = two_pop_vcz
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE),
         "--vcz", vcz, "--out-dir", str(tmp_path / "bad"),
         "--chunk-bp", "500000",
         "--heatmap-subsample", "10",
         "--garud-subsample", "20",
         "--joint-sfs-target", "10"],
        cwd=str(repo_root),
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "not a multiple of every WINDOW_SCALES" in proc.stderr
