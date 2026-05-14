"""Equivalence tests for kernels that accept a StreamingHaplotypeMatrix.

Each public entry point that gains streaming-aware dispatch is checked
twice on the same data: once with the eager HaplotypeMatrix and once
with the StreamingHaplotypeMatrix. Results must agree row-for-row.
"""

import msprime
import numpy as np
import pandas as pd
import pytest

from pg_gpu import HaplotypeMatrix, windowed_analysis
from pg_gpu import sfs as sfs_module


def _simulate_hm(n_samples=20, seq_length=100_000, seed=42):
    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=seq_length,
        recombination_rate=1e-4, random_seed=seed, ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


@pytest.fixture
def vcz_store(tmp_path):
    hm = _simulate_hm()
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "kern.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


def _assert_frames_equivalent(a, b):
    """Compare two windowed_analysis DataFrames, sorted by start, with
    numeric tolerance suitable for float32-ish stats."""
    a = a.sort_values("start").reset_index(drop=True)
    b = b.sort_values("start").reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(
                a[col].to_numpy(), b[col].to_numpy(),
                rtol=1e-6, atol=1e-9,
                err_msg=f"column {col!r} disagrees",
            )
        else:
            assert (a[col] == b[col]).all(), f"column {col!r} disagrees"


def _aligned_pair(vcz_store, **stream_kwargs):
    """Return (eager, streaming) pair with their window grids aligned.

    Eager from_zarr uses ``chrom_start = positions[0]`` (the first variant
    position) by default, while the streaming path uses the chunk-grid
    origin. Both are legitimate placements; for an apples-to-apples
    equivalence check we force the eager matrix onto the streaming grid.
    """
    eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                      **stream_kwargs.get("from_zarr_kwargs", {}))
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                       **stream_kwargs)
    eager.chrom_start = stream.chrom_start
    eager.chrom_end = stream.chrom_end
    return eager, stream


class TestWindowedAnalysisDispatch:

    def test_pi_equivalent(self, vcz_store):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=["pi"])
        df_s = windowed_analysis(stream, window_size=5_000, statistics=["pi"])
        _assert_frames_equivalent(df_e, df_s)

    def test_multiple_stats_equivalent(self, vcz_store):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        stats = ["pi", "theta_w", "tajimas_d", "segregating_sites"]
        df_e = windowed_analysis(eager, window_size=5_000, statistics=stats)
        df_s = windowed_analysis(stream, window_size=5_000, statistics=stats)
        _assert_frames_equivalent(df_e, df_s)

    def test_two_pop_divergence_equivalent(self, vcz_store, tmp_path):
        # Build a two-population pop file so divergence stats have something
        # to compute. Split samples 50/50.
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        n_dip = eager.num_haplotypes // 2
        half = n_dip // 2
        popfile = str(tmp_path / "pops.tsv")
        with open(popfile, "w") as f:
            f.write("sample\tpop\n")
            for i in range(half):
                f.write(f"s{i}\tpop1\n")
            for i in range(half, n_dip):
                f.write(f"s{i}\tpop2\n")

        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          pop_file=popfile)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            pop_file=popfile,
                                            chunk_bp=10_000)
        eager.chrom_start = stream.chrom_start
        eager.chrom_end = stream.chrom_end
        df_e = windowed_analysis(eager, window_size=5_000,
                                 statistics=["fst", "dxy"],
                                 populations=["pop1", "pop2"])
        df_s = windowed_analysis(stream, window_size=5_000,
                                 statistics=["fst", "dxy"],
                                 populations=["pop1", "pop2"])
        _assert_frames_equivalent(df_e, df_s)


SINGLE_POP_STATS = [
    "pi", "theta_w", "tajimas_d", "segregating_sites",
    "theta_h", "theta_l", "fay_wu_h", "singletons",
    "normalized_fay_wu_h", "zeng_e", "zeng_dh", "max_daf",
]
TWO_POP_STATS = ["fst", "fst_hudson", "dxy", "da"]


def _two_pop_pair(vcz_store, tmp_path, **stream_kwargs):
    n_dip = HaplotypeMatrix.from_zarr(vcz_store, streaming="never").num_haplotypes // 2
    half = n_dip // 2
    popfile = str(tmp_path / "two_pops.tsv")
    with open(popfile, "w") as f:
        f.write("sample\tpop\n")
        for i in range(half):
            f.write(f"s{i}\tpop1\n")
        for i in range(half, n_dip):
            f.write(f"s{i}\tpop2\n")
    eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                      pop_file=popfile)
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                       pop_file=popfile,
                                       **stream_kwargs)
    eager.chrom_start = stream.chrom_start
    eager.chrom_end = stream.chrom_end
    return eager, stream


class TestSingleStatParametrized:
    """Every scatter_single stat individually -- catches dispatch breakage that
    a multi-stat test would hide via short-circuit columns."""

    @pytest.mark.parametrize("stat", SINGLE_POP_STATS)
    def test_each_stat_equivalent(self, vcz_store, stat):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=[stat])
        df_s = windowed_analysis(stream, window_size=5_000, statistics=[stat])
        _assert_frames_equivalent(df_e, df_s)

    @pytest.mark.parametrize("missing_data", ["include", "exclude"])
    def test_missing_data_modes(self, vcz_store, missing_data):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=["pi"],
                                 missing_data=missing_data)
        df_s = windowed_analysis(stream, window_size=5_000, statistics=["pi"],
                                 missing_data=missing_data)
        _assert_frames_equivalent(df_e, df_s)

    def test_span_normalize_false(self, vcz_store):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=["pi"],
                                 span_normalize=False)
        df_s = windowed_analysis(stream, window_size=5_000, statistics=["pi"],
                                 span_normalize=False)
        _assert_frames_equivalent(df_e, df_s)


class TestTwoPopStatParametrized:
    """Every scatter_twopop stat individually."""

    @pytest.mark.parametrize("stat", TWO_POP_STATS)
    def test_each_stat_equivalent(self, vcz_store, tmp_path, stat):
        eager, stream = _two_pop_pair(vcz_store, tmp_path, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=[stat],
                                 populations=["pop1", "pop2"])
        df_s = windowed_analysis(stream, window_size=5_000, statistics=[stat],
                                 populations=["pop1", "pop2"])
        _assert_frames_equivalent(df_e, df_s)


class TestAccessibleBedDispatch:
    """The streaming path passes accessible_bed through to each per-chunk
    windowed_analysis call. A BED mask that excludes part of the genome
    should produce the same per-window n_total_sites / span on both
    paths."""

    def _write_bed(self, path):
        # exclude 0-25000 and 75000-100000 (so only the middle 50 kb is
        # accessible), forcing the mask to be applied at chunk boundaries
        # rather than only at the matrix edges.
        with open(path, "w") as f:
            f.write("1\t25000\t75000\n")
        return path

    def test_accessible_bed_equivalent(self, vcz_store, tmp_path):
        bed = self._write_bed(str(tmp_path / "acc.bed"))
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        # eager_chrom must remain "1" for the BED to match the contig name
        df_e = windowed_analysis(eager, window_size=5_000, statistics=["pi"],
                                 accessible_bed=bed, chrom="1")
        df_s = windowed_analysis(stream, window_size=5_000, statistics=["pi"],
                                 accessible_bed=bed, chrom="1")
        _assert_frames_equivalent(df_e, df_s)


class TestGarudGuardrail:
    """Garud is rejected on the streaming path until the fused kernel's
    rounding sensitivity is addressed. Position-deterministic weights
    make the hash basis stable, but second-order rounding still drifts
    the distinct-haplotype count by a few ULP under different prefix
    trajectories."""

    @pytest.mark.parametrize("stat", ["garud_h1", "garud_h12",
                                       "garud_h123", "garud_h2h1"])
    def test_each_stat_rejected(self, vcz_store, stat):
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        with pytest.raises(NotImplementedError, match="Garud"):
            windowed_analysis(stream, window_size=5_000, statistics=[stat])


class TestSFSDispatch:

    def test_sfs_equivalent(self, vcz_store):
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        s_e = sfs_module.sfs(eager)
        s_s = sfs_module.sfs(stream)
        np.testing.assert_array_equal(s_e, s_s)

    def test_joint_sfs_equivalent(self, vcz_store, tmp_path):
        n_dip = HaplotypeMatrix.from_zarr(vcz_store).num_haplotypes // 2
        half = n_dip // 2
        popfile = str(tmp_path / "pops.tsv")
        with open(popfile, "w") as f:
            f.write("sample\tpop\n")
            for i in range(half):
                f.write(f"s{i}\tpop1\n")
            for i in range(half, n_dip):
                f.write(f"s{i}\tpop2\n")

        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          pop_file=popfile)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            pop_file=popfile,
                                            chunk_bp=10_000)
        j_e = sfs_module.joint_sfs(eager, pop1="pop1", pop2="pop2")
        j_s = sfs_module.joint_sfs(stream, pop1="pop1", pop2="pop2")
        np.testing.assert_array_equal(j_e, j_s)


class TestStreamingGuardrails:

    def test_window_must_divide_align(self, vcz_store):
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        with pytest.raises(ValueError, match="must divide"):
            # 7000 does not divide 10000
            windowed_analysis(stream, window_size=7_000, statistics=["pi"])

    def test_sliding_windows_not_supported(self, vcz_store):
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        with pytest.raises(NotImplementedError, match="sliding windows"):
            windowed_analysis(stream, window_size=5_000, step_size=2_500,
                              statistics=["pi"])

    def test_local_pca_rejected(self, vcz_store):
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        with pytest.raises(NotImplementedError, match="local_pca"):
            windowed_analysis(stream, window_size=5_000,
                              statistics=["local_pca"])

