"""Equivalence tests for kernels that accept a StreamingHaplotypeMatrix.

Each public entry point that gains streaming-aware dispatch is checked
twice on the same data: once with the eager HaplotypeMatrix and once
with the StreamingHaplotypeMatrix. Results must agree row-for-row.
"""

import numpy as np
import pandas as pd
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, windowed_analysis
from pg_gpu import sfs as sfs_module

from .conftest import simulate_hm


def _simulate_hm(n_samples=20, seq_length=100_000, seed=42):
    return simulate_hm(n_samples=n_samples, seq_length=seq_length, seed=seed,
                       mutation_model='binary')


@pytest.fixture
def vcz_store(tmp_path):
    hm = _simulate_hm()
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "kern.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


@pytest.fixture
def vcz_store_missing(tmp_path):
    """Like vcz_store but with ~10% missing genotypes, to exercise the
    missing-data paths (include and exclude) over the streaming dispatch."""
    hm = _simulate_hm()
    hap = hm.haplotypes
    hap = (hap.get() if hasattr(hap, "get") else np.asarray(hap)).copy()
    rng = np.random.RandomState(0)
    hap[rng.random(hap.shape) < 0.1] = -1
    pos = hm.positions.get() if hasattr(hm.positions, "get") else hm.positions
    hm2 = HaplotypeMatrix(hap, pos, hm.chrom_start, hm.chrom_end)
    hm2.samples = [f"s{i}" for i in range(hap.shape[0] // 2)]
    path = str(tmp_path / "kern_missing.vcz")
    hm2.to_zarr(path, format="vcz", contig_name="1")
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


# (chunk_bp, window_size) combos that exercise the streaming dispatch:
# * one window per chunk (chunk == window)
# * a couple of windows per chunk (the typical case)
# * many windows per chunk (chunk >> window)
# * the whole-fixture single-chunk case (chunk >= seq_length)
# Streaming forbids chunk_bp < window_size and chunk_bp % window_size
# != 0; those are covered by TestStreamingGuardrails below.
CHUNK_WINDOW_COMBOS = [
    (5_000, 5_000),     # one window per chunk
    (10_000, 5_000),    # two windows per chunk
    (25_000, 5_000),    # five windows per chunk
    (50_000, 10_000),   # five-window chunk at a larger window
    (200_000, 5_000),   # everything in one chunk
]


class TestWindowedAnalysisDispatch:

    @pytest.mark.parametrize("chunk_bp,window_size", CHUNK_WINDOW_COMBOS)
    def test_pi_equivalent(self, vcz_store, chunk_bp, window_size):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=chunk_bp)
        df_e = windowed_analysis(eager, window_size=window_size,
                                  statistics=["pi"])
        df_s = windowed_analysis(stream, window_size=window_size,
                                  statistics=["pi"])
        _assert_frames_equivalent(df_e, df_s)

    @pytest.mark.parametrize("chunk_bp,window_size", CHUNK_WINDOW_COMBOS)
    def test_multiple_stats_equivalent(self, vcz_store, chunk_bp, window_size):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=chunk_bp)
        stats = ["pi", "theta_w", "tajimas_d", "segregating_sites"]
        df_e = windowed_analysis(eager, window_size=window_size,
                                  statistics=stats)
        df_s = windowed_analysis(stream, window_size=window_size,
                                  statistics=stats)
        _assert_frames_equivalent(df_e, df_s)

    @pytest.mark.parametrize("chunk_bp,window_size", CHUNK_WINDOW_COMBOS)
    def test_daf_hist_mu_sfs_equivalent(self, vcz_store, chunk_bp, window_size):
        # daf_hist (normalized histogram) and mu_sfs (edge/segregating ratio) are
        # not plain sums, so a naive per-chunk normalize-then-concatenate would be
        # wrong. It is correct here only because the streaming grid aligns windows
        # to chunk boundaries (a window never straddles two chunks), so each
        # window is computed whole within one chunk. This pins that.
        eager, stream = _aligned_pair(vcz_store, chunk_bp=chunk_bp)
        stats = ["daf_hist", "mu_sfs"]
        df_e = windowed_analysis(eager, window_size=window_size, statistics=stats)
        df_s = windowed_analysis(stream, window_size=window_size, statistics=stats)
        _assert_frames_equivalent(df_e, df_s)

    def test_daf_hist_mu_sfs_missing_include_equivalent(self, vcz_store_missing):
        # daf_hist / mu_sfs over missing data, include mode: per-site n_valid is
        # chunk-invariant (a site's non-missing count does not depend on other
        # sites), so streaming must match eager.
        eager, stream = _aligned_pair(vcz_store_missing, chunk_bp=25_000)
        stats = ["daf_hist", "mu_sfs"]
        df_e = windowed_analysis(eager, window_size=5_000, statistics=stats,
                                 missing_data="include")
        df_s = windowed_analysis(stream, window_size=5_000, statistics=stats,
                                 missing_data="include")
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
                                          pop_assignment=popfile)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            pop_assignment=popfile,
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
                                      pop_assignment=popfile)
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                       pop_assignment=popfile,
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

    def test_load_time_mask_filters_chunks(self, vcz_store, tmp_path):
        # A mask supplied at LOAD time (from_zarr(accessible_bed=)) is attached
        # to every chunk by iter_gpu_chunks, so every streaming consumer sees
        # accessible-filtered variants -- not just genetic_relatedness. #151
        bed = self._write_bed(str(tmp_path / "acc.bed"))   # accessible: [25000, 75000)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000, accessible_bed=bed)
        seen = 0
        for _, _, chunk in stream.iter_gpu_chunks():
            assert chunk.has_accessible_mask
            pos = chunk.positions
            pos = pos.get() if hasattr(pos, "get") else np.asarray(pos)
            assert ((pos >= 25000) & (pos < 75000)).all()
            seen += chunk.num_variants
        assert seen > 0

    def test_load_time_mask_windowed_equivalent(self, vcz_store, tmp_path):
        # A LOAD-TIME mask must reach windowed_analysis without re-passing
        # accessible_bed: iter_gpu_chunks attaches it, and the per-chunk
        # eager call honors the already-attached mask. This is a distinct
        # path from test_accessible_bed_equivalent, which passes the BED as
        # an argument. Both eager and streaming carry the mask from load.
        bed = self._write_bed(str(tmp_path / "acc.bed"))
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          accessible_bed=bed)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000, accessible_bed=bed)
        eager.chrom_start = stream.chrom_start
        eager.chrom_end = stream.chrom_end
        stats = ["pi", "theta_w", "segregating_sites"]
        df_e = windowed_analysis(eager, window_size=5_000, statistics=stats)
        df_s = windowed_analysis(stream, window_size=5_000, statistics=stats)
        _assert_frames_equivalent(df_e, df_s)

        # Guard that the mask actually bit: an unmasked streaming run must
        # differ, else the test would pass even if the mask were dropped.
        unmasked = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                             chunk_bp=10_000)
        df_u = windowed_analysis(unmasked, window_size=5_000,
                                 statistics=["segregating_sites"])
        df_u = df_u.sort_values("start").reset_index(drop=True)
        df_s2 = df_s.sort_values("start").reset_index(drop=True)
        assert not np.allclose(df_u["segregating_sites"].to_numpy(),
                               df_s2["segregating_sites"].to_numpy())


class TestGarudStreamingDispatch:
    """Per-window scatter-reduce assembles each window's hash from the
    same set of variants regardless of chunking, so eager and streaming
    Garud H must agree bit-for-bit (no prefix-sum reorder drift to
    absorb)."""

    @pytest.mark.parametrize("stat", ["garud_h1", "garud_h12",
                                       "garud_h123", "garud_h2h1"])
    def test_each_stat_matches_eager(self, vcz_store, stat):
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        df_e = windowed_analysis(eager, window_size=5_000, statistics=[stat])
        df_s = windowed_analysis(stream, window_size=5_000, statistics=[stat])
        # Sort by window start; demand exact-precision parity on the
        # numeric column. The whole point of the rewrite is that the
        # streaming path's per-window hash is identical to the eager
        # one's, so float jitter is no longer a thing.
        df_e = df_e.sort_values("start").reset_index(drop=True)
        df_s = df_s.sort_values("start").reset_index(drop=True)
        assert list(df_e.columns) == list(df_s.columns)
        np.testing.assert_array_equal(df_e[stat].to_numpy(),
                                       df_s[stat].to_numpy())

    def test_garud_with_pi_in_one_call(self, vcz_store):
        # Mixed stat set -- the streaming dispatch should run pi and
        # Garud H side by side on the same chunks without either
        # disturbing the other.
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        stats = ["pi", "garud_h12"]
        df_e = windowed_analysis(eager, window_size=5_000, statistics=stats)
        df_s = windowed_analysis(stream, window_size=5_000, statistics=stats)
        df_e = df_e.sort_values("start").reset_index(drop=True)
        df_s = df_s.sort_values("start").reset_index(drop=True)
        np.testing.assert_array_equal(df_e["garud_h12"].to_numpy(),
                                       df_s["garud_h12"].to_numpy())
        np.testing.assert_allclose(df_e["pi"].to_numpy(),
                                    df_s["pi"].to_numpy(),
                                    rtol=1e-9, atol=1e-12)

    def test_chunk_boundary_invariance(self, vcz_store):
        # The most direct stress test for prefix-sum drift used to be
        # exactly this: change the chunk_bp and the per-window hash
        # gets accumulated in a different order. After the per-window
        # scatter-reduce rewrite the two chunkings produce identical
        # results.
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        s_a = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=10_000)
        s_b = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=20_000)
        eager.chrom_start = s_a.chrom_start
        eager.chrom_end = s_a.chrom_end
        df_e = windowed_analysis(eager, window_size=5_000,
                                  statistics=["garud_h12"])
        df_a = windowed_analysis(s_a, window_size=5_000,
                                  statistics=["garud_h12"])
        df_b = windowed_analysis(s_b, window_size=5_000,
                                  statistics=["garud_h12"])
        for df in (df_a, df_b):
            df.sort_values("start", inplace=True)
            df.reset_index(drop=True, inplace=True)
        df_e.sort_values("start", inplace=True)
        df_e.reset_index(drop=True, inplace=True)
        np.testing.assert_array_equal(df_e["garud_h12"].to_numpy(),
                                       df_a["garud_h12"].to_numpy())
        np.testing.assert_array_equal(df_a["garud_h12"].to_numpy(),
                                       df_b["garud_h12"].to_numpy())


class TestSFSDispatch:

    def test_sfs_equivalent(self, vcz_store):
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        s_e = sfs_module.sfs(eager)
        s_s = sfs_module.sfs(stream)
        np.testing.assert_array_equal(s_e, s_s)

    def test_sfs_folded_equivalent(self, vcz_store):
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        np.testing.assert_allclose(sfs_module.sfs_folded(eager),
                                   sfs_module.sfs_folded(stream))

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
                                          pop_assignment=popfile)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            pop_assignment=popfile,
                                            chunk_bp=10_000)
        j_e = sfs_module.joint_sfs(eager, pop1="pop1", pop2="pop2")
        j_s = sfs_module.joint_sfs(stream, pop1="pop1", pop2="pop2")
        np.testing.assert_array_equal(j_e, j_s)

    def test_project_joint_sfs_equivalent(self, vcz_store, tmp_path):
        # Eager projection (built from full joint_sfs) must match
        # streaming projection (which never materializes the full
        # histogram) for both an identity target (target == source,
        # recovers float-cast joint_sfs) and a strict downsample.
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
                                          pop_assignment=popfile)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            pop_assignment=popfile,
                                            chunk_bp=10_000)
        # Each pop has 2 * half haplotypes; pick a strict downsample.
        n1 = 2 * half
        n2 = 2 * (n_dip - half)
        target1 = max(1, n1 - 2)
        target2 = max(1, n2 - 2)
        e = sfs_module.project_joint_sfs(eager, pop1="pop1", pop2="pop2",
                                          target_n1=target1,
                                          target_n2=target2)
        s = sfs_module.project_joint_sfs(stream, pop1="pop1", pop2="pop2",
                                          target_n1=target1,
                                          target_n2=target2)
        np.testing.assert_allclose(e, s, rtol=1e-9, atol=1e-9)

        # Identity-target projection of an int joint SFS just casts to
        # float; pmf collapses to the indicator.
        full = sfs_module.joint_sfs(eager, pop1="pop1", pop2="pop2")
        e_id = sfs_module.project_joint_sfs(eager, pop1="pop1", pop2="pop2",
                                             target_n1=n1, target_n2=n2)
        np.testing.assert_allclose(e_id, full.astype(np.float64),
                                    rtol=1e-9, atol=1e-9)


class TestStreamingGuardrails:

    def test_window_must_divide_align(self, vcz_store):
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        with pytest.raises(ValueError, match="must divide"):
            # 7000 does not divide 10000
            windowed_analysis(stream, window_size=7_000, statistics=["pi"])

    def test_window_larger_than_chunk_rejected(self, vcz_store):
        # A window that is bigger than the chunk would have to straddle
        # chunk boundaries; the streaming guardrail rejects it with the
        # same 'must divide' error path (window > chunk is equivalent
        # to a non-divisor since chunk_bp % window_size is window_size's
        # own value, which is greater than zero).
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=5_000)
        with pytest.raises(ValueError, match="must divide"):
            windowed_analysis(stream, window_size=20_000, statistics=["pi"])

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



class TestGeneticRelatednessDispatch:
    """Streaming genetic_relatedness matches the eager result row-for-row."""

    def _both(self, vcz_store, **kw):
        from pg_gpu import relatedness
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        return (relatedness.genetic_relatedness(eager, **kw),
                relatedness.genetic_relatedness(stream, **kw))

    def test_singleton_raw_equivalent(self, vcz_store):
        e, s = self._both(vcz_store, span_normalize=False)
        np.testing.assert_allclose(s, e, rtol=1e-6, atol=1e-9)

    def test_proportion_equivalent(self, vcz_store):
        e, s = self._both(vcz_store, proportion=True)
        np.testing.assert_allclose(s, e, rtol=1e-6, atol=1e-9)

    def test_noncentred_equivalent(self, vcz_store):
        e, s = self._both(vcz_store, centre=False, span_normalize=False)
        np.testing.assert_allclose(s, e, rtol=1e-6, atol=1e-9)

    def test_polarised_equivalent(self, vcz_store):
        e, s = self._both(vcz_store, polarised=True, span_normalize=False)
        np.testing.assert_allclose(s, e, rtol=1e-6, atol=1e-9)

    def test_grouped_sample_sets_equivalent(self, vcz_store):
        from pg_gpu import relatedness
        eager, stream = _aligned_pair(vcz_store, chunk_bp=10_000)
        n_hap = stream.num_haplotypes
        half = n_hap // 2
        sets = [list(range(0, half)), list(range(half, n_hap))]
        e = relatedness.genetic_relatedness(eager, sample_sets=sets,
                                            span_normalize=False)
        s = relatedness.genetic_relatedness(stream, sample_sets=sets,
                                            span_normalize=False)
        np.testing.assert_allclose(s, e, rtol=1e-6, atol=1e-9)

    def test_chunk_boundary_invariance(self, vcz_store):
        from pg_gpu import relatedness
        s_a = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                        chunk_bp=10_000)
        s_b = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                        chunk_bp=20_000)
        a = relatedness.genetic_relatedness(s_a, span_normalize=False)
        b = relatedness.genetic_relatedness(s_b, span_normalize=False)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_span_normalize_no_mask_equivalent(self, vcz_store):
        # Default span_normalize=True with no accessible mask: streaming must
        # normalize by the same variant-position span as the eager path
        # (both use mappable_lo/hi, not the chunk-grid edges). #151
        from pg_gpu import relatedness
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000)
        e = relatedness.genetic_relatedness(eager, span_normalize=True)
        s = relatedness.genetic_relatedness(stream, span_normalize=True)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    def test_accessible_bed_span_and_filter_equivalent(self, vcz_store, tmp_path):
        # accessible_bed both restricts variants to accessible sites and sets
        # the accessible-base span; streaming must match eager on both. #151
        from pg_gpu import relatedness
        bed = str(tmp_path / "acc.bed")
        with open(bed, "w") as f:
            f.write("1\t25000\t75000\n")   # only the middle 50 kb accessible
        full = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          accessible_bed=bed)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000, accessible_bed=bed)
        assert eager.num_variants < full.num_variants   # mask really filters
        e = relatedness.genetic_relatedness(eager, span_normalize=True)
        s = relatedness.genetic_relatedness(stream, span_normalize=True)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("mode", ["per_variant", "sites", "callable", "per_base"])
    def test_span_mode_matches_eager_with_mask(self, vcz_store, tmp_path, mode):
        # Non-default span modes: per_variant/sites/callable count over the
        # accessible-FILTERED variant set (eager does the same via its filtered
        # view); per_base is the raw span. All must agree eager vs streaming.
        from pg_gpu import relatedness
        bed = str(tmp_path / "acc.bed")
        with open(bed, "w") as f:
            f.write("1\t25000\t75000\n")
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          accessible_bed=bed)
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000, accessible_bed=bed)
        assert eager.get_span(mode) == stream.get_span(mode)
        e = relatedness.genetic_relatedness(eager, span_normalize=mode)
        s = relatedness.genetic_relatedness(stream, span_normalize=mode)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("missing_data", ["include", "exclude"])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_missing_data_equivalent(self, vcz_store_missing, tmp_path,
                                     missing_data, use_mask):
        # genetic_relatedness on missing genotypes: per-site per-set frequencies
        # and both denominators -- the accessible span (span_normalize) and the
        # segregating-site count (proportion) -- must match eager, including with
        # an accessible mask. Every other streaming fixture is missing-free, so
        # this is the only test exercising the missing-data path for the
        # streaming accessible-mask handling on the haplotype side.
        from pg_gpu import relatedness
        kw = {}
        if use_mask:
            bed = str(tmp_path / "acc.bed")
            with open(bed, "w") as f:
                f.write("1\t25000\t75000\n")
            kw["accessible_bed"] = bed
        eager = HaplotypeMatrix.from_zarr(vcz_store_missing, streaming="never",
                                          **kw)
        assert bool((eager.haplotypes < 0).any())   # guard: missing present
        stream = HaplotypeMatrix.from_zarr(vcz_store_missing, streaming="always",
                                           chunk_bp=10_000, **kw)
        for extra in ({"span_normalize": True}, {"proportion": True}):
            e = relatedness.genetic_relatedness(eager, missing_data=missing_data,
                                                **extra)
            s = relatedness.genetic_relatedness(stream, missing_data=missing_data,
                                                **extra)
            np.testing.assert_allclose(
                s, e, rtol=1e-6, atol=1e-9,
                err_msg=f"{extra} missing={missing_data} mask={use_mask}")


class TestGrmIbsAccessibleBed:
    """Streaming grm/ibs match eager row-for-row, and a load-time
    accessible_bed filters variants identically on both paths. grm/ibs are
    per-site averages (not span-normalized), so the mask matters here purely
    as variant filtering. #152"""

    # Accessible: [25000, 75000). Excluding the flanks forces the mask to be
    # applied at chunk boundaries, not just the matrix edges.
    def _write_bed(self, path):
        with open(path, "w") as f:
            f.write("1\t25000\t75000\n")
        return path

    @pytest.mark.parametrize("stat", ["grm", "ibs"])
    def test_no_mask_equivalent(self, vcz_store, stat):
        from pg_gpu import relatedness
        fn = getattr(relatedness, stat)
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=10_000)
        np.testing.assert_allclose(fn(stream), fn(eager),
                                   rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("stat", ["grm", "ibs"])
    def test_accessible_bed_equivalent(self, vcz_store, tmp_path, stat):
        from pg_gpu import relatedness
        fn = getattr(relatedness, stat)
        bed = self._write_bed(str(tmp_path / "acc.bed"))
        full = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never",
                                         accessible_bed=bed)
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=10_000, accessible_bed=bed)
        assert eager.num_variants < full.num_variants   # mask really filters
        np.testing.assert_allclose(fn(stream), fn(eager),
                                   rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("stat", ["grm", "ibs"])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_missing_data_equivalent(self, vcz_store_missing, tmp_path, stat,
                                     use_mask):
        # grm/ibs on missing genotypes: the per-pair valid-site normalization
        # (ibs's n_joint; grm's per-site p and n_snps_used) must accumulate
        # across streaming chunks and normalize once, matching eager -- and that
        # must still hold with an accessible mask filtering variants. Every other
        # fixture is missing-free, so this is the only test exercising the
        # accumulate-then-normalize path under missing data.
        from pg_gpu import relatedness
        fn = getattr(relatedness, stat)
        kw = {}
        if use_mask:
            kw["accessible_bed"] = self._write_bed(str(tmp_path / "acc.bed"))
        eager = GenotypeMatrix.from_zarr(vcz_store_missing, streaming="never",
                                         **kw)
        # guard against a missing-free fixture silently making this vacuous
        assert bool((eager.genotypes < 0).any())
        stream = GenotypeMatrix.from_zarr(vcz_store_missing, streaming="always",
                                          chunk_bp=10_000, **kw)
        np.testing.assert_allclose(fn(stream), fn(eager),
                                   rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("stat", ["grm", "ibs"])
    def test_load_time_mask_changes_streaming_result(self, vcz_store, tmp_path,
                                                     stat):
        # Guard against a no-op mask trivially "matching" eager: the masked
        # streaming result must differ from the unmasked one. Without the
        # #152 fix the streaming matrix carries no mask and these are equal.
        from pg_gpu import relatedness
        fn = getattr(relatedness, stat)
        bed = self._write_bed(str(tmp_path / "acc.bed"))
        unmasked = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        masked = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=10_000, accessible_bed=bed)
        assert not np.allclose(fn(masked), fn(unmasked))

    def test_load_time_mask_filters_chunks(self, vcz_store, tmp_path):
        # The mask is attached to every chunk iter_gpu_chunks yields, so each
        # chunk's genotypes are restricted to accessible positions.
        bed = self._write_bed(str(tmp_path / "acc.bed"))
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                          chunk_bp=10_000, accessible_bed=bed)
        seen = 0
        for _, _, chunk in stream.iter_gpu_chunks():
            assert chunk.has_accessible_mask
            pos = chunk.positions
            pos = pos.get() if hasattr(pos, "get") else np.asarray(pos)
            assert ((pos >= 25000) & (pos < 75000)).all()
            seen += chunk.num_variants
        assert seen > 0

    @pytest.mark.parametrize("stat", ["grm", "ibs"])
    def test_auto_fallback_forwards_accessible_bed(self, vcz_store, tmp_path,
                                                   monkeypatch, stat):
        # streaming='auto' forwards accessible_bed on its streaming-fallback
        # branch too, not just streaming='always'. The tiny fixture fits
        # eagerly so 'auto' never falls back on its own; force the fallback
        # by making the size decision return 'streaming' with a real source,
        # which drives the actual _build_streaming(source=..., accessible_bed=)
        # call. Guards against dropping accessible_bed from only that branch.
        import pg_gpu.haplotype_matrix as hm_mod
        from pg_gpu.zarr_source import ZarrGenotypeSource
        from pg_gpu.streaming_matrix import StreamingGenotypeMatrix
        from pg_gpu import relatedness
        fn = getattr(relatedness, stat)
        bed = self._write_bed(str(tmp_path / "acc.bed"))

        def force_streaming(zarr_path, region, streaming, pop_assignment, **kw):
            src = ZarrGenotypeSource(zarr_path, region=region,
                                     pop_assignment=False)
            return "streaming", src
        monkeypatch.setattr(hm_mod, "_decide_streaming_mode", force_streaming)

        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="auto",
                                          chunk_bp=10_000, accessible_bed=bed)
        assert isinstance(stream, StreamingGenotypeMatrix)  # fallback taken
        assert stream.accessible_mask is not None            # bed forwarded
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never",
                                         accessible_bed=bed)
        np.testing.assert_allclose(fn(stream), fn(eager),
                                   rtol=1e-6, atol=1e-9)
