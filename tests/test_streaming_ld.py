"""Equivalence tests for streaming-aware LD statistics.

``compute_ld_statistics_gpu_single_pop`` and ``_two_pops`` accept a
``StreamingHaplotypeMatrix`` directly. Per-chunk bin sums are
sum-reducible (numerators and denominators decompose by pair), and a
tail buffer of the last ``max_bp_dist`` of variants from the previous
chunk captures pairs that straddle chunk boundaries.

Tests verify three properties on small msprime-derived stores:

1. Eager and streaming produce the same per-bin (DD, Dz, pi2) tuples
   (single-pop) and the same 15-stat OrderedDict (two-pop) when chunks
   tile cleanly and no pair crosses a boundary.
2. The tail-buffer machinery captures cross-chunk pairs: with
   ``chunk_bp`` short and ``max_bp_dist`` large enough to span at
   least one chunk boundary, eager vs streaming still agree.
3. Per-chunk biallelic filtering produces the same filtered pair set
   as the global filter the eager path uses.
"""

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix

from .conftest import simulate_hm


@pytest.fixture
def vcz_store(tmp_path):
    """Single-population vcz with enough variants to populate multiple
    chunks and several bp bins."""
    # binary model -- the eager-vs-streaming comparison only needs the
    # two paths to agree on the same data, so the multi-allelic gap
    # would be a confounder rather than something we want to exercise.
    hm = simulate_hm(n_samples=24, seq_length=200_000, seed=11,
                     mutation_model="binary")
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "ld.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


@pytest.fixture
def two_pop_vcz_store(tmp_path):
    """Vcz store with a two-population pop file split 50/50."""
    hm = simulate_hm(n_samples=24, seq_length=200_000, seed=23,
                     mutation_model="binary")
    n_dip = hm.num_haplotypes // 2
    hm.samples = [f"s{i}" for i in range(n_dip)]
    path = str(tmp_path / "ld_twopop.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    half = n_dip // 2
    popfile = str(tmp_path / "pops.tsv")
    with open(popfile, "w") as f:
        f.write("sample\tpop\n")
        for i in range(half):
            f.write(f"s{i}\tpop1\n")
        for i in range(half, n_dip):
            f.write(f"s{i}\tpop2\n")
    return path, popfile


def _aligned_single_pop_pair(vcz_store, **stream_kwargs):
    eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                        **stream_kwargs)
    return eager, stream


def _aligned_two_pop_pair(path, popfile, **stream_kwargs):
    eager = HaplotypeMatrix.from_zarr(path, streaming="never",
                                       pop_file=popfile)
    stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                        pop_file=popfile, **stream_kwargs)
    return eager, stream


def _assert_single_pop_equivalent(eager_dict, stream_dict, *, rtol=1e-9):
    """Compare two single-pop LD dicts of (DD, Dz, pi2) per bin."""
    assert set(eager_dict.keys()) == set(stream_dict.keys())
    for key in eager_dict:
        np.testing.assert_allclose(
            np.array(stream_dict[key]),
            np.array(eager_dict[key]),
            rtol=rtol, atol=1e-12,
            err_msg=f"bin {key} disagrees",
        )


def _assert_two_pop_equivalent(eager_dict, stream_dict, *, rtol=1e-9):
    """Compare two two-pop LD dicts of OrderedDict[stat_name, float]."""
    assert set(eager_dict.keys()) == set(stream_dict.keys())
    for key in eager_dict:
        e_stats = eager_dict[key]
        s_stats = stream_dict[key]
        assert list(e_stats.keys()) == list(s_stats.keys()), (
            f"stat-name order disagrees for bin {key}"
        )
        np.testing.assert_allclose(
            np.array(list(s_stats.values())),
            np.array(list(e_stats.values())),
            rtol=rtol, atol=1e-12,
            err_msg=f"bin {key} disagrees",
        )


class TestSinglePopParity:

    @pytest.mark.parametrize("chunk_bp", [50_000, 25_000])
    def test_parity_single_chunk(self, vcz_store, chunk_bp):
        # Bin upper edge is short relative to chunk width, so no pair
        # crosses a chunk boundary -- the tail buffer should be empty
        # at every chunk and the result must match eager exactly.
        eager, stream = _aligned_single_pop_pair(vcz_store, chunk_bp=chunk_bp)
        bp_bins = [0, 1_000, 5_000]  # max_dist 5kb << 25kb chunk
        e = eager.compute_ld_statistics_gpu_single_pop(bp_bins)
        s = stream.compute_ld_statistics_gpu_single_pop(bp_bins)
        _assert_single_pop_equivalent(e, s)

    def test_cross_chunk_pairs_counted(self, vcz_store):
        # max_bp_dist == chunk_bp guarantees that any chunk's right
        # half feeds the next chunk's tail. Eager vs streaming must
        # still agree on every bin -- this is the test that exercises
        # the tail-buffer pair set.
        eager, stream = _aligned_single_pop_pair(vcz_store, chunk_bp=10_000)
        bp_bins = [0, 2_500, 5_000, 10_000]
        e = eager.compute_ld_statistics_gpu_single_pop(bp_bins)
        s = stream.compute_ld_statistics_gpu_single_pop(bp_bins)
        _assert_single_pop_equivalent(e, s)

    def test_raw_sums_parity(self, vcz_store):
        # raw=True returns per-bin numerator sums. Sum-reducibility is
        # the strongest claim we make; verify it directly without the
        # final divide.
        eager, stream = _aligned_single_pop_pair(vcz_store, chunk_bp=20_000)
        bp_bins = [0, 5_000, 15_000]
        e = eager.compute_ld_statistics_gpu_single_pop(bp_bins, raw=True)
        s = stream.compute_ld_statistics_gpu_single_pop(bp_bins, raw=True)
        _assert_single_pop_equivalent(e, s)

    def test_ac_filter_parity(self, vcz_store):
        # Per-chunk apply_biallelic_filter must produce the same kept
        # variants as the global filter on the eager matrix, because
        # the filter looks only at allele counts at each variant.
        eager, stream = _aligned_single_pop_pair(vcz_store, chunk_bp=10_000)
        bp_bins = [0, 2_500, 7_500]
        e = eager.compute_ld_statistics_gpu_single_pop(bp_bins, ac_filter=True)
        s = stream.compute_ld_statistics_gpu_single_pop(bp_bins, ac_filter=True)
        _assert_single_pop_equivalent(e, s)

    def test_chunk_bp_smaller_than_max_dist(self, vcz_store):
        # When the chunk width is smaller than max_bp_dist, the tail
        # buffer ends up spanning multiple prior chunks. The carry-over
        # logic must still keep every variant whose position is within
        # max_dist of the right edge.
        eager, stream = _aligned_single_pop_pair(vcz_store, chunk_bp=5_000)
        bp_bins = [0, 5_000, 15_000]
        e = eager.compute_ld_statistics_gpu_single_pop(bp_bins)
        s = stream.compute_ld_statistics_gpu_single_pop(bp_bins)
        _assert_single_pop_equivalent(e, s)


class TestTwoPopParity:

    def test_parity_single_chunk(self, two_pop_vcz_store):
        path, popfile = two_pop_vcz_store
        eager, stream = _aligned_two_pop_pair(path, popfile, chunk_bp=50_000)
        bp_bins = [0, 1_000, 5_000]
        e = eager.compute_ld_statistics_gpu_two_pops(bp_bins, "pop1", "pop2")
        s = stream.compute_ld_statistics_gpu_two_pops(bp_bins, "pop1", "pop2")
        _assert_two_pop_equivalent(e, s)

    def test_cross_chunk_pairs_counted(self, two_pop_vcz_store):
        path, popfile = two_pop_vcz_store
        eager, stream = _aligned_two_pop_pair(path, popfile, chunk_bp=10_000)
        bp_bins = [0, 2_500, 5_000, 10_000]
        e = eager.compute_ld_statistics_gpu_two_pops(bp_bins, "pop1", "pop2")
        s = stream.compute_ld_statistics_gpu_two_pops(bp_bins, "pop1", "pop2")
        _assert_two_pop_equivalent(e, s)

    def test_raw_sums_parity(self, two_pop_vcz_store):
        path, popfile = two_pop_vcz_store
        eager, stream = _aligned_two_pop_pair(path, popfile, chunk_bp=20_000)
        bp_bins = [0, 5_000, 15_000]
        e = eager.compute_ld_statistics_gpu_two_pops(
            bp_bins, "pop1", "pop2", raw=True,
        )
        s = stream.compute_ld_statistics_gpu_two_pops(
            bp_bins, "pop1", "pop2", raw=True,
        )
        _assert_two_pop_equivalent(e, s)

    def test_ac_filter_parity(self, two_pop_vcz_store):
        path, popfile = two_pop_vcz_store
        eager, stream = _aligned_two_pop_pair(path, popfile, chunk_bp=10_000)
        bp_bins = [0, 2_500, 7_500]
        e = eager.compute_ld_statistics_gpu_two_pops(
            bp_bins, "pop1", "pop2", ac_filter=True,
        )
        s = stream.compute_ld_statistics_gpu_two_pops(
            bp_bins, "pop1", "pop2", ac_filter=True,
        )
        _assert_two_pop_equivalent(e, s)
