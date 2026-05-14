"""Tests for StreamingHaplotypeMatrix + HostChunkFetcher."""

import msprime
import numpy as np
import pytest
import cupy as cp

from pg_gpu import HaplotypeMatrix
from pg_gpu.streaming_matrix import (
    ChunkFetcher, HostChunkFetcher, StreamingHaplotypeMatrix,
)
from pg_gpu.zarr_source import ZarrGenotypeSource


def _simulate_hm(n_samples=20, seq_length=50_000, seed=42):
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
    path = str(tmp_path / "stream.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path, hm


def _stream_concat(streaming_hm):
    """Walk every chunk and concatenate haps + positions into single arrays
    on the host. Used as the comparison target against the eager from_zarr."""
    haps_parts, pos_parts = [], []
    for left, right, chunk_hm in streaming_hm.iter_gpu_chunks():
        haps_parts.append(cp.asnumpy(chunk_hm.haplotypes))
        pos_parts.append(cp.asnumpy(chunk_hm.positions))
    return np.concatenate(haps_parts, axis=1), np.concatenate(pos_parts)


class TestStreamingFromZarr:

    def test_streaming_always_returns_streaming_class(self, vcz_store):
        path, _ = vcz_store
        hm = HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=5_000)
        assert isinstance(hm, StreamingHaplotypeMatrix)

    def test_streaming_never_returns_eager(self, vcz_store):
        path, _ = vcz_store
        hm = HaplotypeMatrix.from_zarr(path, streaming="never")
        assert isinstance(hm, HaplotypeMatrix)

    def test_auto_picks_eager_on_small_store(self, vcz_store):
        # small msprime store, plenty of free GPU memory -> auto goes eager.
        path, _ = vcz_store
        hm = HaplotypeMatrix.from_zarr(path, streaming="auto")
        assert isinstance(hm, HaplotypeMatrix)

    def test_invalid_streaming_raises(self, vcz_store):
        path, _ = vcz_store
        with pytest.raises(ValueError, match="streaming must be"):
            HaplotypeMatrix.from_zarr(path, streaming="maybe")


class TestStreamingMatrixSurface:

    def test_basic_metadata(self, vcz_store):
        path, eager = vcz_store
        smatrix = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        assert smatrix.num_variants == eager.haplotypes.shape[1]
        assert smatrix.num_haplotypes == eager.haplotypes.shape[0]
        assert smatrix.chrom == "1"
        assert smatrix.chrom_start <= smatrix.chrom_end
        assert "streaming" in repr(smatrix).lower() or "Streaming" in repr(smatrix)

    def test_haplotypes_property_raises(self, vcz_store):
        path, _ = vcz_store
        smatrix = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        with pytest.raises(NotImplementedError, match="no materialized"):
            _ = smatrix.haplotypes

    def test_sample_sets_default_to_all(self, vcz_store):
        path, _ = vcz_store
        smatrix = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        # no pop file present -> property returns the same "all" fallback
        # HaplotypeMatrix uses.
        assert set(smatrix.sample_sets.keys()) == {"all"}
        assert len(smatrix.sample_sets["all"]) == smatrix.num_haplotypes


class TestStreamingEquivalence:

    def test_stream_concat_matches_eager(self, vcz_store):
        # walking every chunk in streaming mode and concatenating the
        # per-chunk haplotype matrices should reproduce the eager matrix
        # bit-for-bit. This is the contract kernel dispatch in the follow-up
        # PR relies on.
        path, _ = vcz_store
        eager = HaplotypeMatrix.from_zarr(path, streaming="never")
        smatrix = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000, prefetch=1)
        haps, pos = _stream_concat(smatrix)
        np.testing.assert_array_equal(haps, cp.asnumpy(eager.haplotypes))
        np.testing.assert_array_equal(pos, cp.asnumpy(eager.positions))

    def test_prefetch_off_matches_prefetch_on(self, vcz_store):
        path, _ = vcz_store
        s0 = HaplotypeMatrix.from_zarr(path, streaming="always",
                                       chunk_bp=5_000, prefetch=0)
        s1 = HaplotypeMatrix.from_zarr(path, streaming="always",
                                       chunk_bp=5_000, prefetch=1)
        h0, p0 = _stream_concat(s0)
        h1, p1 = _stream_concat(s1)
        np.testing.assert_array_equal(h0, h1)
        np.testing.assert_array_equal(p0, p1)


class TestProducerThreadErrorPropagation:

    def test_producer_exception_reaches_consumer(self, vcz_store):
        # Wrap the source's slice_region in a fetcher that raises on the
        # second chunk. The producer's exception must surface at the
        # consumer's next() call, with the original message preserved.
        path, _ = vcz_store
        source = ZarrGenotypeSource(path)

        class FlakySource:
            def __init__(self, inner):
                self.inner = inner
                self._calls = 0
            def slice_region(self, left, right):
                self._calls += 1
                if self._calls == 2:
                    raise RuntimeError("synthetic producer error")
                return self.inner.slice_region(left, right)
            def iter_chunks(self, chunk_bp, align_bp=None):
                return self.inner.iter_chunks(chunk_bp, align_bp)
            @property
            def num_variants(self): return self.inner.num_variants
            @property
            def num_haplotypes(self): return self.inner.num_haplotypes
            @property
            def num_diploids(self): return self.inner.num_diploids
            @property
            def chrom(self): return self.inner.chrom
            @property
            def mappable_lo(self): return self.inner.mappable_lo
            @property
            def mappable_hi(self): return self.inner.mappable_hi
            @property
            def pop_cols(self): return self.inner.pop_cols

        flaky = FlakySource(source)
        fetcher = HostChunkFetcher(flaky)
        smatrix = StreamingHaplotypeMatrix(flaky, fetcher, chunk_bp=5_000,
                                           prefetch=1)
        with pytest.raises(RuntimeError, match="synthetic producer error"):
            for _ in smatrix.iter_gpu_chunks():
                pass


class TestAutoDetection:
    """``streaming='auto'`` should pick eager for stores that fit on the
    device and streaming for stores that don't. The size threshold is
    parameterized so tests can simulate "too big" with a tiny
    ``free_gpu_bytes`` rather than needing a biobank-scale fixture."""

    def test_auto_picks_eager_when_it_fits(self, vcz_store):
        # default free GPU memory, tiny msprime store -> eager
        path, _ = vcz_store
        hm = HaplotypeMatrix.from_zarr(path, streaming="auto")
        assert isinstance(hm, HaplotypeMatrix)
        assert not isinstance(hm, StreamingHaplotypeMatrix)

    def test_auto_picks_streaming_when_oversized(self, vcz_store):
        # Force the "fits in free GPU memory" check to fail by passing a
        # free_gpu_bytes too small for the eager footprint. This exercises
        # the heuristic without needing a multi-GB fixture.
        from pg_gpu.haplotype_matrix import _decide_streaming_mode
        path, hm = vcz_store
        eager_bytes = hm.haplotypes.size
        # Make the eager footprint look like it doesn't fit by claiming
        # less free GPU memory than the projected size requires.
        choice = _decide_streaming_mode(
            path, region=None, streaming="auto", pop_file=False,
            free_gpu_bytes=int(eager_bytes / 0.5 - 1),  # just below threshold
        )
        assert choice == "streaming"

    def test_never_raises_when_oversized(self, vcz_store):
        from pg_gpu.haplotype_matrix import _decide_streaming_mode
        path, hm = vcz_store
        eager_bytes = hm.haplotypes.size
        with pytest.raises(MemoryError, match="streaming='never'"):
            _decide_streaming_mode(
                path, region=None, streaming="never", pop_file=False,
                free_gpu_bytes=int(eager_bytes / 0.5 - 1),
            )

    def test_never_passes_when_it_fits(self, vcz_store):
        # opposite case: free memory is much larger than the matrix, so
        # streaming='never' returns eager without raising.
        path, _ = vcz_store
        hm = HaplotypeMatrix.from_zarr(path, streaming="never")
        assert isinstance(hm, HaplotypeMatrix)
        assert not isinstance(hm, StreamingHaplotypeMatrix)


class TestMaterialize:
    """``.materialize(region=...)`` is the path from streaming to pairwise
    kernels: pull a sub-region eagerly, then run pairwise_r2 / locate_unlinked
    / etc. on the eager HaplotypeMatrix it returns."""

    def test_full_region_matches_eager(self, vcz_store):
        path, _ = vcz_store
        eager = HaplotypeMatrix.from_zarr(path, streaming="never")
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        materialized = stream.materialize()
        assert isinstance(materialized, HaplotypeMatrix)
        # same haplotypes (modulo any positions outside the eager matrix's
        # range -- the streaming chunk grid extends to the chunk-aligned
        # mappable_hi, but the variants are the same).
        np.testing.assert_array_equal(cp.asnumpy(eager.haplotypes),
                                      cp.asnumpy(materialized.haplotypes))

    def test_sub_region(self, vcz_store):
        path, _ = vcz_store
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        sub = stream.materialize(region=(10_000, 30_000))
        assert isinstance(sub, HaplotypeMatrix)
        # all positions inside the requested half-open interval
        pos = cp.asnumpy(sub.positions)
        assert pos.min() >= 10_000
        assert pos.max() < 30_000
        assert sub.chrom_start == 10_000
        assert sub.chrom_end == 30_000

    def test_pairwise_r2_via_materialize(self, vcz_store):
        # The intended user pattern: streaming hm -> .materialize(region=)
        # -> pairwise_r2(). Asserts the composition produces a finite
        # (n_var x n_var) matrix with the kernel's diagonal zeroed.
        # r2 values are not bounded to [0, 1] in this test because the
        # msprime fixture uses the default Jukes-Cantor mutation model;
        # pg_gpu's binary-allele assumptions in pairwise_r2 inflate the
        # numerator at triallelic sites (tracked in kr-colab/pg_gpu#100).
        # The streaming -> materialize -> pairwise_r2 composition is what
        # this test is for; the r2 numerics are validated elsewhere.
        path, _ = vcz_store
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        sub = stream.materialize(region=(10_000, 30_000))
        r2 = sub.pairwise_r2()
        n_var = sub.haplotypes.shape[1]
        assert r2.shape == (n_var, n_var)
        assert bool(cp.isfinite(r2).all()), "pairwise_r2 returned NaN / inf"
        assert float(cp.abs(cp.diag(r2)).sum()) == 0.0

    def test_sample_subset_requires_even(self, vcz_store):
        path, _ = vcz_store
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        with pytest.raises(ValueError, match="even count"):
            stream.materialize(region=(10_000, 20_000),
                               sample_subset=[0, 1, 2])  # odd


class TestChunkFetcherABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            ChunkFetcher()
