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

    def test_auto_defaults_to_eager_today(self, vcz_store):
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


class TestChunkFetcherABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            ChunkFetcher()
