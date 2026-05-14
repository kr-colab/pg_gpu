"""Tests for KvikioChunkFetcher + backend auto-detection.

kvikio + nvidia-nvcomp are hard dependencies, so these tests always
run -- there is no "skipif kvikio missing" path.
"""

import os
import shutil

import cupy as cp
import msprime
import numpy as np
import pytest
import zarr

from pg_gpu import HaplotypeMatrix
from pg_gpu.streaming_matrix import (
    BadlyChunkedWarning, HostChunkFetcher, KvikioChunkFetcher,
    _pick_chunk_fetcher, _store_call_genotype_chunks,
    _store_call_genotype_codec,
)
from pg_gpu.zarr_source import ZarrGenotypeSource

from .conftest import simulate_hm


def _write_vcz_with_sample_chunk(path, hm, *, sample_chunk):
    """Build a VCZ store at ``path`` with a specific call_genotype
    sample-axis chunk size, so tests can pick bio2zarr-shaped vs
    whole-sample-axis chunking explicitly."""
    haps = cp.asnumpy(hm.haplotypes)
    pos = cp.asnumpy(hm.positions)
    gt = HaplotypeMatrix._haplotypes_to_gt(haps)
    n_var, n_dip, _ = gt.shape

    if os.path.exists(path):
        shutil.rmtree(path)

    g = zarr.create_group(store=path, overwrite=True)
    g.create_array("call_genotype",
                   shape=(n_var, n_dip, 2),
                   chunks=(min(n_var, 10_000), sample_chunk, 2),
                   dtype="int8")[:] = gt.astype(np.int8)
    g.create_array("call_genotype_mask",
                   shape=(n_var, n_dip, 2),
                   chunks=(min(n_var, 10_000), sample_chunk, 2),
                   dtype="bool")[:] = (gt < 0)
    g.create_array("variant_position", shape=(n_var,),
                   dtype="int32")[:] = pos.astype(np.int32)
    g.create_array("variant_contig", shape=(n_var,),
                   dtype="int32")[:] = np.zeros(n_var, dtype=np.int32)
    g.create_array("contig_id", shape=(1,), dtype="<U16")[:] = np.asarray(
        ["1"], dtype="<U16")
    g.create_array("sample_id", shape=(n_dip,), dtype="<U16")[:] = np.asarray(
        [f"s{i}" for i in range(n_dip)], dtype="<U16")
    return path


@pytest.fixture
def bio2zarr_store(tmp_path):
    """A VCZ store with bio2zarr-style sample chunking (sample chunk
    smaller than the full sample axis), so the kvikio backend probes
    treat the chunking as friendly."""
    hm = simulate_hm()
    return _write_vcz_with_sample_chunk(
        str(tmp_path / "bio2zarr.vcz"),
        hm,
        sample_chunk=max(1, hm.num_haplotypes // 4),
    )


@pytest.fixture
def whole_axis_store(tmp_path):
    """A VCZ store whose sample axis is one chunk (the default zarr
    writer's behavior). The kvikio path probes this and warns +
    falls back to host on large enough stores."""
    hm = simulate_hm()
    return _write_vcz_with_sample_chunk(
        str(tmp_path / "wholeaxis.vcz"),
        hm,
        sample_chunk=hm.num_haplotypes // 2,  # == n_diploids
    )


class TestStoreProbes:

    def test_codec_probe_picks_zstd(self, bio2zarr_store):
        # zarr 3's create_array defaults to zstd for int dtypes; the
        # probe reads call_genotype/zarr.json and returns the name.
        assert _store_call_genotype_codec(bio2zarr_store) == "zstd"

    def test_chunk_probe_returns_shape(self, bio2zarr_store):
        chunks = _store_call_genotype_chunks(bio2zarr_store)
        assert chunks is not None
        assert len(chunks) == 3
        assert chunks[2] == 2

    def test_missing_store_probes_return_none(self, tmp_path):
        nowhere = str(tmp_path / "does-not-exist.vcz")
        assert _store_call_genotype_codec(nowhere) is None
        assert _store_call_genotype_chunks(nowhere) is None


class TestPickFetcher:

    def test_auto_picks_kvikio_on_bio2zarr_chunks(self, bio2zarr_store):
        source = ZarrGenotypeSource(bio2zarr_store)
        fetcher = _pick_chunk_fetcher(source, backend="auto")
        assert isinstance(fetcher, KvikioChunkFetcher)
        fetcher.close()

    def test_auto_picks_host_on_whole_axis_chunks(self, whole_axis_store):
        source = ZarrGenotypeSource(whole_axis_store)
        fetcher = _pick_chunk_fetcher(source, backend="auto")
        # whole-sample-axis -> host fallback; warning suppressed below
        # the 1 GiB threshold on the test fixture's tiny footprint.
        assert isinstance(fetcher, HostChunkFetcher)

    def test_explicit_host_skips_kvikio_path(self, bio2zarr_store):
        source = ZarrGenotypeSource(bio2zarr_store)
        fetcher = _pick_chunk_fetcher(source, backend="host")
        assert isinstance(fetcher, HostChunkFetcher)

    def test_explicit_kvikio_on_unsupported_codec_raises(self, tmp_path):
        # A store whose call_genotype is uncompressed (no codec in
        # the nvCOMP-supported list). Force compressors=None so the
        # spec ends up with just a bytes codec; the probe then
        # cannot identify a GPU-decodable codec and the explicit
        # backend='kvikio' should raise rather than silently produce
        # wrong output.
        path = str(tmp_path / "uncompressed.vcz")
        if os.path.exists(path):
            shutil.rmtree(path)
        g = zarr.create_group(store=path, overwrite=True)
        n_var, n_dip = 100, 8
        gt = np.zeros((n_var, n_dip, 2), dtype=np.int8)
        g.create_array(
            "call_genotype",
            shape=(n_var, n_dip, 2),
            chunks=(n_var, n_dip, 2),
            dtype="int8",
            compressors=None,
        )[:] = gt
        g.create_array("variant_position", shape=(n_var,),
                       dtype="int32")[:] = np.arange(n_var, dtype=np.int32)
        g.create_array("variant_contig", shape=(n_var,),
                       dtype="int32")[:] = np.zeros(n_var, dtype=np.int32)
        g.create_array("contig_id", shape=(1,), dtype="<U16")[:] = ["1"]
        g.create_array("sample_id", shape=(n_dip,), dtype="<U16")[:] = [
            f"s{i}" for i in range(n_dip)]

        source = ZarrGenotypeSource(path)
        with pytest.raises(ValueError, match="nvCOMP GPU decoder"):
            _pick_chunk_fetcher(source, backend="kvikio")


class TestKvikioFetcherReads:
    """KvikioChunkFetcher should produce the same per-chunk genotype
    bytes the host fetcher does on the same store. Same source, same
    region, same answer -- modulo the result landing on GPU memory."""

    def test_chunks_match_host(self, bio2zarr_store):
        source = ZarrGenotypeSource(bio2zarr_store)

        host = HostChunkFetcher(source)
        host_chunks = []
        for ci, left, right, gt, pos, _ in host.iter_chunks(
                [(0, 10_000), (10_000, 20_000)], prefetch=0):
            host_chunks.append((left, right, np.asarray(gt), np.asarray(pos)))

        kvi = KvikioChunkFetcher(source)
        try:
            kvi_chunks = []
            for ci, left, right, gt, pos, _ in kvi.iter_chunks(
                    [(0, 10_000), (10_000, 20_000)], prefetch=0):
                kvi_chunks.append((left, right,
                                   cp.asnumpy(gt) if isinstance(gt, cp.ndarray) else np.asarray(gt),
                                   np.asarray(pos)))
        finally:
            kvi.close()

        assert len(host_chunks) == len(kvi_chunks)
        for (lh, rh, gth, ph), (lk, rk, gtk, pk) in zip(host_chunks, kvi_chunks):
            assert (lh, rh) == (lk, rk)
            np.testing.assert_array_equal(gth, gtk)
            np.testing.assert_array_equal(ph, pk)


class TestFromZarrBackendKwarg:

    def test_invalid_backend_raises(self, bio2zarr_store):
        with pytest.raises(ValueError, match="backend must be"):
            HaplotypeMatrix.from_zarr(bio2zarr_store, streaming="always",
                                      backend="banana")

    def test_streaming_always_with_kvikio_uses_kvikio_fetcher(
            self, bio2zarr_store):
        hm = HaplotypeMatrix.from_zarr(bio2zarr_store, streaming="always",
                                       backend="kvikio", chunk_bp=5_000)
        from pg_gpu.streaming_matrix import StreamingHaplotypeMatrix
        assert isinstance(hm, StreamingHaplotypeMatrix)
        assert isinstance(hm._fetcher, KvikioChunkFetcher)
