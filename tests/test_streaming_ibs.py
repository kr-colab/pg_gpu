"""Equivalence tests for streaming-aware IBS.

``ibs(matrix, ...)`` dispatches at the top: if the argument is a
``StreamingHaplotypeMatrix`` / ``StreamingGenotypeMatrix``, the variant
axis is streamed chunk-by-chunk and the individual axis is tiled into
row blocks so the (n_ind, n_ind) accumulators never have to fit on the
GPU. Per-chunk contributions sum on the host; the final ratio is
applied once. Tests assert that streaming and eager produce the same
matrix on small msprime stores.
"""

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix
from pg_gpu.relatedness import ibs

from .conftest import simulate_hm


@pytest.fixture
def vcz_store(tmp_path):
    hm = simulate_hm(n_samples=16, seq_length=80_000, seed=31,
                     mutation_model="binary")
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "ibs.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


@pytest.fixture
def two_pop_vcz_store(tmp_path):
    hm = simulate_hm(n_samples=18, seq_length=80_000, seed=37,
                     mutation_model="binary")
    n_dip = hm.num_haplotypes // 2
    hm.samples = [f"s{i}" for i in range(n_dip)]
    path = str(tmp_path / "ibs_twopop.vcz")
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


class TestIbsStreaming:

    def test_haplotype_parity(self, vcz_store):
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        e = ibs(eager)
        s = ibs(stream)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    def test_genotype_parity(self, vcz_store):
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000)
        e = ibs(eager)
        s = ibs(stream)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    def test_missing_data_exclude(self, vcz_store):
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        e = ibs(eager, missing_data='exclude')
        s = ibs(stream, missing_data='exclude')
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    def test_population_subset(self, two_pop_vcz_store):
        path, popfile = two_pop_vcz_store
        eager = HaplotypeMatrix.from_zarr(path, streaming="never",
                                            pop_file=popfile)
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                             pop_file=popfile,
                                             chunk_bp=10_000)
        e = ibs(eager, population='pop1')
        s = ibs(stream, population='pop1')
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("block_size", [1, 4, 1000])
    def test_block_size_invariance(self, vcz_store, block_size):
        # Streaming output must not depend on the row-block tile size
        # -- only the (block_size, n_ind) working memory does.
        from pg_gpu.relatedness import _stream_ibs
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        eager = HaplotypeMatrix.from_zarr(vcz_store, streaming="never")
        e = ibs(eager)
        s = _stream_ibs(stream, population=None, missing_data='include',
                        block_size=block_size)
        np.testing.assert_allclose(s, e, rtol=1e-9, atol=1e-12)

    def test_diagonal_is_one(self, vcz_store):
        # Eager and streaming both pin the diagonal at 1.0 -- check the
        # streaming path keeps that invariant.
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        s = ibs(stream)
        np.testing.assert_allclose(np.diag(s), 1.0)
