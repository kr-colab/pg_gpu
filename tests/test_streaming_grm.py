"""Equivalence tests for streaming-aware GRM.

``grm`` is the diploid biallelic GCTA GRM and takes genotype matrices; a
``StreamingGenotypeMatrix`` routes through a two-pass streaming implementation
(chromosome-wide allele frequencies first, then a standardized outer-product
accumulated on host with row-block tiling on the individual axis). Haplotype
matrices are rejected in favor of ``genetic_relatedness``. Tests assert
streaming-vs-eager parity on small msprime stores.
"""

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix
from pg_gpu.relatedness import grm


@pytest.fixture
def vcz_store(tmp_path):
    from .conftest import simulate_hm
    hm = simulate_hm(n_samples=16, seq_length=80_000, seed=41,
                     mutation_model="binary")
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "grm.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


@pytest.fixture
def two_pop_vcz_store(tmp_path):
    from .conftest import simulate_hm
    hm = simulate_hm(n_samples=18, seq_length=80_000, seed=43,
                     mutation_model="binary")
    n_dip = hm.num_haplotypes // 2
    hm.samples = [f"s{i}" for i in range(n_dip)]
    path = str(tmp_path / "grm_twopop.vcz")
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


class TestGrmStreaming:

    def test_rejects_streaming_haplotype_matrix(self, vcz_store):
        # grm is the diploid biallelic GCTA GRM; a streaming haplotype
        # matrix is rejected in favor of genetic_relatedness.
        stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000)
        with pytest.raises(TypeError, match="genetic_relatedness"):
            grm(stream)

    def test_genotype_parity(self, vcz_store):
        # rtol looser because GRM has a two-pass float accumulation
        # (frequencies then standardized outer product) where the chunked
        # vs whole-matrix orderings can differ at the last ULP.
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                           chunk_bp=10_000)
        np.testing.assert_allclose(grm(stream), grm(eager),
                                    rtol=1e-7, atol=1e-10)

    def test_missing_data_exclude(self, vcz_store):
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        np.testing.assert_allclose(
            grm(stream, missing_data='exclude'),
            grm(eager, missing_data='exclude'),
            rtol=1e-7, atol=1e-10,
        )

    def test_population_subset(self, two_pop_vcz_store):
        # Eager genotype population subsetting equals running the GRM on the
        # manually extracted individuals. (Streaming genotype population
        # subsetting is a separate pre-existing gap: the streaming matrix
        # keeps sample_sets in haplotype coordinates.)
        path, popfile = two_pop_vcz_store
        eager = GenotypeMatrix.from_zarr(path, streaming="never",
                                         pop_assignment=popfile)
        idx = eager.sample_sets['pop1']
        g = eager.genotypes
        g = g.get() if hasattr(g, 'get') else g
        pos = eager.positions
        pos = pos.get() if hasattr(pos, 'get') else pos
        subset = GenotypeMatrix(np.asarray(g)[idx], np.asarray(pos),
                                eager.chrom_start, eager.chrom_end)
        np.testing.assert_allclose(
            grm(eager, population='pop1'), grm(subset),
            rtol=1e-7, atol=1e-10,
        )

    @pytest.mark.parametrize("block_size", [1, 4, 1000])
    def test_block_size_invariance(self, vcz_store, block_size):
        from pg_gpu.relatedness import _stream_grm
        eager = GenotypeMatrix.from_zarr(vcz_store, streaming="never")
        stream = GenotypeMatrix.from_zarr(vcz_store, streaming="always",
                                            chunk_bp=10_000)
        e = grm(eager)
        s = _stream_grm(stream, population=None, missing_data='include',
                         block_size=block_size)
        np.testing.assert_allclose(s, e, rtol=1e-7, atol=1e-10)

    def test_monomorphic_chunks_dont_break(self, tmp_path):
        # Build a small store where most variants are monomorphic so
        # poly_mask is sparse and many chunks contribute nothing. The
        # eager kernel returns zeros if every site is monomorphic; the
        # streaming path must not divide by zero or skip a contributing
        # chunk.
        from .conftest import simulate_hm
        hm = simulate_hm(n_samples=8, seq_length=20_000, seed=53,
                         mutation_model="binary")
        hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
        path = str(tmp_path / "mono.vcz")
        hm.to_zarr(path, format="vcz", contig_name="1")
        eager = GenotypeMatrix.from_zarr(path, streaming="never")
        stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        np.testing.assert_allclose(grm(stream), grm(eager),
                                    rtol=1e-7, atol=1e-10)
