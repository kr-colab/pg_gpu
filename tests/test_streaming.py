"""Tests for StreamingHaplotypeMatrix + HostChunkFetcher."""

import numpy as np
import pytest
import cupy as cp

from pg_gpu import HaplotypeMatrix
from pg_gpu.streaming_matrix import (
    ChunkFetcher, HostChunkFetcher, StreamingHaplotypeMatrix,
)
from pg_gpu.zarr_source import ZarrGenotypeSource

from .conftest import simulate_hm


def _simulate_hm(n_samples=20, seq_length=50_000, seed=42):
    return simulate_hm(n_samples=n_samples, seq_length=seq_length, seed=seed)


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

    def test_pop_assignment_yields_individual_row_sets(self, vcz_store,
                                                        tmp_path):
        """The stream stored pop-file sets in haplotype coordinates while
        its chunks hold individual rows, so chunk construction raised once
        sample_sets grew validation."""
        from pg_gpu import GenotypeMatrix
        path, hm = vcz_store
        n_dip = hm.num_haplotypes // 2
        half = n_dip // 2
        popfile = tmp_path / "pops.tsv"
        lines = ["sample\tpop"] + [
            f"s{i}\t{'p1' if i < half else 'p2'}" for i in range(n_dip)]
        popfile.write_text("\n".join(lines) + "\n")

        stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                          chunk_bp=5_000,
                                          pop_assignment=str(popfile))
        sets = stream.sample_sets
        assert sorted(int(i) for i in sets['p1']) == list(range(half))
        assert sorted(int(i) for i in sets['p2']) == list(range(half, n_dip))
        for _, _, chunk in stream.iter_gpu_chunks():
            assert max(int(i) for i in chunk.sample_sets['p2']) < n_dip
            break

    def test_grm_runs_with_pop_assignment(self, vcz_store, tmp_path):
        from pg_gpu import GenotypeMatrix, relatedness
        path, hm = vcz_store
        n_dip = hm.num_haplotypes // 2
        popfile = tmp_path / "pops.tsv"
        lines = ["sample\tpop"] + [f"s{i}\tp1" for i in range(n_dip)]
        popfile.write_text("\n".join(lines) + "\n")
        stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                          chunk_bp=5_000,
                                          pop_assignment=str(popfile))
        g = relatedness.grm(stream)
        assert g.shape == (n_dip, n_dip)

    def test_materialize_subset_speaks_individual_rows(self, vcz_store,
                                                       tmp_path):
        """A genotype stream's subset and its sample_sets share the
        individual row space, so feeding a population's own rows back as
        the subset is the supported idiom. Both used to disagree: the
        sets moved to individual rows while the subset still meant
        haplotype columns, silently selecting the wrong samples."""
        from pg_gpu import GenotypeMatrix
        path, hm = vcz_store
        n_dip = hm.num_haplotypes // 2
        half = n_dip // 2
        popfile = tmp_path / "pops.tsv"
        lines = ["sample\tpop"] + [
            f"s{i}\t{'p1' if i < half else 'p2'}" for i in range(n_dip)]
        popfile.write_text("\n".join(lines) + "\n")
        stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                          chunk_bp=5_000,
                                          pop_assignment=str(popfile))

        # The documented idiom: a population's rows as the subset.
        m = stream.materialize(sample_subset=list(stream.sample_sets['p2']))
        assert m.num_individuals == n_dip - half
        assert sorted(int(i) for i in m.sample_sets['p2']) == list(
            range(n_dip - half))
        assert 'p1' not in m.sample_sets
        for rows in m.sample_sets.values():
            assert max(int(i) for i in rows) < m.num_individuals

    def test_materialize_subset_is_validated(self, vcz_store, tmp_path):
        """The subset obeys the same rules as a sample_sets value."""
        from pg_gpu import GenotypeMatrix
        path, hm = vcz_store
        stream_h = HaplotypeMatrix.from_zarr(path, streaming="always",
                                             chunk_bp=5_000)
        with pytest.raises(ValueError, match="duplicate"):
            stream_h.materialize(sample_subset=[0, 1, 0, 1])
        with pytest.raises(ValueError, match="rows 0"):
            stream_h.materialize(sample_subset=[0, 10**6])
        stream_g = GenotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        with pytest.raises(ValueError, match="duplicate"):
            stream_g.materialize(sample_subset=[0, 0, 2, 3])

    def test_stream_setter_validates_in_own_row_space(self, vcz_store):
        """Assigning sample_sets on a stream validates like the eager
        classes; the bare assignment used to carry garbage into every
        chunk and out through materialize."""
        from pg_gpu import GenotypeMatrix
        path, hm = vcz_store
        n_dip = hm.num_haplotypes // 2
        stream_h = HaplotypeMatrix.from_zarr(path, streaming="always",
                                             chunk_bp=5_000)
        with pytest.raises(ValueError, match="rows 0"):
            stream_h.sample_sets = {'p': [0, 10**6]}
        stream_h.sample_sets = {'p': [0, 1, 2, 3]}

        stream_g = GenotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        # Individual rows: n_dip is one past the end for this stream.
        with pytest.raises(ValueError, match=f"rows 0..{n_dip - 1}"):
            stream_g.sample_sets = {'p': [0, n_dip]}
        stream_g.sample_sets = {'p': [0, 1]}

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

    @pytest.mark.parametrize("chunk_bp", [1_000, 5_000, 25_000, 200_000])
    @pytest.mark.parametrize("prefetch", [0, 1, 4])
    def test_stream_concat_matches_eager(self, vcz_store, chunk_bp, prefetch):
        # walking every chunk in streaming mode and concatenating the
        # per-chunk haplotype matrices reproduces the eager matrix
        # bit-for-bit -- the invariant streaming-aware kernels rely on.
        # Sweep chunk_bp from one-chunk-per-variant (1 kb) up through
        # chunks that span the whole fixture (200 kb >> 100 kb store
        # length) so the chunk-count edge cases are all exercised, and
        # sweep prefetch off / on / deeper so the producer thread's
        # ordering is checked under different queue depths.
        path, _ = vcz_store
        eager = HaplotypeMatrix.from_zarr(path, streaming="never")
        smatrix = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=chunk_bp,
                                            prefetch=prefetch)
        haps, pos = _stream_concat(smatrix)
        np.testing.assert_array_equal(haps, cp.asnumpy(eager.haplotypes))
        np.testing.assert_array_equal(pos, cp.asnumpy(eager.positions))

    @pytest.mark.parametrize("chunk_bp", [1_000, 5_000, 25_000])
    def test_prefetch_off_matches_prefetch_on(self, vcz_store, chunk_bp):
        path, _ = vcz_store
        s0 = HaplotypeMatrix.from_zarr(path, streaming="always",
                                       chunk_bp=chunk_bp, prefetch=0)
        s1 = HaplotypeMatrix.from_zarr(path, streaming="always",
                                       chunk_bp=chunk_bp, prefetch=1)
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
        choice, source = _decide_streaming_mode(
            path, region=None, streaming="auto", pop_assignment=False,
            free_gpu_bytes=int(eager_bytes / 0.5 - 1),
        )
        assert choice == "streaming"
        # source is returned so _build_streaming can reuse it instead of
        # re-opening the zarr store.
        assert source is not None

    def test_never_raises_when_oversized(self, vcz_store):
        from pg_gpu.haplotype_matrix import _decide_streaming_mode
        path, hm = vcz_store
        eager_bytes = hm.haplotypes.size
        with pytest.raises(MemoryError, match="streaming='never'"):
            _decide_streaming_mode(
                path, region=None, streaming="never", pop_assignment=False,
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
        # -> pairwise_r2(). Asserts the composition produces an (n_var x n_var)
        # matrix with the kernel's diagonal zeroed. The msprime fixture uses the
        # default Jukes-Cantor model, so it has multiallelic sites: pairwise_r2 is
        # biallelic-only and returns NaN for their rows/cols, while the biallelic
        # pairs stay finite. The composition is what this test is for; the r2
        # numerics are validated elsewhere.
        path, _ = vcz_store
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        sub = stream.materialize(region=(10_000, 30_000))
        r2 = sub.pairwise_r2()
        n_var = sub.haplotypes.shape[1]
        assert r2.shape == (n_var, n_var)
        assert bool(cp.isfinite(r2).any()), "no finite biallelic r2 values"
        assert float(cp.abs(cp.diag(r2)).sum()) == 0.0

    def test_sample_subset_requires_even(self, vcz_store):
        path, _ = vcz_store
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                            chunk_bp=5_000)
        with pytest.raises(ValueError, match="even count"):
            stream.materialize(region=(10_000, 20_000),
                               sample_subset=[0, 1, 2])  # odd


class TestMaterializeSampleSetRenumbering:
    """materialize(sample_subset=...) renumbers sample_sets into the
    subset; it used to carry full-matrix row numbers onto a smaller
    matrix."""

    def _stream_with_pops(self, vcz_store, tmp_path):
        path, hm = vcz_store
        n_dip = hm.num_haplotypes // 2
        half = n_dip // 2
        popfile = tmp_path / "pops.tsv"
        lines = ["sample\tpop"] + [
            f"s{i}\t{'p1' if i < half else 'p2'}" for i in range(n_dip)]
        popfile.write_text("\n".join(lines) + "\n")
        stream = HaplotypeMatrix.from_zarr(path, streaming="always",
                                           chunk_bp=5_000,
                                           pop_assignment=str(popfile))
        return stream, half

    def test_subset_sets_are_renumbered(self, vcz_store, tmp_path):
        stream, half = self._stream_with_pops(vcz_store, tmp_path)
        # Samples 0 and 1 from p1, sample `half` from p2, whole samples.
        sub = [0, 1, 2, 3, 2 * half, 2 * half + 1]
        m = stream.materialize(sample_subset=sub)
        assert m.num_haplotypes == 6
        assert sorted(int(i) for i in m.sample_sets['p1']) == [0, 1, 2, 3]
        assert sorted(int(i) for i in m.sample_sets['p2']) == [4, 5]

    def test_population_absent_from_subset_is_dropped(self, vcz_store,
                                                      tmp_path):
        stream, half = self._stream_with_pops(vcz_store, tmp_path)
        m = stream.materialize(sample_subset=[0, 1, 2, 3])
        assert 'p2' not in m.sample_sets
        assert sorted(int(i) for i in m.sample_sets['p1']) == [0, 1, 2, 3]


class TestChunkFetcherABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            ChunkFetcher()


class TestStreamingGenotypeMatrix:
    """StreamingGenotypeMatrix mirrors StreamingHaplotypeMatrix's shape but
    yields per-chunk GenotypeMatrix instances (dosage-coded, n_indiv x
    n_var) instead of haplotype-coded. Sample sets index the diploid
    axis (0..n_indiv) rather than the haplotype axis."""

    def test_streaming_always_returns_streaming_class(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        from pg_gpu.streaming_matrix import StreamingGenotypeMatrix
        path, _ = vcz_store
        gm = GenotypeMatrix.from_zarr(path, streaming="always",
                                       chunk_bp=5_000)
        assert isinstance(gm, StreamingGenotypeMatrix)

    def test_chunk_payload_is_genotype_matrix(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        path, _ = vcz_store
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        for left, right, chunk_gm in gm_stream.iter_gpu_chunks():
            assert isinstance(chunk_gm, GenotypeMatrix)
            # GenotypeMatrix layout: (n_indiv, n_var) with dosage values
            assert chunk_gm.genotypes.shape[0] == gm_stream.num_individuals
            break  # one chunk is enough to verify the contract

    def test_sample_sets_default_to_individual_axis(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        path, _ = vcz_store
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        sets = gm_stream.sample_sets
        assert set(sets.keys()) == {"all"}
        # Genotype matrix indexes individuals, not haplotypes -- length
        # should match num_individuals not 2*num_individuals.
        assert len(sets["all"]) == gm_stream.num_individuals

    def test_genotypes_property_raises(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        path, _ = vcz_store
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        with pytest.raises(NotImplementedError, match="materialized"):
            _ = gm_stream.genotypes

    def test_materialize_returns_eager_genotype_matrix(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        path, _ = vcz_store
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        eager = gm_stream.materialize(region=(0, 10_000))
        assert isinstance(eager, GenotypeMatrix)
        # genotypes are (n_indiv, n_var) dosage int8
        assert eager.genotypes.shape[0] == gm_stream.num_individuals

    def test_grm_streams(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        from pg_gpu.relatedness import grm
        path, _ = vcz_store
        gm_eager = GenotypeMatrix.from_zarr(path, streaming="never")
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        # grm uses a two-pass streaming form (per-variant frequencies on
        # the first pass, standardized outer product on the second) and
        # tiles the individual axis the same way ibs does.
        np.testing.assert_allclose(grm(gm_stream), grm(gm_eager),
                                    rtol=1e-7, atol=1e-10)

    def test_ibs_streams(self, vcz_store):
        from pg_gpu import GenotypeMatrix
        from pg_gpu.relatedness import ibs
        path, _ = vcz_store
        gm_eager = GenotypeMatrix.from_zarr(path, streaming="never")
        gm_stream = GenotypeMatrix.from_zarr(path, streaming="always",
                                              chunk_bp=5_000)
        # ibs streams the variant axis chunk-by-chunk and tiles the
        # individual axis into row blocks. Result must match eager.
        np.testing.assert_allclose(ibs(gm_stream), ibs(gm_eager),
                                    rtol=1e-9, atol=1e-12)
