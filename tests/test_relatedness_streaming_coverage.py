"""Degenerate-source coverage for the streaming relatedness path.

grm / ibs / genetic_relatedness route a streaming matrix through a separate
two-pass implementation. The parity tests use polymorphic, non-empty stores,
so these drive the branches they never reach: an empty streaming source (a
region with no variants), a monomorphic source (no polymorphic site), the
population fallback for the individual count, and the indexes= projection.
"""
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, relatedness

# A region past every variant position, so the streaming source yields no chunks.
_EMPTY_REGION = "1:900000-1000000"


def _write_vcz(hap, path):
    hap = np.asarray(hap, dtype=np.int8)
    pos = np.arange(1, hap.shape[1] + 1) * 100
    hm = HaplotypeMatrix(hap, pos, 0, hap.shape[1] * 100)
    hm.samples = [f"s{i}" for i in range(hap.shape[0] // 2)]
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


@pytest.fixture
def mono_store(tmp_path):
    # Every site is monomorphic (all reference), so no variant is polymorphic.
    return _write_vcz(np.zeros((8, 20), dtype=np.int8),
                      str(tmp_path / "mono.vcz"))


@pytest.fixture
def poly_store(tmp_path):
    rng = np.random.default_rng(0)
    return _write_vcz(rng.integers(0, 2, size=(8, 30)),
                      str(tmp_path / "poly.vcz"))


class TestStreamingGrmDegenerate:

    def test_monomorphic_source_returns_zero_grm(self, mono_store):
        gm = GenotypeMatrix.from_zarr(mono_store, streaming="always",
                                      chunk_bp=500)
        A = relatedness.grm(gm)
        assert A.shape == (4, 4)
        assert np.all(A == 0.0)

    def test_empty_source_returns_zero_grm(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        A = relatedness.grm(gm)
        # The all-zero return must keep the full (n_ind, n_ind) shape, so a
        # regression in the empty-source individual count (an unhalved or
        # zero-size fallback) is caught, not masked by np.all(x == 0).
        assert A.shape == (4, 4)
        assert np.all(A == 0.0)


class TestStreamingIbsDegenerate:

    def test_empty_source_returns_identity_diagonal(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        ibs = relatedness._stream_ibs(gm, population=None,
                                      missing_data='include')
        off_diagonal = ibs[~np.eye(ibs.shape[0], dtype=bool)]
        assert np.all(np.diag(ibs) == 1.0)
        assert np.all(off_diagonal == 0.0)

    def test_empty_source_with_population(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        gm.sample_sets = {"pop1": [0, 1]}
        ibs = relatedness._stream_ibs(gm, population="pop1",
                                      missing_data='include')
        assert ibs.shape == (2, 2)
        off_diagonal = ibs[~np.eye(2, dtype=bool)]
        assert np.all(np.diag(ibs) == 1.0)
        assert np.all(off_diagonal == 0.0)


class TestStreamingGeneticRelatednessIndexes:

    def test_indexes_project_requested_pairs(self, poly_store):
        sh = HaplotypeMatrix.from_zarr(poly_store, streaming="always",
                                       chunk_bp=500)
        # Four sample sets (a 4x4 matrix) and a mix of distinct off-diagonal
        # and diagonal pairs, so a projection that mis-maps the requested
        # row/column (or returns a diagonal element) gives a different value --
        # a single symmetric pair on a 2x2 matrix could not distinguish these.
        sets = [[0, 1], [2, 3], [4, 5], [6, 7]]
        full = np.asarray(relatedness.genetic_relatedness(sh, sample_sets=sets))
        pairs = [(0, 1), (1, 2), (2, 3), (0, 0)]
        idxed = np.asarray(relatedness.genetic_relatedness(
            sh, sample_sets=sets, indexes=pairs))
        assert idxed.shape == (len(pairs),)
        np.testing.assert_allclose(idxed, [full[i, j] for i, j in pairs])
