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

    def test_monomorphic_source_returns_nan_grm(self, mono_store):
        gm = GenotypeMatrix.from_zarr(mono_store, streaming="always",
                                      chunk_bp=500)
        A = relatedness.grm(gm)
        assert A.shape == (4, 4)
        assert np.all(np.isnan(A))

    def test_empty_source_returns_nan_grm(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        A = relatedness.grm(gm)
        # The all-NaN return must keep the full (n_ind, n_ind) shape, so a
        # regression in the empty-source individual count (an unhalved or
        # zero-size fallback) is caught, not masked by np.all(np.isnan(x)).
        assert A.shape == (4, 4)
        assert np.all(np.isnan(A))

    def test_pair_with_no_shared_called_snp_is_nan(self, tmp_path):
        # Same disjoint-missingness pattern as the eager-path test, through
        # the streaming two-pass path. _write_vcz takes haplotypes, so each
        # genotype dosage is expanded to the haplotype pair that reproduces
        # it on read-back (-1 -> (-1, -1), 0 -> (0, 0), 1 -> (0, 1), 2 -> (1, 1)).
        g = np.array([[0, 1, 2, -1, -1, -1],
                     [-1, -1, -1, 2, 1, 0],
                     [0, 1, 2, 2, 1, 0],
                     [2, 1, 0, 0, 1, 2]], dtype=np.int8)
        pairs = {-1: (-1, -1), 0: (0, 0), 1: (0, 1), 2: (1, 1)}
        n_ind, n_var = g.shape
        hap = np.empty((2 * n_ind, n_var), dtype=np.int8)
        for i in range(n_ind):
            for j in range(n_var):
                hap[2 * i, j], hap[2 * i + 1, j] = pairs[int(g[i, j])]
        store = _write_vcz(hap, str(tmp_path / "disjoint_geno.vcz"))
        gm = GenotypeMatrix.from_zarr(store, streaming="always", chunk_bp=500)
        A = relatedness.grm(gm)
        assert np.isnan(A[0, 1]) and np.isnan(A[1, 0])
        np.testing.assert_allclose(A[0, 0], A[2, 2])


class TestStreamingIbsDegenerate:

    def test_empty_source_returns_identity_diagonal(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        ibs = relatedness._stream_ibs(gm, population=None,
                                      missing_data='include')
        # With no variants no pair shares a called site, so every
        # off-diagonal entry is undefined.
        off_diagonal = ibs[~np.eye(ibs.shape[0], dtype=bool)]
        assert np.all(np.diag(ibs) == 1.0)
        assert np.all(np.isnan(off_diagonal))

    def test_empty_source_with_population(self, poly_store):
        gm = GenotypeMatrix.from_zarr(poly_store, streaming="always",
                                      chunk_bp=500, region=_EMPTY_REGION)
        gm.sample_sets = {"pop1": [0, 1]}
        ibs = relatedness._stream_ibs(gm, population="pop1",
                                      missing_data='include')
        assert ibs.shape == (2, 2)
        off_diagonal = ibs[~np.eye(2, dtype=bool)]
        assert np.all(np.diag(ibs) == 1.0)
        assert np.all(np.isnan(off_diagonal))


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


class TestStreamingGeneticRelatednessMissingData:

    def test_pair_with_no_shared_called_site_is_nan(self, tmp_path):
        # Same disjoint-missingness pattern as the eager-path test: haplotypes
        # 0 and 1 are each called at a disjoint half of the sites, so they
        # share no called site and must get NaN through the streaming
        # two-pass path too, not a coincidental 0.0.
        hap = np.array([[0, 0, 0, -1, -1, -1],
                       [-1, -1, -1, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0]], dtype=np.int8)
        store = _write_vcz(hap, str(tmp_path / "disjoint.vcz"))
        sh = HaplotypeMatrix.from_zarr(store, streaming="always", chunk_bp=500)
        R = relatedness.genetic_relatedness(sh, centre=False,
                                            span_normalize=False)
        assert np.isnan(R[0, 1]) and np.isnan(R[1, 0])
        # Rescaled by n_cols / J (its own jointly-called column count out of
        # 12), same as the eager-path test's hand-derivation.
        expected = np.array([[6, np.nan, 6, 0],
                             [np.nan, 6, 6, 0],
                             [6, 6, 6, 0],
                             [0, 0, 0, 6]], dtype=np.float64)
        mask = ~np.isnan(expected)
        np.testing.assert_allclose(R[mask], expected[mask])

    def test_no_column_at_all_zero_unless_proportion(self, tmp_path):
        # One haplotype missing everywhere -> 'exclude' drops every site, an
        # empty analysis: 0.0 (a sum over no sites), except under
        # proportion=True, where the segregating-site denominator is also 0
        # (0/0, undefined).
        rng = np.random.default_rng(0)
        hap = rng.integers(0, 2, size=(8, 20)).astype(np.int8)
        hap[0, :] = -1
        store = _write_vcz(hap, str(tmp_path / "no_complete_site.vcz"))
        sh = HaplotypeMatrix.from_zarr(store, streaming="always", chunk_bp=500)

        R = relatedness.genetic_relatedness(sh, missing_data='exclude',
                                            span_normalize=False)
        assert np.all(R == 0.0)
        Rp = relatedness.genetic_relatedness(sh, missing_data='exclude',
                                             proportion=True)
        assert np.all(np.isnan(Rp))
