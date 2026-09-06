"""Tests for pg_gpu.relatedness (GRM and IBS)."""

import numpy as np
import pytest
from pg_gpu import GenotypeMatrix, HaplotypeMatrix, relatedness


@pytest.fixture
def small_haplotype_matrix():
    """Small test dataset: 6 haplotypes (3 diploids), 20 variants."""
    np.random.seed(42)
    hap = np.random.randint(0, 2, size=(6, 20)).astype(np.int8)
    positions = np.arange(20) * 1000
    return HaplotypeMatrix(hap, positions)


@pytest.fixture
def small_genotype_matrix():
    """Small diploid genotype dataset: 3 individuals, 20 variants (0/1/2)."""
    rng = np.random.RandomState(42)
    geno = rng.randint(0, 3, size=(3, 20)).astype(np.int8)
    positions = np.arange(20) * 1000
    return GenotypeMatrix(geno, positions)


def _reference_grm(g):
    """GCTA GRM from a diploid genotype array (0/1/2), numpy reference."""
    g = g.astype(float)
    p = g.mean(axis=0) / 2
    poly = (p > 0) & (p < 1)
    g_p = g[:, poly]
    p_p = p[poly]
    centered = g_p - 2 * p_p
    scale = np.sqrt(2 * p_p * (1 - p_p))
    std = centered / scale
    return (std @ std.T) / poly.sum()


def _reference_ibs(g):
    """Biallelic diploid IBS from a genotype array (0/1/2), numpy reference.

    Per jointly-called site the shared-allele count is 2 - abs(g_i - g_j); IBS
    is the summed shared count over jointly-called sites divided by twice that
    site count.
    """
    n_ind = g.shape[0]
    ibs_mat = np.eye(n_ind)
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            valid = (g[i] >= 0) & (g[j] >= 0)
            n_joint = int(valid.sum())
            if n_joint == 0:
                continue
            shared = (2 - np.abs(g[i] - g[j]))[valid].sum()
            val = shared / (2 * n_joint)
            ibs_mat[i, j] = val
            ibs_mat[j, i] = val
    return ibs_mat


def _geno_of(gm):
    g = gm.genotypes
    return g.get() if hasattr(g, 'get') else g


class TestGRM:
    def test_shape(self, small_genotype_matrix):
        grm = relatedness.grm(small_genotype_matrix)
        assert grm.shape == (3, 3)

    def test_symmetric(self, small_genotype_matrix):
        grm = relatedness.grm(small_genotype_matrix)
        np.testing.assert_allclose(grm, grm.T)

    def test_matches_reference(self, small_genotype_matrix):
        grm_pg = relatedness.grm(small_genotype_matrix)
        grm_ref = _reference_grm(_geno_of(small_genotype_matrix))
        np.testing.assert_allclose(grm_pg, grm_ref, atol=1e-10)

    def test_identical_individuals(self):
        """Two identical individuals should have GRM off-diagonal = diagonal."""
        geno = np.array([[0, 1, 2, 1, 0],
                         [0, 1, 2, 1, 0],   # identical to individual 0
                         [2, 1, 0, 1, 2]], dtype=np.int8)
        gm = GenotypeMatrix(geno, np.arange(5) * 100)
        grm = relatedness.grm(gm)
        np.testing.assert_allclose(grm[0, 1], grm[0, 0], atol=1e-10)

    def test_returns_numpy(self, small_genotype_matrix):
        grm = relatedness.grm(small_genotype_matrix)
        assert isinstance(grm, np.ndarray)

    def test_rejects_haplotype_matrix(self, small_haplotype_matrix):
        with pytest.raises(TypeError, match="genetic_relatedness"):
            relatedness.grm(small_haplotype_matrix)

    def test_all_monomorphic_returns_zeros(self):
        # No polymorphic sites -> nothing to standardize and accumulate, so
        # the GRM is all zeros rather than a divide-by-zero.
        gm = GenotypeMatrix(np.zeros((3, 5), dtype=np.int8), np.arange(5) * 100)
        grm = relatedness.grm(gm)
        assert grm.shape == (3, 3)
        np.testing.assert_array_equal(grm, np.zeros((3, 3)))


class TestIBS:
    def test_shape(self, small_genotype_matrix):
        ibs_mat = relatedness.ibs(small_genotype_matrix)
        assert ibs_mat.shape == (3, 3)

    def test_diagonal_is_one(self, small_genotype_matrix):
        ibs_mat = relatedness.ibs(small_genotype_matrix)
        np.testing.assert_allclose(ibs_mat.diagonal(), 1.0)

    def test_symmetric(self, small_genotype_matrix):
        ibs_mat = relatedness.ibs(small_genotype_matrix)
        np.testing.assert_allclose(ibs_mat, ibs_mat.T)

    def test_range(self, small_genotype_matrix):
        ibs_mat = relatedness.ibs(small_genotype_matrix)
        assert np.all(ibs_mat >= 0)
        assert np.all(ibs_mat <= 1)

    def test_matches_reference(self, small_genotype_matrix):
        ibs_pg = relatedness.ibs(small_genotype_matrix)
        np.testing.assert_allclose(ibs_pg, _reference_ibs(_geno_of(small_genotype_matrix)),
                                   atol=1e-10)

    def test_missing_matches_reference(self):
        rng = np.random.RandomState(4)
        geno = rng.randint(0, 3, size=(5, 40)).astype(np.int8)
        geno[rng.random(geno.shape) < 0.1] = -1
        gm = GenotypeMatrix(geno, np.arange(40) * 100)
        np.testing.assert_allclose(relatedness.ibs(gm), _reference_ibs(geno),
                                   atol=1e-10)

    def test_identical_individuals(self):
        geno = np.array([[0, 1, 2, 1],
                         [0, 1, 2, 1],   # identical to individual 0
                         [2, 1, 0, 1]], dtype=np.int8)
        gm = GenotypeMatrix(geno, np.arange(4) * 100)
        ibs_mat = relatedness.ibs(gm)
        assert ibs_mat[0, 1] == 1.0

    def test_completely_different(self):
        # individual 0 = 0/0 everywhere, individual 1 = 1/1 everywhere.
        geno = np.array([[0, 0, 0],
                         [2, 2, 2]], dtype=np.int8)
        gm = GenotypeMatrix(geno, np.arange(3) * 100)
        ibs_mat = relatedness.ibs(gm)
        assert ibs_mat[0, 1] == 0.0

    def test_returns_numpy(self, small_genotype_matrix):
        ibs_mat = relatedness.ibs(small_genotype_matrix)
        assert isinstance(ibs_mat, np.ndarray)

    def test_rejects_haplotype_matrix(self, small_haplotype_matrix):
        with pytest.raises(TypeError, match="genetic_relatedness"):
            relatedness.ibs(small_haplotype_matrix)


def _reference_relatedness_columns(hap, sample_sets, n_alleles):
    """Independent host reference for _relatedness_columns (complete or missing).

    Returns (C, col_site): per-set per-allele frequencies centered by the
    unweighted across-set mean, columns ordered site-major then allele-minor.
    """
    n_hap, n_var = hap.shape
    valid = hap >= 0
    sets = ([[h] for h in range(n_hap)] if sample_sets is None
            else [list(s) for s in sample_sets])
    covered = np.zeros(n_hap, dtype=bool)
    for s in sets:
        covered[s] = True
    p_cols, site_cols = [], []
    for s_idx in range(n_var):
        for a in range(n_alleles):
            if np.any(covered & (hap[:, s_idx] == a) & valid[:, s_idx]):
                col = []
                for st in sets:
                    nv = int(valid[st, s_idx].sum())
                    cnt = int(((hap[st, s_idx] == a) & valid[st, s_idx]).sum())
                    col.append(cnt / nv if nv > 0 else 0.0)
                p_cols.append(np.array(col))
                site_cols.append(s_idx)
    D = len(sets)
    P = np.stack(p_cols, axis=1) if p_cols else np.zeros((D, 0))
    C = P - P.mean(axis=0, keepdims=True)
    return C, np.array(site_cols, dtype=np.int64)


class TestRelatednessColumns:
    """Per-set per-allele centered frequency plumbing for genetic_relatedness."""

    def _run(self, hap, sample_sets):
        import cupy as cp
        k = int(hap.max()) + 1 if hap.size else 1
        C, col_site, _ = relatedness._relatedness_columns(
            cp.asarray(hap), sample_sets, k)
        ref_C, ref_site = _reference_relatedness_columns(hap, sample_sets, k)
        return C.get(), col_site.get(), ref_C, ref_site

    def test_singleton_biallelic(self):
        rng = np.random.RandomState(0)
        hap = rng.randint(0, 2, size=(8, 30)).astype(np.int8)
        C, site, ref_C, ref_site = self._run(hap, None)
        np.testing.assert_array_equal(site, ref_site)
        np.testing.assert_allclose(C, ref_C)
        assert C.shape[0] == 8  # one row per haplotype

    def test_singleton_multiallelic(self):
        rng = np.random.RandomState(1)
        hap = rng.randint(0, 3, size=(8, 40)).astype(np.int8)
        C, site, ref_C, ref_site = self._run(hap, None)
        np.testing.assert_array_equal(site, ref_site)
        np.testing.assert_allclose(C, ref_C)

    def test_columns_are_mean_centered(self):
        # Each column is centered by the across-set mean -> column sums ~ 0.
        rng = np.random.RandomState(2)
        hap = rng.randint(0, 3, size=(6, 25)).astype(np.int8)
        C, *_ = self._run(hap, None)
        np.testing.assert_allclose(C.sum(axis=0), 0.0, atol=1e-9)

    def test_grouped_sample_sets_multiallelic(self):
        # Two sample sets of unequal size; centering is the unweighted mean of
        # the two set frequencies (not the pooled frequency).
        rng = np.random.RandomState(3)
        hap = rng.randint(0, 3, size=(10, 30)).astype(np.int8)
        sample_sets = [np.arange(0, 4), np.arange(4, 10)]
        C, site, ref_C, ref_site = self._run(hap, sample_sets)
        np.testing.assert_array_equal(site, ref_site)
        np.testing.assert_allclose(C, ref_C)
        assert C.shape[0] == 2

    def test_subset_of_samples_restricts_present_alleles(self):
        # An allele carried only by an excluded sample must not create a column.
        hap = np.array([[0], [0], [1], [2]], dtype=np.int8)  # 4 haps, 1 site
        sample_sets = [np.array([0, 1, 2])]  # excludes hap 3 (the only allele-2)
        C, site, ref_C, ref_site = self._run(hap, sample_sets)
        np.testing.assert_array_equal(site, ref_site)
        np.testing.assert_allclose(C, ref_C)
        # alleles 0 and 1 present among {0,1,2}; allele 2 excluded -> 2 columns
        assert C.shape[1] == 2

    def test_relatedness_matrix_symmetric_psd_ish(self):
        # C @ C.T (the genetic_relatedness matrix) is symmetric with a
        # non-negative diagonal.
        rng = np.random.RandomState(4)
        hap = rng.randint(0, 3, size=(8, 50)).astype(np.int8)
        C, *_ = self._run(hap, None)
        R = C @ C.T
        np.testing.assert_allclose(R, R.T, atol=1e-9)
        assert bool((np.diag(R) >= -1e-9).all())


class TestGeneticRelatedness:
    """genetic_relatedness pinned to tskit.TreeSequence.genetic_relatedness."""

    def _tskit_matrix(self, ts, sample_sets, proportion=False, **kw):
        # proportion=False gives the raw centered covariance; proportion=True
        # divides by segregating sites (per-site sum of num_alleles - 1).
        D = len(sample_sets)
        idx = [(i, j) for i in range(D) for j in range(D)]
        vals = ts.genetic_relatedness(sample_sets, indexes=idx, mode='site',
                                      proportion=proportion, **kw)
        return np.asarray(vals).reshape(D, D)

    def test_singleton_pin_raw(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        tsk_sets = [[s] for s in ts.samples()]
        exp = self._tskit_matrix(ts, tsk_sets, centre=True, polarised=False,
                                 span_normalise=False)
        got = relatedness.genetic_relatedness(hm, span_normalize=False)
        np.testing.assert_allclose(got, exp, atol=1e-9, rtol=1e-6)

    def test_singleton_pin_span_normalised(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        tsk_sets = [[s] for s in ts.samples()]
        exp = self._tskit_matrix(ts, tsk_sets, centre=True, polarised=False,
                                 span_normalise=True)
        got = relatedness.genetic_relatedness(hm, span_normalize=True)
        np.testing.assert_allclose(got, exp, atol=1e-12, rtol=1e-6)

    def test_grouped_sample_sets_pin(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        samples = list(ts.samples())
        n = len(samples)
        half = n // 2
        pg_sets = [list(range(0, half)), list(range(half, n))]
        tsk_sets = [[samples[i] for i in grp] for grp in pg_sets]
        exp = self._tskit_matrix(ts, tsk_sets, centre=True, polarised=False,
                                 span_normalise=False)
        got = relatedness.genetic_relatedness(hm, sample_sets=pg_sets,
                                              span_normalize=False)
        np.testing.assert_allclose(got, exp, atol=1e-9, rtol=1e-6)

    def test_noncentred_pin(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        tsk_sets = [[s] for s in ts.samples()]
        exp = self._tskit_matrix(ts, tsk_sets, centre=False, polarised=False,
                                 span_normalise=False)
        got = relatedness.genetic_relatedness(hm, centre=False,
                                              span_normalize=False)
        np.testing.assert_allclose(got, exp, atol=1e-9, rtol=1e-6)

    def test_polarised_pin(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        tsk_sets = [[s] for s in ts.samples()]
        exp = self._tskit_matrix(ts, tsk_sets, centre=True, polarised=True,
                                 span_normalise=False)
        got = relatedness.genetic_relatedness(hm, polarised=True,
                                              span_normalize=False)
        np.testing.assert_allclose(got, exp, atol=1e-9, rtol=1e-6)

    def test_proportion_pin(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        tsk_sets = [[s] for s in ts.samples()]
        exp = self._tskit_matrix(ts, tsk_sets, proportion=True, centre=True,
                                 polarised=False, span_normalise=False)
        got = relatedness.genetic_relatedness(hm, proportion=True)
        np.testing.assert_allclose(got, exp, atol=1e-12, rtol=1e-6)

    def test_proportion_ignores_span_normalize(self, multiallelic_hm):
        _, hm = multiallelic_hm
        a = relatedness.genetic_relatedness(hm, proportion=True,
                                            span_normalize=False)
        b = relatedness.genetic_relatedness(hm, proportion=True,
                                            span_normalize=True)
        np.testing.assert_allclose(a, b)

    def test_indexes_selection(self, multiallelic_hm):
        _, hm = multiallelic_hm
        full = relatedness.genetic_relatedness(hm, span_normalize=False)
        pairs = [(0, 1), (2, 3), (0, 0)]
        sub = relatedness.genetic_relatedness(hm, indexes=pairs,
                                              span_normalize=False)
        np.testing.assert_allclose(sub, [full[i, j] for i, j in pairs])

    def test_missing_include_matches_host(self):
        rng = np.random.RandomState(5)
        hap = rng.randint(0, 3, size=(10, 40)).astype(np.int8)
        hap[rng.random(hap.shape) < 0.1] = -1
        hm = HaplotypeMatrix(hap, np.arange(40) * 100)
        got = relatedness.genetic_relatedness(hm, span_normalize=False)
        k = int(hap.max()) + 1
        Cref, _ = _reference_relatedness_columns(hap, None, k)
        np.testing.assert_allclose(got, Cref @ Cref.T, atol=1e-9)

    def test_missing_exclude_matches_host(self):
        rng = np.random.RandomState(6)
        hap = rng.randint(0, 3, size=(10, 40)).astype(np.int8)
        hap[rng.random(hap.shape) < 0.1] = -1
        hm = HaplotypeMatrix(hap, np.arange(40) * 100)
        got = relatedness.genetic_relatedness(hm, missing_data='exclude',
                                              span_normalize=False)
        complete = (hap >= 0).all(axis=0)
        k = int(hap.max()) + 1
        Cref, _ = _reference_relatedness_columns(hap[:, complete], None, k)
        np.testing.assert_allclose(got, Cref @ Cref.T, atol=1e-9)
