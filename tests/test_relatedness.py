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
                ibs_mat[i, j] = ibs_mat[j, i] = np.nan
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

    Returns (C, col_site, valid_mask): per-set per-allele frequencies centered
    by the across-set mean of the sets with a call at that site (a set with no
    call there is excluded from the mean and forced to 0, not a fake 0
    frequency), columns ordered site-major then allele-minor.
    """
    n_hap, n_var = hap.shape
    valid = hap >= 0
    sets = ([[h] for h in range(n_hap)] if sample_sets is None
            else [list(s) for s in sample_sets])
    covered = np.zeros(n_hap, dtype=bool)
    for s in sets:
        covered[s] = True
    p_cols, valid_cols, site_cols = [], [], []
    for s_idx in range(n_var):
        for a in range(n_alleles):
            if np.any(covered & (hap[:, s_idx] == a) & valid[:, s_idx]):
                col, col_valid = [], []
                for st in sets:
                    nv = int(valid[st, s_idx].sum())
                    cnt = int(((hap[st, s_idx] == a) & valid[st, s_idx]).sum())
                    col.append(cnt / nv if nv > 0 else np.nan)
                    col_valid.append(nv > 0)
                p_cols.append(np.array(col))
                valid_cols.append(np.array(col_valid))
                site_cols.append(s_idx)
    D = len(sets)
    P = np.stack(p_cols, axis=1) if p_cols else np.zeros((D, 0))
    VM = np.stack(valid_cols, axis=1) if valid_cols else np.zeros((D, 0), dtype=bool)
    pbar = np.nanmean(P, axis=0, keepdims=True) if P.shape[1] else P
    C = np.where(VM, P - pbar, 0.0)
    return C, np.array(site_cols, dtype=np.int64), VM


class TestRelatednessColumns:
    """Per-set per-allele centered frequency plumbing for genetic_relatedness."""

    def _run(self, hap, sample_sets):
        import cupy as cp
        k = int(hap.max()) + 1 if hap.size else 1
        C, col_site, _, _ = relatedness._relatedness_columns(
            cp.asarray(hap), sample_sets, k)
        ref_C, ref_site, _ = _reference_relatedness_columns(hap, sample_sets, k)
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

    def test_missing_set_forced_to_zero_and_excluded_from_mean(self):
        # Set 1 has no call at this site: its column entry must be exactly 0
        # (not a disguised frequency), and the centering mean must exclude it
        # rather than average it in as 0.
        import cupy as cp
        hap = np.array([[0], [-1], [0], [1]], dtype=np.int8)  # 4 singleton sets
        C, col_site, col_allele, valid_mask = relatedness._relatedness_columns(
            cp.asarray(hap), None, 2)
        C, col_allele, valid_mask = C.get(), col_allele.get(), valid_mask.get()
        a0 = int(np.where(col_allele == 0)[0][0])
        assert not valid_mask[1, a0]
        assert C[1, a0] == 0.0
        # pbar(allele 0) over the valid sets {0, 2, 3} only: (1 + 1 + 0) / 3.
        pbar = 2 / 3
        np.testing.assert_allclose(C[0, a0], 1 - pbar)
        np.testing.assert_allclose(C[2, a0], 1 - pbar)
        np.testing.assert_allclose(C[3, a0], 0 - pbar)


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
        Cref, _, VM = _reference_relatedness_columns(hap, None, k)
        # genetic_relatedness rescales each pair's raw sum by n_cols / J (its
        # own jointly-called column count), so a pair does not shrink just
        # because it called fewer columns -- see relatedness.py's own
        # n_cols/J step.
        raw = Cref @ Cref.T
        J = VM.astype(np.float64) @ VM.astype(np.float64).T
        n_cols = Cref.shape[1]
        expected = np.where(J > 0, raw * n_cols / np.maximum(J, 1.0), np.nan)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_missing_exclude_matches_host(self):
        rng = np.random.RandomState(6)
        hap = rng.randint(0, 3, size=(10, 40)).astype(np.int8)
        hap[rng.random(hap.shape) < 0.1] = -1
        hm = HaplotypeMatrix(hap, np.arange(40) * 100)
        got = relatedness.genetic_relatedness(hm, missing_data='exclude',
                                              span_normalize=False)
        complete = (hap >= 0).all(axis=0)
        k = int(hap.max()) + 1
        Cref, _, _ = _reference_relatedness_columns(hap[:, complete], None, k)
        np.testing.assert_allclose(got, Cref @ Cref.T, atol=1e-9)

    def test_pair_with_no_shared_called_site_is_nan(self):
        # Haplotypes 0 and 1 are each called at a disjoint half of the sites,
        # so they share no called site and must get NaN rather than a
        # coincidental 0.0. Haplotypes 2 and 3 are called everywhere.
        hap = np.array([[0, 0, 0, -1, -1, -1],
                        [-1, -1, -1, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.arange(6) * 10 + 1, 1, 60)

        noncentred = relatedness.genetic_relatedness(hm, centre=False,
                                                      span_normalize=False)
        assert np.isnan(noncentred[0, 1]) and np.isnan(noncentred[1, 0])
        # Every pair's raw sum is rescaled by n_cols / J (its own
        # jointly-called column count out of all 12), so a haplotype called
        # everywhere is not diluted when paired with one missing half the
        # sites: e.g. hap0-hap2 share 6 of 12 columns with 3 matches, scaled
        # to 3 * 12/6 = 6, the same as hap2's own full-coverage count.
        expected = np.array([[6, np.nan, 6, 0],
                             [np.nan, 6, 6, 0],
                             [6, 6, 6, 0],
                             [0, 0, 0, 6]], dtype=np.float64)
        mask = ~np.isnan(expected)
        np.testing.assert_allclose(noncentred[mask], expected[mask])

        centred = relatedness.genetic_relatedness(hm, span_normalize=False)
        assert np.isnan(centred[0, 1]) and np.isnan(centred[1, 0])
        # Hand-derived from the per-site centered columns (pbar excludes the
        # one set missing a call at each site), then rescaled by n_cols / J
        # the same way as the noncentred case above.
        expected_centred = np.array([[4/3, np.nan, 4/3, -8/3],
                                     [np.nan, 4/3, 4/3, -8/3],
                                     [4/3, 4/3, 4/3, -8/3],
                                     [-8/3, -8/3, -8/3, 16/3]])
        mask = ~np.isnan(expected_centred)
        np.testing.assert_allclose(centred[mask], expected_centred[mask])

    def test_self_relatedness_not_diluted_by_own_call_rate(self):
        # A haplotype missing half its calls must not look less related to
        # itself than a fully-called one: each pair's raw sum is over the
        # columns it shares, which is rescaled to the full analyzed column
        # count (n_cols / J), not left to shrink with that pair's coverage.
        rng = np.random.default_rng(0)
        n_hap, n_var = 40, 5000
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[0:20][rng.random((20, n_var)) < 0.5] = -1
        hm = HaplotypeMatrix(hap, np.arange(n_var) + 1, 1, n_var)

        R = relatedness.genetic_relatedness(hm, span_normalize=False)
        diag = np.diag(R)
        ratio = diag[:20].mean() / diag[20:].mean()
        assert abs(ratio - 1.0) < 0.05

    def test_no_column_at_all_is_zero_not_nan(self):
        # 'exclude' with no site called in every analyzed sample leaves zero
        # columns -- an empty analysis, not an undefined pair, so it is a sum
        # over no sites (0.0), not NaN.
        rng = np.random.default_rng(0)
        n_hap, n_var = 8, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[0, :] = -1  # one haplotype missing everywhere -> no complete site
        hm = HaplotypeMatrix(hap, np.arange(n_var) + 1, 1, n_var)

        R = relatedness.genetic_relatedness(hm, missing_data='exclude',
                                            span_normalize=False)
        assert np.all(R == 0.0)

    def test_no_column_at_all_is_nan_under_proportion(self):
        # Same empty analysis as above, but proportion=True divides by the
        # segregating-site count, which is also 0 here -- 0/0, undefined,
        # matching tskit's own division (out /= denominator gives nan, not
        # 0, when both are 0).
        rng = np.random.default_rng(0)
        n_hap, n_var = 8, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[0, :] = -1
        hm = HaplotypeMatrix(hap, np.arange(n_var) + 1, 1, n_var)

        Rp = relatedness.genetic_relatedness(hm, missing_data='exclude',
                                             proportion=True)
        assert np.all(np.isnan(Rp))
