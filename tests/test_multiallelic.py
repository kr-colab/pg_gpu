"""
Multiallelic-site correctness for the diversity family (issue #100).

Oracle: tskit (per-allele pi/SFS; mutation-count segregating/Watterson). Also
scikit-allel where it coincides (pi). The biallelic control confirms parity on
biallelic data. Fixtures live in conftest.py (multiallelic_ts / multiallelic_hm).
"""

import numpy as np
import allel
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu import BiallelicOnlyWarning
from pg_gpu import diversity
from pg_gpu import sfs
from pg_gpu.diversity import FrequencySpectrum


def _a1(n):
    return float(np.sum(1.0 / np.arange(1, n)))


def _strict_biallelic_mask(G):
    """Sites whose sample alleles are exactly {0, 1}."""
    return (G.max(axis=1) == 1) & (G == 0).any(axis=1) & (G == 1).any(axis=1)


def _expected_segregating(hap, exclude=False):
    """Independent numpy reference: mutation-count segregating sites.

    hap is (n_hap, n_var) with -1 for missing. ``exclude`` drops any variant
    with a missing genotype first (the missing_data='exclude' semantics).
    """
    hap = np.asarray(hap)
    total = 0
    for j in range(hap.shape[1]):
        col = hap[:, j]
        if exclude and np.any(col < 0):
            continue
        valid = col[col >= 0]
        if valid.size < 2:
            continue
        total += np.unique(valid).size - 1
    return total


def _expected_pi(hap):
    """Independent numpy reference: per-site per-allele mean pairwise difference."""
    hap = np.asarray(hap)
    total = 0.0
    for j in range(hap.shape[1]):
        valid = hap[:, j][hap[:, j] >= 0]
        n = valid.size
        if n < 2:
            continue
        _, counts = np.unique(valid, return_counts=True)
        total += 1.0 - float(np.sum(counts * (counts - 1))) / (n * (n - 1))
    return total


class TestDiversityCoreVsTskit:
    """pi / segregating / Watterson / Tajima numerator against tskit."""

    def test_fixture_is_multiallelic(self, multiallelic_hm):
        ts, _ = multiallelic_hm
        G = ts.genotype_matrix()
        n_alleles = np.array([np.unique(G[i]).size for i in range(ts.num_sites)])
        assert (n_alleles >= 3).sum() > 0, "fixture has no multiallelic sites"

    def test_pi_matches_tskit(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        pi_pg = diversity.pi(hm, span_normalize=False)
        pi_ts = float(ts.diversity(mode="site", span_normalise=False))
        np.testing.assert_allclose(pi_pg, pi_ts, rtol=1e-9)

    def test_pi_matches_allel(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        ac = allel.HaplotypeArray(ts.genotype_matrix()).count_alleles()
        pi_allel = float(np.nansum(allel.mean_pairwise_difference(ac)))
        np.testing.assert_allclose(diversity.pi(hm, span_normalize=False),
                                   pi_allel, rtol=1e-9)

    def test_segregating_matches_tskit_mutation_count(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        s_pg = diversity.segregating_sites(hm)
        s_ts = ts.segregating_sites(mode="site", span_normalise=False)
        assert s_pg == int(round(float(s_ts)))

    def test_watterson_is_mutation_count_over_a1(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        # no missing data -> n_valid == n everywhere, so theta_w = S / a1(n)
        s_ts = float(ts.segregating_sites(mode="site", span_normalise=False))
        expected = s_ts / _a1(ts.num_samples)
        np.testing.assert_allclose(diversity.theta_w(hm, span_normalize=False),
                                   expected, rtol=1e-9)

    def test_tajimas_d_matches_tskit(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        ct = diversity._compute_thetas(hm, ("pi", "watterson"))
        num_pg = ct["thetas"]["pi"] - ct["thetas"]["watterson"]
        pi_ts = float(ts.diversity(mode="site", span_normalise=False))
        s_ts = float(ts.segregating_sites(mode="site", span_normalise=False))
        num_ts = pi_ts - s_ts / _a1(ts.num_samples)
        np.testing.assert_allclose(num_pg, num_ts, rtol=1e-9)
        # Full D matches tskit exactly on complete data: tskit plugs the
        # mutation-count S straight into the classic Tajima variance
        # sqrt(a*S + (b/c)*S(S-1)), which is what pg_gpu's Achaz framework
        # reduces to for the (pi, watterson) pair with the same S.
        np.testing.assert_allclose(diversity.tajimas_d(hm),
                                   float(ts.Tajimas_D(mode="site")), rtol=1e-9)

    def test_theta_h_matches_tskit_sfs(self, multiallelic_hm):
        # oracle: tskit per-allele SFS through the theta_H weight 2*i^2/(n(n-1))
        ts, hm = multiallelic_hm
        xi = ts.allele_frequency_spectrum(polarised=True, span_normalise=False)
        n = ts.num_samples
        i = np.arange(len(xi), dtype=float)
        ref = float(np.sum(xi * 2.0 * i ** 2 / (n * (n - 1))))
        np.testing.assert_allclose(diversity.theta_h(hm, span_normalize=False),
                                   ref, rtol=1e-9)

    def test_theta_l_matches_tskit_sfs(self, multiallelic_hm):
        # oracle: tskit per-allele SFS through the theta_L weight i/(n-1)
        ts, hm = multiallelic_hm
        xi = ts.allele_frequency_spectrum(polarised=True, span_normalise=False)
        n = ts.num_samples
        i = np.arange(len(xi), dtype=float)
        ref = float(np.sum(xi * i / (n - 1)))
        np.testing.assert_allclose(diversity.theta_l(hm, span_normalize=False),
                                   ref, rtol=1e-9)


class TestBiallelicControl:
    """On strict-biallelic sites, pg_gpu matches both tskit and scikit-allel."""

    def test_biallelic_pi_and_segregating(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        G = ts.genotype_matrix()
        nonbi = np.where(~_strict_biallelic_mask(G))[0].astype(np.int32)
        ts_bi = ts.delete_sites(nonbi)
        hm_bi = hm.apply_biallelic_filter()

        # tskit oracle
        np.testing.assert_allclose(
            diversity.pi(hm_bi, span_normalize=False),
            float(ts_bi.diversity(mode="site", span_normalise=False)), rtol=1e-9)
        assert diversity.segregating_sites(hm_bi) == int(round(
            float(ts_bi.segregating_sites(mode="site", span_normalise=False))))

        # scikit-allel oracle
        ac_bi = allel.HaplotypeArray(G[_strict_biallelic_mask(G)]).count_alleles()
        np.testing.assert_allclose(
            diversity.pi(hm_bi, span_normalize=False),
            float(np.nansum(allel.mean_pairwise_difference(ac_bi))), rtol=1e-9)


class TestMissingData:
    """Missing genotypes (-1): the include per-site path and the exclude filter."""

    # cols: biallelic+missing / triallelic+missing / biallelic+missing /
    #       mostly-missing (n_valid<2) / complete triallelic / complete monomorphic
    HAP = np.array([
        [0, 0, 1, 1, -1, -1],
        [0, 1, 2, -1, -1, -1],
        [0, 0, 0, 1, -1, -1],
        [-1, -1, -1, -1, -1, 0],
        [0, 1, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.int8).T  # -> (6 haplotypes, 6 variants)

    def _hm(self):
        return HaplotypeMatrix(self.HAP.copy(), np.arange(6) * 100, 0, 600)

    def test_segregating_include(self):
        # per-site: n_valid>=2 sites counted by mutation count; the n_valid=1
        # site is dropped. Reference: 1 + 2 + 1 + 0 + 2 + 0 = 6.
        got = diversity.segregating_sites(self._hm(), missing_data="include")
        assert got == _expected_segregating(self.HAP, exclude=False) == 6

    def test_segregating_exclude(self):
        # exclude drops any variant with missing -> only the 2 complete sites
        # remain (triallelic -> 2, monomorphic -> 0). Reference: 2.
        got = diversity.segregating_sites(self._hm(), missing_data="exclude")
        assert got == _expected_segregating(self.HAP, exclude=True) == 2

    def test_pi_include_with_missing(self):
        # include mode uses per-site n_valid. Oracle 1: a numpy per-site
        # per-allele reference. Oracle 2: allel (note HAP is n_hap x n_var, so
        # transpose for allel's n_var x n_hap layout).
        pi_pg = diversity.pi(self._hm(), span_normalize=False, missing_data="include")
        np.testing.assert_allclose(pi_pg, _expected_pi(self.HAP), rtol=1e-9)
        ac = allel.HaplotypeArray(self.HAP.T).count_alleles()
        pi_allel = float(np.nansum(allel.mean_pairwise_difference(ac, fill=0)))
        np.testing.assert_allclose(pi_pg, pi_allel, rtol=1e-9)


class TestPerAlleleSFS:
    """sfs.sfs + FrequencySpectrum on the per-allele SFS."""

    def test_sfs_module_matches_tskit(self, multiallelic_hm):
        # The sfs module's unfolded SFS is per-allele and matches tskit too.
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        afs_pg = np.asarray(sfs_mod.sfs(hm))
        afs_ts = ts.allele_frequency_spectrum(polarised=True, span_normalise=False)
        np.testing.assert_array_equal(afs_pg, afs_ts.astype(int))

    def test_sfs_folded_matches_tskit(self, multiallelic_hm):
        # Folded SFS is per-allele = tskit polarised=False: every allele (ref
        # included) weight 1/2 at min(count, n-count), fixed class dropped.
        # Half-integer bins appear on multiallelic data.
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        n = ts.num_samples
        folded_pg = np.asarray(sfs_mod.sfs_folded(hm))
        folded_ts = ts.allele_frequency_spectrum(polarised=False,
                                                 span_normalise=False)
        # pg_gpu returns the compact folded domain [0, n//2]; tskit pads to n+1
        # with zeros above n//2.
        np.testing.assert_allclose(folded_pg, folded_ts[:n // 2 + 1])
        # multiallelic input really does produce half-integer weights
        assert np.any(folded_pg != np.round(folded_pg))

    def test_sfs_folded_scaled_matches_tskit_base(self, multiallelic_hm):
        # Scaled folded = tskit-pinned folded base * the k*(n-k)/n transform.
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        n = ts.num_samples
        base = np.asarray(sfs_mod.sfs_folded(hm))
        scaled = np.asarray(sfs_mod.sfs_folded_scaled(hm))
        k = np.arange(base.shape[0])
        np.testing.assert_allclose(scaled, base * k * (n - k) / n)

    def test_afs_interior_matches_allel(self, multiallelic_hm):
        # Independent-library check on the segregating interior (1..n-1).
        ts, hm = multiallelic_hm
        n = ts.num_samples
        ac = allel.HaplotypeArray(ts.genotype_matrix()).count_alleles()
        per_allele = ac[:, 1:].flatten()
        per_allele = per_allele[(per_allele > 0) & (per_allele < n)].astype(np.int64)
        ref = np.bincount(per_allele, minlength=n + 1)[:n + 1]
        afs_pg = np.asarray(sfs.sfs(hm))
        np.testing.assert_array_equal(afs_pg[1:n], ref[1:n])

    def test_afs_excludes_fixed_classes(self):
        # Both fixed classes are excluded (matches tskit): a fully-derived site
        # (bin n) and a monomorphic-ancestral site (bin 0) contribute nothing.
        derived = HaplotypeMatrix(np.array([[1], [1], [1], [1]], dtype=np.int8),
                                  np.array([100]), 0, 200)
        ancestral = HaplotypeMatrix(np.array([[0], [0], [0], [0]], dtype=np.int8),
                                    np.array([100]), 0, 200)
        assert np.asarray(sfs.sfs(derived)).sum() == 0
        assert np.asarray(sfs.sfs(ancestral)).sum() == 0

    def test_freqspec_theta_h_l_match_scalar(self, multiallelic_hm):
        # theta_h / theta_l are additive per allele, so the SFS path equals the
        # scalar path even on multiallelic data.
        _, hm = multiallelic_hm
        fs = FrequencySpectrum(hm)
        np.testing.assert_allclose(fs.theta('theta_h'),
                                   diversity.theta_h(hm, span_normalize=False), rtol=1e-9)
        np.testing.assert_allclose(fs.theta('theta_l'),
                                   diversity.theta_l(hm, span_normalize=False), rtol=1e-9)

    def test_freqspec_matches_scalar_biallelic(self, multiallelic_hm):
        # On biallelic data every estimator agrees between the two paths.
        _, hm = multiallelic_hm
        hm_bi = hm.apply_biallelic_filter()
        fs = FrequencySpectrum(hm_bi)
        for name, fn in [('pi', diversity.pi), ('watterson', diversity.theta_w),
                         ('theta_h', diversity.theta_h), ('theta_l', diversity.theta_l)]:
            np.testing.assert_allclose(fs.theta(name),
                                       fn(hm_bi, span_normalize=False), rtol=1e-9)

    def test_freqspec_pi_diverges_from_scalar_on_multiallelic(self, multiallelic_hm):
        # Documented divergence (issue #100): the marginal SFS overcounts pi on
        # multiallelic data; the scalar diversity.pi is authoritative.
        _, hm = multiallelic_hm
        pi_sfs = FrequencySpectrum(hm).theta('pi')
        pi_scalar = diversity.pi(hm, span_normalize=False)
        assert pi_sfs > pi_scalar
        assert not np.isclose(pi_sfs, pi_scalar, rtol=1e-6)


class TestJointSFS:
    """Joint / projected SFS per-allele vs tskit (two sample sets)."""

    @staticmethod
    def _pops(ts):
        s = ts.samples()
        h = len(s) // 2
        return list(s[:h]), list(s[h:])

    def test_joint_sfs_matches_tskit(self, multiallelic_hm):
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        A, B = self._pops(ts)
        j_pg = np.asarray(sfs_mod.joint_sfs(hm, A, B))
        j_ts = ts.allele_frequency_spectrum([A, B], polarised=True,
                                            span_normalise=False)
        np.testing.assert_array_equal(j_pg, j_ts.astype(int))

    def test_joint_sfs_excludes_only_global_mono_corners(self, multiallelic_hm):
        # Edge cells (alleles private to / fixed within one pop) are populated;
        # only the two global-monomorphic corners are dropped.
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        A, B = self._pops(ts)
        j = np.asarray(sfs_mod.joint_sfs(hm, A, B))
        assert j[0, 0] == 0 and j[-1, -1] == 0
        assert j[0, 1:].sum() > 0 and j[1:, 0].sum() > 0  # private-allele edges

    def test_joint_sfs_folded_matches_tskit(self, multiallelic_hm):
        # Folded joint = tskit polarised=False: fold each site as a unit by the
        # global minor (NOT allel's per-axis fold), weight 1/2, corners dropped.
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        A, B = self._pops(ts)
        jf_pg = np.asarray(sfs_mod.joint_sfs_folded(hm, A, B))
        jf_ts = ts.allele_frequency_spectrum([A, B], polarised=False,
                                             span_normalise=False)
        np.testing.assert_allclose(jf_pg, jf_ts)
        assert np.any(jf_pg != np.round(jf_pg))  # half-integers on multiallelic

    def test_joint_sfs_folded_scaled_matches_base(self, multiallelic_hm):
        from pg_gpu import sfs as sfs_mod
        ts, hm = multiallelic_hm
        A, B = self._pops(ts)
        n1, n2 = len(A), len(B)
        base = np.asarray(sfs_mod.joint_sfs_folded(hm, A, B))
        scaled = np.asarray(sfs_mod.joint_sfs_folded_scaled(hm, A, B))
        i = np.arange(base.shape[0])[:, None]
        j = np.arange(base.shape[1])[None, :]
        np.testing.assert_allclose(scaled, base * i * j * (n1 - i) * (n2 - j))

    def test_projection_matches_per_allele_sandwich(self, multiallelic_hm):
        # project_joint_sfs == P1 @ joint_sfs @ P2.T on the per-allele joint.
        from pg_gpu import sfs as sfs_mod
        from pg_gpu.diversity import _projection_matrix
        ts, hm = multiallelic_hm
        A, B = self._pops(ts)
        n1, n2 = len(A), len(B)
        t1, t2 = 8, 9
        full = np.asarray(sfs_mod.joint_sfs(hm, A, B)).astype(np.float64)
        P1 = _projection_matrix(n1, t1)
        P2 = _projection_matrix(n2, t2)
        expected = P1 @ full @ P2.T
        result = sfs_mod.project_joint_sfs(hm, A, B, target_n1=t1, target_n2=t2)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


class TestPairwiseDistanceKernel:
    """Multiallelic pairwise Hamming distance kernel (0003b) vs numpy direct."""

    @staticmethod
    def _direct(hap):
        # raw = # jointly-valid sites that differ; jv = # jointly-valid sites
        n = hap.shape[0]
        raw = np.zeros((n, n))
        jv = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                both = (hap[i] >= 0) & (hap[j] >= 0)
                jv[i, j] = both.sum()
                raw[i, j] = (both & (hap[i] != hap[j])).sum()
        return raw, jv

    def test_multiallelic_with_missing_matches_numpy(self):
        import cupy as cp
        from pg_gpu.distance_stats import _pairwise_diffs_matrix_gpu
        rng = np.random.RandomState(0)
        hap = rng.randint(0, 4, size=(8, 200)).astype(np.int8)  # 4 alleles
        hap[rng.random((8, 200)) < 0.1] = -1                     # 10% missing
        assert hap.max() > 1                                     # genuinely multiallelic
        raw_ref, jv_ref = self._direct(hap)
        norm_ref = np.where(jv_ref > 0, raw_ref / jv_ref, 0.0)
        raw = cp.asnumpy(_pairwise_diffs_matrix_gpu(hap, 'include'))
        norm = cp.asnumpy(_pairwise_diffs_matrix_gpu(hap, 'normalize'))
        np.testing.assert_allclose(raw, raw_ref)
        np.testing.assert_allclose(norm, norm_ref)
        assert np.allclose(np.diag(raw), 0)
        assert not (raw < 0).any()          # the old binary-only identity went negative

    def test_biallelic_bit_identical_to_gram_trick(self):
        import cupy as cp
        from pg_gpu.distance_stats import _pairwise_diffs_matrix_gpu
        rng = np.random.RandomState(1)
        hb = rng.randint(0, 2, size=(6, 150)).astype(np.int8)   # no missing
        old = (hb.sum(1)[:, None] + hb.sum(1)[None, :]
               - 2 * (hb.astype(float) @ hb.astype(float).T))
        new = cp.asnumpy(_pairwise_diffs_matrix_gpu(hb, 'include'))
        np.testing.assert_array_equal(new, old)


class TestDistanceStatsConsumers:
    """Distance-based stats inherit the corrected multiallelic Hamming kernel."""

    @staticmethod
    def _raw_hamming(haps):
        # raw pairwise Hamming count (fixture has no missing data)
        n = haps.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = (haps != haps[i]).sum(axis=1)
        return D

    def _setup(self, ts):
        n = ts.num_samples
        h = n // 2
        hm = HaplotypeMatrix.from_ts(ts, device="GPU")
        hm.sample_sets = {'A': list(range(h)), 'B': list(range(h, n))}
        haps = ts.genotype_matrix().T          # (n_samples, n_sites) allele indices
        # Guard the coverage claim: allele indices must exceed {0,1}, i.e. the
        # exact regime where the old binary-only kernel broke.
        assert haps.max() > 1, "fixture is not multiallelic"
        return hm, h, n, haps

    def test_pairwise_diffs_haploid_matches_numpy(self, multiallelic_ts):
        from pg_gpu import distance_stats as ds
        ts = multiallelic_ts
        hm, h, n, haps = self._setup(ts)
        got = np.asarray(ds.pairwise_diffs_haploid(hm))     # condensed, per-site avg
        D = self._raw_hamming(haps) / haps.shape[1]
        ii, jj = np.triu_indices(n, k=1)
        np.testing.assert_allclose(got, D[ii, jj], rtol=1e-9)

    def test_pairwise_distance_matrix_blocks_match_numpy(self, multiallelic_ts):
        import cupy as cp
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, h, n, haps = self._setup(ts)
        D = self._raw_hamming(haps)
        between, within1, within2 = divergence.pairwise_distance_matrix(hm, 'A', 'B')
        np.testing.assert_allclose(cp.asnumpy(between), D[:h, h:])
        np.testing.assert_allclose(cp.asnumpy(within1), D[:h, :h])
        np.testing.assert_allclose(cp.asnumpy(within2), D[h:, h:])

    # NOTE: zx is excluded -- it is an LD/ZnS ratio (ld_statistics.zns), not a
    # distance-matrix statistic, so it does not consume the Hamming kernel.
    @pytest.mark.parametrize("statname",
                             ["snn", "dxy_min", "gmin", "dd", "dd_rank"])
    def test_distance_stat_consumes_corrected_kernel(self, multiallelic_ts, statname):
        # stat via the pg_gpu kernel == stat fed numpy-reference distance matrices,
        # so each stat correctly consumes the corrected multiallelic distances.
        import cupy as cp
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, h, n, haps = self._setup(ts)
        D = self._raw_hamming(haps)
        ref = (cp.asarray(D[:h, h:]), cp.asarray(D[:h, :h]), cp.asarray(D[h:, h:]))
        fn = getattr(divergence, statname)
        got = fn(hm, 'A', 'B')
        ref_val = fn(hm, 'A', 'B', distance_matrices=ref)
        np.testing.assert_allclose(got, ref_val, rtol=1e-9, equal_nan=True)


class TestDivergenceVsTskit:
    """Count-based divergence per-allele vs tskit (dxy/da) and scikit-allel
    (Hudson FST / PBS), on multiallelic two/three sample sets."""

    @staticmethod
    def _pops(ts):
        s = ts.samples()
        h = len(s) // 2
        return list(s[:h]), list(s[h:])

    def _hm_with_pops(self, ts):
        # HaplotypeMatrix carrying sample_sets so divergence funcs can name pops.
        A, B = self._pops(ts)
        hm = HaplotypeMatrix.from_ts(ts, device="GPU")
        hm.sample_sets = {'A': list(range(len(A))),
                          'B': list(range(len(A), len(A) + len(B)))}
        return hm, A, B

    def test_fst_hudson_matches_allel(self, multiallelic_ts):
        # Classic Hudson FST = sum(num)/sum(den); per-allele == allel.hudson_fst.
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        fst_pg = divergence.fst_hudson(hm, 'A', 'B')
        ha = allel.HaplotypeArray(ts.genotype_matrix())
        ac1 = ha.count_alleles(subpop=hm.sample_sets['A'])
        ac2 = ha.count_alleles(subpop=hm.sample_sets['B'])
        num, den = allel.hudson_fst(ac1, ac2)
        np.testing.assert_allclose(fst_pg, np.sum(num) / np.sum(den), rtol=1e-9)

    def test_fst_tskit_matches_tskit(self, multiallelic_ts):
        # tskit's FST = (Hb - Hw)/(Hb + Hw); a distinct assembly from Hudson.
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        fst_pg = divergence.fst_tskit(hm, 'A', 'B')
        np.testing.assert_allclose(fst_pg, ts.Fst([A, B], mode='site'), rtol=1e-9)
        # dispatcher routes method='tskit' here; and it differs from Hudson
        assert divergence.fst(hm, 'A', 'B', method='tskit') == fst_pg
        assert not np.isclose(fst_pg, divergence.fst_hudson(hm, 'A', 'B'))

    def test_fst_nei_matches_hand_formula(self, multiallelic_ts):
        # Nei GST per-allele: HS = 1 - sum_a p_a^2 (weighted), HT from pooled
        # pbar_a. allel has no Nei GST, so pin to the numpy hand formula.
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        gst_pg = divergence.fst_nei(hm, 'A', 'B')

        G = ts.genotype_matrix()
        K = int(G.max()) + 1
        Ai, Bi = hm.sample_sets['A'], hm.sample_sets['B']
        ac1 = np.stack([(G[:, Ai] == a).sum(1) for a in range(K)], axis=1).astype(float)
        ac2 = np.stack([(G[:, Bi] == a).sum(1) for a in range(K)], axis=1).astype(float)
        n1, n2 = ac1.sum(1), ac2.sum(1)
        p1, p2 = ac1 / n1[:, None], ac2 / n2[:, None]
        h1, h2 = 1 - (p1**2).sum(1), 1 - (p2**2).sum(1)
        tot = n1 + n2
        hs = (h1 * n1 + h2 * n2) / tot
        pbar = (n1[:, None] * p1 + n2[:, None] * p2) / tot[:, None]
        ht = 1 - (pbar**2).sum(1)
        v = ht > 0
        gst_ref = (ht[v] - hs[v]).sum() / ht[v].sum()
        np.testing.assert_allclose(gst_pg, gst_ref, rtol=1e-12)

    def test_fst_weir_cockerham_matches_allel(self, multiallelic_ts):
        # WC per-allele one-vs-rest a/b/c summed over alleles (incl. per-allele
        # observed het) == allel.weir_cockerham_fst on multiallelic diploid data.
        from pg_gpu import divergence
        ts = multiallelic_ts
        G = ts.genotype_matrix()
        nsites, n_nodes = G.shape
        n_ind = n_nodes // 2
        half = n_ind // 2
        gd = G.reshape(nsites, n_ind, 2)
        subpops = [list(range(half)), list(range(half, n_ind))]
        a, b, c = allel.weir_cockerham_fst(allel.GenotypeArray(gd), subpops)
        fst_allel = np.sum(a) / np.sum(a + b + c)

        hm = HaplotypeMatrix.from_ts(ts, device="GPU")
        # individual i == haplotype rows 2i, 2i+1
        hm.sample_sets = {'A': list(range(2 * half)),
                          'B': list(range(2 * half, n_nodes))}
        fst_pg = divergence.fst_weir_cockerham(hm, 'A', 'B')
        np.testing.assert_allclose(fst_pg, fst_allel, rtol=1e-9)

    def test_pbs_matches_allel(self, multiallelic_ts):
        # PBS composes three per-allele Hudson FSTs; matches allel.pbs.
        from pg_gpu import divergence
        ts = multiallelic_ts
        n = ts.num_samples
        t = n // 3
        sets = {'A': list(range(t)), 'B': list(range(t, 2 * t)),
                'C': list(range(2 * t, n))}
        hm = HaplotypeMatrix.from_ts(ts, device="GPU")
        hm.sample_sets = sets
        w = 50
        pbs_pg = divergence.pbs(hm, 'A', 'B', 'C', window_size=w)
        ha = allel.HaplotypeArray(ts.genotype_matrix())
        ac = {k: ha.count_alleles(subpop=v) for k, v in sets.items()}
        pbs_allel = allel.pbs(ac['A'], ac['B'], ac['C'], window_size=w, normed=True)
        valid = ~np.isnan(pbs_pg) & ~np.isnan(pbs_allel)
        np.testing.assert_allclose(pbs_pg[valid], pbs_allel[valid], rtol=1e-6, atol=1e-9)

    def test_dxy_matches_tskit(self, multiallelic_ts):
        # dxy = 1 - sum_a p1_a*p2_a per site. span_normalize=False returns the
        # raw sum over sites, which equals tskit's site divergence sum.
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        dxy_pg = divergence.dxy(hm, 'A', 'B', span_normalize=False)
        dxy_ts = ts.divergence([A, B], mode='site', span_normalise=False)
        np.testing.assert_allclose(dxy_pg, dxy_ts, rtol=1e-12)

    def test_dxy_components_consistent_with_dxy(self, multiallelic_ts):
        # Raw-count path (used by windowed analysis) agrees with dxy's mean.
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        pop1 = hm.haplotypes[hm.sample_sets['A'], :]
        pop2 = hm.haplotypes[hm.sample_sets['B'], :]
        diffs, comps, nsites = divergence.dxy_components(pop1, pop2)
        dxy_pg = divergence.dxy(hm, 'A', 'B', span_normalize=False)
        # dxy (span_normalize=False) is the raw sum of per-site ratios. With no
        # missing data n1*n2 is constant, so the pooled ratio diffs/comps equals
        # the per-site mean == dxy_pg / nsites.
        assert nsites == ts.num_sites
        np.testing.assert_allclose(diffs / comps, dxy_pg / nsites, rtol=1e-12)

    def test_da_uses_per_allele_pieces(self, multiallelic_ts):
        # da = dxy - (pi1+pi2)/2, both per-allele now (internally consistent).
        from pg_gpu import divergence
        ts = multiallelic_ts
        hm, A, B = self._hm_with_pops(ts)
        da = divergence.da(hm, 'A', 'B', span_normalize=False)
        dxy_pg = divergence.dxy(hm, 'A', 'B', span_normalize=False)
        pi1 = diversity.pi(hm, 'A', span_normalize=False)
        pi2 = diversity.pi(hm, 'B', span_normalize=False)
        np.testing.assert_allclose(da, dxy_pg - (pi1 + pi2) / 2.0, rtol=1e-12)


class TestFStatisticsVsTskit:
    """Patterson f2 / f3 per-allele vs tskit (unbiased U-statistics)."""

    @staticmethod
    def _pops(ts):
        s = ts.samples()
        q = len(s) // 4
        return (list(s[:q]), list(s[q:2 * q]), list(s[2 * q:3 * q]),
                list(s[3 * q:]))

    def _hm(self, ts):
        s = ts.samples()
        q = len(s) // 4
        hm = HaplotypeMatrix.from_ts(ts, device="GPU")
        hm.sample_sets = {'A': list(range(q)), 'B': list(range(q, 2 * q)),
                          'C': list(range(2 * q, 3 * q)),
                          'D': list(range(3 * q, len(s)))}
        return hm

    def test_f2_matches_tskit(self, multiallelic_ts):
        from pg_gpu import admixture
        ts = multiallelic_ts
        A, B, _, _ = self._pops(ts)
        hm = self._hm(ts)
        f2_pg = float(np.nansum(admixture.patterson_f2(hm, 'A', 'B')))
        f2_ts = float(ts.f2([A, B], mode='site', span_normalise=False))
        np.testing.assert_allclose(f2_pg, f2_ts, rtol=1e-9)
        assert f2_pg < 0  # unbiased estimator can be negative (and is here)

    def test_f3_matches_tskit(self, multiallelic_ts):
        # pg_gpu patterson_f3(C, A, B) == f3(C; A, B); tskit target set is first.
        from pg_gpu import admixture
        ts = multiallelic_ts
        A, B, C, _ = self._pops(ts)
        hm = self._hm(ts)
        T, _ = admixture.patterson_f3(hm, 'C', 'A', 'B')
        f3_pg = float(np.nansum(T))
        f3_ts = float(ts.f3([C, A, B], mode='site', span_normalise=False))
        np.testing.assert_allclose(f3_pg, f3_ts, rtol=1e-9)

    def test_f4_matches_tskit(self, multiallelic_ts):
        from pg_gpu import admixture
        ts = multiallelic_ts
        A, B, C, D = self._pops(ts)
        hm = self._hm(ts)
        f4_pg = float(np.nansum(admixture.patterson_f4(hm, 'A', 'B', 'C', 'D')))
        f4_ts = float(ts.f4([A, B, C, D], mode='site', span_normalise=False))
        np.testing.assert_allclose(f4_pg, f4_ts, rtol=1e-9)

    def test_patterson_d_is_biallelic_restricted(self, multiallelic_ts):
        # D excludes >2-allele sites (num=den=0) and stays in [-1, 1]; the
        # exclusion now emits a BiallelicOnlyWarning with the dropped-site count.
        from pg_gpu import admixture
        ts = multiallelic_ts
        hm = self._hm(ts)
        with pytest.warns(BiallelicOnlyWarning, match="multiallelic"):
            num, den = admixture.patterson_d(hm, 'A', 'B', 'C', 'D')
        d_val = np.nansum(num) / np.nansum(den)
        assert -1.0 <= d_val <= 1.0
        # reference: strictly-biallelic subset ({0,1} across all four pops) gives
        # the same D (non-biallelic sites contribute 0 to both sums).
        G = ts.genotype_matrix()
        cols = (list(hm.sample_sets['A']) + list(hm.sample_sets['B'])
                + list(hm.sample_sets['C']) + list(hm.sample_sets['D']))
        n_alleles = np.array([np.unique(G[i, cols]).size
                              for i in range(ts.num_sites)])
        assert (n_alleles > 2).any()          # fixture really has multiallelic sites
        d_all = num[n_alleles <= 2]
        assert np.allclose(num[n_alleles > 2], 0.0)   # multiallelic sites zeroed

    @pytest.mark.filterwarnings(
        "ignore::pg_gpu._warnings.BiallelicOnlyWarning")
    def test_moving_and_jackknife_inherit_per_allele_fix(self, multiallelic_ts):
        # moving_* / average_* build on the corrected _patterson_*_gpu helpers,
        # so they inherit the per-allele fix. A single all-variant window
        # reproduces the global estimate; jackknife runs and is finite.
        from pg_gpu import admixture
        ts = multiallelic_ts
        hm = self._hm(ts)
        nv = hm.num_variants

        T, B = admixture.patterson_f3(hm, 'C', 'A', 'B')
        f3_global = float(np.nansum(T) / np.nansum(B))
        f3_win = admixture.moving_patterson_f3(hm, 'C', 'A', 'B', size=nv,
                                               normed=True)
        assert len(f3_win) == 1
        np.testing.assert_allclose(f3_win[0], f3_global, rtol=1e-9)

        num, den = admixture.patterson_d(hm, 'A', 'B', 'C', 'D')
        d_global = float(np.nansum(num) / np.nansum(den))
        d_win = admixture.moving_patterson_d(hm, 'A', 'B', 'C', 'D', size=nv)
        assert len(d_win) == 1
        np.testing.assert_allclose(d_win[0], d_global, rtol=1e-9)

        blen = max(nv // 5, 1)
        f3_est = admixture.average_patterson_f3(hm, 'C', 'A', 'B', blen=blen)[0]
        d_est = admixture.average_patterson_d(hm, 'A', 'B', 'C', 'D', blen=blen)[0]
        assert np.isfinite(f3_est) and np.isfinite(d_est)

    def test_f3_normalizer_is_unbiased_het_of_target(self, multiallelic_ts):
        # B == unbiased heterozygosity of C == tskit diversity(C) per site.
        from pg_gpu import admixture
        ts = multiallelic_ts
        _, _, C, _ = self._pops(ts)
        hm = self._hm(ts)
        _, B = admixture.patterson_f3(hm, 'C', 'A', 'B')
        het_c = float(np.nansum(B))
        div_c = float(np.atleast_1d(
            ts.diversity([C], mode='site', span_normalise=False))[0])
        np.testing.assert_allclose(het_c, div_c, rtol=1e-9)


class TestMultiallelicConsumers:
    """singleton_count / heterozygosity_expected / max_daf / mu_sfs / daf_histogram."""

    def test_singleton_count_per_allele(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        ac = allel.HaplotypeArray(ts.genotype_matrix()).count_alleles()
        # per-allele singletons: derived alleles carried by exactly one sample
        ref = int(np.sum(ac[:, 1:] == 1))
        assert diversity.singleton_count(hm) == ref

    def test_heterozygosity_expected_per_allele(self, multiallelic_hm):
        ts, hm = multiallelic_hm
        n = ts.num_samples
        ac = np.asarray(allel.HaplotypeArray(ts.genotype_matrix()).count_alleles(),
                        dtype=float)
        # He = 1 - sum_a p_a^2 over all alleles (no missing so n_valid = n)
        ref = 1.0 - (ac ** 2).sum(axis=1) / (n ** 2)
        np.testing.assert_allclose(diversity.heterozygosity_expected(hm), ref, rtol=1e-9)

    def test_heterozygosity_expected_triallelic(self):
        # {0:2, 1:1, 2:1}, n=4: He = 1 - (2^2 + 1 + 1)/16 = 1 - 6/16 = 0.625
        hm = HaplotypeMatrix(np.array([[0], [0], [1], [2]], dtype=np.int8),
                             np.array([100]), 0, 200)
        np.testing.assert_allclose(diversity.heterozygosity_expected(hm)[0],
                                   0.625, rtol=1e-12)

    def test_max_daf_is_single_derived_allele(self):
        # {0:1, 1:5, 2:4}, n=10: per-allele max DAF = 0.5 (allele 1), NOT the
        # total non-ancestral fraction 0.9.
        col = [[0]] + [[1]] * 5 + [[2]] * 4
        hm = HaplotypeMatrix(np.array(col, dtype=np.int8), np.array([100]), 0, 200)
        np.testing.assert_allclose(diversity.max_daf(hm), 0.5, rtol=1e-12)

    def test_mu_sfs_and_daf_histogram_run(self, multiallelic_hm):
        _, hm = multiallelic_hm
        assert 0.0 <= diversity.mu_sfs(hm) <= 1.0
        hist, edges = diversity.daf_histogram(hm)
        assert np.isclose(hist.sum(), 1.0)
        assert len(edges) == len(hist) + 1


class TestMultiallelicEdgeCases:
    """Hand-built sites: per-allele pi and mutation-count segregating."""

    def _hm(self, col):
        hap = np.array(col, dtype=np.int8)
        return HaplotypeMatrix(hap, np.array([100]), 0, 200)

    def test_reference_absent_site(self):
        # alleles {1, 2}, ancestral 0 absent: pi = 1 - (C(2,2)+C(2,2))/C(4,2)
        hm = self._hm([[1], [1], [2], [2]])
        np.testing.assert_allclose(diversity.pi(hm, span_normalize=False),
                                   1.0 - 2.0 / 6.0, rtol=1e-12)
        assert diversity.segregating_sites(hm) == 1  # 2 alleles -> 1 mutation

    def test_triallelic_site(self):
        # counts 2,1,1: pi = 1 - C(2,2)/C(4,2) = 1 - 1/6
        hm = self._hm([[0], [0], [1], [2]])
        np.testing.assert_allclose(diversity.pi(hm, span_normalize=False),
                                   1.0 - 1.0 / 6.0, rtol=1e-12)
        assert diversity.segregating_sites(hm) == 2  # 3 alleles -> 2 mutations

    def test_tetraallelic_site(self):
        # all four distinct: every pair differs -> pi = 1.0
        hm = self._hm([[0], [1], [2], [3]])
        np.testing.assert_allclose(diversity.pi(hm, span_normalize=False),
                                   1.0, rtol=1e-12)
        assert diversity.segregating_sites(hm) == 3
