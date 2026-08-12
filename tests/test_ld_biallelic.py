"""Biallelic-only LD restriction (issue #100).

Covers the two matrix filters and the ephemeral indicator, the general-coding
equivalence ({0,2}/{1,2} give the same LD as the {0,1} relabelling), the
keep-shape NaN / False convention for the per-variant/pair outputs, the
BiallelicOnlyWarning behaviour, and biallelic parity of the biased plug-in
r2 / D against tskit's ld_matrix.
"""
import warnings

import numpy as np
import cupy as cp
import pytest
import msprime

from pg_gpu import HaplotypeMatrix, GenotypeMatrix, BiallelicOnlyWarning
from pg_gpu.ld_statistics import zns, omega


def _hm(mat):
    a = np.asarray(mat, dtype=np.int8)
    pos = np.arange(a.shape[1], dtype=np.int64) * 1000
    return HaplotypeMatrix(cp.asarray(a), pos, 0, a.shape[1] * 1000)


def _recode_alt(hap01, alt):
    """Relabel the derived allele 1 -> `alt` (2 or, with ref->1, a {1,2} site)."""
    return np.where(hap01 == 1, alt, hap01)


@pytest.fixture(scope="module")
def biallelic_ts_hm():
    """(ts, hm) with strictly biallelic {0,1} sites for tskit ld_matrix parity."""
    ts = msprime.sim_ancestry(
        samples=20, sequence_length=2e4, recombination_rate=1e-8,
        population_size=1e4, random_seed=7, ploidy=2)
    ts = msprime.sim_mutations(
        ts, rate=5e-8, model=msprime.BinaryMutationModel(), random_seed=8)
    hm = HaplotypeMatrix.from_ts(ts, device="GPU")
    return ts, hm


# ---------------------------------------------------------------------------
# restrict_to_biallelic / restrict_to_segregating / _biallelic_indicator
# ---------------------------------------------------------------------------


class TestRestrictMethods:

    def test_restrict_to_biallelic_keeps_codes_drops_multiallelic(self):
        # cols: {0,1}, {0,2}, {1,2}, {0,1,2}, {0,1}+missing
        base = [[0, 0, 1, 0, 0],
                [1, 2, 2, 1, -1],
                [0, 0, 1, 2, 1],
                [1, 2, 2, 0, 1]]
        out = _hm(base).restrict_to_biallelic()
        r = out.haplotypes.get()
        assert out.num_variants == 4              # dropped the {0,1,2} column
        # codes UNCHANGED (no recode): {0,2} keeps its 2, {1,2} keeps 1 and 2
        assert 2 in r[:, 1] and set(r[:, 2].tolist()) == {1, 2}
        assert (r == -1).sum() == 1               # missing preserved

    def test_restrict_to_segregating_drops_monomorphic(self):
        base = [[0, 0, 0], [1, 2, 0], [0, 1, 0], [1, 2, 0]]  # col2 monomorphic
        assert _hm(base).restrict_to_segregating().num_variants == 2

    def test_genotype_restrict_to_segregating(self):
        g = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]], dtype=np.int8)  # col1 mono
        gm = GenotypeMatrix(cp.asarray(g), np.arange(3) * 100, 0, 300)
        assert gm.restrict_to_segregating().num_variants == 2

    def test_restrict_empty_result_is_valid_not_crash(self):
        # every site multiallelic (>=3 alleles) -> nothing biallelic survives
        multi = [[0, 0], [1, 1], [2, 2], [0, 0], [1, 1], [2, 2]]
        assert _hm(multi).restrict_to_biallelic().num_variants == 0
        # every site monomorphic -> nothing segregating survives
        mono = [[0, 0], [0, 0], [0, 0]]
        assert _hm(mono).restrict_to_segregating().num_variants == 0
        # an all-multiallelic matrix has no biallelic sites: zns/omega return
        # their too-few-sites value (0.0) instead of raising "genotypes cannot
        # be empty" on the empty filter output
        for est in ('auto', 'r2'):
            assert zns(_hm(multi), estimator=est) == 0.0
            assert omega(_hm(multi), estimator=est) == 0.0

    def test_indicator_identity_and_recode(self):
        base = [[0, 0, 1], [1, 2, 2], [0, 0, 1], [1, 2, 2]]  # {0,1},{0,2},{1,2}
        ind = _hm(base)._biallelic_indicator().get()
        # all three columns partition the same haplotypes -> identical indicator
        assert np.array_equal(ind[:, 0], ind[:, 1])
        assert np.array_equal(ind[:, 1], ind[:, 2])
        assert set(np.unique(ind).tolist()) <= {0, 1}

    def test_indicator_preserves_missing(self):
        ind = _hm([[0], [1], [-1], [2]])._biallelic_indicator().get()
        assert ind[2, 0] == -1


# ---------------------------------------------------------------------------
# General-coding equivalence: {0,2}/{1,2} == {0,1} relabelling
# ---------------------------------------------------------------------------


class TestGeneralCodingEquivalence:

    @pytest.fixture
    def h01(self):
        rng = np.random.default_rng(0)
        h = rng.integers(0, 2, (30, 12)).astype(np.int8)
        h[0, :] = 0
        h[1, :] = 1                      # every site segregating
        return h

    @pytest.mark.parametrize("alt", [2])
    def test_pairwise_r2(self, h01, alt):
        r01 = _hm(h01).pairwise_r2().get()
        r02 = _hm(_recode_alt(h01, alt)).pairwise_r2().get()
        np.testing.assert_allclose(r01, r02, equal_nan=True)

    def test_reference_absent_12(self, h01):
        # {1,2} site: ref 0 -> 1, derived 1 -> 2 (allele 0 absent entirely)
        h12 = np.where(h01 == 0, 1, 2).astype(np.int8)
        np.testing.assert_allclose(_hm(h01).pairwise_r2().get(),
                                   _hm(h12).pairwise_r2().get(), equal_nan=True)

    def test_zns_omega(self, h01):
        h02 = _recode_alt(h01, 2)
        for est in ('r2', 'sigma_d2'):
            assert np.isclose(zns(_hm(h01), estimator=est),
                              zns(_hm(h02), estimator=est), equal_nan=True)
            assert np.isclose(omega(_hm(h01), estimator=est),
                              omega(_hm(h02), estimator=est), equal_nan=True)

    def test_windowed_r_squared(self, h01):
        bp = [0, 4000, 12000]
        r1, c1 = _hm(h01).windowed_r_squared(bp)
        r2, c2 = _hm(_recode_alt(h01, 2)).windowed_r_squared(bp)
        np.testing.assert_allclose(r1, r2, equal_nan=True)
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Keep-shape NaN / False for per-variant / per-pair outputs
# ---------------------------------------------------------------------------


class TestKeepShapeOutOfDomain:

    def _data(self):
        rng = np.random.default_rng(1)
        h = rng.integers(0, 2, (20, 6)).astype(np.int8)
        h[0, :] = 0
        h[1, :] = 1
        h[3, 4] = 2                      # site 4 -> multiallelic {0,1,2}
        return h

    def test_pairwise_r2_nan_rows(self):
        r = _hm(self._data()).pairwise_r2().get()
        assert r.shape == (6, 6)                       # shape preserved
        assert np.all(np.isnan(r[4, [0, 1, 2, 3, 5]]))  # multiallelic row NaN
        assert np.isfinite(r[0, 1])                     # biallelic pair finite

    def test_pairwise_r2_monomorphic_nan(self):
        h = np.zeros((10, 3), dtype=np.int8)
        h[:, 1] = np.array([0, 1] * 5)                  # only col 1 segregating
        r = _hm(h).pairwise_r2().get()
        assert np.all(np.isnan(r[0, [1, 2]]))           # monomorphic -> NaN

    def test_pairwise_LD_v_nan_rows(self):
        D = _hm(self._data()).pairwise_LD_v().get()
        assert np.all(np.isnan(D[4, [0, 1, 2, 3, 5]]))
        assert np.isfinite(D[0, 1])

    def test_locate_unlinked_false_and_shape(self):
        loc = _hm(self._data()).locate_unlinked(size=6, step=6, threshold=0.5)
        assert loc.shape == (6,)
        assert loc[4] == False  # noqa: E712 -- multiallelic site not certified


# ---------------------------------------------------------------------------
# BiallelicOnlyWarning: once per top-level call, count = dropped multiallelic
# ---------------------------------------------------------------------------


class TestWarning:

    def _multi(self, n_multi):
        rng = np.random.default_rng(2)
        h = rng.integers(0, 2, (30, 12)).astype(np.int8)
        h[0, :] = 0
        h[1, :] = 1
        for j in range(n_multi):
            h[3, j] = 2                  # make j sites multiallelic
        return h

    @pytest.mark.parametrize("fn", [
        lambda hm: hm.pairwise_r2(),
        lambda hm: hm.pairwise_LD_v(),
        lambda hm: hm.locate_unlinked(size=12, step=12, threshold=0.5),
        lambda hm: zns(hm, estimator='r2'),
        lambda hm: omega(hm, estimator='r2'),
        lambda hm: hm.windowed_r_squared([0, 6000, 13000]),
    ])
    def test_warns_once(self, fn):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fn(_hm(self._multi(2)))
        bw = [x for x in w if issubclass(x.category, BiallelicOnlyWarning)]
        assert len(bw) == 1

    def test_no_warning_when_all_biallelic(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _hm(self._multi(0)).pairwise_r2()
        assert not any(issubclass(x.category, BiallelicOnlyWarning) for x in w)


# ---------------------------------------------------------------------------
# Biallelic parity vs tskit ld_matrix (biased plug-in r2 / D)
# ---------------------------------------------------------------------------


class TestTskitParity:

    @staticmethod
    def _seg(ts):
        G = ts.genotype_matrix()
        n = G.shape[1]
        return np.array([s for s in range(ts.num_sites)
                         if 0 < int(G[s].sum()) < n])

    def test_pairwise_r2_matches_tskit(self, biallelic_ts_hm):
        ts, hm = biallelic_ts_hm
        seg = self._seg(ts)
        assert len(seg) >= 8
        r2 = hm.pairwise_r2().get()[np.ix_(seg, seg)]
        M = ts.ld_matrix(sites=[seg.tolist()], stat='r2')
        iu = np.triu_indices(len(seg), k=1)
        a, b = r2[iu], M[iu]
        fin = np.isfinite(a) & np.isfinite(b)
        assert fin.sum() > 10
        np.testing.assert_allclose(a[fin], b[fin], rtol=1e-9, atol=1e-9)

    def test_pairwise_LD_v_matches_tskit_D(self, biallelic_ts_hm):
        ts, hm = biallelic_ts_hm
        seg = self._seg(ts)
        D = hm.pairwise_LD_v().get()[np.ix_(seg, seg)]
        M = ts.ld_matrix(sites=[seg.tolist()], stat='D')
        iu = np.triu_indices(len(seg), k=1)
        a, b = D[iu], M[iu]
        fin = np.isfinite(a) & np.isfinite(b)
        assert fin.sum() > 10
        np.testing.assert_allclose(a[fin], b[fin], rtol=1e-9, atol=1e-9)
