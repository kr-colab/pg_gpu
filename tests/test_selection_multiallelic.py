"""Multiallelic support for iHS / nSL (issue #140).

scikit-allel's ihs/nsl and tskit have no multiallelic iHS/nSL, so there is no
external oracle. The multiallelic path is validated three ways:

  * biallelic reduction -- on biallelic data the new code returns a 1-D array
    and matches the scikit-allel oracle (the existing parity behaviour).
  * relabel invariance -- recoding the single derived allele 1 -> 2 must move
    the score to column 1 and leave column 0 (now-absent allele 1) all-NaN,
    with the allele-2 score bit-identical to the original biallelic score.
  * brute-force reference -- a pure-numpy one-vs-ancestral nSL scan reproduces
    the GPU per-allele scores on genuinely triallelic data.
"""

import numpy as np
import pytest
import allel

from pg_gpu import HaplotypeMatrix, selection
from pg_gpu.windowed_analysis import windowed_analysis


def _mat(hap, pos=None):
    hap = np.asarray(hap, dtype=np.int8)
    n_var = hap.shape[1]
    if pos is None:
        pos = np.arange(n_var, dtype=np.float64) * 1000 + 1
    return HaplotypeMatrix(hap.copy(), pos, 0, int(pos[-1]) + 1)


# ---------------------------------------------------------------------------
# Brute-force pure-numpy nSL reference (one derived allele vs ancestral 0)
# ---------------------------------------------------------------------------

def _nsl_scan_ref(hap, focal_allele):
    """One-direction mean-SSL scan, mirroring _nsl01_kernel exactly.

    hap : (n_hap, n_var). Returns per-site mean SSL for the ancestral (0) and
    focal-allele homozygous pair classes (NaN where a class is empty).
    """
    n_hap, n_var = hap.shape
    s0 = np.zeros(n_var)
    c0 = np.zeros(n_var, dtype=int)
    sa = np.zeros(n_var)
    ca = np.zeros(n_var, dtype=int)
    for j in range(n_hap):
        for k in range(j + 1, n_hap):
            ssl = 0
            for i in range(n_var):
                a1 = int(hap[j, i])
                a2 = int(hap[k, i])
                if a1 < 0 or a2 < 0:
                    ssl += 1
                elif a1 == a2:
                    ssl += 1
                    if a1 == 0:
                        s0[i] += ssl
                        c0[i] += 1
                    elif a1 == focal_allele:
                        sa[i] += ssl
                        ca[i] += 1
                else:
                    ssl = 0
    nsl0 = np.where(c0 > 0, s0 / np.maximum(c0, 1), np.nan)
    nsla = np.where(ca > 0, sa / np.maximum(ca, 1), np.nan)
    return nsl0, nsla


def _nsl_ref(hap, focal_allele):
    """Full forward+reverse nSL reference for one derived allele."""
    f0, fa = _nsl_scan_ref(hap, focal_allele)
    r0, ra = _nsl_scan_ref(hap[:, ::-1], focal_allele)
    nsl0 = f0 + r0[::-1]
    nsla = fa + ra[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(nsla / nsl0)


# ---------------------------------------------------------------------------
# Biallelic reduction: 1-D output that matches the scikit-allel oracle
# ---------------------------------------------------------------------------

class TestBiallelicReduction:
    @pytest.fixture
    def biallelic(self):
        rng = np.random.RandomState(42)
        hap = rng.randint(0, 2, (40, 200)).astype(np.int8)
        pos = (np.arange(200) * 500 + 1).astype(np.float64)
        return hap, pos

    def test_nsl_biallelic_is_1d_and_matches_allel(self, biallelic):
        hap, pos = biallelic
        score = selection.nsl(_mat(hap, pos))
        assert score.ndim == 1 and score.shape == (200,)
        ref = allel.nsl(hap.T, use_threads=False)  # allel layout (n_var, n_hap)
        both = ~np.isnan(score) & ~np.isnan(ref)
        np.testing.assert_allclose(score[both], ref[both], rtol=1e-5)

    def test_ihs_biallelic_is_1d_and_matches_allel(self, biallelic):
        hap, pos = biallelic
        score = selection.ihs(_mat(hap, pos), include_edges=False)
        assert score.ndim == 1 and score.shape == (200,)
        ref = allel.ihs(hap.T, pos, include_edges=False, use_threads=False)
        both = ~np.isnan(score) & ~np.isnan(ref)
        assert both.sum() > 0
        np.testing.assert_allclose(score[both], ref[both], rtol=1e-5)

    def test_biallelic_filter_of_multiallelic_fixture(self, multiallelic_hm):
        _, hm = multiallelic_hm
        hm_bi = hm.apply_biallelic_filter()
        hap = np.asarray(hm_bi.haplotypes.get()
                         if hasattr(hm_bi.haplotypes, "get") else hm_bi.haplotypes)
        assert hap.max() <= 1  # genuinely biallelic after the filter
        score = selection.nsl(hm_bi)
        assert score.ndim == 1
        ref = allel.nsl(hap.T, use_threads=False)  # allel layout (n_var, n_hap)
        both = ~np.isnan(score) & ~np.isnan(ref)
        np.testing.assert_allclose(score[both], ref[both], rtol=1e-5)


# ---------------------------------------------------------------------------
# Relabel invariance: derived allele 1 -> 2 shifts the score to column 1
# ---------------------------------------------------------------------------

class TestRelabelInvariance:
    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(0)
        hap = rng.randint(0, 2, (30, 120)).astype(np.int8)
        pos = (np.arange(120) * 1000 + 1).astype(np.float64)
        relabel = np.where(hap == 1, np.int8(2), np.int8(0)).astype(np.int8)
        return hap, relabel, pos

    def _check(self, base, re):
        assert base.ndim == 1
        assert re.ndim == 2 and re.shape[1] == 2
        assert np.all(np.isnan(re[:, 0]))  # allele 1 now absent everywhere
        both = np.isfinite(base) & np.isfinite(re[:, 1])
        assert both.sum() > 0
        np.testing.assert_allclose(re[both, 1], base[both], rtol=1e-9, atol=1e-9)

    def test_nsl_relabel(self, data):
        hap, relabel, pos = data
        self._check(selection.nsl(_mat(hap, pos)), selection.nsl(_mat(relabel, pos)))

    def test_ihs_relabel(self, data):
        hap, relabel, pos = data
        self._check(selection.ihs(_mat(hap, pos)), selection.ihs(_mat(relabel, pos)))


# ---------------------------------------------------------------------------
# Brute-force reference on genuinely triallelic data
# ---------------------------------------------------------------------------

class TestMultiallelicBruteForce:
    def test_nsl_triallelic_matches_reference(self):
        rng = np.random.RandomState(7)
        n_hap, n_var = 10, 25
        hap = rng.randint(0, 2, (n_hap, n_var)).astype(np.int8)
        # inject derived allele 2 at a few sites (some {0,2}, some {0,1,2})
        hap[:4, 5] = 2
        hap[:3, 12] = 2
        hap[3:6, 12] = 1
        hap[:5, 18] = 2

        score = selection.nsl(_mat(hap))
        assert score.shape == (n_var, 2)

        ref = np.stack([_nsl_ref(hap, 1), _nsl_ref(hap, 2)], axis=1)

        assert np.array_equal(np.isnan(score), np.isnan(ref))
        both = np.isfinite(score) & np.isfinite(ref)
        assert both.sum() > 0
        np.testing.assert_allclose(score[both], ref[both], rtol=1e-9, atol=1e-9)


class TestReferenceAbsent:
    def test_no_ancestral_allele_is_nan(self):
        # A focal site with no ancestral (0) allele has no reference class, so
        # every derived allele's one-vs-ancestral score is NaN there.
        rng = np.random.RandomState(5)
        n_hap, n_var = 20, 40
        hap = rng.randint(0, 2, (n_hap, n_var)).astype(np.int8)
        focal = 20
        hap[:10, focal] = 1   # no allele 0 remains at the focal site
        hap[10:, focal] = 2
        for score in (selection.ihs(_mat(hap), min_maf=0.0), selection.nsl(_mat(hap))):
            assert score.shape == (n_var, 2)
            assert np.all(np.isnan(score[focal]))


# ---------------------------------------------------------------------------
# Per-allele MAF: a common allele with index >= 2 is now visible to min_maf
# ---------------------------------------------------------------------------

class TestPerAlleleMAF:
    def test_common_high_index_allele_visible(self):
        rng = np.random.RandomState(3)
        n_hap, n_var = 40, 60
        hap = rng.randint(0, 2, (n_hap, n_var)).astype(np.int8)
        focal = 30
        # focal site: ancestral common, allele 1 rare (1/40 = 0.025 < min_maf),
        # allele 2 common (20/40 = 0.5). Old counts (0/1 only) would hide allele 2.
        col = np.zeros(n_hap, dtype=np.int8)
        col[0] = 1
        col[1:21] = 2
        hap[:, focal] = col

        score = selection.ihs(_mat(hap), min_maf=0.05)
        assert score.shape == (n_var, 2)
        assert np.isnan(score[focal, 0])       # allele 1 fails per-allele MAF
        assert np.isfinite(score[focal, 1])    # common allele 2 passes


# ---------------------------------------------------------------------------
# mean_nsl (windowed) -- first coverage; must accept 1-D and 2-D nsl output
# ---------------------------------------------------------------------------

class TestMeanNSL:
    def test_mean_nsl_biallelic_matches_manual(self):
        rng = np.random.RandomState(11)
        n_hap, n_var = 30, 300
        hap = rng.randint(0, 2, (n_hap, n_var)).astype(np.int8)
        pos = (np.arange(n_var) * 100 + 1).astype(np.float64)
        hm = _mat(hap, pos)

        df = windowed_analysis(hm, window_size=5000, statistics=["mean_nsl"])
        nsl = selection.nsl(hm)
        # manual per-window mean of finite nsl
        for _, row in df.iterrows():
            mask = (pos >= row["start"]) & (pos < row["end"]) & np.isfinite(nsl)
            expected = np.nanmean(nsl[mask]) if mask.any() else np.nan
            got = row["mean_nsl"]
            if np.isnan(expected):
                assert np.isnan(got)
            else:
                np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_mean_nsl_multiallelic_runs(self):
        rng = np.random.RandomState(13)
        n_hap, n_var = 30, 300
        hap = rng.randint(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[:6, ::7] = 2  # scatter a derived allele 2
        pos = (np.arange(n_var) * 100 + 1).astype(np.float64)
        hm = _mat(hap, pos)
        assert selection.nsl(hm).ndim == 2  # 2-D per-allele output

        df = windowed_analysis(hm, window_size=5000, statistics=["mean_nsl"])
        assert "mean_nsl" in df.columns
        assert np.isfinite(df["mean_nsl"].to_numpy()).any()
