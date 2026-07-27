"""BiallelicOnlyWarning: the shared warning class and its emit helper (0014).

Site-specific tests (from_vcf, from_haplotype_matrix, patterson_d) are added
alongside their conversions in later tasks.
"""
import warnings

import numpy as np
import pytest

from pg_gpu import BiallelicOnlyWarning, GenotypeMatrix, HaplotypeMatrix
from pg_gpu._warnings import _warn_biallelic_only
from pg_gpu.genotype_matrix import _biallelic_and_alt


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class TestBiallelicAndAlt:
    """The shared definition: biallelic = <= 2 distinct present alleles; alt =
    highest-index present allele (K sentinel for monomorphic)."""

    def test_classification_and_alt(self):
        K = 3
        ac = np.array([
            [3, 1, 0],   # {0,1}      -> biallelic, alt=1
            [3, 0, 1],   # {0,2}      -> biallelic, alt=2 (code-independent)
            [0, 2, 2],   # {1,2}      -> biallelic, alt=2 (reference absent)
            [4, 0, 0],   # mono {0}   -> biallelic, alt=K sentinel (dosage 0)
            [0, 5, 0],   # mono {1}   -> biallelic, alt=1 (fixed alt; dosage stays 2)
            [1, 1, 2],   # {0,1,2}    -> multiallelic
        ], dtype=np.int64)
        bi, alt = _biallelic_and_alt(ac)
        np.testing.assert_array_equal(bi, [True, True, True, True, True, False])
        np.testing.assert_array_equal(alt, [1, 2, 2, K, 1, 2])

    def test_alt_sentinel_counts_zero_and_never_missing(self):
        # the monomorphic sentinel K matches no real call (0..K-1) and not the
        # -1 missing code, so (calls == alt) is all-False -> dosage 0.
        ac = np.array([[5, 0, 0]], dtype=np.int64)   # monomorphic, K=3
        _, alt = _biallelic_and_alt(ac)
        calls = np.array([[0, 0, -1, 0]])            # some ref, one missing
        assert int(alt[0]) == 3
        assert not (calls == alt[0]).any()


_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\n"
)


def _write_vcf(path, rows):
    with open(path, "w") as f:
        f.write(_VCF_HEADER)
        f.write("\n".join(rows) + "\n")
    return str(path)


def test_is_userwarning_subclass():
    assert issubclass(BiallelicOnlyWarning, UserWarning)


def test_helper_warns_with_count_and_context():
    with pytest.warns(BiallelicOnlyWarning, match=r"foo\.bar .*dropped 3 multiallelic"):
        _warn_biallelic_only(3, context="foo.bar")


def test_helper_accepts_numpy_scalar_count():
    with pytest.warns(BiallelicOnlyWarning, match="dropped 7 multiallelic"):
        _warn_biallelic_only(np.int64(7), context="ctx")


def test_helper_no_warn_when_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes an error
        _warn_biallelic_only(0, context="ctx")   # must be a no-op


def test_silenceable_by_category():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=BiallelicOnlyWarning)
        _warn_biallelic_only(4, context="ctx")   # silenced -> no error raised


class TestFromVcfWarns:
    def test_warns_on_multiallelic(self, tmp_path):
        vcf = _write_vcf(tmp_path / "m.vcf", [
            "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1",     # biallelic
            "1\t200\t.\tA\tT,G\t.\tPASS\t.\tGT\t2|1\t0|0",   # triallelic -> dropped
            "1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0",     # biallelic
        ])
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            gm = GenotypeMatrix.from_vcf(vcf)
        assert gm.num_variants == 2   # the multiallelic site is dropped

    def test_no_warn_when_all_biallelic(self, tmp_path):
        vcf = _write_vcf(tmp_path / "b.vcf", [
            "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1",
            "1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0",
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)
            gm = GenotypeMatrix.from_vcf(vcf)   # must not raise
        assert gm.num_variants == 2

    def test_keeps_non_01_biallelic_and_monomorphic(self, tmp_path):
        # {0,2}, reference-absent {1,2}, and monomorphic {0} sites are all
        # biallelic-or-fewer and KEPT (were dropped by is_biallelic_01); only the
        # triallelic site is dropped. Dosage counts the alt (highest present > 0).
        vcf = _write_vcf(tmp_path / "nz.vcf", [
            "1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|0",       # {0,1} -> 1, 1
            "1\t200\t.\tA\tT,G\t.\tPASS\t.\tGT\t0|2\t0|0",     # {0,2}, alt=2 -> 1, 0
            "1\t300\t.\tA\tT,G\t.\tPASS\t.\tGT\t1|2\t1|1",     # {1,2}, alt=2 -> 1, 0
            "1\t400\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|0",       # mono {0} -> 0, 0
            "1\t500\t.\tA\tT,G\t.\tPASS\t.\tGT\t0|1\t2|0",     # {0,1,2} -> dropped
        ])
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            gm = GenotypeMatrix.from_vcf(vcf)
        assert gm.num_variants == 4
        np.testing.assert_array_equal(_host(gm.positions), [100, 200, 300, 400])
        np.testing.assert_array_equal(
            _host(gm.genotypes),
            np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=np.int8))


class TestFromHaplotypeMatrixWarns:
    def test_drops_and_warns_on_multiallelic(self):
        # site 1 (col index 1) carries allele index 2 -> not representable as a
        # 0/1/2 dosage, so it is dropped. Rows are haplotypes; cols are sites.
        hap = np.array([[0, 0, 1],
                        [1, 1, 1],
                        [0, 2, 0],
                        [1, 0, 0]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.array([100, 200, 300]), 0, 400)
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            gm = GenotypeMatrix.from_haplotype_matrix(hm)
        assert gm.num_variants == 2
        # paired sums of the two retained {0,1} sites; no bogus >2 dosage
        np.testing.assert_array_equal(_host(gm.genotypes),
                                      np.array([[1, 2], [1, 0]], dtype=np.int8))
        np.testing.assert_array_equal(_host(gm.positions), [100, 300])
        assert int(_host(gm.genotypes).max()) <= 2

    def test_biallelic_unchanged_no_warn(self):
        hap = np.array([[0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1]], dtype=np.int8)
        hm = HaplotypeMatrix(hap, np.array([100, 200, 300]), 0, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)
            gm = GenotypeMatrix.from_haplotype_matrix(hm)   # must not warn
        assert gm.num_variants == 3
        expected = (hap[0::2] + hap[1::2]).astype(np.int8)   # no missing
        np.testing.assert_array_equal(_host(gm.genotypes), expected)

    @pytest.mark.parametrize("alt_code", [1, 2, 3])
    def test_alt_code_independent(self, alt_code):
        # A biallelic {0, alt} site gives the same dosage regardless of the alt's
        # integer code -- {0,2}/{0,3} are now KEPT (were dropped under codes<=1).
        col = np.array([0, alt_code, alt_code, alt_code], dtype=np.int8)[:, None]
        hm = HaplotypeMatrix(col, np.array([100]), 0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)   # biallelic -> no warn
            gm = GenotypeMatrix.from_haplotype_matrix(hm)
        np.testing.assert_array_equal(_host(gm.genotypes),
                                      np.array([[1], [2]], dtype=np.int8))

    def test_reference_absent_site_kept(self):
        # {1,2} (reference allele 0 absent, e.g. after subsetting) is biallelic:
        # alt=2, ref implicitly allele 1; dosage = count of allele 2.
        col = np.array([1, 2, 1, 1], dtype=np.int8)[:, None]   # ind0=(1,2), ind1=(1,1)
        hm = HaplotypeMatrix(col, np.array([100]), 0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)
            gm = GenotypeMatrix.from_haplotype_matrix(hm)
        np.testing.assert_array_equal(_host(gm.genotypes),
                                      np.array([[1], [0]], dtype=np.int8))

    def test_half_call_is_fully_missing(self):
        # If either haplotype of an individual is missing (-1), the whole diploid
        # genotype is -1 -- even when the called haplotype carries the alt (the
        # count underneath is discarded; a 0/1/2 dosage has no half-missing state).
        # ind0 = (1, -1) alt half-call -> -1; ind1 = (0, 1) -> 1.
        col = np.array([1, -1, 0, 1], dtype=np.int8)[:, None]
        hm = HaplotypeMatrix(col, np.array([100]), 0, 200)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        np.testing.assert_array_equal(_host(gm.genotypes),
                                      np.array([[-1], [1]], dtype=np.int8))


class TestDegenerateColumnsDownstream:
    """The loaders now retain monomorphic and all-missing columns; the diploid
    stats must tolerate them (guarded divisions), not error or return NaN, as
    long as at least one usable column is present."""

    def _gm(self):
        # cols: 0 = polymorphic, 1 = monomorphic (all 0), 2 = all-missing (-1),
        # 3 = polymorphic. Three individuals.
        geno = np.array([
            [2, 0, -1, 1],
            [1, 0, -1, 2],
            [0, 0, -1, 0],
        ], dtype=np.int8)
        return GenotypeMatrix(geno, np.arange(4) * 100)

    def test_grm_finite(self):
        from pg_gpu import relatedness
        g = _host(relatedness.grm(self._gm()))
        assert np.isfinite(g).all()

    def test_pca_dosage_finite(self):
        from pg_gpu import decomposition
        coords, evr = decomposition.pca_dosage(self._gm(), n_components=2)
        assert np.isfinite(_host(coords)).all()
        assert np.isfinite(_host(evr)).all()


class TestPattersonDWarns:
    """patterson_d (and its moving/average variants) drops sites with >2 alleles
    across the four populations; warn once per top-level call."""

    def _hm4(self, sites):
        # ``sites`` is a list of 8-long allele-index rows (one per site); rows
        # are haplotypes, so the matrix is (8 haplotypes, n_sites). Four pops of
        # two haplotypes each.
        hap = np.array(sites, dtype=np.int8).T
        pos = np.arange(hap.shape[1]) * 100
        return HaplotypeMatrix(
            hap, pos, 0, hap.shape[1] * 100,
            sample_sets={"A": [0, 1], "B": [2, 3], "C": [4, 5], "D": [6, 7]})

    def test_warns_on_multiallelic(self):
        from pg_gpu import admixture
        hm = self._hm4([
            [0, 1, 0, 1, 0, 1, 0, 1],   # biallelic
            [0, 1, 2, 0, 1, 0, 1, 0],   # triallelic (0,1,2) -> dropped
            [1, 0, 1, 0, 1, 0, 1, 0],   # biallelic
        ])
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            admixture.patterson_d(hm, "A", "B", "C", "D")

    def test_no_warn_when_biallelic(self):
        from pg_gpu import admixture
        hm = self._hm4([
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)
            admixture.patterson_d(hm, "A", "B", "C", "D")

    def test_moving_warns_at_most_once(self):
        from pg_gpu import admixture
        hm = self._hm4([
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 2, 0, 1, 0, 1, 0],   # multiallelic
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 3, 1, 0, 1, 0, 1, 0],   # multiallelic
            [1, 1, 0, 0, 1, 0, 1, 0],
        ])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            admixture.moving_patterson_d(hm, "A", "B", "C", "D", size=2)
        bo = [w for w in rec if issubclass(w.category, BiallelicOnlyWarning)]
        assert len(bo) == 1
        assert "dropped 2 multiallelic" in str(bo[0].message)
