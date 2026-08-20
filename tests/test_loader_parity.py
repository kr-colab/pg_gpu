"""Cross-loader parity for GenotypeMatrix.

The three GenotypeMatrix loaders -- from_vcf, from_zarr (eager), and from_zarr
(streaming) -- must agree on the dosage matrix they produce from equivalent
data. They historically did not (from_vcf used one biallelic definition, the
zarr paths a raw ploidy sum), and nothing in the suite compared them. These
tests drive all three from a single call_genotype block so a future divergence
fails loudly.
"""
import warnings

import numpy as np
import pytest

from pg_gpu import BiallelicOnlyWarning, GenotypeMatrix
from pg_gpu.zarr_io import write_vcz

from .conftest import canonical_hap_rows


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


_ALT_POOL = ["C", "G", "T", "N"]


def _write_vcf_from_cg(path, cg, positions, sample_names):
    """Write a VCF whose per-diploid genotypes are exactly ``cg[v, d, :]``."""
    n_var, n_dip, _ = cg.shape
    header = (
        "##fileformat=VCFv4.2\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(sample_names) + "\n"
    )
    rows = []
    for v in range(n_var):
        site = cg[v]
        m = int(site.max())            # highest allele index present (-1 => all missing)
        alt = ",".join(_ALT_POOL[:max(m, 1)])
        calls = []
        for d in range(n_dip):
            a, b = site[d]
            sa = "." if a < 0 else str(int(a))
            sb = "." if b < 0 else str(int(b))
            calls.append(f"{sa}|{sb}")
        rows.append(f"1\t{int(positions[v])}\t.\tA\t{alt}\t.\tPASS\t.\tGT\t"
                    + "\t".join(calls))
    with open(path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")
    return str(path)


def _expected_dosage(cg):
    """Unified rule, computed on the host: biallelic = <= 2 distinct present
    alleles; dosage = count of the highest present alt (> 0); any missing ploidy
    -> -1; a >= 3-allele site -> whole row -1."""
    n_var, n_dip, _ = cg.shape
    out = np.zeros((n_dip, n_var), dtype=np.int8)
    for v in range(n_var):
        site = cg[v]
        present = sorted({int(x) for x in site.ravel() if x >= 0})
        alts = [a for a in present if a > 0]
        alt = max(alts) if alts else None
        for d in range(n_dip):
            a, b = int(site[d, 0]), int(site[d, 1])
            if a < 0 or b < 0:
                out[d, v] = -1
            elif alt is None:
                out[d, v] = 0
            else:
                out[d, v] = (a == alt) + (b == alt)
        if len(present) > 2:            # multiallelic -> fully-missing row
            out[:, v] = -1
    return out


# call_genotype (n_var, n_dip, 2); n_dip = 3.
_SAMPLES = ["s0", "s1", "s2"]
_BIALLELIC_CG = np.array([
    [[0, 1], [1, 1], [0, 0]],   # {0,1}          -> 1, 2, 0
    [[0, 2], [0, 0], [2, 2]],   # {0,2} alt=2     -> 1, 0, 2
    [[1, 2], [1, 1], [2, 2]],   # {1,2} ref-absent-> 1, 0, 2
    [[0, 0], [0, 0], [0, 0]],   # mono {0}        -> 0, 0, 0
    [[0, 1], [-1, 1], [0, 0]],  # ind1 half-call  -> 1, -1, 0
], dtype=np.int8)
_BIALLELIC_POS = np.array([100, 200, 300, 400, 500])

_MULTI_CG = np.concatenate([
    _BIALLELIC_CG,
    np.array([[[0, 1], [2, 0], [1, 2]]], dtype=np.int8),   # {0,1,2} triallelic
])
_MULTI_POS = np.array([100, 200, 300, 400, 500, 600])


def _load_all_zarr(tmp_path, cg, pos):
    path = str(tmp_path / "parity.vcz.zarr")
    write_vcz(path, cg, pos, samples=_SAMPLES, contig_name="1")
    eager = GenotypeMatrix.from_zarr(path, streaming="never")
    streaming = GenotypeMatrix.from_zarr(path, streaming="always")
    return path, eager, streaming


class TestBiallelicParity:
    """No >= 3-allele sites: all three loaders must be byte-identical."""

    def test_all_three_loaders_agree(self, tmp_path):
        expected = _expected_dosage(_BIALLELIC_CG)

        with warnings.catch_warnings():
            warnings.simplefilter("error", BiallelicOnlyWarning)   # none expected
            path, eager, streaming = _load_all_zarr(
                tmp_path, _BIALLELIC_CG, _BIALLELIC_POS)
            strm = streaming.materialize()
            vcf = _write_vcf_from_cg(
                tmp_path / "parity.vcf", _BIALLELIC_CG, _BIALLELIC_POS, _SAMPLES)
            from_vcf = GenotypeMatrix.from_vcf(vcf)

        np.testing.assert_array_equal(_host(eager.genotypes), expected)
        np.testing.assert_array_equal(_host(strm.genotypes), expected)
        np.testing.assert_array_equal(_host(from_vcf.genotypes), expected)
        for gm in (eager, strm, from_vcf):
            np.testing.assert_array_equal(_host(gm.positions), _BIALLELIC_POS)


class TestMultiallelicParity:
    """A >= 3-allele site: the two zarr paths agree (row -> -1); from_vcf drops
    it instead. Both surface exactly one BiallelicOnlyWarning."""

    def test_zarr_eager_and_streaming_agree(self, tmp_path):
        expected = _expected_dosage(_MULTI_CG)   # last row all -1
        path = str(tmp_path / "m.vcz.zarr")
        write_vcz(path, _MULTI_CG, _MULTI_POS, samples=_SAMPLES, contig_name="1")

        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            eager = GenotypeMatrix.from_zarr(path, streaming="never")
        streaming = GenotypeMatrix.from_zarr(path, streaming="always")
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            strm = streaming.materialize()

        np.testing.assert_array_equal(_host(eager.genotypes), expected)
        np.testing.assert_array_equal(_host(strm.genotypes), expected)
        assert eager.num_variants == 6
        # the multiallelic row is present but fully missing
        np.testing.assert_array_equal(_host(eager.genotypes)[:, 5], [-1, -1, -1])

    def test_from_vcf_drops_the_site(self, tmp_path):
        vcf = _write_vcf_from_cg(
            tmp_path / "m.vcf", _MULTI_CG, _MULTI_POS, _SAMPLES)
        with pytest.warns(BiallelicOnlyWarning, match="dropped 1 multiallelic"):
            from_vcf = GenotypeMatrix.from_vcf(vcf)
        # from_vcf drops the multiallelic site entirely (5 kept, not 6)...
        assert from_vcf.num_variants == 5
        np.testing.assert_array_equal(_host(from_vcf.positions), _BIALLELIC_POS)
        # ...and the retained biallelic rows match the zarr loaders' first 5.
        np.testing.assert_array_equal(
            _host(from_vcf.genotypes), _expected_dosage(_BIALLELIC_CG))

    def test_streaming_warns_once_across_chunks(self, tmp_path):
        # Two chunks, each carrying a multiallelic site; the warning must fire
        # exactly once per streaming load, not once per affected chunk.
        cg = np.array([
            [[0, 1], [2, 0], [1, 2]],   # chunk 0: {0,1,2} multiallelic
            [[0, 1], [1, 1], [0, 0]],   # chunk 0: biallelic
            [[0, 1], [3, 0], [1, 3]],   # chunk 1: {0,1,3} multiallelic
            [[0, 0], [0, 1], [1, 1]],   # chunk 1: biallelic
        ], dtype=np.int8)
        pos = np.array([100, 200, 2_000_000, 2_000_100])
        path = str(tmp_path / "chunks.vcz.zarr")
        write_vcz(path, cg, pos, samples=_SAMPLES, contig_name="1")

        streaming = GenotypeMatrix.from_zarr(
            path, streaming="always", chunk_bp=1_000_000)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            chunks = list(streaming.iter_gpu_chunks())
        assert len(chunks) >= 2   # the sites really did split across chunks
        bo = [w for w in rec if issubclass(w.category, BiallelicOnlyWarning)]
        assert len(bo) == 1


class TestHaplotypeRowOrder:
    """Every loader must place a sample's two gametes at rows 2i and 2i+1.

    The loaders used to disagree: from_ts emitted that order while the VCF and
    zarr paths emitted ploidy-0 rows followed by ploidy-1 rows. Whichever
    convention a consumer assumed, data from the other loader was paired across
    individuals -- silently, since a mispaired matrix is still well formed.
    """

    @pytest.fixture(scope="class")
    def ts_and_vcf(self, tmp_path_factory):
        """A diploid tree sequence and the VCF written from it."""
        msprime = pytest.importorskip("msprime")
        ts = msprime.sim_ancestry(samples=8, population_size=1e4,
                                  sequence_length=2e4, recombination_rate=1e-8,
                                  random_seed=11, ploidy=2)
        ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=11,
                                   model="binary")
        assert ts.num_sites > 0
        path = str(tmp_path_factory.mktemp("rows") / "ts.vcf")
        with open(path, "w") as fh:
            ts.write_vcf(fh, individual_names=[f"s{i}" for i
                                               in range(ts.num_individuals)])
        return ts, path

    def test_vcf_and_zarr_agree_with_canonical_order(self, tmp_path):
        from pg_gpu import HaplotypeMatrix

        cg, pos = _BIALLELIC_CG, _BIALLELIC_POS
        vcf = str(tmp_path / "rows.vcf")
        _write_vcf_from_cg(vcf, cg, pos, _SAMPLES)
        vcz = str(tmp_path / "rows.vcz.zarr")
        write_vcz(vcz, cg, pos, samples=_SAMPLES, contig_name="1")

        expected = canonical_hap_rows(cg)
        np.testing.assert_array_equal(_host(HaplotypeMatrix.from_vcf(vcf).haplotypes),
                                      expected)
        np.testing.assert_array_equal(_host(HaplotypeMatrix.from_zarr(vcz).haplotypes),
                                      expected)

    def test_from_ts_matches_the_vcf_loader(self, ts_and_vcf):
        """A tree sequence and the VCF written from it load identically."""
        from pg_gpu import HaplotypeMatrix

        ts, vcf = ts_and_vcf
        hm_ts = HaplotypeMatrix.from_ts(ts)
        hm_vcf = HaplotypeMatrix.from_vcf(vcf)
        np.testing.assert_array_equal(_host(hm_ts.haplotypes),
                                      _host(hm_vcf.haplotypes))

        # tskit sample nodes 2i / 2i+1 are individual i, so summing adjacent
        # rows is the per-individual dosage the loaders must reproduce.
        g = ts.genotype_matrix()
        truth = np.stack([g[:, 2 * i] + g[:, 2 * i + 1]
                          for i in range(ts.num_individuals)], axis=1).T
        gm = GenotypeMatrix.from_haplotype_matrix(hm_ts)
        np.testing.assert_array_equal(_host(gm.genotypes), truth)

    def test_consumers_read_the_canonical_order(self, ts_and_vcf):
        """Statistics that rebuild individuals agree with a host reference.

        Loader-against-loader would prove nothing once the rows are known to be
        byte-identical, so each statistic is checked against a value computed
        directly from the tree sequence instead.
        """
        from pg_gpu import HaplotypeMatrix, diversity, relatedness

        ts, vcf = ts_and_vcf
        hm = HaplotypeMatrix.from_vcf(vcf)
        g = ts.genotype_matrix()
        dosage = np.stack([g[:, 2 * i] + g[:, 2 * i + 1]
                           for i in range(ts.num_individuals)], axis=1).T

        ho = float(np.nanmean(_host(diversity.heterozygosity_observed(hm))))
        assert ho == pytest.approx(float((dosage == 1).mean()), rel=1e-12)

        # For ibs and grm, build the reference GenotypeMatrix straight from the
        # tree sequence dosages so the comparison isolates the pairing without
        # restating either statistic's formula.
        pos = np.asarray([s.position for s in ts.sites()], dtype=np.int64)
        ref = GenotypeMatrix(dosage.astype(np.int8), pos,
                             int(pos[0]), int(pos[-1]))
        via_loader = GenotypeMatrix.from_haplotype_matrix(hm)
        np.testing.assert_allclose(_host(relatedness.ibs(via_loader)),
                                   _host(relatedness.ibs(ref)), rtol=1e-12)
        np.testing.assert_allclose(_host(relatedness.grm(via_loader)),
                                   _host(relatedness.grm(ref)), rtol=1e-12)

    def test_to_zarr_round_trip_preserves_pairing(self, tmp_path):
        """The writer must read the same row order the loaders emit."""
        from pg_gpu import HaplotypeMatrix

        cg, pos = _BIALLELIC_CG, _BIALLELIC_POS
        vcz = str(tmp_path / "in.vcz.zarr")
        write_vcz(vcz, cg, pos, samples=_SAMPLES, contig_name="1")
        hm = HaplotypeMatrix.from_zarr(vcz)

        out = str(tmp_path / "out.vcz.zarr")
        hm.to_zarr(out, format="vcz", contig_name="1")
        np.testing.assert_array_equal(_host(HaplotypeMatrix.from_zarr(out).haplotypes),
                                      _host(hm.haplotypes))

    def test_populations_pair_within_their_own_subset(self, tmp_path):
        """Pairing must survive subsetting to a population.

        Statistics like fst_weir_cockerham pair consecutive rows *within* the
        population subset, a second pairing step layered on the row order. The
        populations here alternate through the samples, so each one's rows are
        non-contiguous in the full matrix and the subset has to carry the
        adjacency with it.
        """
        from pg_gpu import HaplotypeMatrix, divergence, diversity

        rng = np.random.default_rng(4)
        n_var, n_dip = 40, 6
        cg = rng.integers(0, 2, size=(n_var, n_dip, 2)).astype(np.int8)
        pos = np.arange(1, n_var + 1) * 100
        names = [f"s{i}" for i in range(n_dip)]

        vcf = str(tmp_path / "multipop.vcf")
        _write_vcf_from_cg(vcf, cg, pos, names)
        pop_file = tmp_path / "multipop.tsv"
        pop_file.write_text("".join(
            f"s{i}\t{'p1' if i % 2 == 0 else 'p2'}\n" for i in range(n_dip)))

        hm = HaplotypeMatrix.from_vcf(vcf)
        hm.load_pop_file(str(pop_file))
        # p1 = samples 0, 2, 4 -> both gametes of each, still adjacent in pairs
        assert sorted(hm.sample_sets["p1"]) == [0, 1, 4, 5, 8, 9]
        assert sorted(hm.sample_sets["p2"]) == [2, 3, 6, 7, 10, 11]

        def only(dips):
            """A matrix holding just these samples, in canonical order."""
            return HaplotypeMatrix(canonical_hap_rows(cg[:, dips, :]), pos,
                                   int(pos[0]), int(pos[-1]))

        # Per-population statistic against a matrix of just that population.
        for pop, dips in (("p1", [0, 2, 4]), ("p2", [1, 3, 5])):
            got = float(np.nanmean(_host(
                diversity.heterozygosity_observed(hm, population=pop))))
            ref = float(np.nanmean(_host(
                diversity.heterozygosity_observed(only(dips)))))
            assert got == pytest.approx(ref, rel=1e-12), pop

        # Two-population statistic against the same populations laid out
        # contiguously, which must give the identical estimate.
        stacked = only([0, 2, 4, 1, 3, 5])
        stacked.sample_sets = {"p1": list(range(6)), "p2": list(range(6, 12))}
        assert float(divergence.fst_weir_cockerham(hm, "p1", "p2")) == pytest.approx(
            float(divergence.fst_weir_cockerham(stacked, "p1", "p2")), rel=1e-12)

    def test_load_pop_file_lists_both_gametes(self, tmp_path):
        """Population sets must hold rows 2i and 2i+1 for each member."""
        from pg_gpu import HaplotypeMatrix

        cg, pos = _BIALLELIC_CG, _BIALLELIC_POS
        vcf = str(tmp_path / "pops.vcf")
        _write_vcf_from_cg(vcf, cg, pos, _SAMPLES)
        pop_file = tmp_path / "pops.tsv"
        pop_file.write_text("s0\tp1\ns1\tp1\ns2\tp2\n")

        hm = HaplotypeMatrix.from_vcf(vcf)
        hm.load_pop_file(str(pop_file))
        assert sorted(hm.sample_sets["p1"]) == [0, 1, 2, 3]
        assert sorted(hm.sample_sets["p2"]) == [4, 5]
