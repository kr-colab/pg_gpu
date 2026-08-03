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
