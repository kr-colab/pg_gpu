"""Tests for the VCF/VCZ FORMAT/INFO ``fields=`` kwarg on
``HaplotypeMatrix.from_vcf`` / ``from_zarr`` and
``GenotypeMatrix.from_vcf`` / ``from_zarr``.

Covers the read path for all three zarr layouts (VCZ, scikit-allel
flat, scikit-allel grouped) plus a small hand-written VCF for the
allel.read_vcf path.
"""

import textwrap

import cupy as cp
import msprime
import numpy as np
import pytest
import zarr

from pg_gpu import GenotypeMatrix, HaplotypeMatrix


# ── Helpers ──────────────────────────────────────────────────────────────


def _simulate_hm(n_samples=8, seq_length=10_000, seed=42):
    """Small msprime fixture matching the helper in test_zarr_io."""
    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=seq_length,
        recombination_rate=1e-4, random_seed=seed, ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


def _host(arr):
    return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr


def _write_vcz_with_qc(tmp_path, hm, *, mq_values=None, gq_values=None,
                       dp_values=None, name="qc.vcz"):
    """Write a VCZ store and inject synthetic QC arrays in place.

    bio2zarr would produce these alongside ``call_genotype`` from a real
    VCF; for the read tests we forge them directly so the fixture stays
    msprime-only.
    """
    path = str(tmp_path / name)
    hm.to_zarr(path, format="vcz", contig_name="chr1")
    store = zarr.open_group(path, mode="r+")
    n_var = int(hm.haplotypes.shape[1])
    n_samples = int(hm.haplotypes.shape[0] // 2)
    if mq_values is None:
        mq_values = np.linspace(20.0, 60.0, n_var, dtype=np.float32)
    if gq_values is None:
        rng = np.random.default_rng(0)
        gq_values = rng.integers(0, 99, size=(n_var, n_samples),
                                 dtype=np.int16)
    if dp_values is None:
        rng = np.random.default_rng(1)
        dp_values = rng.integers(1, 50, size=(n_var, n_samples),
                                 dtype=np.int16)
    store.create_array("variant_MQ", data=mq_values)
    store.create_array("call_GQ", data=gq_values)
    store.create_array("call_DP", data=dp_values)
    return path, mq_values, gq_values, dp_values


def _write_allel_with_qc(tmp_path, hm, name="qc_allel.zarr"):
    """Write a scikit-allel layout store and add INFO/FORMAT arrays."""
    path = str(tmp_path / name)
    hm.to_zarr(path, format="scikit-allel")
    store = zarr.open_group(path, mode="r+")
    n_var = int(hm.haplotypes.shape[1])
    n_samples = int(hm.haplotypes.shape[0] // 2)
    mq = np.linspace(10.0, 80.0, n_var, dtype=np.float32)
    rng = np.random.default_rng(7)
    gq = rng.integers(0, 99, size=(n_var, n_samples), dtype=np.int16)
    store.create_array("variants/MQ", data=mq)
    store.create_array("calldata/GQ", data=gq)
    return path, mq, gq


def _write_grouped_with_qc(tmp_path, hm, name="qc_grouped.zarr"):
    """Write a chromosome-grouped scikit-allel store with QC arrays.

    Mirrors the ``grouped_store`` fixture in test_zarr_io but reuses one
    chromosome so the QC fields can be asserted against a single
    msprime-derived matrix.
    """
    path = str(tmp_path / name)
    store = zarr.open(path, mode="w")
    chrom = "chr1"
    hap = hm.haplotypes
    pos = hm.positions
    gt = HaplotypeMatrix._haplotypes_to_gt(hap)
    grp = store.create_group(chrom)
    grp.create_array("calldata/GT", data=gt)
    grp.create_array("variants/POS", data=pos)
    if hm.samples is not None:
        grp.create_array("samples",
                          data=np.array(hm.samples, dtype="U"))
    n_var = int(hap.shape[1])
    n_samples = int(hap.shape[0] // 2)
    mq = np.linspace(5.0, 50.0, n_var, dtype=np.float32)
    rng = np.random.default_rng(11)
    gq = rng.integers(0, 99, size=(n_var, n_samples), dtype=np.int16)
    grp.create_array("variants/MQ", data=mq)
    grp.create_array("calldata/GQ", data=gq)
    region = f"{chrom}:{int(pos[0])}-{int(pos[-1]) + 1}"
    return path, region, mq, gq


def _write_small_vcf(tmp_path, name="small.vcf"):
    """Write a 4-variant, 3-sample VCF with GT, GQ, DP, and INFO/MQ.

    Hand-written so the test doesn't depend on bcftools, pysam, etc.
    """
    body = textwrap.dedent(
        """\
        ##fileformat=VCFv4.2
        ##contig=<ID=chr1,length=10000>
        ##INFO=<ID=MQ,Number=1,Type=Float,Description="Mapping quality">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
        chr1\t100\t.\tA\tG\t30\tPASS\tMQ=50\tGT:GQ:DP\t0|0:20:10\t0|1:30:12\t1|1:40:15
        chr1\t200\t.\tC\tT\t30\tPASS\tMQ=35\tGT:GQ:DP\t0|1:25:8\t1|1:15:5\t0|0:35:14
        chr1\t300\t.\tG\tA\t30\tPASS\tMQ=55\tGT:GQ:DP\t1|1:45:20\t0|0:50:18\t0|1:55:22
        chr1\t400\t.\tT\tC\t30\tPASS\tMQ=42\tGT:GQ:DP\t0|0:33:11\t1|1:28:9\t0|1:40:13
        """
    )
    path = tmp_path / name
    path.write_text(body)
    return str(path)


# ── HaplotypeMatrix.from_zarr ─────────────────────────────────────────────


class TestHaplotypeMatrixFromZarrFields:

    def test_vcz_info_field_shape(self, tmp_path):
        hm = _simulate_hm()
        path, mq, _, _ = _write_vcz_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(path, fields=["MQ"],
                                           streaming="never")
        assert "MQ" in loaded.fields
        assert loaded.fields["MQ"].shape == (mq.shape[0],)
        np.testing.assert_array_equal(loaded.fields["MQ"], mq)

    def test_vcz_format_field_shape(self, tmp_path):
        hm = _simulate_hm()
        path, _, gq, _ = _write_vcz_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(path, fields=["GQ"],
                                           streaming="never")
        assert "GQ" in loaded.fields
        assert loaded.fields["GQ"].shape == gq.shape
        np.testing.assert_array_equal(loaded.fields["GQ"], gq)

    def test_vcz_both_kinds_together(self, tmp_path):
        hm = _simulate_hm()
        path, mq, gq, dp = _write_vcz_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(
            path, fields=["MQ", "GQ", "DP"], streaming="never")
        assert set(loaded.fields) == {"MQ", "GQ", "DP"}
        np.testing.assert_array_equal(loaded.fields["MQ"], mq)
        np.testing.assert_array_equal(loaded.fields["GQ"], gq)
        np.testing.assert_array_equal(loaded.fields["DP"], dp)

    def test_missing_field_warns_and_drops(self, tmp_path):
        hm = _simulate_hm()
        path, mq, _, _ = _write_vcz_with_qc(tmp_path, hm)
        with pytest.warns(UserWarning, match="NONEXISTENT"):
            loaded = HaplotypeMatrix.from_zarr(
                path, fields=["MQ", "NONEXISTENT"], streaming="never")
        assert set(loaded.fields) == {"MQ"}
        np.testing.assert_array_equal(loaded.fields["MQ"], mq)

    def test_no_fields_kwarg_leaves_fields_empty(self, tmp_path):
        hm = _simulate_hm()
        path, _, _, _ = _write_vcz_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(path, streaming="never")
        assert loaded.fields == {}

    def test_allel_flat_layout_loads_fields(self, tmp_path):
        hm = _simulate_hm()
        path, mq, gq = _write_allel_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(path, fields=["MQ", "GQ"],
                                           streaming="never")
        assert set(loaded.fields) == {"MQ", "GQ"}
        np.testing.assert_array_equal(loaded.fields["MQ"], mq)
        np.testing.assert_array_equal(loaded.fields["GQ"], gq)

    def test_allel_grouped_layout_loads_fields(self, tmp_path):
        hm = _simulate_hm()
        path, region, mq, gq = _write_grouped_with_qc(tmp_path, hm)
        loaded = HaplotypeMatrix.from_zarr(
            path, region=region, fields=["MQ", "GQ"], streaming="never")
        assert set(loaded.fields) == {"MQ", "GQ"}
        np.testing.assert_array_equal(loaded.fields["MQ"], mq)
        np.testing.assert_array_equal(loaded.fields["GQ"], gq)

    def test_region_subsets_keep_qc_aligned(self, tmp_path):
        hm = _simulate_hm()
        path, mq, gq, _ = _write_vcz_with_qc(tmp_path, hm)
        # Take the middle ~half of positions.
        pos = _host(hm.positions)
        lo = int(pos[len(pos) // 4])
        hi = int(pos[3 * len(pos) // 4])
        region = f"chr1:{lo}-{hi + 1}"
        loaded = HaplotypeMatrix.from_zarr(
            path, region=region, fields=["MQ", "GQ"], streaming="never")
        n_loaded = loaded.haplotypes.shape[1]
        # The QC arrays must have been sliced down to match the loaded
        # variant axis, not still hold the full-store length.
        assert loaded.fields["MQ"].shape == (n_loaded,)
        assert loaded.fields["GQ"].shape[0] == n_loaded

    def test_streaming_with_fields_raises(self, tmp_path):
        hm = _simulate_hm()
        path, _, _, _ = _write_vcz_with_qc(tmp_path, hm)
        with pytest.raises(NotImplementedError, match="streaming"):
            HaplotypeMatrix.from_zarr(path, fields=["MQ"], streaming="always")


# ── GenotypeMatrix.from_zarr ──────────────────────────────────────────────


class TestGenotypeMatrixFromZarrFields:

    def test_vcz_loads_fields(self, tmp_path):
        hm = _simulate_hm()
        path, mq, gq, _ = _write_vcz_with_qc(tmp_path, hm)
        gm = GenotypeMatrix.from_zarr(path, fields=["MQ", "GQ"],
                                       streaming="never")
        assert set(gm.fields) == {"MQ", "GQ"}
        np.testing.assert_array_equal(gm.fields["MQ"], mq)
        np.testing.assert_array_equal(gm.fields["GQ"], gq)

    def test_streaming_with_fields_raises(self, tmp_path):
        hm = _simulate_hm()
        path, _, _, _ = _write_vcz_with_qc(tmp_path, hm)
        with pytest.raises(NotImplementedError, match="streaming"):
            GenotypeMatrix.from_zarr(path, fields=["MQ"], streaming="always")


# ── *.from_vcf ────────────────────────────────────────────────────────────


class TestFromVcfFields:

    def test_haplotype_matrix_loads_info_and_format(self, tmp_path):
        vcf = _write_small_vcf(tmp_path)
        hm = HaplotypeMatrix.from_vcf(vcf, fields=["MQ", "GQ", "DP"])
        assert set(hm.fields) == {"MQ", "GQ", "DP"}
        # 4 variants, 3 samples
        np.testing.assert_array_equal(
            hm.fields["MQ"], np.array([50.0, 35.0, 55.0, 42.0]))
        np.testing.assert_array_equal(
            hm.fields["GQ"],
            np.array([[20, 30, 40], [25, 15, 35],
                      [45, 50, 55], [33, 28, 40]]))
        np.testing.assert_array_equal(
            hm.fields["DP"],
            np.array([[10, 12, 15], [8, 5, 14],
                      [20, 18, 22], [11, 9, 13]]))

    def test_genotype_matrix_loads_info_and_format(self, tmp_path):
        vcf = _write_small_vcf(tmp_path)
        gm = GenotypeMatrix.from_vcf(vcf, fields=["MQ", "GQ"])
        # All four variants are biallelic so nothing is dropped by the
        # biallelic filter; the QC arrays match the input row-for-row.
        np.testing.assert_array_equal(
            gm.fields["MQ"], np.array([50.0, 35.0, 55.0, 42.0]))
        np.testing.assert_array_equal(
            gm.fields["GQ"],
            np.array([[20, 30, 40], [25, 15, 35],
                      [45, 50, 55], [33, 28, 40]]))

    def test_missing_field_warns_and_drops(self, tmp_path):
        vcf = _write_small_vcf(tmp_path)
        with pytest.warns(UserWarning, match="NONEXISTENT"):
            hm = HaplotypeMatrix.from_vcf(vcf, fields=["MQ", "NONEXISTENT"])
        assert set(hm.fields) == {"MQ"}
