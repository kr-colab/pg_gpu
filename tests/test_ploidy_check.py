"""Haploid / hemizygous detection at load time.

pg_gpu's loaders are diploid-only. These tests pin the behaviour added to make
non-diploid input fail loudly instead of silently producing doubled haplotype
counts and biased statistics: genuine haploid calls raise, and samples that
look hemizygous (homozygous at every variant they carry) warn.
"""
import os
import tempfile
import warnings

import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, HaploidDataWarning
from pg_gpu._warnings import check_diploid_encoding


def _i8(rows):
    return np.array(rows, dtype=np.int8)


# --- direct unit tests of the detector -------------------------------------

def test_normal_diploid_passes_cleanly():
    # Every sample is heterozygous at site 0, so nothing looks haploid.
    gt = _i8([[[0, 1], [1, 0]],
              [[1, 1], [0, 0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        check_diploid_encoding(gt, sample_names=["a", "b"], source="test")


def test_true_haploid_padding_raises():
    # scikit-allel pads GT=1 to [1, -1]: first allele called, second missing.
    gt = _i8([[[0, -1], [1, -1]]])
    with pytest.raises(ValueError, match="haploid genotype calls"):
        check_diploid_encoding(gt, sample_names=["m1", "m2"], source="test")


def test_haploid_error_names_samples():
    gt = _i8([[[0, 1], [1, -1]]])  # only the second sample is haploid
    with pytest.raises(ValueError, match="m2"):
        check_diploid_encoding(gt, sample_names=["f1", "m2"], source="test")


def test_mixed_haploid_and_diploid_raises():
    # One sample haploid-padded, one a normal diploid het: still an error,
    # since the haploid sample cannot be loaded correctly.
    gt = _i8([[[0, -1], [0, 1]]])
    with pytest.raises(ValueError, match="haploid"):
        check_diploid_encoding(gt, source="test")


def test_pseudo_diploid_homozygous_warns():
    # Polymorphic but no heterozygote anywhere -- looks hemizygous.
    gt = _i8([[[1, 1], [0, 0]],
              [[0, 0], [1, 1]]])
    with pytest.warns(HaploidDataWarning, match="no heterozygous"):
        check_diploid_encoding(gt, sample_names=["x", "y"], source="test")


def test_one_heterozygote_suppresses_warning():
    # A single real heterozygote anywhere rules out the haploid encoding, so
    # ordinary diploid data with some incidentally homozygous samples is quiet.
    gt = _i8([[[1, 1], [0, 1]],   # sample 1 is heterozygous here
              [[0, 0], [1, 0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_diploid_encoding(gt, sample_names=["hom", "het"], source="test")


def test_monomorphic_reference_does_not_warn():
    # No alternate allele present -> nothing to diagnose, no warning.
    gt = _i8([[[0, 0], [0, 0]],
              [[0, 0], [0, 0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_diploid_encoding(gt, source="test")


def test_partial_missing_diploid_not_treated_as_haploid():
    # A diploid sample with a half-missing call (GT=1/.) yields [1, -1], the
    # same shape as haploid padding -- but its other sites are fully called,
    # so it must not be misread as haploid.
    gt = _i8([[[1, -1], [0, 1]],   # sample 0 half-missing here
              [[0, 1], [1, 0]]])   # ... but fully called (and het) here
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_diploid_encoding(gt, source="test")


def test_fully_missing_diploid_not_treated_as_haploid():
    # [-1, -1] is a missing diploid, not a padded haploid.
    gt = _i8([[[-1, -1], [0, 1]],
              [[0, 1], [1, 0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_diploid_encoding(gt, source="test")


@pytest.mark.parametrize("ploidy", [1, 3, 4])
def test_non_diploid_ploidy_raises(ploidy):
    gt = np.zeros((4, 3, ploidy), dtype=np.int8)
    with pytest.raises(ValueError, match=f"ploidy {ploidy}"):
        check_diploid_encoding(gt, source="test")


# --- integration through the VCF loaders -----------------------------------

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chrX>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tm1\tm2\tm3\n"
)


def _write_vcf(rows):
    text = _HEADER + "\n".join(rows) + "\n"
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False)
    f.write(text)
    f.close()
    return f.name


@pytest.fixture
def haploid_vcf():
    path = _write_vcf([
        "chrX\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0\t1\t1",
        "chrX\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1\t1\t0",
    ])
    yield path
    os.unlink(path)


@pytest.fixture
def pseudo_diploid_vcf():
    path = _write_vcf([
        "chrX\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t1/1\t1/1",
        "chrX\t200\t.\tC\tG\t.\tPASS\t.\tGT\t1/1\t1/1\t0/0",
    ])
    yield path
    os.unlink(path)


def test_haplotype_from_vcf_rejects_haploid(haploid_vcf):
    with pytest.raises(ValueError, match="haploid genotype calls"):
        HaplotypeMatrix.from_vcf(haploid_vcf)


def test_genotype_from_vcf_rejects_haploid(haploid_vcf):
    with pytest.raises(ValueError, match="haploid genotype calls"):
        GenotypeMatrix.from_vcf(haploid_vcf)


def test_haplotype_from_vcf_warns_pseudo_diploid(pseudo_diploid_vcf):
    with pytest.warns(HaploidDataWarning):
        HaplotypeMatrix.from_vcf(pseudo_diploid_vcf)


def test_genotype_from_vcf_warns_pseudo_diploid(pseudo_diploid_vcf):
    with pytest.warns(HaploidDataWarning):
        GenotypeMatrix.from_vcf(pseudo_diploid_vcf)


def test_normal_vcf_loads_without_warning(simple_vcf_file):
    with warnings.catch_warnings():
        warnings.simplefilter("error", HaploidDataWarning)
        hm = HaplotypeMatrix.from_vcf(simple_vcf_file)
    assert hm.num_haplotypes == 4  # 2 diploid samples -> 4 haplotypes
