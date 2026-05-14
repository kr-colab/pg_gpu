"""Tests for ``BiobankScaleWarning`` and ``_maybe_biobank_warn``.

The size thresholds are kwargs on ``_maybe_biobank_warn`` so tests can
trip them on tiny on-disk fixtures rather than building genuine 10 GiB
VCFs.
"""

import os
import warnings

import pytest

from pg_gpu import BiobankScaleWarning
from pg_gpu._biobank_warning import (
    _maybe_biobank_warn, _region_span_bp, _vcf_header_sample_count,
    _warned_paths,
)


def _write_minimal_vcf(path, n_samples, n_variants=2):
    """Write a tiny synthetic VCF with the requested sample count and a
    few variants. Used to drive the header-parse path without needing
    a real msprime/bcftools roundtrip."""
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    )
    cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
            "INFO", "FORMAT"] + [f"s{i}" for i in range(n_samples)]
    header += "\t".join(cols) + "\n"

    variant_lines = []
    for v in range(n_variants):
        fields = ["1", str(100 * (v + 1)), ".", "A", "T", ".", "PASS",
                  ".", "GT"] + ["0|1"] * n_samples
        variant_lines.append("\t".join(fields))

    with open(path, "w") as f:
        f.write(header + "\n".join(variant_lines) + "\n")


@pytest.fixture(autouse=True)
def _clear_warned_cache():
    """``_warned_paths`` is process-global; clear between tests so each
    test starts from a clean slate."""
    _warned_paths.clear()
    yield
    _warned_paths.clear()


class TestHeaderSampleCount:

    def test_counts_samples_in_chrom_line(self, tmp_path):
        path = str(tmp_path / "small.vcf")
        _write_minimal_vcf(path, n_samples=7)
        assert _vcf_header_sample_count(path) == 7

    def test_handles_zero_samples(self, tmp_path):
        path = str(tmp_path / "nosamples.vcf")
        _write_minimal_vcf(path, n_samples=0)
        assert _vcf_header_sample_count(path) == 0

    def test_missing_file_returns_none(self, tmp_path):
        assert _vcf_header_sample_count(str(tmp_path / "nope.vcf")) is None


class TestRegionSpan:

    def test_parses_chrom_start_end(self):
        assert _region_span_bp("chr1:1000-5000") == 4000

    def test_none_is_zero(self):
        assert _region_span_bp(None) == 0

    def test_malformed_returns_zero(self):
        assert _region_span_bp("not a region") == 0


class TestMaybeBiobankWarn:

    def test_small_file_no_warn(self, tmp_path):
        path = str(tmp_path / "small.vcf")
        _write_minimal_vcf(path, n_samples=10)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiobankScaleWarning)
            _maybe_biobank_warn(path)  # default thresholds; ~few hundred bytes

    def test_big_file_no_region_warns(self, tmp_path):
        path = str(tmp_path / "big.vcf")
        _write_minimal_vcf(path, n_samples=10)
        # tiny file; spoof the byte threshold low so the size check trips
        size = os.path.getsize(path)
        with pytest.warns(BiobankScaleWarning, match="will be slow"):
            _maybe_biobank_warn(path, warn_bytes=size - 1)

    def test_big_file_small_region_does_not_warn(self, tmp_path):
        # The "right tool" case: huge VCF, but we asked for a small
        # tabix-region read. Don't warn -- region reads are fast.
        path = str(tmp_path / "big-tabix.vcf")
        _write_minimal_vcf(path, n_samples=10)
        size = os.path.getsize(path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiobankScaleWarning)
            _maybe_biobank_warn(path, region="1:0-1000",
                                warn_bytes=size - 1,
                                warn_region_bp=10_000)

    def test_many_samples_warns_regardless_of_size(self, tmp_path):
        # Sample count alone trips the warning even on a small file --
        # 10k-sample VCFs are slow to parse regardless.
        path = str(tmp_path / "many-samples.vcf")
        _write_minimal_vcf(path, n_samples=20)
        with pytest.warns(BiobankScaleWarning, match="20 samples"):
            _maybe_biobank_warn(path, warn_samples=10)

    def test_cached_per_path(self, tmp_path):
        path = str(tmp_path / "twice.vcf")
        _write_minimal_vcf(path, n_samples=20)
        with pytest.warns(BiobankScaleWarning):
            _maybe_biobank_warn(path, warn_samples=10)
        # Same path again should not re-warn within the process.
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiobankScaleWarning)
            _maybe_biobank_warn(path, warn_samples=10)

    def test_silenced_by_filter(self, tmp_path):
        path = str(tmp_path / "silenced.vcf")
        _write_minimal_vcf(path, n_samples=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiobankScaleWarning)
            _maybe_biobank_warn(path, warn_samples=10)


class TestFromVcfIntegration:

    def test_haplotype_matrix_from_vcf_warns(self, tmp_path):
        # End-to-end: from_vcf with a small sample threshold trips the
        # warning before parsing. Uses the existing simple_vcf_file
        # shape but with enough samples to clear the test threshold.
        from pg_gpu import HaplotypeMatrix
        from pg_gpu._biobank_warning import BIOBANK_VCF_WARN_SAMPLES
        path = str(tmp_path / "hm.vcf")
        _write_minimal_vcf(path, n_samples=BIOBANK_VCF_WARN_SAMPLES + 10)
        with pytest.warns(BiobankScaleWarning):
            HaplotypeMatrix.from_vcf(path)

    def test_genotype_matrix_from_vcf_warns(self, tmp_path):
        from pg_gpu import GenotypeMatrix
        from pg_gpu._biobank_warning import BIOBANK_VCF_WARN_SAMPLES
        path = str(tmp_path / "gm.vcf")
        _write_minimal_vcf(path, n_samples=BIOBANK_VCF_WARN_SAMPLES + 10)
        with pytest.warns(BiobankScaleWarning):
            GenotypeMatrix.from_vcf(path)

    def test_haplotype_matrix_from_vcf_small_does_not_warn(self, tmp_path):
        from pg_gpu import HaplotypeMatrix
        path = str(tmp_path / "small.vcf")
        _write_minimal_vcf(path, n_samples=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BiobankScaleWarning)
            HaplotypeMatrix.from_vcf(path)
