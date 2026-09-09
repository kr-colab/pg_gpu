"""Argument-validation, recombination-map, and progress-report coverage for the
moments.LD-format LD pipeline (pg_gpu.moments_ld).

compute_ld_statistics validates its argument combinations before doing any
work, and the recombination-distance path parses a genetic map and derives a
physical-distance cap. These drive the guard, parse, and report branches the
parity tests skip. The module needs no moments install, so these run in the
default environment.
"""
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.moments_ld import (compute_ld_statistics,
                               _interpolate_genetic_distances,
                               _max_bp_for_r_dist)


def _hm(n_hap=8, n_var=40, seed=0):
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, size=(n_hap, n_var)).astype(np.int8)
    return HaplotypeMatrix(hap, np.arange(1, n_var + 1) * 100, 0, n_var * 100)


def _gm(n_ind=12, n_var=40, seed=0):
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n_ind, n_var)).astype(np.int8)
    return GenotypeMatrix(geno, np.arange(1, n_var + 1) * 100)


class TestArgValidation:

    def test_pop_file_and_pop_assignment_conflict(self):
        with pytest.raises(TypeError, match="only one of pop_file"):
            compute_ld_statistics(pop_file="a", pop_assignment="b")

    def test_too_many_populations(self):
        with pytest.raises(ValueError, match="1-4 populations"):
            compute_ld_statistics(pops=list("abcde"), bp_bins=[0, 1000])

    def test_no_bins_provided(self):
        # pops omitted -> the default two-population pair, then the bins guard.
        with pytest.raises(ValueError, match="r_bins or bp_bins"):
            compute_ld_statistics()

    def test_pop_assignment_aliases_pop_file(self):
        # pop_assignment is an alias for pop_file; passing it alone is accepted
        # (the later bins guard is what raises here).
        with pytest.raises(ValueError, match="r_bins or bp_bins"):
            compute_ld_statistics(pop_assignment="x", pops=["p"])

    def test_genotype_path_needs_a_source(self):
        with pytest.raises(ValueError, match="vcf_file or genotype_matrix"):
            compute_ld_statistics(pops=["p"], bp_bins=[0, 1000],
                                  use_genotypes=True)

    def test_genotype_vcf_needs_pop_file(self):
        with pytest.raises(ValueError, match="pop_file is required"):
            compute_ld_statistics(pops=["p"], bp_bins=[0, 1000],
                                  use_genotypes=True, vcf_file="x.vcf")

    def test_haplotype_path_needs_a_source(self):
        with pytest.raises(ValueError, match="vcf_file or haplotype_matrix"):
            compute_ld_statistics(pops=["p"], bp_bins=[0, 1000],
                                  use_genotypes=False)

    def test_haplotype_vcf_needs_pop_file(self):
        with pytest.raises(ValueError, match="pop_file is required"):
            compute_ld_statistics(pops=["p"], bp_bins=[0, 1000],
                                  use_genotypes=False, vcf_file="x.vcf")

    def test_r_bins_needs_rec_map(self):
        with pytest.raises(ValueError, match="rec_map_file required"):
            compute_ld_statistics(haplotype_matrix=_hm(), use_genotypes=False,
                                  pops=["p"], r_bins=[0, 1e-5], report=False)


class TestRecMapHelpers:

    def test_interpolate_skips_malformed_lines(self, tmp_path):
        rec = tmp_path / "rec.map"
        rec.write_text("header line\n0 0.0\n1000 1.0\nbad row\n2000 2.0\n")
        gd = _interpolate_genetic_distances(np.array([500.0, 1500.0]), str(rec))
        # cM / 100 -> Morgans, linearly interpolated between the two valid rows.
        np.testing.assert_allclose(gd, [0.005, 0.015])

    def test_max_bp_single_position_returns_span(self):
        # Fewer than two positions -> the chromosome span (here zero).
        assert _max_bp_for_r_dist(np.array([100.0]), np.array([0.0]), 1e-5) == 0.0

    def test_max_bp_no_positive_bp_diff_returns_span(self):
        # Duplicate positions leave no positive physical gap.
        out = _max_bp_for_r_dist(np.array([100.0, 100.0]),
                                 np.array([0.0, 0.01]), 1e-5)
        assert out == 0.0

    def test_max_bp_flat_map_uses_rate_fallback(self):
        # A flat genetic map has no positive local rate, so the cap falls back
        # to max_r / span rather than a per-interval rate.
        out = _max_bp_for_r_dist(np.array([0.0, 1000.0]),
                                 np.array([0.0, 0.0]), 1e-5)
        assert np.isfinite(out) and out > 0

    def test_max_bp_normal_path(self):
        out = _max_bp_for_r_dist(np.array([0.0, 1000.0, 2000.0]),
                                 np.array([0.0, 0.01, 0.02]), 1e-5)
        assert out > 0


class TestReportRun:
    """A full report=True run over each compute path prints progress and
    returns the moments-format result dict."""

    def test_genotype_report_run(self):
        gm = _gm()
        gm.sample_sets = {"pop0": list(range(12))}
        res = compute_ld_statistics(genotype_matrix=gm, use_genotypes=True,
                                    pops=["pop0"], bp_bins=[0, 500, 4000],
                                    report=True)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        assert res["pops"] == ["pop0"]

    def test_recombination_binned_run(self, tmp_path):
        # r_bins routes through the genetic-map interpolation and the physical
        # distance cap, then bins pairs by recombination distance.
        rec = tmp_path / "rmap.map"
        rec.write_text("0 0.0\n5000 5.0\n")
        hm = _hm()
        hm.sample_sets = {"pop0": list(range(8))}
        res = compute_ld_statistics(haplotype_matrix=hm, use_genotypes=False,
                                    pops=["pop0"], r_bins=[0, 1e-5, 5e-5],
                                    rec_map_file=str(rec), report=False)
        assert set(res) == {"bins", "sums", "stats", "pops"}

    def test_haplotype_report_run(self):
        hm = _hm()
        hm.sample_sets = {"pop0": list(range(8))}
        res = compute_ld_statistics(haplotype_matrix=hm, use_genotypes=False,
                                    pops=["pop0"], bp_bins=[0, 500, 4000],
                                    report=True)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        assert res["pops"] == ["pop0"]
