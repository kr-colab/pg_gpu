"""Argument-validation, recombination-map, and progress-report coverage for the
moments.LD-format LD pipeline (pg_gpu.moments_ld).

compute_ld_statistics validates its argument combinations before doing any
work, and the recombination-distance path parses a genetic map and derives a
physical-distance cap. These drive the guard, parse, and report branches the
parity tests skip. The module needs no moments install, so these run in the
default environment.
"""
from contextlib import nullcontext

import numpy as np
import cupy as cp
import pytest

from pg_gpu import BiallelicOnlyWarning, GenotypeMatrix, HaplotypeMatrix
from pg_gpu import ld_statistics
from pg_gpu.ld_pipeline import compute_counts_for_pairs as _compute_counts_for_pairs
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
    """Full runs over each compute path: progress reporting on and off,
    recombination-distance binning, and the union filter across
    populations, each returning the moments-format result dict."""

    def test_genotype_report_run(self):
        gm = _gm()
        gm.sample_sets = {"pop0": list(range(12))}
        res = compute_ld_statistics(genotype_matrix=gm, use_genotypes=True,
                                    pops=["pop0"], bp_bins=[0, 500, 4000],
                                    report=True)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        assert res["pops"] == ["pop0"]

    def test_recombination_binned_run(self, tmp_path, capsys):
        # r_bins routes through the genetic-map interpolation and the physical
        # distance cap, then bins pairs by recombination distance. The map is
        # 1e-5 M/bp, so bins up to 1e-2 M admit pairs up to about 1 kb apart;
        # the inner edge sits between the 100 bp multiples of the variant
        # spacing so no pair lands on a bin boundary.
        rec = tmp_path / "rmap.map"
        rec.write_text("0 0.0\n5000 5.0\n")
        hm = _hm()
        hm.sample_sets = {"pop0": list(range(8))}
        res = compute_ld_statistics(haplotype_matrix=hm, use_genotypes=False,
                                    pops=["pop0"], r_bins=[0, 1.5e-3, 1e-2],
                                    rec_map_file=str(rec), report=False)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        # Both recombination bins receive pairs; the last entry holds H.
        assert all(np.any(np.asarray(s) != 0) for s in res["sums"][:-1])
        # report=False keeps the run silent.
        assert capsys.readouterr().out == ""

    def test_shared_individual_counted_once_in_union_filter(self):
        # The biallelic filter runs over the union of the populations and
        # counts an individual listed in both once. A site whose only call is
        # that individual's stays below the two-call minimum, so no pair forms
        # and every LD bin stays at zero.
        geno = np.ones((12, 2), dtype=np.int8)  # site 0: heterozygous in all
        geno[:, 1] = -1
        geno[4, 1] = 1                           # site 1: one call, individual 4
        gm = GenotypeMatrix(geno, np.array([100, 200]))
        gm.sample_sets = {"pop0": list(range(0, 8)), "pop1": list(range(4, 12))}
        res = compute_ld_statistics(genotype_matrix=gm, use_genotypes=True,
                                    pops=["pop0", "pop1"], bp_bins=[0, 500],
                                    report=False)
        assert all(np.all(np.asarray(s) == 0) for s in res["sums"][:-1])

    def test_haplotype_report_run(self):
        hm = _hm()
        hm.sample_sets = {"pop0": list(range(8))}
        res = compute_ld_statistics(haplotype_matrix=hm, use_genotypes=False,
                                    pops=["pop0"], bp_bins=[0, 500, 4000],
                                    report=True)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        assert res["pops"] == ["pop0"]


class TestVcfLoadingPath:
    """The VCF-loading branches -- from_vcf, load_pop_file, the
    accessible_bed mask attached at load, and the progress line -- on both
    compute paths."""

    @pytest.fixture
    def pop_and_bed(self, tmp_path):
        pop = tmp_path / "pops.tsv"
        pop.write_text("sample\tpop\n"
                       + "".join(f"tsk_{i}\tpop0\n" for i in range(10)))
        bed = tmp_path / "acc.bed"
        bed.write_text("1\t0\t1000\n")
        return str(pop), str(bed)

    @pytest.mark.parametrize("use_genotypes", [True, False],
                             ids=["genotype", "haplotype"])
    def test_vcf_with_accessible_bed(self, sample_vcf, pop_and_bed,
                                     use_genotypes, capsys):
        pop, bed = pop_and_bed
        # The genotype loader is biallelic-only and drops the fixture's
        # multiallelic sites with a warning; the haplotype loader keeps them.
        expect = (pytest.warns(BiallelicOnlyWarning) if use_genotypes
                  else nullcontext())
        with expect:
            res = compute_ld_statistics(vcf_file=sample_vcf, pop_file=pop,
                                        pops=["pop0"], bp_bins=[0, 100, 1000],
                                        use_genotypes=use_genotypes,
                                        accessible_bed=bed, report=True)
        assert set(res) == {"bins", "sums", "stats", "pops"}
        assert res["pops"] == ["pop0"]
        assert f"Loading {sample_vcf}" in capsys.readouterr().out


class TestMissingDataSums:
    """compute_ld_statistics's raw sums must exclude an undefined pair
    (NaN) rather than leak it, and the sum of the remaining pairs must be
    exact."""

    def _three_site_matrix(self):
        # Site 0 vs site 1 share 0 jointly-valid haplotypes (undefined);
        # site 0 vs site 2 share 3 (< 4, still undefined); site 1 vs site 2
        # share 17 (fully defined) -- exactly one of the three pairs
        # carries any information.
        hap = np.zeros((20, 3), dtype=np.int8)
        hap[:, 0] = [0, 1] * 10
        hap[:, 1] = [1, 0] * 10
        hap[:, 2] = [0, 1] * 10
        hap[3:, 0] = -1
        hap[0:3, 1] = -1
        hm = HaplotypeMatrix(hap, np.array([100, 200, 300], dtype=np.int64),
                            0, 1000)
        hm.sample_sets = {"pop0": list(range(20))}
        return hm

    def test_single_pop_sum_has_no_nan_and_matches_the_defined_pair(self):
        hm = self._three_site_matrix()
        out = compute_ld_statistics(haplotype_matrix=hm, pops=["pop0"],
                                    bp_bins=[0, 1000], use_genotypes=False,
                                    report=False)
        sums = out["sums"][0]
        assert not np.any(np.isnan(sums))

        # Hand reference: only pair (site1, site2) is defined; its own
        # per-pair DD/Dz/pi2 is exactly what the sum should equal, since
        # the two undefined pairs must contribute nothing.
        idx_i = cp.array([1], dtype=cp.int32)
        idx_j = cp.array([2], dtype=cp.int32)
        hm.transfer_to_gpu()
        counts, n_valid = _compute_counts_for_pairs(
            hm.haplotypes, idx_i, idx_j, has_missing=True)
        expected = [
            float(ld_statistics.dd(counts, n_valid=n_valid)[0]),
            float(ld_statistics.dz(counts, n_valid=n_valid)[0]),
            float(ld_statistics.pi2(counts, n_valid=n_valid)[0]),
        ]
        names = out["stats"][0]
        np.testing.assert_allclose(
            [sums[names.index(n)] for n in ("DD_0_0", "Dz_0_0_0", "pi2_0_0_0_0")],
            expected, rtol=1e-9)

    def test_haplotype_matrix_mean_excludes_rather_than_dilutes(self):
        # Same fixture through HaplotypeMatrix.compute_ld_statistics_gpu_
        # single_pop's averaged (raw=False) path instead of the raw-sum
        # pipeline above: the single bin holds one defined and two
        # undefined pairs, so the mean must equal the defined pair's own
        # value, not that value diluted by the total pair count (/3).
        hm = self._three_site_matrix()
        stats = hm.compute_ld_statistics_gpu_single_pop(
            bp_bins=[0, 1000], raw=False)[(0.0, 1000.0)]
        assert not any(np.isnan(s) for s in stats)

        idx_i = cp.array([1], dtype=cp.int32)
        idx_j = cp.array([2], dtype=cp.int32)
        hm.transfer_to_gpu()
        counts, n_valid = _compute_counts_for_pairs(
            hm.haplotypes, idx_i, idx_j, has_missing=True)
        expected = [
            float(ld_statistics.dd(counts, n_valid=n_valid)[0]),
            float(ld_statistics.dz(counts, n_valid=n_valid)[0]),
            float(ld_statistics.pi2(counts, n_valid=n_valid)[0]),
        ]
        np.testing.assert_allclose(stats, expected, rtol=1e-9)

    @pytest.mark.parametrize("use_genotypes", [True, False],
                             ids=["genotype", "haplotype"])
    def test_three_pop_sum_has_no_nan(self, use_genotypes):
        # 3 populations with real missingness: the N-population fused
        # kernel path (not the 2-pop fast path) must also exclude an
        # undefined pair from the sum rather than poison it with NaN.
        rng = np.random.default_rng(7)
        n_hap, n_var = 60, 8
        hap = (rng.random((n_hap, n_var)) < 0.4).astype(np.int8)
        for v in range(n_var):
            if hap[:, v].sum() in (0, n_hap):
                hap[0, v] = 1 - hap[0, v]
        hap[rng.random((n_hap, n_var)) < 0.15] = -1
        pos = (np.arange(n_var) + 1) * 100
        hm = HaplotypeMatrix(hap, pos.astype(np.int64), 0, 1000)
        hm.sample_sets = {"p0": list(range(20)), "p1": list(range(20, 40)),
                          "p2": list(range(40, 60))}
        out = compute_ld_statistics(
            haplotype_matrix=hm, pops=["p0", "p1", "p2"], bp_bins=[0, 1000],
            use_genotypes=use_genotypes, report=False)
        assert not np.any(np.isnan(out["sums"][0]))
