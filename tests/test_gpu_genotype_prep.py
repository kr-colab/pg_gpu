"""Tests for build_haplotype_matrix: GPU-side prep of (n_var, n_dip, 2) blocks."""

import numpy as np
import pytest
import cupy as cp

from pg_gpu._gpu_genotype_prep import build_haplotype_matrix


def _reference_reshape(gt):
    """Host-only equivalent of build_haplotype_matrix's reshape, used as
    ground truth in equivalence tests. Matches the layout
    HaplotypeMatrix.load_pop_file expects (ploidy 0 first, then ploidy 1)."""
    n_var, n_dip, _ = gt.shape
    haps = np.empty((n_var, 2 * n_dip), dtype=gt.dtype)
    haps[:, :n_dip] = gt[:, :, 0]
    haps[:, n_dip:] = gt[:, :, 1]
    return haps.T


@pytest.fixture
def synthetic_gt():
    """A small but representative (n_var, n_dip, 2) int8 block with a couple
    of missing rows so the -1 propagation path is exercised."""
    rng = np.random.default_rng(0)
    n_var, n_dip = 100, 50
    gt = rng.integers(0, 2, size=(n_var, n_dip, 2), dtype=np.int8)
    gt[3] = -1
    gt[17] = -1
    gt[42] = -1
    pos = np.arange(n_var, dtype=np.int64) * 100 + 1000
    return gt, pos


class TestBuildHaplotypeMatrix:

    def test_layout_matches_load_pop_file_convention(self, synthetic_gt):
        gt, pos = synthetic_gt
        hm = build_haplotype_matrix(gt, pos)
        haps_host = cp.asnumpy(hm.haplotypes)
        np.testing.assert_array_equal(haps_host, _reference_reshape(gt))

    def test_missing_rows_preserved(self, synthetic_gt):
        # downstream kernels use 'include' / 'exclude' missing-data modes,
        # so the helper must keep -1 cells rather than dropping them.
        gt, pos = synthetic_gt
        hm = build_haplotype_matrix(gt, pos)
        haps_host = cp.asnumpy(hm.haplotypes)
        # rows 3, 17, 42 should be all -1 across haplotypes.
        assert (haps_host[:, 3] == -1).all()
        assert (haps_host[:, 17] == -1).all()
        assert (haps_host[:, 42] == -1).all()

    def test_positions_passed_through(self, synthetic_gt):
        gt, pos = synthetic_gt
        hm = build_haplotype_matrix(gt, pos)
        pos_host = cp.asnumpy(hm.positions)
        np.testing.assert_array_equal(pos_host, pos)

    def test_kwargs_forwarded(self, synthetic_gt):
        gt, pos = synthetic_gt
        hm = build_haplotype_matrix(
            gt, pos,
            chrom_start=1000, chrom_end=20000,
            samples=[f"s{i}" for i in range(50)],
            sample_sets={"all": list(range(100))},
        )
        assert hm.chrom_start == 1000
        assert hm.chrom_end == 20000
        assert hm.samples[0] == "s0"
        assert "all" in hm.sample_sets

    def test_accepts_device_inputs(self, synthetic_gt):
        gt, pos = synthetic_gt
        hm = build_haplotype_matrix(cp.asarray(gt), cp.asarray(pos))
        np.testing.assert_array_equal(cp.asnumpy(hm.haplotypes),
                                      _reference_reshape(gt))

    def test_mismatched_shapes_raise(self, synthetic_gt):
        gt, pos = synthetic_gt
        with pytest.raises(ValueError, match="disagree on n_var"):
            build_haplotype_matrix(gt, pos[:50])

    def test_wrong_ploidy_raises(self):
        gt = np.zeros((10, 5, 3), dtype=np.int8)
        pos = np.arange(10, dtype=np.int64)
        with pytest.raises(ValueError, match="must have shape"):
            build_haplotype_matrix(gt, pos)
