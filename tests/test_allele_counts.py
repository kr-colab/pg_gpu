"""
Tests for the per-allele allele-count primitive (_memutil.allele_counts).

This is the multiallelic-correct counting primitive behind the diversity
family (issue #100). Oracles: a cupy per-column bincount reference and
scikit-allel's count_alleles.
"""

import numpy as np
import cupy as cp
import allel
import pytest

from pg_gpu._memutil import allele_counts


def _ref_counts(hap, K):
    """Ground-truth per-allele counts via per-value reduction (n_hap, n_var)."""
    ac = np.zeros((hap.shape[1], K), dtype=np.int64)
    for a in range(K):
        ac[:, a] = (hap == a).sum(axis=0)
    return ac


class TestAlleleCounts:

    def test_biallelic(self):
        # site0: 3x ref, 1x alt ; site1: all alt
        hap = np.array([[0, 1],
                        [0, 1],
                        [0, 1],
                        [1, 1]], dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        assert ac.shape == (2, 2)
        np.testing.assert_array_equal(ac.get(), [[3, 1], [0, 4]])
        np.testing.assert_array_equal(n.get(), [4, 4])

    def test_triallelic(self):
        # one site: 2x allele0, 1x allele1, 1x allele2
        hap = np.array([[0], [0], [1], [2]], dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        assert ac.shape == (1, 3)
        np.testing.assert_array_equal(ac.get(), [[2, 1, 1]])
        np.testing.assert_array_equal(n.get(), [4])

    def test_tetraallelic(self):
        hap = np.array([[0], [1], [2], [3]], dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        np.testing.assert_array_equal(ac.get(), [[1, 1, 1, 1]])
        np.testing.assert_array_equal(n.get(), [4])

    def test_reference_absent(self):
        # site with only alleles 1 and 2 (ancestral index 0 absent)
        hap = np.array([[1], [1], [2], [2]], dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        np.testing.assert_array_equal(ac.get(), [[0, 2, 2]])
        np.testing.assert_array_equal(n.get(), [4])

    def test_missing_data(self):
        # -1 excluded from n_valid and from all allele columns
        hap = np.array([[0], [1], [-1], [2]], dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        np.testing.assert_array_equal(ac.get(), [[1, 1, 1]])
        np.testing.assert_array_equal(n.get(), [3])
        # counts sum to n_valid (K covers the range)
        assert int(ac.get().sum()) == int(n.get().sum())

    def test_sum_equals_n_valid(self):
        rng = np.random.default_rng(0)
        hap = rng.integers(-1, 4, size=(30, 200)).astype(np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        np.testing.assert_array_equal(ac.get().sum(axis=1), n.get())

    def test_matches_bincount_reference(self):
        rng = np.random.default_rng(1)
        hap = rng.integers(-1, 4, size=(40, 500)).astype(np.int8)
        ac, _ = allele_counts(cp.asarray(hap))
        K = ac.shape[1]
        np.testing.assert_array_equal(ac.get(), _ref_counts(hap, K))

    def test_matches_scikit_allel(self):
        rng = np.random.default_rng(2)
        hap = rng.integers(0, 4, size=(24, 300)).astype(np.int8)
        ac, _ = allele_counts(cp.asarray(hap))
        # allel expects (n_var, n_hap); count_alleles gives (n_var, n_alleles)
        allel_ac = np.asarray(allel.HaplotypeArray(hap.T).count_alleles())
        assert ac.shape == allel_ac.shape
        np.testing.assert_array_equal(ac.get(), allel_ac)

    def test_n_alleles_override_pads(self):
        hap = np.array([[0], [1], [0], [1]], dtype=np.int8)  # biallelic
        ac, n = allele_counts(cp.asarray(hap), n_alleles=4)
        assert ac.shape == (1, 4)
        np.testing.assert_array_equal(ac.get(), [[2, 2, 0, 0]])

    def test_all_missing(self):
        hap = np.full((5, 3), -1, dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        assert ac.shape == (3, 1)  # K floored to 1
        np.testing.assert_array_equal(ac.get(), np.zeros((3, 1)))
        np.testing.assert_array_equal(n.get(), [0, 0, 0])

    def test_empty(self):
        hap = np.empty((5, 0), dtype=np.int8)
        ac, n = allele_counts(cp.asarray(hap))
        assert ac.shape[0] == 0 and n.shape[0] == 0
