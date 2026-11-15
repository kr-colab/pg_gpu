"""
Validation tests comparing pg_gpu SFS functions against scikit-allel.
"""

import pytest
import numpy as np
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import sfs


@pytest.fixture
def single_pop_data():
    """Deterministic single-population haplotype data."""
    np.random.seed(42)
    n_hap = 20
    n_var = 100
    hap = np.random.randint(0, 2, (n_hap, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    matrix = HaplotypeMatrix(hap, pos, 0, n_var * 1000)
    return matrix, hap


@pytest.fixture
def two_pop_data():
    """Deterministic two-population data."""
    np.random.seed(123)
    n_var = 100
    n1, n2 = 10, 12
    hap = np.random.randint(0, 2, (n1 + n2, n_var), dtype=np.int8)
    pos = np.arange(n_var) * 1000
    matrix = HaplotypeMatrix(
        hap, pos, 0, n_var * 1000,
        sample_sets={'pop1': list(range(n1)), 'pop2': list(range(n1, n1 + n2))}
    )
    return matrix, hap, n1, n2


class TestSFSComparison:
    """Validate SFS against scikit-allel."""

    def test_sfs(self, single_pop_data):
        matrix, hap = single_pop_data
        n = hap.shape[0]
        # pg_gpu
        result = sfs.sfs(matrix)
        # allel: pass n explicitly to match full-size output
        dac = np.sum(hap, axis=0)
        expected = allel.sfs(dac, n=n)
        # sfs.sfs is now the per-allele polarised SFS (tskit convention): both
        # fixed classes (bins 0 and n) are excluded, whereas allel.sfs includes
        # them. Compare the segregating interior; assert the endpoints are dropped.
        np.testing.assert_array_equal(result[1:n], expected[1:n])
        assert result[0] == 0 and result[n] == 0

    def test_sfs_folded(self, single_pop_data):
        matrix, hap = single_pop_data
        result = sfs.sfs_folded(matrix)
        # allel: needs allele counts (n_variants, 2)
        dac = np.sum(hap, axis=0)
        n = hap.shape[0]
        ac = np.column_stack([n - dac, dac])
        expected = allel.sfs_folded(ac)
        np.testing.assert_array_equal(result, expected)

    def test_sfs_scaled(self, single_pop_data):
        matrix, hap = single_pop_data
        n = hap.shape[0]
        result = sfs.sfs_scaled(matrix)
        dac = np.sum(hap, axis=0)
        expected = allel.sfs_scaled(dac, n=n)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sfs_folded_scaled(self, single_pop_data):
        matrix, hap = single_pop_data
        result = sfs.sfs_folded_scaled(matrix)
        dac = np.sum(hap, axis=0)
        n = hap.shape[0]
        ac = np.column_stack([n - dac, dac])
        expected = allel.sfs_folded_scaled(ac)
        np.testing.assert_array_almost_equal(result, expected)


class TestJointSFSComparison:
    """Validate joint SFS against scikit-allel."""

    def test_joint_sfs(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        expected = allel.joint_sfs(dac1, dac2, n1=n1, n2=n2)
        np.testing.assert_array_equal(result, expected)

    # NOTE: joint_sfs_folded / joint_sfs_folded_scaled deliberately do NOT match
    # scikit-allel. The 2D fold is the one place tskit and allel disagree even on
    # biallelic data -- allel folds each axis independently, tskit folds the site
    # as a unit by the global minor. We pin to tskit (see the ledger). Parity is
    # tested in test_multiallelic.py::TestJointSFS::test_joint_sfs_folded_matches_tskit.

    def test_joint_sfs_scaled(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.joint_sfs_scaled(matrix, 'pop1', 'pop2')
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        expected = allel.joint_sfs_scaled(dac1, dac2, n1=n1, n2=n2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_joint_sfs_no_segregating_alleles(self):
        # No allele segregates in the A+B cohort, so the joint SFS is all zeros.
        n1, n2 = 4, 3
        hap = np.zeros((n1 + n2, 5), dtype=np.int8)
        hm = HaplotypeMatrix(
            hap, np.arange(5) * 1000, 0, 5000,
            sample_sets={'p1': list(range(n1)),
                         'p2': list(range(n1, n1 + n2))})
        result = sfs.joint_sfs(hm, 'p1', 'p2')
        assert result.shape == (n1 + 1, n2 + 1)
        np.testing.assert_array_equal(result, 0)


class TestProjectionSandwich:
    """Validate project_joint_sfs against the explicit P1 @ S @ P2.T."""

    def test_eager_matches_sandwich(self, two_pop_data):
        # Reference: build the full joint SFS with allel, then apply
        # the exact-integer projection matrix on each axis. The new
        # function uses a gammaln-based projection matrix instead, but
        # both should agree to float64 round-off for small n.
        from pg_gpu.diversity import _projection_matrix
        matrix, hap, n1, n2 = two_pop_data
        target1, target2 = 4, 5
        dac1 = np.sum(hap[:n1], axis=0)
        dac2 = np.sum(hap[n1:], axis=0)
        ref_full = allel.joint_sfs(dac1, dac2, n1=n1, n2=n2)
        P1 = _projection_matrix(n1, target1)
        P2 = _projection_matrix(n2, target2)
        expected = P1 @ ref_full @ P2.T
        result = sfs.project_joint_sfs(matrix, "pop1", "pop2",
                                        target_n1=target1,
                                        target_n2=target2)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)

    def test_identity_target_recovers_full(self, two_pop_data):
        # When target_n equals source n, hypergeometric pmf is the
        # identity so projection collapses to a float cast of joint_sfs.
        matrix, hap, n1, n2 = two_pop_data
        result = sfs.project_joint_sfs(matrix, "pop1", "pop2",
                                        target_n1=n1, target_n2=n2)
        full = sfs.joint_sfs(matrix, "pop1", "pop2")
        np.testing.assert_allclose(result, full.astype(np.float64),
                                    rtol=1e-9, atol=1e-9)

    def test_over_projection_raises(self, two_pop_data):
        matrix, hap, n1, n2 = two_pop_data
        with pytest.raises(ValueError, match="Cannot project up"):
            sfs.project_joint_sfs(matrix, "pop1", "pop2",
                                  target_n1=n1 + 1, target_n2=n2)

    def test_no_alleles_returns_zeros(self):
        # Monomorphic input: no allele survives the cohort filter.
        n1, n2 = 5, 4
        hap = np.zeros((n1 + n2, 8), dtype=np.int8)
        hm = HaplotypeMatrix(
            hap, np.arange(8) * 1000, 0, 8000,
            sample_sets={'p1': list(range(n1)),
                         'p2': list(range(n1, n1 + n2))})
        result = sfs.project_joint_sfs(hm, "p1", "p2",
                                       target_n1=3, target_n2=2)
        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result, 0.0)


class TestScalingComparison:
    """Validate scaling and folding utilities."""

    def test_scale_sfs(self):
        s = np.array([10, 5, 3, 2, 1])
        np.testing.assert_array_equal(
            sfs.scale_sfs(s), allel.scale_sfs(s))

    def test_scale_sfs_folded(self):
        s = np.array([10, 5, 3])
        np.testing.assert_array_almost_equal(
            sfs.scale_sfs_folded(s, 4), allel.scale_sfs_folded(s, 4))

    def test_scale_joint_sfs(self):
        s = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(
            sfs.scale_joint_sfs(s), allel.scale_joint_sfs(s))

    def test_fold_sfs_odd_n_matches_allel(self):
        # scikit-allel's fold_sfs is correct for odd n; even n is not,
        # so it is not a reference there -- see test_fold_sfs_even_n below.
        for n in [7, 9, 21]:
            np.random.seed(42)
            s = np.random.randint(0, 20, size=n + 1)
            np.testing.assert_array_equal(
                sfs.fold_sfs(s, n), allel.fold_sfs(s, n))

    def test_fold_sfs_even_n_matches_sfs_folded(self):
        # sfs_folded computes the fold directly from per-site allele counts,
        # independently of fold_sfs's array manipulation; the two must agree
        # on biallelic data, where fold_sfs(sfs(hm), n) is the same quantity.
        rng = np.random.default_rng(1)
        for n_hap in [8, 20]:
            hap = (rng.random((n_hap, 60)) < 0.4).astype(np.int8)
            hm = HaplotypeMatrix(hap, np.arange(60) * 10 + 1, 1, 1000)
            np.testing.assert_array_equal(
                sfs.fold_sfs(sfs.sfs(hm), n_hap), sfs.sfs_folded(hm))

    def test_fold_sfs_pads_short_input(self):
        # A short unfolded SFS is zero-padded to n+1 before folding.
        n = 8
        short = np.array([0, 3, 5, 2])
        padded = np.zeros(n + 1, dtype=short.dtype)
        padded[:len(short)] = short
        np.testing.assert_array_equal(sfs.fold_sfs(short, n),
                                      sfs.fold_sfs(padded, n))

    @staticmethod
    def _ref_fold_joint_sfs(s, n1, n2):
        """Brute-force per-cell reference: bin (i, j) reflects past the
        midpoint of each axis independently, the definition fold_joint_sfs
        implements. scikit-allel is not a reference for an even axis."""
        m1, m2 = n1 // 2, n2 // 2
        o = np.zeros((m1 + 1, m2 + 1), dtype=np.asarray(s).dtype)
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                oi = i if i <= m1 else n1 - i
                oj = j if j <= m2 else n2 - j
                o[oi, oj] += s[i, j]
        return o

    def test_fold_joint_sfs(self):
        rng = np.random.default_rng(0)
        for n1, n2 in [(6, 5), (7, 4), (8, 8), (7, 5), (9, 9)]:
            s = rng.integers(0, 20, size=(n1 + 1, n2 + 1))
            np.testing.assert_array_equal(
                sfs.fold_joint_sfs(s, n1, n2),
                self._ref_fold_joint_sfs(s, n1, n2))

    def test_fold_joint_sfs_pads_short_input(self):
        # A short joint SFS is zero-padded on both axes before folding.
        n1, n2 = 6, 5
        short = np.arange(1, 3 * 4 + 1).reshape(3, 4)
        padded = np.zeros((n1 + 1, n2 + 1), dtype=short.dtype)
        padded[:3, :4] = short
        np.testing.assert_array_equal(sfs.fold_joint_sfs(short, n1, n2),
                                      sfs.fold_joint_sfs(padded, n1, n2))


class TestProjectionMatrixVec:
    """Edge cases of the hypergeometric projection-matrix builder."""

    def test_rejects_upsampling(self):
        with pytest.raises(ValueError, match=r"n_to <= n_from"):
            sfs._projection_matrix_vec(5, 6)

    def test_projects_to_zero_sample(self):
        # A size-0 target puts all mass in the empty-sample bin.
        P = sfs._projection_matrix_vec(5, 0)
        assert P.shape == (1, 6)
        np.testing.assert_array_equal(P, 1.0)
