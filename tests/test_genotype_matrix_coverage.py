"""Edge-case coverage for GenotypeMatrix: input validation, device transfer,
filtering, population-file loading, and the accessible-mask lifecycle.

These are public paths on the core diploid data structure that the broader
suite exercises only incidentally; each test drives one branch directly.
"""
import numpy as np
import cupy as cp
import pytest

from pg_gpu import GenotypeMatrix
from pg_gpu.accessible import AccessibleMask


def _gm(genotypes, samples=None):
    """A small CPU GenotypeMatrix; genotypes is (n_individuals, n_variants)."""
    geno = np.asarray(genotypes, dtype=np.int8)
    positions = np.arange(geno.shape[1]) * 100 + 1
    chrom_end = int(positions[-1]) if positions.size else 1
    return GenotypeMatrix(geno, positions, 0, chrom_end, samples=samples)


class TestInitValidation:

    def test_empty_genotypes_raises(self):
        with pytest.raises(ValueError, match="genotypes cannot be empty"):
            GenotypeMatrix(np.empty((0, 0), dtype=np.int8), np.array([1]))

    def test_empty_positions_raises(self):
        with pytest.raises(ValueError, match="positions cannot be empty"):
            GenotypeMatrix(np.zeros((2, 3), dtype=np.int8), np.array([]))


class TestDeviceTransfer:

    def test_round_trip_cpu_gpu_cpu(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])
        assert gm.device == "CPU"
        gm.transfer_to_gpu()
        assert gm.device == "GPU"
        assert isinstance(gm.genotypes, cp.ndarray)
        gm.transfer_to_cpu()
        assert gm.device == "CPU"
        assert isinstance(gm.genotypes, np.ndarray)
        np.testing.assert_array_equal(gm.genotypes, [[0, 1, 2], [2, 1, 0]])

    def test_transfer_to_gpu_is_idempotent(self):
        # A second transfer on a store already on the GPU is a no-op.
        gm = _gm([[0, 1], [2, 0]])
        gm.transfer_to_gpu()
        gm.transfer_to_gpu()
        assert gm.device == "GPU"

    def test_round_trip_carries_accessible_index(self):
        # A partial mask sets _accessible_idx; both transfer directions must
        # move it across devices (the accessible-index branch of each).
        gm = _gm([[0, 1, 2], [2, 1, 0]])
        arr = np.ones(300, dtype=bool)
        arr[101] = False
        gm.set_accessible_mask(AccessibleMask(arr, offset=0))
        assert gm._accessible_idx is not None
        gm.transfer_to_gpu()
        assert isinstance(gm._accessible_idx, cp.ndarray)
        gm.transfer_to_cpu()
        assert isinstance(gm._accessible_idx, np.ndarray)


class TestFilter:

    def test_variants_mask_shape_mismatch_raises(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])  # n_variants = 3
        with pytest.raises(ValueError, match="variants mask shape mismatch"):
            gm.filter(variants=np.array([True, False]))

    def test_genotypes_mask_shape_mismatch_raises(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])  # expects a (n_var, n_ind) = (3, 2) mask
        with pytest.raises(ValueError, match="genotypes mask shape mismatch"):
            gm.filter(genotypes=np.ones((2, 3), dtype=bool))

    def test_filtering_all_variants_out_returns_empty(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])
        out = gm.filter(variants=np.zeros(3, dtype=bool))
        assert out.genotypes.shape == (2, 0)
        assert out.positions.shape == (0,)
        assert out.n_total_sites is None


class TestLoadPopFile:

    def test_no_sample_names_raises(self):
        gm = _gm([[0, 1], [2, 0]])  # no samples=
        with pytest.raises(ValueError, match="No sample names stored"):
            gm.load_pop_file({"s0": "A"})

    def test_dict_assignment_populates_sample_sets(self):
        gm = _gm([[0, 1], [2, 0]], samples=["s0", "s1"])
        gm.load_pop_file({"s0": "A", "s1": "B"})
        assert gm.sample_sets["A"] == [0]
        assert gm.sample_sets["B"] == [1]

    def test_file_assignment_populates_sample_sets(self, tmp_path):
        gm = _gm([[0, 1], [2, 0]], samples=["s0", "s1"])
        pop_file = tmp_path / "pops.tsv"
        pop_file.write_text("sample\tpop\ns0\tA\ns1\tB\n")
        gm.load_pop_file(str(pop_file))
        assert gm.sample_sets["A"] == [0]
        assert gm.sample_sets["B"] == [1]

    def test_population_with_no_member_is_dropped_with_warning(self):
        gm = _gm([[0, 1], [2, 0]], samples=["s0", "s1"])
        with pytest.warns(UserWarning, match="no member in this matrix"):
            gm.load_pop_file({"s0": "A", "s1": "A"}, pops=["A", "C"])
        assert "C" not in gm.sample_sets
        assert gm.sample_sets["A"] == [0, 1]


class TestAccessibleMaskLifecycle:

    def test_set_full_mask_then_remove(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])  # positions 1, 101, 201
        mask = AccessibleMask(np.ones(300, dtype=bool), offset=0)
        gm.set_accessible_mask(mask)
        assert gm.accessible_mask is mask
        assert gm.n_total_sites is not None
        # every position accessible -> no per-variant index needed
        assert gm._accessible_idx is None
        gm.remove_accessible_mask()
        assert gm.accessible_mask is None
        assert gm.n_total_sites is None

    def test_partial_mask_records_accessible_index(self):
        gm = _gm([[0, 1, 2], [2, 1, 0]])  # positions 1, 101, 201
        arr = np.ones(300, dtype=bool)
        arr[101] = False  # mask out the second variant's position
        gm.set_accessible_mask(AccessibleMask(arr, offset=0))
        idx = gm._accessible_idx
        assert idx is not None
        assert 1 not in (idx.get() if hasattr(idx, "get") else idx)
