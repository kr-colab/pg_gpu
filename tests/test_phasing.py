"""Tests for the ``phased`` flag on HaplotypeMatrix and StreamingHaplotypeMatrix."""

import numpy as np
import pytest
import cupy as cp

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix
from pg_gpu._gpu_genotype_prep import build_haplotype_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_hm(phased=False):
    """Minimal HaplotypeMatrix for flag-propagation tests."""
    hap = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.int8)
    pos = np.array([100, 200, 300], dtype=np.int64)
    return HaplotypeMatrix(hap, pos, chrom_start=0, chrom_end=400, phased=phased)


# ---------------------------------------------------------------------------
# Constructor defaults and overrides
# ---------------------------------------------------------------------------

class TestConstructorDefaults:

    def test_raw_constructor_default_false(self):
        hm = _small_hm()
        assert hm.phased is False

    def test_raw_constructor_explicit_true(self):
        hm = _small_hm(phased=True)
        assert hm.phased is True

    def test_from_vcf_default_false(self, simple_vcf_file):
        hm = HaplotypeMatrix.from_vcf(simple_vcf_file)
        assert hm.phased is False

    def test_from_vcf_override_true(self, simple_vcf_file):
        hm = HaplotypeMatrix.from_vcf(simple_vcf_file, phased=True)
        assert hm.phased is True

    def test_from_ts_default_true(self):
        msprime = pytest.importorskip("msprime")
        ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=1)
        hm = HaplotypeMatrix.from_ts(ts)
        assert hm.phased is True

    def test_from_ts_override_false(self):
        msprime = pytest.importorskip("msprime")
        ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=1)
        hm = HaplotypeMatrix.from_ts(ts, phased=False)
        assert hm.phased is False


# ---------------------------------------------------------------------------
# set_phased
# ---------------------------------------------------------------------------

class TestSetPhased:

    def test_set_phased_flips_to_true(self):
        hm = _small_hm(phased=False)
        ret = hm.set_phased()
        assert hm.phased is True
        assert ret is hm

    def test_set_phased_explicit_false(self):
        hm = _small_hm(phased=True)
        hm.set_phased(False)
        assert hm.phased is False

    def test_set_phased_returns_self(self):
        hm = _small_hm()
        assert hm.set_phased(True) is hm


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_phased_in_repr(self):
        hm = _small_hm(phased=True)
        assert "phased=True" in repr(hm)

    def test_unphased_in_repr(self):
        hm = _small_hm(phased=False)
        assert "phased=False" in repr(hm)


# ---------------------------------------------------------------------------
# Propagation through derived matrices
# ---------------------------------------------------------------------------

class TestPropagation:

    @pytest.mark.parametrize("phased", [True, False])
    def test_get_subset(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.get_subset(cp.array([0, 1]))
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_get_subset_empty(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.get_subset(cp.array([], dtype=np.int64))
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_get_subset_from_range(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.get_subset_from_range(50, 250)
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_get_subset_from_range_empty(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.get_subset_from_range(500, 600)
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_keep_sites(self, phased):
        hm = _small_hm(phased=phased)
        hm.transfer_to_gpu()
        sub = hm._keep_sites(cp.array([0]))
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_restrict_to_biallelic(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.restrict_to_biallelic()
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_restrict_to_segregating(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.restrict_to_segregating()
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_filter(self, phased):
        hm = _small_hm(phased=phased)
        sub = hm.filter()
        assert sub.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_get_population_matrix(self, phased):
        from pg_gpu._utils import get_population_matrix
        hm = _small_hm(phased=phased)
        hm.sample_sets = {"pop": [0, 1, 2, 3]}
        pop_hm = get_population_matrix(hm, "pop")
        assert pop_hm.phased is phased


# ---------------------------------------------------------------------------
# GM round-trip gives phased=False
# ---------------------------------------------------------------------------

class TestGenotypeRoundTrip:

    def test_to_haplotype_matrix_always_unphased(self):
        hm = _small_hm(phased=True)
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        hm2 = gm.to_haplotype_matrix()
        assert hm2.phased is False


# ---------------------------------------------------------------------------
# build_haplotype_matrix requires phased
# ---------------------------------------------------------------------------

class TestBuildHaplotypeMatrixRequired:

    def test_missing_phased_raises_typeerror(self):
        gt = np.zeros((10, 5, 2), dtype=np.int8)
        pos = np.arange(10, dtype=np.int64)
        with pytest.raises(TypeError):
            build_haplotype_matrix(gt, pos)

    def test_explicit_phased_works(self):
        gt = np.zeros((10, 5, 2), dtype=np.int8)
        pos = np.arange(10, dtype=np.int64)
        hm = build_haplotype_matrix(gt, pos, phased=True)
        assert hm.phased is True


# ---------------------------------------------------------------------------
# Zarr round-trip (eager and streaming)
# ---------------------------------------------------------------------------

class TestZarrPhased:

    @pytest.fixture
    def zarr_store(self, tmp_path, simple_vcf_file):
        hm = HaplotypeMatrix.from_vcf(simple_vcf_file, phased=True)
        path = str(tmp_path / "test.vcz.zarr")
        hm.to_zarr(path, contig_name="1")
        return path

    @pytest.mark.parametrize("phased", [True, False])
    def test_eager_from_zarr(self, zarr_store, phased):
        hm = HaplotypeMatrix.from_zarr(zarr_store, streaming="never",
                                       phased=phased)
        assert hm.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_streaming_iter_chunks(self, zarr_store, phased):
        stream = HaplotypeMatrix.from_zarr(zarr_store, streaming="always",
                                           phased=phased)
        for _left, _right, chunk in stream.iter_gpu_chunks():
            assert chunk.phased is phased

    @pytest.mark.parametrize("phased", [True, False])
    def test_streaming_materialize(self, zarr_store, phased):
        stream = HaplotypeMatrix.from_zarr(zarr_store, streaming="always",
                                           phased=phased)
        mat = stream.materialize()
        assert mat.phased is phased

    def test_streaming_set_phased_propagates_to_chunks(self, zarr_store):
        stream = HaplotypeMatrix.from_zarr(zarr_store, streaming="always",
                                           phased=False)
        assert stream.phased is False
        stream.set_phased(True)
        assert stream.phased is True
        for _left, _right, chunk in stream.iter_gpu_chunks():
            assert chunk.phased is True
