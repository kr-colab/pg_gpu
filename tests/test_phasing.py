"""Tests for the ``phased`` flag on HaplotypeMatrix and StreamingHaplotypeMatrix."""

import warnings

import numpy as np
import pytest
import cupy as cp
import zarr

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.genotype_matrix import GenotypeMatrix
from pg_gpu._gpu_genotype_prep import build_haplotype_matrix
from pg_gpu.zarr_io import PHASED_ATTR

from .conftest import simulate_hm


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

    def test_set_unphased_flips_to_false(self):
        hm = _small_hm(phased=True)
        ret = hm.set_unphased()
        assert hm.phased is False
        assert ret is hm

    def test_setters_are_idempotent(self):
        hm = _small_hm(phased=True)
        assert hm.set_phased().phased is True
        hm = _small_hm(phased=False)
        assert hm.set_unphased().phased is False

    def test_set_phased_takes_no_argument(self):
        hm = _small_hm()
        with pytest.raises(TypeError):
            hm.set_phased(False)


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
        stream.set_phased()
        assert stream.phased is True
        for _left, _right, chunk in stream.iter_gpu_chunks():
            assert chunk.phased is True

    def test_streaming_set_unphased_propagates_to_chunks(self, zarr_store):
        stream = HaplotypeMatrix.from_zarr(zarr_store, streaming="always",
                                           phased=True)
        assert stream.set_unphased() is stream
        assert stream.phased is False
        for _left, _right, chunk in stream.iter_gpu_chunks():
            assert chunk.phased is False


# ---------------------------------------------------------------------------
# to_zarr records the flag and from_zarr gives it back
# ---------------------------------------------------------------------------

class TestZarrKeepsPhase:
    """A matrix keeps its ``phased`` value through ``to_zarr`` and
    ``from_zarr``. The loader reads the record that ``to_zarr`` left in the
    store; it does not infer phase from the calls, so a store from another
    tool loads as unphased."""

    @staticmethod
    def _saved(tmp_path, phased, format="vcz", fields=None):
        hm = _small_hm(phased=phased)
        hm.samples = ["s0", "s1"]
        hm.fields = fields
        path = str(tmp_path / f"{format}_{phased}.zarr")
        hm.to_zarr(path, format=format, contig_name="1")
        return path

    @pytest.mark.parametrize("phased", [True, False])
    @pytest.mark.parametrize("format,streaming", [
        ("vcz", "never"), ("vcz", "always"), ("scikit-allel", "never")])
    def test_round_trip(self, tmp_path, format, streaming, phased):
        path = self._saved(tmp_path, phased, format=format)
        assert HaplotypeMatrix.from_zarr(path, streaming=streaming).phased is phased

    def test_saved_simulation_does_not_warn(self, tmp_path):
        # The case that showed the fault: a phased simulation, saved and
        # loaded, then given to a phase-dependent statistic.
        from pg_gpu import UnphasedDataWarning, selection
        hm = simulate_hm(n_samples=10, mutation_model="binary")
        assert hm.phased is True
        path = str(tmp_path / "sim.vcz")
        hm.to_zarr(path, contig_name="1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnphasedDataWarning)
            selection.garud_h(HaplotypeMatrix.from_zarr(path))

    def test_phased_store_has_the_vcz_phase_field(self, tmp_path):
        # call_genotype_phased is the VCZ field for phase: one value for
        # each variant and sample.
        field = zarr.open_group(self._saved(tmp_path, True), mode="r")["call_genotype_phased"]
        assert field.shape == (3, 2) and field.dtype == bool
        assert np.asarray(field).all()

    def test_unphased_store_has_no_phase_field(self, tmp_path):
        # phased=False means that phase was not declared; the store must not
        # state that every call is unphased.
        store = zarr.open_group(self._saved(tmp_path, False), mode="r")
        assert "call_genotype_phased" not in store
        assert store.attrs[PHASED_ATTR] is False

    @pytest.mark.parametrize("phased", [True, False])
    def test_phase_field_among_the_matrix_fields_is_kept(self, tmp_path, phased):
        # A matrix loaded with fields=['genotype_phased'] carries the
        # per-call values; to_zarr writes those, not its own field.
        per_call = np.array([[True, False], [False, False], [True, True]])
        path = self._saved(tmp_path, phased, fields={"genotype_phased": per_call})
        store = zarr.open_group(path, mode="r")
        np.testing.assert_array_equal(np.asarray(store["call_genotype_phased"]), per_call)
        assert HaplotypeMatrix.from_zarr(path).phased is phased

    def test_store_without_the_record_loads_unphased(self, tmp_path):
        # As a store from bio2zarr: every call is marked phased, but the
        # store has no record from to_zarr.
        path = self._saved(tmp_path, True)
        del zarr.open_group(path, mode="r+").attrs[PHASED_ATTR]
        assert HaplotypeMatrix.from_zarr(path).phased is False
        assert HaplotypeMatrix.from_zarr(path, streaming="always").phased is False

    def test_genotype_store_has_no_record(self, tmp_path):
        gm = GenotypeMatrix.from_haplotype_matrix(_small_hm(phased=True))
        path = str(tmp_path / "genotypes.vcz")
        gm.to_zarr(path, contig_name="1")
        assert PHASED_ATTR not in zarr.open_group(path, mode="r").attrs
        assert HaplotypeMatrix.from_zarr(path).phased is False

    @pytest.mark.parametrize("phased", [True, False])
    def test_layout_conversion_keeps_the_record(self, tmp_path, phased):
        from pg_gpu.zarr_io import allel_zarr_to_vcz
        vcz = str(tmp_path / "converted.vcz")
        allel_zarr_to_vcz(self._saved(tmp_path, phased, format="scikit-allel"), vcz,
                          contig="1")
        assert HaplotypeMatrix.from_zarr(vcz).phased is phased
