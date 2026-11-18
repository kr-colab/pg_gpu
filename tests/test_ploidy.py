"""The ``ploidy`` of a HaplotypeMatrix: rows per sample, 1 or 2.

A diploid matrix has sample ``i`` at rows ``2i`` and ``2i + 1``; a haploid
matrix has one row per sample. The store that ``to_zarr`` writes has the
matrix's ploidy on its last axis, ``from_zarr`` reads it back on the eager
and the streaming path, and the consumers that pair rows into individuals
refuse a haploid matrix.
"""

import msprime
import numpy as np
import pytest
import zarr

from pg_gpu import (GenotypeMatrix, HaplotypeMatrix, UnpairedRowsWarning,
                    UnphasedDataWarning, divergence, diversity, selection)
from pg_gpu._utils import get_population_matrix
from pg_gpu.zarr_io import PHASED_ATTR, allel_zarr_to_vcz

from .conftest import force_chunked_engine

N_VAR = 40
POS = np.arange(N_VAR) * 10 + 1
NAMES = ["a", "b", "c", "d", "e"]


def _calls(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, (n_rows, N_VAR)).astype(np.int8)
    hap[rng.random(hap.shape) < 0.1] = -1
    return hap


def _haploid(n_rows=5, **kwargs):
    kwargs.setdefault("samples", NAMES[:n_rows])
    return HaplotypeMatrix(_calls(n_rows), POS, 0, 500, phased=True, ploidy=1, **kwargs)


def _rows(matrix):
    hap = matrix.haplotypes
    return hap.get() if hasattr(hap, "get") else hap


def _saved(tmp_path, matrix, format="vcz"):
    path = str(tmp_path / f"store_{format}.zarr")
    matrix.to_zarr(path, format=format, contig_name="1")
    return path


# ── the attribute ─────────────────────────────────────────────────────
class TestAttribute:

    def test_default_is_diploid(self):
        assert HaplotypeMatrix(_calls(4), POS, 0, 500).ploidy == 2

    @pytest.mark.parametrize("ploidy", [0, 3, 1.5, True, None, "2"])
    def test_other_values_raise(self, ploidy):
        with pytest.raises(ValueError, match="ploidy must be 1 or 2"):
            HaplotypeMatrix(_calls(4), POS, 0, 500, ploidy=ploidy)

    def test_can_be_assigned(self):
        hm = HaplotypeMatrix(_calls(4), POS, 0, 500)
        hm.ploidy = 1
        assert hm.ploidy == 1
        with pytest.raises(ValueError, match="ploidy must be 1 or 2"):
            hm.ploidy = 4

    def test_in_repr(self):
        assert "ploidy=1" in repr(_haploid())

    @pytest.mark.parametrize("derive", [
        lambda m: m.get_subset(np.array([0, 3, 5])),
        lambda m: m.get_subset(np.array([], dtype=np.int64)),
        lambda m: m.get_subset_from_range(0, 100),
        lambda m: m.get_subset_from_range(5000, 6000),
        lambda m: m.restrict_to_biallelic(),
        lambda m: m.restrict_to_segregating(),
        lambda m: m.filter(variants=np.arange(N_VAR) % 2 == 0),
        lambda m: m.filter(variants=np.zeros(N_VAR, dtype=bool)),
        lambda m: get_population_matrix(m, [3, 0, 4]),
    ], ids=["get_subset", "get_subset_empty", "range", "range_empty", "biallelic",
            "segregating", "filter", "filter_empty", "population"])
    def test_derived_matrices_keep_it(self, derive):
        assert derive(_haploid()).ploidy == 1

    def test_population_subset_keeps_names_by_row(self):
        assert get_population_matrix(_haploid(), [3, 0, 4]).samples == ["d", "a", "e"]

    def test_genotype_filter_masks_one_row_per_sample(self):
        hm = HaplotypeMatrix(np.ones((5, N_VAR), dtype=np.int8), POS, 0, 500, ploidy=1)
        keep = np.ones((N_VAR, 5), dtype=bool)
        keep[:, 2] = False
        out = _rows(hm.filter(genotypes=keep, drop_all_missing=False))
        assert (out[2] == -1).all() and (out[[0, 1, 3, 4]] == 1).all()

    def test_pop_file_gives_one_row_per_sample(self):
        hm = _haploid()
        hm.load_pop_file({"a": "x", "b": "x", "c": "y", "d": "y", "e": "y"})
        assert hm.sample_sets == {"x": [0, 1], "y": [2, 3, 4]}

    def test_pop_file_refuses_names_that_do_not_fit_the_rows(self):
        # Two names for four rows, read at one row per sample, would send
        # sample "t" to a row of sample "s".
        hm = HaplotypeMatrix(_calls(4), POS, 0, 500, samples=["s", "t"])
        hm.ploidy = 1
        with pytest.raises(ValueError, match="2 names for 4 rows"):
            hm.load_pop_file({"s": "x", "t": "y"})

    @pytest.mark.parametrize("rows", [[0, 2, 4], [0, 2, 4, 6], [0, 1, 2]],
                             ids=["one-per-individual-odd", "one-per-individual-even",
                                  "odd"])
    def test_unpaired_subset_of_a_diploid_matrix_is_haploid(self, rows):
        # The rows no longer make whole individuals, so pairing them again,
        # as to_zarr does at ploidy 2, would join unrelated gametes.
        hm = HaplotypeMatrix(_calls(8), POS, 0, 500, samples=["s", "t", "u", "v"])
        with pytest.warns(UnpairedRowsWarning, match="has ploidy 1"):
            assert get_population_matrix(hm, rows).ploidy == 1

    def test_paired_subset_of_a_diploid_matrix_stays_diploid(self):
        hm = HaplotypeMatrix(_calls(8), POS, 0, 500, samples=["s", "t", "u", "v"])
        sub = get_population_matrix(hm, [4, 5, 0, 1])
        assert sub.ploidy == 2 and sub.samples == ["u", "s"]


class TestHaploidIsPhased:
    """One row per sample is one chromosome, so a haploid matrix counts as
    phased whatever flag it was given."""

    def test_flag_is_true_at_ploidy_one(self):
        hm = HaplotypeMatrix(_calls(5), POS, 0, 500, phased=False, ploidy=1)
        assert hm.phased is True and hm.set_unphased().phased is True
        assert "phased=True" in repr(hm)

    def test_flag_follows_the_ploidy(self):
        hm = HaplotypeMatrix(_calls(4), POS, 0, 500, phased=False, ploidy=1)
        hm.ploidy = 2
        assert hm.phased is False

    @pytest.mark.parametrize("streaming", ["never", "always"])
    def test_store_with_no_phase_record(self, tmp_path, streaming):
        # As a haploid store from bio2zarr.
        path = _saved(tmp_path, _haploid())
        del zarr.open_group(path, mode="r+").attrs[PHASED_ATTR]
        assert HaplotypeMatrix.from_zarr(path, streaming=streaming).phased is True

    def test_phase_dependent_statistic_does_not_warn(self):
        import warnings
        hm = HaplotypeMatrix(_calls(8), POS, 0, 500, phased=False, ploidy=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnphasedDataWarning)
            selection.garud_h(hm)


# ── from_ts ───────────────────────────────────────────────────────────
def _ts(n, ploidy):
    return msprime.sim_mutations(
        msprime.sim_ancestry(n, ploidy=ploidy, sequence_length=1e4, recombination_rate=1e-8,
                             population_size=1e4, random_seed=1),
        rate=1e-7, random_seed=1)


class TestFromTs:

    def test_diploid_individuals(self):
        assert HaplotypeMatrix.from_ts(_ts(4, 2)).ploidy == 2

    def test_one_node_per_individual_is_haploid(self):
        hm = HaplotypeMatrix.from_ts(_ts(7, 1))
        assert hm.num_haplotypes == 7 and hm.ploidy == 1

    def test_no_individuals_is_haploid(self):
        tables = _ts(3, 2).dump_tables()
        tables.individuals.clear()
        tables.nodes.individual = np.full(tables.nodes.num_rows, -1, dtype=np.int32)
        assert HaplotypeMatrix.from_ts(tables.tree_sequence()).ploidy == 1

    def test_explicit_ploidy_declares_the_pairs(self):
        assert HaplotypeMatrix.from_ts(_ts(6, 1), ploidy=2).ploidy == 2


# ── to_zarr and from_zarr ─────────────────────────────────────────────
class TestHaploidStore:

    @pytest.mark.parametrize("format,array", [("vcz", "call_genotype"),
                                              ("scikit-allel", "calldata/GT")])
    def test_store_has_a_ploidy_axis_of_one(self, tmp_path, format, array):
        store = zarr.open_group(_saved(tmp_path, _haploid(), format), mode="r")
        assert store[array].shape == (N_VAR, 5, 1)

    @pytest.mark.parametrize("format,streaming", [
        ("vcz", "auto"), ("vcz", "never"), ("scikit-allel", "auto")])
    def test_eager_round_trip(self, tmp_path, format, streaming):
        hm = _haploid()
        back = HaplotypeMatrix.from_zarr(_saved(tmp_path, hm, format), streaming=streaming)
        assert back.ploidy == 1 and back.phased is True
        np.testing.assert_array_equal(_rows(back), _rows(hm))
        assert [str(s) for s in back.samples] == NAMES

    @pytest.mark.parametrize("backend", ["host", "kvikio"])
    def test_streaming_round_trip(self, tmp_path, backend):
        hm = _haploid()
        stream = HaplotypeMatrix.from_zarr(_saved(tmp_path, hm), streaming="always",
                                           backend=backend, chunk_bp=100)
        assert stream.ploidy == 1 and stream.num_haplotypes == 5
        chunks = [m for _, _, m in stream.iter_gpu_chunks()]
        assert len(chunks) > 1 and {m.ploidy for m in chunks} == {1}
        np.testing.assert_array_equal(np.concatenate([_rows(m) for m in chunks], axis=1),
                                      _rows(hm))
        whole = stream.materialize()
        assert whole.ploidy == 1
        np.testing.assert_array_equal(_rows(whole), _rows(hm))

    @pytest.mark.parametrize("backend", ["host", "kvikio"])
    def test_streaming_sample_subset_takes_single_rows(self, tmp_path, backend):
        hm = _haploid()
        stream = HaplotypeMatrix.from_zarr(_saved(tmp_path, hm), streaming="always",
                                           backend=backend)
        sub = stream.materialize(sample_subset=[4, 0, 2])
        assert sub.ploidy == 1
        np.testing.assert_array_equal(_rows(sub), _rows(hm)[[4, 0, 2]])

    @pytest.mark.parametrize("streaming", ["never", "always"])
    def test_pop_assignment_gives_one_row_per_sample(self, tmp_path, streaming):
        loaded = HaplotypeMatrix.from_zarr(
            _saved(tmp_path, _haploid()), streaming=streaming,
            pop_assignment={"a": "x", "b": "x", "c": "y", "d": "y", "e": "y"})
        assert {p: [int(r) for r in rows] for p, rows in loaded.sample_sets.items()} == \
            {"x": [0, 1], "y": [2, 3, 4]}

    def test_layout_conversion_keeps_the_ploidy(self, tmp_path):
        hm = _haploid()
        vcz = str(tmp_path / "converted.vcz")
        allel_zarr_to_vcz(_saved(tmp_path, hm, "scikit-allel"), vcz, contig="1")
        back = HaplotypeMatrix.from_zarr(vcz)
        assert back.ploidy == 1
        np.testing.assert_array_equal(_rows(back), _rows(hm))

    @pytest.mark.parametrize("format", ["vcz", "scikit-allel"])
    @pytest.mark.parametrize("streaming", ["auto", "always"])
    def test_genotype_loader_refuses_it(self, tmp_path, format, streaming):
        if (format, streaming) == ("scikit-allel", "always"):
            pytest.skip("the streaming reader takes VCZ stores only")
        with pytest.raises(ValueError, match="ploidy 1"):
            GenotypeMatrix.from_zarr(_saved(tmp_path, _haploid(), format), streaming=streaming)


class TestDiploidStore:

    def test_round_trip_is_unchanged(self, tmp_path):
        hm = HaplotypeMatrix(_calls(6), POS, 0, 500, samples=["s", "t", "u"], phased=True)
        path = _saved(tmp_path, hm)
        assert zarr.open_group(path, mode="r")["call_genotype"].shape == (N_VAR, 3, 2)
        back = HaplotypeMatrix.from_zarr(path)
        assert back.ploidy == 2
        np.testing.assert_array_equal(_rows(back), _rows(hm))

    @pytest.mark.parametrize("ploidy", [3, 4])
    def test_higher_ploidy_store_raises(self, tmp_path, ploidy):
        path = str(tmp_path / "poly.vcz")
        store = zarr.open_group(path, mode="w")
        store.create_array("call_genotype", data=np.zeros((4, 2, ploidy), dtype=np.int8))
        store.create_array("call_genotype_mask", data=np.zeros((4, 2, ploidy), dtype=bool))
        store.create_array("variant_position", data=np.arange(1, 5, dtype=np.int32))
        store.create_array("contig_id", data=np.array(["1"], dtype="U"))
        store.create_array("variant_contig", data=np.zeros(4, dtype=np.int8))
        for streaming in ("auto", "always"):
            with pytest.raises(ValueError, match="ploidy 1 and 2 only"):
                HaplotypeMatrix.from_zarr(path, streaming=streaming)


# ── to_zarr refuses a store that would not agree with itself ──────────
class TestWriterChecks:

    def test_odd_rows_at_ploidy_two(self, tmp_path):
        hm = HaplotypeMatrix(_calls(5), POS, 0, 500)
        with pytest.raises(ValueError, match="set matrix.ploidy = 1"):
            _saved(tmp_path, hm)

    def test_one_name_per_row_at_ploidy_two(self, tmp_path):
        # Four haploid samples built without ploidy=1: the rows pair, but the
        # names say that they are four samples.
        hm = HaplotypeMatrix(_calls(4), POS, 0, 500, samples=NAMES[:4])
        with pytest.raises(ValueError, match="samples has 4 names"):
            _saved(tmp_path, hm)
        hm.ploidy = 1
        assert HaplotypeMatrix.from_zarr(_saved(tmp_path, hm)).num_haplotypes == 4

    def test_no_names_writes_no_names(self, tmp_path):
        path = _saved(tmp_path, _haploid(samples=None))
        assert "sample_id" not in zarr.open_group(path, mode="r")

    def test_genotype_matrix_names_must_match_individuals(self, tmp_path):
        gm = GenotypeMatrix(np.zeros((5, N_VAR), dtype=np.int8), POS, 0, 500,
                            samples=["a", "b", "c"])
        with pytest.raises(ValueError, match="samples has 3 names"):
            _saved(tmp_path, gm)

    def test_scikit_allel_layout_checks_names_too(self, tmp_path):
        hm = HaplotypeMatrix(_calls(4), POS, 0, 500, samples=NAMES[:4])
        with pytest.raises(ValueError, match="samples has 4 names"):
            _saved(tmp_path, hm, "scikit-allel")

    def test_genotype_matrix_value_above_two_raises(self, tmp_path):
        geno = np.zeros((5, N_VAR), dtype=np.int8)
        geno[0, 0] = 3
        with pytest.raises(ValueError, match="genotypes 0, 1 and 2"):
            _saved(tmp_path, GenotypeMatrix(geno, POS, 0, 500))


# ── consumers that pair rows into individuals ─────────────────────────
class TestPairingConsumers:

    @pytest.fixture
    def hm(self):
        hm = HaplotypeMatrix(_calls(8), POS, 0, 500, phased=True, ploidy=1)
        hm.sample_sets = {"x": [0, 1, 2, 3], "y": [4, 5, 6, 7]}
        return hm

    @pytest.mark.parametrize("call", [
        lambda m: GenotypeMatrix.from_haplotype_matrix(m),
        lambda m: diversity.heterozygosity_observed(m),
        lambda m: diversity.inbreeding_coefficient(m),
        lambda m: divergence.fst_weir_cockerham(m, "x", "y"),
    ], ids=["from_haplotype_matrix", "heterozygosity_observed",
            "inbreeding_coefficient", "fst_weir_cockerham"])
    def test_raise_on_a_haploid_matrix(self, hm, call):
        with pytest.raises(ValueError, match="has ploidy 1"):
            call(hm)

    @pytest.mark.parametrize("statistics", [["fst_wc"], ["fst_wc", "dxy"],
                                            ["fst_wc", "garud_h1"]])
    @pytest.mark.parametrize("chunked", [False, True])
    def test_windowed_fst_wc_raises(self, hm, statistics, chunked, monkeypatch):
        from pg_gpu import windowed_analysis
        if chunked:
            force_chunked_engine(monkeypatch, 7)
        with pytest.raises(ValueError, match="has ploidy 1"):
            windowed_analysis(hm, window_size=100, statistics=statistics,
                              populations=["x", "y"])

    def test_genotype_estimator_of_ld_raises(self, hm):
        with pytest.raises(ValueError, match="has ploidy 1"):
            hm.pairwise_r2(estimator="rogers_huff")

    def test_gamete_statistics_run(self, hm):
        assert np.isfinite(float(diversity.pi(hm)))
        assert np.isfinite(float(divergence.fst_hudson(hm, "x", "y")))

    def test_declared_pairs_run(self, hm):
        hm.ploidy = 2
        assert GenotypeMatrix.from_haplotype_matrix(hm).genotypes.shape[0] == 4
