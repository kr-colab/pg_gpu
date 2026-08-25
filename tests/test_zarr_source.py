"""Tests for ZarrGenotypeSource: chunked + subset reads on a VCZ store."""

import msprime
import numpy as np
import pytest
import zarr

from pg_gpu import HaplotypeMatrix
from pg_gpu.zarr_source import ZarrGenotypeSource

from .conftest import canonical_hap_rows


def _simulate_hm(n_samples=20, seq_length=50_000, seed=42):
    """Build a HaplotypeMatrix with msprime, matching test_zarr_io.py's style."""
    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=seq_length,
        recombination_rate=1e-4, random_seed=seed, ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


@pytest.fixture
def vcz_store(tmp_path):
    """Write a single-contig VCZ store with stable sample names."""
    hm = _simulate_hm()
    if hm.samples is None:
        hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "test.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path, hm


@pytest.fixture
def multi_contig_store(tmp_path):
    """Two VCZ groups merged into one store so the multi-contig branch is hit.
    The cleanest way to do this is write two stores and stitch their
    call_genotype + variant_position together via zarr's own API."""
    hm1 = _simulate_hm(seed=1)
    hm2 = _simulate_hm(seed=2)
    path = str(tmp_path / "multi.vcz")
    out = zarr.open_group(path, mode="w")
    n1, n2 = hm1.haplotypes.shape[1], hm2.haplotypes.shape[1]
    n_dip = hm1.num_haplotypes // 2

    pos = np.concatenate([np.asarray(hm1.positions), np.asarray(hm2.positions)])
    contig = np.concatenate([np.zeros(n1, np.int32), np.ones(n2, np.int32)])
    out.create_array("variant_position", shape=pos.shape, dtype="int32")[:] = pos.astype(np.int32)
    out.create_array("variant_contig", shape=contig.shape, dtype="int32")[:] = contig
    out.create_array("contig_id", shape=(2,), dtype="<U16")[:] = np.array(["chrA", "chrB"])
    out.create_array("sample_id", shape=(n_dip,), dtype="<U16")[:] = np.array(
        [f"s{i}" for i in range(n_dip)]
    )
    # Combine genotypes: shape (n1 + n2, n_dip, 2).
    gt1 = HaplotypeMatrix._haplotypes_to_gt(np.asarray(hm1.haplotypes))
    gt2 = HaplotypeMatrix._haplotypes_to_gt(np.asarray(hm2.haplotypes))
    gt = np.concatenate([gt1, gt2], axis=0)
    out.create_array("call_genotype", shape=gt.shape, dtype="int8",
                     chunks=(10_000, n_dip, 2))[:] = gt
    return path, hm1, hm2


@pytest.fixture
def pop_tsv(tmp_path):
    """A two-pop TSV in the format ``HaplotypeMatrix.load_pop_file`` expects."""
    path = str(tmp_path / "pops.tsv")
    with open(path, "w") as f:
        f.write("sample\tpop\n")
        for i in range(10):
            f.write(f"s{i}\tpop1\n")
        for i in range(10, 20):
            f.write(f"s{i}\tpop2\n")
    return path


class TestConstruction:

    def test_single_contig_no_region(self, vcz_store):
        path, hm = vcz_store
        src = ZarrGenotypeSource(path)
        assert src.chrom == "1"
        assert src.num_diploids == hm.num_haplotypes // 2
        assert src.num_haplotypes == hm.num_haplotypes
        assert src.num_variants == hm.haplotypes.shape[1]

    def test_region_subset(self, vcz_store):
        path, hm = vcz_store
        full = ZarrGenotypeSource(path)
        first_half = full.site_pos[len(full.site_pos) // 2]
        src = ZarrGenotypeSource(path, region=f"1:0-{int(first_half)}")
        assert src.num_variants < full.num_variants
        # The end bound is inclusive, and first_half is a real site.
        assert int(src.site_pos[-1]) == int(first_half)
        assert src.num_variants == int(np.sum(full.site_pos <= first_half))

    def test_single_position_region(self, vcz_store):
        path, hm = vcz_store
        p = int(np.asarray(hm.positions)[5])
        src = ZarrGenotypeSource(path, region=f"1:{p}-{p}")
        assert set(src.site_pos.tolist()) == {p}

    def test_multi_contig_requires_pick(self, multi_contig_store):
        path, _, _ = multi_contig_store
        with pytest.raises(ValueError, match="contigs"):
            ZarrGenotypeSource(path)

    def test_multi_contig_with_contig_id(self, multi_contig_store):
        path, hm1, _ = multi_contig_store
        src = ZarrGenotypeSource(path, contig_id="chrA")
        assert src.chrom == "chrA"
        assert src.num_variants == hm1.haplotypes.shape[1]

    def test_unknown_contig_raises(self, multi_contig_store):
        path, _, _ = multi_contig_store
        with pytest.raises(ValueError, match="chrC"):
            ZarrGenotypeSource(path, contig_id="chrC")

    def test_rejects_non_vcz_layout(self, tmp_path):
        hm = _simulate_hm()
        path = str(tmp_path / "allel.zarr")
        hm.to_zarr(path, format="scikit-allel")
        with pytest.raises(ValueError, match="VCZ layout only"):
            ZarrGenotypeSource(path)


class TestSliceRegion:

    def test_shapes_and_dtype(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        gt, pos = src.slice_region(0, src.mappable_hi)
        assert gt.shape == (src.num_variants, src.num_diploids, 2)
        assert gt.dtype == np.int8
        assert pos.shape == (src.num_variants,)

    def test_round_trip_to_haplotype_matrix(self, vcz_store):
        path, hm = vcz_store
        src = ZarrGenotypeSource(path)
        gt, pos = src.slice_region(0, src.mappable_hi)
        # Reproduce the canonical row order and compare to the original
        # HaplotypeMatrix bytes.
        haps = canonical_hap_rows(gt).T
        np.testing.assert_array_equal(haps.T, np.asarray(hm.haplotypes))

    def test_empty_region(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        # below the first variant
        gt, pos = src.slice_region(0, max(0, src.mappable_lo - 1))
        assert gt.shape[0] == 0
        assert pos.shape[0] == 0


class TestSliceSubsample:

    def test_oindex_matches_full_then_slice(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        gt_full, _ = src.slice_region(0, src.mappable_hi)

        # Mix both gametes of some samples with a single gamete of another so
        # the (dip, ploidy) translation is exercised in both parities.
        cols = np.array([0, 1, 2, 5, 8], dtype=np.int64)

        gm_sub, _ = src.slice_subsample(0, src.mappable_hi, cols)
        # Build the expected (n_var, len(cols)) from gt_full.
        expected = np.empty((gt_full.shape[0], len(cols)), dtype=gt_full.dtype)
        for j, c in enumerate(cols):
            expected[:, j] = gt_full[:, c // 2, c % 2]
        np.testing.assert_array_equal(gm_sub, expected)

    def test_empty_region(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        gm, pos = src.slice_subsample(0, max(0, src.mappable_lo - 1),
                                      np.array([0, 1]))
        assert gm.shape == (0, 2)
        assert pos.shape == (0,)

    def test_to_gpu_matches_host(self, vcz_store):
        import cupy as cp
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        cols = np.array([0, 1, 2, 5, 8], dtype=np.int64)
        gm_host, pos_host = src.slice_subsample(
            0, src.mappable_hi, cols, to_gpu=False
        )
        gm_gpu, pos_gpu = src.slice_subsample(
            0, src.mappable_hi, cols, to_gpu=True
        )
        assert isinstance(gm_gpu, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(gm_gpu), gm_host)
        np.testing.assert_array_equal(pos_gpu, pos_host)

    def test_to_gpu_empty_region(self, vcz_store):
        import cupy as cp
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        gm, pos = src.slice_subsample(
            0, max(0, src.mappable_lo - 1), np.array([0, 1]), to_gpu=True,
        )
        assert isinstance(gm, cp.ndarray)
        assert gm.shape == (0, 2)
        assert pos.shape == (0,)


class TestIterChunks:

    def test_yields_aligned_intervals(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        chunks = list(src.iter_chunks(chunk_bp=10_000, align_bp=10_000))
        # alignment respected
        for left, right in chunks:
            assert left % 10_000 == 0
            assert right - left <= 10_000
        # cover the whole mappable range
        assert chunks[0][0] == 0
        assert chunks[-1][1] == src.mappable_hi

    def test_alignment_smaller_than_chunk(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path)
        chunks = list(src.iter_chunks(chunk_bp=10_000, align_bp=5_000))
        # chunk_bp / align_bp = 2 windows per chunk, step = 10000
        assert chunks[0] == (0, min(10_000, src.mappable_hi))


class TestPopAssignmentResolution:

    def test_explicit_pop_assignment(self, vcz_store, pop_tsv):
        path, _ = vcz_store
        src = ZarrGenotypeSource(path, pop_assignment=pop_tsv)
        assert set(src.pop_cols.keys()) == {"pop1", "pop2"}
        # pop1 = first 10 diploids -> haps [0..20), both gametes of each
        expected1 = np.arange(2 * 10)
        np.testing.assert_array_equal(np.sort(src.pop_cols["pop1"]), expected1)

    def test_auto_load_companion(self, vcz_store, pop_tsv, tmp_path, capsys):
        path, _ = vcz_store
        companion = path + ".pops.tsv"
        # copy pop_tsv to companion location
        with open(pop_tsv) as src_f, open(companion, "w") as dst:
            dst.write(src_f.read())
        src = ZarrGenotypeSource(path)
        assert src.pop_cols is not None
        # auto-load announces to stderr so it doesn't pollute pipelines that
        # capture stdout for table output.
        assert "auto-loaded" in capsys.readouterr().err

    def test_pop_assignment_false_disables_autoload(self, vcz_store, pop_tsv):
        path, _ = vcz_store
        companion = path + ".pops.tsv"
        with open(pop_tsv) as src_f, open(companion, "w") as dst:
            dst.write(src_f.read())
        src = ZarrGenotypeSource(path, pop_assignment=False)
        assert src.pop_cols is None

    def test_unknown_sample_in_pop_assignment_warns(self, vcz_store, tmp_path):
        path, _ = vcz_store
        bad_pop = str(tmp_path / "bad.tsv")
        with open(bad_pop, "w") as f:
            f.write("sample\tpop\ns0\tpop1\nnobody\tpop2\n")
        with pytest.warns(UserWarning, match="not in store"):
            src = ZarrGenotypeSource(path, pop_assignment=bad_pop)
        assert "pop1" in src.pop_cols
        assert "pop2" not in src.pop_cols

    def test_pop_assignment_accepts_dict(self, vcz_store):
        path, _ = vcz_store
        src = ZarrGenotypeSource(
            path, pop_assignment={f"s{i}": "pop1" if i < 5 else "pop2"
                                  for i in range(20)},
        )
        assert set(src.pop_cols.keys()) == {"pop1", "pop2"}
        # pop1 = first 5 diploids -> haps [0..10), both gametes of each
        expected1 = np.arange(2 * 5)
        np.testing.assert_array_equal(np.sort(src.pop_cols["pop1"]),
                                       expected1)

    def test_pop_assignment_accepts_array(self, vcz_store):
        path, _ = vcz_store
        # 20 diploids in the fixture; first 12 are pop1, rest pop2.
        labels = np.array(["pop1"] * 12 + ["pop2"] * 8)
        src = ZarrGenotypeSource(path, pop_assignment=labels)
        assert set(src.pop_cols.keys()) == {"pop1", "pop2"}
        expected1 = np.arange(2 * 12)
        np.testing.assert_array_equal(np.sort(src.pop_cols["pop1"]),
                                       expected1)

    def test_pop_assignment_rejects_mismatched_array_length(self, vcz_store):
        path, _ = vcz_store
        with pytest.raises(ValueError, match="does not match sample"):
            ZarrGenotypeSource(path, pop_assignment=np.array(["pop1", "pop2"]))

    def test_pop_assignment_accepts_zarr_key(self, vcz_store):
        path, _ = vcz_store
        # Stamp a 1-D population array onto the store under a non-VCZ
        # key, then look it up by name. Mirrors the case where bio2zarr
        # was extended with a sample-axis population field.
        store = zarr.open_group(path, mode="r+")
        labels = np.array(["pop1"] * 12 + ["pop2"] * 8)
        store.create_array("sample_population", shape=labels.shape,
                            dtype="<U8")[:] = labels
        src = ZarrGenotypeSource(path, pop_assignment="sample_population")
        assert set(src.pop_cols.keys()) == {"pop1", "pop2"}
