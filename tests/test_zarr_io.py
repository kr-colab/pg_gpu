"""Tests for zarr I/O: VCZ and scikit-allel format support."""

import numpy as np
import pytest
import zarr
import msprime

import cupy as cp

from pg_gpu import HaplotypeMatrix, GenotypeMatrix
from pg_gpu.zarr_io import (
    detect_zarr_layout, read_genotypes, write_vcz, write_allel, vcf_to_zarr,
)


def _host(arr):
    """Bring an array back to the host for byte-equal comparison. from_zarr
    returns GPU haplotypes after the GPU-side prep refactor; round-trip
    tests compare against a host-built msprime hm, so callers need to be
    explicit about which device they're on."""
    return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr


# ── Fixtures ────────────────────────────────────────────────────────────


def _simulate_hm(n_samples=10, seq_length=10_000, seed=42):
    """Simulate a HaplotypeMatrix with msprime."""
    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=seq_length,
        recombination_rate=1e-4, random_seed=seed, ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed)
    return HaplotypeMatrix.from_ts(ts)


@pytest.fixture
def hm():
    return _simulate_hm()


@pytest.fixture
def vcz_store(tmp_path, hm):
    """Write a VCZ zarr store and return its path."""
    path = str(tmp_path / "test.vcz.zarr")
    hm.to_zarr(path, format='vcz', contig_name='chr1')
    return path


@pytest.fixture
def allel_store(tmp_path, hm):
    """Write a scikit-allel zarr store and return its path."""
    path = str(tmp_path / "test.allel.zarr")
    hm.to_zarr(path, format='scikit-allel')
    return path


@pytest.fixture
def grouped_store(tmp_path):
    """Write a chromosome-grouped scikit-allel zarr store."""
    path = str(tmp_path / "test.grouped.zarr")
    store = zarr.open(path, mode='w')

    hm1 = _simulate_hm(seed=1)
    hm2 = _simulate_hm(seed=2)

    for chrom, hm_data in [('chr1', hm1), ('chr2', hm2)]:
        hap = hm_data.haplotypes
        pos = hm_data.positions
        gt = HaplotypeMatrix._haplotypes_to_gt(hap)
        grp = store.create_group(chrom)
        grp.create_array('calldata/GT', data=gt)
        grp.create_array('variants/POS', data=pos)
        if hm_data.samples is not None:
            grp.create_array(
                'samples',
                data=np.array(hm_data.samples, dtype='U'),
            )

    return path


# ── Layout Detection ────────────────────────────────────────────────────


class TestDetectLayout:

    def test_vcz(self, vcz_store):
        store = zarr.open(vcz_store, mode='r')
        assert detect_zarr_layout(store) == 'vcz'

    def test_allel(self, allel_store):
        store = zarr.open(allel_store, mode='r')
        assert detect_zarr_layout(store) == 'scikit-allel'

    def test_grouped(self, grouped_store):
        store = zarr.open(grouped_store, mode='r')
        assert detect_zarr_layout(store) == 'scikit-allel-grouped'

    def test_unknown_raises(self, tmp_path):
        path = str(tmp_path / "empty.zarr")
        store = zarr.open(path, mode='w')
        store.create_array('random_data', data=np.array([1, 2, 3]))
        store = zarr.open(path, mode='r')
        with pytest.raises(ValueError, match="Unrecognized zarr layout"):
            detect_zarr_layout(store)


# ── HaplotypeMatrix Round-Trip ──────────────────────────────────────────


class TestHaplotypeMatrixRoundTrip:

    def test_vcz_roundtrip(self, tmp_path, hm):
        path = str(tmp_path / "rt.vcz.zarr")
        hm.to_zarr(path, format='vcz', contig_name='chr1')
        hm2 = HaplotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(_host(hm.haplotypes), _host(hm2.haplotypes))
        np.testing.assert_array_equal(_host(hm.positions), _host(hm2.positions))

    def test_allel_roundtrip(self, tmp_path, hm):
        path = str(tmp_path / "rt.allel.zarr")
        hm.to_zarr(path, format='scikit-allel')
        hm2 = HaplotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(_host(hm.haplotypes), _host(hm2.haplotypes))
        np.testing.assert_array_equal(_host(hm.positions), _host(hm2.positions))

    def test_cross_format_read(self, allel_store, hm):
        """Write in allel format, read with auto-detect."""
        hm2 = HaplotypeMatrix.from_zarr(allel_store)
        np.testing.assert_array_equal(_host(hm.haplotypes), _host(hm2.haplotypes))

    def test_samples_preserved(self, tmp_path):
        """Verify sample names survive round-trip when present."""
        hm = _simulate_hm()
        # Manually set sample names (from_ts doesn't set them)
        n_samples = hm.num_haplotypes // 2
        hm.samples = [f"sample_{i}" for i in range(n_samples)]
        path = str(tmp_path / "samples.vcz.zarr")
        hm.to_zarr(path, format='vcz', contig_name='chr1')
        hm2 = HaplotypeMatrix.from_zarr(path)
        assert hm2.samples == hm.samples

    def test_invalid_format_raises(self, tmp_path, hm):
        with pytest.raises(ValueError, match="Unknown format"):
            hm.to_zarr(str(tmp_path / "bad.zarr"), format='hdf5')


class TestPopFileKwarg:

    def _write_pops_tsv(self, path, n_dip):
        with open(path, "w") as f:
            f.write("sample\tpop\n")
            half = n_dip // 2
            for i in range(half):
                f.write(f"sample_{i}\tpop1\n")
            for i in range(half, n_dip):
                f.write(f"sample_{i}\tpop2\n")

    def _hm_with_samples(self, tmp_path, name):
        hm = _simulate_hm()
        n_dip = hm.num_haplotypes // 2
        hm.samples = [f"sample_{i}" for i in range(n_dip)]
        path = str(tmp_path / name)
        hm.to_zarr(path, format='vcz', contig_name='chr1')
        return path, n_dip

    def test_explicit_pop_file(self, tmp_path):
        path, n_dip = self._hm_with_samples(tmp_path, "explicit.vcz")
        popfile = str(tmp_path / "pops.tsv")
        self._write_pops_tsv(popfile, n_dip)
        hm = HaplotypeMatrix.from_zarr(path, pop_file=popfile)
        assert set(hm.sample_sets.keys()) == {"pop1", "pop2"}

    def test_companion_auto_load(self, tmp_path, capsys):
        path, n_dip = self._hm_with_samples(tmp_path, "auto.vcz")
        self._write_pops_tsv(path + ".pops.tsv", n_dip)
        hm = HaplotypeMatrix.from_zarr(path)
        assert hm.sample_sets is not None
        assert "auto-loaded" in capsys.readouterr().err

    def test_pop_file_false_disables_companion(self, tmp_path):
        path, n_dip = self._hm_with_samples(tmp_path, "disabled.vcz")
        self._write_pops_tsv(path + ".pops.tsv", n_dip)
        hm = HaplotypeMatrix.from_zarr(path, pop_file=False)
        # _sample_sets stays None; the .sample_sets property then returns
        # the default {"all": [...]} fallback rather than custom pops.
        assert hm._sample_sets is None

    def test_no_companion_no_pop_file(self, tmp_path):
        path, _ = self._hm_with_samples(tmp_path, "none.vcz")
        hm = HaplotypeMatrix.from_zarr(path)
        assert hm._sample_sets is None


# ── GenotypeMatrix Round-Trip ───────────────────────────────────────────


class TestGenotypeMatrixRoundTrip:

    def test_vcz_roundtrip(self, tmp_path, hm):
        from pg_gpu.genotype_matrix import GenotypeMatrix
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        path = str(tmp_path / "gm.vcz.zarr")
        gm.to_zarr(path, format='vcz', contig_name='chr1')
        gm2 = GenotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(gm.genotypes, gm2.genotypes)
        np.testing.assert_array_equal(gm.positions, gm2.positions)

    def test_allel_roundtrip(self, tmp_path, hm):
        from pg_gpu.genotype_matrix import GenotypeMatrix
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        path = str(tmp_path / "gm.allel.zarr")
        gm.to_zarr(path, format='scikit-allel')
        gm2 = GenotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(gm.genotypes, gm2.genotypes)


# ── Region Queries ──────────────────────────────────────────────────────


class TestRegionQueries:

    def test_vcz_region(self, vcz_store, hm):
        pos = hm.positions
        mid = int(pos[len(pos) // 2])
        region = f"chr1:{int(pos[0])}-{mid}"
        hm2 = HaplotypeMatrix.from_zarr(vcz_store, region=region)
        assert hm2.num_variants < hm.num_variants
        assert np.all(hm2.positions < mid)

    def test_vcz_multi_contig_no_region_raises(self, tmp_path):
        """Multi-contig VCZ without region should raise."""
        path = str(tmp_path / "multi.vcz.zarr")
        store = zarr.open(path, mode='w')
        n = 100
        gt = np.zeros((n, 5, 2), dtype=np.int8)
        pos = np.arange(n, dtype=np.int32)
        contig = np.array([0] * 50 + [1] * 50, dtype=np.int8)
        store.create_array('call_genotype', data=gt)
        store.create_array('variant_position', data=pos)
        store.create_array('variant_contig', data=contig)
        store.create_array('contig_id',
                           data=np.array(['chr1', 'chr2'], dtype='U'))
        with pytest.raises(ValueError, match="contains 2 contigs"):
            HaplotypeMatrix.from_zarr(path)

    def test_vcz_bad_contig_raises(self, vcz_store):
        with pytest.raises(ValueError, match="not found"):
            HaplotypeMatrix.from_zarr(vcz_store, region='chrZ:1-100')

    def test_vcz_empty_region_raises(self, vcz_store):
        with pytest.raises(ValueError, match="No variants"):
            HaplotypeMatrix.from_zarr(vcz_store,
                                      region='chr1:999999999-999999999')

    def test_grouped_region(self, grouped_store):
        hm = HaplotypeMatrix.from_zarr(grouped_store, region='chr1:1-100000')
        assert hm.num_variants > 0

    def test_grouped_no_region_raises(self, grouped_store):
        with pytest.raises(ValueError, match="requires region"):
            HaplotypeMatrix.from_zarr(grouped_store)

    def test_grouped_bad_chrom_raises(self, grouped_store):
        with pytest.raises(ValueError, match="not found"):
            HaplotypeMatrix.from_zarr(grouped_store, region='chrZ:1-100')

    def test_allel_region(self, allel_store, hm):
        pos = hm.positions
        mid = int(pos[len(pos) // 2])
        region = f"chr1:{int(pos[0])}-{mid}"
        hm2 = HaplotypeMatrix.from_zarr(allel_store, region=region)
        assert hm2.num_variants < hm.num_variants


# ── Missing Data ────────────────────────────────────────────────────────


class TestMissingData:

    def test_missing_data_roundtrip(self, tmp_path):
        """Verify -1 values survive VCZ round-trip."""
        hm = _simulate_hm()
        hap = hm.haplotypes.copy()
        # Inject missing data
        rng = np.random.default_rng(99)
        mask = rng.random(hap.shape) < 0.1
        hap[mask] = -1
        hm_missing = HaplotypeMatrix(
            hap, hm.positions, hm.chrom_start, hm.chrom_end
        )

        path = str(tmp_path / "missing.vcz.zarr")
        hm_missing.to_zarr(path, format='vcz', contig_name='chr1')
        hm2 = HaplotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(_host(hm_missing.haplotypes),
                                      _host(hm2.haplotypes))

    def test_genotype_mask_written(self, tmp_path):
        """Verify call_genotype_mask reflects missing values."""
        hm = _simulate_hm()
        hap = hm.haplotypes.copy()
        hap[0, 0] = -1
        hm_missing = HaplotypeMatrix(
            hap, hm.positions, hm.chrom_start, hm.chrom_end
        )

        path = str(tmp_path / "mask.vcz.zarr")
        hm_missing.to_zarr(path, format='vcz', contig_name='chr1')
        store = zarr.open(path, mode='r')
        gt_mask = np.array(store['call_genotype_mask'])
        gt = np.array(store['call_genotype'])
        # Where gt < 0, mask should be True
        np.testing.assert_array_equal(gt_mask, gt < 0)

    def test_genotype_missing_roundtrip(self, tmp_path):
        """Verify -1 survives GenotypeMatrix VCZ round-trip."""
        hm = _simulate_hm()
        gm = GenotypeMatrix.from_haplotype_matrix(hm)
        geno = gm.genotypes.copy()
        geno[0, 0] = -1
        gm_missing = GenotypeMatrix(
            geno, gm.positions, gm.chrom_start, gm.chrom_end
        )

        path = str(tmp_path / "gm_missing.vcz.zarr")
        gm_missing.to_zarr(path, format='vcz', contig_name='chr1')
        gm2 = GenotypeMatrix.from_zarr(path)
        np.testing.assert_array_equal(gm_missing.genotypes, gm2.genotypes)


# ── vcf_to_zarr ─────────────────────────────────────────────────────────


class TestVcfToZarr:

    def test_basic_conversion(self, tmp_path):
        """Convert a simulated VCF to zarr and verify loadable."""
        ts = msprime.sim_ancestry(
            samples=5, sequence_length=10_000,
            recombination_rate=1e-4, random_seed=42, ploidy=2,
        )
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=42)

        vcf_path = str(tmp_path / "test.vcf")
        with open(vcf_path, 'w') as f:
            ts.write_vcf(f)

        # bgzip and index
        import subprocess
        subprocess.run(['bgzip', vcf_path], check=True)
        subprocess.run(['tabix', '-p', 'vcf', vcf_path + '.gz'], check=True)

        zarr_path = str(tmp_path / "test.zarr")
        vcf_to_zarr(vcf_path + '.gz', zarr_path,
                     worker_processes=1, show_progress=False)

        # Verify it loads
        hm = HaplotypeMatrix.from_zarr(zarr_path)
        assert hm.num_variants > 0
        assert hm.num_haplotypes == 10  # 5 diploid samples

    def test_icf_cleanup(self, tmp_path):
        """Verify ICF temp directory is cleaned up."""
        ts = msprime.sim_ancestry(
            samples=3, sequence_length=5_000,
            recombination_rate=1e-4, random_seed=43, ploidy=2,
        )
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=43)

        vcf_path = str(tmp_path / "test2.vcf")
        with open(vcf_path, 'w') as f:
            ts.write_vcf(f)

        import subprocess
        subprocess.run(['bgzip', vcf_path], check=True)
        subprocess.run(['tabix', '-p', 'vcf', vcf_path + '.gz'], check=True)

        zarr_path = str(tmp_path / "test2.zarr")
        icf_path = str(tmp_path / "custom_icf")
        vcf_to_zarr(vcf_path + '.gz', zarr_path,
                     worker_processes=1, icf_path=icf_path,
                     show_progress=False)

        import os
        assert os.path.exists(zarr_path)
        assert not os.path.exists(icf_path)

    def test_static_method(self, tmp_path):
        """Verify HaplotypeMatrix.vcf_to_zarr works."""
        ts = msprime.sim_ancestry(
            samples=3, sequence_length=5_000,
            recombination_rate=1e-4, random_seed=44, ploidy=2,
        )
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=44)

        vcf_path = str(tmp_path / "test3.vcf")
        with open(vcf_path, 'w') as f:
            ts.write_vcf(f)

        import subprocess
        subprocess.run(['bgzip', vcf_path], check=True)
        subprocess.run(['tabix', '-p', 'vcf', vcf_path + '.gz'], check=True)

        zarr_path = str(tmp_path / "test3.zarr")
        HaplotypeMatrix.vcf_to_zarr(
            vcf_path + '.gz', zarr_path,
            worker_processes=1, show_progress=False
        )
        hm = HaplotypeMatrix.from_zarr(zarr_path)
        assert hm.num_variants > 0


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_single_variant(self, tmp_path):
        """Store with a single variant."""
        gt = np.array([[[0, 1]]], dtype=np.int8)  # (1, 1, 2)
        pos = np.array([100], dtype=np.int32)
        path = str(tmp_path / "single.zarr")
        write_vcz(path, gt, pos, samples=['s1'], contig_name='chr1')
        hm = HaplotypeMatrix.from_zarr(path)
        assert hm.num_variants == 1
        assert hm.num_haplotypes == 2

    def test_all_missing(self, tmp_path):
        """Store where every genotype is missing."""
        gt = np.full((5, 3, 2), -1, dtype=np.int8)
        pos = np.arange(5, dtype=np.int32) * 100
        path = str(tmp_path / "all_missing.zarr")
        write_vcz(path, gt, pos, samples=['a', 'b', 'c'],
                  contig_name='chr1')
        hm = HaplotypeMatrix.from_zarr(path)
        assert hm.num_variants == 5
        assert np.all(hm.haplotypes == -1)
