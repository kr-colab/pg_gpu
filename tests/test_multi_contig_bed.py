"""A BED file with more than one contig selects only the data's contig.

Each path that resolves a BED takes the data's own contig (the zarr loaders,
the streaming windowed path, the VCF loaders), or raises when it cannot know
it and the BED has more than one contig.
"""
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu.accessible import resolve_accessible_mask
from pg_gpu.moments_ld import compute_ld_statistics
from pg_gpu.windowed_analysis import windowed_analysis

from .conftest import simulate_hm


@pytest.fixture
def beds(tmp_path):
    """The data's contig "1" is callable on 25001..75000. The two-contig BED
    adds contig "2" intervals that a merge would count as contig 1's."""
    one = tmp_path / "one.bed"
    one.write_text("1\t25000\t75000\n")
    two = tmp_path / "two.bed"
    two.write_text("1\t25000\t75000\n2\t0\t25000\n2\t75000\t100000\n")
    return str(one), str(two)


@pytest.fixture
def vcz_store(tmp_path):
    hm = simulate_hm(n_samples=20, seq_length=100_000, seed=42,
                     mutation_model='binary')
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "data.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


def test_resolve_needs_chrom_only_for_multi_contig(beds):
    one, two = beds
    with pytest.raises(ValueError, match="2 contigs"):
        resolve_accessible_mask(two, 1, 100_000)
    assert resolve_accessible_mask(one, 1, 100_000).total_accessible == 50_000
    assert resolve_accessible_mask(two, 1, 100_000, chrom="1").total_accessible == 50_000


@pytest.mark.parametrize("cls", [HaplotypeMatrix, GenotypeMatrix])
def test_eager_from_zarr_uses_store_contig(vcz_store, beds, cls):
    # With no region, the eager loader takes the store's contig, as the
    # streaming loader does, so both see the same accessible bases.
    one, two = beds
    eager = cls.from_zarr(vcz_store, streaming="never", accessible_bed=two)
    stream = cls.from_zarr(vcz_store, streaming="always", chunk_bp=10_000,
                           accessible_bed=two)
    single = cls.from_zarr(vcz_store, streaming="never", accessible_bed=one)
    assert eager.n_total_sites == single.n_total_sites == 50_000
    assert stream.accessible_mask.total_accessible == 50_000
    assert eager.num_variants == single.num_variants


def _pi(matrix, bed, **kw):
    df = windowed_analysis(matrix, window_size=10_000, statistics=["pi"],
                           accessible_bed=bed, **kw)
    return df.sort_values("start").reset_index(drop=True)


def test_windowed_analysis_eager_needs_chrom(vcz_store, beds):
    one, two = beds

    def eager():
        return HaplotypeMatrix.from_zarr(vcz_store, streaming="never")

    with pytest.raises(ValueError, match="pass chrom"):
        _pi(eager(), two)
    got, want = _pi(eager(), two, chrom="1"), _pi(eager(), one)
    np.testing.assert_allclose(got["pi"], want["pi"], equal_nan=True)
    np.testing.assert_array_equal(got["n_variants"], want["n_variants"])


def test_windowed_analysis_streaming_uses_stream_contig(vcz_store, beds):
    one, two = beds
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                       chunk_bp=10_000)
    got = _pi(stream, two)
    want = _pi(HaplotypeMatrix.from_zarr(vcz_store, streaming="never"), one)
    np.testing.assert_allclose(got["pi"], want["pi"], rtol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(got["n_variants"], want["n_variants"])


# The genotype loader drops the fixture's multiallelic sites with this
# warning; the test is about the mask, not that drop.
@pytest.mark.filterwarnings("ignore::pg_gpu.BiallelicOnlyWarning")
@pytest.mark.parametrize("use_genotypes", [True, False],
                         ids=["genotype", "haplotype"])
def test_moments_ld_vcf_path_uses_vcf_contig(sample_vcf, tmp_path,
                                             use_genotypes):
    # The fixture VCF is contig "1" over 1 kb; contig "2" covers what
    # contig 1's BED leaves out, so a merge would accept every variant.
    pop = tmp_path / "pops.tsv"
    pop.write_text("sample\tpop\n" + "".join(f"tsk_{i}\tpop0\n" for i in range(10)))
    one = tmp_path / "one.bed"
    one.write_text("1\t0\t500\n")
    two = tmp_path / "two.bed"
    two.write_text("1\t0\t500\n2\t500\t1000\n")
    kw = dict(vcf_file=sample_vcf, pop_file=str(pop), pops=["pop0"],
              bp_bins=[0, 100, 500], use_genotypes=use_genotypes, report=False)
    got = compute_ld_statistics(accessible_bed=str(two), **kw)
    want = compute_ld_statistics(accessible_bed=str(one), **kw)
    for g, w in zip(got["sums"], want["sums"]):
        np.testing.assert_allclose(g, w)
