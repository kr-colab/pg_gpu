"""A BED file with more than one contig selects only the data's contig.

Each path that resolves a BED takes the data's own contig (the zarr loaders,
the streaming windowed path, the VCF loaders), or raises when it cannot know
it and the BED has more than one contig. A BED that names the contig
differently ('chr1' for '1') is matched by name with a ContigNameWarning.
"""
import numpy as np
import pytest

from pg_gpu import ContigNameWarning, GenotypeMatrix, HaplotypeMatrix
from pg_gpu.accessible import parse_bed, resolve_accessible_mask
from pg_gpu.moments_ld import compute_ld_statistics
from pg_gpu.windowed_analysis import windowed_analysis

from .conftest import simulate_hm


def _write_beds(tmp_path, prefix):
    """One- and two-contig BEDs. The data's contig "1" is callable on
    25001..75000; the second contig's intervals are what a merge would count
    as contig 1's. ``prefix`` gives the contig names ("" or UCSC "chr")."""
    one = tmp_path / f"{prefix}one.bed"
    one.write_text(f"{prefix}1\t25000\t75000\n")
    two = tmp_path / f"{prefix}two.bed"
    two.write_text(f"{prefix}1\t25000\t75000\n{prefix}2\t0\t25000\n"
                   f"{prefix}2\t75000\t100000\n")
    return str(one), str(two)


@pytest.fixture
def beds(tmp_path):
    return _write_beds(tmp_path, "")


@pytest.fixture
def chr_beds(tmp_path):
    return _write_beds(tmp_path, "chr")


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

    with pytest.raises(ValueError, match=r"region= \(from_zarr, from_vcf\)"):
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
    # UCSC contig names in the BED ('chr1' for the VCF's '1') match by name.
    chr_bed = tmp_path / "chr.bed"
    chr_bed.write_text("chr1\t0\t500\nchr2\t500\t1000\n")
    with pytest.warns(ContigNameWarning):
        got = compute_ld_statistics(accessible_bed=str(chr_bed), **kw)
    for g, w in zip(got["sums"], want["sums"]):
        np.testing.assert_allclose(g, w)


def test_resolve_matches_contig_names_without_chr(chr_beds):
    _, chr_two = chr_beds
    # parse_bed applies the same name rule as the mask builders.
    with pytest.warns(ContigNameWarning):
        assert parse_bed(chr_two, chrom="1") == [("chr1", 25000, 75000)]
    with pytest.warns(ContigNameWarning, match="'chr1'.*'1'"):
        mask = resolve_accessible_mask(chr_two, 1, 100_000, chrom="1")
    assert mask.total_accessible == 50_000
    # No contig of that name, even without 'chr': the error lists the BED's.
    with pytest.raises(ValueError, match="its contigs are: chr1, chr2"):
        resolve_accessible_mask(chr_two, 1, 100_000, chrom="3")


@pytest.mark.parametrize("which", [0, 1], ids=["one_contig", "two_contigs"])
@pytest.mark.parametrize("cls", [HaplotypeMatrix, GenotypeMatrix])
@pytest.mark.parametrize("streaming", ["never", "always"])
def test_from_zarr_matches_chr_named_bed(vcz_store, chr_beds, cls, streaming,
                                         which):
    with pytest.warns(ContigNameWarning):
        m = cls.from_zarr(vcz_store, streaming=streaming, chunk_bp=10_000,
                          accessible_bed=chr_beds[which])
    assert m.accessible_mask.count_accessible(1, 100_001) == 50_000


def test_windowed_analysis_matches_chr_named_bed(vcz_store, beds, chr_beds):
    want = _pi(HaplotypeMatrix.from_zarr(vcz_store, streaming="never"), beds[0])
    stream = HaplotypeMatrix.from_zarr(vcz_store, streaming="always",
                                       chunk_bp=10_000)
    with pytest.warns(ContigNameWarning):
        got = _pi(stream, chr_beds[1])
    np.testing.assert_allclose(got["pi"], want["pi"], rtol=1e-6, equal_nan=True)
    with pytest.warns(ContigNameWarning):
        got = _pi(HaplotypeMatrix.from_zarr(vcz_store, streaming="never"),
                  chr_beds[1], chrom="1")
    np.testing.assert_allclose(got["pi"], want["pi"], equal_nan=True)


def test_flat_allel_store_uses_its_chrom_column(tmp_path, beds):
    # A flat scikit-allel store has no contig table; with one value in its
    # CHROM column, that value names the contig.
    import zarr
    hm = simulate_hm(n_samples=20, seq_length=100_000, seed=42,
                     mutation_model='binary')
    path = str(tmp_path / "flat.allel.zarr")
    hm.to_zarr(path, format="scikit-allel")
    zarr.open(path, mode="a").create_array(
        "variants/CHROM", data=np.full(hm.num_variants, "1", dtype="<U1"))
    m = HaplotypeMatrix.from_zarr(path, accessible_bed=beds[1])
    assert m.n_total_sites == 50_000


_TWO_CONTIG_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##contig=<ID=2,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t0|0\t1|1\t0|1
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t0|1\t1|1
2\t600\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1\t0|0\t0|1
2\t700\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|1\t0|1\t0|0
"""


@pytest.fixture
def two_contig_vcf(tmp_path):
    """A VCF with variants on contigs 1 and 2, and a BED for both."""
    vcf = tmp_path / "two.vcf"
    vcf.write_text(_TWO_CONTIG_VCF)
    bed = tmp_path / "two.bed"
    bed.write_text("1\t0\t150\n2\t0\t650\n")
    return str(vcf), str(bed)


@pytest.mark.parametrize("cls", [HaplotypeMatrix, GenotypeMatrix])
def test_from_vcf_with_several_contigs_raises(two_contig_vcf, cls):
    vcf, bed = two_contig_vcf
    with pytest.raises(ValueError, match="variants on 2 contigs"):
        cls.from_vcf(vcf, accessible_bed=bed)


def test_from_vcf_region_selects_one_contig(two_contig_vcf):
    vcf, bed = two_contig_vcf
    hm = HaplotypeMatrix.from_vcf(vcf, region="2", accessible_bed=bed)
    assert hm.positions.tolist() == [600]


def test_flat_allel_store_with_several_contigs_raises(tmp_path, beds):
    # A flat scikit-allel store has no contig table; its CHROM column holds
    # both contigs, so no single contig names the mask.
    import zarr
    hm = simulate_hm(n_samples=20, seq_length=100_000, seed=42,
                     mutation_model='binary')
    path = str(tmp_path / "flat2.allel.zarr")
    hm.to_zarr(path, format="scikit-allel")
    chrom = np.where(np.arange(hm.num_variants) < hm.num_variants // 2, "1", "2")
    zarr.open(path, mode="a").create_array("variants/CHROM", data=chrom)
    with pytest.raises(ValueError, match="variants on 2 contigs"):
        HaplotypeMatrix.from_zarr(path, accessible_bed=beds[0])


def test_contig_name_warning_points_at_caller(vcz_store, chr_beds):
    # Each entry point reaches the warning through a different number of
    # pg_gpu frames; the warning must name this file every time.
    _, chr_two = chr_beds
    calls = [
        lambda: parse_bed(chr_two, chrom="1"),
        lambda: resolve_accessible_mask(chr_two, 1, 100_000, chrom="1"),
        lambda: HaplotypeMatrix.from_zarr(vcz_store, streaming="never",
                                          accessible_bed=chr_two),
    ]
    for call in calls:
        with pytest.warns(ContigNameWarning) as record:
            call()
        assert all(w.filename == __file__ for w in record
                   if issubclass(w.category, ContigNameWarning))


@pytest.mark.parametrize("region, want", [
    ("1-5", ["1"]),        # positions only: contig 1's variants
    ("2:1-5", ["1"]),      # the flat filter reads positions, so still contig 1
    (None, ["1", "2"]),
])
def test_flat_allel_contigs_follow_the_variants_read(region, want):
    import zarr
    from pg_gpu.zarr_io import read_genotypes_allel
    store = zarr.group()
    store["variants/POS"] = np.array([1, 2, 3, 4, 5, 1000, 1001, 1002, 1003, 1004])
    store["variants/CHROM"] = np.array(["1"] * 5 + ["2"] * 5)
    store["calldata/GT"] = np.zeros((10, 2, 2), dtype=np.int8)
    store["samples"] = np.array(["s1", "s2"])
    assert read_genotypes_allel(store, region=region)["contigs"] == want
