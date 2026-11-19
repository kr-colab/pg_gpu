"""Polarization by an ancestral-allele tag.

pg_gpu reads allele code 0 as the ancestral allele. A VCF gives code 0 to
REF, which is not the ancestral allele in general. ``ancestral=`` on the
loaders, and ``HaplotypeMatrix.polarize``, recode each site so that code 0
is the allele that an ``AA`` tag names, and drop the sites where it names
none.

A tree sequence is the truth: its own matrix has the ancestral state at
code 0. The fixtures write it to a VCF whose REF is a random allele of each
site, with the true ancestral allele in ``AA``.
"""

import warnings

import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import (GenotypeMatrix, HaplotypeMatrix, UnknownAncestralWarning,
                    diversity, selection, sfs)
from pg_gpu._gpu_genotype_prep import swap_ancestral
from pg_gpu.zarr_io import ancestral_codes, swap_allele_columns

from .conftest import bgzip_index

pytestmark = pytest.mark.filterwarnings(
    "ignore::pg_gpu.UnknownAncestralWarning")

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length={length}>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n")


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _simulate(model, seed=3):
    ts = msprime.sim_ancestry(8, sequence_length=40_000, recombination_rate=1e-8,
                              population_size=1e4, random_seed=seed)
    return msprime.sim_mutations(ts, rate=2e-7, model=model, random_seed=seed)


def _scrambled_vcf(ts, path, seed=0):
    """Write ``ts`` as a VCF whose REF is a random allele of each site and
    whose ``AA`` is the ancestral allele, lower case at some sites and a
    placeholder at others. Returns the index of each site with a known
    ancestral allele, and of each one written in upper case."""
    rng = np.random.default_rng(seed)
    known, upper, lines = [], [], []
    for var in ts.variants():
        alleles = [a for a in var.alleles if a is not None]
        ref = int(rng.integers(len(alleles)))
        order = [ref] + [i for i in range(len(alleles)) if i != ref]
        new_code = np.empty(len(alleles), dtype=np.int64)
        new_code[order] = np.arange(len(alleles))
        calls = new_code[var.genotypes]
        draw = rng.random()
        if draw < 0.15:
            aa = str(rng.choice([".", "N", "-"]))
        elif draw < 0.30:
            aa = alleles[0].lower()
            known.append(var.site.id)
        else:
            aa = alleles[0]
            known.append(var.site.id)
            upper.append(var.site.id)
        gts = "\t".join(f"{a}|{b}" for a, b in zip(calls[0::2], calls[1::2]))
        alt = ",".join(alleles[i] for i in order[1:])
        lines.append(f"1\t{int(var.site.position) + 1}\t.\t{alleles[ref]}\t{alt}"
                     f"\t.\t.\tAA={aa}\tGT\t{gts}\n")
    with open(path, "w") as f:
        f.write(_HEADER.format(length=int(ts.sequence_length),
                               samples="\t".join(f"s{i}" for i in range(ts.num_samples // 2))))
        f.writelines(lines)
    return np.array(known), np.array(upper)


class _Case:
    """A simulation, its scrambled VCF and VCZ store, and the truth."""

    def __init__(self, tmp_path_factory, model):
        self.ts = _simulate(model)
        d = tmp_path_factory.mktemp(f"ancestral_{model}")
        vcf = str(d / "scrambled.vcf")
        self.known, self.upper = _scrambled_vcf(self.ts, vcf)
        self.vcf = bgzip_index(vcf)
        self.vcz = str(d / "scrambled.vcz")
        HaplotypeMatrix.vcf_to_zarr(self.vcf, self.vcz, show_progress=False,
                                    worker_processes=1, zarr_format=3)
        self.truth = HaplotypeMatrix.from_ts(self.ts)

    def expected(self, sites=None):
        return self.truth.get_subset(self.known if sites is None else sites)

    def load(self, how, **kwargs):
        kwargs.setdefault("ancestral", "AA")
        if how == "vcf":
            return HaplotypeMatrix.from_vcf(self.vcf, phased=True, **kwargs)
        if how == "eager":
            return HaplotypeMatrix.from_zarr(self.vcz, streaming="never", **kwargs)
        stream = self.stream(how.split(":")[1], **kwargs)
        assert len(stream._chunks) > 1
        return stream.materialize()

    def stream(self, backend, **kwargs):
        kwargs.setdefault("ancestral", "AA")
        return HaplotypeMatrix.from_zarr(self.vcz, streaming="always", backend=backend,
                                         chunk_bp=7_000, **kwargs)


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    case = _Case(tmp_path_factory, "binary")
    assert 0 < case.known.size < case.ts.num_sites
    return case


@pytest.fixture(scope="module")
def multiallelic(tmp_path_factory):
    case = _Case(tmp_path_factory, "jc69")
    assert (_host(case.truth.haplotypes) > 1).any()
    return case


PATHS = ["vcf", "eager", "stream:host", "stream:kvikio"]


# ── the loaders against the tree sequence ─────────────────────────────
class TestLoadersGiveTheTruth:

    @pytest.mark.parametrize("how", PATHS)
    def test_biallelic_matrix_equals_the_tree_sequence(self, binary, how):
        loaded = binary.load(how)
        np.testing.assert_array_equal(_host(loaded.haplotypes),
                                      _host(binary.expected().haplotypes))
        np.testing.assert_array_equal(_host(loaded.positions),
                                      _host(binary.expected().positions) + 1)

    def test_the_test_can_fail(self, binary):
        # With no ancestral tag, code 0 is the scrambled REF.
        raw = HaplotypeMatrix.from_zarr(binary.vcz).get_subset(binary.known)
        assert not np.array_equal(_host(raw.haplotypes),
                                  _host(binary.expected().haplotypes))
        assert not np.array_equal(sfs.sfs(raw), sfs.sfs(binary.expected()))

    @pytest.mark.parametrize("how", PATHS)
    def test_multiallelic_sites_have_the_ancestral_allele_at_code_zero(self, multiallelic, how):
        # The derived alleles keep the codes of the VCF, which are not
        # those of the tree sequence; the ancestral class is the same.
        loaded = multiallelic.load(how)
        np.testing.assert_array_equal(_host(loaded.haplotypes) == 0,
                                      _host(multiallelic.expected().haplotypes) == 0)

    @pytest.mark.parametrize("how", ["vcf", "stream:host"])
    @pytest.mark.parametrize("statistic", [
        lambda m: sfs.sfs(m),
        lambda m: diversity.theta_h(m, span_normalize=False),
        lambda m: diversity.theta_l(m, span_normalize=False),
        lambda m: diversity.fay_wus_h(m),
        lambda m: diversity.zeng_e(m),
        lambda m: diversity.max_daf(m),
    ], ids=["sfs", "theta_h", "theta_l", "fay_wus_h", "zeng_e", "max_daf"])
    def test_derived_state_statistics(self, multiallelic, how, statistic):
        np.testing.assert_allclose(_host(statistic(multiallelic.load(how))),
                                   _host(statistic(multiallelic.expected())), rtol=1e-12)

    def test_sign_of_ihs(self, binary):
        got = _host(selection.ihs(binary.load("eager")))
        want = _host(selection.ihs(binary.expected()))
        np.testing.assert_allclose(got, want, rtol=1e-9, equal_nan=True)

    @pytest.mark.parametrize("how", PATHS)
    def test_lowercase_as_unknown(self, multiallelic, how):
        # The nucleotide model: the binary alleles "0" and "1" have no case.
        case = multiallelic
        loaded = case.load(how, ancestral_lowercase_known=False)
        assert case.upper.size < case.known.size
        np.testing.assert_array_equal(_host(loaded.haplotypes) == 0,
                                      _host(case.expected(case.upper).haplotypes) == 0)

    @pytest.mark.parametrize("how", ["eager", "stream:host", "stream:kvikio"])
    def test_region(self, binary, how):
        whole = binary.load("eager")
        pos = _host(whole.positions)
        lo, hi = int(pos[pos.size // 4]), int(pos[3 * pos.size // 4])
        part = binary.load(how, region=f"1:{lo}-{hi}")
        keep = (pos >= lo) & (pos <= hi)
        np.testing.assert_array_equal(_host(part.haplotypes), _host(whole.haplotypes)[:, keep])

    @pytest.mark.parametrize("backend", ["host", "kvikio"])
    def test_stream_counts_and_subsets_the_kept_sites(self, binary, backend):
        stream = binary.stream(backend)
        want = _host(binary.expected().haplotypes)
        assert stream.num_variants == binary.known.size
        chunks = [_host(m.haplotypes) for _, _, m in stream.iter_gpu_chunks()]
        np.testing.assert_array_equal(np.concatenate(chunks, axis=1), want)
        rows = [5, 0, 9, 2]
        np.testing.assert_array_equal(_host(stream.materialize(sample_subset=rows).haplotypes),
                                      want[rows])

    def test_from_vcf_gives_a_host_matrix(self, binary):
        assert binary.load("vcf").device == "CPU"

    def test_fields_stay_aligned_and_alleles_follow_the_codes(self, binary):
        loaded = binary.load("eager", fields=["AA", "allele"])
        aa = np.strings.upper(loaded.fields["AA"])
        assert loaded.fields["AA"].shape == (binary.known.size,)
        # Code 0 is the ancestral allele, so column 0 names it.
        np.testing.assert_array_equal(np.strings.upper(loaded.fields["allele"][:, 0]), aa)

    def test_accessible_mask_is_set_on_the_kept_sites(self, binary, tmp_path):
        bed = tmp_path / "half.bed"
        bed.write_text("1\t0\t20000\n")
        loaded = binary.load("eager", accessible_bed=str(bed))
        want = binary.expected()
        keep = _host(want.positions) + 1 <= 20_000
        np.testing.assert_array_equal(_host(loaded.haplotypes), _host(want.haplotypes)[:, keep])


class TestRoundTrip:

    def test_second_polarization_changes_nothing(self, binary):
        # The allele strings of the matrix follow the codes, so the same
        # tag names code 0 at every site the second time.
        first = binary.load("eager", fields=["AA", "allele"])
        again = first.polarize(first.fields["AA"])
        np.testing.assert_array_equal(_host(again.haplotypes), _host(first.haplotypes))

    def test_store_without_the_strings_cannot_be_polarized_again(self, binary, tmp_path):
        path = str(tmp_path / "saved.vcz")
        binary.load("eager").to_zarr(path, contig_name="1")
        for streaming in ("never", "always"):
            with pytest.raises(ValueError, match="variant_AA"):
                HaplotypeMatrix.from_zarr(path, ancestral="AA", streaming=streaming)


# ── unknown sites ─────────────────────────────────────────────────────
class TestUnknownSites:

    @pytest.mark.parametrize("how", PATHS)
    def test_drop_warns_with_the_count(self, binary, how):
        n_drop = binary.ts.num_sites - binary.known.size
        with pytest.warns(UnknownAncestralWarning,
                          match=f"{n_drop} of {binary.ts.num_sites} sites"):
            binary.load(how)

    def test_no_warning_when_every_site_is_known(self):
        hm = HaplotypeMatrix(np.array([[0, 1], [1, 0]], dtype=np.int8), np.array([1, 2]), 0, 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnknownAncestralWarning)
            hm.polarize(np.array([1, 0]))

    def test_all_unknown_raises(self):
        hm = HaplotypeMatrix(np.array([[0, 1], [1, 0]], dtype=np.int8), np.array([1, 2]), 0, 3)
        with pytest.raises(ValueError, match="none of the 2 sites"):
            hm.polarize(np.array([-1, -1]))

    def test_bounds_are_those_of_all_the_sites(self, tmp_path):
        # The first and the last site have no known ancestral allele. The
        # drop must not move the bounds, eager or streaming: they set the
        # span, the window grid and the range of a BED mask.
        vcf = str(tmp_path / "ends.vcf")
        rows = [(100, "."), (200, "A"), (300, "T"), (400, "N")]
        with open(vcf, "w") as f:
            f.write(_HEADER.format(length=1000, samples="s0\ts1"))
            for pos, aa in rows:
                f.write(f"1\t{pos}\t.\tA\tT\t.\t.\tAA={aa}\tGT\t0|1\t1|1\n")
        vcz = str(tmp_path / "ends.vcz")
        HaplotypeMatrix.vcf_to_zarr(bgzip_index(vcf), vcz, show_progress=False,
                                    worker_processes=1, zarr_format=3)
        eager = HaplotypeMatrix.from_zarr(vcz, ancestral="AA")
        stream = HaplotypeMatrix.from_zarr(vcz, ancestral="AA", streaming="always")
        assert _host(eager.positions).tolist() == [200, 300]
        assert stream.num_variants == 2
        assert (eager.chrom_start, eager.chrom_end) == (100, 400)
        assert (stream.chrom_start, stream.chrom_end) == (100, 400)
        assert stream.get_span() == eager.get_span()


# ── the match of ancestral strings to allele codes ────────────────────
class TestAncestralCodes:

    ALLELES = np.array([["A", "T", ""], ["C", "G", "T"], ["CA", "C", ""]])

    @pytest.mark.parametrize("aa,codes", [
        (["A", "G", "CA"], [0, 1, 0]),
        (["T", "T", "C"], [1, 2, 1]),
        (["t", "c", "ca"], [1, 0, 0]),
        # '' pads the allele columns; a missing AA must not take its code.
        (["", "", ""], [-1, -1, -1]),
        ([".", "N", "-"], [-1, -1, -1]),
        (["G", "A", "CAT"], [-1, -1, -1]),
        (["T|||", "g|Hsap|x", " CA "], [1, 1, 0]),
        (["|||", ".|||", "N|T"], [-1, -1, -1]),
    ], ids=["ref", "alt", "lowercase", "empty", "placeholder", "not-an-allele",
            "ensembl-form", "ensembl-form-unknown"])
    def test_codes(self, aa, codes):
        np.testing.assert_array_equal(ancestral_codes(aa, self.ALLELES), codes)

    def test_lowercase_as_unknown(self):
        np.testing.assert_array_equal(
            ancestral_codes(["t", "C", "Ca"], self.ALLELES, lowercase_known=False), [-1, 0, -1])

    @pytest.mark.parametrize("dtype", [object, "U8", np.dtypes.StringDType()])
    def test_string_dtypes(self, dtype):
        got = ancestral_codes(np.array(["T", "T", "C"], dtype=dtype),
                              self.ALLELES.astype(dtype))
        assert got.dtype == np.int8 and got.tolist() == [1, 2, 1]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="one value per variant"):
            ancestral_codes(["A"], self.ALLELES)

    def test_swap_allele_columns(self):
        out = swap_allele_columns(self.ALLELES, np.array([1, 2, -1]))
        assert out.tolist() == [["T", "A", ""], ["T", "G", "C"], ["CA", "C", ""]]


# ── the recode ────────────────────────────────────────────────────────
class TestSwapKernel:

    def test_codes_change_places(self):
        haps = cp.asarray(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2],
                                    [-1, -1, -1, -1], [-2, -2, -2, -2]], dtype=np.int8))
        out = swap_ancestral(haps.copy(), np.array([0, 1, 2, -1], dtype=np.int8)).get()
        assert out.tolist() == [[0, 1, 2, 0], [1, 0, 1, 1], [2, 2, 0, 2],
                                [-1, -1, -1, -1], [-2, -2, -2, -2]]

    def test_twice_is_the_identity(self):
        rng = np.random.default_rng(0)
        haps = cp.asarray(rng.integers(-1, 4, (30, 500)).astype(np.int8))
        codes = rng.integers(-1, 4, 500).astype(np.int8)
        once = swap_ancestral(haps.copy(), codes)
        assert not (once == haps).all()
        assert (swap_ancestral(once, codes) == haps).all()

    def test_wrong_number_of_codes_raises(self):
        with pytest.raises(ValueError, match="one ancestral code per variant"):
            swap_ancestral(cp.zeros((2, 3), dtype=cp.int8), np.zeros(4, dtype=np.int8))


# ── HaplotypeMatrix.polarize ──────────────────────────────────────────
class TestPolarize:

    @staticmethod
    def _hm(device="GPU", **kwargs):
        hap = np.array([[0, 0, 1, 0], [1, 0, 2, 1], [1, 1, 0, -1], [0, 1, 2, 1]], dtype=np.int8)
        pos = np.array([10, 20, 30, 40])
        if device == "GPU":
            hap, pos = cp.asarray(hap), cp.asarray(pos)
        return HaplotypeMatrix(hap, pos, 0, 50, **kwargs)

    @pytest.mark.parametrize("device", ["CPU", "GPU"])
    def test_integer_codes(self, device):
        hm = self._hm(device, phased=True, samples=["s", "t"])
        out = hm.polarize(np.array([1, -1, 2, 0]))
        assert out.device == device and out.phased is True and out.samples == ["s", "t"]
        assert _host(out.positions).tolist() == [10, 30, 40]
        assert _host(out.haplotypes).tolist() == [[1, 1, 0], [0, 0, 1], [0, 2, -1], [1, 0, 1]]
        # The parent is not changed.
        assert _host(hm.haplotypes)[0].tolist() == [0, 0, 1, 0]

    def test_strings_need_the_alleles(self):
        with pytest.raises(ValueError, match="allele strings"):
            self._hm().polarize(["A", "T", "G", "C"])

    def test_strings_with_alleles(self):
        alleles = np.array([["A", "T", ""], ["A", "T", ""], ["C", "G", "T"], ["C", "G", ""]])
        by_string = self._hm().polarize(["T", ".", "T", "C"], alleles=alleles)
        by_code = self._hm().polarize(np.array([1, -1, 2, 0]))
        np.testing.assert_array_equal(_host(by_string.haplotypes), _host(by_code.haplotypes))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="one ancestral allele per variant"):
            self._hm().polarize(np.array([0, 1]))

    def test_fields_are_cut_to_the_kept_sites(self):
        hm = self._hm(fields={"QUAL": np.array([1.0, 2.0, 3.0, 4.0])})
        assert hm.polarize(np.array([1, -1, 2, 0])).fields["QUAL"].tolist() == [1.0, 3.0, 4.0]

    def test_ploidy_is_kept(self):
        assert self._hm(ploidy=1).polarize(np.array([1, 0, 2, 0])).ploidy == 1


# ── inputs that cannot be polarized ───────────────────────────────────
class TestErrors:

    def test_tag_not_in_the_store(self, binary):
        for streaming in ("never", "always"):
            with pytest.raises(ValueError, match="variant_ANC"):
                HaplotypeMatrix.from_zarr(binary.vcz, ancestral="ANC", streaming=streaming)

    def test_tag_not_in_the_vcf_header(self, binary):
        with pytest.raises(ValueError, match="no such INFO tag"):
            HaplotypeMatrix.from_vcf(binary.vcf, ancestral="ANC")

    def test_scikit_allel_store(self, binary, tmp_path):
        path = str(tmp_path / "allel.zarr")
        binary.truth.to_zarr(path, format="scikit-allel")
        with pytest.raises(ValueError, match="scikit-allel store"):
            HaplotypeMatrix.from_zarr(path, ancestral="AA")

    def test_genotype_loaders_do_not_take_the_argument(self, binary):
        with pytest.raises(TypeError):
            GenotypeMatrix.from_zarr(binary.vcz, ancestral="AA")
