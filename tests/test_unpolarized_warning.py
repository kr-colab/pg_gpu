"""The ``polarized`` flag and UnpolarizedDataWarning.

pg_gpu reads allele code 0 as the ancestral allele. A matrix carries
``polarized``: whether that is true of its codes. A statistic of a derived
state warns on a matrix that is not marked.

Which statistics those are is decided here by experiment, not by a list:
exchange codes 0 and 1 at half of the sites of a biallelic matrix, and a
statistic must warn exactly when its value changes. A second experiment on
multiallelic data pins the statistics that depend on code 0 only there,
which do not warn.
"""

import tomllib
import warnings
from pathlib import Path

import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import (GenotypeMatrix, HaplotypeMatrix, UnpolarizedDataWarning,
                    decomposition, divergence, diversity, relatedness,
                    selection, sfs, windowed_analysis)
from pg_gpu._gpu_genotype_prep import swap_ancestral
from pg_gpu._utils import get_population_matrix
from pg_gpu._warnings import _polarization_dependent, _warn_unpolarized
from pg_gpu.zarr_io import POLARIZED_ATTR, allel_zarr_to_vcz

from .conftest import bgzip_index

MAP = tomllib.loads(
    (Path(__file__).parents[1] / "docs/dev/pipeline_map.toml").read_text())
SETS = {"a": list(range(0, 20)), "b": list(range(20, 40))}


def _ts(seed=5, model="binary", rate=1e-8):
    ts = msprime.sim_ancestry(20, sequence_length=2e5, recombination_rate=1e-8,
                              population_size=1e4, random_seed=seed)
    return msprime.sim_mutations(ts, rate=rate, model=model, random_seed=seed)


def _matrix(haps, like, polarized):
    hm = HaplotypeMatrix(haps, cp.asarray(like.positions), like.chrom_start,
                         like.chrom_end, phased=True, polarized=polarized,
                         samples=[f"s{i}" for i in range(haps.shape[0] // 2)])
    hm.sample_sets = SETS
    return hm


@pytest.fixture(scope="module")
def pair():
    """The same data twice: as simulated, and with codes 0 and 1 exchanged
    at a random half of the sites. Neither is marked polarized."""
    base = HaplotypeMatrix.from_ts(_ts())
    haps = cp.ascontiguousarray(cp.asarray(base.haplotypes).astype(cp.int8))
    codes = np.random.default_rng(1).integers(0, 2, haps.shape[1]).astype(np.int8)
    return (_matrix(haps, base, False),
            _matrix(swap_ancestral(haps.copy(), codes), base, False))


@pytest.fixture
def hm(pair):
    return _matrix(cp.asarray(pair[0].haplotypes), pair[0], False)


@pytest.fixture
def polarized(pair):
    return _matrix(cp.asarray(pair[0].haplotypes), pair[0], True)


def _host(x):
    if isinstance(x, (tuple, list)):
        return [_host(v) for v in x]
    if isinstance(x, dict):
        return {k: _host(v) for k, v in x.items()}
    if hasattr(x, "columns"):
        return {c: x[c].to_numpy() for c in x.columns}
    return x.get() if hasattr(x, "get") else x


def _same(x, y):
    if isinstance(x, list):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        return str(x) == str(y)
    return x.shape == y.shape and np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)


def _warned(call, matrix):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(matrix)
    return result, [w for w in caught if issubclass(w.category, UnpolarizedDataWarning)]


def _changes_and_warnings(call, pair):
    a, b = pair
    value_a, hits = _warned(call, a)
    value_b, _ = _warned(call, b)
    return not _same(_host(value_a), _host(value_b)), len(hits)


def _assert_warns_iff_value_changes(call, pair):
    changes, n_warnings = _changes_and_warnings(call, pair)
    assert n_warnings == (1 if changes else 0), (
        f"value changes under a swap: {changes}; warnings: {n_warnings}")


# ── every statistic, by experiment ────────────────────────────────────
SCALARS = {
    "pi": lambda m: diversity.pi(m),
    "theta_w": lambda m: diversity.theta_w(m),
    "tajimas_d": lambda m: diversity.tajimas_d(m),
    "segregating_sites": lambda m: diversity.segregating_sites(m),
    "singleton_count": lambda m: diversity.singleton_count(m),
    "theta_h": lambda m: diversity.theta_h(m),
    "theta_l": lambda m: diversity.theta_l(m),
    "fay_wus_h": lambda m: diversity.fay_wus_h(m),
    "normalized_fay_wus_h": lambda m: diversity.normalized_fay_wus_h(m),
    "zeng_e": lambda m: diversity.zeng_e(m),
    "zeng_dh": lambda m: diversity.zeng_dh(m),
    "max_daf": lambda m: diversity.max_daf(m),
    "daf_histogram": lambda m: diversity.daf_histogram(m),
    "neutrality_tests": lambda m: diversity.neutrality_tests(m),
    "heterozygosity_expected": lambda m: diversity.heterozygosity_expected(m),
    "haplotype_diversity": lambda m: diversity.haplotype_diversity(m),
    "mu_sfs": lambda m: diversity.mu_sfs(m),
    "sfs": lambda m: sfs.sfs(m),
    "sfs_scaled": lambda m: sfs.sfs_scaled(m),
    "sfs_folded": lambda m: sfs.sfs_folded(m),
    "sfs_folded_scaled": lambda m: sfs.sfs_folded_scaled(m),
    "joint_sfs": lambda m: sfs.joint_sfs(m, "a", "b"),
    "joint_sfs_scaled": lambda m: sfs.joint_sfs_scaled(m, "a", "b"),
    "joint_sfs_folded": lambda m: sfs.joint_sfs_folded(m, "a", "b"),
    "joint_sfs_folded_scaled": lambda m: sfs.joint_sfs_folded_scaled(m, "a", "b"),
    "project_joint_sfs": lambda m: sfs.project_joint_sfs(m, "a", "b", 6, 6),
    "ihs": lambda m: selection.ihs(m),
    "nsl": lambda m: selection.nsl(m),
    "xpehh": lambda m: selection.xpehh(m, "a", "b"),
    "xpnsl": lambda m: selection.xpnsl(m, "a", "b"),
    "garud_h": lambda m: selection.garud_h(m),
    "fst_hudson": lambda m: divergence.fst_hudson(m, "a", "b"),
    "dxy": lambda m: divergence.dxy(m, "a", "b"),
    "pca": lambda m: np.abs(_host(decomposition.pca(m))[0]),
    "genetic_relatedness": lambda m: relatedness.genetic_relatedness(m),
    "genetic_relatedness_polarised_uncentred":
        lambda m: relatedness.genetic_relatedness(m, polarised=True, centre=False),
    "genetic_relatedness_uncentred": lambda m: relatedness.genetic_relatedness(m, centre=False),
}


@pytest.mark.parametrize("name", sorted(SCALARS))
def test_scalar_warns_iff_value_changes(pair, name):
    _assert_warns_iff_value_changes(SCALARS[name], pair)


def test_centred_polarised_relatedness_warns_for_its_multiallelic_sites(pair, multiallelic_pair):
    # Centring makes a biallelic site the same either way, so the value
    # changes only on multiallelic data; the caller asked for the
    # polarised form, so it warns on both.
    call = lambda m: relatedness.genetic_relatedness(m, polarised=True)  # noqa: E731
    assert _changes_and_warnings(call, pair) == (False, 1)
    assert _changes_and_warnings(call, multiallelic_pair) == (True, 1)


@pytest.mark.parametrize("name", sorted(diversity._DIVERSITY_STATS_NAMES))
def test_diversity_stats_name_warns_iff_value_changes(pair, name):
    _assert_warns_iff_value_changes(
        lambda m: diversity.diversity_stats(m, statistics=[name]), pair)


@pytest.mark.parametrize("weight", sorted(diversity.WEIGHT_REGISTRY))
def test_frequency_spectrum_weight_warns_iff_value_changes(pair, weight):
    _assert_warns_iff_value_changes(
        lambda m: diversity.FrequencySpectrum(m).theta(weight), pair)


@pytest.mark.parametrize("method", ["sfs", "all_thetas", "all_tests", "fay_wu_h",
                                    "zeng_e", "tajimas_d"])
def test_frequency_spectrum_method_warns_iff_value_changes(pair, method):
    _assert_warns_iff_value_changes(
        lambda m: getattr(diversity.FrequencySpectrum(m), method)(), pair)


# Statistics that need arguments the generic call does not give.
_WINDOWED_NEEDS_ARGUMENTS = {"ld_decay", "mean_r2"}
WINDOWED = sorted(set(MAP["windowed"]) - _WINDOWED_NEEDS_ARGUMENTS)


def _windowed(m, stat):
    kw = dict(window_size=20_000, statistics=[stat], progress_bar=False)
    try:
        return windowed_analysis(m, **kw)
    except (ValueError, TypeError):
        return windowed_analysis(m, populations=["a", "b"], **kw)


@pytest.mark.parametrize("stat", WINDOWED)
def test_windowed_name_warns_iff_value_changes(pair, stat):
    _assert_warns_iff_value_changes(lambda m: _windowed(m, stat), pair)


# ── multiallelic sites ────────────────────────────────────────────────
# Statistics that count each derived allele of a site apart, and so change
# with the choice of code 0 at a site with three or more alleles, although
# they are symmetric at a biallelic site. They do not warn: on data that
# are mostly biallelic the warning would be a false alarm for nearly every
# user. The docs name them.
MULTIALLELIC_ONLY = {
    "scalar mu_sfs", "windowed mu_sfs",
    "diversity_stats eta1_star", "diversity_stats minus_eta1_star",
}
# Its windows differ on the multiallelic pair by more than the tolerance
# of this test; whether that is a dependence on code 0 is not established.
_NOT_IN_THE_MULTIALLELIC_EXPERIMENT = {"local_pca_jackknife"}


@pytest.fixture(scope="module")
def multiallelic_pair():
    """As ``pair``, on data with sites of three and more alleles: code 0
    changes places with a random allele of each site."""
    base = HaplotypeMatrix.from_ts(_ts(model="jc69", rate=6e-8))
    haps = cp.ascontiguousarray(cp.asarray(base.haplotypes).astype(cp.int8))
    n_alleles = _host(haps.max(axis=0)) + 1
    assert (n_alleles > 2).any()
    rng = np.random.default_rng(1)
    codes = np.array([rng.integers(0, k) for k in n_alleles], dtype=np.int8)
    return (_matrix(haps, base, False),
            _matrix(swap_ancestral(haps.copy(), codes), base, False))


def test_silent_dependence_on_code_zero_is_the_known_multiallelic_set(multiallelic_pair):
    calls = {f"scalar {n}": f for n, f in SCALARS.items()}
    calls |= {f"diversity_stats {n}": (lambda m, n=n: diversity.diversity_stats(m, statistics=[n]))
              for n in diversity._DIVERSITY_STATS_NAMES}
    calls |= {f"FrequencySpectrum.theta {w}": (lambda m, w=w: diversity.FrequencySpectrum(m).theta(w))
              for w in diversity.WEIGHT_REGISTRY}
    calls |= {f"FrequencySpectrum.{k}": (lambda m, k=k: getattr(diversity.FrequencySpectrum(m), k)())
              for k in ("sfs", "all_thetas", "all_tests", "tajimas_d")}
    calls |= {f"windowed {s}": (lambda m, s=s: _windowed(m, s))
              for s in WINDOWED if s not in _NOT_IN_THE_MULTIALLELIC_EXPERIMENT}
    silent = set()
    for label, call in calls.items():
        changes, n_warnings = _changes_and_warnings(call, multiallelic_pair)
        if changes and not n_warnings:
            silent.add(label)
    assert silent == MULTIALLELIC_ONLY


# ── a marked matrix is silent ─────────────────────────────────────────
@pytest.mark.parametrize("name", ["theta_h", "sfs", "joint_sfs", "ihs", "max_daf",
                                  "daf_histogram", "singleton_count",
                                  "genetic_relatedness_polarised_uncentred"])
def test_silent_on_a_polarized_matrix(polarized, name):
    assert _warned(SCALARS[name], polarized)[1] == []


def test_windowed_silent_on_a_polarized_matrix(polarized):
    assert _warned(lambda m: _windowed(m, "fay_wu_h"), polarized)[1] == []


def test_genotype_matrix_is_silent(hm):
    gm = GenotypeMatrix.from_haplotype_matrix(hm)
    assert _warned(lambda m: diversity.daf_histogram(m), gm)[1] == []


# ── one user call, one warning ────────────────────────────────────────
def test_message_carries_filter_phrase_and_class_name(hm):
    # pyproject.toml's filter keys on this phrase.
    @_polarization_dependent
    def f(m):
        _warn_unpolarized(m, context="f")
    _, hits = _warned(f, hm)
    msg = str(hits[0].message)
    assert "reads allele code 0 as the ancestral allele" in msg
    assert "this HaplotypeMatrix" in msg and ".polarize()" in msg


def test_nested_calls_warn_once(hm):
    # zeng_dh calls two other statistics of the derived state.
    assert len(_warned(lambda m: diversity.zeng_dh(m), hm)[1]) == 1
    assert len(_warned(lambda m: diversity.neutrality_tests(m), hm)[1]) == 1


def test_windowed_fallback_warns_once(hm):
    # The per-window fallback calls the scalar function for each window.
    call = lambda m: windowed_analysis(  # noqa: E731
        m, window_size=20_000, statistics=["theta_h", "haplotype_count"],
        progress_bar=False)
    assert len(_warned(call, hm)[1]) == 1


@pytest.mark.parametrize("entry", ["compute", "compute_region", "compute_streaming"])
def test_windowed_analyzer_warns_once(hm, entry):
    from pg_gpu import WindowedAnalyzer
    analyzer = WindowedAnalyzer(window_size=20_000, statistics=["theta_h", "pi"],
                                progress_bar=False)
    call = {"compute": lambda m: analyzer.compute(m),
            "compute_region": lambda m: analyzer.compute_region(m, "1", 0, 100_000),
            "compute_streaming": lambda m: list(analyzer.compute_streaming(m))}[entry]
    assert len(_warned(call, hm)[1]) == 1


@pytest.mark.parametrize("engine", ["windowed_statistics_fused",
                                    "windowed_statistics_fused_chunked"])
@pytest.mark.parametrize("statistics,n_warnings", [(("theta_h", "fay_wu_h"), 1),
                                                   (("pi", "tajimas_d"), 0)])
def test_direct_engine_call_warns_by_the_names_asked_for(hm, engine, statistics, n_warnings):
    # The kernel sums theta_h for every request, so the names decide.
    import importlib
    wa = importlib.import_module("pg_gpu.windowed_analysis")
    call = lambda m: getattr(wa, engine)(  # noqa: E731
        m, bp_bins=np.arange(0, 200_001, 20_000), statistics=statistics)
    assert len(_warned(call, hm)[1]) == n_warnings


def test_custom_statistic_warns_as_a_direct_call_does(hm):
    # A caller's own function runs outside the guard of the scan, so the
    # statistic it calls is not silenced by it.
    def my_theta_h(window):
        return float(diversity.theta_h(window.matrix))
    call = lambda m: windowed_analysis(  # noqa: E731
        m, window_size=200_000, statistics=["pi", my_theta_h], progress_bar=False)
    _, hits = _warned(call, hm)
    assert len(hits) == 1 and "theta_h reads" in str(hits[0].message)


def test_frequency_spectrum_warns_once_per_object(hm):
    def call(m):
        fs = diversity.FrequencySpectrum(m)
        fs.all_thetas()
        fs.all_tests()
        fs.sfs()
    assert len(_warned(call, hm)[1]) == 1


def test_silenceable_by_category(hm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=UnpolarizedDataWarning)
        diversity.theta_h(hm)


# ── the flag ──────────────────────────────────────────────────────────
class TestFlag:

    def test_default_is_false(self):
        hm = HaplotypeMatrix(np.zeros((2, 3), dtype=np.int8) + [0, 1, 0], np.arange(3), 0, 3)
        assert hm.polarized is False and "polarized=False" in repr(hm)

    def test_setters(self, hm):
        assert hm.set_polarized() is hm and hm.polarized is True
        assert hm.set_unpolarized() is hm and hm.polarized is False

    def test_set_polarized_changes_no_code(self, hm):
        before = _host(hm.haplotypes).copy()
        np.testing.assert_array_equal(_host(hm.set_polarized().haplotypes), before)

    def test_from_ts_is_polarized(self):
        assert HaplotypeMatrix.from_ts(_ts()).polarized is True

    @pytest.mark.parametrize("derive", [
        lambda m: m.get_subset(np.array([0, 3, 5])),
        lambda m: m.get_subset(np.array([], dtype=np.int64)),
        lambda m: m.get_subset_from_range(0, 50_000),
        lambda m: m.restrict_to_biallelic(),
        lambda m: m.restrict_to_segregating(),
        lambda m: m.filter(variants=np.arange(m.num_variants) % 2 == 0),
        lambda m: get_population_matrix(m, "a"),
        lambda m: get_population_matrix(m, [0, 2, 4]),
    ], ids=["get_subset", "get_subset_empty", "range", "biallelic", "segregating",
            "filter", "population", "unpaired_population"])
    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.filterwarnings("ignore::pg_gpu.UnpairedRowsWarning")
    def test_derived_matrices_keep_it(self, hm, derive, flag):
        if flag:
            hm.set_polarized()
        assert derive(hm).polarized is flag

    def test_genotype_round_trip_is_not_polarized(self, polarized):
        back = GenotypeMatrix.from_haplotype_matrix(polarized).to_haplotype_matrix()
        assert back.polarized is False

    def test_polarize_marks_the_result(self, hm):
        out = hm.polarize(np.zeros(hm.num_variants, dtype=np.int8))
        assert out.polarized is True and hm.polarized is False

    def test_polarize_twice_raises(self, hm):
        codes = np.ones(hm.num_variants, dtype=np.int8)
        once = hm.polarize(codes)
        with pytest.raises(ValueError, match="marked polarized"):
            once.polarize(codes)
        # The mark can be taken off on purpose.
        np.testing.assert_array_equal(
            _host(once.set_unpolarized().polarize(codes).haplotypes), _host(hm.haplotypes))

    def test_polarize_again_with_code_zero_everywhere_is_allowed(self, polarized):
        out = polarized.polarize(np.zeros(polarized.num_variants, dtype=np.int8))
        np.testing.assert_array_equal(_host(out.haplotypes), _host(polarized.haplotypes))


# ── stores ────────────────────────────────────────────────────────────
class TestStore:

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.parametrize("format,streaming", [
        ("vcz", "never"), ("vcz", "always"), ("scikit-allel", "never")])
    def test_round_trip(self, hm, tmp_path, format, streaming, flag):
        if flag:
            hm.set_polarized()
        path = str(tmp_path / "store.zarr")
        hm.to_zarr(path, format=format, contig_name="1")
        assert HaplotypeMatrix.from_zarr(path, streaming=streaming).polarized is flag

    @pytest.mark.parametrize("streaming", ["never", "always"])
    def test_explicit_argument_wins(self, polarized, tmp_path, streaming):
        path = str(tmp_path / "store.vcz")
        polarized.to_zarr(path, contig_name="1")
        assert HaplotypeMatrix.from_zarr(path, streaming=streaming,
                                         polarized=False).polarized is False

    def test_store_with_no_record_is_not_polarized(self, polarized, tmp_path):
        import zarr
        path = str(tmp_path / "store.vcz")
        polarized.to_zarr(path, contig_name="1")
        del zarr.open_group(path, mode="r+").attrs[POLARIZED_ATTR]
        for streaming in ("never", "always"):
            assert HaplotypeMatrix.from_zarr(path, streaming=streaming).polarized is False

    def test_layout_conversion_keeps_the_record(self, polarized, tmp_path):
        allel = str(tmp_path / "allel.zarr")
        polarized.to_zarr(allel, format="scikit-allel")
        vcz = str(tmp_path / "converted.vcz")
        allel_zarr_to_vcz(allel, vcz, contig="1")
        assert HaplotypeMatrix.from_zarr(vcz).polarized is True

    def test_stream_flag_reaches_chunks_and_setters_work(self, polarized, tmp_path):
        path = str(tmp_path / "store.vcz")
        polarized.to_zarr(path, contig_name="1")
        stream = HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=50_000)
        assert stream.polarized is True and "polarized=True" in repr(stream)
        assert all(m.polarized for _, _, m in stream.iter_gpu_chunks())
        assert stream.materialize().polarized is True
        assert stream.set_unpolarized() is stream
        assert not any(m.polarized for _, _, m in stream.iter_gpu_chunks())
        assert stream.set_polarized().polarized is True

    def test_stream_warns_once_and_names_no_polarize_method(self, hm, tmp_path):
        path = str(tmp_path / "store.vcz")
        hm.to_zarr(path, contig_name="1")
        stream = HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=40_000)
        assert len(stream._chunks) > 1
        for call in (lambda m: sfs.sfs(m),
                     lambda m: windowed_analysis(m, window_size=20_000,
                                                 statistics=["theta_h"])):
            _, hits = _warned(call, stream)
            assert len(hits) == 1
            msg = str(hits[0].message)
            assert "this StreamingHaplotypeMatrix" in msg and ".polarize()" not in msg


# ── loads with an ancestral tag ───────────────────────────────────────
_HEADER = (
    "##fileformat=VCFv4.2\n##contig=<ID=1,length=1000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n")


@pytest.fixture(scope="module")
def tagged(tmp_path_factory):
    """A VCF and its VCZ store with an AA tag: REF at two sites, ALT at one."""
    d = tmp_path_factory.mktemp("tagged")
    vcf = str(d / "t.vcf")
    with open(vcf, "w") as f:
        f.write(_HEADER)
        for pos, aa in ((100, "A"), (200, "T"), (300, "A")):
            f.write(f"1\t{pos}\t.\tA\tT\t.\t.\tAA={aa}\tGT\t0|1\t1|1\n")
    vcf = bgzip_index(vcf)
    vcz = str(d / "t.vcz")
    HaplotypeMatrix.vcf_to_zarr(vcf, vcz, show_progress=False, worker_processes=1,
                                zarr_format=3)
    return vcf, vcz


class TestAncestralLoads:

    def test_plain_loads_are_not_polarized(self, tagged):
        vcf, vcz = tagged
        assert HaplotypeMatrix.from_vcf(vcf).polarized is False
        assert HaplotypeMatrix.from_zarr(vcz).polarized is False
        assert HaplotypeMatrix.from_zarr(vcz, streaming="always").polarized is False

    def test_ancestral_loads_are_polarized(self, tagged):
        vcf, vcz = tagged
        assert HaplotypeMatrix.from_vcf(vcf, ancestral="AA").polarized is True
        assert HaplotypeMatrix.from_zarr(vcz, ancestral="AA").polarized is True
        stream = HaplotypeMatrix.from_zarr(vcz, ancestral="AA", streaming="always")
        assert stream.polarized is True and stream.materialize().polarized is True
        # The stream owns the flag of its chunks, with a recode too.
        assert stream.set_unpolarized().materialize().polarized is False

    def test_declared_polarized(self, tagged):
        vcf, vcz = tagged
        assert HaplotypeMatrix.from_vcf(vcf, polarized=True).polarized is True
        assert HaplotypeMatrix.from_zarr(vcz, polarized=True).polarized is True
        assert HaplotypeMatrix.from_zarr(vcz, polarized=True,
                                         streaming="always").polarized is True

    @pytest.mark.parametrize("streaming", ["never", "always"])
    def test_tag_on_data_declared_polarized_raises(self, tagged, streaming):
        # The tag names ALT at one site, so the declaration and the tag
        # disagree; a recode would move the allele that was declared
        # ancestral.
        with pytest.raises(ValueError, match="marked polarized"):
            HaplotypeMatrix.from_zarr(tagged[1], ancestral="AA", polarized=True,
                                      streaming=streaming)

    def test_tag_on_vcf_declared_polarized_raises(self, tagged):
        with pytest.raises(ValueError, match="marked polarized"):
            HaplotypeMatrix.from_vcf(tagged[0], ancestral="AA", polarized=True)

    def test_polarized_load_is_silent(self, tagged):
        loaded = HaplotypeMatrix.from_zarr(tagged[1], ancestral="AA")
        assert _warned(lambda m: sfs.sfs(m), loaded)[1] == []
