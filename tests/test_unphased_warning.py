"""Tests for UnphasedDataWarning: the emit helper, the phase-depth guard, and
every call site that raises it.

Every warned case checks that the warning is emitted exactly once and that it
points at the line that made the call: same file and same line. The file check
catches a public function that reaches a warned function without being
decorated itself (the inner warning would point inside pg_gpu); the line check
catches a stacklevel that is too large, which would land on a helper in this
file.
"""
import importlib
import warnings

import cupy as cp
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, UnphasedDataWarning
from pg_gpu import (decomposition, distance_stats, divergence, diversity,
                    ld_statistics, moments_ld, relatedness, selection, sfs)
from pg_gpu._warnings import (_phase_dependent, _phase_depth, _splits_individual,
                              _warn_unphased)

from .conftest import simulate_hm

# pg_gpu.windowed_analysis names the function; the module is needed here.
wa = importlib.import_module("pg_gpu.windowed_analysis")

POPS = ['pop1', 'pop2']
BP_BINS = [0, 1_000, 5_000, 20_000]
WIN = 5_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _hm(phased=False, seed=7):
    """Biallelic msprime matrix: 10 diploids, two whole-individual pops."""
    hm = simulate_hm(n_samples=10, seq_length=20_000, seed=seed,
                     mutation_model='binary')
    hm.sample_sets = {'pop1': list(range(10)), 'pop2': list(range(10, 20))}
    return hm.set_phased() if phased else hm.set_unphased()


@pytest.fixture
def hm():
    return _hm(phased=False)


@pytest.fixture
def phased():
    return _hm(phased=True)


@pytest.fixture
def gm(hm):
    return GenotypeMatrix.from_haplotype_matrix(hm)


@pytest.fixture
def stream(tmp_path, hm):
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "unphased.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    s = HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=10_000)
    assert s.phased is False
    s.sample_sets = hm.sample_sets
    return s


@pytest.fixture
def vcf_and_pops(tmp_path, hm):
    """A VCF + pop file built from ``hm``, for the moments_ld VCF path."""
    hap = cp.asnumpy(hm.haplotypes)
    pos = cp.asnumpy(hm.positions)
    n_ind = hap.shape[0] // 2
    names = [f"ind{i}" for i in range(n_ind)]
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=1>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
             + "\t".join(names)]
    for j, p in enumerate(pos):
        gts = [f"{hap[2 * i, j]}|{hap[2 * i + 1, j]}" for i in range(n_ind)]
        lines.append(f"1\t{int(p) + 1}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    vcf = tmp_path / "x.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    pop_file = tmp_path / "pops.txt"
    pop_file.write_text("sample\tpop\n" + "".join(
        f"{n}\t{'pop1' if i < n_ind // 2 else 'pop2'}\n" for i, n in enumerate(names)))
    return str(vcf), str(pop_file)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _caught(block):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block()
    return [w for w in caught if issubclass(w.category, UnphasedDataWarning)]


def _warns_once(block, context, origin=None):
    """``origin`` is the lambda whose body makes the warned call; it defaults
    to ``block``. The warning must point at that lambda's (first) line."""
    got = _caught(block)
    assert len(got) == 1, [str(w.message) for w in got]
    msg = str(got[0].message)
    assert msg.startswith(context), msg
    line = (origin or block).__code__.co_firstlineno
    assert (got[0].filename, got[0].lineno) == (__file__, line), (
        got[0].filename, got[0].lineno, line, msg)
    return msg


def _silent(block):
    got = _caught(block)
    assert not got, [str(w.message) for w in got]


# ---------------------------------------------------------------------------
# Helper and guard
# ---------------------------------------------------------------------------

def test_is_userwarning_subclass():
    assert issubclass(UnphasedDataWarning, UserWarning)


def test_message_carries_filter_phrase_and_class_name(hm, stream):
    # pyproject.toml's filter keys on this phrase.
    @_phase_dependent
    def f(m):
        _warn_unphased(m, context="f", alternative="use g")
    msg = _warns_once(lambda: f(hm), "f")
    assert "depends on haplotype phase" in msg
    assert "this HaplotypeMatrix" in msg and "Use g." in msg
    assert "this StreamingHaplotypeMatrix" in _warns_once(lambda: f(stream), "f")


def test_helper_silent_for_phased_gm_and_array(phased, gm):
    @_phase_dependent
    def f(m):
        _warn_unphased(m, context="f")
    for m in (phased, gm, cp.zeros((3, 3))):
        _silent(lambda: f(m))


def test_silenceable_by_category(hm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=UnphasedDataWarning)
        diversity.haplotype_count(hm)


def test_depth_resets_after_return_and_raise():
    @_phase_dependent
    def ok():
        return 1

    @_phase_dependent
    def boom():
        raise ValueError

    ok()
    assert _phase_depth.get() == 0
    with pytest.raises(ValueError):
        boom()
    assert _phase_depth.get() == 0


def test_nested_decorated_warns_once_from_outer(hm):
    @_phase_dependent
    def inner(m):
        _warn_unphased(m, context="inner")

    @_phase_dependent
    def outer(m):
        _warn_unphased(m, context="outer")
        inner(m)

    _warns_once(lambda: outer(hm), "outer")


@pytest.mark.parametrize("rows, splits", [
    ([0, 1], False), ([0, 1, 2, 3], False), ([0, 2, 1, 3], False),
    ([0, 2], True), ([1, 4], True), ([3], True),
])
def test_splits_individual(rows, splits):
    assert _splits_individual(rows) is splits
    assert _splits_individual(np.array(rows)) is splits
    assert _splits_individual(cp.array(rows)) is splits


# ---------------------------------------------------------------------------
# Positive: every warned function, once, pointing at this file
# ---------------------------------------------------------------------------

_P1, _P2 = 'pop1', 'pop2'

WARNED = {
    # selection
    "garud_h": lambda m: selection.garud_h(m),
    "moving_garud_h": lambda m: selection.moving_garud_h(m, size=10),
    "nsl": lambda m: selection.nsl(m),
    "xpnsl": lambda m: selection.xpnsl(m, _P1, _P2),
    "ihs": lambda m: selection.ihs(m),
    "xpehh": lambda m: selection.xpehh(m, _P1, _P2),
    "ehh_decay": lambda m: selection.ehh_decay(m),
    # diversity
    "haplotype_diversity": lambda m: diversity.haplotype_diversity(m),
    "haplotype_count": lambda m: diversity.haplotype_count(m),
    "diversity_stats": lambda m: diversity.diversity_stats(
        m, statistics=['pi', 'haplotype_diversity']),
    # divergence
    "pairwise_distance_matrix": lambda m: divergence.pairwise_distance_matrix(m, _P1, _P2),
    "snn": lambda m: divergence.snn(m, _P1, _P2),
    "dxy_min": lambda m: divergence.dxy_min(m, _P1, _P2),
    "gmin": lambda m: divergence.gmin(m, _P1, _P2),
    "dd": lambda m: divergence.dd(m, _P1, _P2),
    "dd_rank": lambda m: divergence.dd_rank(m, _P1, _P2),
    "zx": lambda m: divergence.zx(m, _P1, _P2),
    "distance_based_stats": lambda m: divergence.distance_based_stats(m, _P1, _P2),
    # distance_stats
    "pairwise_diffs_haploid": lambda m: distance_stats.pairwise_diffs_haploid(m),
    "pairwise_diffs": lambda m: distance_stats.pairwise_diffs(m),
    "dist_moments": lambda m: distance_stats.dist_moments(m),
    "dist_var": lambda m: distance_stats.dist_var(m),
    "dist_skew": lambda m: distance_stats.dist_skew(m),
    "dist_kurt": lambda m: distance_stats.dist_kurt(m),
    # ld
    "zns": lambda m: ld_statistics.zns(m, estimator='r2'),
    "zns(auto)": lambda m: ld_statistics.zns(m),
    "omega": lambda m: ld_statistics.omega(m, estimator='r2'),
    "omega(auto)": lambda m: ld_statistics.omega(m),
    "mu_ld": lambda m: ld_statistics.mu_ld(m),
    "compute_ld_statistics": lambda m: moments_ld.compute_ld_statistics(
        haplotype_matrix=m, pops=POPS, bp_bins=BP_BINS,
        use_genotypes=False, report=False),
    # decomposition
    "pca": lambda m: decomposition.pca(m, n_components=2),
    "randomized_pca": lambda m: decomposition.randomized_pca(
        m, n_components=2, random_state=0),
    "pairwise_distance": lambda m: decomposition.pairwise_distance(m),
    "local_pca": lambda m: decomposition.local_pca(m, k=2, window_size=50),
    "local_pca_jackknife": lambda m: decomposition.local_pca_jackknife(
        m, k=2, window_size=50, n_blocks=3),
    "lostruct": lambda m: decomposition.lostruct(m, k=2, window_size=50),
    "lostruct(jackknife)": lambda m: decomposition.lostruct(
        m, k=2, window_size=50, jackknife=True, n_blocks=3),
    # relatedness
    "genetic_relatedness": lambda m: relatedness.genetic_relatedness(m),
    # HaplotypeMatrix methods
    "pairwise_LD_v": lambda m: m.pairwise_LD_v(),
    "pairwise_r2": lambda m: m.pairwise_r2(),
    "locate_unlinked": lambda m: m.locate_unlinked(),
    "windowed_r_squared": lambda m: m.windowed_r_squared(BP_BINS),
    "tally_gpu_haplotypes": lambda m: m.tally_gpu_haplotypes(),
    "tally_gpu_haplotypes_two_pops": lambda m: m.tally_gpu_haplotypes_two_pops(_P1, _P2),
    "tally_gpu_haplotypes_two_pops_with_missing":
        lambda m: m.tally_gpu_haplotypes_two_pops_with_missing(_P1, _P2),
    "compute_ld_statistics_gpu_single_pop":
        lambda m: m.compute_ld_statistics_gpu_single_pop(BP_BINS),
    "compute_ld_statistics_gpu_two_pops":
        lambda m: m.compute_ld_statistics_gpu_two_pops(BP_BINS, _P1, _P2),
}


@pytest.mark.parametrize("name", sorted(WARNED))
def test_warns_once_on_unphased(hm, name):
    _warns_once(lambda: WARNED[name](hm), name.split("(")[0], origin=WARNED[name])


@pytest.mark.parametrize("name", sorted(WARNED))
def test_silent_on_phased(phased, name):
    _silent(lambda: WARNED[name](phased))


# ---------------------------------------------------------------------------
# Negative: unphased-safe functions and branches
# ---------------------------------------------------------------------------

SILENT = {
    "pi": lambda m: diversity.pi(m),
    "fst_hudson": lambda m: divergence.fst_hudson(m, _P1, _P2),
    "fst_weir_cockerham": lambda m: divergence.fst_weir_cockerham(m, _P1, _P2),
    "sfs": lambda m: sfs.sfs(m),
    "heterozygosity_observed": lambda m: diversity.heterozygosity_observed(m),
    "diversity_stats(pi)": lambda m: diversity.diversity_stats(m, statistics=['pi']),
    "zns(rogers_huff)": lambda m: ld_statistics.zns(m, estimator='rogers_huff'),
    "omega(rogers_huff)": lambda m: ld_statistics.omega(m, estimator='rogers_huff'),
    "pairwise_r2(rogers_huff)": lambda m: m.pairwise_r2(estimator='rogers_huff'),
    "windowed_r_squared(rogers_huff)":
        lambda m: m.windowed_r_squared(BP_BINS, estimator='rogers_huff'),
    "compute_ld_statistics(use_genotypes=True)": lambda m: moments_ld.compute_ld_statistics(
        haplotype_matrix=m, pops=POPS, bp_bins=BP_BINS,
        use_genotypes=True, report=False),
    "genetic_relatedness(whole individuals)":
        lambda m: relatedness.genetic_relatedness(m, sample_sets=[[0, 1], [2, 3]]),
    "genetic_relatedness(shuffled whole individuals)":
        lambda m: relatedness.genetic_relatedness(m, sample_sets=[[0, 2, 1, 3]]),
}


@pytest.mark.parametrize("name", sorted(SILENT))
def test_phase_free_is_silent(hm, name):
    _silent(lambda: SILENT[name](hm))


def test_gm_branches_are_silent(gm):
    _silent(lambda: selection.garud_h(gm))
    _silent(lambda: distance_stats.dist_moments(gm))
    _silent(lambda: distance_stats.pairwise_diffs(gm))
    _silent(lambda: ld_statistics.zns(gm))


def test_precomputed_distance_matrices_are_silent(hm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = divergence.pairwise_distance_matrix(hm, _P1, _P2)
    for f in (divergence.snn, divergence.dxy_min, divergence.gmin,
              divergence.dd, divergence.dd_rank):
        _silent(lambda: f(hm, _P1, _P2, distance_matrices=dm))


def test_moments_ld_vcf_path_is_silent(vcf_and_pops):
    vcf, pop_file = vcf_and_pops
    _silent(lambda: moments_ld.compute_ld_statistics(
        vcf, pop_file=pop_file, pops=POPS, bp_bins=BP_BINS,
        use_genotypes=False, report=False))


@pytest.mark.parametrize("sets, warns", [
    (None, True), ([[0, 2]], True), ([[0, 1], [2, 3]], False),
    ([[0, 2, 1, 3]], False), ([cp.array([0, 1]), cp.array([2, 3])], False),
    ([cp.array([0, 2])], True),
])
def test_genetic_relatedness_sets(hm, stream, sets, warns):
    for m in (hm, stream):
        # A lambda, not a def: _warns_once checks the warning's line against
        # the block's first line, which must be the line that makes the call.
        block = lambda: relatedness.genetic_relatedness(m, sample_sets=sets)  # noqa: E731
        if warns:
            _warns_once(block, "genetic_relatedness")
        else:
            _silent(block)


def test_from_ts_is_silent():
    hm = simulate_hm(n_samples=10, seq_length=20_000, seed=7,
                     mutation_model='binary')
    assert hm.phased is True
    _silent(lambda: selection.garud_h(hm))
    _silent(lambda: decomposition.pca(hm, n_components=2))


# ---------------------------------------------------------------------------
# Streaming LD methods
# ---------------------------------------------------------------------------

def test_streaming_ld_methods_warn_once(stream):
    _warns_once(lambda: stream.compute_ld_statistics_gpu_single_pop(BP_BINS),
                "compute_ld_statistics_gpu_single_pop")
    _warns_once(lambda: stream.compute_ld_statistics_gpu_two_pops(BP_BINS, _P1, _P2),
                "compute_ld_statistics_gpu_two_pops")


# ---------------------------------------------------------------------------
# Windowed scans: one warning per scan, naming the phase-dependent stats
# ---------------------------------------------------------------------------

SCANS = {
    "fused garud": dict(statistics=['garud_h12', 'pi']),
    "fused diploshic": dict(statistics=['omega', 'mu_ld', 'dist_var']),
    "mean_nsl": dict(statistics=['mean_nsl']),
    "fallback mean_r2": dict(statistics=['mean_r2'],
                             custom_stat_kwargs={'mean_r2': {'max_dist': 2_000}}),
    "local_pca": dict(statistics=['local_pca']),
}


@pytest.mark.parametrize("name", sorted(SCANS))
def test_windowed_analysis_warns_once(hm, name):
    kw = SCANS[name]
    msg = _warns_once(lambda: wa.windowed_analysis(hm, window_size=WIN, **kw),
                      "windowed_analysis")
    for s in set(kw['statistics']) & wa._PHASE_DEPENDENT_STATS:
        assert s in msg


def test_windowed_analysis_phase_free_is_silent(hm):
    _silent(lambda: wa.windowed_analysis(
        hm, window_size=WIN, statistics=['pi', 'fst'], populations=POPS))


def test_windowed_analysis_streaming_warns_once(stream):
    _warns_once(lambda: wa.windowed_analysis(
        stream, window_size=WIN, statistics=['garud_h12']), "windowed_analysis")


def test_fused_entry_points_warn_once(hm):
    bins = np.arange(0, 20_001, WIN)
    for f in (wa.windowed_statistics_fused, wa.windowed_statistics_fused_chunked):
        _warns_once(lambda: f(hm, bins, statistics=('garud_h12', 'pi')), f.__name__)
        _silent(lambda: f(hm, bins, statistics=('pi',)))


def test_windowed_analyzer_methods_warn_once(hm):
    an = wa.WindowedAnalyzer(window_size=WIN, statistics=['garud_h12', 'mean_r2'],
                             custom_stat_kwargs={'mean_r2': {'max_dist': 2_000}},
                             progress_bar=False)
    _warns_once(lambda: an.compute(hm), "WindowedAnalyzer.compute")
    _warns_once(lambda: an.compute_region(hm, "1", 0, 10_000),
                "WindowedAnalyzer.compute_region")
    # The warning fires at the call; iterating stays silent although each
    # window reaches pairwise_r2 after the call has returned.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gen = an.compute_streaming(hm, batch_size=1)
        n_at_call = len(caught)
        list(gen)
        got = [w for w in caught if issubclass(w.category, UnphasedDataWarning)]
    assert n_at_call == 1 and len(got) == 1, [str(w.message) for w in got]
    assert got[0].filename == __file__


def test_custom_callable_stat_warns_from_the_callable(hm, phased):
    # The scan can't see what a callable computes, so the phase-dependent
    # function the callable calls warns itself, pointing at the callable.
    def h12(w):
        return selection.garud_h(w.matrix)[1]

    an = wa.WindowedAnalyzer(window_size=WIN, statistics=[h12], progress_bar=False)
    got = _caught(lambda: an.compute(hm))
    assert got and all(str(w.message).startswith("garud_h") for w in got)
    assert {(w.filename, w.lineno) for w in got} == {
        (__file__, h12.__code__.co_firstlineno + 1)}
    _silent(lambda: an.compute(phased))
    _silent(lambda: wa.WindowedAnalyzer(window_size=WIN, statistics=['pi'],
                                        progress_bar=False).compute(hm))


def test_phase_dependent_stats_pinned():
    # Update phasing.rst and the † markers in features.rst/quickstart.rst
    # when this changes.
    assert wa._PHASE_DEPENDENT_STATS == frozenset({
        'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1', 'haplotype_count',
        'zns', 'omega', 'mu_ld', 'ld_decay', 'mean_r2',
        'dist_var', 'dist_skew', 'dist_kurt', 'mean_nsl',
        'local_pca', 'local_pca_jackknife',
    })


# ---------------------------------------------------------------------------
# Re-phasing: the classification matches behavior
# ---------------------------------------------------------------------------

def _rephase(hm, seed=0):
    """Swap rows 2i and 2i+1 at a random half of heterozygous sites."""
    hap = cp.asnumpy(hm.haplotypes).copy()
    rng = np.random.default_rng(seed)
    a, b = hap[0::2], hap[1::2]
    flip = (a != b) & (rng.random(a.shape) < 0.5)
    a2 = np.where(flip, b, a)
    b2 = np.where(flip, a, b)
    hap[0::2], hap[1::2] = a2, b2
    out = HaplotypeMatrix(hap, cp.asnumpy(hm.positions), hm.chrom_start,
                          hm.chrom_end, phased=True)
    out.sample_sets = hm.sample_sets
    return out


_PHASE_FREE_SCAN = ['pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                    'singletons', 'theta_h', 'fay_wu_h', 'max_daf', 'fst',
                    'fst_hudson', 'fst_wc', 'dxy', 'da', 'mu_var', 'mu_sfs',
                    'snp_dist_mean', 'snp_dist_var', 'snp_dist_min',
                    'snp_dist_max']
_PHASE_SCAN = ['garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1',
               'haplotype_count', 'zns', 'omega', 'mu_ld',
               'dist_var', 'dist_skew', 'dist_kurt', 'mean_nsl']


def _scan(m, stats, **kw):
    return wa.windowed_analysis(m, window_size=2_000, statistics=stats,
                                populations=POPS, **kw)


def _numeric_stat_cols(df):
    skip = set(wa.CANONICAL_WINDOW_PREFIX) | {'window_id'}
    return [c for c in df.columns if c not in skip and not c.startswith('n_complete')
            and np.issubdtype(df[c].dtype, np.number)]


def test_rephasing_matches_classification():
    base = _hm(phased=True, seed=11)
    flipped = _rephase(base)

    a, b = _scan(base, _PHASE_FREE_SCAN), _scan(flipped, _PHASE_FREE_SCAN)
    for c in _numeric_stat_cols(a):
        np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                                   rtol=1e-9, atol=1e-12, equal_nan=True,
                                   err_msg=f"unphased-safe stat {c} changed")

    a, b = _scan(base, _PHASE_SCAN), _scan(flipped, _PHASE_SCAN)
    for s in _PHASE_SCAN:
        cols = [c for c in _numeric_stat_cols(a) if c == s or c.startswith(s + "_pop")]
        assert cols, s
        assert any(not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                                   equal_nan=True) for c in cols), \
            f"phase-dependent stat {s} unchanged by re-phasing"

    kw = dict(custom_stat_kwargs={'mean_r2': {'max_dist': 1_000}})
    a, b = _scan(base, ['mean_r2'], **kw), _scan(flipped, ['mean_r2'], **kw)
    assert any(not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                               equal_nan=True) for c in _numeric_stat_cols(a))

    la = decomposition.local_pca(base, k=2, window_size=50)
    lb = decomposition.local_pca(flipped, k=2, window_size=50)
    assert not np.allclose(la.eigvals, lb.eigvals, equal_nan=True)

    sa = _scan(base, ['local_pca'])
    sb = _scan(flipped, ['local_pca'])
    assert not np.allclose(sa.eigvals, sb.eigvals, equal_nan=True)

    ja = decomposition.local_pca_jackknife(base, k=2, window_size=50, n_blocks=3)
    jb = decomposition.local_pca_jackknife(flipped, k=2, window_size=50, n_blocks=3)
    assert not np.allclose(ja, jb, equal_nan=True)


# ---------------------------------------------------------------------------
# Older warnings still point at the caller through a decorated function
# ---------------------------------------------------------------------------

def _multiallelic(m):
    hap = cp.asnumpy(m.haplotypes).copy()
    # A third allele: at a site with at least two 0s and two 1s, turn one 0
    # into a 2, so 0, 1 and 2 are all present.
    j = int(np.flatnonzero(((hap == 0).sum(0) >= 2) & ((hap == 1).sum(0) >= 2))[0])
    hap[np.flatnonzero(hap[:, j] == 0)[0], j] = 2
    out = HaplotypeMatrix(hap, cp.asnumpy(m.positions), m.chrom_start,
                          m.chrom_end, phased=True)
    out.sample_sets = m.sample_sets
    return out


def _points_here(block, category, origin=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        block()
    got = [w for w in caught if issubclass(w.category, category)]
    assert got, f"no {category.__name__}"
    line = (origin or block).__code__.co_firstlineno
    assert {(w.filename, w.lineno) for w in got} == {(__file__, line)}, \
        [(w.filename, w.lineno, str(w.message)) for w in got]


@pytest.mark.parametrize("call", [
    lambda m: m.pairwise_r2(),
    lambda m: m.pairwise_LD_v(),
    lambda m: m.tally_gpu_haplotypes(),
    lambda m: ld_statistics.zns(m, estimator='r2'),
], ids=["pairwise_r2", "pairwise_LD_v", "tally_gpu_haplotypes", "zns"])
def test_biallelic_warning_points_at_caller(call):
    from pg_gpu import BiallelicOnlyWarning
    m = _multiallelic(_hm(phased=True))
    _points_here(lambda: call(m), BiallelicOnlyWarning, origin=call)


def test_unpaired_rows_warning_points_at_caller():
    from pg_gpu import UnpairedRowsWarning
    m = _hm(phased=True)
    m.sample_sets = {'a': [0, 2, 1, 3], 'b': list(range(10, 20))}
    bins = np.arange(0, 20_001, WIN)
    _points_here(lambda: wa.windowed_statistics_fused_chunked(m, bins, statistics=('fst_wc',), pop1='a', pop2='b'),
                 UnpairedRowsWarning)
