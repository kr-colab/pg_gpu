"""Cross-implementation parity tests.

The same population-genetic statistics are computed by several independent code
paths in pg_gpu:

  1. scalar functions            -- ``diversity.pi``, ``divergence.dxy``, ...
  2. ``FrequencySpectrum``       -- SFS dot-product path
  3. windowed "scatter" engine   -- ``_windowed_thetas_scatter`` / ``_windowed_twopop_scatter``
  4. windowed "fused" engine     -- ``windowed_statistics_fused`` (CUDA kernels)
  5. windowed "python-loop"      -- ``WindowedAnalyzer`` per-window fallback

Nothing else in the suite asserts these agree. This module does. Each statistic
is described once in the ``_STATS`` registry -- which paths compute it, how to
invoke each, and the span-independent "canonical quantity" to compare -- and the
parametrized test drives every supported path across four data conditions
(clean, missing/include, missing/exclude, multiallelic) and asserts it matches
the scalar reference.

Every path is invoked so its return value is the same span-independent
quantity, compared by ``kind``:

  * value -- the statistic as returned (raw per-site sum, ratio, or
             dimensionless test statistic), compared with a tolerance
  * count -- an integer count, compared exactly

Known divergences between paths are recorded in ``_XFAILS`` as ``strict``
xfails with a reason. A ``(stat, path)`` may carry several condition-scoped
rules (e.g. one divergence under missing data and a different one under
multiallelic sites). Each is a defect to fix in a follow-up, at which point
the rule is removed and the cell flips to a pass.
"""

from collections import namedtuple

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, diversity, divergence
from pg_gpu.diversity import FrequencySpectrum
from pg_gpu.windowed_analysis import (
    WindowedAnalyzer,
    _windowed_thetas_scatter,
    _windowed_twopop_scatter,
    windowed_statistics_fused,
    _DAF_N_BINS,
)

from .conftest import simulate_hm

try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - import/driver failure means no GPU
    _GPU = False

pytestmark = pytest.mark.skipif(not _GPU, reason="parity suite requires a CUDA GPU")

POP1, POP2 = "pop1", "pop2"
RTOL, ATOL = 1e-9, 1e-12


def _asnumpy(x):
    """Host numpy array regardless of whether ``x`` is a CuPy or numpy array."""
    return np.asarray(x.get() if hasattr(x, "get") else x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_pop_split(hm):
    # Even haplotype count per population: fst_weir_cockerham pairs consecutive
    # haplotypes into diploid individuals and requires an even count.
    nhap = hm.num_haplotypes
    half = nhap // 2
    hm.sample_sets = {POP1: list(range(half)), POP2: list(range(half, nhap))}
    return hm


@pytest.fixture(scope="module")
def clean_hm():
    """Clean biallelic matrix (no missing) split into two populations.

    ``mutation_model='binary'`` forces biallelic sites, sidestepping the
    multiallelic folding gap so the paths are expected to agree exactly here.
    """
    hm = simulate_hm(n_samples=24, seq_length=200_000, seed=7,
                     mutation_model="binary")
    hm.transfer_to_gpu()
    return _two_pop_split(hm)


@pytest.fixture(scope="module")
def missing_hm():
    """Biallelic matrix with structured missingness.

    Even-indexed sites get ~30% missing entries except one protected
    haplotype per population, so every site keeps at least one valid sample in
    each pop (the per-site dxy/da denominator stays the full variant count
    under ``include``). Odd-indexed sites are fully complete, so ``exclude``
    mode retains a substantial, well-defined subset of sites.
    """
    hm0 = simulate_hm(n_samples=24, seq_length=200_000, seed=7,
                      mutation_model="binary")

    hap = _asnumpy(hm0.haplotypes).astype(np.int8).copy()
    pos = _asnumpy(hm0.positions)

    nhap, nvar = hap.shape
    half = nhap // 2
    protected = {0, half}
    rng = np.random.RandomState(99)
    for v in range(0, nvar, 2):
        for h in range(nhap):
            if h not in protected and rng.random() < 0.3:
                hap[h, v] = -1

    # The per-site dxy/da denominator under 'include' relies on every site
    # keeping at least one valid sample in each pop; make that a loud contract
    # so a future fixture edit that breaks it fails here, not silently.
    assert (hap[:half] >= 0).any(axis=0).all()
    assert (hap[half:] >= 0).any(axis=0).all()

    cs = hm0.chrom_start if hm0.chrom_start is not None else int(pos[0])
    ce = hm0.chrom_end if hm0.chrom_end is not None else int(pos[-1])
    hm = HaplotypeMatrix(hap, pos, cs, ce)
    hm.transfer_to_gpu()
    return _two_pop_split(hm)


@pytest.fixture(scope="module")
def multiallelic_hm():
    """Matrix with multiallelic sites and no missing data.

    msprime's default (Jukes-Cantor) mutation model produces triallelic sites,
    stored with allele codes 0/1/2. Keeping missingness out isolates the
    multiallelic folding behavior (#100) from the missing-data divergences.
    """
    hm = simulate_hm(n_samples=24, seq_length=200_000, seed=7, mutation_model=None)
    hm.transfer_to_gpu()
    return _two_pop_split(hm)


# Data conditions, each declaring the data shape that drives path divergences.
# ``partial_sites`` is True when the condition retains sites of variable
# per-site sample size (the driver of the fs/scatter/fused divergences under
# missing data); ``multiallelic`` is True when sites carry allele codes >= 2.
_Condition = namedtuple("_Condition",
                        ["fixture", "missing_data", "partial_sites", "multiallelic"])
_CONDITIONS = {
    "clean": _Condition("clean_hm", "include", False, False),
    "missing_include": _Condition("missing_hm", "include", True, False),
    "missing_exclude": _Condition("missing_hm", "exclude", False, False),
    "multiallelic": _Condition("multiallelic_hm", "include", False, True),
}
_WHEN_PARTIAL = frozenset(
    name for name, c in _CONDITIONS.items() if c.partial_sites)
_WHEN_MULTIALLELIC = frozenset(
    name for name, c in _CONDITIONS.items() if c.multiallelic)
_WHEN_EXCLUDE = frozenset(
    name for name, c in _CONDITIONS.items() if c.missing_data == "exclude")


# ---------------------------------------------------------------------------
# Whole-region window helpers
# ---------------------------------------------------------------------------

def _region_bounds(hm):
    positions = _asnumpy(hm.positions)
    start = hm.chrom_start if hm.chrom_start is not None else int(positions[0])
    end = hm.chrom_end if hm.chrom_end is not None else int(positions[-1])
    return min(int(start), int(positions[0])), max(int(end), int(positions[-1]))


def _whole_window_size(hm):
    """window_size/step_size yielding a single window covering everything.

    Making both equal to (span + 1) guarantees exactly one window whose stop
    lies past the last variant, in every engine regardless of whether it takes
    its bounds from the matrix's chrom_start/end or from the positions array.
    """
    start, end = _region_bounds(hm)
    return int(end - start) + 1


def _whole_bp_bins(hm):
    """Two bin edges defining one window covering the whole region (for fused)."""
    start, end = _region_bounds(hm)
    return np.array([start, end + 1], dtype=np.float64)


def _one_row(df, name):
    """Pull the single-window value of column ``name`` (ignoring any pop suffix)."""
    if name in df.columns:
        return float(df[name].iloc[0])
    candidates = [c for c in df.columns if c.startswith(name + "_")]
    assert len(candidates) == 1, f"ambiguous column for {name!r}: {candidates}"
    return float(df[candidates[0]].iloc[0])


# ---------------------------------------------------------------------------
# Statistic registry -- one row per stat holds all per-stat knowledge:
#   two_pop  : needs pop1/pop2 rather than a single population
#   kind     : 'value' | 'count' (see module docstring)
#   scalar   : (hm, missing_data) -> float, the reference implementation
#   fs       : (FrequencySpectrum) -> float, or None if unsupported
#   scatter/fused/pyloop : the windowed engine's name for this stat, or None
# ---------------------------------------------------------------------------

_Stat = namedtuple("_Stat", ["two_pop", "kind", "scalar", "fs",
                             "scatter", "fused", "pyloop"])

_STATS = {
    "pi": _Stat(
        False, "value",
        lambda hm, md: diversity.pi(hm, span_normalize=False, missing_data=md),
        lambda fs: fs.theta("pi", span_normalize=False),
        "pi", "pi", "pi"),
    "theta_w": _Stat(
        False, "value",
        lambda hm, md: diversity.theta_w(hm, span_normalize=False, missing_data=md),
        lambda fs: fs.theta("watterson", span_normalize=False),
        "theta_w", "theta_w", "theta_w"),
    "tajimas_d": _Stat(
        False, "value",
        lambda hm, md: diversity.tajimas_d(hm, missing_data=md),
        lambda fs: fs.tajimas_d(),
        "tajimas_d", "tajimas_d", "tajimas_d"),
    "segregating_sites": _Stat(
        False, "count",
        lambda hm, md: diversity.segregating_sites(hm, missing_data=md),
        lambda fs: fs.n_segregating,
        "segregating_sites", "segregating_sites", "segregating_sites"),
    "theta_h": _Stat(
        False, "value",
        lambda hm, md: diversity.theta_h(hm, span_normalize=False, missing_data=md),
        lambda fs: fs.theta("theta_h", span_normalize=False),
        "theta_h", "theta_h", None),
    "theta_l": _Stat(
        False, "value",
        lambda hm, md: diversity.theta_l(hm, span_normalize=False, missing_data=md),
        lambda fs: fs.theta("theta_l", span_normalize=False),
        "theta_l", None, None),
    "fay_wu_h": _Stat(
        False, "value",
        lambda hm, md: diversity.fay_wus_h(hm, missing_data=md),
        lambda fs: fs.fay_wu_h(),
        "fay_wu_h", "fay_wu_h", None),
    "normalized_fay_wu_h": _Stat(
        False, "value",
        lambda hm, md: diversity.normalized_fay_wus_h(hm, missing_data=md),
        lambda fs: fs.fay_wu_h(normalized=True),
        "normalized_fay_wu_h", None, None),
    "zeng_e": _Stat(
        False, "value",
        lambda hm, md: diversity.zeng_e(hm, missing_data=md),
        lambda fs: fs.zeng_e(),
        "zeng_e", None, None),
    "zeng_dh": _Stat(
        False, "value",
        lambda hm, md: diversity.zeng_dh(hm, missing_data=md),
        None, "zeng_dh", None, None),
    "singletons": _Stat(
        False, "count",
        lambda hm, md: diversity.singleton_count(hm, missing_data=md),
        None, "singletons", "singletons", "n_singletons"),
    "max_daf": _Stat(
        False, "value",
        lambda hm, md: diversity.max_daf(hm, missing_data=md),
        None, "max_daf", "max_daf", None),
    "mu_sfs": _Stat(
        False, "value",
        lambda hm, md: diversity.mu_sfs(hm, missing_data=md),
        None, None, "mu_sfs", "mu_sfs"),
    "fst_hudson": _Stat(
        True, "value",
        lambda hm, md: divergence.fst_hudson(hm, POP1, POP2, missing_data=md),
        None, "fst_hudson", "fst_hudson", "fst_hudson"),
    "dxy": _Stat(
        True, "value",
        lambda hm, md: divergence.dxy(hm, POP1, POP2, span_normalize=False, missing_data=md),
        None, "dxy", "dxy", "dxy"),
    "da": _Stat(
        True, "value",
        lambda hm, md: divergence.da(hm, POP1, POP2, span_normalize=False, missing_data=md),
        None, "da", "da", "da"),
    "fst_wc": _Stat(
        True, "value",
        lambda hm, md: divergence.fst_weir_cockerham(hm, POP1, POP2, missing_data=md),
        None, None, "fst_wc", "fst_wc"),
}


# ---------------------------------------------------------------------------
# Per-path adapters -- each returns the canonical quantity for its stat
# ---------------------------------------------------------------------------

def _path_scalar(hm, stat, missing_data):
    return float(_STATS[stat].scalar(hm, missing_data))


def _path_fs(hm, stat, missing_data):
    fs = FrequencySpectrum(hm, missing_data=missing_data)
    return float(_STATS[stat].fs(fs))


def _path_scatter(hm, stat, missing_data):
    spec = _STATS[stat]
    W = _whole_window_size(hm)
    if spec.two_pop:
        df = _windowed_twopop_scatter(hm, W, W, [spec.scatter], [POP1, POP2],
                                      missing_data, False)
    else:
        df = _windowed_thetas_scatter(hm, W, W, [spec.scatter], None,
                                      missing_data, False)
    return _one_row(df, spec.scatter)


def _path_fused(hm, stat, missing_data):
    spec = _STATS[stat]
    bins = _whole_bp_bins(hm)
    pop_kw = {"pop1": POP1, "pop2": POP2} if spec.two_pop else {"population": None}
    out = windowed_statistics_fused(hm, bins, statistics=(spec.fused,),
                                    per_base=False, missing_data=missing_data, **pop_kw)
    return float(np.asarray(out[spec.fused])[0])


def _path_pyloop(hm, stat, missing_data):
    spec = _STATS[stat]
    W = _whole_window_size(hm)
    populations = [POP1, POP2] if spec.two_pop else None
    analyzer = WindowedAnalyzer(
        window_type="bp", window_size=W, step_size=W, statistics=[spec.pyloop],
        populations=populations, missing_data=missing_data, span_normalize=False,
    )
    return _one_row(analyzer.compute(hm), spec.pyloop)


_PATHS = {
    "fs": _path_fs,
    "scatter": _path_scatter,
    "fused": _path_fused,
    "pyloop": _path_pyloop,
}


def _supported_paths(stat):
    """Non-scalar paths that compute ``stat`` (those with a name/callable set)."""
    spec = _STATS[stat]
    return [p for p in _PATHS if getattr(spec, p) is not None]


# ---------------------------------------------------------------------------
# Known divergences and engine limitations -> xfail with a reason
# ---------------------------------------------------------------------------

_FUSED_MISSING = (
    "the fused kernel's per-site sample-size handling under missing data still "
    "diverges from the scalar for this estimator. #135"
)
_FS_VARIANCE = (
    "FrequencySpectrum computes the Achaz neutrality-test variance at a single "
    "modal sample size while summing the numerator over variable per-site "
    "sample sizes, diverging from the scalar under missing data. #135"
)
_SCATTER_VARIANCE = (
    "the scatter neutrality-test variance uses full-sample harmonic numbers "
    "rather than per-site valid counts, diverging from the scalar under "
    "missing data. #135"
)
_FST_WC_FUSED = (
    "the fused fst_wc kernel was not converted to the per-allele one-vs-rest "
    "Weir-Cockerham ANOVA the scalar/python-loop path now uses. #135"
)
_DA_SCATTER_EXCLUDE = (
    "under missing_data='exclude' the scatter da's within-population pi terms "
    "use a different site set than the scalar (dxy agrees), so da diverges. #135"
)
_FS_MULTIALLELIC = (
    "on multiallelic sites the FrequencySpectrum builds a per-allele SFS whose "
    "segregating-class count and pi differ from the scalar's per-allele counts, "
    "so the SFS-derived estimators that depend on them diverge. #100"
)

# (stat, path) -> list of (reason, conditions) rules; the first rule whose
# ``conditions`` set contains the condition (or is None, meaning every
# condition where the path runs) applies. A key carries more than one rule when
# it diverges for different reasons under different conditions. The fused/exclude
# skip is handled separately, so a fused rule with conditions=None never reaches
# the (skipped) missing_exclude cell.
_XFAILS = {
    # fused kernel still mishandles missing data for these two estimators.
    ("theta_w", "fused"): [(_FUSED_MISSING, _WHEN_PARTIAL)],
    ("tajimas_d", "fused"): [(_FUSED_MISSING, _WHEN_PARTIAL)],

    # neutrality-test variance differs under variable per-site sample sizes,
    # and the FS path additionally diverges under multiallelic sites.
    ("tajimas_d", "fs"): [(_FS_VARIANCE, _WHEN_PARTIAL),
                          (_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("tajimas_d", "scatter"): [(_SCATTER_VARIANCE, _WHEN_PARTIAL)],
    ("normalized_fay_wu_h", "fs"): [(_FS_VARIANCE, _WHEN_PARTIAL),
                                    (_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("normalized_fay_wu_h", "scatter"): [(_SCATTER_VARIANCE, _WHEN_PARTIAL)],
    ("zeng_e", "fs"): [(_FS_VARIANCE, _WHEN_PARTIAL),
                       (_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("zeng_e", "scatter"): [(_SCATTER_VARIANCE, _WHEN_PARTIAL)],

    # FS per-allele SFS diverges from the scalar per-allele counts (multiallelic).
    ("pi", "fs"): [(_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("theta_w", "fs"): [(_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("segregating_sites", "fs"): [(_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],
    ("fay_wu_h", "fs"): [(_FS_MULTIALLELIC, _WHEN_MULTIALLELIC)],

    # fused fst_wc kernel not updated to the per-allele WC ANOVA (all conditions).
    ("fst_wc", "fused"): [(_FST_WC_FUSED, None)],

    # scatter da within-pop pi term uses a different site set under exclude.
    ("da", "scatter"): [(_DA_SCATTER_EXCLUDE, _WHEN_EXCLUDE)],
}


def _status(stat, path, condition):
    """Return ('ok'|'xfail'|'skip', reason) for one parametrized cell."""
    cond = _CONDITIONS[condition]

    # The fused engine is only dispatched for missing_data='include'.
    if path == "fused" and cond.missing_data == "exclude":
        return "skip", "fused engine requires missing_data='include'"

    for reason, conditions in _XFAILS.get((stat, path), ()):
        if conditions is None or condition in conditions:
            return "xfail", reason

    return "ok", None


def _cases():
    for condition in _CONDITIONS:
        for stat in _STATS:
            for path in _supported_paths(stat):
                kind, reason = _status(stat, path, condition)
                marks = []
                if kind == "xfail":
                    marks.append(pytest.mark.xfail(reason=reason, strict=True))
                elif kind == "skip":
                    marks.append(pytest.mark.skip(reason=reason))
                yield pytest.param(condition, stat, path, marks=marks,
                                   id=f"{condition}-{stat}-{path}")


def _assert_agrees(stat, reference, value):
    if _STATS[stat].kind == "count":
        assert int(value) == int(reference)
    elif np.isnan(reference):
        assert np.isnan(value)
    else:
        assert np.isclose(value, reference, rtol=RTOL, atol=ATOL), (
            f"{stat}: got {value!r}, expected {reference!r}")


@pytest.mark.parametrize("condition,stat,path", list(_cases()))
def test_path_matches_scalar(request, condition, stat, path):
    """Every supported path reproduces the scalar reference for the condition."""
    cond = _CONDITIONS[condition]
    hm = request.getfixturevalue(cond.fixture)
    reference = _path_scalar(hm, stat, cond.missing_data)
    value = _PATHS[path](hm, stat, cond.missing_data)
    _assert_agrees(stat, reference, value)


@pytest.mark.parametrize("condition", list(_CONDITIONS))
def test_daf_hist_matches_scalar(request, condition):
    """daf_hist is vector-valued (daf_bin_0 .. daf_bin_{_DAF_N_BINS - 1}), so it
    sits outside the scalar _STATS matrix. Check its whole-region histogram
    against diversity.daf_histogram for every engine that computes it: the fused
    engine (include only) and the WindowedAnalyzer fallback (include and exclude).

    This checks the whole-region histogram; multi-window and window-boundary
    coverage for daf_hist (and mean_nsl) lives in dedicated windowed tests."""
    cond = _CONDITIONS[condition]
    hm = request.getfixturevalue(cond.fixture)
    md = cond.missing_data
    ref, _ = diversity.daf_histogram(hm, n_bins=_DAF_N_BINS, missing_data=md)

    # fallback (pyloop) computes daf_hist for every missing mode
    W = _whole_window_size(hm)
    an = WindowedAnalyzer(window_type="bp", window_size=W, step_size=W,
                          statistics=["daf_hist"], missing_data=md,
                          span_normalize=False)
    df = an.compute(hm)
    got_pyloop = np.array([_one_row(df, f"daf_bin_{b}") for b in range(_DAF_N_BINS)])
    np.testing.assert_allclose(got_pyloop, ref, rtol=RTOL, atol=ATOL)

    # fused engine computes daf_hist only under missing_data='include'
    if md == "include":
        out = windowed_statistics_fused(hm, _whole_bp_bins(hm),
                                        statistics=("daf_hist",), per_base=False,
                                        missing_data=md, population=None)
        got_fused = np.array([np.asarray(out[f"daf_bin_{b}"])[0]
                              for b in range(_DAF_N_BINS)])
        np.testing.assert_allclose(got_fused, ref, rtol=RTOL, atol=ATOL)
