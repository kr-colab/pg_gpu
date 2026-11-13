"""Coverage for the counts-based multi-population LD moment API.

`ld_statistics.dz` / `ld_statistics.pi2` accept a concatenated per-population
counts array and a population-index tuple, and dispatch to a per-index-pattern
formula for each population configuration.

The two statistics follow different, self-consistent conventions:

* `dz` returns the raw single-index term for every pattern, so it is
  cross-checked directly against the fused kernel (`compute_all_dz_hap`) on the
  same counts.
* `pi2` returns the *symmetrized* statistic for every pattern. pi2 is invariant
  under swapping its two loci and under swapping the two populations within a
  locus, so the named statistic averages the raw term over that symmetry orbit.
  It is cross-checked against the same average of the fused kernel
  (`compute_all_pi2_hap`) over the orbit -- the average the fused pipeline
  applies one layer up in `generate_stat_specs`.

Because the fused kernels are validated against moments.LD (see
test_moments_ld_multipop), agreement pins each counts-API formula.
"""
import numpy as np
import cupy as cp
import pytest

from pg_gpu import ld_statistics
from pg_gpu.haplotype_kernels import compute_all_dz_hap, compute_all_pi2_hap
from pg_gpu.ld_pipeline import PopData

N_PAIRS = 8
N_POPS = 4


@pytest.fixture(scope="module")
def counts_and_pops():
    """Synthetic per-population 4-way haplotype counts (N_PAIRS x 4*N_POPS)
    with per-population valid sizes, plus the fused-kernel PopData view of the
    same data. Counts are arbitrary but valid (each population's four counts
    sum to a size >= 4, so the projection estimators are defined)."""
    rng = np.random.default_rng(0)
    counts = np.zeros((N_PAIRS, 4 * N_POPS), dtype=np.float64)
    for p in range(N_POPS):
        for r in range(N_PAIRS):
            counts[r, p * 4:p * 4 + 4] = rng.multinomial(16, [0.4, 0.2, 0.2, 0.2])
    counts = cp.asarray(counts)
    n_valid = cp.stack(
        [counts[:, p * 4:p * 4 + 4].sum(axis=1) for p in range(N_POPS)], axis=1)
    pops = [PopData(counts[:, p * 4:p * 4 + 4], n_valid[:, p])
            for p in range(N_POPS)]
    return counts, n_valid, pops


def _agree(a, b):
    np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(b),
                               rtol=1e-9, atol=1e-12, equal_nan=True)


def _pi2_orbit(cfg):
    """The (i,j,k,l) arrangements pi2 is symmetric over: swap the two loci,
    and swap the two populations within each locus."""
    i, j, k, l = cfg
    return {(i, j, k, l), (i, j, l, k), (j, i, k, l), (j, i, l, k),
            (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)}


def _pi2_symmetrized_fused(pops, cfg):
    """The fused pi2 kernel averaged over cfg's symmetry orbit -- the
    symmetrized statistic ld_statistics.pi2 returns for every pattern."""
    orbit = _pi2_orbit(cfg)
    return sum(compute_all_pi2_hap(pops, [c])[0] for c in orbit) / len(orbit)


# Dz(i,j,k): the i,i,j / i,j,i / i,j,j (two-pop) and all-different (three
# distinct) branches. Dz returns a single term for every pattern -> all agree
# with the raw fused call.
DZ_CONFIGS = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)]

# pi2 patterns across every branch: iiij (degenerate), ijij, iikk, the three-
# and four-distinct branches, and the shared-population orderings. Each returns
# the symmetrized statistic, cross-checked against the orbit-averaged kernel.
PI2_CONFIGS = [
    (0, 0, 0, 1),                                          # iiij (degenerate)
    (0, 1, 0, 1),                                          # ijij
    (0, 0, 1, 1),                                          # iikk
    (0, 0, 1, 2),                                          # iikl (3 distinct)
    (0, 1, 2, 2),                                          # ijkk (3 distinct)
    (0, 1, 2, 0), (0, 1, 0, 2), (0, 1, 1, 2), (0, 1, 2, 1),  # shared, orderings
    (0, 1, 2, 3),                                          # all-different
]


@pytest.mark.parametrize("cfg", DZ_CONFIGS, ids=[str(c) for c in DZ_CONFIGS])
def test_dz_counts_api_matches_fused(counts_and_pops, cfg):
    counts, n_valid, pops = counts_and_pops
    _agree(ld_statistics.dz(counts, cfg, n_valid),
           compute_all_dz_hap(pops, [cfg])[0])


@pytest.mark.parametrize("cfg", PI2_CONFIGS, ids=[str(c) for c in PI2_CONFIGS])
def test_pi2_counts_api_matches_symmetrized_fused(counts_and_pops, cfg):
    counts, n_valid, pops = counts_and_pops
    _agree(ld_statistics.pi2(counts, cfg, n_valid),
           _pi2_symmetrized_fused(pops, cfg))


def test_pi2_is_symmetric_across_orbit(counts_and_pops):
    # pi2 returns one quantity for every arrangement in an index pattern's
    # symmetry orbit, so a locus swap or a within-locus population swap leaves
    # the result unchanged.
    counts, n_valid, _ = counts_and_pops
    base = ld_statistics.pi2(counts, (0, 0, 1, 2), n_valid)
    for cfg in [(0, 0, 2, 1), (1, 2, 0, 0), (2, 1, 0, 0)]:
        _agree(ld_statistics.pi2(counts, cfg, n_valid), base)


def test_multi_pop_configs_are_non_degenerate(counts_and_pops):
    # Guard against a fixture that trivially agrees everywhere: the
    # four-distinct pi2 and three-distinct Dz must carry real signal.
    counts, n_valid, _ = counts_and_pops
    assert np.any(cp.asnumpy(ld_statistics.pi2(counts, (0, 1, 2, 3), n_valid)) != 0)
    assert np.any(cp.asnumpy(ld_statistics.dz(counts, (0, 1, 2), n_valid)) != 0)


@pytest.fixture(scope="module")
def degenerate_counts_and_pops():
    """Same shape as ``counts_and_pops``, but population 0 is entirely
    missing (n=0) at every pair -- below every pattern's threshold -- while
    populations 1-3 stay well-sampled. Every DZ_CONFIGS/PI2_CONFIGS pattern
    involves population 0, so this drives ``ld_statistics.dz``/``pi2`` and
    the fused kernels through the undefined regime together: if the two
    independent implementations disagreed about *where* NaN starts, one
    side would return a real number against the other's NaN and ``_agree``
    (``equal_nan=True``) would catch it."""
    rng = np.random.default_rng(1)
    counts = np.zeros((N_PAIRS, 4 * N_POPS), dtype=np.float64)
    for p in range(1, N_POPS):
        for r in range(N_PAIRS):
            counts[r, p * 4:p * 4 + 4] = rng.multinomial(16, [0.4, 0.2, 0.2, 0.2])
    counts = cp.asarray(counts)
    n_valid = cp.stack(
        [counts[:, p * 4:p * 4 + 4].sum(axis=1) for p in range(N_POPS)], axis=1)
    pops = [PopData(counts[:, p * 4:p * 4 + 4], n_valid[:, p])
            for p in range(N_POPS)]
    return counts, n_valid, pops


@pytest.mark.parametrize("cfg", DZ_CONFIGS, ids=[str(c) for c in DZ_CONFIGS])
def test_dz_counts_api_matches_fused_when_degenerate(degenerate_counts_and_pops, cfg):
    counts, n_valid, pops = degenerate_counts_and_pops
    _agree(ld_statistics.dz(counts, cfg, n_valid),
           compute_all_dz_hap(pops, [cfg])[0])


@pytest.mark.parametrize("cfg", PI2_CONFIGS, ids=[str(c) for c in PI2_CONFIGS])
def test_pi2_counts_api_matches_symmetrized_fused_when_degenerate(
        degenerate_counts_and_pops, cfg):
    counts, n_valid, pops = degenerate_counts_and_pops
    _agree(ld_statistics.pi2(counts, cfg, n_valid),
           _pi2_symmetrized_fused(pops, cfg))


def test_degenerate_fixture_actually_hits_the_undefined_regime(
        degenerate_counts_and_pops):
    # Guard against a fixture that trivially agrees because both sides
    # stayed finite: every config touches population 0, so both must be NaN.
    counts, n_valid, _ = degenerate_counts_and_pops
    assert np.all(np.isnan(cp.asnumpy(ld_statistics.dz(counts, (0, 1, 2), n_valid))))
    assert np.all(np.isnan(cp.asnumpy(
        ld_statistics.pi2(counts, (0, 1, 2, 3), n_valid))))


# ── the forms of n_valid ───────────────────────────────────────────────
DD_CONFIGS = [None, (0, 0), (2, 2), (0, 1), (1, 2), (3, 1)]
_N_VALID_CASES = ([(ld_statistics.dd, c) for c in DD_CONFIGS]
                  + [(ld_statistics.dz, c) for c in [None] + DZ_CONFIGS]
                  + [(ld_statistics.pi2, c) for c in [None] + PI2_CONFIGS])


@pytest.fixture(scope="module")
def uneven_n_valid(counts_and_pops):
    """Sample sizes that differ between populations and from the count
    totals, so a size read from the wrong population, or from the counts,
    changes the value."""
    _, n_valid, _ = counts_and_pops
    return n_valid + cp.arange(1, N_POPS + 1, dtype=n_valid.dtype)


@pytest.mark.parametrize("stat,cfg", _N_VALID_CASES,
                         ids=[f"{s.__name__}{c}" for s, c in _N_VALID_CASES])
def test_tuple_n_valid_equals_two_dimensional(counts_and_pops, uneven_n_valid, stat, cfg):
    counts, n_valid, _ = counts_and_pops
    as_tuple = tuple(uneven_n_valid[:, p] for p in range(N_POPS))
    expected = stat(counts, cfg, uneven_n_valid)
    _agree(stat(counts, cfg, as_tuple), expected)
    # The sizes are in use: the count totals give another value.
    assert not np.allclose(cp.asnumpy(expected), cp.asnumpy(stat(counts, cfg, n_valid)))


@pytest.mark.parametrize("stat", [ld_statistics.dd, ld_statistics.dz, ld_statistics.pi2],
                         ids=lambda s: s.__name__)
def test_tuple_n_valid_on_one_population_counts(counts_and_pops, uneven_n_valid, stat):
    counts, _, _ = counts_and_pops
    one = counts[:, :4]
    _agree(stat(one, None, (uneven_n_valid[:, 0],)), stat(one, None, uneven_n_valid[:, 0]))


@pytest.mark.parametrize("stat,cfg", [(ld_statistics.dd, (1, 2)), (ld_statistics.dd, (1, 1)),
                                      (ld_statistics.dz, (1, 1, 2)), (ld_statistics.pi2, (1, 1, 2, 2))],
                         ids=["dd_between", "dd_within", "dz", "pi2"])
def test_tuple_n_valid_none_entry_uses_count_total(counts_and_pops, uneven_n_valid, stat, cfg):
    # A population with None in the tuple takes its size from its counts.
    counts, n_valid, _ = counts_and_pops
    mixed = uneven_n_valid.copy()
    mixed[:, 1] = n_valid[:, 1]
    as_tuple = tuple(None if p == 1 else uneven_n_valid[:, p] for p in range(N_POPS))
    _agree(stat(counts, cfg, as_tuple), stat(counts, cfg, mixed))


def test_tuple_n_valid_without_the_population_raises(counts_and_pops, uneven_n_valid):
    # A tuple for two populations says nothing about population 2.
    counts, _, _ = counts_and_pops
    short = (uneven_n_valid[:, 0], uneven_n_valid[:, 1])
    with pytest.raises(IndexError):
        ld_statistics.dd(counts, (1, 2), short)


@pytest.mark.parametrize("stat", [ld_statistics.r, ld_statistics.r_squared, ld_statistics.dd_within],
                         ids=lambda s: s.__name__)
def test_one_population_statistics_read_every_n_valid_form(counts_and_pops, uneven_n_valid, stat):
    counts, _, _ = counts_and_pops
    one = counts[:, :4]
    expected = stat(one, uneven_n_valid[:, 0])
    _agree(stat(one, (uneven_n_valid[:, 0],)), expected)
    _agree(stat(one, uneven_n_valid), expected)
