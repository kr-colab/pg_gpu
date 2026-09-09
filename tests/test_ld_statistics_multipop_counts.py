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
