"""Coverage for the counts-based multi-population LD moment API, and a guard on
its symmetrization consistency.

`ld_statistics.dz` / `ld_statistics.pi2` accept a concatenated per-population
counts array and a population-index tuple, and dispatch through `_dz_multi` /
`_pi2_multi` to per-index-pattern formulas -- including the three- and
four-distinct-population branches that have no in-package caller (the pipeline
computes multi-population moments through the fused CUDA kernels) and were
untested, though the module is public.

Each pattern is cross-checked against the fused kernels (`compute_all_dz_hap` /
`compute_all_pi2_hap`) on the same counts. Those kernels return the raw
single-index term for every pattern and are validated against moments.LD (see
test_moments_ld_multipop), so agreement pins the counts-API formula for that
term.

`dz` returns the raw single-index term for every pattern, so it agrees
throughout. `pi2` does not: the `_pi2_multi` iikk branch self-symmetrizes
(returns `0.5*(numer1+numer2)`, the two-locus average) while every other pi2
branch returns a single term. So `ld_statistics.pi2` returns a different kind
of quantity depending on the index pattern -- an inconsistent public contract.
That case is marked xfail(strict) below so the mismatch is pinned and a fix
(unifying the convention) trips the marker.
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


# Dz(i,j,k): the i,i,j / i,j,i / i,j,j (two-pop) and all-different (three
# distinct) branches. Dz returns a single term for every pattern -> all agree.
DZ_CONFIGS = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)]

# pi2 patterns across every branch. All return a single term matching the
# fused raw call, EXCEPT iikk, whose _pi2_multi branch self-symmetrizes --
# an inconsistent contract, pinned as an expected failure.
PI2_CONFIGS = [
    (0, 0, 0, 1),                                          # iiij
    (0, 1, 0, 1),                                          # ijij
    (0, 0, 1, 2),                                          # iikl (3 distinct)
    (0, 1, 2, 2),                                          # ijkk (3 distinct)
    (0, 1, 2, 0), (0, 1, 0, 2), (0, 1, 1, 2), (0, 1, 2, 1),  # shared, orderings
    (0, 1, 2, 3),                                          # all-different
    pytest.param(
        (0, 0, 1, 1),                                     # iikk -- self-symmetrizes
        marks=pytest.mark.xfail(
            strict=True,
            reason="_pi2_multi iikk branch returns the two-locus average "
                   "0.5*(numer1+numer2) while every other pi2 branch returns "
                   "a single index term; ld_statistics.pi2 is inconsistent "
                   "across index patterns")),
]


@pytest.mark.parametrize("cfg", DZ_CONFIGS, ids=[str(c) for c in DZ_CONFIGS])
def test_dz_counts_api_matches_fused(counts_and_pops, cfg):
    counts, n_valid, pops = counts_and_pops
    _agree(ld_statistics.dz(counts, cfg, n_valid),
           compute_all_dz_hap(pops, [cfg])[0])


@pytest.mark.parametrize("cfg", PI2_CONFIGS,
                         ids=[str(c.values[0]) if hasattr(c, "values") else str(c)
                              for c in PI2_CONFIGS])
def test_pi2_counts_api_matches_fused(counts_and_pops, cfg):
    counts, n_valid, pops = counts_and_pops
    _agree(ld_statistics.pi2(counts, cfg, n_valid),
           compute_all_pi2_hap(pops, [cfg])[0])


def test_multi_pop_configs_are_non_degenerate(counts_and_pops):
    # Guard against a fixture that trivially agrees everywhere: the
    # four-distinct pi2 and three-distinct Dz must carry real signal.
    counts, n_valid, _ = counts_and_pops
    assert np.any(cp.asnumpy(ld_statistics.pi2(counts, (0, 1, 2, 3), n_valid)) != 0)
    assert np.any(cp.asnumpy(ld_statistics.dz(counts, (0, 1, 2), n_valid)) != 0)
