"""Genotype LD with degenerate per-pair / per-site sample counts (issue #123).

Two layers:

* Correctness -- pg_gpu's genotype kernels match moments' Python reference
  (``stats_from_genotype_counts``, with the PR #258 ``-20`` pi2(i,i;i,i)
  coefficient) to machine precision on well-sampled count vectors, applying the
  same index symmetrization moments' pipeline uses. This needs the moments env.

* Robustness -- when missing data drops a population's per-pair sample count
  below the order of a statistic's denominator (``n*(n-1)*(n-2)`` etc.), the
  per-pair statistic is 0/0, which is *undefined* in moments too (its formula
  raises / yields nan). pg_gpu contributes 0 there -- excluding the undefined
  pair -- instead of poisoning the whole r-bin sum with a NaN. These checks
  need only a GPU.
"""
import numpy as np
import cupy as cp
import pytest

from pg_gpu import GenotypeMatrix
from pg_gpu.genotype_kernels import compute_multi_pop_statistics_batch_geno
from pg_gpu.ld_pipeline import compute_genotype_counts_for_pairs
from pg_gpu.moments_ld import _compute_heterozygosity, _generate_stat_specs, _ld_names

try:
    from moments.LD import stats_from_genotype_counts as sgc
    HAVE_MOMENTS = True
except Exception:
    HAVE_MOMENTS = False


# pg_gpu's count columns are (n00,n01,n02,n10,n11,n12,n20,n21,n22); moments'
# n1..n9 are (n22,n21,...,n00) -- the reverse.
def _to_pg(mom9):
    return np.asarray(mom9, dtype=np.int32)[::-1]


def _moments_ref(name, counts):
    """Reference value for ``name`` from moments' Python per-pair functions,
    with the same symmetrization ``moments.LD.Parsing._call_sgc`` applies.
    ``counts`` is a list of per-population moments-order 9-count vectors."""
    parts = name.split("_")
    kind, idx = parts[0], [int(x) for x in parts[1:]]
    if kind == "DD":
        return sgc.DD(counts, idx)
    if kind == "Dz":
        i, j, k = idx
        if j == k:
            return sgc.Dz(counts, [i, j, k])
        return 0.5 * (sgc.Dz(counts, [i, j, k]) + sgc.Dz(counts, [i, k, j]))
    i, j, k, l = idx
    if i == j and k == l and i == k:
        return sgc.pi2(counts, [i, j, k, l])
    if i == j and k == l:
        return 0.5 * (sgc.pi2(counts, [i, j, k, l]) + sgc.pi2(counts, [k, l, i, j]))
    if i == j:
        return 0.25 * sum(sgc.pi2(counts, o) for o in
                          ([i, j, k, l], [i, j, l, k], [k, l, i, j], [l, k, i, j]))
    if k == l:
        return 0.25 * sum(sgc.pi2(counts, o) for o in
                          ([i, j, k, l], [j, i, k, l], [k, l, i, j], [k, l, j, i]))
    return (1.0 / 8) * sum(sgc.pi2(counts, o) for o in (
        [i, j, k, l], [i, j, l, k], [j, i, k, l], [j, i, l, k],
        [k, l, i, j], [l, k, i, j], [k, l, j, i], [l, k, j, i]))


@pytest.mark.skipif(not HAVE_MOMENTS, reason="needs the moments pixi env")
def test_genotype_stats_match_moments_python():
    # Random, well-sampled 3-population genotype counts (n>=8 per pair) exercise
    # every DD/Dz/pi2 case, including the between-population kernels. pg_gpu must
    # equal moments' Python reference to floating-point precision.
    rng = np.random.default_rng(0)
    n_pops, n_pairs = 3, 300
    counts_mom, counts_pg = [], []
    for _ in range(n_pops):
        mom = [rng.multinomial(int(rng.integers(8, 40)), np.ones(9) / 9)
               for _ in range(n_pairs)]
        counts_mom.append(mom)
        counts_pg.append(cp.asarray(np.array([_to_pg(c) for c in mom]),
                                    dtype=cp.int32))

    names = _ld_names(n_pops)
    gpu = cp.asnumpy(compute_multi_pop_statistics_batch_geno(
        counts_pg, [None] * n_pops, None, _generate_stat_specs(n_pops)))

    for k, nm in enumerate(names):
        ref = np.array([_moments_ref(nm, [counts_mom[p][pp] for p in range(n_pops)])
                        for pp in range(n_pairs)])
        np.testing.assert_allclose(gpu[:, k], ref, rtol=1e-9, atol=1e-12,
                                   err_msg=f"{nm} disagrees with moments")


def _pop_counts(g, pop_idx, n_var):
    i, j = cp.triu_indices(n_var, k=1)
    return compute_genotype_counts_for_pairs(
        cp.asarray(g, dtype=cp.int8),
        i.astype(cp.int32), j.astype(cp.int32),
        cp.asarray(pop_idx, dtype=cp.int32))


def test_between_pop_dz_degenerate_pair_is_finite():
    # pop0 small, with missing data leaving one individual typed at both loci of
    # the pair -- the n*(n-1)*(n-2)=0 case. pops 1, 2 fully typed.
    M = -1
    g = np.array([
        [1, 1], [0, M], [M, 2], [M, M],
        [0, 1], [1, 1], [2, 0], [1, 2], [0, 0], [1, 1], [2, 1], [0, 2],
        [1, 0], [0, 1], [1, 1], [2, 2], [0, 1], [1, 0], [2, 1], [1, 1],
    ], dtype=np.int8)
    pops = [list(range(0, 4)), list(range(4, 12)), list(range(12, 20))]
    counts, n_valid = zip(*(_pop_counts(g, p, g.shape[1]) for p in pops))
    assert int(counts[0].sum()) == 1

    stats = cp.asnumpy(compute_multi_pop_statistics_batch_geno(
        list(counts), list(n_valid), None, _generate_stat_specs(3)))
    assert np.all(np.isfinite(stats))

    names = _ld_names(3)
    for nm in ("Dz_0_0_1", "Dz_0_1_1", "Dz_0_1_2"):
        assert stats[0, names.index(nm)] == 0.0


def test_between_pop_pi2_zero_sample_pop_is_finite():
    # pop0 entirely missing at the second locus -> zero jointly-typed -> the
    # normalized-frequency pi2 kernels (alldiff, iikl, ijkk, shared) would emit
    # NaN from 1/n=inf without the guard.
    M = -1
    g = np.array([
        [1, M], [0, M], [2, M],
        [0, 1], [1, 1], [2, 0], [1, 2],
        [1, 0], [0, 1], [1, 1], [2, 2],
        [0, 2], [1, 0], [2, 1], [1, 1],
    ], dtype=np.int8)
    pops = [list(range(0, 3)), list(range(3, 7)),
            list(range(7, 11)), list(range(11, 15))]
    counts, n_valid = zip(*(_pop_counts(g, p, g.shape[1]) for p in pops))
    assert int(counts[0].sum()) == 0

    stats = cp.asnumpy(compute_multi_pop_statistics_batch_geno(
        list(counts), list(n_valid), None, _generate_stat_specs(4)))
    assert np.all(np.isfinite(stats))

    names = _ld_names(4)
    for nm in ("pi2_0_1_2_3", "pi2_0_2_1_3"):
        assert stats[0, names.index(nm)] == 0.0


def test_heterozygosity_zero_sample_pop_site_is_finite():
    # A site where a whole population is missing makes the haploid-size
    # denominator zero; the per-site contribution is 0, not NaN.
    M = -1
    geno = np.array([[1, M], [0, M], [2, M],
                     [0, 1], [1, 0], [2, 1], [1, 2]], dtype=np.int8)
    gm = GenotypeMatrix(geno, np.array([100, 200]), 100, 200)
    gm.transfer_to_gpu()
    gm.sample_sets = {"A": [0, 1, 2], "B": [3, 4, 5, 6]}
    het = _compute_heterozygosity(gm, ["A", "B"], use_genotypes=True)
    assert all(np.isfinite(v) for v in het.values())
