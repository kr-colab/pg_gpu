"""Residual coverage for divergence + selection.

Covers the FST method dispatcher, the distance_based_stats aggregator,
garud_h on a GenotypeMatrix, the allele-count standardization default, cupy
position input to the gap computation, and the missing-aware haplotype
fallback. Each gap is pinned to an independent oracle -- a direct function, a
constituent statistic, or structured data with a known answer -- rather than a
golden constant. (admixture's normed=False paths live in the allel comparison
suite, which already has the fixtures and oracle for them.)
"""
import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, divergence, selection


def _two_pop_hm(seed=7, n_per_pop=8, seq_length=500_000):
    """Two named populations, biallelic, fully called."""
    demography = msprime.Demography()
    demography.add_population(name="p1", initial_size=10_000)
    demography.add_population(name="p2", initial_size=10_000)
    demography.add_population(name="anc", initial_size=10_000)
    demography.add_population_split(time=5000, derived=["p1", "p2"],
                                    ancestral="anc")
    ts = msprime.sim_ancestry(samples={"p1": n_per_pop, "p2": n_per_pop},
                              demography=demography, sequence_length=seq_length,
                              recombination_rate=1e-8, random_seed=seed,
                              ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=seed, model="binary")
    hm = HaplotypeMatrix.from_ts(ts)
    hm.sample_sets = dict(hm.sample_sets)
    return hm


@pytest.fixture(scope="module")
def hm():
    return _two_pop_hm()


# ── divergence: fst() method dispatcher ────────────────────────────────
@pytest.mark.parametrize("method,direct", [
    ("weir_cockerham", divergence.fst_weir_cockerham),
    ("nei", divergence.fst_nei),
], ids=["weir_cockerham", "nei"])
def test_fst_dispatch_matches_direct(hm, method, direct):
    assert divergence.fst(hm, "p1", "p2", method=method) == pytest.approx(
        direct(hm, "p1", "p2"))


def test_fst_unknown_method_raises(hm):
    with pytest.raises(ValueError, match="Unknown FST method"):
        divergence.fst(hm, "p1", "p2", method="nope")


# ── divergence: distance_based_stats aggregator (never called internally) ──
def test_distance_based_stats_matches_constituents(hm):
    d = divergence.distance_based_stats(hm, "p1", "p2")
    # snn, dxy_min, gmin, dd_rank are recomputed identically to the
    # standalone functions, so they must match exactly.
    assert d['snn'] == pytest.approx(divergence.snn(hm, "p1", "p2"))
    assert d['dxy_min'] == pytest.approx(divergence.dxy_min(hm, "p1", "p2"))
    assert d['gmin'] == pytest.approx(divergence.gmin(hm, "p1", "p2"))
    r1, r2 = divergence.dd_rank(hm, "p1", "p2")
    assert d['dd_rank1'] == pytest.approx(r1)
    assert d['dd_rank2'] == pytest.approx(r2)
    # dd1/dd2 use mean within-pop distance for pi, not diversity.pi, so only
    # sanity-check them here.
    assert d['dd1'] > 0
    assert d['dd2'] > 0


# ── selection: garud_h GenotypeMatrix dispatch ─────────────────────────
def test_garud_h_genotype_matrix_dispatch(hm):
    gm = GenotypeMatrix.from_haplotype_matrix(hm)
    assert selection.garud_h(gm) == pytest.approx(selection.garud_h_diploid(gm))


# ── selection: standardize_by_allele_count default bin count ───────────
def test_standardize_by_allele_count_defaults_n_bins():
    rng = np.random.default_rng(0)
    score = rng.normal(size=300)
    aac = rng.integers(1, 20, size=300)
    # bins and n_bins both None -> n_bins defaults to max(1, max(aac)//2).
    out, bins = selection.standardize_by_allele_count(score, aac)
    assert out.shape == score.shape
    assert len(bins) - 1 == max(1, int(np.max(aac) // 2))
    assert np.isfinite(out).any()


# ── selection: _compute_gaps accepts cupy position arrays ──────────────
def test_compute_gaps_accepts_cupy_positions():
    pos = np.array([100, 200, 450, 900], dtype=np.int64)
    map_pos = np.array([0.0, 1.0, 2.5, 5.0])
    ref = selection._compute_gaps(pos, map_pos=map_pos)
    got = selection._compute_gaps(cp.asarray(pos), map_pos=cp.asarray(map_pos))
    np.testing.assert_allclose(got, ref)


# ── selection: missing-aware haplotype fallback ────────────────────────
# With missing calls the grouping falls back to wildcard matching. Structured
# data keeps the answer deterministic: identical rows collapse to one group,
# and a -1 inside a group still matches only that group.
def test_distinct_haplotype_frequencies_missing_all_identical():
    hap = cp.zeros((6, 5), dtype=cp.int8)
    hap[0, 0] = -1
    hap[3, 4] = -1
    freqs = selection._distinct_haplotype_frequencies_missing(hap)
    np.testing.assert_allclose(freqs, [1.0])


def test_distinct_haplotype_frequencies_missing_two_groups():
    a = np.zeros((3, 6), dtype=np.int8)
    b = np.ones((3, 6), dtype=np.int8)
    hap = np.vstack([a, b])
    hap[0, 0] = -1   # wildcard, still matches only the all-0 group
    hap[4, 2] = -1   # wildcard, still matches only the all-1 group
    freqs = selection._distinct_haplotype_frequencies_missing(cp.asarray(hap))
    np.testing.assert_allclose(sorted(freqs), [0.5, 0.5])


def test_distinct_haplotype_frequencies_missing_numpy_input():
    # A host array with missing takes the numpy has_missing / CPU branch.
    hap = np.zeros((4, 5), dtype=np.int8)
    hap[1, 2] = -1
    freqs = selection._distinct_haplotype_frequencies_missing(hap)
    np.testing.assert_allclose(freqs, [1.0])


def test_distinct_haplotype_frequencies_missing_no_missing_passthrough():
    # No missing, host input -> uploaded and delegated to the exact-grouping
    # implementation.
    hap = np.zeros((4, 5), dtype=np.int8)
    freqs = selection._distinct_haplotype_frequencies_missing(hap)
    np.testing.assert_allclose(freqs, [1.0])


def test_moving_garud_h_missing_fallback():
    # Two haplotype groups with intra-group missing -> one window, H1 = 0.5.
    a = np.zeros((4, 10), dtype=np.int8)
    b = np.ones((4, 10), dtype=np.int8)
    hap = np.vstack([a, b])
    hap[0, 0] = -1
    hap[5, 3] = -1
    pos = np.arange(10, dtype=np.int64) * 100
    m = HaplotypeMatrix(hap, pos, 0, 1000)
    h1, h12, h123, h2_h1 = selection.moving_garud_h(m, size=10)
    np.testing.assert_allclose(h1, [0.5])
