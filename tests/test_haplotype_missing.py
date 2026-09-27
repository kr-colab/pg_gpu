"""Haplotype identity with missing calls: the EM rule and every path that uses it.

Complete haplotypes set the distinct haplotypes; EM splits each incomplete
haplotype across the complete ones it matches at its called sites; an
incomplete haplotype that matches none is a singleton; a window with no
complete haplotype is NaN.
"""
import cupy as cp
import numpy as np
import pytest

from pg_gpu import (
    GenotypeMatrix, HaplotypeMatrix, diversity, ld_statistics, selection,
    windowed_analysis,
)
from pg_gpu._haplotype_groups import frequencies, haplotype_groups
from pg_gpu.windowed_analysis import (
    windowed_statistics_fused, windowed_statistics_fused_chunked,
)

GARUD = ["garud_h1", "garud_h12", "garud_h123", "garud_h2h1"]
SPACING = 10


def _hm(rows):
    hap = np.asarray(rows, dtype=np.int8)
    n_var = hap.shape[1]
    return HaplotypeMatrix(hap, np.arange(n_var) * SPACING + 1, 0,
                           n_var * SPACING + 1)


def _freqs(rows):
    return frequencies(cp.asarray(np.asarray(rows, dtype=np.int8))).get()


def _founder_data(seed, n_hap=60, n_var=400, n_founders=6, rate=0.002):
    """Founder haplotypes with scattered missing calls."""
    rng = np.random.default_rng(seed)
    founders = rng.integers(0, 2, size=(n_founders, n_var)).astype(np.int8)
    hap = founders[rng.integers(0, n_founders, size=n_hap)]
    hap[rng.random(hap.shape) < rate] = -1
    return hap


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_no_shared_site_without_complete_haplotype_is_nan():
    # Two haplotypes with no jointly called site and no complete haplotype:
    # nothing anchors a frequency, so every frequency statistic is NaN.
    h = _hm([[0, -1], [-1, 1]])
    assert np.all(np.isnan(selection.garud_h(h)))
    assert np.isnan(diversity.haplotype_count(h))
    assert np.isnan(diversity.haplotype_diversity(h))


def test_incomplete_rows_join_the_complete_haplotype_they_match():
    # [0, 1] is complete, and both incomplete rows match it at their calls.
    h = _hm([[0, -1], [-1, 1], [0, 1]])
    assert selection.garud_h(h)[0] == pytest.approx(1.0)
    assert diversity.haplotype_count(h) == 1


def test_result_does_not_depend_on_row_order():
    x, y, m = [[0, 0]] * 4, [[1, 1]] * 3, [[-1, -1]]
    # A row with no calls leaves the EM fixed point at the complete
    # frequencies 4/7 and 3/7.
    expected = (4 / 7) ** 2 + (3 / 7) ** 2
    for rows in (x + y + m, m + x + y, y + m + x):
        assert selection.garud_h(_hm(rows))[0] == pytest.approx(expected)


def test_em_split_is_the_maximum_likelihood_estimate():
    # Three [0,0], one [0,1], one [0,-1]. The MLE solves p = (3 + p) / 5,
    # so p = 3/4.
    np.testing.assert_allclose(
        _freqs([[0, 0]] * 3 + [[0, 1]] + [[0, -1]]), [0.75, 0.25])


def test_em_matches_independent_numpy_em():
    hap = _founder_data(11, n_hap=80, n_var=6, n_founders=5, rate=0.25)
    counts = haplotype_groups(cp.asarray(hap))[0].get()

    complete = hap[(hap >= 0).all(axis=1)]
    groups, n_g = np.unique(complete, axis=0, return_counts=True)
    incomplete = hap[(hap < 0).any(axis=1)]
    compat = np.array([[np.all((r < 0) | (r == g)) for g in groups]
                       for r in incomplete])
    anchored = compat.any(axis=1)
    total = n_g.sum() + anchored.sum()
    q = n_g / n_g.sum()
    for _ in range(20000):
        w = compat[anchored] * q
        w = w / w.sum(axis=1, keepdims=True)
        q = (n_g + w.sum(axis=0)) / total

    # Group order differs (content hash vs lexicographic), so compare the
    # sorted group counts; singletons follow the groups.
    np.testing.assert_allclose(np.sort(counts[:len(groups)]),
                               np.sort(q * total), rtol=1e-8)
    np.testing.assert_array_equal(counts[len(groups):], 1.0)
    assert counts[len(groups):].size == (~anchored).sum()
    assert counts.sum() == pytest.approx(hap.shape[0])


def test_unmatched_incomplete_rows_are_singletons():
    # [1,1,-1] and its copy match no complete row: two singletons, never
    # joined with each other.
    rows = [[0, 0, 0], [0, 0, 0], [1, 1, -1], [1, 1, -1]]
    assert diversity.haplotype_count(_hm(rows)) == 3
    np.testing.assert_allclose(_freqs(rows), [0.5, 0.25, 0.25])


def test_multiallelic_and_wide_dtypes_read_in_place():
    rng = np.random.default_rng(4)
    hap = rng.integers(0, 4, size=(5, 50)).astype(np.int8)[rng.integers(0, 5, size=40)]
    hap[rng.random(hap.shape) < 0.01] = -1
    ref = frequencies(cp.asarray(hap)).get()
    for dt in (np.int16, np.int32):
        for order in ("C", "F"):
            wide = cp.asarray(np.asarray(hap, dtype=dt, order=order))
            np.testing.assert_allclose(frequencies(wide).get(), ref)


def test_no_missing_matches_exact_distinct_count():
    rng = np.random.default_rng(5)
    hap = rng.integers(0, 2, size=(8, 30)).astype(np.int8)[rng.integers(0, 8, size=50)]
    h = _hm(hap)
    assert diversity.haplotype_count(h) == len({r.tobytes() for r in hap})


# ---------------------------------------------------------------------------
# Row-order invariance of every public entry point
# ---------------------------------------------------------------------------

def test_every_statistic_is_row_order_invariant():
    hap = _founder_data(7)
    perm = np.random.default_rng(0).permutation(hap.shape[0])
    a, b = _hm(hap), _hm(hap[perm])
    np.testing.assert_allclose(selection.garud_h(a), selection.garud_h(b))
    assert diversity.haplotype_count(a) == diversity.haplotype_count(b)
    assert diversity.haplotype_diversity(a) == pytest.approx(
        diversity.haplotype_diversity(b))
    assert ld_statistics.mu_ld(a) == pytest.approx(ld_statistics.mu_ld(b))


# ---------------------------------------------------------------------------
# Every windowed path agrees with the scalar
# ---------------------------------------------------------------------------

def _scalar_per_window(hap, win, missing_data='include'):
    out = []
    for lo, hi in win:
        sub = _hm(hap[:, lo:hi])
        out.append([*selection.garud_h(sub, missing_data=missing_data),
                    diversity.haplotype_count(sub, missing_data=missing_data)])
    return np.array(out, dtype=np.float64)


@pytest.fixture(scope="module")
def windowed_case():
    hap = _founder_data(21, n_hap=40, n_var=400, rate=0.003)
    # One window in which every row has a missing call: NaN everywhere.
    hap[:, 250] = -1
    win = [(0, 100), (100, 200), (200, 300), (300, 400)]
    return hap, win


def test_moving_garud_h_matches_scalar(windowed_case):
    hap, win = windowed_case
    ref = _scalar_per_window(hap, win)[:, :4]
    got = np.array(selection.moving_garud_h(_hm(hap), size=100)).T
    np.testing.assert_allclose(got, ref, equal_nan=True)
    assert np.isnan(got[2]).all() and not np.isnan(got[[0, 1, 3]]).any()


@pytest.mark.parametrize("engine", ["windowed_analysis", "fused", "chunked"])
def test_windowed_engines_match_scalar(windowed_case, engine):
    hap, win = windowed_case
    ref = _scalar_per_window(hap, win)
    h = _hm(hap)
    stats = GARUD + ["haplotype_count"]
    size = 100 * SPACING
    if engine == "windowed_analysis":
        df = windowed_analysis(h, window_size=size, step_size=size,
                               statistics=stats)
        got = df[stats].to_numpy(dtype=np.float64)[:len(win)]
    else:
        fn = (windowed_statistics_fused if engine == "fused"
              else windowed_statistics_fused_chunked)
        edges = [lo * SPACING for lo, _ in win] + [len(hap[0]) * SPACING]
        res = fn(h, bp_bins=np.array(edges), statistics=tuple(stats),
                 per_base=False)
        got = np.column_stack([np.asarray(res[s], dtype=np.float64)
                               for s in stats])
    np.testing.assert_allclose(got, ref, equal_nan=True)


def test_windowed_haplotype_count_is_float_only_with_nan(windowed_case):
    hap, _ = windowed_case
    size = 100 * SPACING
    df = windowed_analysis(_hm(hap), window_size=size, step_size=size,
                           statistics=["haplotype_count"])
    assert df["haplotype_count"].dtype.kind == "f"
    df = windowed_analysis(_hm(hap[:, :200]), window_size=size, step_size=size,
                           statistics=["haplotype_count"])
    assert df["haplotype_count"].dtype.kind == "i"


def test_windowed_exclude_matches_scalar(windowed_case):
    hap, win = windowed_case
    ref = _scalar_per_window(hap, win, missing_data="exclude")
    h = _hm(hap)
    size = 100 * SPACING
    stats = GARUD + ["haplotype_count"]
    fused = windowed_analysis(h, window_size=size, step_size=size,
                              statistics=stats, missing_data="exclude")
    np.testing.assert_allclose(
        fused[stats].to_numpy(dtype=np.float64)[:len(win)], ref)
    # A mixed request takes the per-window fallback; its Garud columns
    # equal the scalar ones too.
    mixed = windowed_analysis(h, window_size=size, step_size=size,
                              statistics=stats + ["pi"], missing_data="exclude")
    np.testing.assert_allclose(
        mixed[stats].to_numpy(dtype=np.float64)[:len(win)], ref)


# ---------------------------------------------------------------------------
# GenotypeMatrix
# ---------------------------------------------------------------------------

def _geno(hap):
    g = np.where((hap[::2] < 0) | (hap[1::2] < 0), -1, hap[::2] + hap[1::2])
    n_var = hap.shape[1]
    return GenotypeMatrix(g.astype(np.int8), np.arange(n_var) * SPACING + 1,
                          0, n_var * SPACING + 1)


def test_genotype_garud_uses_missing_data():
    hap = _founder_data(31, n_hap=80, rate=0.002)
    gm = _geno(hap)
    inc = selection.garud_h(gm)
    exc = selection.garud_h(gm, missing_data="exclude")
    g = cp.asnumpy(gm.genotypes)
    complete = g[:, (g >= 0).all(axis=0)]
    ref = selection.garud_h(GenotypeMatrix(complete, np.arange(complete.shape[1]) + 1))
    np.testing.assert_allclose(exc, ref)
    assert not np.allclose(inc, exc)


def test_genotype_garud_is_row_order_invariant():
    hap = _founder_data(32, n_hap=80, rate=0.002)
    gm = _geno(hap)
    g = cp.asnumpy(gm.genotypes)
    perm = np.random.default_rng(1).permutation(g.shape[0])
    shuffled = GenotypeMatrix(g[perm], np.arange(g.shape[1]) * SPACING + 1)
    np.testing.assert_allclose(selection.garud_h(gm), selection.garud_h(shuffled))


# ---------------------------------------------------------------------------
# mu_ld
# ---------------------------------------------------------------------------

def test_mu_ld_unambiguous_missing_matches_clean():
    rng = np.random.default_rng(41)
    founders = rng.integers(0, 2, size=(4, 40)).astype(np.int8)
    clean = founders[np.repeat(np.arange(4), 5)]
    masked = clean.copy()
    # One call masked in one copy of each founder; the founders differ at
    # many sites, so each masked row still matches only its own founder.
    for f in range(4):
        masked[f * 5, rng.integers(0, 40)] = -1
    assert ld_statistics.mu_ld(_hm(masked)) == pytest.approx(
        ld_statistics.mu_ld(_hm(clean)))


def test_mu_ld_is_nan_without_complete_half():
    hap = np.zeros((6, 10), dtype=np.int8)
    hap[:, 2] = -1   # every row misses a call in the left half
    assert np.isnan(ld_statistics.mu_ld(_hm(hap)))


def test_window_longer_than_one_hash_segment():
    # 9000 sites span three hash segments; identical rows must still group.
    rng = np.random.default_rng(51)
    founders = rng.integers(0, 2, size=(3, 9000)).astype(np.int8)
    hap = founders[np.repeat(np.arange(3), [5, 3, 2])]
    hap[0, 8500] = -1
    hap[5, 10] = -1
    np.testing.assert_allclose(_freqs(hap), [0.5, 0.3, 0.2])


def test_em_repeats_bit_for_bit():
    # The E-step gathers in a fixed order with no atomics.
    hap = cp.asarray(_founder_data(61, n_hap=300, n_var=8, n_founders=12, rate=0.2))
    a = haplotype_groups(hap)[0].get()
    b = haplotype_groups(hap)[0].get()
    np.testing.assert_array_equal(a, b)


def test_streaming_matches_scalar_with_missing(tmp_path):
    from .conftest import simulate_hm
    hm = simulate_hm(n_samples=20, seq_length=50_000, seed=42,
                     mutation_model='binary')
    hap = cp.asnumpy(hm.haplotypes).astype(np.int8)
    pos = cp.asnumpy(hm.positions)
    rng = np.random.default_rng(71)
    hap[rng.random(hap.shape) < 0.002] = -1
    hm = HaplotypeMatrix(hap, pos, 0, 50_000)
    hm.samples = [f"s{i}" for i in range(hap.shape[0] // 2)]
    path = str(tmp_path / "missing.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")

    stats = GARUD + ["haplotype_count"]
    stream = windowed_analysis(
        HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=10_000),
        window_size=5_000, step_size=5_000, statistics=stats)
    # Each streaming window against the scalar rule on the same variants.
    ref = []
    for lo, hi in zip(stream["start"], stream["end"]):
        cols = np.nonzero((pos >= lo) & (pos < hi))[0]
        sub = _hm(hap[:, cols])
        ref.append([*selection.garud_h(sub), diversity.haplotype_count(sub)])
    assert (hap < 0).any()
    np.testing.assert_allclose(stream[stats].to_numpy(dtype=np.float64),
                               np.array(ref, dtype=np.float64), equal_nan=True)
