"""Simulation-based bias tests for missing data handling.

Verifies that pg_gpu statistics are unbiased under MCAR (missing
completely at random) missing data at various rates. Uses msprime
simulations under the standard neutral model.

Run with: pixi run pytest tests/test_missing_data_bias.py --slow -v
"""

import numpy as np
import pytest
import msprime

from pg_gpu import HaplotypeMatrix, diversity, divergence
from pg_gpu.diversity import FrequencySpectrum

N_REPS = 50
N_HAPS_PER_POP = 100
SEQ_LENGTH = 200_000
MISSING_RATES = [0.0, 0.10, 0.30, 0.60]
BIAS_TOLERANCE = 0.05  # 5% relative bias


def _simulate(seed):
    """Simulate two populations under an isolation model."""
    demography = msprime.Demography()
    demography.add_population(name="pop1", initial_size=10_000)
    demography.add_population(name="pop2", initial_size=10_000)
    demography.add_population(name="anc", initial_size=10_000)
    demography.add_population_split(
        time=5000, derived=["pop1", "pop2"], ancestral="anc")
    ts = msprime.sim_ancestry(
        samples={"pop1": N_HAPS_PER_POP // 2,
                 "pop2": N_HAPS_PER_POP // 2},
        demography=demography,
        sequence_length=SEQ_LENGTH,
        recombination_rate=1e-8,
        random_seed=seed, ploidy=2)
    return msprime.sim_mutations(ts, rate=1e-8, random_seed=seed)


def _ts_to_hm(ts):
    # sample_sets are auto-populated from the named populations in ts.
    return HaplotypeMatrix.from_ts(ts)


def _add_missing(hm, rate, rng):
    hap = hm.haplotypes
    if hasattr(hap, 'get'):
        hap = hap.get()
    hap = hap.copy()
    hap[rng.random(hap.shape) < rate] = -1
    hm_miss = HaplotypeMatrix(
        hap, hm.positions, hm.chrom_start, hm.chrom_end)
    hm_miss.sample_sets = hm.sample_sets
    return hm_miss


def _check_bias(results, stat, miss_rate, near_zero=False,
                rel_tol=BIAS_TOLERANCE, abs_tol=2.0):
    """Assert that mean estimate at miss_rate matches truth."""
    truth_vals = results[stat][0.0]
    test_vals = results[stat][miss_rate]
    assert len(truth_vals) >= 20, f"Too few truth replicates for {stat}"
    assert len(test_vals) >= 20, f"Too few test replicates for {stat}"
    truth_mean = np.mean(truth_vals)
    test_mean = np.mean(test_vals)
    if near_zero or abs(truth_mean) < 1e-10:
        diff = abs(test_mean - truth_mean)
        assert diff < abs_tol, (
            f"{stat} at {miss_rate*100:.0f}% missing: "
            f"diff={diff:.3f} (truth={truth_mean:.3f}, "
            f"est={test_mean:.3f})")
    else:
        rel_bias = abs(test_mean / truth_mean - 1)
        assert rel_bias < rel_tol, (
            f"{stat} at {miss_rate*100:.0f}% missing: "
            f"bias={rel_bias*100:.1f}% (truth={truth_mean:.6f}, "
            f"est={test_mean:.6f})")


@pytest.fixture(scope="module")
def simulation_results():
    """Run all simulations and collect results.

    Returns dict[stat_name][miss_rate] = list of values.
    """
    def _pop1(fn, **kw):
        return lambda hm: fn(hm, population="pop1", missing_data='include',
                             **kw)

    def _twopop(fn, **kw):
        return lambda hm: fn(hm, "pop1", "pop2", missing_data='include',
                             **kw)

    stats = {
        "pi": _pop1(diversity.pi, span_normalize=False),
        "theta_w": _pop1(diversity.theta_w, span_normalize=False),
        "theta_h": _pop1(diversity.theta_h, span_normalize=False),
        "theta_l": _pop1(diversity.theta_l, span_normalize=False),
        # Same four estimators, span-normalized (the default): exercises the
        # missing-data rescale, since the physical span is the same at every
        # missing rate (missingness never moves chrom_start/chrom_end) while
        # the raw sum above was already shown unbiased, so any bias here is
        # from the rescale alone.
        "pi_rate": _pop1(diversity.pi),
        "theta_w_rate": _pop1(diversity.theta_w),
        "theta_h_rate": _pop1(diversity.theta_h),
        "theta_l_rate": _pop1(diversity.theta_l),
        "tajd": _pop1(diversity.tajimas_d),
        "fay_wus_h": _pop1(diversity.fay_wus_h),
        "zeng_e": _pop1(diversity.zeng_e),
        "dxy": _twopop(divergence.dxy),
        "fst_hudson": _twopop(divergence.fst_hudson),
        "fst_wc": _twopop(divergence.fst_weir_cockerham),
        "da": _twopop(divergence.da),
    }

    achaz_stats = {
        "achaz_pi": lambda fs: fs.theta("pi"),
        "achaz_watterson": lambda fs: fs.theta("watterson"),
        "achaz_theta_h": lambda fs: fs.theta("theta_h"),
        "achaz_theta_l": lambda fs: fs.theta("theta_l"),
        "achaz_tajd": lambda fs: fs.tajimas_d(),
        "achaz_zeng_e": lambda fs: fs.zeng_e(),
    }

    all_names = list(stats) + list(achaz_stats)
    results = {name: {r: [] for r in MISSING_RATES} for name in all_names}

    for rep in range(N_REPS):
        rng = np.random.default_rng(rep + 10001)
        ts = _simulate(rep + 1)
        hm_clean = _ts_to_hm(ts)
        if hm_clean.num_variants < 10:
            continue
        hm_clean.transfer_to_gpu()

        for rate in MISSING_RATES:
            hm = hm_clean if rate == 0.0 else _add_missing(hm_clean, rate, rng)
            if rate > 0:
                hm.transfer_to_gpu()

            for name, fn in stats.items():
                try:
                    val = fn(hm)
                    if np.isfinite(val):
                        results[name][rate].append(val)
                except Exception:
                    pass

            try:
                fs = FrequencySpectrum(hm, population="pop1")
                for name, fn in achaz_stats.items():
                    try:
                        val = fn(fs)
                        if np.isfinite(val):
                            results[name][rate].append(val)
                    except Exception:
                        pass
            except Exception:
                pass

    return results


# Statistics near zero under neutrality (differences of nearly-equal
# theta estimators: H = pi - theta_H, E = theta_L - theta_W, etc.)
NEAR_ZERO = {"tajd", "fay_wus_h", "zeng_e", "achaz_tajd", "achaz_zeng_e"}

# All stats that should be unbiased under MCAR
ALL_UNBIASED = [
    "pi", "theta_w", "theta_h", "theta_l",
    "pi_rate", "theta_w_rate", "theta_h_rate", "theta_l_rate",
    "tajd", "fay_wus_h", "zeng_e",
    "dxy", "fst_hudson", "fst_wc", "da",
    "achaz_pi", "achaz_watterson", "achaz_theta_h", "achaz_theta_l",
    "achaz_tajd", "achaz_zeng_e",
]


@pytest.mark.slow
class TestUnbiasedUnderMCAR:
    """Verify all statistics are unbiased under MCAR missing data."""

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    @pytest.mark.parametrize("stat", ALL_UNBIASED)
    def test_unbiased(self, simulation_results, stat, miss_rate):
        _check_bias(simulation_results, stat, miss_rate,
                     near_zero=(stat in NEAR_ZERO))


@pytest.mark.slow
class TestExcludeConsistency:
    """Verify exclude mode matches include on complete data."""

    def test_pi_exclude_equals_include(self):
        ts = _simulate(seed=999)
        hm = _ts_to_hm(ts)
        hm.transfer_to_gpu()
        pi_inc = diversity.pi(hm, population="pop1",
                              missing_data='include', span_normalize=False)
        pi_exc = diversity.pi(hm, population="pop1",
                              missing_data='exclude', span_normalize=False)
        assert abs(pi_inc - pi_exc) < 1e-10

    def test_dxy_exclude_equals_include(self):
        ts = _simulate(seed=999)
        hm = _ts_to_hm(ts)
        hm.transfer_to_gpu()
        dxy_inc = divergence.dxy(hm, "pop1", "pop2", missing_data='include')
        dxy_exc = divergence.dxy(hm, "pop1", "pop2", missing_data='exclude')
        assert abs(dxy_inc - dxy_exc) < 1e-10


class TestSpanNormalizedRescale:
    """pi/theta_w/theta_h/theta_l rescale their raw sum to the sites where
    the population has at least two calls before dividing by the span:
    without it, a site a population's own missing calls dropped from the
    sum stayed in the span denominator, reading as invariant rather than
    unmeasured, and silently pulling the per-base rate down."""

    def _pop_b(self, hap, n_var):
        hm = HaplotypeMatrix(hap, np.arange(n_var) * 10 + 1, 1, n_var * 10)
        hm.sample_sets = {"b": list(range(hap.shape[0] - 4, hap.shape[0]))}
        return hm

    def test_rate_matches_complete_data_under_heavy_missingness(self):
        # Half of population b's calls are missing at random; the fixed
        # per-base pi must stay close to the complete-data value instead of
        # shrinking toward it the way the raw sum alone would (unrescaled,
        # it would be cut roughly in half).
        rng = np.random.default_rng(0)
        n_hap, n_var = 20, 4000
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        pi_complete = diversity.pi(self._pop_b(hap.copy(), n_var), "b")

        miss = hap.copy()
        miss[-4:][rng.random((4, n_var)) < 0.5] = -1
        pi_missing = diversity.pi(self._pop_b(miss, n_var), "b")

        assert abs(pi_missing / pi_complete - 1) < 0.1

    def test_raw_sum_untouched_by_rescale(self):
        # span_normalize=False is explicitly out of scope for the rescale:
        # the raw sum (and so the 0.0 it gives over an empty analysis) is
        # unchanged, matching every other raw-sum statistic in the package.
        rng = np.random.default_rng(1)
        n_hap, n_var = 10, 200
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._pop_b(hap, n_var)
        for fn in (diversity.pi, diversity.theta_w, diversity.theta_h,
                  diversity.theta_l):
            assert np.isfinite(fn(hm, "b", span_normalize=False))

    def test_nan_when_population_has_no_call_anywhere(self):
        # Population b has zero calls at every site: no information at all,
        # so the span-normalized rate is NaN, not the old silent 0.0; the
        # raw sum stays the empty-sum 0.0.
        rng = np.random.default_rng(2)
        n_hap, n_var = 10, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-4:] = -1
        hm = self._pop_b(hap, n_var)
        for fn in (diversity.pi, diversity.theta_w, diversity.theta_h,
                  diversity.theta_l):
            assert np.isnan(fn(hm, "b"))
            assert fn(hm, "b", span_normalize=False) == 0.0

    def test_complete_data_unaffected(self):
        # No missing data: n_sites_with_data == num_variants, so the
        # rescale factor is exactly 1 and every value matches what the
        # function returned before this fix.
        rng = np.random.default_rng(3)
        n_hap, n_var = 12, 300
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._pop_b(hap, n_var)
        for fn in (diversity.pi, diversity.theta_w, diversity.theta_h,
                  diversity.theta_l):
            rescaled = fn(hm, "b")
            raw = fn(hm, "b", span_normalize=False)
            np.testing.assert_allclose(rescaled, raw / hm.get_span(), rtol=1e-12)

    def test_diversity_stats_eta_family_shares_the_rescale(self):
        # eta1/eta1_star/minus_eta1/minus_eta1_star share _compute_thetas'
        # has_data gate with pi/theta_w/theta_h/theta_l, so they pick up
        # the same fix through diversity_stats' one _apply_span_normalize
        # call site.
        rng = np.random.default_rng(4)
        n_hap, n_var = 10, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-4:] = -1
        hm = self._pop_b(hap, n_var)
        stats = diversity.diversity_stats(
            hm, "b", statistics=["eta1", "eta1_star", "minus_eta1",
                                 "minus_eta1_star"])
        assert all(np.isnan(v) for v in stats.values())


class TestDivergenceSpanNormalizedRescale:
    """dxy/da rescale their raw sum to the sites where each population has
    enough calls (one each for dxy, two each for da) before dividing by the
    span, the same fix as pi/theta_w/theta_h/theta_l but for the two-pop
    site gates in _dxy_per_site_from_counts / _site_components_from_counts."""

    def _hm(self, hap, n_var):
        hm = HaplotypeMatrix(hap, np.arange(n_var) * 10 + 1, 1, n_var * 10)
        hm.sample_sets = {"a": list(range(hap.shape[0] - 8)),
                          "b": list(range(hap.shape[0] - 8, hap.shape[0]))}
        return hm

    def test_rate_matches_complete_data_under_heavy_missingness(self):
        rng = np.random.default_rng(10)
        n_hap, n_var = 24, 4000
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm_complete = self._hm(hap.copy(), n_var)
        dxy_complete = divergence.dxy(hm_complete, "a", "b")
        da_complete = divergence.da(hm_complete, "a", "b")

        miss = hap.copy()
        miss[-8:][rng.random((8, n_var)) < 0.5] = -1
        hm_missing = self._hm(miss, n_var)
        dxy_missing = divergence.dxy(hm_missing, "a", "b")
        da_missing = divergence.da(hm_missing, "a", "b")

        assert abs(dxy_missing / dxy_complete - 1) < 0.1
        assert abs(da_missing - da_complete) < 0.01

    def test_raw_sum_untouched_by_rescale(self):
        rng = np.random.default_rng(11)
        n_hap, n_var = 16, 200
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        assert np.isfinite(divergence.dxy(hm, "a", "b", span_normalize=False))
        assert np.isfinite(divergence.da(hm, "a", "b", span_normalize=False))

    def test_nan_when_population_has_no_call_anywhere(self):
        rng = np.random.default_rng(12)
        n_hap, n_var = 16, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:] = -1
        hm = self._hm(hap, n_var)
        assert np.isnan(divergence.dxy(hm, "a", "b"))
        assert np.isnan(divergence.da(hm, "a", "b"))
        assert divergence.dxy(hm, "a", "b", span_normalize=False) == 0.0
        assert divergence.da(hm, "a", "b", span_normalize=False) == 0.0

    def test_nan_when_population_has_a_single_call(self):
        # da needs its own pair (two calls) in each population at a site;
        # exactly one call everywhere is enough for dxy but not for da.
        rng = np.random.default_rng(13)
        n_hap, n_var = 16, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:-1] = -1  # population b keeps exactly one haplotype with calls
        hm = self._hm(hap, n_var)
        assert np.isfinite(divergence.dxy(hm, "a", "b"))
        assert np.isnan(divergence.da(hm, "a", "b"))

    def test_per_site_dxy_nan_where_either_population_has_no_call(self):
        rng = np.random.default_rng(14)
        n_hap, n_var = 16, 20
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:, 5] = -1  # population b missing entirely at site 5
        hm = self._hm(hap, n_var)
        per_site = divergence.dxy(hm, "a", "b", per_site=True)
        assert np.isnan(per_site[5])
        assert np.isfinite(per_site[:5]).all() and np.isfinite(per_site[6:]).all()

    def test_complete_data_unaffected(self):
        rng = np.random.default_rng(15)
        n_hap, n_var = 20, 300
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        dxy_raw = divergence.dxy(hm, "a", "b", span_normalize=False)
        da_raw = divergence.da(hm, "a", "b", span_normalize=False)
        np.testing.assert_allclose(divergence.dxy(hm, "a", "b"),
                                   dxy_raw / hm.get_span(), rtol=1e-12)
        np.testing.assert_allclose(divergence.da(hm, "a", "b"),
                                   da_raw / hm.get_span(), rtol=1e-12)

    def test_exclude_to_empty_is_nan(self):
        rng = np.random.default_rng(16)
        n_hap, n_var = 16, 30
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[0, :] = -1  # population a missing at every site
        hm = self._hm(hap, n_var)
        assert np.isnan(divergence.dxy(hm, "a", "b", missing_data='exclude'))
        assert np.isnan(divergence.da(hm, "a", "b", missing_data='exclude'))


class TestWindowedSpanNormalizedRescale:
    """The windowed scatter engine's pi/theta_w/theta_h/theta_l pick up the
    same per-window rescale as the scalar functions above."""

    def _hm(self, hap, n_var):
        hm = HaplotypeMatrix(hap, np.arange(n_var, dtype=np.int64) * 10 + 1,
                             1, n_var * 10)
        hm.sample_sets = {"b": list(range(hap.shape[0] - 4, hap.shape[0]))}
        return hm

    def test_rate_matches_scalar_under_missingness(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(5)
        n_hap, n_var = 20, 400
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-4:][rng.random((4, n_var)) < 0.5] = -1
        hm = self._hm(hap, n_var)

        stats = ["pi", "theta_w", "theta_h", "theta_l"]
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=stats, populations=["b"])
        for stat, fn in zip(stats, (diversity.pi, diversity.theta_w,
                                    diversity.theta_h, diversity.theta_l)):
            np.testing.assert_allclose(df[stat].iloc[0], fn(hm, "b"), rtol=1e-9)

    def test_raw_sum_untouched_by_rescale(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(6)
        n_hap, n_var = 10, 200
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["pi"], populations=["b"],
                               span_normalize=False)
        np.testing.assert_allclose(
            df["pi"].iloc[0], diversity.pi(hm, "b", span_normalize=False))

    def test_nan_when_population_has_no_call_in_window(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(7)
        n_hap, n_var = 10, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-4:] = -1
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["pi", "theta_w"], populations=["b"])
        assert np.isnan(df["pi"].iloc[0]) and np.isnan(df["theta_w"].iloc[0])

    def test_complete_data_unaffected(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(8)
        n_hap, n_var = 12, 300
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = HaplotypeMatrix(hap, np.arange(n_var, dtype=np.int64) * 10 + 1,
                             1, n_var * 10)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["pi", "theta_w"])
        np.testing.assert_allclose(df["pi"].iloc[0], diversity.pi(hm), rtol=1e-9)
        np.testing.assert_allclose(df["theta_w"].iloc[0], diversity.theta_w(hm),
                                   rtol=1e-9)


class TestWindowedDivergenceSpanNormalizedRescale:
    """The windowed scatter engine's dxy/da pick up the same per-window
    rescale as the scalar functions above, each against its own site gate."""

    def _hm(self, hap, n_var):
        hm = HaplotypeMatrix(hap, np.arange(n_var, dtype=np.int64) * 10 + 1,
                             1, n_var * 10)
        hm.sample_sets = {"a": list(range(hap.shape[0] - 8)),
                          "b": list(range(hap.shape[0] - 8, hap.shape[0]))}
        return hm

    def test_rate_matches_scalar_under_missingness(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(18)
        n_hap, n_var = 24, 400
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:][rng.random((8, n_var)) < 0.5] = -1
        hm = self._hm(hap, n_var)

        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"])
        np.testing.assert_allclose(df["dxy"].iloc[0], divergence.dxy(hm, "a", "b"),
                                   rtol=1e-9)
        np.testing.assert_allclose(df["da"].iloc[0], divergence.da(hm, "a", "b"),
                                   rtol=1e-9)

    def test_raw_sum_untouched_by_rescale(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(19)
        n_hap, n_var = 16, 200
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"],
                               span_normalize=False)
        np.testing.assert_allclose(
            df["dxy"].iloc[0], divergence.dxy(hm, "a", "b", span_normalize=False))
        np.testing.assert_allclose(
            df["da"].iloc[0], divergence.da(hm, "a", "b", span_normalize=False))

    def test_nan_when_population_has_no_call_in_window(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(20)
        n_hap, n_var = 16, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:] = -1
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"])
        assert np.isnan(df["dxy"].iloc[0]) and np.isnan(df["da"].iloc[0])

    def test_nan_when_population_has_a_single_call_in_window(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(21)
        n_hap, n_var = 16, 50
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[-8:-1] = -1  # population b keeps exactly one haplotype with calls
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"])
        assert np.isfinite(df["dxy"].iloc[0])
        assert np.isnan(df["da"].iloc[0])

    def test_complete_data_unaffected(self):
        from pg_gpu.windowed_analysis import windowed_analysis
        rng = np.random.default_rng(22)
        n_hap, n_var = 20, 300
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"])
        np.testing.assert_allclose(df["dxy"].iloc[0], divergence.dxy(hm, "a", "b"),
                                   rtol=1e-9)
        np.testing.assert_allclose(df["da"].iloc[0], divergence.da(hm, "a", "b"),
                                   rtol=1e-9)

    def test_exclude_to_empty_is_zero_not_nan(self):
        # Mirrors the single-pop windowed convention: a window with literally
        # zero variants (the whole matrix excluded to empty) gives 0, not
        # NaN -- NaN is reserved for a window that has variants but no
        # population-pair overlap among them.
        from pg_gpu.windowed_analysis import windowed_analysis
        n_hap, n_var = 16, 6
        hap = np.full((n_hap, n_var), -1, dtype=np.int8)
        hm = self._hm(hap, n_var)
        df = windowed_analysis(hm, window_size=n_var * 10, step_size=n_var * 10,
                               statistics=["dxy", "da"], populations=["a", "b"],
                               missing_data='exclude')
        assert (df["n_variants"] == 0).all()
        assert (df["dxy"] == 0).all() and (df["da"] == 0).all()


class TestFusedSinglePopSpanNormalizedRescale:
    """windowed_statistics_fused/_chunked's pi/theta_w/theta_h pick up the
    same per-window rescale as the scatter engine, from the per-window
    counts (var_count, count_valid_w) the kernel already returns."""

    def _engines(self):
        from pg_gpu.windowed_analysis import (
            windowed_statistics_fused, windowed_statistics_fused_chunked)
        return [windowed_statistics_fused, windowed_statistics_fused_chunked]

    def _hm(self, hap, n_var):
        return HaplotypeMatrix(hap, np.arange(n_var, dtype=np.int64) * 10 + 1,
                               1, n_var * 10)

    def test_rate_matches_scalar_under_missingness(self):
        rng = np.random.default_rng(23)
        n_hap, n_var = 10, 200
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[rng.random((n_hap, n_var)) < 0.85] = -1
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins,
                         statistics=('pi', 'theta_w', 'theta_h'))
            np.testing.assert_allclose(out['pi'][0], diversity.pi(hm), rtol=1e-9)
            np.testing.assert_allclose(out['theta_w'][0], diversity.theta_w(hm),
                                       rtol=1e-9)
            np.testing.assert_allclose(out['theta_h'][0], diversity.theta_h(hm),
                                       rtol=1e-9)

    def test_raw_sum_untouched_by_rescale(self):
        rng = np.random.default_rng(24)
        n_hap, n_var = 10, 100
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[rng.random((n_hap, n_var)) < 0.5] = -1
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins,
                         statistics=('pi', 'theta_w', 'theta_h'),
                         per_base=False)
            np.testing.assert_allclose(
                out['pi'][0], diversity.pi(hm, span_normalize=False))
            np.testing.assert_allclose(
                out['theta_w'][0], diversity.theta_w(hm, span_normalize=False))
            np.testing.assert_allclose(
                out['theta_h'][0], diversity.theta_h(hm, span_normalize=False))

    def test_tajimas_d_and_fay_wu_h_unaffected(self):
        # tajimas_d and fay_wu_h must keep reading the raw, unrescaled sums.
        rng = np.random.default_rng(25)
        n_hap, n_var = 20, 400
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hap[rng.random((n_hap, n_var)) < 0.3] = -1
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins,
                         statistics=('pi', 'theta_w', 'theta_h', 'tajimas_d',
                                     'fay_wu_h'))
            np.testing.assert_allclose(out['tajimas_d'][0],
                                       diversity.tajimas_d(hm), rtol=1e-6)
            np.testing.assert_allclose(out['fay_wu_h'][0],
                                       diversity.fay_wus_h(hm) / hm.get_span(),
                                       rtol=1e-6)

    def test_nan_when_no_site_has_two_calls(self):
        n_hap, n_var = 6, 20
        hap = np.zeros((n_hap, n_var), dtype=np.int8)
        hap[1:, :] = -1  # only one haplotype called anywhere
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins,
                         statistics=('pi', 'theta_w', 'theta_h'))
            assert np.isnan(out['pi'][0])
            assert np.isnan(out['theta_w'][0])
            assert np.isnan(out['theta_h'][0])

    def test_zero_variant_window_is_zero_not_nan(self):
        # Mirrors the scatter engine's convention: a window with literally
        # no variants gives 0, not NaN.
        n_hap, n_var = 6, 10
        hap = np.zeros((n_hap, n_var), dtype=np.int8)
        hap[:, :] = -1
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins, statistics=('pi', 'theta_w'),
                         missing_data='exclude')
            assert out['n_variants'][0] == 0
            assert out['pi'][0] == 0.0
            assert out['theta_w'][0] == 0.0

    def test_complete_data_unaffected(self):
        rng = np.random.default_rng(26)
        n_hap, n_var = 12, 150
        hap = rng.integers(0, 2, (n_hap, n_var)).astype(np.int8)
        hm = self._hm(hap, n_var)
        bp_bins = np.array([1, n_var * 10 + 1])
        for engine in self._engines():
            out = engine(hm, bp_bins=bp_bins,
                         statistics=('pi', 'theta_w', 'theta_h'))
            np.testing.assert_allclose(out['pi'][0], diversity.pi(hm), rtol=1e-9)
            np.testing.assert_allclose(out['theta_w'][0], diversity.theta_w(hm),
                                       rtol=1e-9)
            np.testing.assert_allclose(out['theta_h'][0], diversity.theta_h(hm),
                                       rtol=1e-9)


class TestExcludeKeepsMatrixContext:
    """exclude_missing_sites keeps the parent chromosome bounds and
    accessibility mask, like the restrict_to_* filters, so exclude-mode
    span normalization divides by the chromosome, not the surviving
    variant hull."""

    def _hm(self):
        rng = np.random.default_rng(17)
        hap = rng.integers(0, 2, (10, 50), dtype=np.int8)
        hap[0, 0] = -1            # first site incomplete (row in pop a)
        hap[3, 49] = -1           # last site incomplete (row in pop a)
        hap[8, 20] = -1           # mid site incomplete only in pop b
        pos = (np.arange(50) * 190 + 200).astype(np.int64)  # 200..9510
        return HaplotypeMatrix(hap, pos, 0, 10000)

    def test_bounds_preserved(self):
        h = self._hm()
        ex = h.exclude_missing_sites()
        assert ex.num_variants == 47
        assert (ex.chrom_start, ex.chrom_end) == (0, 10000)

    def test_bounds_preserved_with_populations(self):
        h = self._hm()
        h.sample_sets = {"a": list(range(6)), "b": list(range(6, 10))}
        ex = h.exclude_missing_sites(populations=["a"])
        assert ex.num_variants == 48   # only rows 0-5 checked
        assert (ex.chrom_start, ex.chrom_end) == (0, 10000)

    def test_mask_preserved(self):
        from pg_gpu.accessible import AccessibleMask
        h = self._hm()
        h.set_accessible_mask(AccessibleMask(np.ones(10001, dtype=bool)))
        ex = h.exclude_missing_sites()
        assert ex.has_accessible_mask

    def test_filter_by_missing_keeps_context(self):
        h = self._hm()
        f = h.filter_variants_by_missing(max_missing_freq=0.05)
        assert f.num_variants == 47
        assert (f.chrom_start, f.chrom_end) == (0, 10000)

    def test_exclude_pi_uses_parent_span(self):
        from pg_gpu import diversity
        h = self._hm()
        raw = diversity.pi(h, missing_data='exclude', span_normalize=False)
        per_bp = diversity.pi(h, missing_data='exclude', span_normalize=True)
        np.testing.assert_allclose(per_bp, raw / 10001, rtol=1e-12)
