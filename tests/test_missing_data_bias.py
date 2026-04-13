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
