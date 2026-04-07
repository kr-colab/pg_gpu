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
from pg_gpu.achaz import FrequencySpectrum

N_REPS = 50
N_HAPS_PER_POP = 100
SEQ_LENGTH = 200_000
MISSING_RATES = [0.0, 0.10, 0.30, 0.60]
BIAS_TOLERANCE = 0.05  # 5% relative bias


# ── Fixtures ────────────────────────────────────────────────────────


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
        random_seed=seed,
        ploidy=2,
    )
    return msprime.sim_mutations(ts, rate=1e-8, random_seed=seed)


def _ts_to_hm(ts):
    """Convert tree sequence to HaplotypeMatrix with populations."""
    hm = HaplotypeMatrix.from_ts(ts)
    hm.sample_sets = {
        "pop1": list(range(0, N_HAPS_PER_POP)),
        "pop2": list(range(N_HAPS_PER_POP, 2 * N_HAPS_PER_POP)),
    }
    return hm


def _add_missing(hm, rate, rng):
    """Add MCAR missing data."""
    hap = hm.haplotypes
    if hasattr(hap, 'get'):
        hap = hap.get()
    hap = hap.copy()
    hap[rng.random(hap.shape) < rate] = -1
    hm_miss = HaplotypeMatrix(
        hap, hm.positions, hm.chrom_start, hm.chrom_end)
    hm_miss.sample_sets = hm.sample_sets
    return hm_miss


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

    def _achaz_theta(name):
        return lambda hm: FrequencySpectrum(
            hm, population="pop1").theta(name)

    stats = {
        # ── Diversity (single-pop, span_normalize=False for raw sums) ──
        "pi": (_pop1(diversity.pi, span_normalize=False), True),
        "theta_w": (_pop1(diversity.theta_w, span_normalize=False), True),
        "theta_h": (_pop1(diversity.theta_h, span_normalize=False), True),
        "theta_l": (_pop1(diversity.theta_l, span_normalize=False), True),
        "tajd": (_pop1(diversity.tajimas_d), True),
        "fay_wus_h": (_pop1(diversity.fay_wus_h), True),
        "norm_fay_wus_h": (_pop1(diversity.normalized_fay_wus_h), True),
        "zeng_e": (_pop1(diversity.zeng_e), True),
        "seg_sites": (_pop1(diversity.segregating_sites), True),
        "het_exp": (_pop1(diversity.heterozygosity_expected), True),
        # ── Divergence (two-pop) ──
        "dxy": (_twopop(divergence.dxy), False),
        "fst_hudson": (_twopop(divergence.fst_hudson), False),
        "fst_wc": (_twopop(divergence.fst_weir_cockerham), False),
        "da": (_twopop(divergence.da), False),
        # ── Achaz SFS framework ──
        "achaz_pi": (_achaz_theta("pi"), True),
        "achaz_watterson": (_achaz_theta("watterson"), True),
        "achaz_theta_h": (_achaz_theta("theta_h"), True),
        "achaz_theta_l": (_achaz_theta("theta_l"), True),
        "achaz_tajd": (lambda hm: FrequencySpectrum(
            hm, population="pop1").tajimas_d(), True),
        "achaz_zeng_e": (lambda hm: FrequencySpectrum(
            hm, population="pop1").zeng_e(), True),
    }

    results = {name: {r: [] for r in MISSING_RATES}
               for name in stats}

    for rep in range(N_REPS):
        seed = rep + 1
        rng = np.random.default_rng(seed + 10000)

        ts = _simulate(seed)
        hm_clean = _ts_to_hm(ts)
        if hm_clean.num_variants < 10:
            continue
        hm_clean.transfer_to_gpu()

        for rate in MISSING_RATES:
            if rate == 0.0:
                hm = hm_clean
            else:
                hm = _add_missing(hm_clean, rate, rng)
                hm.transfer_to_gpu()

            for name, (fn, _) in stats.items():
                try:
                    val = fn(hm)
                    if np.isfinite(val):
                        results[name][rate].append(val)
                except Exception:
                    pass

    return results


# ── Tests ───────────────────────────────────────────────────────────


def _check_bias(results, stat_name, miss_rate, tolerance=BIAS_TOLERANCE):
    """Check that mean estimate at miss_rate is within tolerance of truth."""
    truth_vals = results[stat_name][0.0]
    test_vals = results[stat_name][miss_rate]
    assert len(truth_vals) >= 20, f"Too few truth replicates for {stat_name}"
    assert len(test_vals) >= 20, f"Too few test replicates for {stat_name}"
    truth_mean = np.mean(truth_vals)
    test_mean = np.mean(test_vals)
    if abs(truth_mean) < 1e-15:
        return  # skip if truth is ~0 (e.g., Tajima's D near neutrality)
    rel_bias = abs(test_mean / truth_mean - 1)
    assert rel_bias < tolerance, (
        f"{stat_name} at {miss_rate*100:.0f}% missing: "
        f"bias={rel_bias*100:.1f}% (truth={truth_mean:.6f}, "
        f"est={test_mean:.6f})")


def _check_near_zero_bias(results, stat_name, miss_rate, abs_tol=0.3):
    """For statistics near zero (neutrality tests), use absolute tolerance."""
    truth = results[stat_name][0.0]
    test = results[stat_name][miss_rate]
    if len(truth) < 20 or len(test) < 20:
        pytest.skip(f"Too few replicates for {stat_name}")
    diff = abs(np.mean(test) - np.mean(truth))
    assert diff < abs_tol, (
        f"{stat_name} at {miss_rate*100:.0f}% missing: "
        f"diff={diff:.3f} (truth={np.mean(truth):.3f}, "
        f"est={np.mean(test):.3f})")


# Statistics near zero under neutrality (use absolute tolerance)
NEAR_ZERO_STATS = ["tajd", "achaz_tajd", "achaz_zeng_e"]


@pytest.mark.slow
class TestIncludeModeUnbiased:
    """Verify include mode is unbiased under MCAR."""

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    @pytest.mark.parametrize("stat", [
        "pi", "theta_w", "theta_h", "theta_l",
    ])
    def test_diversity(self, simulation_results, stat, miss_rate):
        _check_bias(simulation_results, stat, miss_rate)

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    def test_tajimas_d(self, simulation_results, miss_rate):
        _check_near_zero_bias(simulation_results, "tajd", miss_rate)

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    @pytest.mark.parametrize("stat", ["dxy", "fst_hudson", "da"])
    def test_divergence(self, simulation_results, stat, miss_rate):
        _check_bias(simulation_results, stat, miss_rate)


@pytest.mark.slow
class TestAchazUnbiased:
    """Verify Achaz SFS framework is unbiased under MCAR."""

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    @pytest.mark.parametrize("stat", [
        "achaz_pi", "achaz_watterson", "achaz_theta_h", "achaz_theta_l",
    ])
    def test_achaz_theta(self, simulation_results, stat, miss_rate):
        _check_bias(simulation_results, stat, miss_rate)

    @pytest.mark.parametrize("miss_rate", [0.10, 0.30, 0.60])
    @pytest.mark.parametrize("stat", ["achaz_tajd", "achaz_zeng_e"])
    def test_achaz_neutrality(self, simulation_results, stat, miss_rate):
        _check_near_zero_bias(simulation_results, stat, miss_rate)


@pytest.mark.slow
class TestKnownBiasedStats:
    """Document statistics that ARE biased under MCAR missing data.

    These tests verify the bias exists (not that it's fixed). They serve
    as documentation and regression markers for future improvements.
    """

    @pytest.mark.parametrize("miss_rate", [0.30, 0.60])
    def test_fst_wc_biased(self, simulation_results, miss_rate):
        """Weir-Cockerham FST is biased upward with missing data."""
        truth = np.mean(simulation_results["fst_wc"][0.0])
        est = np.mean(simulation_results["fst_wc"][miss_rate])
        # WC FST inflates with missing data — verify bias > 10%
        assert est > truth * 1.1, (
            f"Expected upward bias in WC FST at {miss_rate*100:.0f}% "
            f"missing but got truth={truth:.4f}, est={est:.4f}")

    @pytest.mark.parametrize("miss_rate", [0.30, 0.60])
    def test_seg_sites_biased(self, simulation_results, miss_rate):
        """Segregating sites decreases with missing data (some sites
        appear monomorphic when alleles are masked)."""
        truth = np.mean(simulation_results["seg_sites"][0.0])
        est = np.mean(simulation_results["seg_sites"][miss_rate])
        assert est < truth * 0.99, (
            f"Expected seg_sites to decrease with missing data but got "
            f"truth={truth:.0f}, est={est:.0f}")


@pytest.mark.slow
class TestExcludeConsistency:
    """Verify exclude mode matches include on complete data."""

    def test_pi_exclude_equals_include(self, simulation_results):
        """On clean data, exclude and include should give same pi."""
        # Run a single rep with no missing data both ways
        ts = _simulate(seed=999)
        hm = _ts_to_hm(ts)
        hm.transfer_to_gpu()

        pi_inc = diversity.pi(hm, population="pop1",
                              missing_data='include', span_normalize=False)
        pi_exc = diversity.pi(hm, population="pop1",
                              missing_data='exclude', span_normalize=False)
        assert abs(pi_inc - pi_exc) < 1e-10, (
            f"include={pi_inc}, exclude={pi_exc}")

    def test_dxy_exclude_equals_include(self, simulation_results):
        ts = _simulate(seed=999)
        hm = _ts_to_hm(ts)
        hm.transfer_to_gpu()

        dxy_inc = divergence.dxy(hm, "pop1", "pop2",
                                 missing_data='include')
        dxy_exc = divergence.dxy(hm, "pop1", "pop2",
                                 missing_data='exclude')
        assert abs(dxy_inc - dxy_exc) < 1e-10, (
            f"include={dxy_inc}, exclude={dxy_exc}")
