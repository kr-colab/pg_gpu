"""Scalar statistics that are undefined for too few sites or haplotypes.

Each case here has no value to report (an empty mean, a ratio of 0/0, or
n / (n - 1) at n = 1), so the statistic must return NaN; 0.0 would read as
"no LD", "no sweep signal" or "no diversity".
"""
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, diversity, divergence, ld_statistics
from pg_gpu.windowed_analysis import windowed_analysis


def _hm(hap, spacing=100):
    hap = np.asarray(hap, dtype=np.int8)
    pos = np.arange(hap.shape[1]) * spacing + 1
    return HaplotypeMatrix(hap, pos, 1, hap.shape[1] * spacing)


MONOMORPHIC = np.zeros((10, 6), dtype=np.int8)
ALL_MISSING = np.full((10, 6), -1, dtype=np.int8)


@pytest.mark.parametrize("estimator", ["sigma_d2", "r2", "rogers_huff"])
@pytest.mark.parametrize("hap", [MONOMORPHIC, ALL_MISSING], ids=["monomorphic", "all_missing"])
def test_zns_and_omega_nan_without_defined_sites(hap, estimator):
    hm = _hm(hap)
    assert np.isnan(ld_statistics.zns(hm, estimator=estimator))
    assert np.isnan(ld_statistics.omega(hm, estimator=estimator))


def test_zns_nan_on_undefined_r2_matrix():
    r2 = np.full((4, 4), np.nan)
    assert np.isnan(ld_statistics.zns(r2))


def test_omega_nan_below_five_sites():
    rng = np.random.default_rng(1)
    hm = _hm(rng.integers(0, 2, (20, 4)))
    assert np.isfinite(ld_statistics.zns(hm))
    assert np.isnan(ld_statistics.omega(hm))


@pytest.mark.parametrize("hap,missing_data", [
    (MONOMORPHIC, "include"),
    (ALL_MISSING, "include"),
    ([[0, 1], [1, -1], [-1, 0], [1, 1]], "exclude"),
], ids=["monomorphic", "all_missing", "exclude_no_complete_site"])
def test_mu_sfs_nan_without_segregating_site(hap, missing_data):
    assert np.isnan(diversity.mu_sfs(_hm(hap), missing_data=missing_data))


def test_mu_var_nan_for_zero_length_window():
    hm = _hm([[0, 1], [1, 0]])
    assert np.isnan(diversity.mu_var(hm, window_length=0))
    assert diversity.mu_var(hm, window_length=100) == pytest.approx(0.02)


def test_haplotype_diversity_nan_for_one_haplotype():
    hm = _hm([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert np.isnan(diversity.haplotype_diversity(hm, population=[0]))
    assert diversity.haplotype_diversity(hm, population=[0, 1]) == pytest.approx(1.0)


def test_zx_nan_when_a_population_has_no_ld_sites():
    rng = np.random.default_rng(2)
    hap = rng.integers(0, 2, (20, 30)).astype(np.int8)
    hap[:10] = 0
    hm = _hm(hap)
    hm.sample_sets = {"a": list(range(10)), "b": list(range(10, 20))}
    assert np.isnan(divergence.zx(hm, "a", "b"))


_SCALARS = {"zns": ld_statistics.zns, "omega": ld_statistics.omega,
            "mu_ld": ld_statistics.mu_ld, "mu_sfs": diversity.mu_sfs}


def test_windowed_monomorphic_window_is_nan():
    # Window 0 is monomorphic; window 1 segregates. The monomorphic window
    # must be NaN, not 0.0, and the other must match the scalar.
    rng = np.random.default_rng(3)
    hap = np.zeros((20, 40), dtype=np.int8)
    hap[:, 20:] = rng.integers(0, 2, (20, 20))
    df = windowed_analysis(_hm(hap), window_size=2000, step_size=2000,
                           statistics=["zns", "omega", "mu_sfs"])
    second = _hm(hap[:, 20:])
    for stat in ("zns", "omega", "mu_sfs"):
        assert np.isnan(df[stat].iloc[0]), stat
        assert df[stat].iloc[1] == pytest.approx(_SCALARS[stat](second)), stat


@pytest.mark.parametrize("n_var", [1, 2, 3])
def test_windowed_small_window_matches_scalar(n_var):
    # A window with fewer variants than omega needs, or fewer than the
    # distance moments need, takes each statistic's own scalar value.
    rng = np.random.default_rng(4)
    hap = rng.integers(0, 2, (20, n_var)).astype(np.int8)
    hm = _hm(hap)
    df = windowed_analysis(hm, window_size=n_var * 100, step_size=n_var * 100,
                           statistics=list(_SCALARS))
    for stat, fn in _SCALARS.items():
        np.testing.assert_allclose(df[stat].iloc[0], fn(hm), rtol=1e-12,
                                   equal_nan=True, err_msg=stat)
