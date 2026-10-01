"""Neutrality tests and the frequency spectrum at their edges.

Few segregating sites leave the variance-based tests undefined; a statistic
that needs no variance must still be defined, and one built on an undefined
test must be undefined too. The frequency spectrum must check custom weights
and report what a projection leaves out. Windowed Fay & Wu's H must be in the
same units whichever engine computes it.
"""
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, diversity, windowed_analysis
from pg_gpu.diversity import FrequencySpectrum

from .conftest import simulate_hm


@pytest.fixture
def two_site_hm():
    # 5 haplotypes, 2 segregating sites: Tajima's D is undefined (S < 3).
    hap = np.array([[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.int8)
    return HaplotypeMatrix(hap, np.array([100, 200], dtype=np.int64), 0, 1000)


def test_diversity_stats_fay_wus_h_needs_no_minimum_sites(two_site_hm):
    got = diversity.diversity_stats(two_site_hm, statistics=['fay_wus_h'])
    assert got['fay_wus_h'] == pytest.approx(diversity.fay_wus_h(two_site_hm))
    assert np.isfinite(got['fay_wus_h'])


def test_zeng_dh_is_nan_when_tajimas_d_is_undefined(two_site_hm):
    assert np.isnan(diversity.tajimas_d(two_site_hm))
    assert np.isnan(diversity.zeng_dh(two_site_hm))


def test_windowed_zeng_dh_is_nan_in_undefined_windows():
    hm = simulate_hm(n_samples=20, seq_length=50_000, seed=5,
                     mutation_model='binary')
    df = windowed_analysis(hm, window_size=200, step_size=200,
                           statistics=['tajimas_d', 'zeng_dh'])
    undefined = df['tajimas_d'].isna().to_numpy()
    assert undefined.any() and (~undefined).any()
    assert df['zeng_dh'].isna().to_numpy()[undefined].all()
    assert np.isfinite(df['zeng_dh'].to_numpy()[~undefined]).all()


def test_custom_weights_must_cover_every_bin(two_site_hm):
    fs = FrequencySpectrum(two_site_hm)
    with pytest.raises(ValueError, match="length"):
        fs.theta(weights=lambda n: np.ones(n))


def test_projection_warns_about_dropped_groups():
    # Sites with 10 valid haplotypes and sites with 6: projecting to 8 can
    # keep only the first group.
    rng = np.random.default_rng(2)
    hap = rng.integers(0, 2, size=(10, 40)).astype(np.int8)
    hap[6:, 20:] = -1
    hm = HaplotypeMatrix(hap, np.arange(40) * 10 + 1, 1, 400)
    fs = FrequencySpectrum(hm)
    dropped = int(np.sum(fs.sfs_by_n[6][1:6]))
    assert dropped > 0
    with pytest.warns(UserWarning, match=f"{dropped} segregating"):
        fs.project(8)


def test_windowed_fay_wu_h_units_do_not_depend_on_the_engine():
    # A Garud statistic sends the request to the fused engine; alone,
    # fay_wu_h takes the scatter engine. Both must be per base.
    hm = simulate_hm(n_samples=20, seq_length=50_000, seed=3,
                     mutation_model='binary')
    kw = dict(window_size=10_000, step_size=10_000)
    alone = windowed_analysis(hm, statistics=['fay_wu_h'], **kw)
    fused = windowed_analysis(hm, statistics=['fay_wu_h', 'garud_h1'], **kw)
    np.testing.assert_allclose(fused['fay_wu_h'], alone['fay_wu_h'],
                               rtol=1e-9, equal_nan=True)
