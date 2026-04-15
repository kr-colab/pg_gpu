"""
Tests for pg_gpu.resampling: block_jackknife and block_bootstrap.
"""

import numpy as np
import pytest
from scipy import stats

from pg_gpu.resampling import block_jackknife, block_bootstrap


# -----------------------------------------------------------------------------
# block_jackknife
# -----------------------------------------------------------------------------


class TestBlockJackknife:
    def test_mean_closed_form_se(self):
        """For statistic=mean, jackknife SE equals sample_std / sqrt(n)."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        est, se, per_iter = block_jackknife(x, statistic=np.mean)

        assert per_iter.shape == (200,)
        # Unweighted point estimate is mean of leave-one-out means, which
        # equals the full-sample mean for a linear statistic.
        assert est == pytest.approx(x.mean(), abs=1e-12)
        # Closed-form: delete-1 jackknife SE of the mean is s/sqrt(n) with
        # the ((n-1)/n) * sum((...)**2) formula → equals sample-std / sqrt(n)
        # with the usual n-1 sample-std denominator.
        expected_se = x.std(ddof=1) / np.sqrt(len(x))
        assert se == pytest.approx(expected_se, rel=1e-10)

    def test_tuple_ratio_matches_handwritten(self):
        """Tuple-input ratio-of-sums matches an explicit delete-1 loop."""
        rng = np.random.default_rng(0)
        num = rng.normal(1.0, 0.5, size=30)
        den = rng.normal(2.0, 0.5, size=30)

        def stat(a, b):
            return np.sum(a) / np.sum(b)
        est, se, per_iter = block_jackknife((num, den), statistic=stat)

        # Reference: explicit leave-one-out loop
        vj_ref = np.array([stat(np.delete(num, i), np.delete(den, i))
                           for i in range(len(num))])
        assert np.allclose(per_iter, vj_ref)
        n = len(num)
        se_ref = np.sqrt(((n - 1) / n) * np.sum((vj_ref - vj_ref.mean()) ** 2))
        assert se == pytest.approx(se_ref, rel=1e-12)
        assert est == pytest.approx(vj_ref.mean(), abs=1e-12)

    def test_weighted_reduces_to_unweighted_when_uniform(self):
        """Uniform weights must give identical SE to unweighted case."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        _, se_u, _ = block_jackknife(x, statistic=np.mean)
        _, se_w, _ = block_jackknife(x, statistic=np.mean,
                                     weights=np.full(40, 250.0))
        assert se_w == pytest.approx(se_u, rel=1e-12)

    def test_weighted_busing_handcomputed(self):
        """Weighted jackknife SE against an explicit Busing 1999 implementation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([10.0, 20.0, 10.0, 40.0, 20.0], dtype=np.float64)

        est, se, per_iter = block_jackknife(x, statistic=np.sum, weights=w)

        # Reference: explicit Busing 1999 pseudo-values + weighted variance.
        g = len(x)
        theta_hat = x.sum()
        theta_neg = np.array([np.delete(x, i).sum() for i in range(g)])
        h = w.sum() / w
        phi = h * theta_hat - (h - 1.0) * theta_neg
        est_ref = phi.mean()
        var_ref = np.mean((phi - est_ref) ** 2 / (h - 1.0))
        se_ref = np.sqrt(var_ref)

        assert np.allclose(per_iter, theta_neg)
        assert est == pytest.approx(est_ref, rel=1e-12)
        assert se == pytest.approx(se_ref, rel=1e-12)

    def test_weights_validated(self):
        with pytest.raises(ValueError):
            block_jackknife(np.arange(5.0), statistic=np.mean,
                            weights=np.ones(4))
        with pytest.raises(ValueError):
            block_jackknife(np.arange(5.0), statistic=np.mean,
                            weights=np.array([1.0, 1.0, 0.0, 1.0, 1.0]))

    def test_tuple_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            block_jackknife((np.arange(5.0), np.arange(6.0)),
                            statistic=lambda a, b: a.sum() / b.sum())


# -----------------------------------------------------------------------------
# block_bootstrap
# -----------------------------------------------------------------------------


class TestBlockBootstrap:
    def test_replicates_shape_and_estimate(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=50)

        est, se, reps = block_bootstrap(x, statistic=np.mean,
                                        n_replicates=500, rng=0)
        # Plug-in estimate equals the sample mean exactly — not the
        # replicate mean.
        assert est == pytest.approx(x.mean(), abs=1e-12)
        assert reps.shape == (500,)
        assert se > 0

    def test_seed_reproducibility(self):
        x = np.arange(20.0)
        _, se1, reps1 = block_bootstrap(x, statistic=np.mean,
                                        n_replicates=200, rng=123)
        _, se2, reps2 = block_bootstrap(x, statistic=np.mean,
                                        n_replicates=200, rng=123)
        assert se1 == se2
        assert np.array_equal(reps1, reps2)

    def test_agreement_with_scipy_bootstrap(self):
        """SE should agree with scipy.stats.bootstrap on a simple mean."""
        rng = np.random.default_rng(2024)
        x = rng.normal(size=100)
        _, se_ours, _ = block_bootstrap(x, statistic=np.mean,
                                        n_replicates=4000, rng=1)
        res = stats.bootstrap(
            (x,), statistic=np.mean, n_resamples=4000,
            method="percentile",
            random_state=np.random.default_rng(1),
        )
        assert se_ours == pytest.approx(res.standard_error, rel=0.1)

    def test_tuple_same_indices_invariant(self):
        """Same block indices apply across tuple entries.

        If resampling were independent across entries, a statistic that
        consumes (a, b) as paired blocks would not reproduce any relation
        between them. Here we use b = 2 * a, so every replicate of
        np.sum(b)/np.sum(a) must equal exactly 2 regardless of the draw.
        """
        a = np.arange(1.0, 21.0)
        b = 2.0 * a
        est, se, reps = block_bootstrap(
            (a, b), statistic=lambda x, y: np.sum(y) / np.sum(x),
            n_replicates=300, rng=42,
        )
        assert est == pytest.approx(2.0)
        assert np.allclose(reps, 2.0)
        # With perfectly collinear entries, bootstrap SE is exactly zero.
        assert se == pytest.approx(0.0, abs=1e-12)

    def test_patterson_d_style_ratio_bootstrap(self):
        """Ratio-of-sums bootstrap SE is in the same ballpark as jackknife SE."""
        rng = np.random.default_rng(9)
        num = rng.normal(0.1, 0.05, size=60)
        den = rng.normal(0.3, 0.05, size=60)

        def stat(n, d):
            return np.sum(n) / np.sum(d)

        _, se_jk, _ = block_jackknife((num, den), statistic=stat)
        _, se_bs, _ = block_bootstrap((num, den), statistic=stat,
                                      n_replicates=2000, rng=0)
        # Not equal — but should agree to within a factor of ~1.5-2.
        assert 0.5 * se_jk < se_bs < 2.0 * se_jk
