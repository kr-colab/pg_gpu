"""
Block-resampling utilities for calibrated SE / CI on genome-wide statistics.

This module provides general-purpose resampling estimators that operate on
pre-binned per-block values:

* :func:`block_jackknife` — delete-1 block jackknife SE, including the
  Busing et al. (1999) weighted variant for unequal block sizes.
* :func:`block_bootstrap` — block bootstrap: resample blocks with replacement
  to obtain an empirical SE and replicate distribution.

Both accept either a single 1D array of per-block values or a tuple of
1D arrays (one entry per block), plus a callable ``statistic`` that maps
the block values to a scalar. The tuple form is intended for ratio-of-sums
estimators (normed F3, Patterson D, windowed Tajima's D means, etc.), where
the numerator and denominator share the same block indexing.

References
----------
Busing, F. M. T. A., Meijer, E., & Van der Leeden, R. (1999). Delete-m
jackknife for unequal m. *Statistics and Computing*, 9, 3-8.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*,
Chapter 6. Chapman & Hall/CRC.

Notes
-----
Future work: delete-m jackknife (contiguous block deletion for m > 1),
jackknife-after-bootstrap, and BCa confidence intervals. Input block values
are expected on host (NumPy); the companion CuPy helpers
``_moving_nansum`` / ``_moving_nanmean`` in this module produce them from
per-variant arrays.
"""

import numpy as np
import cupy as cp


def _moving_nansum(values, size, start=0, stop=None, step=None):
    """Windowed nansum on GPU via cumulative sums.

    Parameters
    ----------
    values : cupy.ndarray, shape (n,)
    size, start, stop, step : int

    Returns
    -------
    cupy.ndarray, shape (n_windows,)
    """
    n = len(values)
    if stop is None:
        stop = n
    if step is None:
        step = size

    v = cp.where(cp.isfinite(values), values, 0.0)
    cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), cp.cumsum(v)])
    w_starts = cp.arange(start, stop - size + 1, step)
    return cs[w_starts + size] - cs[w_starts]


def _moving_nanmean(values, size, start=0, stop=None, step=None):
    """Windowed nanmean on GPU via cumulative sums.

    Parameters
    ----------
    values : cupy.ndarray, shape (n,)
    size, start, stop, step : int

    Returns
    -------
    cupy.ndarray, shape (n_windows,)
    """
    n = len(values)
    if stop is None:
        stop = n
    if step is None:
        step = size

    finite = cp.isfinite(values)
    v = cp.where(finite, values, 0.0)
    cs = cp.concatenate([cp.zeros(1, dtype=cp.float64), cp.cumsum(v)])
    cn = cp.concatenate([cp.zeros(1, dtype=cp.float64),
                         cp.cumsum(finite.astype(cp.float64))])
    w_starts = cp.arange(start, stop - size + 1, step)
    sums = cs[w_starts + size] - cs[w_starts]
    counts = cn[w_starts + size] - cn[w_starts]
    return cp.where(counts > 0, sums / counts, cp.nan)


def _as_tuple(values):
    """Normalise a single array or tuple of arrays to a tuple."""
    if isinstance(values, tuple):
        arrays = tuple(np.asarray(v) for v in values)
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError(
                    "all arrays in `values` tuple must have the same length"
                )
        return arrays, n, True
    arr = np.asarray(values)
    return (arr,), len(arr), False


def block_jackknife(values, statistic, *, weights=None):
    """Block-jackknife standard error for a statistic over pre-binned blocks.

    Implements the delete-1 block jackknife: for each block ``j``, the
    statistic is recomputed on the remaining blocks; the spread of those
    leave-one-out estimates gives a calibrated SE. When ``weights`` is
    provided, the Busing et al. (1999) weighted delete-:math:`m_j`
    formulation is used so unequal block sizes do not skew the SE.

    Parameters
    ----------
    values : ndarray or tuple of ndarrays
        Per-block values. Use a single 1D array for statistics of the form
        ``statistic(vb)``; use a tuple ``(num, den, ...)`` for ratio-of-sums
        statistics where multiple per-block arrays are consumed by
        ``statistic``. All arrays in the tuple must have the same length.
    statistic : callable
        ``statistic(*values) -> float``. Called once on the full data and
        once per block with that block masked out.
    weights : ndarray, optional
        Length-``n`` array of block sizes (e.g., SNPs per block). When
        ``None`` (default), blocks are treated as equally weighted and the
        formula reduces to the unweighted block jackknife. See Notes.

    Returns
    -------
    estimate : float
        Point estimate. Unweighted case: mean of leave-one-out values
        (matches the current admixture jackknife). Weighted case: mean of
        the Busing pseudo-values.
    se : float
        Jackknife standard error.
    per_iter : ndarray, shape (n,)
        Per-block leave-one-out statistic values.

    Notes
    -----
    For uniform block sizes the weighted formulation reduces exactly to
    ``sqrt(((n-1)/n) * sum((vj - mean(vj))**2))``.

    Weighted formulation (Busing 1999 eq. 4-5):

    .. math::

        h_j = N / m_j, \\qquad \\phi_j = h_j \\hat\\theta - (h_j - 1) \\hat\\theta_{-j}

        \\tilde\\theta = \\frac{1}{g} \\sum_j \\phi_j, \\qquad
        \\mathrm{Var}(\\tilde\\theta) = \\frac{1}{g} \\sum_j
            \\frac{(\\phi_j - \\tilde\\theta)^2}{h_j - 1}

    where ``g`` is the number of blocks, ``m_j`` is the size of block ``j``,
    and ``N = sum(m_j)``.

    References
    ----------
    Busing, F. M. T. A., Meijer, E., & Van der Leeden, R. (1999). Delete-m
    jackknife for unequal m. *Statistics and Computing*, 9, 3-8.
    """
    arrays, n, is_tuple = _as_tuple(values)

    def _eval(mask):
        if is_tuple:
            masked = [a[mask] for a in arrays]
            return float(statistic(*masked))
        return float(statistic(arrays[0][mask]))

    keep = np.ones(n, dtype=bool)
    per_iter = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep[i] = False
        per_iter[i] = _eval(keep)
        keep[i] = True

    if weights is None:
        m = per_iter.mean()
        sv = ((n - 1) / n) * np.sum((per_iter - m) ** 2)
        return m, float(np.sqrt(sv)), per_iter

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(
            f"`weights` must have length {n}; got shape {w.shape}"
        )
    if np.any(w <= 0):
        raise ValueError("`weights` must be strictly positive")

    theta_hat = _eval(np.ones(n, dtype=bool))
    h = w.sum() / w
    phi = h * theta_hat - (h - 1.0) * per_iter
    estimate = float(phi.mean())
    var = float(np.mean((phi - estimate) ** 2 / (h - 1.0)))
    return estimate, float(np.sqrt(var)), per_iter


def block_bootstrap(values, statistic, *, n_replicates=1000, rng=None):
    """Block-bootstrap distribution for a statistic over pre-binned blocks.

    Resamples block indices with replacement ``n_replicates`` times and
    evaluates ``statistic`` on each resample. When ``values`` is a tuple,
    the **same** sampled indices are applied to every array — required for
    ratio-of-sums statistics where numerator and denominator share block
    indexing (e.g. normed Patterson F3, Patterson D). To bootstrap two
    *independent* arrays, call ``block_bootstrap`` on each separately.

    Parameters
    ----------
    values : ndarray or tuple of ndarrays
        Per-block values. Tuple entries must have the same length.
    statistic : callable
        ``statistic(*values) -> float``.
    n_replicates : int, default 1000
        Number of bootstrap replicates.
    rng : int, numpy.random.Generator, or None
        Seed or Generator for reproducibility. ``None`` uses fresh entropy.

    Returns
    -------
    estimate : float
        Plug-in point estimate ``statistic(*values)`` on the full data.
        This follows Efron & Tibshirani (1993) §6.2: the bootstrap mean is
        a noisy estimate of the plug-in estimate, not a replacement for it.
    se : float
        Bootstrap standard error, ``std(replicates, ddof=1)``.
    replicates : ndarray, shape (n_replicates,)
        Bootstrap replicate values. Use e.g. ``np.quantile(replicates,
        [0.025, 0.975])`` for a percentile 95% CI.

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the
    Bootstrap*. Chapman & Hall/CRC.
    """
    arrays, n, is_tuple = _as_tuple(values)
    gen = np.random.default_rng(rng)

    if is_tuple:
        estimate = float(statistic(*arrays))
    else:
        estimate = float(statistic(arrays[0]))

    replicates = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        idx = gen.integers(0, n, size=n)
        if is_tuple:
            replicates[r] = float(statistic(*(a[idx] for a in arrays)))
        else:
            replicates[r] = float(statistic(arrays[0][idx]))

    se = float(replicates.std(ddof=1)) if n_replicates > 1 else float("nan")
    return estimate, se, replicates
