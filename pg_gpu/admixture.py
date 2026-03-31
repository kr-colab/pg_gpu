"""
GPU-accelerated admixture and Patterson F-statistics.

This module provides functions for computing Patterson's F2, F3, and D (F4)
statistics, including windowed and block-jackknife variants.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix


def _allele_freq(haplotype_matrix, missing_data='include'):
    """Compute alternate allele frequency only (no heterozygosity)."""
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes

    if missing_data == 'exclude':
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        n1 = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
        freq = cp.where(n_valid > 0, n1 / n_valid, cp.nan)
        # mark sites with any missing as NaN
        has_missing = cp.any(hap < 0, axis=0)
        freq[has_missing] = cp.nan
        return freq
    else:
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        n1 = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
        return cp.where(n_valid > 0, n1 / n_valid, 0.0)


def _allele_freq_and_het(haplotype_matrix, missing_data='include'):
    """Compute alternate allele frequency and unbiased heterozygosity.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    missing_data : str

    Returns
    -------
    freq : cupy.ndarray, float64, shape (n_variants,)
        Alternate allele frequency.
    h : cupy.ndarray, float64, shape (n_variants,)
        Unbiased heterozygosity estimator: n0*n1 / (n*(n-1)).
    n : cupy.ndarray, float64, shape (n_variants,)
        Allele number per site (n_valid).
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes  # (n_haplotypes, n_variants)

    valid_mask = hap >= 0
    an = cp.sum(valid_mask, axis=0).astype(cp.float64)  # per-site n
    n1 = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
    n0 = an - n1

    freq = cp.where(an > 0, n1 / an, 0.0)
    h = cp.where(an > 1, (n0 * n1) / (an * (an - 1)), 0.0)

    if missing_data == 'exclude':
        has_missing = cp.any(hap < 0, axis=0)
        freq[has_missing] = cp.nan
        h[has_missing] = cp.nan
        an[has_missing] = 0

    return freq, h, an


def _moving_statistic(values, statistic, size, start=0, stop=None, step=None):
    """Apply a statistic to moving windows of an array.

    Parameters
    ----------
    values : ndarray, shape (n,)
    statistic : callable
    size : int
    start, stop, step : int, optional

    Returns
    -------
    ndarray, shape (n_windows,)
    """
    n = len(values)
    if stop is None:
        stop = n
    if step is None:
        step = size

    results = []
    for w_start in range(start, stop - size + 1, step):
        w_end = w_start + size
        results.append(statistic(values[w_start:w_end]))

    return np.array(results)


def _jackknife(values, statistic):
    """Block-jackknife for standard error estimation.

    Parameters
    ----------
    values : ndarray or tuple of ndarrays
        Block-level values.
    statistic : callable
        Statistic to compute from values.

    Returns
    -------
    m : float
        Mean of jackknife estimates.
    se : float
        Standard error estimate.
    vj : ndarray
        Per-iteration jackknife values.
    """
    if isinstance(values, tuple):
        n = len(values[0])
        masked = [np.ma.asarray(v) for v in values]
        for m in masked:
            m.mask = np.zeros(m.shape, dtype=bool)
    else:
        n = len(values)
        masked = np.ma.asarray(values)
        masked.mask = np.zeros(masked.shape, dtype=bool)

    vj = []
    for i in range(n):
        if isinstance(values, tuple):
            for m in masked:
                m.mask[i] = True
            x = statistic(*masked)
            for m in masked:
                m.mask[i] = False
        else:
            masked.mask[i] = True
            x = statistic(masked)
            masked.mask[i] = False
        vj.append(x)

    vj = np.array(vj)
    m = vj.mean()
    sv = ((n - 1) / n) * np.sum((vj - m) ** 2)
    se = np.sqrt(sv)

    return m, se, vj


# ---------------------------------------------------------------------------
# Public API: Per-variant F-statistics
# ---------------------------------------------------------------------------

def patterson_f2(haplotype_matrix: HaplotypeMatrix,
                 pop_a: Union[str, list],
                 pop_b: Union[str, list],
                 missing_data: str = 'include'):
    """Unbiased estimator for F2(A, B), the branch length between populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b : str or list
        Population names or sample indices.
    missing_data : str
        'include' or 'pairwise' - per-site n_valid for frequencies
        'exclude' - NaN at sites with any missing

    Returns
    -------
    f2 : ndarray, float64, shape (n_variants,)
        Per-variant F2 estimates.
    """
    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)

    a, ha, sa = _allele_freq_and_het(ma, missing_data)
    b, hb, sb = _allele_freq_and_het(mb, missing_data)

    f2 = ((a - b) ** 2) - (ha / sa) - (hb / sb)
    return f2.get()


def patterson_f3(haplotype_matrix: HaplotypeMatrix,
                 pop_c: Union[str, list],
                 pop_a: Union[str, list],
                 pop_b: Union[str, list],
                 missing_data: str = 'include'):
    """Unbiased estimator for F3(C; A, B), the three-population admixture test.

    A significantly negative F3 indicates that population C is admixed
    between populations A and B.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c : str or list
        Test population.
    pop_a, pop_b : str or list
        Source populations.
    missing_data : str
        'include' or 'pairwise' - per-site n_valid
        'exclude' - NaN at sites with any missing

    Returns
    -------
    T : ndarray, float64, shape (n_variants,)
        Un-normalized F3 estimates per variant.
    B : ndarray, float64, shape (n_variants,)
        Heterozygosity estimates (2 * h_hat) for population C.
    """
    mc = _get_population_matrix(haplotype_matrix, pop_c)
    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)

    c, hc, sc = _allele_freq_and_het(mc, missing_data)
    a, _, _ = _allele_freq_and_het(ma, missing_data)
    b, _, _ = _allele_freq_and_het(mb, missing_data)

    T = ((c - a) * (c - b)) - (hc / sc)
    B = 2 * hc

    return T.get(), B.get()


def patterson_d(haplotype_matrix: HaplotypeMatrix,
                pop_a: Union[str, list],
                pop_b: Union[str, list],
                pop_c: Union[str, list],
                pop_d: Union[str, list],
                missing_data: str = 'include'):
    """Unbiased estimator for D(A, B; C, D), the ABBA-BABA test.

    Tests for admixture between (A or B) and (C or D).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
        Population names or sample indices.
    missing_data : str
        'include' or 'pairwise' - per-site n_valid
        'exclude' - NaN at sites with any missing

    Returns
    -------
    num : ndarray, float64, shape (n_variants,)
        Numerator (un-normalized F4 estimates).
    den : ndarray, float64, shape (n_variants,)
        Denominator.
    """
    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)
    mc = _get_population_matrix(haplotype_matrix, pop_c)
    md = _get_population_matrix(haplotype_matrix, pop_d)

    a = _allele_freq(ma, missing_data)
    b = _allele_freq(mb, missing_data)
    c = _allele_freq(mc, missing_data)
    d = _allele_freq(md, missing_data)

    num = (a - b) * (c - d)
    den = (a + b - 2 * a * b) * (c + d - 2 * c * d)

    return num.get(), den.get()


# ---------------------------------------------------------------------------
# Public API: Moving window variants
# ---------------------------------------------------------------------------

def moving_patterson_f3(haplotype_matrix: HaplotypeMatrix,
                        pop_c: Union[str, list],
                        pop_a: Union[str, list],
                        pop_b: Union[str, list],
                        size: int,
                        start: int = 0,
                        stop: Optional[int] = None,
                        step: Optional[int] = None,
                        normed: bool = True,
                        missing_data: str = 'include'):
    """Estimate F3(C; A, B) in moving windows.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c, pop_a, pop_b : str or list
    size : int
        Window size (number of variants).
    start, stop, step : int, optional
    normed : bool
        If True, compute normalized F3* per window.
    missing_data : str

    Returns
    -------
    f3 : ndarray, float64, shape (n_windows,)
    """
    T, B = patterson_f3(haplotype_matrix, pop_c, pop_a, pop_b,
                         missing_data=missing_data)

    if normed:
        T_bsum = _moving_statistic(T, np.nansum, size, start, stop, step)
        B_bsum = _moving_statistic(B, np.nansum, size, start, stop, step)
        f3 = T_bsum / B_bsum
    else:
        f3 = _moving_statistic(T, np.nanmean, size, start, stop, step)

    return f3


def moving_patterson_d(haplotype_matrix: HaplotypeMatrix,
                       pop_a: Union[str, list],
                       pop_b: Union[str, list],
                       pop_c: Union[str, list],
                       pop_d: Union[str, list],
                       size: int,
                       start: int = 0,
                       stop: Optional[int] = None,
                       step: Optional[int] = None,
                       missing_data: str = 'include'):
    """Estimate D(A, B; C, D) in moving windows.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
    size : int
    start, stop, step : int, optional
    missing_data : str

    Returns
    -------
    d : ndarray, float64, shape (n_windows,)
    """
    num, den = patterson_d(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                            missing_data=missing_data)
    num_sum = _moving_statistic(num, np.nansum, size, start, stop, step)
    den_sum = _moving_statistic(den, np.nansum, size, start, stop, step)
    return num_sum / den_sum


# ---------------------------------------------------------------------------
# Public API: Block-jackknife averaged variants
# ---------------------------------------------------------------------------

def average_patterson_f3(haplotype_matrix: HaplotypeMatrix,
                         pop_c: Union[str, list],
                         pop_a: Union[str, list],
                         pop_b: Union[str, list],
                         blen: int,
                         normed: bool = True,
                         missing_data: str = 'include'):
    """Estimate F3(C; A, B) with standard error via block-jackknife.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c, pop_a, pop_b : str or list
    blen : int
        Block size (number of variants).
    normed : bool
        If True, compute normalized F3*.
    missing_data : str

    Returns
    -------
    f3 : float
        Overall estimate.
    se : float
        Standard error.
    z : float
        Z-score.
    vb : ndarray
        Per-block values.
    vj : ndarray
        Jackknife resampled values.
    """
    T, B = patterson_f3(haplotype_matrix, pop_c, pop_a, pop_b,
                         missing_data=missing_data)

    if normed:
        f3 = np.nansum(T) / np.nansum(B)
        T_bsum = _moving_statistic(T, np.nansum, blen)
        B_bsum = _moving_statistic(B, np.nansum, blen)
        vb = T_bsum / B_bsum
        _, se, vj = _jackknife(
            (T_bsum, B_bsum),
            statistic=lambda t, b: np.sum(t) / np.sum(b)
        )
    else:
        f3 = np.nanmean(T)
        vb = _moving_statistic(T, np.nanmean, blen)
        _, se, vj = _jackknife(vb, statistic=np.mean)

    z = f3 / se
    return f3, se, z, vb, vj


def average_patterson_d(haplotype_matrix: HaplotypeMatrix,
                        pop_a: Union[str, list],
                        pop_b: Union[str, list],
                        pop_c: Union[str, list],
                        pop_d: Union[str, list],
                        blen: int,
                        missing_data: str = 'include'):
    """Estimate D(A, B; C, D) with standard error via block-jackknife.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
    blen : int
        Block size (number of variants).
    missing_data : str

    Returns
    -------
    d : float
        Overall estimate.
    se : float
        Standard error.
    z : float
        Z-score.
    vb : ndarray
        Per-block values.
    vj : ndarray
        Jackknife resampled values.
    """
    num, den = patterson_d(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                            missing_data=missing_data)

    d_avg = np.nansum(num) / np.nansum(den)

    num_bsum = _moving_statistic(num, np.nansum, blen)
    den_bsum = _moving_statistic(den, np.nansum, blen)
    vb = num_bsum / den_bsum

    _, se, vj = _jackknife(
        (num_bsum, den_bsum),
        statistic=lambda n, d: np.sum(n) / np.sum(d)
    )

    z = d_avg / se
    return d_avg, se, z, vb, vj
