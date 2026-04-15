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
from .resampling import block_jackknife, _moving_nansum, _moving_nanmean


def _allele_freq(haplotype_matrix):
    """Compute alternate allele frequency from per-site valid data."""
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes
    valid_mask = hap >= 0
    n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
    n1 = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
    return cp.where(n_valid > 0, n1 / n_valid, 0.0)


def _allele_freq_and_het(haplotype_matrix):
    """Compute alternate allele frequency and unbiased heterozygosity.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix

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

    hap = haplotype_matrix.haplotypes

    valid_mask = hap >= 0
    an = cp.sum(valid_mask, axis=0).astype(cp.float64)
    n1 = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
    n0 = an - n1

    freq = cp.where(an > 0, n1 / an, 0.0)
    h = cp.where(an > 1, (n0 * n1) / (an * (an - 1)), 0.0)

    return freq, h, an



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
        'include' - per-site n_valid for frequencies
        'exclude' - filter to sites with no missing data

    Returns
    -------
    f2 : ndarray, float64, shape (n_variants,)
        Per-variant F2 estimates.
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b])
        if haplotype_matrix.num_variants == 0:
            return np.array([])

    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)

    a, ha, sa = _allele_freq_and_het(ma)
    b, hb, sb = _allele_freq_and_het(mb)

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
        'include' - per-site n_valid
        'exclude' - filter to sites with no missing data

    Returns
    -------
    T : ndarray, float64, shape (n_variants,)
        Un-normalized F3 estimates per variant.
    B : ndarray, float64, shape (n_variants,)
        Heterozygosity estimates (2 * h_hat) for population C.
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_c, pop_a, pop_b])
        if haplotype_matrix.num_variants == 0:
            return np.array([]), np.array([])

    mc = _get_population_matrix(haplotype_matrix, pop_c)
    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)

    c, hc, sc = _allele_freq_and_het(mc)
    a, _, _ = _allele_freq_and_het(ma)
    b, _, _ = _allele_freq_and_het(mb)

    T = ((c - a) * (c - b)) - (hc / sc)
    B = 2 * hc

    return T.get(), B.get()


def _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                      missing_data='include'):
    """Like patterson_f3 but returns CuPy arrays (no D2H transfer)."""
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_c, pop_a, pop_b])
        if haplotype_matrix.num_variants == 0:
            return cp.array([]), cp.array([])

    mc = _get_population_matrix(haplotype_matrix, pop_c)
    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)

    c, hc, sc = _allele_freq_and_het(mc)
    a, _, _ = _allele_freq_and_het(ma)
    b, _, _ = _allele_freq_and_het(mb)

    T = ((c - a) * (c - b)) - (hc / sc)
    B = 2 * hc
    return T, B


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
        'include' - per-site n_valid
        'exclude' - filter to sites with no missing data

    Returns
    -------
    num : ndarray, float64, shape (n_variants,)
        Numerator (un-normalized F4 estimates).
    den : ndarray, float64, shape (n_variants,)
        Denominator.
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b, pop_c, pop_d])
        if haplotype_matrix.num_variants == 0:
            return np.array([]), np.array([])

    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)
    mc = _get_population_matrix(haplotype_matrix, pop_c)
    md = _get_population_matrix(haplotype_matrix, pop_d)

    a = _allele_freq(ma)
    b = _allele_freq(mb)
    c = _allele_freq(mc)
    d = _allele_freq(md)

    num = (a - b) * (c - d)
    den = (a + b - 2 * a * b) * (c + d - 2 * c * d)

    return num.get(), den.get()


def _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                     missing_data='include'):
    """Like patterson_d but returns CuPy arrays (no D2H transfer)."""
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b, pop_c, pop_d])
        if haplotype_matrix.num_variants == 0:
            return cp.array([]), cp.array([])

    ma = _get_population_matrix(haplotype_matrix, pop_a)
    mb = _get_population_matrix(haplotype_matrix, pop_b)
    mc = _get_population_matrix(haplotype_matrix, pop_c)
    md = _get_population_matrix(haplotype_matrix, pop_d)

    a = _allele_freq(ma)
    b = _allele_freq(mb)
    c = _allele_freq(mc)
    d = _allele_freq(md)

    num = (a - b) * (c - d)
    den = (a + b - 2 * a * b) * (c + d - 2 * c * d)
    return num, den


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
    T, B = _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                              missing_data=missing_data)

    if normed:
        T_bsum = _moving_nansum(T, size, start, stop, step)
        B_bsum = _moving_nansum(B, size, start, stop, step)
        f3 = cp.where(B_bsum != 0, T_bsum / B_bsum, cp.nan)
    else:
        f3 = _moving_nanmean(T, size, start, stop, step)

    return f3.get()


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
    num, den = _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                                missing_data=missing_data)
    num_sum = _moving_nansum(num, size, start, stop, step)
    den_sum = _moving_nansum(den, size, start, stop, step)
    return cp.where(den_sum != 0, num_sum / den_sum, cp.nan).get()


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
    T, B = _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                              missing_data=missing_data)

    if normed:
        T_finite = cp.where(cp.isfinite(T), T, 0.0)
        B_finite = cp.where(cp.isfinite(B), B, 0.0)
        f3 = float((cp.sum(T_finite) / cp.sum(B_finite)).get())
        T_bsum = _moving_nansum(T, blen).get()
        B_bsum = _moving_nansum(B, blen).get()
        vb = T_bsum / B_bsum
        _, se, vj = block_jackknife(
            (T_bsum, B_bsum),
            statistic=lambda t, b: np.sum(t) / np.sum(b)
        )
    else:
        finite = cp.isfinite(T)
        f3 = float((cp.sum(cp.where(finite, T, 0.0)) / cp.sum(finite)).get())
        vb = _moving_nanmean(T, blen).get()
        _, se, vj = block_jackknife(vb, statistic=np.mean)

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
    num, den = _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                                missing_data=missing_data)

    num_f = cp.where(cp.isfinite(num), num, 0.0)
    den_f = cp.where(cp.isfinite(den), den, 0.0)
    d_avg = float((cp.sum(num_f) / cp.sum(den_f)).get())

    num_bsum = _moving_nansum(num, blen).get()
    den_bsum = _moving_nansum(den, blen).get()
    vb = num_bsum / den_bsum

    _, se, vj = block_jackknife(
        (num_bsum, den_bsum),
        statistic=lambda n, d: np.sum(n) / np.sum(d)
    )

    z = d_avg / se
    return d_avg, se, z, vb, vj
