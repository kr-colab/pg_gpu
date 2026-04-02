"""
GPU-accelerated population divergence statistics.

This module provides efficient computation of population divergence metrics
including FST, Dxy, and related statistics using GPU acceleration.
"""

import warnings
from collections import namedtuple

import numpy as np
import cupy as cp
from typing import Union, Tuple, Optional, Dict
from .haplotype_matrix import HaplotypeMatrix
from .diversity import PairwiseResult, _pairwise_pi_components
from ._memutil import chunked_dac_and_n as _pop_dac_and_n


def _pairwise_dxy_components(pop1_haps, pop2_haps, n_total_sites=None,
                             n_pop1_full=None, n_pop2_full=None):
    """Compute between-population pairwise differences and comparisons.

    Parameters
    ----------
    pop1_haps, pop2_haps : cp.ndarray, shape (n_haplotypes, n_variants)
        Haplotype data for each population.
    n_total_sites : int, optional
        Total callable sites. Invariant sites contribute 0 diffs and
        n_pop1_full * n_pop2_full comps each.
    n_pop1_full, n_pop2_full : int, optional
        Full sample sizes per population (for invariant site comps).

    Returns
    -------
    total_diffs, total_comps, total_missing, n_sites : float, float, float, int
    """
    pop1_derived, pop1_n = _pop_dac_and_n(pop1_haps)
    pop2_derived, pop2_n = _pop_dac_and_n(pop2_haps)
    pop1_n = pop1_n.astype(cp.float64)
    pop2_n = pop2_n.astype(cp.float64)
    pop1_derived = pop1_derived.astype(cp.float64)
    pop2_derived = pop2_derived.astype(cp.float64)
    pop1_ancestral = pop1_n - pop1_derived
    pop2_ancestral = pop2_n - pop2_derived

    # Per-site diffs: pop1_derived * pop2_ancestral + pop1_ancestral * pop2_derived
    site_diffs = pop1_derived * pop2_ancestral + pop1_ancestral * pop2_derived
    # Per-site comps: n_pop1 * n_pop2
    site_comps = pop1_n * pop2_n

    usable = (pop1_n > 0) & (pop2_n > 0)
    total_diffs = float(cp.sum(site_diffs[usable]).get())
    total_comps = float(cp.sum(site_comps[usable]).get())
    n_sites = int(cp.sum(usable).get())

    # Invariant site contribution
    if n_total_sites is not None:
        n1 = n_pop1_full or pop1_haps.shape[0]
        n2 = n_pop2_full or pop2_haps.shape[0]
        n_invariant = n_total_sites - n_sites
        if n_invariant > 0:
            total_comps += n_invariant * (n1 * n2)
            n_sites += n_invariant

    # Missing
    n1 = n_pop1_full or pop1_haps.shape[0]
    n2 = n_pop2_full or pop2_haps.shape[0]
    total_possible = (n1 * n2) * n_sites
    total_missing = total_possible - total_comps

    return total_diffs, total_comps, total_missing, n_sites


def fst(haplotype_matrix: HaplotypeMatrix,
        pop1: Union[str, list],
        pop2: Union[str, list],
        method: str = 'hudson',
        missing_data: str = 'include') -> float:
    """
    Compute FST between two populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population (name from sample_sets or list of indices)
    pop2 : str or list
        Second population (name from sample_sets or list of indices)
    method : str
        FST estimation method ('hudson', 'weir_cockerham', 'nei')
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data

    Returns
    -------
    float
        FST value between populations
    """
    if method == 'hudson':
        return fst_hudson(haplotype_matrix, pop1, pop2, missing_data)
    elif method == 'weir_cockerham':
        return fst_weir_cockerham(haplotype_matrix, pop1, pop2, missing_data)
    elif method == 'nei':
        return fst_nei(haplotype_matrix, pop1, pop2, missing_data)
    else:
        raise ValueError(f"Unknown FST method: {method}")


def fst_hudson(haplotype_matrix: HaplotypeMatrix,
               pop1: Union[str, list],
               pop2: Union[str, list],
               missing_data: str = 'include') -> float:
    """
    Compute Hudson's FST estimator between two populations.

    Hudson (1992) estimator:
    FST = 1 - (Hw / Hb)
    where Hw is within-population diversity and Hb is between-population diversity

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'pairwise' - Comparison-counting: FST = 1 - (pi_within / dxy)

    Returns
    -------
    float
        Hudson's FST estimate
    """
    # Get population indices
    pop1_idx = _get_population_indices(haplotype_matrix, pop1)
    pop2_idx = _get_population_indices(haplotype_matrix, pop2)

    # Ensure data is on GPU if available
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # Pairwise mode: FST = 1 - (pi_within / dxy) using pairwise estimates
    if missing_data == 'pairwise':
        pop1_haps = haplotype_matrix.haplotypes[pop1_idx, :]
        pop2_haps = haplotype_matrix.haplotypes[pop2_idx, :]
        n_ts = haplotype_matrix.n_total_sites

        # Pairwise pi for each population
        d1, c1, _, _ = _pairwise_pi_components(
            pop1_haps, n_total_sites=n_ts, n_haplotypes_full=len(pop1_idx))
        d2, c2, _, _ = _pairwise_pi_components(
            pop2_haps, n_total_sites=n_ts, n_haplotypes_full=len(pop2_idx))

        # Pairwise dxy
        d_between, c_between, _, _ = _pairwise_dxy_components(
            pop1_haps, pop2_haps, n_total_sites=n_ts,
            n_pop1_full=len(pop1_idx), n_pop2_full=len(pop2_idx))

        pi_within = (d1 + d2) / (c1 + c2) if (c1 + c2) > 0 else float('nan')
        dxy_val = d_between / c_between if c_between > 0 else float('nan')
        if dxy_val == 0 or dxy_val != dxy_val:  # nan check
            return 0.0
        fst_val = 1.0 - (pi_within / dxy_val)
        return fst_val

    # Get haplotype data
    pop1_haps = haplotype_matrix.haplotypes[pop1_idx, :]
    pop2_haps = haplotype_matrix.haplotypes[pop2_idx, :]

    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]

    # Get allele frequencies - calculate from non-missing data per site
    pop1_counts, pop1_n = _pop_dac_and_n(pop1_haps)
    pop2_counts, pop2_n = _pop_dac_and_n(pop2_haps)
    pop1_counts = pop1_counts.astype(cp.float64)
    pop2_counts = pop2_counts.astype(cp.float64)
    n1 = pop1_n.astype(cp.float64)
    n2 = pop2_n.astype(cp.float64)

    pop1_freqs = cp.where(n1 > 0, pop1_counts / n1, 0.0)
    pop2_freqs = cp.where(n2 > 0, pop2_counts / n2, 0.0)

    # Per-site within-population mean pairwise difference
    # mpd(p, n) = p*(1-p)*n/(n-1)  (unbiased heterozygosity)
    within1 = cp.zeros_like(pop1_freqs)
    within2 = cp.zeros_like(pop2_freqs)
    valid1 = n1 > 1
    valid2 = n2 > 1
    within1[valid1] = (pop1_freqs[valid1] * (1 - pop1_freqs[valid1])
                       * n1[valid1] / (n1[valid1] - 1))
    within2[valid2] = (pop2_freqs[valid2] * (1 - pop2_freqs[valid2])
                       * n2[valid2] / (n2[valid2] - 1))
    within = (within1 + within2) / 2.0

    # Per-site between-population mean pairwise difference
    between = (pop1_freqs * (1 - pop2_freqs)
               + pop2_freqs * (1 - pop1_freqs)) / 2.0

    # Numerator and denominator per SNP (ratio-of-averages)
    num = between - within
    den = between

    valid_mask = (den > 0) & (n1 > 0) & (n2 > 0)

    if cp.any(valid_mask):
        fst_val = float((cp.sum(num[valid_mask]) / cp.sum(den[valid_mask])).get())
        return fst_val
    else:
        return 0.0


def fst_weir_cockerham(haplotype_matrix: HaplotypeMatrix,
                       pop1: Union[str, list],
                       pop2: Union[str, list],
                       missing_data: str = 'include') -> float:
    """
    Compute Weir & Cockerham's (1984) FST estimator.

    This is the method of moments estimator that accounts for sampling variance.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes, ratio-of-sums
        'exclude' - Only use sites with no missing data

    Returns
    -------
    float
        Weir & Cockerham's FST estimate
    """
    # pairwise uses same ratio-of-sums as include for W-C
    if missing_data == 'pairwise':
        missing_data = 'include'

    # Get population indices
    pop1_idx = _get_population_indices(haplotype_matrix, pop1)
    pop2_idx = _get_population_indices(haplotype_matrix, pop2)

    # Ensure data is on GPU if available
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # Get haplotype data
    pop1_haps = haplotype_matrix.haplotypes[pop1_idx, :]
    pop2_haps = haplotype_matrix.haplotypes[pop2_idx, :]

    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]

    # Get allele counts and frequencies from non-missing data per site
    pop1_counts, pop1_n = _pop_dac_and_n(pop1_haps)
    pop2_counts, pop2_n = _pop_dac_and_n(pop2_haps)
    pop1_counts = pop1_counts.astype(cp.float64)
    pop2_counts = pop2_counts.astype(cp.float64)
    n1 = pop1_n.astype(cp.float64)
    n2 = pop2_n.astype(cp.float64)

    pop1_freqs = cp.where(n1 > 0, pop1_counts / n1, 0.0)
    pop2_freqs = cp.where(n2 > 0, pop2_counts / n2, 0.0)

    # Number of populations
    r = 2

    # Total sample size and average sample size (in haplotypes)
    n_total = n1 + n2
    n_bar = n_total / float(r)

    # Sample size scaling factor n_C (per site)
    # n_C = (n_total - sum(n_i^2)/n_total) / (r - 1)
    nc = cp.zeros_like(n_total)
    valid = n_total > 0
    nc[valid] = (n_total[valid] - (n1[valid]**2 + n2[valid]**2) / n_total[valid]) / (r - 1)

    # Average allele frequency weighted by sample size
    p_bar = cp.zeros_like(pop1_freqs)
    valid = n_total > 0
    p_bar[valid] = (n1[valid] * pop1_freqs[valid] + n2[valid] * pop2_freqs[valid]) / n_total[valid]

    # Sample variance of allele frequencies
    # s^2 = sum(n_i * (p_i - p_bar)^2) / ((r-1) * n_bar)
    s_squared = cp.zeros_like(p_bar)
    valid = n_bar > 0
    s_squared[valid] = (n1[valid] * (pop1_freqs[valid] - p_bar[valid])**2 +
                       n2[valid] * (pop2_freqs[valid] - p_bar[valid])**2) / ((r - 1) * n_bar[valid])

    # For haploid data, observed heterozygosity h_bar = 0
    # This simplifies the W-C variance components:
    #   a = (n_bar/n_C) * (s^2 - (1/(n_bar-1)) * (p_bar*(1-p_bar) - (r-1)*s^2/r))
    #   b = (n_bar/(n_bar-1)) * (p_bar*(1-p_bar) - (r-1)*s^2/r)
    #   c = 0
    a = cp.zeros_like(p_bar)
    b = cp.zeros_like(p_bar)
    c = cp.zeros_like(p_bar)

    valid = (n_bar > 1) & (nc > 0)
    pq = p_bar[valid] * (1 - p_bar[valid])
    s2 = s_squared[valid]
    nb = n_bar[valid]
    ncc = nc[valid]

    a[valid] = (nb / ncc) * (s2 - (1.0 / (nb - 1)) * (pq - (r - 1) * s2 / r))
    b[valid] = (nb / (nb - 1)) * (pq - (r - 1) * s2 / r)

    # Calculate FST for each locus - only for sites with sufficient data
    valid_mask = ((a + b + c) > 0) & (n1 > 0) & (n2 > 0)
    fst_per_snp = cp.zeros_like(a)
    fst_per_snp[valid_mask] = a[valid_mask] / (a[valid_mask] + b[valid_mask] + c[valid_mask])

    # Global FST estimate (ratio of averages)
    if cp.any(valid_mask):
        global_fst = float(cp.sum(a[valid_mask]).get() /
                          cp.sum(a[valid_mask] + b[valid_mask] + c[valid_mask]).get())
        return global_fst
    else:
        return 0.0


def fst_nei(haplotype_matrix: HaplotypeMatrix,
            pop1: Union[str, list],
            pop2: Union[str, list],
            missing_data: str = 'include') -> float:
    """
    Compute Nei's GST (1973).

    GST = (HT - HS) / HT
    where HT is total heterozygosity and HS is within-subpopulation heterozygosity

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'pairwise' - Ratio-of-sums: sum(HT-HS) / sum(HT)

    Returns
    -------
    float
        Nei's GST estimate
    """
    # Get population indices
    pop1_idx = _get_population_indices(haplotype_matrix, pop1)
    pop2_idx = _get_population_indices(haplotype_matrix, pop2)

    # Ensure data is on GPU if available
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # Get haplotype data
    pop1_haps = haplotype_matrix.haplotypes[pop1_idx, :]
    pop2_haps = haplotype_matrix.haplotypes[pop2_idx, :]

    # Pairwise mode: ratio-of-sums GST = sum(HT-HS) / sum(HT)
    _is_pairwise = missing_data == 'pairwise'
    if _is_pairwise:
        missing_data = 'include'

    # Handle missing data
    if missing_data == 'exclude':
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]

    # Get allele frequencies from non-missing data per site
    pop1_counts, pop1_n = _pop_dac_and_n(pop1_haps)
    pop2_counts, pop2_n = _pop_dac_and_n(pop2_haps)
    pop1_counts = pop1_counts.astype(cp.float64)
    pop2_counts = pop2_counts.astype(cp.float64)
    n1 = pop1_n.astype(cp.float64)
    n2 = pop2_n.astype(cp.float64)

    pop1_freqs = cp.where(n1 > 0, pop1_counts / n1, 0.0)
    pop2_freqs = cp.where(n2 > 0, pop2_counts / n2, 0.0)

    # Within-population heterozygosity
    hs1 = 2.0 * pop1_freqs * (1 - pop1_freqs)
    hs2 = 2.0 * pop2_freqs * (1 - pop2_freqs)

    hs = cp.zeros_like(hs1)
    p_total = cp.zeros_like(pop1_freqs)
    valid = (n1 + n2) > 0
    hs[valid] = (hs1[valid] * n1[valid] + hs2[valid] * n2[valid]) / (n1[valid] + n2[valid])
    p_total[valid] = (pop1_freqs[valid] * n1[valid] + pop2_freqs[valid] * n2[valid]) / (n1[valid] + n2[valid])

    ht = 2.0 * p_total * (1 - p_total)

    # Calculate GST - only for sites with sufficient data
    valid_mask = (ht > 0) & (n1 > 0) & (n2 > 0)

    if not cp.any(valid_mask):
        return 0.0

    # Ratio-of-averages: sum(HT-HS) / sum(HT)
    sum_ht = float(cp.sum(ht[valid_mask]).get())
    sum_hs = float(cp.sum(hs[valid_mask]).get())
    if sum_ht == 0:
        return 0.0
    return (sum_ht - sum_hs) / sum_ht


def dxy(haplotype_matrix: HaplotypeMatrix,
        pop1: Union[str, list],
        pop2: Union[str, list],
        per_site: bool = False,
        missing_data: str = 'include',
        span_denominator: bool = False,
        return_components: bool = False) -> Union[float, cp.ndarray, 'PairwiseResult']:
    """
    Compute absolute divergence (Dxy) between two populations.

    Dxy measures the average number of nucleotide differences between
    sequences from two populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    per_site : bool
        If True, return per-site values; if False, return mean
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'pairwise' - Comparison-counting normalization (pixy-style):
            dxy = sum(diffs) / sum(comps) across all sites.
    span_denominator : bool
        If True, normalize by total sites; if False, normalize by sites with data
    return_components : bool
        If True, return PairwiseResult. Only meaningful for 'pairwise' mode.

    Returns
    -------
    float, cp.ndarray, or PairwiseResult
        Mean Dxy, per-site Dxy values, or components
    """
    # Get population indices
    pop1_idx = _get_population_indices(haplotype_matrix, pop1)
    pop2_idx = _get_population_indices(haplotype_matrix, pop2)

    # Ensure data is on GPU if available
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # Get haplotype data
    pop1_haps = haplotype_matrix.haplotypes[pop1_idx, :]
    pop2_haps = haplotype_matrix.haplotypes[pop2_idx, :]
    total_sites = pop1_haps.shape[1]

    # Pairwise mode: comparison-counting normalization
    if missing_data == 'pairwise':
        if not haplotype_matrix.has_invariant_info:
            warnings.warn(
                "No invariant site information available (n_total_sites not set). "
                "Pairwise-mode dxy will be computed from variant sites only and "
                "may overestimate divergence.",
                stacklevel=2)
        total_diffs, total_comps, total_missing, n_sites = _pairwise_dxy_components(
            pop1_haps, pop2_haps,
            n_total_sites=haplotype_matrix.n_total_sites,
            n_pop1_full=len(pop1_idx),
            n_pop2_full=len(pop2_idx))
        dxy_value = total_diffs / total_comps if total_comps > 0 else float('nan')
        if return_components:
            return PairwiseResult(dxy_value, total_diffs, total_comps,
                                 total_missing, n_sites)
        return dxy_value

    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0 if not per_site else cp.zeros(total_sites)
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]
    n_sites = total_sites

    # Get allele frequencies from non-missing data per site
    pop1_counts, pop1_n = _pop_dac_and_n(pop1_haps)
    pop2_counts, pop2_n = _pop_dac_and_n(pop2_haps)
    pop1_counts = pop1_counts.astype(cp.float64)
    pop2_counts = pop2_counts.astype(cp.float64)
    pop1_n = pop1_n.astype(cp.float64)
    pop2_n = pop2_n.astype(cp.float64)

    pop1_freqs = cp.where(pop1_n > 0, pop1_counts / pop1_n, 0.0)
    pop2_freqs = cp.where(pop2_n > 0, pop2_counts / pop2_n, 0.0)

    # Calculate Dxy only for sites with data
    valid_mask = (pop1_n > 0) & (pop2_n > 0)
    dxy_per_site = cp.zeros(total_sites)
    dxy_per_site[valid_mask] = (pop1_freqs[valid_mask] + pop2_freqs[valid_mask] -
                               2 * pop1_freqs[valid_mask] * pop2_freqs[valid_mask])

    # Count sites with data for normalization
    if not span_denominator:
        n_sites = int(cp.sum(valid_mask).get())

    if per_site:
        if missing_data == 'exclude':
            result = cp.zeros(total_sites)
            result[valid_sites] = dxy_per_site
            return result.get()
        else:
            return dxy_per_site.get()
    else:
        if span_denominator:
            # Normalize by total sites
            return float(cp.sum(dxy_per_site).get() / total_sites)
        elif n_sites > 0:
            # Normalize by sites with data
            return float(cp.sum(dxy_per_site).get() / n_sites)
        else:
            return 0.0


def da(haplotype_matrix: HaplotypeMatrix,
       pop1: Union[str, list],
       pop2: Union[str, list],
       missing_data: str = 'include',
       span_denominator: bool = False) -> float:
    """
    Compute net divergence (Da) between two populations.

    Da = Dxy - (pi1 + pi2) / 2
    where pi1 and pi2 are within-population diversities

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
    span_denominator : bool
        If True, normalize by total sites; if False, normalize by sites with data

    Returns
    -------
    float
        Net divergence (Da)
    """
    # Get Dxy
    dxy_value = dxy(haplotype_matrix, pop1, pop2, missing_data=missing_data,
                   span_denominator=span_denominator)

    # Get within-population diversities
    pi1 = pi_within_population(haplotype_matrix, pop1, missing_data=missing_data,
                              span_denominator=span_denominator)
    pi2 = pi_within_population(haplotype_matrix, pop2, missing_data=missing_data,
                              span_denominator=span_denominator)

    # Calculate Da
    da_value = dxy_value - (pi1 + pi2) / 2.0

    return da_value


def pi_within_population(haplotype_matrix: HaplotypeMatrix,
                        pop: Union[str, list],
                        missing_data: str = 'include',
                        span_denominator: bool = False) -> float:
    """
    Compute nucleotide diversity (pi) within a population.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop : str or list
        Population name or list of indices
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
    span_denominator : bool
        If True, normalize by total sites; if False, normalize by sites with data

    Returns
    -------
    float
        Nucleotide diversity
    """
    # Get population indices
    pop_idx = _get_population_indices(haplotype_matrix, pop)

    # Ensure data is on GPU if available
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # Pairwise mode: comparison-counting pi
    if missing_data == 'pairwise':
        pop_haps = haplotype_matrix.haplotypes[pop_idx, :]
        total_diffs, total_comps, _, _ = _pairwise_pi_components(
            pop_haps, n_total_sites=haplotype_matrix.n_total_sites,
            n_haplotypes_full=len(pop_idx))
        return total_diffs / total_comps if total_comps > 0 else 0.0

    # Extract population haplotypes
    pop_haplotypes = haplotype_matrix.haplotypes[pop_idx, :]
    total_sites = pop_haplotypes.shape[1]

    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop_haplotypes >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop_haplotypes = pop_haplotypes[:, valid_sites]
    n_sites = total_sites

    # Calculate frequencies from non-missing data per site
    pop_counts, n = _pop_dac_and_n(pop_haplotypes)
    pop_counts = pop_counts.astype(cp.float64)
    n = n.astype(cp.float64)

    pop_freq = cp.where(n > 0, pop_counts / n, 0.0)

    # Pi = 2 * p * (1 - p) * n / (n - 1) for sites with n > 1
    pi_per_site = cp.zeros(total_sites)
    valid_mask = n > 1
    pi_per_site[valid_mask] = (2.0 * pop_freq[valid_mask] * (1 - pop_freq[valid_mask]) *
                              n[valid_mask] / (n[valid_mask] - 1))

    # Count sites with sufficient data for normalization
    if not span_denominator:
        n_sites = int(cp.sum(valid_mask).get())

    if span_denominator:
        # Normalize by total sites
        return float(cp.sum(pi_per_site).get() / total_sites) if total_sites > 0 else 0.0
    elif n_sites > 0:
        # Normalize by sites with data
        return float(cp.sum(pi_per_site).get() / n_sites)
    else:
        return 0.0


def divergence_stats(haplotype_matrix: HaplotypeMatrix,
                    pop1: Union[str, list],
                    pop2: Union[str, list],
                    statistics: list = ['fst', 'dxy', 'da'],
                    missing_data: str = 'include',
                    span_denominator: bool = False) -> Dict[str, float]:
    """
    Compute multiple divergence statistics between two populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop1 : str or list
        First population
    pop2 : str or list
        Second population
    statistics : list
        List of statistics to compute
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
    span_denominator : bool
        If True, normalize by total sites; if False, normalize by sites with data
        (only applies to dxy, da, pi1, pi2)

    Returns
    -------
    dict
        Dictionary of statistic names to values
    """
    results = {}

    for stat in statistics:
        if stat == 'fst':
            results['fst'] = fst(haplotype_matrix, pop1, pop2, missing_data=missing_data)
        elif stat == 'fst_hudson':
            results['fst_hudson'] = fst_hudson(haplotype_matrix, pop1, pop2, missing_data=missing_data)
        elif stat == 'fst_wc':
            results['fst_wc'] = fst_weir_cockerham(haplotype_matrix, pop1, pop2, missing_data=missing_data)
        elif stat == 'fst_nei':
            results['fst_nei'] = fst_nei(haplotype_matrix, pop1, pop2, missing_data=missing_data)
        elif stat == 'dxy':
            results['dxy'] = dxy(haplotype_matrix, pop1, pop2, missing_data=missing_data,
                               span_denominator=span_denominator)
        elif stat == 'da':
            results['da'] = da(haplotype_matrix, pop1, pop2, missing_data=missing_data,
                             span_denominator=span_denominator)
        elif stat == 'pi1':
            results['pi1'] = pi_within_population(haplotype_matrix, pop1, missing_data=missing_data,
                                                span_denominator=span_denominator)
        elif stat == 'pi2':
            results['pi2'] = pi_within_population(haplotype_matrix, pop2, missing_data=missing_data,
                                                span_denominator=span_denominator)
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return results


def pairwise_fst(haplotype_matrix: HaplotypeMatrix,
                 populations: Optional[list] = None,
                 method: str = 'hudson',
                 missing_data: str = 'include') -> Tuple[cp.ndarray, list]:
    """
    Compute pairwise FST matrix for multiple populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    populations : list, optional
        List of population names. If None, uses all populations in sample_sets
    method : str
        FST method to use
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data

    Returns
    -------
    fst_matrix : cp.ndarray
        Pairwise FST matrix
    pop_names : list
        Population names in matrix order
    """
    if populations is None:
        if haplotype_matrix.sample_sets is None:
            raise ValueError("No populations defined in haplotype matrix")
        populations = list(haplotype_matrix.sample_sets.keys())

    n_pops = len(populations)
    fst_matrix = cp.zeros((n_pops, n_pops))

    for i in range(n_pops):
        for j in range(i + 1, n_pops):
            fst_value = fst(haplotype_matrix, populations[i], populations[j], method, missing_data)
            fst_matrix[i, j] = fst_value
            fst_matrix[j, i] = fst_value

    return fst_matrix, populations


def _get_population_indices(haplotype_matrix: HaplotypeMatrix,
                           pop: Union[str, list]) -> list:
    """
    Get population indices from name or list.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop : str or list
        Population name or list of indices

    Returns
    -------
    list
        Population indices
    """
    if isinstance(pop, str):
        if haplotype_matrix.sample_sets is None:
            raise ValueError("No sample_sets defined in haplotype matrix")
        if pop not in haplotype_matrix.sample_sets:
            raise ValueError(f"Population {pop} not found in sample_sets")
        return haplotype_matrix.sample_sets[pop]
    else:
        return list(pop)


def _pop_allele_counts(haplotype_matrix, pop, missing_data='include'):
    """Compute per-variant allele counts for a population on GPU.

    Returns (ac_0, ac_1, n) as CuPy arrays. n is per-site (array)
    for 'include'/'pairwise' modes, and also per-site after filtering
    for 'exclude' mode.
    """
    pop_idx = _get_population_indices(haplotype_matrix, pop)
    h = haplotype_matrix.haplotypes[pop_idx, :]

    if missing_data == 'exclude':
        valid_sites = cp.all(h >= 0, axis=0)
        h = h[:, valid_sites]

    # Use per-site non-missing counts
    ac_1, n = _pop_dac_and_n(h)
    n = n.astype(cp.float64)
    ac_1 = ac_1.astype(cp.float64)
    ac_0 = n - ac_1
    return ac_0, ac_1, n


def _hudson_fst_from_counts(ac1_0, ac1_1, n1, ac2_0, ac2_1, n2):
    """Compute per-variant Hudson FST num/den from precomputed allele counts.

    Returns (num, den) as numpy arrays.
    """
    n1_pairs = n1 * (n1 - 1) / 2
    n1_same = (ac1_0 * (ac1_0 - 1) + ac1_1 * (ac1_1 - 1)) / 2
    mpd1 = cp.where(n1_pairs > 0, (n1_pairs - n1_same) / n1_pairs, 0.0)

    n2_pairs = n2 * (n2 - 1) / 2
    n2_same = (ac2_0 * (ac2_0 - 1) + ac2_1 * (ac2_1 - 1)) / 2
    mpd2 = cp.where(n2_pairs > 0, (n2_pairs - n2_same) / n2_pairs, 0.0)

    within = (mpd1 + mpd2) / 2.0

    n_between = n1 * n2
    n_between_same = ac1_0 * ac2_0 + ac1_1 * ac2_1
    between = cp.where(n_between > 0,
                       (n_between - n_between_same) / n_between, 0.0)

    return (between - within).get(), between.get()


def _windowed_fst(num, den, size, start=0, stop=None, step=None):
    """Compute windowed FST from per-variant numerator/denominator."""
    n = len(num)
    if stop is None:
        stop = n
    if step is None:
        step = size

    fst_vals = []
    for w_start in range(start, stop - size + 1, step):
        w_end = w_start + size
        den_sum = np.nansum(den[w_start:w_end])
        if den_sum != 0:
            fst_vals.append(np.nansum(num[w_start:w_end]) / den_sum)
        else:
            fst_vals.append(np.nan)

    return np.array(fst_vals)


def pbs(haplotype_matrix: HaplotypeMatrix,
        pop1: Union[str, list],
        pop2: Union[str, list],
        pop3: Union[str, list],
        window_size: int,
        window_start: int = 0,
        window_stop: Optional[int] = None,
        window_step: Optional[int] = None,
        normed: bool = True,
        missing_data: str = 'include'):
    """Compute the Population Branch Statistic (PBS).

    PBS detects genomic regions unusually differentiated in pop1 relative
    to pop2 and pop3, using pairwise Hudson FST estimates.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data containing all three populations.
    pop1, pop2, pop3 : str or list
        Population names or sample indices.
    window_size : int
        Number of variants per window.
    window_start : int, optional
        Starting variant index.
    window_stop : int, optional
        Stopping variant index.
    window_step : int, optional
        Stride between windows. Defaults to window_size (non-overlapping).
    normed : bool, optional
        If True (default), return normalized PBS (PBSn1).
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes
        'exclude' - only use sites with no missing data

    Returns
    -------
    ndarray, float64, shape (n_windows,)
        PBS values per window.
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # precompute allele counts once per population
    ac1_0, ac1_1, n1 = _pop_allele_counts(haplotype_matrix, pop1, missing_data)
    ac2_0, ac2_1, n2 = _pop_allele_counts(haplotype_matrix, pop2, missing_data)
    ac3_0, ac3_1, n3 = _pop_allele_counts(haplotype_matrix, pop3, missing_data)

    # compute all three pairwise FST num/den from shared counts
    num12, den12 = _hudson_fst_from_counts(ac1_0, ac1_1, n1, ac2_0, ac2_1, n2)
    num13, den13 = _hudson_fst_from_counts(ac1_0, ac1_1, n1, ac3_0, ac3_1, n3)
    num23, den23 = _hudson_fst_from_counts(ac2_0, ac2_1, n2, ac3_0, ac3_1, n3)

    fst12 = _windowed_fst(num12, den12, window_size, window_start,
                          window_stop, window_step)
    fst13 = _windowed_fst(num13, den13, window_size, window_start,
                          window_stop, window_step)
    fst23 = _windowed_fst(num23, den23, window_size, window_start,
                          window_stop, window_step)

    np.clip(fst12, 0, 0.99999, out=fst12)
    np.clip(fst13, 0, 0.99999, out=fst13)
    np.clip(fst23, 0, 0.99999, out=fst23)

    t12 = -np.log(1 - fst12)
    t13 = -np.log(1 - fst13)
    t23 = -np.log(1 - fst23)

    ret = (t12 + t13 - t23) / 2

    if normed:
        norm = 1 + (t12 + t13 + t23) / 2
        ret = ret / norm

    return ret
