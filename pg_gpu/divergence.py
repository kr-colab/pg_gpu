"""
GPU-accelerated population divergence statistics.

This module provides efficient computation of population divergence metrics
including FST, Dxy, and related statistics using GPU acceleration.
"""

import cupy as cp
from typing import Union, Tuple, Optional, Dict
from .haplotype_matrix import HaplotypeMatrix


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
        'ignore' - Treat missing as reference allele (original behavior)
        
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
        'ignore' - Treat missing as reference allele (original behavior)
        
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
    elif missing_data == 'include':
        # Will handle missing data per site below
        pass
    elif missing_data == 'ignore':
        # Treat missing as reference allele
        pop1_haps = cp.where(pop1_haps < 0, 0, pop1_haps)
        pop2_haps = cp.where(pop2_haps < 0, 0, pop2_haps)
    
    # Get allele frequencies
    if missing_data == 'include':
        # Calculate frequencies from non-missing data per site
        pop1_mask = pop1_haps >= 0
        pop2_mask = pop2_haps >= 0
        pop1_counts = cp.sum(cp.where(pop1_mask, pop1_haps, 0), axis=0)
        pop2_counts = cp.sum(cp.where(pop2_mask, pop2_haps, 0), axis=0)
        pop1_n = cp.sum(pop1_mask, axis=0)
        pop2_n = cp.sum(pop2_mask, axis=0)
        
        # Avoid division by zero
        pop1_freqs = cp.divide(pop1_counts, pop1_n, out=cp.zeros_like(pop1_counts, dtype=float), where=pop1_n > 0)
        pop2_freqs = cp.divide(pop2_counts, pop2_n, out=cp.zeros_like(pop2_counts, dtype=float), where=pop2_n > 0)
        
        # Use actual sample sizes per site
        n1 = pop1_n
        n2 = pop2_n
    else:
        pop1_freqs = cp.mean(pop1_haps, axis=0)
        pop2_freqs = cp.mean(pop2_haps, axis=0)
        n1 = len(pop1_idx)
        n2 = len(pop2_idx)
    
    # Calculate within-population heterozygosity
    if missing_data == 'include':
        # Handle per-site sample sizes
        hw1 = cp.zeros_like(pop1_freqs)
        hw2 = cp.zeros_like(pop2_freqs)
        valid1 = n1 > 1
        valid2 = n2 > 1
        hw1[valid1] = 2.0 * pop1_freqs[valid1] * (1 - pop1_freqs[valid1]) * n1[valid1] / (n1[valid1] - 1)
        hw2[valid2] = 2.0 * pop2_freqs[valid2] * (1 - pop2_freqs[valid2]) * n2[valid2] / (n2[valid2] - 1)
        hw = cp.divide(hw1 * n1 + hw2 * n2, n1 + n2, out=cp.zeros_like(hw1), where=(n1 + n2) > 0)
    else:
        hw1 = 2.0 * pop1_freqs * (1 - pop1_freqs) * n1 / (n1 - 1)
        hw2 = 2.0 * pop2_freqs * (1 - pop2_freqs) * n2 / (n2 - 1)
        hw = (hw1 * n1 + hw2 * n2) / (n1 + n2)
    
    # Calculate between-population heterozygosity
    if missing_data == 'include':
        p_avg = cp.divide(pop1_freqs * n1 + pop2_freqs * n2, n1 + n2, 
                         out=cp.zeros_like(pop1_freqs), where=(n1 + n2) > 0)
    else:
        p_avg = (pop1_freqs * n1 + pop2_freqs * n2) / (n1 + n2)
    hb = 2.0 * p_avg * (1 - p_avg)
    
    # Calculate FST for each SNP
    if missing_data == 'include':
        # Only calculate for sites with sufficient data
        valid_mask = (hb > 0) & (n1 > 0) & (n2 > 0)
    else:
        valid_mask = hb > 0
    fst_per_snp = cp.zeros_like(hb)
    fst_per_snp[valid_mask] = 1 - (hw[valid_mask] / hb[valid_mask])
    
    # Average across SNPs (excluding invalid values)
    if cp.any(valid_mask):
        mean_fst = float(cp.mean(fst_per_snp[valid_mask]).get())
        return max(0.0, mean_fst)  # FST should be non-negative
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
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    float
        Weir & Cockerham's FST estimate
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
    
    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]
    elif missing_data == 'include':
        # Will handle missing data per site below
        pass
    elif missing_data == 'ignore':
        # Treat missing as reference allele
        pop1_haps = cp.where(pop1_haps < 0, 0, pop1_haps)
        pop2_haps = cp.where(pop2_haps < 0, 0, pop2_haps)
    
    # Get allele counts and frequencies
    if missing_data == 'include':
        # Calculate from non-missing data per site
        pop1_mask = pop1_haps >= 0
        pop2_mask = pop2_haps >= 0
        pop1_counts = cp.sum(cp.where(pop1_mask, pop1_haps, 0), axis=0).astype(float)
        pop2_counts = cp.sum(cp.where(pop2_mask, pop2_haps, 0), axis=0).astype(float)
        n1 = cp.sum(pop1_mask, axis=0).astype(float)
        n2 = cp.sum(pop2_mask, axis=0).astype(float)
        
        pop1_freqs = cp.divide(pop1_counts, n1, out=cp.zeros_like(pop1_counts), where=n1 > 0)
        pop2_freqs = cp.divide(pop2_counts, n2, out=cp.zeros_like(pop2_counts), where=n2 > 0)
    else:
        pop1_counts = cp.sum(pop1_haps, axis=0).astype(float)
        pop2_counts = cp.sum(pop2_haps, axis=0).astype(float)
        n1 = float(len(pop1_idx))
        n2 = float(len(pop2_idx))
        pop1_freqs = pop1_counts / n1
        pop2_freqs = pop2_counts / n2
    
    # Total sample size and average sample size
    if missing_data == 'include':
        n_total = n1 + n2
        n_bar = n_total / 2.0  # For two populations
        
        # Sample size scaling factor (per site)
        nc = cp.zeros_like(n_total)
        valid = n_total > 0
        nc[valid] = (n_total[valid] - (n1[valid]**2 + n2[valid]**2) / n_total[valid]) / 1.0
        
        # Average allele frequency
        p_bar = cp.divide(n1 * pop1_freqs + n2 * pop2_freqs, n_total, 
                         out=cp.zeros_like(pop1_freqs), where=n_total > 0)
    else:
        n_total = n1 + n2
        n_bar = n_total / 2.0  # For two populations
        nc = (n_total - (n1**2 + n2**2) / n_total) / 1.0  # r-1 = 1 for 2 pops
        p_bar = (n1 * pop1_freqs + n2 * pop2_freqs) / n_total
    
    # Sample variance of allele frequencies
    if missing_data == 'include':
        s_squared = cp.divide(n1 * (pop1_freqs - p_bar)**2 + n2 * (pop2_freqs - p_bar)**2, 
                             n_bar, out=cp.zeros_like(p_bar), where=n_bar > 0)
    else:
        s_squared = (n1 * (pop1_freqs - p_bar)**2 + 
                     n2 * (pop2_freqs - p_bar)**2) / (1 * n_bar)
    
    # Calculate variance components
    # He = expected heterozygosity
    He = 2.0 * p_bar * (1 - p_bar)
    
    if missing_data == 'include':
        # a = between population variance (per site)
        a = cp.zeros_like(He)
        b = cp.zeros_like(He)
        c = He / 2.0
        
        valid = (n_bar > 0.5) & (nc > 0)
        a[valid] = ((s_squared[valid] - He[valid] / (2 * n_bar[valid] - 1)) * 
                    (n_bar[valid] / nc[valid]))
        b[valid] = He[valid] * (2 * n_bar[valid] - 1) / (2 * n_bar[valid])
    else:
        # a = between population variance
        a = (s_squared - He / (2 * n_bar - 1)) * (n_bar / nc)
        # b = between individual within population variance
        b = He * (2 * n_bar - 1) / (2 * n_bar)
        # c = within individual variance
        c = He / 2.0
    
    # Calculate FST for each locus
    if missing_data == 'include':
        # Only calculate for sites with sufficient data
        valid_mask = ((a + b + c) > 0) & (n1 > 0) & (n2 > 0)
    else:
        valid_mask = (a + b + c) > 0
    fst_per_snp = cp.zeros_like(a)
    fst_per_snp[valid_mask] = a[valid_mask] / (a[valid_mask] + b[valid_mask] + c[valid_mask])
    
    # Global FST estimate (ratio of averages)
    if cp.any(valid_mask):
        global_fst = float(cp.sum(a[valid_mask]).get() / 
                          cp.sum(a[valid_mask] + b[valid_mask] + c[valid_mask]).get())
        return max(0.0, global_fst)
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
        'ignore' - Treat missing as reference allele (original behavior)
        
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
    
    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]
    elif missing_data == 'include':
        # Will handle missing data per site below
        pass
    elif missing_data == 'ignore':
        # Treat missing as reference allele
        pop1_haps = cp.where(pop1_haps < 0, 0, pop1_haps)
        pop2_haps = cp.where(pop2_haps < 0, 0, pop2_haps)
    
    # Get allele frequencies
    if missing_data == 'include':
        # Calculate frequencies from non-missing data per site
        pop1_mask = pop1_haps >= 0
        pop2_mask = pop2_haps >= 0
        pop1_counts = cp.sum(cp.where(pop1_mask, pop1_haps, 0), axis=0)
        pop2_counts = cp.sum(cp.where(pop2_mask, pop2_haps, 0), axis=0)
        n1 = cp.sum(pop1_mask, axis=0)
        n2 = cp.sum(pop2_mask, axis=0)
        
        pop1_freqs = cp.divide(pop1_counts, n1, out=cp.zeros_like(pop1_counts, dtype=float), where=n1 > 0)
        pop2_freqs = cp.divide(pop2_counts, n2, out=cp.zeros_like(pop2_counts, dtype=float), where=n2 > 0)
    else:
        pop1_freqs = cp.mean(pop1_haps, axis=0)
        pop2_freqs = cp.mean(pop2_haps, axis=0)
        n1 = len(pop1_idx)
        n2 = len(pop2_idx)
    
    # Within-population heterozygosity
    hs1 = 2.0 * pop1_freqs * (1 - pop1_freqs)
    hs2 = 2.0 * pop2_freqs * (1 - pop2_freqs)
    
    if missing_data == 'include':
        hs = cp.divide(hs1 * n1 + hs2 * n2, n1 + n2, 
                      out=cp.zeros_like(hs1), where=(n1 + n2) > 0)
        p_total = cp.divide(pop1_freqs * n1 + pop2_freqs * n2, n1 + n2, 
                           out=cp.zeros_like(pop1_freqs), where=(n1 + n2) > 0)
    else:
        hs = (hs1 * n1 + hs2 * n2) / (n1 + n2)
        p_total = (pop1_freqs * n1 + pop2_freqs * n2) / (n1 + n2)
    
    ht = 2.0 * p_total * (1 - p_total)
    
    # Calculate GST for each SNP
    if missing_data == 'include':
        # Only calculate for sites with sufficient data
        valid_mask = (ht > 0) & (n1 > 0) & (n2 > 0)
    else:
        valid_mask = ht > 0
    gst_per_snp = cp.zeros_like(ht)
    gst_per_snp[valid_mask] = (ht[valid_mask] - hs[valid_mask]) / ht[valid_mask]
    
    # Average across SNPs
    if cp.any(valid_mask):
        mean_gst = float(cp.mean(gst_per_snp[valid_mask]).get())
        return max(0.0, mean_gst)
    else:
        return 0.0


def dxy(haplotype_matrix: HaplotypeMatrix,
        pop1: Union[str, list],
        pop2: Union[str, list],
        per_site: bool = False,
        missing_data: str = 'include',
        span_denominator: bool = False) -> Union[float, cp.ndarray]:
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
        'ignore' - Treat missing as reference allele (original behavior)
    span_denominator : bool
        If True, normalize by total sites; if False, normalize by sites with data
        
    Returns
    -------
    float or cp.ndarray
        Mean Dxy or per-site Dxy values
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
    
    # Handle missing data
    if missing_data == 'exclude':
        # Only use sites with no missing data
        valid_sites = cp.all(pop1_haps >= 0, axis=0) & cp.all(pop2_haps >= 0, axis=0)
        if not cp.any(valid_sites):
            return 0.0 if not per_site else cp.zeros(total_sites)
        pop1_haps = pop1_haps[:, valid_sites]
        pop2_haps = pop2_haps[:, valid_sites]
        n_sites = cp.sum(valid_sites)
    elif missing_data == 'include':
        # Will handle missing data per site below
        n_sites = total_sites
    elif missing_data == 'ignore':
        # Treat missing as reference allele
        pop1_haps = cp.where(pop1_haps < 0, 0, pop1_haps)
        pop2_haps = cp.where(pop2_haps < 0, 0, pop2_haps)
        n_sites = total_sites
    
    # Get allele frequencies
    if missing_data == 'include':
        # Calculate frequencies from non-missing data per site
        pop1_mask = pop1_haps >= 0
        pop2_mask = pop2_haps >= 0
        pop1_counts = cp.sum(cp.where(pop1_mask, pop1_haps, 0), axis=0)
        pop2_counts = cp.sum(cp.where(pop2_mask, pop2_haps, 0), axis=0)
        pop1_n = cp.sum(pop1_mask, axis=0)
        pop2_n = cp.sum(pop2_mask, axis=0)
        
        pop1_freqs = cp.divide(pop1_counts, pop1_n, out=cp.zeros_like(pop1_counts, dtype=float), where=pop1_n > 0)
        pop2_freqs = cp.divide(pop2_counts, pop2_n, out=cp.zeros_like(pop2_counts, dtype=float), where=pop2_n > 0)
        
        # Calculate Dxy only for sites with data
        valid_mask = (pop1_n > 0) & (pop2_n > 0)
        dxy_per_site = cp.zeros(total_sites)
        dxy_per_site[valid_mask] = (pop1_freqs[valid_mask] + pop2_freqs[valid_mask] - 
                                   2 * pop1_freqs[valid_mask] * pop2_freqs[valid_mask])
        
        # Count sites with data for normalization
        if not span_denominator:
            n_sites = cp.sum(valid_mask)
    else:
        pop1_freqs = cp.mean(pop1_haps, axis=0)
        pop2_freqs = cp.mean(pop2_haps, axis=0)
        # Dxy = p1(1-p2) + p2(1-p1) = p1 + p2 - 2*p1*p2
        dxy_per_site = pop1_freqs + pop2_freqs - 2 * pop1_freqs * pop2_freqs
    
    if per_site:
        if missing_data == 'exclude':
            # Return values for valid sites only
            result = cp.zeros(total_sites)
            result[valid_sites] = dxy_per_site
            return result
        else:
            return dxy_per_site
    else:
        if span_denominator and missing_data == 'include':
            # Normalize by total sites
            return float(cp.sum(dxy_per_site).get() / total_sites)
        elif missing_data == 'include' and n_sites > 0:
            # Normalize by sites with data
            return float(cp.sum(dxy_per_site).get() / n_sites)
        elif n_sites > 0:
            return float(cp.mean(dxy_per_site).get())
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
        'ignore' - Treat missing as reference allele (original behavior)
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
    
    return max(0.0, da_value)  # Da should be non-negative


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
        'ignore' - Treat missing as reference allele (original behavior)
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
        n_sites = cp.sum(valid_sites)
    elif missing_data == 'include':
        # Will handle missing data per site below
        n_sites = total_sites
    elif missing_data == 'ignore':
        # Treat missing as reference allele
        pop_haplotypes = cp.where(pop_haplotypes < 0, 0, pop_haplotypes)
        n_sites = total_sites
    
    if missing_data == 'include':
        # Calculate frequencies from non-missing data per site
        pop_mask = pop_haplotypes >= 0
        pop_counts = cp.sum(cp.where(pop_mask, pop_haplotypes, 0), axis=0)
        n = cp.sum(pop_mask, axis=0)
        
        pop_freq = cp.divide(pop_counts, n, out=cp.zeros_like(pop_counts, dtype=float), where=n > 0)
        
        # Pi = 2 * p * (1 - p) * n / (n - 1) for sites with n > 1
        pi_per_site = cp.zeros(total_sites)
        valid_mask = n > 1
        pi_per_site[valid_mask] = (2.0 * pop_freq[valid_mask] * (1 - pop_freq[valid_mask]) * 
                                  n[valid_mask] / (n[valid_mask] - 1))
        
        # Count sites with sufficient data for normalization
        if not span_denominator:
            n_sites = cp.sum(valid_mask)
        
        if span_denominator:
            # Normalize by total sites
            return float(cp.sum(pi_per_site).get() / total_sites) if total_sites > 0 else 0.0
        elif n_sites > 0:
            # Normalize by sites with data
            return float(cp.sum(pi_per_site).get() / n_sites)
        else:
            return 0.0
    else:
        n = len(pop_idx)
        # Calculate allele frequency
        pop_freq = cp.mean(pop_haplotypes, axis=0)
        
        # Pi = 2 * p * (1 - p) * n / (n - 1)
        if n > 1:
            pi_per_site = 2.0 * pop_freq * (1 - pop_freq) * n / (n - 1)
            if span_denominator and missing_data == 'exclude':
                # Normalize by total sites
                return float(cp.sum(pi_per_site).get() / total_sites) if total_sites > 0 else 0.0
            else:
                return float(cp.mean(pi_per_site).get())
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
        'ignore' - Treat missing as reference allele (original behavior)
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
        'ignore' - Treat missing as reference allele (original behavior)
        
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
