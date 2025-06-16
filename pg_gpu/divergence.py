"""
GPU-accelerated population divergence statistics.

This module provides efficient computation of population divergence metrics
including FST, Dxy, and related statistics using GPU acceleration.
"""

import numpy as np
import cupy as cp
from typing import Union, Tuple, Optional, Dict
from .haplotype_matrix import HaplotypeMatrix


def fst(haplotype_matrix: HaplotypeMatrix,
        pop1: Union[str, list],
        pop2: Union[str, list],
        method: str = 'hudson') -> float:
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
        
    Returns
    -------
    float
        FST value between populations
    """
    if method == 'hudson':
        return fst_hudson(haplotype_matrix, pop1, pop2)
    elif method == 'weir_cockerham':
        return fst_weir_cockerham(haplotype_matrix, pop1, pop2)
    elif method == 'nei':
        return fst_nei(haplotype_matrix, pop1, pop2)
    else:
        raise ValueError(f"Unknown FST method: {method}")


def fst_hudson(haplotype_matrix: HaplotypeMatrix,
               pop1: Union[str, list],
               pop2: Union[str, list]) -> float:
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
    
    # Get allele frequencies
    pop1_freqs = cp.mean(haplotype_matrix.haplotypes[pop1_idx, :], axis=0)
    pop2_freqs = cp.mean(haplotype_matrix.haplotypes[pop2_idx, :], axis=0)
    
    n1 = len(pop1_idx)
    n2 = len(pop2_idx)
    
    # Calculate within-population heterozygosity
    hw1 = 2.0 * pop1_freqs * (1 - pop1_freqs) * n1 / (n1 - 1)
    hw2 = 2.0 * pop2_freqs * (1 - pop2_freqs) * n2 / (n2 - 1)
    hw = (hw1 * n1 + hw2 * n2) / (n1 + n2)
    
    # Calculate between-population heterozygosity
    p_avg = (pop1_freqs * n1 + pop2_freqs * n2) / (n1 + n2)
    hb = 2.0 * p_avg * (1 - p_avg)
    
    # Calculate FST for each SNP
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
                       pop2: Union[str, list]) -> float:
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
    
    # Get allele counts and frequencies
    pop1_counts = cp.sum(haplotype_matrix.haplotypes[pop1_idx, :], axis=0)
    pop2_counts = cp.sum(haplotype_matrix.haplotypes[pop2_idx, :], axis=0)
    
    n1 = len(pop1_idx)
    n2 = len(pop2_idx)
    
    pop1_freqs = pop1_counts / n1
    pop2_freqs = pop2_counts / n2
    
    # Total sample size and average sample size
    n_total = n1 + n2
    n_bar = n_total / 2.0  # For two populations
    
    # Sample size scaling factor
    nc = (n_total - (n1**2 + n2**2) / n_total) / 1.0  # r-1 = 1 for 2 pops
    
    # Average allele frequency
    p_bar = (n1 * pop1_freqs + n2 * pop2_freqs) / n_total
    
    # Sample variance of allele frequencies
    s_squared = (n1 * (pop1_freqs - p_bar)**2 + 
                 n2 * (pop2_freqs - p_bar)**2) / (1 * n_bar)
    
    # Calculate variance components
    # He = expected heterozygosity
    He = 2.0 * p_bar * (1 - p_bar)
    
    # a = between population variance
    a = (s_squared - He / (2 * n_bar - 1)) * (n_bar / nc)
    
    # b = between individual within population variance
    b = He * (2 * n_bar - 1) / (2 * n_bar)
    
    # c = within individual variance
    c = He / 2.0
    
    # Calculate FST for each locus
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
            pop2: Union[str, list]) -> float:
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
    
    # Get allele frequencies
    pop1_freqs = cp.mean(haplotype_matrix.haplotypes[pop1_idx, :], axis=0)
    pop2_freqs = cp.mean(haplotype_matrix.haplotypes[pop2_idx, :], axis=0)
    
    n1 = len(pop1_idx)
    n2 = len(pop2_idx)
    
    # Within-population heterozygosity
    hs1 = 2.0 * pop1_freqs * (1 - pop1_freqs)
    hs2 = 2.0 * pop2_freqs * (1 - pop2_freqs)
    hs = (hs1 * n1 + hs2 * n2) / (n1 + n2)
    
    # Total heterozygosity
    p_total = (pop1_freqs * n1 + pop2_freqs * n2) / (n1 + n2)
    ht = 2.0 * p_total * (1 - p_total)
    
    # Calculate GST for each SNP
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
        per_site: bool = False) -> Union[float, cp.ndarray]:
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
    
    # Get allele frequencies
    pop1_freqs = cp.mean(haplotype_matrix.haplotypes[pop1_idx, :], axis=0)
    pop2_freqs = cp.mean(haplotype_matrix.haplotypes[pop2_idx, :], axis=0)
    
    # Dxy = p1(1-p2) + p2(1-p1) = p1 + p2 - 2*p1*p2
    dxy_per_site = pop1_freqs + pop2_freqs - 2 * pop1_freqs * pop2_freqs
    
    if per_site:
        return dxy_per_site
    else:
        return float(cp.mean(dxy_per_site).get())


def da(haplotype_matrix: HaplotypeMatrix,
       pop1: Union[str, list],
       pop2: Union[str, list]) -> float:
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
        
    Returns
    -------
    float
        Net divergence (Da)
    """
    # Get Dxy
    dxy_value = dxy(haplotype_matrix, pop1, pop2)
    
    # Get within-population diversities
    pi1 = pi_within_population(haplotype_matrix, pop1)
    pi2 = pi_within_population(haplotype_matrix, pop2)
    
    # Calculate Da
    da_value = dxy_value - (pi1 + pi2) / 2.0
    
    return max(0.0, da_value)  # Da should be non-negative


def pi_within_population(haplotype_matrix: HaplotypeMatrix,
                        pop: Union[str, list]) -> float:
    """
    Compute nucleotide diversity (pi) within a population.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    pop : str or list
        Population name or list of indices
        
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
    n = len(pop_idx)
    
    # Calculate allele frequency
    pop_freq = cp.mean(pop_haplotypes, axis=0)
    
    # Pi = 2 * p * (1 - p) * n / (n - 1)
    if n > 1:
        pi_per_site = 2.0 * pop_freq * (1 - pop_freq) * n / (n - 1)
        return float(cp.mean(pi_per_site).get())
    else:
        return 0.0


def divergence_stats(haplotype_matrix: HaplotypeMatrix,
                    pop1: Union[str, list],
                    pop2: Union[str, list],
                    statistics: list = ['fst', 'dxy', 'da']) -> Dict[str, float]:
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
        
    Returns
    -------
    dict
        Dictionary of statistic names to values
    """
    results = {}
    
    for stat in statistics:
        if stat == 'fst':
            results['fst'] = fst(haplotype_matrix, pop1, pop2)
        elif stat == 'fst_hudson':
            results['fst_hudson'] = fst_hudson(haplotype_matrix, pop1, pop2)
        elif stat == 'fst_wc':
            results['fst_wc'] = fst_weir_cockerham(haplotype_matrix, pop1, pop2)
        elif stat == 'fst_nei':
            results['fst_nei'] = fst_nei(haplotype_matrix, pop1, pop2)
        elif stat == 'dxy':
            results['dxy'] = dxy(haplotype_matrix, pop1, pop2)
        elif stat == 'da':
            results['da'] = da(haplotype_matrix, pop1, pop2)
        elif stat == 'pi1':
            results['pi1'] = pi_within_population(haplotype_matrix, pop1)
        elif stat == 'pi2':
            results['pi2'] = pi_within_population(haplotype_matrix, pop2)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    return results


def pairwise_fst(haplotype_matrix: HaplotypeMatrix,
                 populations: Optional[list] = None,
                 method: str = 'hudson') -> Tuple[cp.ndarray, list]:
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
            fst_value = fst(haplotype_matrix, populations[i], populations[j], method)
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