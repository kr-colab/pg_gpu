"""
GPU-accelerated diversity and polymorphism statistics.

This module provides efficient computation of within-population genetic diversity
metrics including nucleotide diversity (π), Watterson's theta, Tajima's D, and
related statistics.
"""

import warnings
from collections import namedtuple

import numpy as np
import cupy as cp
from typing import Union, Optional, Dict, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix

PairwiseResult = namedtuple(
    'PairwiseResult',
    ['value', 'total_diffs', 'total_comps', 'total_missing', 'n_sites'])


def _pairwise_pi_components(haplotypes, n_total_sites=None, n_haplotypes_full=None):
    """Compute pairwise differences and comparisons across all sites.

    Parameters
    ----------
    haplotypes : cp.ndarray, shape (n_haplotypes, n_variants)
        Haplotype data with -1 for missing.
    n_total_sites : int, optional
        Total callable sites (variant + invariant). If provided, invariant
        sites contribute 0 diffs and C(n_haplotypes_full, 2) comps each.
    n_haplotypes_full : int, optional
        Full sample size (used for invariant site comps). Required when
        n_total_sites is set. Defaults to haplotypes.shape[0].

    Returns
    -------
    total_diffs : float
    total_comps : float
    total_missing : float
    n_sites : int
    """
    valid_mask = haplotypes >= 0
    n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)  # per site

    hap_clean = cp.where(valid_mask, haplotypes, 0)
    derived = cp.sum(hap_clean, axis=0).astype(cp.float64)
    ancestral = n_valid - derived

    # Per-site: diffs = derived * ancestral (number of mismatched pairs)
    site_diffs = derived * ancestral
    # Per-site: comps = C(n_valid, 2)
    site_comps = n_valid * (n_valid - 1) / 2.0

    # Only count sites with >= 2 valid samples
    usable = n_valid >= 2
    total_diffs = float(cp.sum(site_diffs[usable]).get())
    total_comps = float(cp.sum(site_comps[usable]).get())
    n_sites = int(cp.sum(usable).get())

    # Invariant site contribution
    if n_total_sites is not None:
        n_full = n_haplotypes_full or haplotypes.shape[0]
        n_invariant = n_total_sites - n_sites
        if n_invariant > 0:
            invariant_comps = n_invariant * (n_full * (n_full - 1) / 2.0)
            total_comps += invariant_comps
            n_sites += n_invariant

    # Total possible comparisons (for missing count)
    n_full = n_haplotypes_full or haplotypes.shape[0]
    possible_per_site = n_full * (n_full - 1) / 2.0
    total_possible = possible_per_site * n_sites
    total_missing = total_possible - total_comps

    return total_diffs, total_comps, total_missing, n_sites


def pi(haplotype_matrix: HaplotypeMatrix,
       population: Optional[Union[str, list]] = None,
       span_normalize: bool = True,
       missing_data: str = 'ignore',
       span_denominator: str = 'total',
       return_components: bool = False) -> Union[float, 'PairwiseResult']:
    """
    Calculate nucleotide diversity (pi) for a population.

    Nucleotide diversity is the average number of nucleotide differences
    per site between two randomly chosen sequences from the population.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    span_normalize : bool
        If True, normalize by genomic span; if False, return raw diversity
    missing_data : str
        'include' - Use all sites, calculate pi from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        'pairwise' - Comparison-counting normalization (pixy-style):
            pi = sum(diffs) / sum(comps) across all sites.
            Sites with more observed data contribute proportionally more.
            Invariant sites are included in the denominator when available.
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
    return_components : bool
        If True, return a PairwiseResult namedtuple with component statistics
        (value, total_diffs, total_comps, total_missing, n_sites).
        Only meaningful for 'pairwise' mode.

    Returns
    -------
    float or PairwiseResult
        Nucleotide diversity value, or components if return_components=True
    """
    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    # Handle missing data strategies
    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]

        if len(valid_sites) == 0:
            return PairwiseResult(0.0, 0, 0, 0, 0) if return_components else 0.0

        # Create subset with only valid sites
        matrix = matrix.get_subset(valid_sites)

    if missing_data == 'pairwise':
        if not matrix.has_invariant_info:
            warnings.warn(
                "No invariant site information available (n_total_sites not set). "
                "Pairwise-mode pi will be computed from variant sites only and "
                "may overestimate diversity. Use include_invariant=True when "
                "loading data to correct this.",
                stacklevel=2)
        total_diffs, total_comps, total_missing, n_sites = _pairwise_pi_components(
            matrix.haplotypes,
            n_total_sites=matrix.n_total_sites,
            n_haplotypes_full=matrix.num_haplotypes)
        pi_value = total_diffs / total_comps if total_comps > 0 else float('nan')
        if return_components:
            return PairwiseResult(pi_value, total_diffs, total_comps,
                                 total_missing, n_sites)
        return pi_value

    if missing_data == 'ignore':
        # Original behavior - use standard AFS calculation
        afs = allele_frequency_spectrum(matrix, missing_data='ignore')
        n_haplotypes = matrix.num_haplotypes

        i = cp.arange(1, n_haplotypes, dtype=cp.float64)
        weight = (2 * i * (n_haplotypes - i)) / (n_haplotypes * (n_haplotypes - 1))
        pi_value = cp.sum((weight * afs[1:n_haplotypes]).astype(cp.float64))

    else:  # missing_data == 'include'
        # Calculate pi per site using only non-missing data at each site (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)

        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)

        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)

        # Only consider sites with at least 2 valid samples
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            pi_value = cp.float64(0.0)
        else:
            # For each site, count derived alleles among valid samples
            # Set missing data to 0 for counting, but use valid_mask to exclude from counts
            hap_clean = cp.where(valid_mask, haplotypes, 0)

            # Count derived alleles per site (only among valid samples)
            derived_counts = cp.sum(hap_clean, axis=0)  # shape: (n_variants,)

            # Only compute for sites with valid data
            valid_sites = cp.where(sites_with_data)[0]
            n_valid = n_valid_per_site[valid_sites].astype(cp.float64)
            derived = derived_counts[valid_sites].astype(cp.float64)

            # Calculate frequencies
            freq_derived = derived / n_valid
            freq_ancestral = (n_valid - derived) / n_valid

            # Calculate pi per site with Nei's correction
            site_pi = 2 * freq_ancestral * freq_derived * n_valid / (n_valid - 1)

            # Sum across all valid sites
            pi_value = cp.sum(site_pi)

    # Apply span normalization
    if span_normalize:
        span = matrix.get_span(span_denominator)
        if span > 0:
            return float(pi_value / span)
        else:
            return float('nan')

    return float(pi_value.get() if hasattr(pi_value, 'get') else pi_value)


def theta_w(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True,
            missing_data: str = 'ignore',
            span_denominator: str = 'total',
            return_components: bool = False) -> Union[float, 'PairwiseResult']:
    """
    Calculate Watterson's theta for a population.

    Watterson's theta is an estimator of the population mutation rate based
    on the number of segregating sites.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    span_normalize : bool
        If True, normalize by genomic span; if False, return raw theta
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        'pairwise' - Per-site 1/a(n_i) for segregating sites, normalized
            by n_total_sites when invariant info is available.
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
    return_components : bool
        If True, return PairwiseResult with raw_theta as total_diffs and
        n_total_sites as total_comps. Only meaningful for 'pairwise' mode.

    Returns
    -------
    float or PairwiseResult
        Watterson's theta value
    """
    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    
    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    
    # Pairwise mode: per-site 1/a(n_i) for segregating sites
    if missing_data == 'pairwise':
        if not matrix.has_invariant_info:
            warnings.warn(
                "No invariant site information available (n_total_sites not set). "
                "Pairwise-mode theta_w will be computed from variant sites only.",
                stacklevel=2)
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid_per_site = cp.sum(valid_mask, axis=0)
        sites_with_data = n_valid_per_site >= 2
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        derived_counts = cp.sum(hap_clean, axis=0)

        # Identify segregating sites
        seg_mask = sites_with_data & (derived_counts > 0) & (derived_counts < n_valid_per_site)
        if not cp.any(seg_mask):
            raw_theta = 0.0
        else:
            seg_n_valid = n_valid_per_site[seg_mask]
            unique_n = cp.unique(seg_n_valid)
            theta_sum = 0.0
            for n in unique_n:
                count_with_n = int(cp.sum(seg_n_valid == n).get())
                a1 = float(cp.sum(1.0 / cp.arange(1, int(n), dtype=cp.float64)).get())
                theta_sum += count_with_n / a1
            raw_theta = theta_sum

        n_sites = matrix.n_total_sites if matrix.has_invariant_info else int(cp.sum(sites_with_data).get())
        theta_value = raw_theta / n_sites if n_sites > 0 else float('nan')

        if return_components:
            S = int(cp.sum(seg_mask).get())
            return PairwiseResult(theta_value, raw_theta, n_sites, 0, S)
        return theta_value

    # Handle missing data strategies
    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]

        if len(valid_sites) == 0:
            return PairwiseResult(0.0, 0, 0, 0, 0) if return_components else 0.0

        # Create subset with only valid sites
        matrix = matrix.get_subset(valid_sites)
        n_haplotypes = matrix.num_haplotypes

        # Count segregating sites in the filtered data
        seg_sites = segregating_sites(matrix, missing_data='exclude')

    elif missing_data == 'ignore':
        # Original behavior
        n_haplotypes = matrix.num_haplotypes
        seg_sites = segregating_sites(matrix, missing_data='ignore')

    else:  # missing_data == 'include'
        # Calculate theta using site-specific sample sizes (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
        
        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)
        
        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)
        
        # Only consider sites with at least 2 valid samples
        sites_with_data = n_valid_per_site >= 2
        
        if not cp.any(sites_with_data):
            theta = cp.float64(0.0)
        else:
            # For each site, check if it's segregating among valid samples
            # Count derived alleles among valid samples
            hap_clean = cp.where(valid_mask, haplotypes, 0)
            derived_counts = cp.sum(hap_clean, axis=0)
            
            # A site is segregating if 0 < derived_count < n_valid
            valid_sites = cp.where(sites_with_data)[0]
            n_valid_sites = n_valid_per_site[valid_sites]
            derived_sites = derived_counts[valid_sites]
            
            # Check which sites are segregating (not monomorphic)
            segregating_mask = (derived_sites > 0) & (derived_sites < n_valid_sites)
            
            if not cp.any(segregating_mask):
                theta = cp.float64(0.0)
            else:
                # For each segregating site, compute 1/a1 where a1 is harmonic number
                seg_n_valid = n_valid_sites[segregating_mask]
                
                # Compute harmonic numbers for each sample size
                # This is the most complex part to vectorize efficiently
                unique_n = cp.unique(seg_n_valid)
                theta_sum = cp.float64(0.0)
                
                for n in unique_n:
                    # Count how many sites have this sample size
                    count_with_n = cp.sum(seg_n_valid == n)
                    # Compute harmonic number for this sample size
                    a1 = cp.sum(1.0 / cp.arange(1, int(n), dtype=cp.float64))
                    # Add contribution
                    theta_sum += count_with_n / a1
                
                theta = theta_sum
        
    # For exclude and ignore modes, compute theta the standard way
    if missing_data in ['exclude', 'ignore']:
        a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes, dtype=cp.float64))
        theta = seg_sites / a1
    
    # Apply span normalization
    if span_normalize:
        span = matrix.get_span(span_denominator)
        if span > 0:
            return float(theta / span)
        else:
            return float('nan')
    
    return float(theta.get() if hasattr(theta, 'get') else theta)


def tajimas_d(haplotype_matrix: HaplotypeMatrix,
              population: Optional[Union[str, list]] = None,
              missing_data: str = 'ignore') -> float:
    """
    Calculate Tajima's D statistic.
    
    Tajima's D tests the neutral mutation hypothesis by comparing two
    estimates of the population mutation rate.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    float
        Tajima's D value
    """
    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    
    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    
    if missing_data == 'pairwise':
        # Pairwise Tajima's D: use raw pairwise pi and per-site theta_w
        # with harmonic mean of sample sizes for variance terms
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid_per_site = cp.sum(valid_mask, axis=0)
        sites_with_data = n_valid_per_site >= 2
        if not cp.any(sites_with_data):
            return float("nan")

        # Raw pairwise pi (sum of diffs / sum of comps, then multiply by comps
        # to get raw count-equivalent; or just use raw diffs directly)
        total_diffs, total_comps, _, _ = _pairwise_pi_components(
            haplotypes, n_total_sites=None)  # no invariant correction for D
        if total_comps == 0:
            return float("nan")
        # pi in "number of differences" scale: diffs/comps * C(n_mean, 2)
        # But Tajima's D uses raw pi = sum of per-site heterozygosity
        # Use the 'include'-style raw pi for consistency with theta_w
        pi_value = pi(matrix, span_normalize=False, missing_data='include')

        # Harmonic mean of sample sizes for variance terms
        valid_n = n_valid_per_site[sites_with_data]
        n_haplotypes = float(len(valid_n) / cp.sum(1.0 / valid_n).get())

        # Segregating sites
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        derived_counts = cp.sum(hap_clean, axis=0)
        seg_mask = sites_with_data & (derived_counts > 0) & (derived_counts < n_valid_per_site)
        S = int(cp.sum(seg_mask).get())
        if S == 0:
            return float("nan")

        # Raw theta_w using per-site sample sizes (same as 'include' mode)
        seg_n_valid = n_valid_per_site[seg_mask]
        unique_n = cp.unique(seg_n_valid)
        theta_raw = 0.0
        for n in unique_n:
            count_with_n = int(cp.sum(seg_n_valid == n).get())
            a1 = float(cp.sum(1.0 / cp.arange(1, int(n), dtype=cp.float64)).get())
            theta_raw += count_with_n / a1

        # Variance terms using harmonic mean n
        n = n_haplotypes
        a1 = sum(1.0 / i for i in range(1, int(round(n))))
        a2 = sum(1.0 / (i ** 2) for i in range(1, int(round(n))))
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1 ** 2))
        e1 = c1 / a1
        e2 = c2 / ((a1 ** 2) + a2)
        V = np.sqrt((e1 * S) + (e2 * S * (S - 1)))
        if V == 0:
            return float("nan")
        return (pi_value - theta_raw) / V

    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]

        if len(valid_sites) == 0:
            return float("nan")  # No valid sites

        # Create subset with only valid sites
        matrix = matrix.get_subset(valid_sites)

    # Get pi and theta with consistent missing data handling
    pi_value = pi(matrix, span_normalize=False, missing_data=missing_data)

    if missing_data == 'include':
        # For Tajima's D with missing data, we need to use an average sample size
        # Calculate the harmonic mean of sample sizes across sites
        n_valid_per_site = matrix.count_called(axis=0)

        # Filter to sites with at least 2 samples
        valid_site_mask = n_valid_per_site >= 2
        if not cp.any(valid_site_mask):
            return float("nan")

        # Harmonic mean of sample sizes
        n_haplotypes = float(len(n_valid_per_site[valid_site_mask]) /
                           cp.sum(1.0 / n_valid_per_site[valid_site_mask]).get())

        # Count segregating sites considering missing data (vectorized)
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid_per_site = cp.sum(valid_mask, axis=0)

        # Only consider sites with at least 2 valid samples
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            S = 0
        else:
            # Check which sites are segregating among valid samples
            hap_clean = cp.where(valid_mask, haplotypes, 0)
            derived_counts = cp.sum(hap_clean, axis=0)

            # Filter to sites with valid data
            valid_sites = cp.where(sites_with_data)[0]
            n_valid_sites = n_valid_per_site[valid_sites]
            derived_sites = derived_counts[valid_sites]

            # A site is segregating if 0 < derived_count < n_valid
            segregating_mask = (derived_sites > 0) & (derived_sites < n_valid_sites)
            S = int(cp.sum(segregating_mask).get())
    else:
        # For 'exclude' and 'ignore' modes
        n_haplotypes = matrix.num_haplotypes
        S = segregating_sites(matrix, missing_data=missing_data)
    
    # If no segregating sites, return NaN
    if S == 0:
        return float("nan")
    
    # Calculate theta directly (to avoid span normalization)
    a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes, dtype=cp.float64)) if isinstance(n_haplotypes, int) else sum(1.0 / i for i in range(1, int(n_haplotypes)))
    theta = S / a1
    
    # Variance term for Tajima's D
    n = n_haplotypes
    a2 = cp.sum(1.0 / (cp.arange(1, n, dtype=cp.float64) ** 2)) if isinstance(n, int) else sum(1.0 / (i ** 2) for i in range(1, int(n)))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1 ** 2))
    e1 = c1 / a1
    e2 = c2 / ((a1 ** 2) + a2)
    V = cp.sqrt((e1 * S) + (e2 * S * (S - 1))) if isinstance(S, cp.ndarray) else np.sqrt((e1 * S) + (e2 * S * (S - 1)))
    
    # Calculate D
    if V != 0:
        D = (pi_value - float(theta)) / float(V.get() if hasattr(V, 'get') else V)
        return D
    else:
        return float("nan")


def allele_frequency_spectrum(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            missing_data: str = 'include') -> cp.ndarray:
    """
    Calculate the allele frequency spectrum (AFS).
    
    The AFS is a histogram of allele frequencies across all sites.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Calculate AFS using available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    cp.ndarray
        Array where element i contains the number of sites with i derived alleles
    """
    # pairwise mode uses same per-site logic as include for AFS
    if missing_data == 'pairwise':
        missing_data = 'include'

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = missing_per_variant == 0
        
        if not cp.any(valid_sites):
            # No valid sites, return empty AFS
            return cp.zeros(matrix.num_haplotypes + 1, dtype=cp.int64)
        
        # Use only valid sites
        valid_haplotypes = matrix.haplotypes[:, valid_sites]
        n_haplotypes = matrix.num_haplotypes
        
        # Count derived alleles at each valid site
        freqs = cp.sum(valid_haplotypes, axis=0)
        
    elif missing_data == 'ignore':
        # Original behavior - count missing as ref (0)
        n_haplotypes = matrix.num_haplotypes
        freqs = cp.sum(cp.nan_to_num(matrix.haplotypes, nan=0).astype(cp.int32), axis=0)
        
    else:  # missing_data == 'include'
        # Build AFS considering variable sample sizes per site (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
        max_n = matrix.num_haplotypes
        
        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)
        
        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)
        
        # Only consider sites with valid data
        sites_with_data = n_valid_per_site > 0
        
        if not cp.any(sites_with_data):
            return cp.zeros(max_n + 1, dtype=cp.int64)
        
        # For sites with valid data, count derived alleles among valid samples
        # Set missing data to 0 for counting, but only count where valid
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        
        # Count derived alleles per site (only among valid samples)
        derived_counts = cp.sum(hap_clean, axis=0)  # shape: (n_variants,)
        
        # Filter to sites with valid data and check they're biallelic
        valid_sites = cp.where(sites_with_data)[0]
        derived_at_valid = derived_counts[valid_sites]
        n_valid_at_valid = n_valid_per_site[valid_sites]
        
        # Check biallelic assumption: derived count should be <= n_valid
        biallelic_mask = derived_at_valid <= n_valid_at_valid
        final_derived = derived_at_valid[biallelic_mask]
        
        # Create AFS histogram
        # Use bincount which is more efficient than a loop
        if len(final_derived) > 0:
            # Ensure derived counts don't exceed max_n
            final_derived = cp.minimum(final_derived, max_n)
            afs = cp.bincount(final_derived, minlength=max_n + 1)
            # Ensure correct size and type
            if len(afs) < max_n + 1:
                afs_full = cp.zeros(max_n + 1, dtype=cp.int64)
                afs_full[:len(afs)] = afs
                afs = afs_full
            else:
                afs = afs[:max_n + 1].astype(cp.int64)
        else:
            afs = cp.zeros(max_n + 1, dtype=cp.int64)
        
        return afs
    
    # For exclude and ignore modes, create standard histogram
    return cp.histogram(freqs, bins=cp.arange(n_haplotypes + 2))[0]


def segregating_sites(haplotype_matrix: HaplotypeMatrix,
                     population: Optional[Union[str, list]] = None,
                     missing_data: str = 'include') -> int:
    """
    Count the number of segregating sites.
    
    A site is segregating if it has more than one allele present.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Count sites as segregating based on non-missing data only
        'exclude' - Only count sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    int
        Number of segregating sites
    """
    # pairwise mode uses same per-site logic as include for counting
    if missing_data == 'pairwise':
        missing_data = 'include'

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    
    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    
    if missing_data == 'exclude':
        # Only count sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = missing_per_variant == 0
        
        # Count alleles only at valid sites
        valid_haplotypes = matrix.haplotypes[:, valid_sites]
        allele_counts = cp.sum(valid_haplotypes, axis=0)
        n_haplotypes = matrix.num_haplotypes
        
        # Site is segregating if not all 0s or all 1s
        segregating = (allele_counts > 0) & (allele_counts < n_haplotypes)
        
    elif missing_data == 'ignore':
        # Original behavior - missing treated as ref (0)
        allele_counts = cp.sum(matrix.haplotypes, axis=0)
        n_haplotypes = matrix.num_haplotypes
        segregating = (allele_counts > 0) & (allele_counts < n_haplotypes)
        
    else:  # missing_data == 'include'
        # Count segregating sites based on non-missing data only (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
        
        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)
        
        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)
        
        # Only consider sites with at least 2 valid samples
        sites_with_data = n_valid_per_site >= 2
        
        if not cp.any(sites_with_data):
            return 0
        
        # For each site, check if it's segregating among valid samples
        # Count derived alleles among valid samples
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        derived_counts = cp.sum(hap_clean, axis=0)
        
        # Filter to sites with valid data
        valid_sites = cp.where(sites_with_data)[0]
        n_valid_sites = n_valid_per_site[valid_sites]
        derived_sites = derived_counts[valid_sites]
        
        # A site is segregating if 0 < derived_count < n_valid
        segregating_mask = (derived_sites > 0) & (derived_sites < n_valid_sites)
        
        return int(cp.sum(segregating_mask).get())
    
    return int(cp.sum(segregating).get())


def singleton_count(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None,
                   missing_data: str = 'include') -> int:
    """
    Count the number of singleton variants.
    
    A singleton is a variant present in exactly one haplotype.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Count singletons based on non-missing data only
        'exclude' - Only count singletons at sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    int
        Number of singleton variants
    """
    # pairwise mode uses same per-site logic as include for counting
    if missing_data == 'pairwise':
        missing_data = 'include'

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        # Only count singletons at sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = missing_per_variant == 0
        
        if not cp.any(valid_sites):
            return 0  # No valid sites
        
        # Count alleles only at valid sites
        valid_haplotypes = matrix.haplotypes[:, valid_sites]
        allele_counts = cp.sum(valid_haplotypes, axis=0)
        
    elif missing_data == 'ignore':
        # Original behavior - missing treated as ref (0)
        allele_counts = cp.sum(matrix.haplotypes, axis=0)
        
    else:  # missing_data == 'include'
        # Count singletons based on non-missing data at each site (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
        
        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)
        
        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)
        
        # Only consider sites with at least 1 valid sample
        sites_with_data = n_valid_per_site >= 1
        
        if not cp.any(sites_with_data):
            return 0
        
        # For each site, count derived alleles among valid samples
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        derived_counts = cp.sum(hap_clean, axis=0)
        
        # Filter to sites with valid data
        valid_sites = cp.where(sites_with_data)[0]
        derived_at_valid = derived_counts[valid_sites]
        
        # Count sites where exactly 1 derived allele is present
        singleton_mask = derived_at_valid == 1
        
        return int(cp.sum(singleton_mask).get())
    
    # For exclude and ignore modes
    return int(cp.sum(allele_counts == 1).get())


def diversity_stats(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None,
                   statistics: list = ['pi', 'theta_w', 'tajimas_d'],
                   span_normalize: bool = True,
                   missing_data: str = 'include',
                   span_denominator: str = 'total') -> Dict[str, float]:
    """
    Compute multiple diversity statistics at once.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    statistics : list
        List of statistics to compute
    span_normalize : bool
        Whether to normalize pi and theta_w by genomic span
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
        
    Returns
    -------
    dict
        Dictionary mapping statistic names to values
    """
    results = {}
    
    for stat in statistics:
        if stat == 'pi':
            results['pi'] = pi(haplotype_matrix, population, span_normalize, missing_data, span_denominator)
        elif stat == 'theta_w':
            results['theta_w'] = theta_w(haplotype_matrix, population, span_normalize, missing_data, span_denominator)
        elif stat == 'tajimas_d':
            results['tajimas_d'] = tajimas_d(haplotype_matrix, population, missing_data)
        elif stat == 'segregating_sites':
            results['segregating_sites'] = segregating_sites(haplotype_matrix, population, missing_data)
        elif stat == 'singletons':
            results['singletons'] = singleton_count(haplotype_matrix, population, missing_data)
        elif stat == 'n_variants':
            if population is not None:
                matrix = _get_population_matrix(haplotype_matrix, population)
                results['n_variants'] = matrix.num_variants
            else:
                results['n_variants'] = haplotype_matrix.num_variants
        elif stat == 'haplotype_diversity':
            results['haplotype_diversity'] = haplotype_diversity(haplotype_matrix, population)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    return results


def fay_wus_h(haplotype_matrix: HaplotypeMatrix,
              population: Optional[Union[str, list]] = None,
              missing_data: str = 'include') -> float:
    """
    Calculate Fay and Wu's H statistic.
    
    Tests for an excess of high-frequency derived alleles, which can indicate
    positive selection.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        'pairwise' - Per-site sample sizes (same as include for H)

    Returns
    -------
    float
        Fay and Wu's H value
    """
    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]

        if len(valid_sites) == 0:
            return float("nan")  # No valid sites

        # Create subset with only valid sites
        matrix = matrix.get_subset(valid_sites)

    # For composite test statistics, 'pairwise' uses 'include' internally
    # so all components are on the same raw-sum scale.
    md = 'include' if missing_data == 'pairwise' else missing_data
    th_val = theta_h(matrix, span_normalize=False, missing_data=md)
    pi_value = pi(matrix, span_normalize=False, missing_data=md)

    # H = pi - theta_H
    return pi_value - th_val


def haplotype_diversity(haplotype_matrix: HaplotypeMatrix,
                       population: Optional[Union[str, list]] = None,
                       missing_data: str = 'ignore') -> float:
    """
    Calculate haplotype diversity for a population.

    Haplotype diversity is defined as 1 - sum(p_i^2) where p_i is the
    frequency of the i-th unique haplotype in the population.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' or 'pairwise' - exclude haplotypes with any missing data
        'exclude' - filter to sites with no missing data
        'ignore' - treat -1 as 0

    Returns
    -------
    float
        Haplotype diversity value
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    haplotypes = matrix.haplotypes
    haplotypes_cpu = haplotypes.get() if matrix.device == 'GPU' else haplotypes

    if missing_data == 'exclude':
        missing_per_var = np.sum(haplotypes_cpu < 0, axis=0)
        complete = missing_per_var == 0
        haplotypes_cpu = haplotypes_cpu[:, complete]
    elif missing_data == 'include':
        # exclude haplotypes with any missing site
        has_missing = np.any(haplotypes_cpu < 0, axis=1)
        haplotypes_cpu = haplotypes_cpu[~has_missing]
    else:
        haplotypes_cpu = np.maximum(haplotypes_cpu, 0)

    n_haplotypes = haplotypes_cpu.shape[0]
    if n_haplotypes <= 1:
        return 0.0

    from collections import Counter
    hap_strings = [''.join(map(str, hap)) for hap in haplotypes_cpu]
    hap_counts = Counter(hap_strings)

    frequencies = np.array(list(hap_counts.values())) / n_haplotypes
    diversity = (1.0 - np.sum(frequencies ** 2)) * n_haplotypes / (n_haplotypes - 1)

    return float(diversity)


_get_population_matrix = get_population_matrix


def theta_h(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True,
            missing_data: str = 'ignore') -> float:
    """Compute theta_H (homozygosity-based diversity estimator).

    theta_H = sum_i [ i^2 * S_i ] * 2 / (n*(n-1)) where S_i is the count
    of variants with derived allele count i. Used to compute Fay and Wu's H.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    span_normalize : bool
        If True, normalize by genomic span.
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]
        if len(valid_sites) == 0:
            return 0.0
        matrix = matrix.get_subset(valid_sites)

    if missing_data in ('include', 'pairwise'):
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)

        usable = n_valid > 1
        th = cp.sum(2.0 * dac[usable] ** 2 / (n_valid[usable] * (n_valid[usable] - 1)))
    else:
        hap = cp.maximum(matrix.haplotypes, 0)
        n = hap.shape[0]
        dac = cp.sum(hap, axis=0).astype(cp.float64)
        th = cp.sum(dac * dac) * 2.0 / (n * (n - 1))

    if span_normalize:
        span = matrix.get_span()
        if span > 0:
            th = th / span

    return float(th.get())


def theta_l(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True,
            missing_data: str = 'ignore') -> float:
    """Compute theta_L diversity estimator.

    theta_L = sum_i(i * xi_i) / (n - 1), where xi_i is the count of
    sites with derived allele count i. Weights variants linearly by
    derived allele frequency, bridging theta_pi and theta_H.

    With missing data ('include'/'pairwise'), each site contributes
    d_i / (n_i - 1) using its own sample size.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (8).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    span_normalize : bool
        If True, normalize by genomic span.
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]
        if len(valid_sites) == 0:
            return 0.0
        matrix = matrix.get_subset(valid_sites)

    if missing_data in ('include', 'pairwise'):
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        hap_clean = cp.where(valid_mask, haplotypes, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)

        usable = n_valid > 1
        tl = cp.sum(dac[usable] / (n_valid[usable] - 1))
    else:
        hap = cp.maximum(matrix.haplotypes, 0)
        n = hap.shape[0]
        dac = cp.sum(hap, axis=0).astype(cp.float64)
        tl = cp.sum(dac) / (n - 1)

    if span_normalize:
        span = matrix.get_span()
        if span > 0:
            tl = tl / span

    return float(tl.get())


def _effective_n_and_S(matrix, missing_data):
    """Compute effective sample size and segregating site count.

    For 'include'/'pairwise': uses harmonic mean of per-site n_valid
    as effective n, and counts segregating sites among valid data.
    For 'exclude'/'ignore': uses the fixed sample size.

    Returns (n_eff, S, matrix) where matrix may be subset-filtered.
    """
    if missing_data == 'exclude':
        missing_per_variant = matrix.count_missing(axis=0)
        valid_idx = cp.where(missing_per_variant == 0)[0]
        if len(valid_idx) == 0:
            return 0.0, 0.0, matrix
        matrix = matrix.get_subset(valid_idx)

    if missing_data in ('include', 'pairwise'):
        haplotypes = matrix.haplotypes
        valid_mask = haplotypes >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        usable = n_valid >= 2
        if not cp.any(usable):
            return 0.0, 0.0, matrix

        hap_clean = cp.where(valid_mask, haplotypes, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)
        seg_mask = usable & (dac > 0) & (dac < n_valid)
        S = float(cp.sum(seg_mask).get())

        # harmonic mean of per-site sample sizes (across usable sites)
        n_usable = n_valid[usable]
        n_eff = float(len(n_usable)) / float(cp.sum(1.0 / n_usable).get())
    else:
        # 'ignore' mode — treat missing as ref
        n_eff = float(matrix.num_haplotypes)
        hap = cp.maximum(matrix.haplotypes, 0)
        dac = cp.sum(hap, axis=0).astype(cp.float64)
        n = int(matrix.num_haplotypes)
        seg_mask = (dac > 0) & (dac < n)
        S = float(cp.sum(seg_mask).get())

    return n_eff, S, matrix


def normalized_fay_wus_h(haplotype_matrix: HaplotypeMatrix,
                         population: Optional[Union[str, list]] = None,
                         missing_data: str = 'ignore') -> float:
    """Compute normalized Fay and Wu's H (H*).

    H = theta_pi - theta_H, normalized by its standard deviation under
    the standard neutral model. The normalization allows comparison
    across samples with different numbers of segregating sites.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (11).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes, harmonic mean n
            for variance terms
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
        Normalized H*. Negative values indicate excess high-frequency
        derived alleles (directional selection signal).
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    n_eff, S, matrix = _effective_n_and_S(matrix, missing_data)

    if S < 2:
        return 0.0

    # H = pi - theta_H (both raw, not span-normalized)
    # For composite tests, 'pairwise' uses 'include' internally
    md = 'include' if missing_data == 'pairwise' else missing_data
    pi_val = pi(matrix, span_normalize=False, missing_data=md)
    th_val = theta_h(matrix, span_normalize=False, missing_data=md)
    H = pi_val - th_val

    # variance of H under neutrality (Zeng et al. 2006, below eq 11)
    n_f = n_eff
    n = int(round(n_f))
    a_n = sum(1.0 / i for i in range(1, n))
    b_n = sum(1.0 / (i * i) for i in range(1, n))

    n2 = n_f * n_f
    n3 = n2 * n_f

    e1 = (n_f - 2) / (6 * (n_f - 1))

    e2_num = (18 * n2 * (3 * n_f + 2) * b_n
              - (88 * n3 + 9 * n2 - 13 * n_f + 6))
    e2_den = 9 * n_f * (n2 - n_f) * a_n + 9 * n_f * (n2 - n_f) * b_n
    e2 = e2_num / e2_den if e2_den != 0 else 0.0

    var_H = e1 * S + e2 * S * (S - 1)

    if var_H <= 0:
        return 0.0

    return float(H / (var_H ** 0.5))


def zeng_e(haplotype_matrix: HaplotypeMatrix,
           population: Optional[Union[str, list]] = None,
           missing_data: str = 'ignore') -> float:
    """Compute Zeng's E test statistic.

    E = theta_L - theta_W, normalized by its standard deviation.
    Contrasts high-frequency variants (theta_L) against rare variants
    (theta_W). Negative values indicate excess high-frequency derived
    alleles relative to rare variants.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (13).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site sample sizes, harmonic mean n
            for variance terms
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
        Normalized E statistic.
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    n_eff, S, matrix = _effective_n_and_S(matrix, missing_data)

    if S < 2:
        return 0.0

    # theta_L and theta_W (raw, not span-normalized)
    # For composite tests, 'pairwise' uses 'include' internally
    md = 'include' if missing_data == 'pairwise' else missing_data
    tl = theta_l(matrix, span_normalize=False, missing_data=md)
    tw = theta_w(matrix, span_normalize=False, missing_data=md)

    E = tl - tw

    # variance of E under neutrality (Zeng et al. 2006, eq 14)
    n_f = n_eff
    n = int(round(n_f))
    a_n = sum(1.0 / i for i in range(1, n))
    b_n = sum(1.0 / (i * i) for i in range(1, n))

    theta = S / a_n

    e1 = (n_f / (2 * (n_f - 1)) - 1.0 / a_n)
    e2_num = (b_n / (a_n * a_n)
              + 2 * (n_f / (n_f - 1)) ** 2 * b_n
              - 2 * (n_f * b_n - n_f + 1) / ((n_f - 1) * a_n)
              - (3 * n_f + 1) / (n_f - 1))
    e2 = e2_num / (a_n * a_n + b_n)

    var_E = e1 * theta + e2 * theta * theta

    if var_E <= 0:
        return 0.0

    return float(E / (var_E ** 0.5))


def zeng_dh(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            missing_data: str = 'ignore') -> float:
    """Compute Zeng's DH joint test statistic.

    Combines Tajima's D and Fay & Wu's H into a single test with
    improved power to detect directional selection. Defined as the
    product D * H when both are negative, zero otherwise.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (15).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        Passed through to tajimas_d and fay_wus_h.

    Returns
    -------
    float
        DH statistic. Positive when both D and H are negative
        (consistent with a selective sweep).
    """
    D = tajimas_d(haplotype_matrix, population, missing_data=missing_data)
    H = fay_wus_h(haplotype_matrix, population, missing_data=missing_data)

    # DH is the product when both are negative (sweep signal)
    if D < 0 and H < 0:
        return float(D * H)
    else:
        return 0.0


def max_daf(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            missing_data: str = 'ignore') -> float:
    """Maximum derived allele frequency across all variants.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site n_valid for frequency
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
        Maximum DAF in [0, 1].
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'include':
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        dac = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
        usable = n_valid > 0
        freqs = cp.where(usable, dac / n_valid, 0.0)
    elif missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap_clean = cp.where(hap >= 0, hap, 0)
        n = hap.shape[0]
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)
        freqs = cp.where(complete, dac / n, -1.0)  # -1 so excluded sites aren't max
    else:
        hap_clean = cp.maximum(hap, 0)
        n = hap.shape[0]
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)
        freqs = dac / n

    return float(cp.max(freqs).get())


def haplotype_count(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None,
                    missing_data: str = 'ignore') -> int:
    """Count distinct haplotypes.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - exclude haplotypes with any missing
        'exclude' - filter to sites with no missing data
        'ignore' - treat -1 as 0

    Returns
    -------
    int
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap_cpu = matrix.haplotypes.get().astype(np.int8)

    if missing_data == 'exclude':
        missing_per_var = np.sum(hap_cpu < 0, axis=0)
        hap_cpu = hap_cpu[:, missing_per_var == 0]
    elif missing_data == 'include':
        has_missing = np.any(hap_cpu < 0, axis=1)
        hap_cpu = hap_cpu[~has_missing]
    else:
        hap_cpu = np.maximum(hap_cpu, 0)

    hap_bytes = np.array([row.tobytes() for row in hap_cpu])
    return len(np.unique(hap_bytes))


def daf_histogram(matrix, n_bins: int = 20,
                  population: Optional[Union[str, list]] = None,
                  missing_data: str = 'ignore'):
    """Normalized histogram of derived allele frequencies.

    Accepts HaplotypeMatrix or GenotypeMatrix. For diploid data,
    DAF = sum(genotypes) / (2 * n_individuals).

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    n_bins : int
        Number of frequency bins spanning [0, 1].
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site n_valid for frequency
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    hist : ndarray, float64, shape (n_bins,)
        Normalized counts (sum to 1).
    bin_edges : ndarray, float64, shape (n_bins + 1,)
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    from .genotype_matrix import GenotypeMatrix

    if isinstance(matrix, GenotypeMatrix):
        return _daf_histogram_diploid(matrix, n_bins, population)

    if population is not None:
        matrix = _get_population_matrix(matrix, population)

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'include':
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        dac = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.float64)
        usable = n_valid > 0
        dafs = cp.where(usable, dac / n_valid, 0.0)
    elif missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap_clean = cp.where(hap >= 0, hap, 0)
        n = hap.shape[0]
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)
        dafs = dac / n
        dafs = dafs[complete]  # filter to complete sites
    else:
        hap_clean = cp.maximum(hap, 0)
        n = hap.shape[0]
        dafs = cp.sum(hap_clean, axis=0).astype(cp.float64) / n

    return _histogram_from_dafs(dafs, n_bins)


def diplotype_frequency_spectrum(genotype_matrix,
                                 population: Optional[Union[str, list]] = None):
    """Count distinct multi-locus genotype patterns (diplotypes).

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    population : str or list, optional

    Returns
    -------
    freqs : ndarray, float64, sorted descending
        Diplotype frequencies.
    n_diplotypes : int
        Number of distinct diplotypes.
    """
    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if isinstance(geno, cp.ndarray):
        geno = geno.get()

    geno = np.maximum(geno, 0).astype(np.int8)
    n_ind = geno.shape[0]

    geno_bytes = np.array([row.tobytes() for row in geno])
    _, counts = np.unique(geno_bytes, return_counts=True)

    freqs = counts / n_ind
    freqs = np.sort(freqs)[::-1]

    return freqs, len(counts)


def _histogram_from_dafs(dafs, n_bins):
    """Shared: compute normalized histogram from DAF CuPy array."""
    bin_edges = cp.linspace(0, 1, n_bins + 1)
    hist = cp.histogram(dafs, bins=bin_edges)[0].astype(cp.float64)
    total = cp.sum(hist)
    if total > 0:
        hist = hist / total
    return hist.get(), bin_edges.get()


def _daf_histogram_diploid(genotype_matrix, n_bins=20, population=None):
    """DAF histogram from diploid genotypes (internal)."""
    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    geno = cp.maximum(geno, 0).astype(cp.float64)
    n_ind = geno.shape[0]
    dafs = cp.sum(geno, axis=0) / (2.0 * n_ind)

    return _histogram_from_dafs(dafs, n_bins)


# backward compat alias
daf_histogram_diploid = _daf_histogram_diploid


# Summary statistics combinations commonly used

def neutrality_tests(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None,
                    missing_data: str = 'include') -> Dict[str, float]:
    """
    Compute common neutrality test statistics.
    
    Returns Tajima's D, Fay and Wu's H, and related values.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
        'ignore' - Treat missing as reference allele (original behavior)
        
    Returns
    -------
    dict
        Dictionary with neutrality test results
    """
    return {
        'tajimas_d': tajimas_d(haplotype_matrix, population, missing_data),
        'fay_wus_h': fay_wus_h(haplotype_matrix, population, missing_data),
        'pi': pi(haplotype_matrix, population, span_normalize=False, missing_data=missing_data),
        'theta_w': theta_w(haplotype_matrix, population, span_normalize=False, missing_data=missing_data),
        'segregating_sites': segregating_sites(haplotype_matrix, population, missing_data)
    }


def heterozygosity_expected(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            missing_data: str = 'ignore'):
    """
    Compute expected heterozygosity (gene diversity) per variant.

    He = 1 - sum(p_i^2) for each variant, where p_i are allele frequencies.
    For biallelic sites this simplifies to He = 2*p*(1-p).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'ignore' - treat missing as reference allele
        'exclude' - NaN at sites with any missing data
        'include' or 'pairwise' - per-site n_valid for frequency calculation

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Expected heterozygosity per variant.
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    has_missing = cp.any(hap < 0)

    if missing_data == 'include' and has_missing:
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        hap_clean = cp.where(valid_mask, hap, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.float64)
        p = cp.where(n_valid > 0, dac / n_valid, 0.0)
        he = 2.0 * p * (1.0 - p)
        he = cp.where(n_valid >= 2, he, cp.nan)
    else:
        n = hap.shape[0]
        hap_for_sum = cp.where(hap >= 0, hap, 0) if has_missing else hap
        dac = cp.sum(hap_for_sum, axis=0).astype(cp.float64)
        p = dac / n
        he = 2.0 * p * (1.0 - p)

        if missing_data == 'exclude' and has_missing:
            missing_per_var = cp.sum(hap < 0, axis=0)
            he[missing_per_var > 0] = cp.nan

    return he.get()


def heterozygosity_observed(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            ploidy: int = 2,
                            missing_data: str = 'include'):
    """
    Compute observed heterozygosity per variant.

    Assumes consecutive haplotypes belong to the same individual
    (standard for diploid VCF data). A site is heterozygous in an
    individual if the two haplotypes differ.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    ploidy : int
        Ploidy level. Default 2 (diploid).
    missing_data : str
        'include' or 'pairwise' - skip missing individuals per site (default)
        'exclude' - NaN at sites with any missing data
        'ignore' - treat -1 as 0

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Observed heterozygosity per variant.
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    n_hap = hap.shape[0]

    if n_hap % ploidy != 0:
        raise ValueError(
            f"Number of haplotypes ({n_hap}) not divisible by ploidy ({ploidy})")

    if missing_data == 'ignore':
        hap = cp.maximum(hap, 0)

    n_individuals = n_hap // ploidy

    if ploidy == 2:
        h1 = hap[0::2]
        h2 = hap[1::2]

        valid = (h1 >= 0) & (h2 >= 0)
        het = (h1 != h2) & valid
        n_valid = cp.sum(valid, axis=0).astype(cp.float64)
        n_het = cp.sum(het, axis=0).astype(cp.float64)

        ho = cp.where(n_valid > 0, n_het / n_valid, cp.nan)
    else:
        n_variants = hap.shape[1]
        n_het = cp.zeros(n_variants, dtype=cp.float64)
        n_valid_ind = cp.zeros(n_variants, dtype=cp.float64)

        for ind in range(n_individuals):
            ind_haps = hap[ind * ploidy:(ind + 1) * ploidy]
            all_valid = cp.all(ind_haps >= 0, axis=0)
            all_same = cp.all(ind_haps == ind_haps[0:1], axis=0)
            n_valid_ind += all_valid.astype(cp.float64)
            n_het += (all_valid & ~all_same).astype(cp.float64)

        ho = cp.where(n_valid_ind > 0, n_het / n_valid_ind, cp.nan)

    if missing_data == 'exclude':
        has_missing = cp.any(hap < 0)
        if has_missing:
            missing_per_var = cp.sum(hap < 0, axis=0)
            ho[missing_per_var > 0] = cp.nan

    return ho.get()


def inbreeding_coefficient(haplotype_matrix: HaplotypeMatrix,
                           population: Optional[Union[str, list]] = None,
                           ploidy: int = 2,
                           missing_data: str = 'include'):
    """
    Compute Wright's inbreeding coefficient F per variant.

    F = 1 - Ho/He, where Ho is observed heterozygosity and He is
    expected heterozygosity.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    ploidy : int
        Ploidy level for observed heterozygosity computation.
    missing_data : str
        Passed to heterozygosity_expected and heterozygosity_observed.

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Inbreeding coefficient per variant. NaN where He = 0.
    """
    ho = heterozygosity_observed(haplotype_matrix, population, ploidy,
                                 missing_data=missing_data)
    he = heterozygosity_expected(haplotype_matrix, population,
                                 missing_data=missing_data)

    # F = 1 - Ho/He; undefined where He = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(he > 0, 1.0 - ho / he, np.nan)

    return f


def mu_var(haplotype_matrix: HaplotypeMatrix,
           window_length: Optional[float] = None,
           population: Optional[Union[str, list]] = None) -> float:
    """mu_VAR: SNP density statistic (RAiSD).

    Number of SNPs per base pair. Elevated near sweeps due to
    hitchhiking effects on local variant density.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    window_length : float, optional
        Window length in bp. If None, uses chrom_end - chrom_start.
    population : str or list, optional

    Returns
    -------
    float
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    n_snps = matrix.num_variants
    if window_length is None:
        window_length = matrix.chrom_end - matrix.chrom_start
    if window_length <= 0:
        return 0.0

    return float(n_snps / window_length)


def mu_sfs(haplotype_matrix: HaplotypeMatrix,
           population: Optional[Union[str, list]] = None,
           missing_data: str = 'ignore') -> float:
    """mu_SFS: fraction of SNPs at SFS edges (RAiSD).

    Counts singletons (DAC=1) and near-fixed variants (DAC=n-1),
    divided by total segregating sites. Elevated near selective sweeps.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site n_valid for edge classification
        'exclude' - only sites with no missing data
        'ignore' - treat missing as reference allele

    Returns
    -------
    float
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'include':
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0)
        dac = cp.sum(cp.where(valid_mask, hap, 0), axis=0)
        usable = n_valid >= 2
        is_seg = usable & (dac > 0) & (dac < n_valid)
        is_edge = usable & ((dac == 1) | (dac == n_valid - 1))
    elif missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap_clean = cp.where(hap >= 0, hap, 0)
        n = hap.shape[0]
        dac = cp.sum(hap_clean, axis=0)
        is_seg = complete & (dac > 0) & (dac < n)
        is_edge = complete & ((dac == 1) | (dac == n - 1))
    else:
        hap_clean = cp.maximum(hap, 0)
        n = hap.shape[0]
        dac = cp.sum(hap_clean, axis=0)
        is_seg = (dac > 0) & (dac < n)
        is_edge = (dac == 1) | (dac == n - 1)

    n_seg = cp.sum(is_seg)
    if int(n_seg.get()) == 0:
        return 0.0

    n_edge = cp.sum(is_edge)
    return float((n_edge.astype(cp.float64) / n_seg.astype(cp.float64)).get())