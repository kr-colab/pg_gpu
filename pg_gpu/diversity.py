"""
GPU-accelerated diversity and polymorphism statistics.

This module provides efficient computation of within-population genetic diversity
metrics including nucleotide diversity (π), Watterson's theta, Tajima's D, and
related statistics.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional, Dict, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix


def pi(haplotype_matrix: HaplotypeMatrix,
       population: Optional[Union[str, list]] = None,
       span_normalize: bool = True,
       missing_data: str = 'ignore',
       span_denominator: str = 'total') -> float:
    """
    Calculate nucleotide diversity (π) for a population.
    
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
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
        
    Returns
    -------
    float
        Nucleotide diversity value
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
            return 0.0  # No valid sites
            
        # Create subset with only valid sites
        matrix = matrix.get_subset(valid_sites)
    
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
            
            # Calculate allele frequencies and pi for each site
            # For biallelic sites: freq_0 = (n_valid - derived) / n_valid, freq_1 = derived / n_valid
            # pi = 2 * freq_0 * freq_1 * n_valid / (n_valid - 1)
            
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
            span_denominator: str = 'total') -> float:
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
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
        
    Returns
    -------
    float
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
    
    # Handle missing data strategies
    if missing_data == 'exclude':
        # Filter to sites with no missing data
        missing_per_variant = matrix.count_missing(axis=0)
        valid_sites = cp.where(missing_per_variant == 0)[0]
        
        if len(valid_sites) == 0:
            return 0.0  # No valid sites
            
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
    
    if missing_data in ['exclude', 'ignore']:
        # For exclude (filtered data) and ignore (treat missing as ref)
        n = matrix.num_haplotypes
        
        # Get allele frequencies
        allele_counts = cp.sum(matrix.haplotypes, axis=0)
        
        # Calculate theta_H (uses squared frequencies) - fully vectorized
        # theta_H = sum over sites of 2 * i^2 / (n * (n-1)) where i is the derived allele count
        # This is equivalent to: sum over i of 2 * i^2 * count(i) / (n * (n-1))
        
        # Direct vectorized computation: sum over all sites of 2 * count^2 / (n * (n-1))
        # Only include segregating sites (0 < count < n)
        segregating_mask = (allele_counts > 0) & (allele_counts < n)
        segregating_counts = allele_counts[segregating_mask].astype(cp.float64)
        
        if len(segregating_counts) > 0:
            # Calculate theta_H: sum of 2 * i^2 / (n * (n-1)) for all segregating sites
            theta_h = float(cp.sum(2.0 * segregating_counts * segregating_counts / (n * (n - 1))).get())
        else:
            theta_h = 0.0
    
    else:  # missing_data == 'include'
        # Calculate theta_H considering variable sample sizes per site (vectorized)
        haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
        
        # Create mask for valid (non-missing) data
        valid_mask = haplotypes >= 0  # shape: (n_haplotypes, n_variants)
        
        # Count valid samples per site
        n_valid_per_site = cp.sum(valid_mask, axis=0)  # shape: (n_variants,)
        
        # Only consider sites with at least 2 valid samples
        sites_with_data = n_valid_per_site > 1
        
        if not cp.any(sites_with_data):
            theta_h = 0.0
        else:
            # For each site, count derived alleles among valid samples
            hap_clean = cp.where(valid_mask, haplotypes, 0)
            derived_counts = cp.sum(hap_clean, axis=0)
            
            # Filter to sites with valid data
            valid_sites = cp.where(sites_with_data)[0]
            n_valid_sites = n_valid_per_site[valid_sites].astype(cp.float64)
            derived_sites = derived_counts[valid_sites].astype(cp.float64)
            
            # Filter to sites with at least one derived allele
            has_derived = derived_sites > 0
            if cp.any(has_derived):
                n_valid_final = n_valid_sites[has_derived]
                derived_final = derived_sites[has_derived]
                
                # Calculate theta_H contribution for each site
                # theta_H = sum over sites of 2 * i^2 / (n * (n-1))
                site_contributions = 2.0 * derived_final * derived_final / (n_valid_final * (n_valid_final - 1))
                theta_h = float(cp.sum(site_contributions).get())
            else:
                theta_h = 0.0
    
    # Get pi with consistent missing data handling
    pi_value = pi(matrix, span_normalize=False, missing_data=missing_data)
    
    # H = pi - theta_H
    return pi_value - theta_h


def haplotype_diversity(haplotype_matrix: HaplotypeMatrix,
                       population: Optional[Union[str, list]] = None) -> float:
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
        
    Returns
    -------
    float
        Haplotype diversity value
    """
    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    
    # Ensure on GPU for efficient computation
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    
    # Get haplotypes array
    haplotypes = matrix.haplotypes  # shape: (n_haplotypes, n_variants)
    n_haplotypes = matrix.num_haplotypes
    
    if n_haplotypes <= 1:
        return 0.0
    
    # Find unique haplotypes and their counts
    # Convert each haplotype to a string representation for hashing
    if matrix.device == 'GPU':
        # Convert to CPU for unique operations (CuPy doesn't have good unique support for 2D)
        haplotypes_cpu = haplotypes.get()
    else:
        haplotypes_cpu = haplotypes
    
    # Convert haplotypes to string representation for finding uniques
    hap_strings = [''.join(map(str, hap)) for hap in haplotypes_cpu]
    
    # Count unique haplotypes
    from collections import Counter
    hap_counts = Counter(hap_strings)
    
    # Calculate frequencies
    frequencies = np.array(list(hap_counts.values())) / n_haplotypes
    
    # Calculate diversity: 1 - sum(p_i^2)
    # Apply Nei's correction for finite sample size: multiply by n/(n-1)
    diversity = (1.0 - np.sum(frequencies ** 2)) * n_haplotypes / (n_haplotypes - 1)
    
    return float(diversity)


_get_population_matrix = get_population_matrix


def theta_h(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True) -> float:
    """Compute theta_H (homozygosity-based diversity estimator).

    theta_H = sum_i [ i^2 * S_i ] * 2 / (n*(n-1)) where S_i is the count
    of variants with derived allele count i. Used to compute Fay and Wu's H.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    span_normalize : bool
        If True, normalize by genomic span.

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

    hap = cp.maximum(matrix.haplotypes, 0)
    n = hap.shape[0]
    dac = cp.sum(hap, axis=0).astype(cp.float64)

    # theta_H = (2 / (n*(n-1))) * sum(dac_i^2) per site, summed
    th = cp.sum(dac * dac) * 2.0 / (n * (n - 1))

    if span_normalize:
        span = matrix.get_span()
        if span > 0:
            th = th / span

    return float(th.get())


def max_daf(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None) -> float:
    """Maximum derived allele frequency across all variants.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional

    Returns
    -------
    float
        Maximum DAF in [0, 1].
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = cp.maximum(matrix.haplotypes, 0)
    n = hap.shape[0]
    dac = cp.sum(hap, axis=0).astype(cp.float64)
    freqs = dac / n

    return float(cp.max(freqs).get())


def haplotype_count(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None) -> int:
    """Count distinct haplotypes.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional

    Returns
    -------
    int
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap_cpu = matrix.haplotypes.get().astype(np.int8)
    hap_bytes = np.array([row.tobytes() for row in hap_cpu])
    return len(np.unique(hap_bytes))


def daf_histogram(haplotype_matrix: HaplotypeMatrix,
                  n_bins: int = 20,
                  population: Optional[Union[str, list]] = None):
    """Normalized histogram of derived allele frequencies.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    n_bins : int
        Number of frequency bins spanning [0, 1].
    population : str or list, optional

    Returns
    -------
    hist : ndarray, float64, shape (n_bins,)
        Normalized counts (sum to 1).
    bin_edges : ndarray, float64, shape (n_bins + 1,)
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = cp.maximum(matrix.haplotypes, 0)
    n = hap.shape[0]
    dafs = cp.sum(hap, axis=0).astype(cp.float64) / n

    bin_edges = cp.linspace(0, 1, n_bins + 1)
    hist = cp.histogram(dafs, bins=bin_edges)[0].astype(cp.float64)

    total = cp.sum(hist)
    if total > 0:
        hist = hist / total

    return hist.get(), bin_edges.get()


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


def daf_histogram_diploid(genotype_matrix,
                          n_bins: int = 20,
                          population: Optional[Union[str, list]] = None):
    """Normalized DAF histogram from diploid genotypes.

    DAF = sum(genotypes) / (2 * n_individuals) since each individual
    contributes 2 alleles.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    n_bins : int
    population : str or list, optional

    Returns
    -------
    hist : ndarray, float64, shape (n_bins,)
    bin_edges : ndarray, float64, shape (n_bins + 1,)
    """
    from .genotype_matrix import GenotypeMatrix

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

    bin_edges = cp.linspace(0, 1, n_bins + 1)
    hist = cp.histogram(dafs, bins=bin_edges)[0].astype(cp.float64)

    total = cp.sum(hist)
    if total > 0:
        hist = hist / total

    return hist.get(), bin_edges.get()


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
        'exclude' - skip sites with any missing data

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Expected heterozygosity per variant.
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    n = hap.shape[0]

    # only create a cleaned copy if missing data actually exists
    has_missing = cp.any(hap < 0)
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
                            ploidy: int = 2):
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
        Ploidy level. Default 2 (diploid). Consecutive haplotypes in
        groups of `ploidy` are treated as belonging to one individual.

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Observed heterozygosity per variant.
    """
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

    n_individuals = n_hap // ploidy

    if ploidy == 2:
        # fast path for diploid: compare consecutive pairs
        h1 = hap[0::2]  # even-indexed haplotypes
        h2 = hap[1::2]  # odd-indexed haplotypes

        # a site is het if h1 != h2 (and neither is missing)
        valid = (h1 >= 0) & (h2 >= 0)
        het = (h1 != h2) & valid
        n_valid = cp.sum(valid, axis=0).astype(cp.float64)
        n_het = cp.sum(het, axis=0).astype(cp.float64)

        ho = cp.where(n_valid > 0, n_het / n_valid, cp.nan)
    else:
        # general ploidy: het if not all alleles identical within individual
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

    return ho.get()


def inbreeding_coefficient(haplotype_matrix: HaplotypeMatrix,
                           population: Optional[Union[str, list]] = None,
                           ploidy: int = 2):
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

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Inbreeding coefficient per variant. NaN where He = 0.
    """
    ho = heterozygosity_observed(haplotype_matrix, population, ploidy)
    he = heterozygosity_expected(haplotype_matrix, population)

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
           population: Optional[Union[str, list]] = None) -> float:
    """mu_SFS: fraction of SNPs at SFS edges (RAiSD).

    Counts singletons (DAC=1) and near-fixed variants (DAC=n-1),
    divided by total segregating sites. Elevated near selective sweeps.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional

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

    hap = cp.maximum(matrix.haplotypes, 0)
    n = hap.shape[0]
    dac = cp.sum(hap, axis=0)

    is_seg = (dac > 0) & (dac < n)
    n_seg = cp.sum(is_seg)

    if int(n_seg.get()) == 0:
        return 0.0

    is_edge = (dac == 1) | (dac == n - 1)
    n_edge = cp.sum(is_edge)

    return float((n_edge.astype(cp.float64) / n_seg.astype(cp.float64)).get())