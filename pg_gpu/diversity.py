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


def pi(haplotype_matrix: HaplotypeMatrix,
       population: Optional[Union[str, list]] = None,
       span_normalize: bool = True,
       missing_data: str = 'include',
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
        afs = allele_frequency_spectrum(matrix)
        n_haplotypes = matrix.num_haplotypes
        
        i = cp.arange(1, n_haplotypes, dtype=cp.float64)
        weight = (2 * i * (n_haplotypes - i)) / (n_haplotypes * (n_haplotypes - 1))
        pi_value = cp.sum((weight * afs[1:n_haplotypes]).astype(cp.float64))
        
    else:  # missing_data == 'include'
        # Calculate pi per site using only non-missing data at each site
        pi_value = cp.float64(0.0)
        
        for variant_idx in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, variant_idx]
            valid_mask = variant_data >= 0
            n_valid = int(cp.sum(valid_mask).get())
            
            if n_valid > 1:  # Need at least 2 samples to calculate diversity
                valid_data = variant_data[valid_mask]
                
                # Count alleles for this site
                allele_counts = cp.zeros(2, dtype=cp.int32)  # Assuming biallelic
                unique_alleles = cp.unique(valid_data)
                
                for i, allele in enumerate(unique_alleles):
                    if i < 2:  # Safety check for biallelic assumption
                        allele_counts[allele] = cp.sum(valid_data == allele)
                
                # Calculate pi for this site using the standard formula
                site_pi = cp.float64(0.0)
                total = cp.float64(n_valid)
                
                for i in range(len(allele_counts)):
                    freq_i = allele_counts[i] / total
                    for j in range(i + 1, len(allele_counts)):
                        freq_j = allele_counts[j] / total
                        # Nei's correction for finite sample size
                        site_pi += 2 * freq_i * freq_j * total / (total - 1)
                
                pi_value += site_pi
    
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
            missing_data: str = 'include',
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
        seg_sites = segregating_sites(matrix)
        
    elif missing_data == 'ignore':
        # Original behavior
        n_haplotypes = matrix.num_haplotypes
        seg_sites = segregating_sites(matrix)
        
    else:  # missing_data == 'include'
        # Calculate theta using site-specific sample sizes
        theta_sum = cp.float64(0.0)
        
        for variant_idx in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, variant_idx]
            valid_mask = variant_data >= 0
            n_valid = int(cp.sum(valid_mask).get())
            
            if n_valid > 1:  # Need at least 2 samples
                valid_data = variant_data[valid_mask]
                
                # Check if site is segregating among valid samples
                unique_alleles = cp.unique(valid_data)
                if len(unique_alleles) > 1:
                    # This site is segregating - add its contribution to theta
                    # Use harmonic number for the actual sample size at this site
                    a1_site = cp.sum(1.0 / cp.arange(1, n_valid, dtype=cp.float64))
                    theta_sum += 1.0 / a1_site
        
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
              missing_data: str = 'include') -> float:
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
        
        # Count segregating sites considering missing data
        S = 0
        for i in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, i]
            valid_mask = variant_data >= 0
            if cp.sum(valid_mask) >= 2:
                valid_data = variant_data[valid_mask]
                if len(cp.unique(valid_data)) > 1:
                    S += 1
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
        # Build AFS considering variable sample sizes per site
        # Maximum possible sample size
        max_n = matrix.num_haplotypes
        
        # Initialize AFS array with size for maximum possible allele count
        afs = cp.zeros(max_n + 1, dtype=cp.int64)
        
        for i in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, i]
            valid_mask = variant_data >= 0
            n_valid = int(cp.sum(valid_mask).get())
            
            if n_valid > 0:
                # Count derived alleles among valid samples
                derived_count = int(cp.sum(variant_data[valid_mask]).get())
                # Add to AFS at the appropriate bin
                afs[derived_count] += 1
        
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
        # Count each site as segregating based on non-missing data only
        seg_count = 0
        
        for i in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, i]
            valid_mask = variant_data >= 0
            n_valid = cp.sum(valid_mask)
            
            if n_valid >= 2:  # Need at least 2 samples
                valid_data = variant_data[valid_mask]
                unique_alleles = cp.unique(valid_data)
                if len(unique_alleles) > 1:
                    seg_count += 1
        
        return seg_count
    
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
        # Count singletons based on non-missing data at each site
        singleton_count = 0
        
        for i in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, i]
            valid_mask = variant_data >= 0
            
            if cp.any(valid_mask):
                # Count derived alleles among valid samples
                derived_count = cp.sum(variant_data[valid_mask])
                if derived_count == 1:
                    singleton_count += 1
        
        return singleton_count
    
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
        
        # Calculate theta_H (uses squared frequencies)
        theta_h = 0.0
        for i in range(1, n):
            count_i = cp.sum(allele_counts == i)
            if count_i > 0:
                theta_h += 2.0 * i * i * float(count_i.get()) / (n * (n - 1))
    
    else:  # missing_data == 'include'
        # Calculate theta_H considering variable sample sizes per site
        theta_h = 0.0
        
        for variant_idx in range(matrix.num_variants):
            variant_data = matrix.haplotypes[:, variant_idx]
            valid_mask = variant_data >= 0
            n_valid = int(cp.sum(valid_mask).get())
            
            if n_valid > 1:  # Need at least 2 samples
                # Count derived alleles at this site
                derived_count = int(cp.sum(variant_data[valid_mask]).get())
                
                # Add contribution to theta_H
                # theta_H = sum over sites of 2 * i^2 / (n * (n-1))
                # where i is the derived allele count
                if derived_count > 0:
                    theta_h += 2.0 * derived_count * derived_count / (n_valid * (n_valid - 1))
    
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


def _get_population_matrix(haplotype_matrix: HaplotypeMatrix,
                          population: Union[str, list]) -> HaplotypeMatrix:
    """
    Extract population-specific haplotype matrix.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The full haplotype data
    population : str or list
        Population name or list of sample indices
        
    Returns
    -------
    HaplotypeMatrix
        Subset containing only the specified population
    """
    if isinstance(population, str):
        if haplotype_matrix.sample_sets is None:
            raise ValueError("No sample_sets defined in haplotype matrix")
        if population not in haplotype_matrix.sample_sets:
            raise ValueError(f"Population {population} not found in sample_sets")
        pop_indices = haplotype_matrix.sample_sets[population]
    else:
        pop_indices = list(population)
    
    # Extract population haplotypes
    pop_haplotypes = haplotype_matrix.haplotypes[pop_indices, :]
    
    # Create new matrix for this population
    return HaplotypeMatrix(
        pop_haplotypes,
        haplotype_matrix.positions,
        haplotype_matrix.chrom_start,
        haplotype_matrix.chrom_end,
        sample_sets={'all': list(range(len(pop_indices)))}
    )


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