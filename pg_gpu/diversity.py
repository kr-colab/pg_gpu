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
       span_normalize: bool = True) -> float:
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
    
    # Calculate allele frequency spectrum
    afs = allele_frequency_spectrum(matrix)
    n_haplotypes = matrix.num_haplotypes
    
    # Compute the weight factor for each allele frequency
    # Note: We only consider frequencies from 1 to n-1 (excluding fixed sites)
    i = cp.arange(1, n_haplotypes, dtype=cp.float64)
    weight = (2 * i * (n_haplotypes - i)) / (n_haplotypes * (n_haplotypes - 1))
    
    # Compute π as a weighted sum over the allele frequency spectrum
    # afs[0] = sites fixed for ancestral, afs[n] = sites fixed for derived
    # We only sum over segregating sites (indices 1 to n-1)
    pi_value = cp.sum((weight * afs[1:n_haplotypes]).astype(cp.float64))
    
    if span_normalize and matrix.chrom_start is not None and matrix.chrom_end is not None:
        span = cp.float64(matrix.chrom_end - matrix.chrom_start)
        return float(pi_value / span)
    return float(pi_value.get() if hasattr(pi_value, 'get') else pi_value)


def theta_w(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True) -> float:
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
    
    n_haplotypes = matrix.num_haplotypes
    
    # Count segregating sites
    seg_sites = segregating_sites(matrix)
    
    # Compute the harmonic number a_n
    a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes, dtype=cp.float64))
    theta = seg_sites / a1
    
    if span_normalize and matrix.chrom_start is not None and matrix.chrom_end is not None:
        span = cp.float64(matrix.chrom_end - matrix.chrom_start)
        return float(theta / span)
    return float(theta.get() if hasattr(theta, 'get') else theta)


def tajimas_d(haplotype_matrix: HaplotypeMatrix,
              population: Optional[Union[str, list]] = None) -> float:
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
    
    # Get pi and theta
    pi_value = pi(matrix, span_normalize=False)
    
    n_haplotypes = matrix.num_haplotypes
    S = segregating_sites(matrix)
    
    # If no segregating sites, return NaN
    if S == 0:
        return float("nan")
    
    # Calculate theta directly (to avoid span normalization)
    a1 = cp.sum(1.0 / cp.arange(1, n_haplotypes, dtype=cp.float64))
    theta = S / a1
    
    # Variance term for Tajima's D
    a2 = cp.sum(1.0 / (cp.arange(1, n_haplotypes, dtype=cp.float64) ** 2))
    b1 = (n_haplotypes + 1) / (3 * (n_haplotypes - 1))
    b2 = 2 * (n_haplotypes**2 + n_haplotypes + 3) / (9 * n_haplotypes * (n_haplotypes - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n_haplotypes + 2) / (a1 * n_haplotypes)) + (a2 / (a1 ** 2))
    e1 = c1 / a1
    e2 = c2 / ((a1 ** 2) + a2)
    V = cp.sqrt((e1 * S) + (e2 * S * (S - 1)))
    
    # Calculate D
    if V != 0:
        D = (pi_value - float(theta.get() if hasattr(theta, 'get') else theta)) / float(V.get() if hasattr(V, 'get') else V)
        return D
    else:
        return float("nan")


def allele_frequency_spectrum(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None) -> cp.ndarray:
    """
    Calculate the allele frequency spectrum (AFS).
    
    The AFS is a histogram of allele frequencies across all sites.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
        
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
    
    n_haplotypes = matrix.num_haplotypes
    
    # Count derived alleles at each site
    freqs = cp.sum(cp.nan_to_num(matrix.haplotypes, nan=0).astype(cp.int32), axis=0)
    
    # Create histogram
    # Note: bins should go from 0 to n_haplotypes inclusive (n_haplotypes + 1 edges)
    # This gives us n_haplotypes + 1 bins
    return cp.histogram(freqs, bins=cp.arange(n_haplotypes + 2))[0]


def segregating_sites(haplotype_matrix: HaplotypeMatrix,
                     population: Optional[Union[str, list]] = None) -> int:
    """
    Count the number of segregating sites.
    
    A site is segregating if it has more than one allele present.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
        
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
    
    # Count alleles at each site
    allele_counts = cp.sum(matrix.haplotypes, axis=0)
    n_haplotypes = matrix.num_haplotypes
    
    # Site is segregating if not all 0s or all 1s
    segregating = (allele_counts > 0) & (allele_counts < n_haplotypes)
    
    return int(cp.sum(segregating).get())


def singleton_count(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None) -> int:
    """
    Count the number of singleton variants.
    
    A singleton is a variant present in exactly one haplotype.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
        
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
    
    # Count alleles at each site
    allele_counts = cp.sum(matrix.haplotypes, axis=0)
    
    # Count singletons
    return int(cp.sum(allele_counts == 1).get())


def diversity_stats(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None,
                   statistics: list = ['pi', 'theta_w', 'tajimas_d'],
                   span_normalize: bool = True) -> Dict[str, float]:
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
        
    Returns
    -------
    dict
        Dictionary mapping statistic names to values
    """
    results = {}
    
    for stat in statistics:
        if stat == 'pi':
            results['pi'] = pi(haplotype_matrix, population, span_normalize)
        elif stat == 'theta_w':
            results['theta_w'] = theta_w(haplotype_matrix, population, span_normalize)
        elif stat == 'tajimas_d':
            results['tajimas_d'] = tajimas_d(haplotype_matrix, population)
        elif stat == 'segregating_sites':
            results['segregating_sites'] = segregating_sites(haplotype_matrix, population)
        elif stat == 'singletons':
            results['singletons'] = singleton_count(haplotype_matrix, population)
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
              population: Optional[Union[str, list]] = None) -> float:
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
    
    n = matrix.num_haplotypes
    
    # Get allele frequencies
    allele_counts = cp.sum(matrix.haplotypes, axis=0)
    
    # Calculate theta_H (uses squared frequencies)
    theta_h = 0.0
    for i in range(1, n):
        count_i = cp.sum(allele_counts == i)
        if count_i > 0:
            theta_h += 2.0 * i * i * float(count_i.get()) / (n * (n - 1))
    
    # Get pi
    pi_value = pi(matrix, span_normalize=False)
    
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
                    population: Optional[Union[str, list]] = None) -> Dict[str, float]:
    """
    Compute common neutrality test statistics.
    
    Returns Tajima's D, Fay and Wu's H, and related values.
    
    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices
        
    Returns
    -------
    dict
        Dictionary with neutrality test results
    """
    return {
        'tajimas_d': tajimas_d(haplotype_matrix, population),
        'fay_wus_h': fay_wus_h(haplotype_matrix, population),
        'pi': pi(haplotype_matrix, population, span_normalize=False),
        'theta_w': theta_w(haplotype_matrix, population, span_normalize=False),
        'segregating_sites': segregating_sites(haplotype_matrix, population)
    }