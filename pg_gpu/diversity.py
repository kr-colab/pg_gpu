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


def _dac_and_n(haplotypes):
    """Shared helper: derived allele counts and valid sample counts per site.

    Uses adaptive chunking from _memutil for memory safety on large matrices.

    Parameters
    ----------
    haplotypes : cupy.ndarray, int8, shape (n_hap, n_var)

    Returns
    -------
    dac : cupy.ndarray, int64, shape (n_var,)
    n_valid : cupy.ndarray, int64, shape (n_var,)
    """
    from ._memutil import chunked_dac_and_n
    return chunked_dac_and_n(haplotypes)


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
    dac, n_valid_i = _dac_and_n(haplotypes)
    n_valid = n_valid_i.astype(cp.float64)
    derived = dac.astype(cp.float64)
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
       missing_data: str = 'include',
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
        ``'include'`` (default) uses per-site valid data.
        ``'exclude'`` filters to sites with no missing data.
        ``'pairwise'`` uses comparison-counting normalization (pixy-style):
        pi = sum(diffs) / sum(comps) across all sites, where sites with
        more observed data contribute proportionally more and invariant
        sites are included in the denominator when available.
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

    # Default: 'include' mode - calculate pi per site using only non-missing data
    derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)

    sites_with_data = n_valid_per_site >= 2

    if not cp.any(sites_with_data):
        pi_value = cp.float64(0.0)
    else:
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
            missing_data: str = 'include',
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
        ``'include'`` (default) uses per-site valid data.
        ``'exclude'`` filters to sites with no missing data.
        ``'pairwise'`` uses per-site 1/a(n_i) for segregating sites,
        normalized by n_total_sites when invariant info is available.
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
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2

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

    else:  # missing_data == 'include'
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            theta = cp.float64(0.0)
        else:

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

    # For exclude mode, compute theta the standard way
    if missing_data == 'exclude':
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
        ``'include'`` (default) uses per-site valid data with harmonic
        mean of per-site sample sizes for variance terms.
        ``'exclude'`` filters to sites with no missing data.
        ``'pairwise'`` uses pairwise pi and per-site theta_w with
        harmonic mean sample sizes for variance terms.

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
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2
        if not cp.any(sites_with_data):
            return float("nan")

        pi_value = pi(matrix, span_normalize=False, missing_data='include')

        valid_n = n_valid_per_site[sites_with_data]
        n_haplotypes = round(float(len(valid_n) / cp.sum(1.0 / valid_n).get()))

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

        # Harmonic mean of sample sizes (round to avoid off-by-one from
        # float truncation, e.g. 199.9999 -> int 199 instead of 200)
        n_haplotypes = round(float(len(n_valid_per_site[valid_site_mask]) /
                           cp.sum(1.0 / n_valid_per_site[valid_site_mask]).get()))

        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            S = 0
        else:
            valid_sites = cp.where(sites_with_data)[0]
            segregating_mask = (derived_counts[valid_sites] > 0) & (derived_counts[valid_sites] < n_valid_per_site[valid_sites])
            S = int(cp.sum(segregating_mask).get())
    else:
        # For 'exclude' mode
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
        'include' - Calculate AFS using available data per site.
        'exclude' - Only use sites with no missing data.
        'pairwise' - Same as 'include' for AFS computation.

    Returns
    -------
    ndarray
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
            return np.zeros(matrix.num_haplotypes + 1, dtype=np.int64)

        # Use only valid sites
        valid_haplotypes = matrix.haplotypes[:, valid_sites]
        n_haplotypes = matrix.num_haplotypes

        # Count derived alleles at each valid site
        freqs = cp.sum(valid_haplotypes, axis=0)

    else:  # missing_data == 'include'
        max_n = matrix.num_haplotypes
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)

        sites_with_data = n_valid_per_site > 0
        if not cp.any(sites_with_data):
            return np.zeros(max_n + 1, dtype=np.int64)

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
            return np.zeros(max_n + 1, dtype=np.int64)

        return afs.get()

    # For exclude mode, create standard histogram
    return cp.histogram(freqs, bins=cp.arange(n_haplotypes + 2))[0].get()


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
        'include' - Count sites as segregating based on non-missing data only.
        'exclude' - Only count sites with no missing data.
        'pairwise' - Same as 'include' for segregating site counting.

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

    else:  # missing_data == 'include'
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            return 0

        valid_sites = cp.where(sites_with_data)[0]
        segregating_mask = (derived_counts[valid_sites] > 0) & (derived_counts[valid_sites] < n_valid_per_site[valid_sites])
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
        'include' - Count singletons based on non-missing data only.
        'exclude' - Only count singletons at sites with no missing data.
        'pairwise' - Same as 'include' for singleton counting.

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

    else:  # missing_data == 'include'
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 1

        if not cp.any(sites_with_data):
            return 0

        valid_sites = cp.where(sites_with_data)[0]
        return int(cp.sum(derived_counts[valid_sites] == 1).get())

    # For exclude mode
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
            results['haplotype_diversity'] = haplotype_diversity(haplotype_matrix, population, missing_data)
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return results


def diversity_stats_fast(haplotype_matrix: HaplotypeMatrix,
                         population=None,
                         span_normalize: bool = True,
                         missing_data: str = 'include',
                         span_denominator: str = 'total',
                         projection_n: Optional[int] = None):
    """Compute all diversity and neutrality statistics from a single SFS.

    Uses the Achaz (2009) framework: one GPU pass for allele counting,
    then all estimators as weight vector dot products. 2-24x faster than
    calling individual functions when computing multiple statistics.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data.
    population : str or list, optional
        Population name or sample indices.
    span_normalize : bool
        Whether to normalize theta estimators by genomic span.
    missing_data : str
        'include' - per-site sample sizes, group by n_valid
        'exclude' - only sites with no missing data
    span_denominator : str
        'total', 'sites', 'callable', or 'accessible'.
    projection_n : int, optional
        Project SFS to this sample size before computing statistics.
        Useful for handling missing data or cross-population comparison.

    Returns
    -------
    dict
        All theta estimators and neutrality tests.
    """
    from .achaz import FrequencySpectrum

    fs = FrequencySpectrum(haplotype_matrix, population=population,
                           missing_data=missing_data)

    if projection_n is not None:
        fs = fs.project(projection_n)

    span = None
    if span_normalize:
        if population is not None:
            matrix = _get_population_matrix(haplotype_matrix, population)
        else:
            matrix = haplotype_matrix
        span = matrix.get_span(span_denominator)

    results = {}
    for name in ['pi', 'watterson', 'theta_h', 'theta_l',
                 'eta1', 'eta1_star', 'minus_eta1', 'minus_eta1_star']:
        results[name] = fs.theta(name, span_normalize=span_normalize, span=span)

    results['segregating_sites'] = fs.n_segregating
    results.update(fs.all_tests())

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
                       missing_data: str = 'include') -> float:
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

    haplotypes = matrix.haplotypes  # (n_hap, n_var)

    if missing_data == 'exclude':
        missing_per_var = cp.sum(haplotypes < 0, axis=0)
        complete = cp.where(missing_per_var == 0)[0]
        haplotypes = haplotypes[:, complete]

    n_haplotypes = haplotypes.shape[0]
    if n_haplotypes <= 1:
        return 0.0

    has_missing = bool(cp.any(haplotypes < 0).get())

    if has_missing:
        # Fallback: wildcard matching requires CPU pairwise comparison
        haplotypes_cpu = haplotypes.get() if hasattr(haplotypes, 'get') else haplotypes
        cluster_id = _cluster_haplotypes_with_missing(haplotypes_cpu)
        from collections import Counter
        counts = Counter(cluster_id)
        frequencies = np.array(list(counts.values())) / n_haplotypes
    else:
        # GPU fast path: dot-product hashing with two random weight vectors
        n_var = haplotypes.shape[1]
        rng = cp.random.RandomState(seed=42)
        w1 = rng.standard_normal(n_var, dtype=cp.float32)
        w2 = rng.standard_normal(n_var, dtype=cp.float32)
        h_f32 = haplotypes.astype(cp.float32)
        hash1 = h_f32 @ w1
        hash2 = h_f32 @ w2
        order = cp.lexsort(cp.stack([hash2, hash1]))
        s1 = hash1[order]
        s2 = hash2[order]
        diff = (cp.abs(s1[1:] - s1[:-1]) > 1e-3) | (cp.abs(s2[1:] - s2[:-1]) > 1e-3)
        boundaries = cp.concatenate([cp.array([True]), diff])
        boundary_idx = cp.where(boundaries)[0]
        counts_gpu = cp.diff(cp.concatenate([boundary_idx,
                                              cp.array([n_haplotypes])]))
        frequencies = (counts_gpu.astype(cp.float64) / n_haplotypes).get()

    diversity = (1.0 - np.sum(frequencies ** 2)) * n_haplotypes / (n_haplotypes - 1)
    return float(diversity)


def _cluster_haplotypes_with_missing(haps):
    """Cluster haplotypes treating -1 as compatible with any allele.

    Two haplotypes are in the same cluster if they match at all positions
    where both are non-missing. Uses greedy assignment: each haplotype
    joins the first compatible cluster.

    Parameters
    ----------
    haps : ndarray, shape (n_haplotypes, n_variants)

    Returns
    -------
    labels : list of int, length n_haplotypes
    """
    n = haps.shape[0]
    has_any_missing = np.any(haps < 0)

    if not has_any_missing:
        # fast path: no missing data, use string hashing
        hap_strings = [''.join(map(str, h)) for h in haps]
        label_map = {}
        labels = []
        next_id = 0
        for s in hap_strings:
            if s not in label_map:
                label_map[s] = next_id
                next_id += 1
            labels.append(label_map[s])
        return labels

    # slow path: pairwise comparison with wildcard matching
    # representative haplotype per cluster (index into haps)
    cluster_reps = [0]
    labels = [0]

    for i in range(1, n):
        matched = False
        for c_idx, rep in enumerate(cluster_reps):
            # check if haps[i] matches haps[rep] at jointly non-missing sites
            both_valid = (haps[i] >= 0) & (haps[rep] >= 0)
            if np.all(haps[i][both_valid] == haps[rep][both_valid]):
                labels.append(c_idx)
                matched = True
                break
        if not matched:
            cluster_reps.append(i)
            labels.append(len(cluster_reps) - 1)

    return labels


_get_population_matrix = get_population_matrix


def theta_h(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True,
            missing_data: str = 'include') -> float:
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

    # 'include' / 'pairwise' mode (default fallback)
    haplotypes = matrix.haplotypes
    dac_i, n_valid_i = _dac_and_n(haplotypes)
    n_valid = n_valid_i.astype(cp.float64)
    dac = dac_i.astype(cp.float64)

    # Only segregating sites: 0 < dac < n_valid
    usable = (n_valid > 1) & (dac > 0) & (dac < n_valid)
    th = cp.sum(2.0 * dac[usable] ** 2 / (n_valid[usable] * (n_valid[usable] - 1)))

    if span_normalize:
        span = matrix.get_span()
        if span > 0:
            th = th / span

    return float(th.get())


def theta_l(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize: bool = True,
            missing_data: str = 'include') -> float:
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

    # 'include' / 'pairwise' mode (default fallback)
    haplotypes = matrix.haplotypes
    dac_i, n_valid_i = _dac_and_n(haplotypes)
    n_valid = n_valid_i.astype(cp.float64)
    dac = dac_i.astype(cp.float64)

    # Only segregating sites: 0 < dac < n_valid
    usable = (n_valid > 1) & (dac > 0) & (dac < n_valid)
    tl = cp.sum(dac[usable] / (n_valid[usable] - 1))

    if span_normalize:
        span = matrix.get_span()
        if span > 0:
            tl = tl / span

    return float(tl.get())


def _effective_n_and_S(matrix, missing_data):
    """Compute effective sample size and segregating site count.

    For 'include'/'pairwise': uses harmonic mean of per-site n_valid
    as effective n, and counts segregating sites among valid data.
    For 'exclude': uses the fixed sample size.

    Returns (n_eff, S, matrix) where matrix may be subset-filtered.
    """
    if missing_data == 'exclude':
        missing_per_variant = matrix.count_missing(axis=0)
        valid_idx = cp.where(missing_per_variant == 0)[0]
        if len(valid_idx) == 0:
            return 0.0, 0.0, matrix
        matrix = matrix.get_subset(valid_idx)

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    n_valid = n_valid_i.astype(cp.float64)
    dac = dac_i.astype(cp.float64)
    usable = n_valid >= 2
    if not cp.any(usable):
        return 0.0, 0.0, matrix

    seg_mask = usable & (dac > 0) & (dac < n_valid)
    S = float(cp.sum(seg_mask).get())

    # harmonic mean of per-site sample sizes (across usable sites)
    # round to avoid off-by-one from float truncation in downstream int()
    n_usable = n_valid[usable]
    n_eff = round(float(len(n_usable)) / float(cp.sum(1.0 / n_usable).get()))

    return n_eff, S, matrix


def normalized_fay_wus_h(haplotype_matrix: HaplotypeMatrix,
                         population: Optional[Union[str, list]] = None,
                         missing_data: str = 'include') -> float:
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
        ``'include'`` (default) or ``'pairwise'`` uses per-site sample
        sizes with harmonic mean n for variance terms.
        ``'exclude'`` filters to sites with no missing data.

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
           missing_data: str = 'include') -> float:
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
        ``'include'`` (default) or ``'pairwise'`` uses per-site sample
        sizes with harmonic mean n for variance terms.
        ``'exclude'`` filters to sites with no missing data.

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
            missing_data: str = 'include') -> float:
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
            missing_data: str = 'include') -> float:
    """Maximum derived allele frequency across all variants.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - per-site n_valid for frequency
        'exclude' - only sites with no missing data

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

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)

    if missing_data == 'exclude':
        complete = n_valid_i == matrix.haplotypes.shape[0]
        freqs = cp.where(complete, dac / n_valid, -1.0)
    else:
        usable = n_valid > 0
        freqs = cp.where(usable, dac / n_valid, 0.0)

    return float(cp.max(freqs).get())


def haplotype_count(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None,
                    missing_data: str = 'include') -> int:
    """Count distinct haplotypes.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - exclude haplotypes with any missing
        'exclude' - filter to sites with no missing data

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

    labels = _cluster_haplotypes_with_missing(hap_cpu)
    return len(set(labels))


def daf_histogram(matrix, n_bins: int = 20,
                  population: Optional[Union[str, list]] = None,
                  missing_data: str = 'include'):
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

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)

    if missing_data == 'exclude':
        complete = n_valid_i == matrix.haplotypes.shape[0]
        dafs = (dac / n_valid)[complete]
    else:
        usable = n_valid > 0
        dafs = cp.where(usable, dac / n_valid, 0.0)

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

    geno = np.asarray(geno, dtype=np.int8)
    n_ind = geno.shape[0]

    # treat missing (-1) as wildcard for diplotype identity
    labels = _cluster_haplotypes_with_missing(geno)
    from collections import Counter
    counts = Counter(labels)
    freqs = np.array(sorted(counts.values(), reverse=True)) / n_ind

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

    valid_mask = geno >= 0
    geno_clean = cp.where(valid_mask, geno, 0).astype(cp.float64)
    n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
    usable = n_valid > 0
    dafs = cp.where(usable, cp.sum(geno_clean, axis=0) / (2.0 * n_valid), 0.0)

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
                            missing_data: str = 'include'):
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

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)
    n = matrix.haplotypes.shape[0]

    if missing_data == 'include':
        p = cp.where(n_valid > 0, dac / n_valid, 0.0)
        he = 2.0 * p * (1.0 - p)
        he = cp.where(n_valid >= 2, he, cp.nan)
    else:
        p = dac / n
        he = 2.0 * p * (1.0 - p)

        if missing_data == 'exclude':
            incomplete = n_valid_i < n
            he[incomplete] = cp.nan

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
           missing_data: str = 'include') -> float:
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

    dac, n_valid = _dac_and_n(matrix.haplotypes)

    if missing_data == 'exclude':
        n = matrix.haplotypes.shape[0]
        complete = n_valid == n
        is_seg = complete & (dac > 0) & (dac < n)
        is_edge = complete & ((dac == 1) | (dac == n - 1))
    else:
        usable = n_valid >= 2
        is_seg = usable & (dac > 0) & (dac < n_valid)
        is_edge = usable & ((dac == 1) | (dac == n_valid - 1))

    n_seg = cp.sum(is_seg)
    if int(n_seg.get()) == 0:
        return 0.0

    n_edge = cp.sum(is_edge)
    return float((n_edge.astype(cp.float64) / n_seg.astype(cp.float64)).get())
