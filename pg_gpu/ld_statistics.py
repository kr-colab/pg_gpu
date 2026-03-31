"""
GPU-accelerated linkage disequilibrium statistics.

This module provides an API for computing LD statistics
on GPUs with automatic missing data handling.
"""

import cupy as cp
from typing import Optional, Union, Tuple, List, Dict


def dd(counts: cp.ndarray, 
       populations: Optional[Union[Tuple[int, int], int]] = None,
       n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² statistic for any population configuration.
    
    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array:
        - Single population: shape (N, 4)
        - Two populations: shape (N, 8)
        - Multi-population: shape (N, 4*P)
    populations : tuple of int, optional
        Population indices. None for single population,
        (i, j) for between populations i and j
    n_valid : cp.ndarray, optional
        Valid sample counts per population. Shape depends on configuration:
        - Single pop: shape (N,)
        - Two pops: shape (N, 2) or tuple of (N,) arrays
        
    Returns
    -------
    cp.ndarray
        D² values for each locus
    """
    # Handle different input formats
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _dd_single(counts, n_valid)
        else:
            # Default to first population if counts has multiple
            return _dd_single(counts[:, :4], n_valid[:, 0] if n_valid is not None and n_valid.ndim == 2 else n_valid)
    
    # Two population case
    pop1, pop2 = populations
    if pop1 == pop2:
        # Within population
        start_idx = pop1 * 4
        pop_counts = counts[:, start_idx:start_idx + 4]
        pop_n_valid = None
        if n_valid is not None:
            if n_valid.ndim == 2:
                pop_n_valid = n_valid[:, pop1]
            elif isinstance(n_valid, tuple):
                pop_n_valid = n_valid[pop1]
            else:
                pop_n_valid = n_valid
        return _dd_single(pop_counts, pop_n_valid)
    else:
        # Between populations
        return _dd_between(counts, pop1, pop2, n_valid)


def dz(counts: cp.ndarray,
       populations: Optional[Tuple[int, int, int]] = None,
       n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute Dz statistic for any population configuration.
    
    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    populations : tuple of int, optional
        Three population indices (i, j, k) for Dz(i,j,k).
        None defaults to single population (0, 0, 0)
    n_valid : cp.ndarray, optional
        Valid sample counts per population
        
    Returns
    -------
    cp.ndarray
        Dz values for each locus
    """
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _dz_single(counts, n_valid)
        else:
            # Default to first population
            populations = (0, 0, 0)
    
    return _dz_multi(counts, populations, n_valid)


def pi2(counts: cp.ndarray,
        populations: Optional[Tuple[int, int, int, int]] = None,
        n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute π₂ statistic for any population configuration.
    
    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    populations : tuple of int, optional
        Four population indices (i, j, k, l) for π₂(i,j,k,l).
        None defaults to single population (0, 0, 0, 0)
    n_valid : cp.ndarray, optional
        Valid sample counts per population
        
    Returns
    -------
    cp.ndarray
        π₂ values for each locus
    """
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _pi2_single(counts, n_valid)
        else:
            # Default to first population
            populations = (0, 0, 0, 0)
    
    return _pi2_multi(counts, populations, n_valid)


def dd_within(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² within a single population.
    
    Convenience function equivalent to dd(counts, populations=None)
    """
    return _dd_single(counts, n_valid)


def dd_between(counts: cp.ndarray, 
               pop1_idx: int, 
               pop2_idx: int,
               n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² between two populations.
    
    Convenience function equivalent to dd(counts, populations=(pop1_idx, pop2_idx))
    """
    return _dd_between(counts, pop1_idx, pop2_idx, n_valid)


def r(counts: cp.ndarray,
      n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute Pearson correlation coefficient r between variant pairs
    from haplotype counts.

    Parameters
    ----------
    counts : cp.ndarray, shape (N, 4)
        Haplotype counts [n11, n10, n01, n00] for each variant pair.
    n_valid : cp.ndarray, optional
        Valid sample counts per pair. Shape (N,).

    Returns
    -------
    cp.ndarray, float64, shape (N,)
        Pearson r values. NaN where computation is undefined
        (monomorphic at either locus).
    """
    c11, c10, c01, c00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid.astype(cp.float64) if n_valid is not None else cp.sum(counts, axis=1).astype(cp.float64)

    p_A = (c11 + c10) / n
    p_B = (c11 + c01) / n
    D = (c11 * c00 - c10 * c01).astype(cp.float64) / (n * n)
    denom = p_A * (1 - p_A) * p_B * (1 - p_B)

    valid_mask = denom > 0
    result = cp.full(n.shape[0], cp.nan, dtype=cp.float64)
    result[valid_mask] = D[valid_mask] / cp.sqrt(denom[valid_mask])

    return result


def r_squared(counts: cp.ndarray,
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute r-squared (squared Pearson correlation) between variant pairs
    from haplotype counts.

    Parameters
    ----------
    counts : cp.ndarray, shape (N, 4)
        Haplotype counts [n11, n10, n01, n00] for each variant pair.
    n_valid : cp.ndarray, optional
        Valid sample counts per pair. Shape (N,).

    Returns
    -------
    cp.ndarray, float64, shape (N,)
        r-squared values. NaN where computation is undefined.
    """
    return r(counts, n_valid) ** 2


def compute_ld_statistics(counts: cp.ndarray,
                         statistics: List[str] = ['dd', 'dz', 'pi2'],
                         populations: Optional[Dict[str, Union[Tuple, None]]] = None,
                         n_valid: Optional[cp.ndarray] = None) -> Dict[str, cp.ndarray]:
    """
    Compute multiple LD statistics in one pass.
    
    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    statistics : list of str
        Statistics to compute ('dd', 'dz', 'pi2')
    populations : dict, optional
        Population configurations for each statistic.
        E.g., {'dd': (0, 1), 'dz': (0, 0, 1), 'pi2': (0, 0, 1, 1)}
    n_valid : cp.ndarray, optional
        Valid sample counts per population
        
    Returns
    -------
    dict
        Dictionary mapping statistic names to computed values
    """
    if populations is None:
        populations = {}
    
    results = {}
    
    for stat in statistics:
        if stat == 'dd':
            pop_config = populations.get('dd', None)
            results['dd'] = dd(counts, pop_config, n_valid)
        elif stat == 'dz':
            pop_config = populations.get('dz', None)
            results['dz'] = dz(counts, pop_config, n_valid)
        elif stat == 'pi2':
            pop_config = populations.get('pi2', None)
            results['pi2'] = pi2(counts, pop_config, n_valid)
        elif stat == 'r':
            results['r'] = r(counts, n_valid)
        elif stat == 'r_squared':
            results['r_squared'] = r_squared(counts, n_valid)
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    return results


# Internal implementation functions

def _dd_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute D² for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)
    
    numer = c1 * (c1 - 1) * c4 * (c4 - 1) + c2 * (c2 - 1) * c3 * (c3 - 1) - 2 * c1 * c2 * c3 * c4
    denom = n * (n - 1) * (n - 2) * (n - 3)
    
    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def _dd_between(counts: cp.ndarray, 
                pop1_idx: int, 
                pop2_idx: int,
                n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute D² between two populations."""
    # Extract counts for each population
    start1 = pop1_idx * 4
    start2 = pop2_idx * 4
    
    c11, c12, c13, c14 = counts[:, start1], counts[:, start1+1], counts[:, start1+2], counts[:, start1+3]
    c21, c22, c23, c24 = counts[:, start2], counts[:, start2+1], counts[:, start2+2], counts[:, start2+3]
    
    # Get valid sample sizes
    if n_valid is not None:
        if isinstance(n_valid, tuple):
            n1 = n_valid[0] if n_valid[0] is not None else cp.sum(counts[:, start1:start1+4], axis=1)
            n2 = n_valid[1] if n_valid[1] is not None else cp.sum(counts[:, start2:start2+4], axis=1)
        elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
            n1 = n_valid[:, pop1_idx]
            n2 = n_valid[:, pop2_idx]
        else:
            # Assume n_valid is for between-population pairs
            n1 = n_valid
            n2 = n_valid
    else:
        n1 = cp.sum(counts[:, start1:start1+4], axis=1)
        n2 = cp.sum(counts[:, start2:start2+4], axis=1)
    
    D1 = c12 * c13 - c11 * c14
    D2 = c22 * c23 - c21 * c24
    
    numer = D1 * D2
    denom = n1 * (n1 - 1) * n2 * (n2 - 1)
    
    valid_mask = (n1 >= 2) & (n2 >= 2)
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def _dz_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute Dz for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)
    
    diff = c1 * c4 - c2 * c3
    sum_34_12 = (c3 + c4) - (c1 + c2)
    sum_24_13 = (c2 + c4) - (c1 + c3)
    sum_23_14 = (c2 + c3) - (c1 + c4)
    
    numer = diff * sum_34_12 * sum_24_13 + diff * sum_23_14 + 2 * (c2 * c3 + c1 * c4)
    denom = n * (n - 1) * (n - 2) * (n - 3)
    
    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def _dz_multi(counts: cp.ndarray, 
              populations: Tuple[int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute Dz for multiple populations."""
    pop1, pop2, pop3 = populations
    
    # Helper to extract counts and valid sizes
    def get_pop_data(pop_idx):
        start = pop_idx * 4
        pop_counts = counts[:, start:start+4]
        if n_valid is not None:
            if isinstance(n_valid, tuple):
                if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                    pop_n = n_valid[pop_idx]
                else:
                    pop_n = cp.sum(pop_counts, axis=1)
            elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
                pop_n = n_valid[:, pop_idx]
            else:
                pop_n = n_valid
        else:
            pop_n = cp.sum(pop_counts, axis=1)
        return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n
    
    if pop1 == pop2 == pop3:
        # Single population
        if n_valid is not None and isinstance(n_valid, tuple):
            # Handle tuple case
            pop_n_valid = n_valid[pop1] if pop1 < len(n_valid) and n_valid[pop1] is not None else None
        elif n_valid is not None and hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
            pop_n_valid = n_valid[:, pop1]
        else:
            pop_n_valid = n_valid
        return _dz_single(counts[:, pop1*4:(pop1+1)*4], pop_n_valid)
    
    elif pop1 == pop2:  # Dz(i,i,j)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop3)
        
        numer = (
            (-c11 - c12 + c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 + c22 - c23 + c24)
        )
        denom = n2 * n1 * (n1 - 1) * (n1 - 2)
        
        valid_mask = (n1 >= 3) & (n2 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    elif pop1 == pop3:  # Dz(i,j,i)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop2)
        
        numer = (
            (-c11 + c12 - c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 - c22 + c23 + c24)
        )
        denom = n2 * n1 * (n1 - 1) * (n1 - 2)
        
        valid_mask = (n1 >= 3) & (n2 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    elif pop2 == pop3:  # Dz(i,j,j)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop2)
        
        numer = (-(c12 * c13) + c11 * c14) * (-c21 + c22 + c23 - c24) + (
            -(c12 * c13) + c11 * c14
        ) * (-c21 + c22 - c23 + c24) * (-c21 - c22 + c23 + c24)
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    else:
        # All different populations - return zeros
        n1 = n_valid[:, 0] if n_valid is not None and n_valid.ndim == 2 else cp.sum(counts[:, :4], axis=1)
        result = cp.zeros_like(n1, dtype=cp.float64)
    
    return result


def _pi2_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute π₂ for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)
    
    s12 = c1 + c2
    s13 = c1 + c3
    s24 = c2 + c4
    s34 = c3 + c4
    
    term_a = s12 * s13 * s24 * s34
    term_b = c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
    term_c = c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4)
    
    numer = term_a - term_b - term_c
    denom = n * (n - 1) * (n - 2) * (n - 3)
    
    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def _pi2_multi(counts: cp.ndarray,
               populations: Tuple[int, int, int, int],
               n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute π₂ for multiple populations."""
    i, j, k, l = populations
    
    # Helper to extract counts and valid sizes
    def get_pop_data(pop_idx):
        start = pop_idx * 4
        pop_counts = counts[:, start:start+4]
        if n_valid is not None:
            if isinstance(n_valid, tuple):
                if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                    pop_n = n_valid[pop_idx]
                else:
                    pop_n = cp.sum(pop_counts, axis=1)
            elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
                pop_n = n_valid[:, pop_idx]
            else:
                pop_n = n_valid
        else:
            pop_n = cp.sum(pop_counts, axis=1)
        return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n
    
    if i == j == k == l:
        # Single population
        return _pi2_single(counts[:, i*4:(i+1)*4],
                          n_valid[:, i] if n_valid is not None and n_valid.ndim == 2 else n_valid)
    
    elif i == j and k == l and i != k:
        # pi2(i,i,j,j) - average of two permutations
        c11, c12, c13, c14, n1 = get_pop_data(i)
        c21, c22, c23, c24, n2 = get_pop_data(k)
        
        numer1 = (c11 + c12) * (c13 + c14) * (c21 + c23) * (c22 + c24)
        numer2 = (c21 + c22) * (c23 + c24) * (c11 + c13) * (c12 + c14)
        
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = 0.5 * (numer1[valid_mask] + numer2[valid_mask]) / denom[valid_mask]
        
    elif j == k == l and i != j:
        # pi2(i,j,j,j) type
        result = _pi2_iiij(counts, populations, n_valid)
        
    elif i == j and k != l:
        # pi2(i,i,k,l) type - need to check specific cases
        if i == k or i == l:
            # Cases like (0,0,0,1) or (0,0,1,0)
            result = _pi2_iikl(counts, populations, n_valid)
        else:
            # Other cases - for now return zeros
            n1 = get_pop_data(0)[4]
            result = cp.zeros_like(n1, dtype=cp.float64)
            
    elif i != j and k == l:
        # pi2(i,j,k,k) type
        result = _pi2_ijkk(counts, populations, n_valid)
        
    elif (i == k and j == l) or (i == l and j == k):
        # pi2(i,j,i,j) or pi2(i,j,j,i) type
        c11, c12, c13, c14, n1 = get_pop_data(i)
        c21, c22, c23, c24, n2 = get_pop_data(j)
        
        numer = (
            ((c12 + c14) * (c13 + c14) * (c21 + c22) * (c21 + c23)) / 4.0
            + ((c11 + c13) * (c13 + c14) * (c21 + c22) * (c22 + c24)) / 4.0
            + ((c11 + c12) * (c12 + c14) * (c21 + c23) * (c23 + c24)) / 4.0
            + ((c11 + c12) * (c11 + c13) * (c22 + c24) * (c23 + c24)) / 4.0
            + (
                -(c12 * c13 * c21)
                + c14 * c21
                - c12 * c14 * c21
                - c13 * c14 * c21
                - c14 ** 2 * c21
                - c14 * c21 ** 2
                + c13 * c22
                - c11 * c13 * c22
                - c13 ** 2 * c22
                - c11 * c14 * c22
                - c13 * c14 * c22
                - c13 * c21 * c22
                - c14 * c21 * c22
                - c13 * c22 ** 2
                + c12 * c23
                - c11 * c12 * c23
                - c12 ** 2 * c23
                - c11 * c14 * c23
                - c12 * c14 * c23
                - c12 * c21 * c23
                - c14 * c21 * c23
                - c11 * c22 * c23
                - c14 * c22 * c23
                - c12 * c23 ** 2
                + c11 * c24
                - c11 ** 2 * c24
                - c11 * c12 * c24
                - c11 * c13 * c24
                - c12 * c13 * c24
                - c12 * c21 * c24
                - c13 * c21 * c24
                - c11 * c22 * c24
                - c13 * c22 * c24
                - c11 * c23 * c24
                - c12 * c23 * c24
                - c11 * c24 ** 2
            ) / 4.0
        )
        
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    else:
        # Other cases - for now return zeros
        # This would need full implementation of all permutation patterns
        n1 = get_pop_data(0)[4]
        result = cp.zeros_like(n1, dtype=cp.float64)
    
    return result


def _pi2_iiij(counts: cp.ndarray, 
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,j,j,j) configurations."""
    i, j, k, l = populations
    
    # Helper to extract counts and valid sizes
    def get_pop_data(pop_idx):
        start = pop_idx * 4
        pop_counts = counts[:, start:start+4]
        if n_valid is not None:
            if isinstance(n_valid, tuple):
                if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                    pop_n = n_valid[pop_idx]
                else:
                    pop_n = cp.sum(pop_counts, axis=1)
            elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
                pop_n = n_valid[:, pop_idx]
            else:
                pop_n = n_valid
        else:
            pop_n = cp.sum(pop_counts, axis=1)
        return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n
    
    # For pi2(i,j,j,j) where j==k==l and i!=j
    c11, c12, c13, c14, n1 = get_pop_data(j)  # The population that appears 3 times
    c21, c22, c23, c24, n2 = get_pop_data(i)  # The population that appears once
    
    # From moments _pi2_iiij formula
    numer = (
        -((c11 + c12) * c14 * (c21 + c23))
        - (c12 * (c13 + c14) * (c21 + c23))
        + ((c11 + c12) * (c12 + c14) * (c13 + c14) * (c21 + c23))
        + ((c11 + c12) * (c13 + c14) * (-2 * c22 - 2 * c24))
        + ((c11 + c12) * c14 * (c22 + c24))
        + (c12 * (c13 + c14) * (c22 + c24))
        + ((c11 + c12) * (c11 + c13) * (c13 + c14) * (c22 + c24))
    ) / 2.0
    
    denom = n2 * n1 * (n1 - 1) * (n1 - 2)
    valid_mask = (n1 >= 3) & (n2 >= 1)
    
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def _pi2_iikl(counts: cp.ndarray,
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,i,k,l) configurations."""
    i, j, k, l = populations
    
    # Helper to extract counts and valid sizes
    def get_pop_data(pop_idx):
        start = pop_idx * 4
        pop_counts = counts[:, start:start+4]
        if n_valid is not None:
            if isinstance(n_valid, tuple):
                if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                    pop_n = n_valid[pop_idx]
                else:
                    pop_n = cp.sum(pop_counts, axis=1)
            elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
                pop_n = n_valid[:, pop_idx]
            else:
                pop_n = n_valid
        else:
            pop_n = cp.sum(pop_counts, axis=1)
        return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n
    
    # Get all unique populations involved
    unique_pops = list(set([i, k, l]))
    
    if len(unique_pops) == 2:  # Cases like (0,0,0,1) 
        # Population i appears 3 times, other population appears once
        pop_major = i
        pop_minor = k if k != i else l
        
        c_major1, c_major2, c_major3, c_major4, n_major = get_pop_data(pop_major)
        c_minor1, c_minor2, c_minor3, c_minor4, n_minor = get_pop_data(pop_minor)
        
        # From moments pi2 formula for (i,i,i,j) case
        numer = (
            -((c_major1 + c_major2) * c_major4 * (c_minor1 + c_minor3))
            - (c_major2 * (c_major3 + c_major4) * (c_minor1 + c_minor3))
            + ((c_major1 + c_major2) * (c_major2 + c_major4) * (c_major3 + c_major4) * (c_minor1 + c_minor3))
            + ((c_major1 + c_major2) * (c_major3 + c_major4) * (-2 * c_minor2 - 2 * c_minor4))
            + ((c_major1 + c_major2) * c_major4 * (c_minor2 + c_minor4))
            + (c_major2 * (c_major3 + c_major4) * (c_minor2 + c_minor4))
            + ((c_major1 + c_major2) * (c_major1 + c_major3) * (c_major3 + c_major4) * (c_minor2 + c_minor4))
        ) / 2.0
        
        denom = n_minor * n_major * (n_major - 1) * (n_major - 2)
        valid_mask = (n_major >= 3) & (n_minor >= 1)
        result = cp.zeros_like(n_major, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    else:
        # Other cases - return zeros for now
        n1 = get_pop_data(0)[4]
        result = cp.zeros_like(n1, dtype=cp.float64)
        
    return result


def _pi2_ijkk(counts: cp.ndarray,
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,j,k,k) configurations."""
    i, j, k, l = populations
    
    # Helper to extract counts and valid sizes
    def get_pop_data(pop_idx):
        start = pop_idx * 4
        pop_counts = counts[:, start:start+4]
        if n_valid is not None:
            if isinstance(n_valid, tuple):
                if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                    pop_n = n_valid[pop_idx]
                else:
                    pop_n = cp.sum(pop_counts, axis=1)
            elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
                pop_n = n_valid[:, pop_idx]
            else:
                pop_n = n_valid
        else:
            pop_n = cp.sum(pop_counts, axis=1)
        return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n
    
    # From moments: pi2(i,j,k,k) where pop3 == pop4
    c11, c12, c13, c14, n1 = get_pop_data(k)  # pop3/pop4 (k)
    c21, c22, c23, c24, n2 = get_pop_data(i)  # pop1 (i)
    
    # Special case: if j == k, cs3 is the same as cs1
    if j == k:
        c31, c32, c33, c34, n3 = c11, c12, c13, c14, n1
    else:
        c31, c32, c33, c34, n3 = get_pop_data(j)  # pop2 (j)
    
    # From moments formula
    numer = (
        (c11 + c13)
        * (c12 + c14)
        * (c23 * (c31 + c32) + c24 * (c31 + c32) + (c21 + c22) * (c33 + c34))
    ) / 2.0
    
    denom = n1 * (n1 - 1) * n2 * n3
    valid_mask = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


# Backward compatibility layer
def DD(counts, n_valid=None):
    """Deprecated: Use dd() instead."""
    import warnings
    warnings.warn(
        "DD() is deprecated. Use ld_statistics.dd() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return dd_within(counts, n_valid)


def DD_two_pops(counts, pop1_idx, pop2_idx, n_valid1=None, n_valid2=None):
    """Deprecated: Use dd() with populations parameter instead."""
    import warnings
    warnings.warn(
        "DD_two_pops() is deprecated. Use ld_statistics.dd(counts, populations=(pop1_idx, pop2_idx)) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Reconstruct the expected format
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return dd(counts, populations=(pop1_idx, pop2_idx), n_valid=n_valid)


def Dz_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """Deprecated: Use dz() with populations parameter instead."""
    import warnings
    warnings.warn(
        "Dz_two_pops() is deprecated. Use ld_statistics.dz(counts, populations=pop_indices) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return dz(counts, populations=pop_indices, n_valid=n_valid)


def pi2_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """Deprecated: Use pi2() with populations parameter instead."""
    import warnings
    warnings.warn(
        "pi2_two_pops() is deprecated. Use ld_statistics.pi2(counts, populations=pop_indices) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return pi2(counts, populations=pop_indices, n_valid=n_valid)