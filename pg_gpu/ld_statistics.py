"""
GPU-accelerated linkage disequilibrium statistics.

This module provides an API for computing LD statistics
on GPUs with automatic missing data handling.
"""

import numpy as np
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


def zns(r2_matrix):
    """Kelly's ZnS: mean pairwise r-squared across all SNP pairs.

    Parameters
    ----------
    r2_matrix : cupy or numpy ndarray, shape (m, m)
        Square r-squared matrix (e.g. from HaplotypeMatrix.pairwise_r2()).

    Returns
    -------
    float
        Mean r-squared (excluding diagonal).
    """
    if not isinstance(r2_matrix, cp.ndarray):
        r2_matrix = cp.asarray(r2_matrix, dtype=cp.float64)

    m = r2_matrix.shape[0]
    if m < 2:
        return 0.0
    total = cp.sum(r2_matrix) - cp.trace(r2_matrix)
    return float((total / (m * (m - 1))).get())


def omega(r2_matrix):
    """Kim and Nielsen's Omega: max ratio of within-partition to
    cross-partition mean LD.

    For each possible SNP partition point, computes the ratio of
    mean r-squared within the two blocks to mean r-squared between
    blocks. Returns the maximum ratio across all partition points.

    Uses GPU prefix sums to evaluate all partition points without
    a Python loop.

    Parameters
    ----------
    r2_matrix : cupy or numpy ndarray, shape (m, m)
        Square r-squared matrix.

    Returns
    -------
    float
        Maximum omega value. Returns 0 if fewer than 4 SNPs.
    """
    if not isinstance(r2_matrix, cp.ndarray):
        r2_matrix = cp.asarray(r2_matrix, dtype=cp.float64)

    m = r2_matrix.shape[0]
    if m < 4:
        return 0.0

    r2 = r2_matrix.astype(cp.float64)

    # precompute 2D prefix sums for O(1) block sum queries
    # S[i,j] = sum of r2[0:i, 0:j]
    S = cp.cumsum(cp.cumsum(r2, axis=0), axis=1)

    # prefix sum of diagonal for trace computation
    diag = cp.diag(r2)
    diag_cumsum = cp.cumsum(diag)

    # total sum and trace
    total_sum = S[m-1, m-1]
    total_trace = diag_cumsum[m-1]

    # for partition at l (left=[0:l], right=[l:m]):
    # sum_left = S[l-1, l-1]
    # sum_cross = S[l-1, m-1] - S[l-1, l-1]  (top-right block)
    #           + S[m-1, l-1] - S[l-1, l-1]  (bottom-left block)
    # but r2 is symmetric so cross = 2 * (S[l-1, m-1] - S[l-1, l-1])
    # sum_right = total_sum - S[l-1, m-1] - S[m-1, l-1] + S[l-1, l-1]

    # evaluate all partition points l = 2..m-2 in parallel
    l_vals = cp.arange(2, m - 1)

    sum_left = S[l_vals - 1, l_vals - 1]
    trace_left = diag_cumsum[l_vals - 1]

    sum_right = total_sum - S[l_vals - 1, m-1] - S[m-1, l_vals - 1] + sum_left
    trace_right = total_trace - trace_left

    # cross = total - left_block - right_block
    # S[l-1, m-1] = sum of rows 0..l-1, all cols
    # left block = S[l-1, l-1], right block needs careful extraction
    sum_cross = total_sum - sum_left - sum_right
    # cross has no diagonal contribution so no trace adjustment needed
    # but we double-counted: cross appears in both off-diagonal blocks
    # Actually: sum_left + sum_right + sum_cross = total_sum
    # where sum_cross is the sum of both off-diagonal blocks combined

    n_left = l_vals * (l_vals - 1)
    n_right = (m - l_vals) * (m - l_vals - 1)
    n_cross = 2 * l_vals * (m - l_vals)  # both off-diagonal blocks

    # within = (left excl diag + right excl diag)
    within_sum = (sum_left - trace_left) + (sum_right - trace_right)
    n_within = n_left + n_right

    valid = (n_within > 0) & (n_cross > 0) & (sum_cross > 0)

    omega_vals = cp.zeros_like(l_vals, dtype=cp.float64)
    mean_within = cp.where(n_within > 0, within_sum / n_within, 0.0)
    mean_cross = cp.where(n_cross > 0, sum_cross / n_cross, 1.0)
    omega_vals = cp.where(valid, mean_within / mean_cross, 0.0)

    return float(cp.max(omega_vals).get())


def mu_ld(haplotype_matrix):
    """mu_LD: haplotype pattern exclusivity between left/right halves (RAiSD).

    Splits variants at midpoint and measures how exclusively haplotype
    patterns associate across halves. Elevated at sweep boundaries where
    LD structure changes abruptly.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix

    Returns
    -------
    float
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = cp.maximum(haplotype_matrix.haplotypes, 0)
    n_hap, n_var = hap.shape

    if n_var < 2:
        return 0.0

    mid = n_var // 2

    # transfer to CPU for unique-row hashing (no GPU equivalent for byte-string uniqueness)
    left = hap[:, :mid].get().astype(np.int8)
    right = hap[:, mid:].get().astype(np.int8)

    left_bytes = [row.tobytes() for row in left]
    right_bytes = [row.tobytes() for row in right]

    # for each distinct left pattern, count how many distinct right patterns it pairs with
    left_to_right = {}
    right_to_left = {}
    for i in range(n_hap):
        lb, rb = left_bytes[i], right_bytes[i]
        left_to_right.setdefault(lb, set()).add(rb)
        right_to_left.setdefault(rb, set()).add(lb)

    n_left = len(left_to_right)
    n_right = len(right_to_left)

    if n_left == 0 or n_right == 0:
        return 0.0

    # exclusive = pattern paired with only one pattern in the other half
    n_excl_left = sum(1 for v in left_to_right.values() if len(v) == 1)
    n_excl_right = sum(1 for v in right_to_left.values() if len(v) == 1)

    return float((n_excl_left / n_left + n_excl_right / n_right) / 2.0)


def r2_matrix_diploid(genotype_matrix):
    """Compute r-squared matrix from diploid genotypes (0/1/2) on GPU.

    Uses genotype correlation: treats 0/1/2 as continuous dosage values,
    computes Pearson correlation, then squares.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix or cupy.ndarray
        If GenotypeMatrix, uses .genotypes. If array, shape (n_individuals, n_variants).

    Returns
    -------
    r2 : cupy.ndarray, float64, shape (n_variants, n_variants)
    """
    from .genotype_matrix import GenotypeMatrix

    if isinstance(genotype_matrix, GenotypeMatrix):
        if genotype_matrix.device == 'CPU':
            genotype_matrix.transfer_to_gpu()
        geno = genotype_matrix.genotypes
    else:
        geno = genotype_matrix

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    geno = cp.maximum(geno, 0).astype(cp.float64)
    n_ind = geno.shape[0]

    # center genotypes per variant
    mean = cp.mean(geno, axis=0)
    gn = geno - mean

    # variance per variant
    var = cp.sum(gn ** 2, axis=0)
    valid = var > 0

    # correlation via matrix multiply
    cov = gn.T @ gn  # (n_var, n_var)

    # normalize: r_ij = cov_ij / sqrt(var_i * var_j)
    denom = cp.sqrt(cp.outer(var, var))
    r2 = cp.where(denom > 0, (cov / denom) ** 2, 0.0)
    cp.fill_diagonal(r2, 0.0)

    return r2


def zns_diploid(genotype_matrix):
    """Kelly's ZnS from diploid genotypes.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix

    Returns
    -------
    float
    """
    r2 = r2_matrix_diploid(genotype_matrix)
    return zns(r2)


def omega_diploid(genotype_matrix):
    """Kim and Nielsen's Omega from diploid genotypes.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix

    Returns
    -------
    float
    """
    r2 = r2_matrix_diploid(genotype_matrix)
    return omega(r2)


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