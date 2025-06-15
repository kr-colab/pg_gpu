import cupy as cp

def DD_two_pops(counts, pop1_idx, pop2_idx, n_valid1=None, n_valid2=None):
    """
    Compute D^2 statistic for two populations with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 8) with haplotype counts for both populations
        pop1_idx (int): Population index (0 or 1) for first population
        pop2_idx (int): Population index (0 or 1) for second population
        n_valid1 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop1
        n_valid2 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop2
        
    Returns:
        cp.ndarray: The computed D^2 values
    """
    # Extract counts for each population
    if pop1_idx == pop2_idx:
        # Same population DD
        if pop1_idx == 0:
            c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            c1, c2, c3, c4 = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        
        numer = c1 * (c1 - 1) * c4 * (c4 - 1) + c2 * (c2 - 1) * c3 * (c3 - 1) - 2 * c1 * c2 * c3 * c4
        denom = n * (n - 1) * (n - 2) * (n - 3)
        
        valid_mask = n >= 4
        result = cp.zeros_like(n, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    else:
        # Cross-population DD
        c11, c12, c13, c14 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
        c21, c22, c23, c24 = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
        
        # Use provided valid counts or compute from haplotype counts
        n1 = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        n2 = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        
        D1 = c12 * c13 - c11 * c14
        D2 = c22 * c23 - c21 * c24
        
        numer = D1 * D2
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def Dz_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """
    Compute Dz statistic for two populations with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 8) with haplotype counts for both populations
        pop_indices (tuple): Three population indices (pop1, pop2, pop3) where each is 0 or 1
        n_valid1 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop1
        n_valid2 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop2
        
    Returns:
        cp.ndarray: The computed Dz values
    """
    pop1, pop2, pop3 = pop_indices
    
    # Helper to get counts and sample sizes for a population
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    if pop1 == pop2 == pop3:
        # Single population Dz
        c1, c2, c3, c4, n = get_counts(pop1)
        
        diff = c1 * c4 - c2 * c3
        sum_34_12 = (c3 + c4) - (c1 + c2)
        sum_24_13 = (c2 + c4) - (c1 + c3)
        sum_23_14 = (c2 + c3) - (c1 + c4)
        
        numer = diff * sum_34_12 * sum_24_13 + diff * sum_23_14 + 2 * (c2 * c3 + c1 * c4)
        denom = n * (n - 1) * (n - 2) * (n - 3)
        
        valid_mask = n >= 4
        result = cp.zeros_like(n, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    elif pop1 == pop2:  # Dz(i,i,j)
        c11, c12, c13, c14, n1 = get_counts(pop1)
        c21, c22, c23, c24, n2 = get_counts(pop3)
        
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
        c11, c12, c13, c14, n1 = get_counts(pop1)
        c21, c22, c23, c24, n2 = get_counts(pop2)
        
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
        c11, c12, c13, c14, n1 = get_counts(pop1)
        c21, c22, c23, c24, n2 = get_counts(pop2)
        
        numer = (-(c12 * c13) + c11 * c14) * (-c21 + c22 + c23 - c24) + (
            -(c12 * c13) + c11 * c14
        ) * (-c21 + c22 - c23 + c24) * (-c21 - c22 + c23 + c24)
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    else:  # All different populations - not supported for two populations
        # Return zeros with proper shape
        n1 = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        result = cp.zeros_like(n1, dtype=cp.float64)
    
    return result


def pi2_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """
    Compute π₂ statistic for two populations with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 8) with haplotype counts for both populations
        pop_indices (tuple): Four population indices (i,j,k,l) where each is 0 or 1
        n_valid1 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop1
        n_valid2 (cp.ndarray, optional): Array of shape (N,) with valid sample sizes for pop2
        
    Returns:
        cp.ndarray: The computed π₂ values
    """
    i, j, k, l = pop_indices
    
    # Helper to get counts and sample sizes
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    if i == j == k == l:
        # Single population pi2
        c1, c2, c3, c4, n = get_counts(i)
        
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
        
    elif i == j and k == l and i != k:
        # pi2(i,i,j,j) - average of two permutations
        c11, c12, c13, c14, n1 = get_counts(i)
        c21, c22, c23, c24, n2 = get_counts(k)
        
        # First permutation: pi2(i,i,j,j)
        numer1 = (c11 + c12) * (c13 + c14) * (c21 + c23) * (c22 + c24)
        # Second permutation: pi2(j,j,i,i)
        numer2 = (c21 + c22) * (c23 + c24) * (c11 + c13) * (c12 + c14)
        
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = 0.5 * (numer1[valid_mask] + numer2[valid_mask]) / denom[valid_mask]
        
    elif i == j and k != l:
        # pi2(i,i,k,l) type - average 4 permutations
        if i == k or i == l:
            # Cases like (0,0,0,1) or (0,0,1,0)
            result = _pi2_iikl(counts, pop_indices, n_valid1, n_valid2)
        else:
            # Cases like (0,0,1,1) already handled above
            n1 = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
            result = cp.zeros_like(n1, dtype=cp.float64)
            
    elif j == k == l and i != j:
        # pi2(i,j,j,j) type - use _pi2_iiij with populations swapped
        result = _pi2_iiij(counts, pop_indices, n_valid1, n_valid2)
        
    elif i != j and k == l:
        # pi2(i,j,k,k) type - average 4 permutations  
        result = _pi2_ijkk(counts, pop_indices, n_valid1, n_valid2)
        
    elif (i == k and j == l) or (i == l and j == k):
        # pi2(i,j,i,j) or pi2(i,j,j,i) type
        result = _pi2_ijij(counts, pop_indices, n_valid1, n_valid2)
        
    else:
        # General case - for now return zeros
        n1 = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        result = cp.zeros_like(n1, dtype=cp.float64)
    
    return result


def _pi2_iikl(counts, indices, n_valid1=None, n_valid2=None):
    """Helper for pi2(i,i,k,l) configurations."""
    i, j, k, l = indices
    
    # Helper to get counts and sample sizes
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    # Need to average 4 permutations
    n1 = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
    result = cp.zeros_like(n1, dtype=cp.float64)
    
    # Get all unique populations involved
    unique_pops = list(set([i, k, l]))
    
    if len(unique_pops) == 2:  # Cases like (0,0,0,1) 
        # Population i appears 3 times, other population appears once
        pop_major = i
        pop_minor = k if k != i else l
        
        c_major1, c_major2, c_major3, c_major4, n_major = get_counts(pop_major)
        c_minor1, c_minor2, c_minor3, c_minor4, n_minor = get_counts(pop_minor)
        
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
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]
        
    return result


def _pi2_iiij(counts, indices, n_valid1=None, n_valid2=None):
    """Helper for pi2(i,j,j,j) configurations."""
    i, j, k, l = indices
    
    # Helper to get counts and sample sizes
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    # For pi2(i,j,j,j) where j==k==l and i!=j
    # This maps to moments' _pi2_iiij(counts[j], counts[i])
    c11, c12, c13, c14, n1 = get_counts(j)  # The population that appears 3 times
    c21, c22, c23, c24, n2 = get_counts(i)  # The population that appears once
    
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


def _pi2_ijkk(counts, indices, n_valid1=None, n_valid2=None):
    """Helper for pi2(i,j,k,k) configurations."""
    i, j, k, l = indices
    
    # Helper to get counts and sample sizes
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    # From moments: pi2(i,j,k,k) where pop3 == pop4
    # cs1 = counts[pop3] = counts[k]
    # cs2 = counts[pop1] = counts[i]  
    # cs3 = counts[pop2] = counts[j]
    
    c11, c12, c13, c14, n1 = get_counts(k)  # pop3/pop4 (k)
    c21, c22, c23, c24, n2 = get_counts(i)  # pop1 (i)
    
    # Special case: if j == k, cs3 is the same as cs1
    if j == k:
        c31, c32, c33, c34, n3 = c11, c12, c13, c14, n1
    else:
        c31, c32, c33, c34, n3 = get_counts(j)  # pop2 (j)
    
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


def _pi2_ijij(counts, indices, n_valid1=None, n_valid2=None):
    """Helper for pi2(i,j,i,j) configurations."""
    i, j, k, l = indices
    
    def get_counts(pop_idx):
        if pop_idx == 0:
            counts_pop = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
            n = n_valid1 if n_valid1 is not None else cp.sum(counts[:, :4], axis=1)
        else:
            counts_pop = counts[:, 4], counts[:, 5], counts[:, 6], counts[:, 7]
            n = n_valid2 if n_valid2 is not None else cp.sum(counts[:, 4:], axis=1)
        return counts_pop + (n,)
    
    c11, c12, c13, c14, n1 = get_counts(i)
    c21, c22, c23, c24, n2 = get_counts(j)
    
    # From moments implementation for pi2(i,j,i,j)
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
    
    return result


# Single population convenience functions
def DD(counts, n_valid=None):
    """
    Compute D^2 statistic for a single population with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 4) with haplotype counts
        n_valid (cp.ndarray, optional): Array of shape (N,) with valid sample sizes
        
    Returns:
        cp.ndarray: The computed D^2 values
    """
    # Direct implementation for single population
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)
    
    numer = c1 * (c1 - 1) * c4 * (c4 - 1) + c2 * (c2 - 1) * c3 * (c3 - 1) - 2 * c1 * c2 * c3 * c4
    denom = n * (n - 1) * (n - 2) * (n - 3)
    
    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    
    return result


def Dz(counts, n_valid=None):
    """
    Compute Dz statistic for a single population with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 4) with haplotype counts
        n_valid (cp.ndarray, optional): Array of shape (N,) with valid sample sizes
        
    Returns:
        cp.ndarray: The computed Dz values
    """
    # Direct implementation for single population
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


def pi2(counts, n_valid=None):
    """
    Compute π₂ statistic for a single population with automatic missing data handling.
    
    Parameters:
        counts (cp.ndarray): Array of shape (N, 4) with haplotype counts
        n_valid (cp.ndarray, optional): Array of shape (N,) with valid sample sizes
        
    Returns:
        cp.ndarray: The computed π₂ values
    """
    # Direct implementation for single population
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