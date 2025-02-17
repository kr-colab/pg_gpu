import cupy as cp

def D(counts):
    """
    Compute the linkage disequilibrium statistic D given haplotype counts.
    
    For each set of counts [n11, n10, n01, n00]:
      D = (n11 * n00 - n10 * n01) / (n * (n - 1))
    where n = n11 + n10 + n01 + n00.
    
    Parameters:
        counts (cp.ndarray): An array of shape (4,) or (N, 4) where each row represents 
                             the 2x2 table counts [n11, n10, n01, n00].
                             
    Returns:
        cp.ndarray: The computed D values (scalar if counts was 1D or an array for multiple rows).
    """
    # Ensure counts is two-dimensional.
    if counts.ndim == 1:
        counts = counts[None, :]
    c1 = counts[:, 0]
    c2 = counts[:, 1]
    c3 = counts[:, 2]
    c4 = counts[:, 3]
    n = cp.sum(counts, axis=1)
    numer = c1 * c4 - c2 * c3
    D_val = numer / (n * (n - 1))
    return cp.squeeze(D_val)

def DD(counts):
    """
    Compute the squared disequilibrium statistic (D^2) from haplotype counts.
    
    For a single population (counts from a single group), if counts = [n11, n10, n01, n00],
    we compute:
      numer = n11*(n11-1)*n00*(n00-1) + n10*(n10-1)*n01*(n01-1) - 2*n11*n10*n01*n00
      DD = numer / (n * (n - 1) * (n - 2) * (n - 3))
    where n = n11 + n10 + n01 + n00.
    
    Parameters:
        counts (cp.ndarray): An array of shape (4,) or (N, 4) with the counts.
        
    Returns:
        cp.ndarray: The computed D^2 values.
    """
    if counts.ndim == 1:
        counts = counts[None, :]
    c1 = counts[:, 0]
    c2 = counts[:, 1]
    c3 = counts[:, 2]
    c4 = counts[:, 3]
    n = cp.sum(counts, axis=1)
    numer = (c1 * (c1 - 1) * c4 * (c4 - 1) +
             c2 * (c2 - 1) * c3 * (c3 - 1) -
             2 * c1 * c2 * c3 * c4)
    DD_val = numer / (n * (n - 1) * (n - 2) * (n - 3))
    return cp.squeeze(DD_val)

def Dz(counts):
    """
    Compute the Dz statistic from haplotype counts.
    
    For a single population with counts [n11, n10, n01, n00], one formulation is:
      Dz = { (n11*n00 - n10*n01) * [(n01+n00) - (n11+n10)] * [(n10+n00) - (n11+n01)]
             + (n11*n00 - n10*n01) * [(n10+n01) - (n11+n00)]
             + 2*(n10*n01 + n11*n00) } / (n*(n-1)*(n-2)*(n-3))
    where n = n11 + n10 + n01 + n00.
    
    Parameters:
        counts (cp.ndarray): An array of shape (4,) or (N, 4) with the counts.
        
    Returns:
        cp.ndarray: The computed Dz values.
    """
    if counts.ndim == 1:
        counts = counts[None, :]
    c1 = counts[:, 0]
    c2 = counts[:, 1]
    c3 = counts[:, 2]
    c4 = counts[:, 3]
    n = cp.sum(counts, axis=1)
    # Compute the components of the formula.
    diff = c1 * c4 - c2 * c3
    term1 = (c3 + c4 - c1 - c2)
    term2 = (c2 + c4 - c1 - c3)
    term3 = (c2 + c3 - c1 - c4)
    numer = diff * term1 * term2 + diff * term3 + 2 * (c2 * c3 + c1 * c4)
    Dz_val = numer / (n * (n - 1) * (n - 2) * (n - 3))
    return cp.squeeze(Dz_val)

def pi2(counts):
    """
    Compute the π₂ statistic from haplotype counts.
    
    For a single population with counts [n11, n10, n01, n00],
    one formulation is:
      π₂ = { ( (n11+n10)*(n11+n01)*(n10+n00)*(n01+n00)
              - n11*n00*( -1 + n11+3*n10+3*n01+n00 )
              - n10*n01*( -1 + 3*n11+n10+n01+3*n00 ) )
             } / (n*(n-1)*(n-2)*(n-3))
    where n = n11 + n10 + n01 + n00.
    
    Parameters:
        counts (cp.ndarray): An array of shape (4,) or (N, 4) with the counts.
        
    Returns:
        cp.ndarray: The computed π₂ values.
    """
    if counts.ndim == 1:
        counts = counts[None, :]
    c1 = counts[:, 0]
    c2 = counts[:, 1]
    c3 = counts[:, 2]
    c4 = counts[:, 3]
    n = cp.sum(counts, axis=1)
    term_a = (c1 + c2) * (c1 + c3) * (c2 + c4) * (c3 + c4)
    term_b = c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
    term_c = c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4)
    numer = term_a - term_b - term_c
    pi2_val = numer / (n * (n - 1) * (n - 2) * (n - 3))
    return cp.squeeze(pi2_val)

# (Additional GPU-optimized functions could be added here as needed) 