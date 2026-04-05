"""Memory-safe GPU utilities for chunked operations over variants."""

import cupy as cp


def estimate_variant_chunk_size(n_hap, bytes_per_element=4, n_intermediates=3,
                                 memory_fraction=0.4):
    """Estimate how many variants can be processed per chunk.

    Parameters
    ----------
    n_hap : int
        Number of haplotypes (rows).
    bytes_per_element : int
        Bytes per element in the working dtype (4 for int32/float32).
    n_intermediates : int
        Number of intermediate arrays of size (n_hap, chunk_size) created.
    memory_fraction : float
        Fraction of free GPU memory to use.

    Returns
    -------
    int
        Number of variants per chunk.
    """
    free = cp.cuda.Device().mem_info[0]
    budget = int(free * memory_fraction)
    per_variant = n_hap * bytes_per_element * n_intermediates
    chunk = max(1, budget // per_variant)
    return chunk


def estimate_fused_chunk_size(n_hap, memory_fraction=0.35):
    """Estimate max variants for a transposed int8 chunk in fused kernels.

    Parameters
    ----------
    n_hap : int
        Number of haplotypes.
    memory_fraction : float
        Fraction of free GPU memory to budget for the transposed copy.

    Returns
    -------
    int
        Number of variants per chunk.
    """
    free = cp.cuda.Device().mem_info[0]
    budget = int(free * memory_fraction)
    # Each variant needs n_hap bytes (int8) in the transposed copy
    chunk = max(1, budget // max(n_hap, 1))
    return chunk


def free_gpu_pool():
    """Release unused GPU memory back to the device."""
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()


def chunked_sum_int32(hap, axis=0):
    """Sum haplotype matrix along axis 0 using int32 chunks.

    Avoids creating a full int32 copy of the matrix by processing
    variant columns in chunks.

    Parameters
    ----------
    hap : cupy.ndarray, int8, shape (n_hap, n_var)

    Returns
    -------
    cupy.ndarray, int64, shape (n_var,)
    """
    n_hap, n_var = hap.shape
    chunk_size = estimate_variant_chunk_size(n_hap, bytes_per_element=4,
                                             n_intermediates=1)
    result = cp.empty(n_var, dtype=cp.int64)
    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        result[start:end] = cp.sum(hap[:, start:end].astype(cp.int32), axis=0)
    return result


def chunked_dac_and_n(hap):
    """Compute derived allele counts and valid counts, memory-safe.

    Uses the fast inline path when the matrix fits in memory, falling
    back to chunked processing for large matrices.

    Parameters
    ----------
    hap : cupy.ndarray, int8, shape (n_hap, n_var)

    Returns
    -------
    dac : cupy.ndarray, int64, shape (n_var,)
    n_valid : cupy.ndarray, int64, shape (n_var,)
    """
    n_hap, n_var = hap.shape

    # Fast path: if int32 intermediates fit in ~40% of free memory, no chunking
    free = cp.cuda.Device().mem_info[0]
    needed = n_hap * n_var * 4 * 2  # two int32 intermediates
    if needed < free * 0.4:
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask.astype(cp.int32), axis=0).astype(cp.int64)
        dac = cp.sum((hap * valid_mask).astype(cp.int32), axis=0).astype(cp.int64)
        return dac, n_valid

    # Chunked path for large matrices
    chunk_size = estimate_variant_chunk_size(n_hap, bytes_per_element=4,
                                             n_intermediates=2)
    dac = cp.empty(n_var, dtype=cp.int64)
    n_valid = cp.empty(n_var, dtype=cp.int64)

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        chunk = hap[:, start:end]
        valid = (chunk >= 0).astype(cp.int32)
        n_valid[start:end] = cp.sum(valid, axis=0)
        dac[start:end] = cp.sum((chunk * valid).astype(cp.int32), axis=0)

    return dac, n_valid


def chunked_matmul_accumulate(X, chunk_size=None):
    """Compute X @ X.T by accumulating partial outer products.

    Splits X along columns (variant axis) to control memory.
    Result is exact (no approximation).

    Parameters
    ----------
    X : cupy.ndarray, shape (n, m)
    chunk_size : int, optional
        Columns per chunk. Auto-estimated if None.

    Returns
    -------
    cupy.ndarray, shape (n, n)
    """
    n, m = X.shape
    if chunk_size is None:
        free = cp.cuda.Device().mem_info[0]
        # Each chunk needs (n, chunk) working memory + (n, n) output
        output_bytes = n * n * 8  # float64
        budget = int(free * 0.4) - output_bytes
        per_col = n * 8  # float64
        chunk_size = max(1, budget // per_col)

    result = cp.zeros((n, n), dtype=cp.float64)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        chunk = X[:, start:end]
        result += chunk @ chunk.T

    return result
