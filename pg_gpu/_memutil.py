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


def estimate_indiv_block_size(n_ind, bytes_per_element=8,
                              n_intermediates=4, memory_fraction=0.25):
    """Pick how many individuals to process per row block when streaming
    relatedness kernels accumulate an (n_ind, n_ind) output by tiling
    the individual axis.

    Each row block holds ~``n_intermediates`` working arrays of shape
    ``(block_size, n_ind)`` on the GPU (typically the row-block slice
    of the indicator matmul plus the matmul output). Budgets the
    requested fraction of free GPU memory for those.

    Returns at least 1 and at most ``n_ind`` (a single block covers
    every individual, equivalent to no tiling).
    """
    free = cp.cuda.Device().mem_info[0]
    budget = int(free * memory_fraction)
    per_row = n_ind * bytes_per_element * n_intermediates
    block = max(1, budget // per_row)
    return min(block, n_ind)


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


_dac_and_n_kernel = cp.RawKernel(r'''
extern "C" __global__
void dac_and_n(const signed char* hap, int n_hap, int n_var,
               long long stride0, long long stride1,
               long long* out_dac, long long* out_n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_var) return;
    int d = 0, nv = 0;
    for (int i = 0; i < n_hap; i++) {
        signed char v = hap[i * stride0 + j * stride1];
        if (v >= 0) nv++;
        if (v > 0) d++;
    }
    out_dac[j] = d;
    out_n[j] = nv;
}
''', 'dac_and_n')


def dac_and_n(hap):
    """Compute derived allele counts and valid counts via fused CUDA kernel.

    Single-pass kernel: reads each element once, no intermediate arrays.
    Counts non-reference alleles (hap > 0) as derived, handling
    multiallelic sites correctly.

    Parameters
    ----------
    hap : cupy.ndarray, int8, shape (n_hap, n_var)

    Returns
    -------
    dac : cupy.ndarray, int64, shape (n_var,)
        Derived (non-reference) allele count per site.
    n_valid : cupy.ndarray, int64, shape (n_var,)
        Number of non-missing haplotypes per site.
    """
    n_hap, n_var = hap.shape
    out_dac = cp.empty(n_var, dtype=cp.int64)
    out_n = cp.empty(n_var, dtype=cp.int64)
    s0, s1 = hap.strides
    threads = 256
    blocks = (n_var + threads - 1) // threads
    _dac_and_n_kernel((blocks,), (threads,),
                      (hap, n_hap, n_var, s0, s1, out_dac, out_n))
    return out_dac, out_n


# Backward-compatible alias
chunked_dac_and_n = dac_and_n


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
