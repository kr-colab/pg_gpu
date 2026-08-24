"""Memory-safe GPU utilities for chunked operations over variants."""

import cupy as cp

# Threads per block for the one-thread-per-variant fused kernels.
_THREADS_PER_BLOCK = 256


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

    The ``cp.cuda.Device().mem_info`` read is single-shot per call;
    pg_gpu's streaming pipelines run on the default CUDA stream from
    one Python thread so no other code is touching the memory pool
    concurrently while this estimate is being computed.
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



_allele_counts_kernel = cp.RawKernel(r'''
extern "C" __global__
void allele_counts(const signed char* hap, int n_hap, int n_var,
                   long long stride0, long long stride1, int K,
                   long long* out_ac, long long* out_n) {
    // Per-thread histograms live in shared memory, laid out (allele, tid)
    // so consecutive threads hit consecutive banks. Tallying in shared
    // costs one 4-byte RMW per element; a global int64 tally would pay a
    // 16-byte round trip per 1-byte read, a ~17x bandwidth amplification.
    extern __shared__ int cnt[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    for (int a = 0; a < K; a++) cnt[a * blockDim.x + tid] = 0;
    if (j >= n_var) return;
    int nv = 0;
    for (int i = 0; i < n_hap; i++) {
        signed char v = hap[i * stride0 + j * stride1];
        if (v >= 0) {
            nv++;
            if (v < K) cnt[v * blockDim.x + tid]++;
        }
    }
    long long base = (long long)j * K;
    for (int a = 0; a < K; a++)
        out_ac[base + a] = (long long)cnt[a * blockDim.x + tid];
    out_n[j] = nv;
}
''', 'allele_counts')


# Fallback for allele counts too wide for a shared-memory histogram (a
# per-thread row would blow the 48 KB block budget). Same contract; the
# output row doubles as the histogram, one global RMW per element.
_allele_counts_kernel_widek = cp.RawKernel(r'''
extern "C" __global__
void allele_counts_widek(const signed char* hap, int n_hap, int n_var,
                         long long stride0, long long stride1, int K,
                         long long* out_ac, long long* out_n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_var) return;
    long long base = (long long)j * K;
    for (int a = 0; a < K; a++) out_ac[base + a] = 0;
    int nv = 0;
    for (int i = 0; i < n_hap; i++) {
        signed char v = hap[i * stride0 + j * stride1];
        if (v >= 0) {
            nv++;
            if (v < K) out_ac[base + v]++;
        }
    }
    out_n[j] = nv;
}
''', 'allele_counts_widek')


def allele_counts(hap, n_alleles=None):
    """Compute per-allele sample counts and valid counts via fused CUDA kernel.

    Single-pass kernel: one thread per variant reads each element once and
    tallies a per-allele histogram directly into its own output row (no
    intermediate array, no atomics). This is the multiallelic-correct
    counting primitive: allele ``a`` at a site gets its own count, rather
    than all non-reference alleles being collapsed into one derived
    class.

    Parameters
    ----------
    hap : cupy.ndarray, int8, shape (n_hap, n_var)
        Allele indices (0 = reference/ancestral, 1.. = alternate), -1 missing.
    n_alleles : int, optional
        Output width ``K`` (number of allele columns). If None, derived as
        ``max(hap) + 1`` via a one-shot reduction. Pass a global value when
        several calls (e.g. per population) must produce aligned matrices;
        it must be at least ``max(hap) + 1`` (counts for larger indices are
        dropped from ``ac`` but still counted in ``n_valid``).

    Returns
    -------
    ac : cupy.ndarray, int64, shape (n_var, K)
        Per-allele sample count; column ``a`` is the count of allele ``a``.
    n_valid : cupy.ndarray, int64, shape (n_var,)
        Number of non-missing haplotypes per site (== ac.sum(axis=1) when
        n_alleles covers the true allele range).
    """
    n_hap, n_var = hap.shape
    if n_alleles is None:
        K = (max(int(hap.max()), 0) + 1) if hap.size else 1
    else:
        K = int(n_alleles)
    if n_var == 0:
        return cp.empty((0, K), dtype=cp.int64), cp.empty(0, dtype=cp.int64)
    out_ac = cp.empty((n_var, K), dtype=cp.int64)
    out_n = cp.empty(n_var, dtype=cp.int64)
    s0, s1 = hap.strides
    threads = _THREADS_PER_BLOCK
    blocks = (n_var + threads - 1) // threads
    shared_bytes = K * threads * 4
    if shared_bytes <= 48 * 1024:
        _allele_counts_kernel((blocks,), (threads,),
                              (hap, n_hap, n_var, s0, s1, K, out_ac, out_n),
                              shared_mem=shared_bytes)
    else:
        _allele_counts_kernel_widek((blocks,), (threads,),
                                    (hap, n_hap, n_var, s0, s1, K,
                                     out_ac, out_n))
    return out_ac, out_n


