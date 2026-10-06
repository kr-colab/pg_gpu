"""Exact haplotype-identity hashing shared by Garud's H and haplotype counting.

Each haplotype in a window is reduced to two float64 hashes: the in-order sum
of a per-variant weight over the alleles it carries. One thread computes one
haplotype's sum in variant-index order, so two identical haplotypes give
bit-identical hashes, and a window with no variants hashes to zero. Distinct
haplotypes differ by a random combination of weights that is never within
rounding distance of zero, so grouping by exact hash equality counts distinct
haplotypes exactly. The kernels read the int8 matrix directly; nothing here
materializes a float copy of the haplotype matrix, and the per-window sort runs
in global memory, so the haplotype count is unbounded.
"""

import cupy as cp
import numpy as np

from . import _memutil

_GARUD_SALT1 = 0x9E3779B97F4A7C15   # golden-ratio constant
_GARUD_SALT2 = 0xC6BC279692B5C323   # phi^-2-based companion

_THREADS = _memutil._THREADS_PER_BLOCK

# Variants per segment when hashing whole rows. Splitting a long row into
# segments gives the GPU many threads per haplotype instead of one.
_ROW_SEGMENT = 4096


def _index_weights(n_var, salt):
    """Float64 weights in ``[-1, 1)`` for variant indices ``0 .. n_var - 1``.

    Splitmix64-scrambles each index with the salt. Two salts give two
    independent weight columns; together they make a collision between
    distinct haplotype patterns vanishingly rare. The distribution is
    uniform rather than normal because the weights only tell patterns
    apart.
    """
    s = cp.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    x = cp.arange(n_var, dtype=cp.uint64) + s
    x = (x ^ (x >> cp.uint64(30))) * cp.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> cp.uint64(27))) * cp.uint64(0x94D049BB133111EB)
    x = x ^ (x >> cp.uint64(31))
    # Mantissa-pack the low 52 bits as a float64 in [1, 2); subtract 1 to
    # get [0, 1); rescale to [-1, 1).
    mant = (x & cp.uint64(0x000FFFFFFFFFFFFF)) | cp.uint64(0x3FF0000000000000)
    u = mant.view(cp.float64) - 1.0
    return 2.0 * u - 1.0


def hash_weights(n_var):
    """The two weight vectors ``(w1, w2)`` for a matrix with ``n_var`` variants."""
    return _index_weights(n_var, _GARUD_SALT1), _index_weights(n_var, _GARUD_SALT2)


# One thread per (window, haplotype). The matrix is addressed with byte
# strides through a signed char pointer, so int8 and wider signed integer
# matrices in C or F order are read in place: a wider element is read through
# its low byte, which holds the allele index for the small values a haplotype
# matrix carries (-1 for missing, 0.. for alleles). All offsets are 64-bit.
_window_hash_kernel = cp.RawKernel(r'''
extern "C" __global__
void garud_window_hash(const signed char* hap, long long stride0, long long stride1,
                       const double* w1, const double* w2,
                       const long long* win_start, const long long* win_stop,
                       int n_hap, int n_windows, int blocks_per_window,
                       double* out1, double* out2) {
    long long b = blockIdx.x;
    int w = (int)(b / blocks_per_window);
    int h = (int)(b % blocks_per_window) * blockDim.x + threadIdx.x;
    if (w >= n_windows || h >= n_hap) return;
    const signed char* row = hap + (long long)h * stride0;
    long long lo = win_start[w];
    long long hi = win_stop[w];
    double a1 = 0.0;
    double a2 = 0.0;
    // Sequential in-order accumulation with explicit fma: identical
    // haplotypes give bit-identical sums, and an empty window sums to zero.
    for (long long v = lo; v < hi; v++) {
        double x = (double)row[v * stride1];
        a1 = fma(x, w1[v], a1);
        a2 = fma(x, w2[v], a2);
    }
    long long o = (long long)w * n_hap + h;
    out1[o] = a1;
    out2[o] = a2;
}
''', 'garud_window_hash')


# One thread per window walks that window's hashes, sorted by (h1, h2), and
# counts runs of equal pairs: each run is one distinct haplotype. It emits the
# sum of squared frequencies and the three largest frequencies; the H
# statistics are derived from those in garud_from_moments. Run lengths are
# tallied as exact integers and divided once, so the output depends only on
# the multiset of run lengths, not on the order the sort put them in.
_garud_walk_kernel = cp.RawKernel(r'''
extern "C" __global__
void garud_walk(const double* s1, const double* s2, int n_hap, int n_windows,
                double* out_sum_f2, double* out_top0, double* out_top1,
                double* out_top2, double* out_n_distinct) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_windows) return;
    const double* a = s1 + (long long)w * n_hap;
    const double* b = s2 + (long long)w * n_hap;
    long long sum_r2 = 0;
    int top0 = 0, top1 = 0, top2 = 0;
    int run = 1;
    int n_distinct = 0;
    for (int i = 1; i <= n_hap; i++) {
        bool boundary = (i == n_hap) || (a[i] != a[i - 1]) || (b[i] != b[i - 1]);
        if (!boundary) {
            run++;
            continue;
        }
        n_distinct++;
        sum_r2 += (long long)run * run;
        if (run > top0) {
            top2 = top1; top1 = top0; top0 = run;
        } else if (run > top1) {
            top2 = top1; top1 = run;
        } else if (run > top2) {
            top2 = run;
        }
        run = 1;
    }
    double n = (double)n_hap;
    out_sum_f2[w] = (double)sum_r2 / (n * n);
    out_top0[w] = (double)top0 / n;
    out_top1[w] = (double)top1 / n;
    out_top2[w] = (double)top2 / n;
    out_n_distinct[w] = (double)n_distinct;
}
''', 'garud_walk')


def garud_from_moments(sum_f2, top0, top1, top2):
    """H1, H12, H123 and H2/H1 from haplotype-frequency moments.

    ``sum_f2`` is the sum of squared haplotype frequencies (H1) and
    ``top0 >= top1 >= top2`` the three largest frequencies, zero when fewer
    haplotypes exist. Works on scalars and on arrays of any array module.
    """
    t12 = top0 + top1
    t123 = t12 + top2
    h12 = t12 * t12 + (sum_f2 - top0 * top0 - top1 * top1)
    h123 = t123 * t123 + (sum_f2 - top0 * top0 - top1 * top1 - top2 * top2)
    h2h1 = (sum_f2 - top0 * top0) / sum_f2
    return sum_f2, h12, h123, h2h1


def _as_kernel_input(hap):
    """A view the hash kernel can read in place: any signed integer dtype."""
    if hap.dtype.kind != 'i':
        hap = hap.astype(cp.int8)
    return hap


def window_hashes(hap, w1, w2, win_start, win_stop):
    """Per-window haplotype hashes.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_hap, n_var), signed integer
        Allele indices, -1 for missing. Any memory order.
    w1, w2 : cupy.ndarray, float64, shape (n_var,)
        Weight per variant, indexed by variant index (see ``hash_weights``).
    win_start, win_stop : array_like of int
        Right-open variant index range of each window, within ``[0, n_var]``.

    Returns
    -------
    h1, h2 : cupy.ndarray, float64, shape (n_windows, n_hap)
        A window with no variants hashes to 0.
    """
    hap = _as_kernel_input(hap)
    n_hap, n_var = hap.shape
    win_start = cp.ascontiguousarray(cp.asarray(win_start, dtype=cp.int64))
    win_stop = cp.ascontiguousarray(cp.asarray(win_stop, dtype=cp.int64))
    n_windows = int(win_start.shape[0])
    out1 = cp.empty((n_windows, n_hap), dtype=cp.float64)
    out2 = cp.empty((n_windows, n_hap), dtype=cp.float64)
    if n_windows == 0:
        return out1, out2
    # The kernel reads the ranges unchecked.
    if bool((win_start < 0).any()) or bool((win_stop > n_var).any()):
        raise ValueError(f"window ranges must lie within [0, {n_var}]")
    blocks_per_window = max(1, (n_hap + _THREADS - 1) // _THREADS)
    if n_windows * blocks_per_window > 2 ** 31 - 1:
        raise ValueError("too many windows for one launch; batch them")
    w1 = cp.ascontiguousarray(w1, dtype=cp.float64)
    w2 = cp.ascontiguousarray(w2, dtype=cp.float64)
    stride0, stride1 = hap.strides
    _window_hash_kernel(
        (n_windows * blocks_per_window,), (_THREADS,),
        (hap, np.int64(stride0), np.int64(stride1), w1, w2, win_start, win_stop,
         np.int32(n_hap), np.int32(n_windows), np.int32(blocks_per_window),
         out1, out2))
    return out1, out2


def sort_order(h1, h2):
    """Per-row permutation that sorts each window's hash pairs by (h1, h2).

    Two stable per-row argsorts (by h2, then by h1) give the lexicographic
    order within every window, in global memory.
    """
    o2 = cp.argsort(h2, axis=1)
    o1 = cp.argsort(cp.take_along_axis(h1, o2, axis=1), axis=1)
    return cp.take_along_axis(o2, o1, axis=1)


def sort_window_hashes(h1, h2):
    """Each window's hash pairs sorted by (h1, h2), as sorted copies."""
    order = sort_order(h1, h2)
    return cp.take_along_axis(h1, order, axis=1), cp.take_along_axis(h2, order, axis=1)


def garud_walk(s1, s2):
    """Garud statistics from per-window sorted hashes.

    Returns
    -------
    h1, h12, h123, h2h1, n_distinct : cupy.ndarray, float64, shape (n_windows,)
    """
    n_windows, n_hap = s1.shape
    moments = tuple(cp.empty(n_windows, dtype=cp.float64) for _ in range(5))
    grid = max(1, (n_windows + _THREADS - 1) // _THREADS)
    _garud_walk_kernel((grid,), (_THREADS,),
                       (s1, s2, np.int32(n_hap), np.int32(n_windows), *moments))
    return (*garud_from_moments(*moments[:4]), moments[4])


def n_segments(win_start, win_stop):
    """Segments that ``segmented_hashes`` splits the longest window into."""
    longest = int(cp.max(win_stop - win_start).get()) if win_start.size else 0
    return max(1, (longest + _ROW_SEGMENT - 1) // _ROW_SEGMENT)


def segmented_hashes(hap, w1, w2, win_start, win_stop, n_seg):
    """(n_windows, n_hap) hashes, each window summed over fixed segments.

    Splitting a long window into segments gives the GPU many threads per
    haplotype. Every row of a window uses the same segments and the same
    reduction, so identical rows stay bit-identical; an empty segment
    hashes to zero. ``n_seg`` must cover the longest window (see
    ``n_segments``).
    """
    n_windows = int(win_start.size)
    n_hap = hap.shape[0]
    offs = cp.arange(n_seg, dtype=cp.int64) * _ROW_SEGMENT
    seg_lo = cp.minimum(win_start[:, None] + offs, win_stop[:, None])
    seg_hi = cp.minimum(seg_lo + _ROW_SEGMENT, win_stop[:, None])
    h1, h2 = window_hashes(hap, w1, w2, seg_lo.ravel(), seg_hi.ravel())
    shape = (n_windows, n_seg, n_hap)
    return h1.reshape(shape).sum(axis=1), h2.reshape(shape).sum(axis=1)


def row_hashes(hap):
    """Whole-row hashes for counting the distinct haplotypes of a matrix.

    Returns
    -------
    hash1, hash2 : cupy.ndarray, float64, shape (n_hap,)
    """
    n_var = hap.shape[1]
    start = cp.zeros(1, dtype=cp.int64)
    stop = cp.full(1, n_var, dtype=cp.int64)
    h1, h2 = segmented_hashes(hap, *hash_weights(n_var), start, stop,
                              n_segments(start, stop))
    return h1[0], h2[0]


def garud_h_windows(hap, win_start, win_stop, missing_data='include'):
    """Garud's H1, H12, H123, H2/H1 and distinct-haplotype count per window.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_hap, n_var), signed integer
    win_start, win_stop : array_like of int
        Right-open variant index range of each window.
    missing_data : {'include', 'exclude'}
        'include' - a window with missing calls gets EM haplotype
        frequencies (see ``_haplotype_groups``); a window with no complete
        haplotype gives NaN.
        'exclude' - each window uses only its sites with no missing call.

    Returns
    -------
    h1, h12, h123, h2h1, n_distinct : numpy.ndarray, float64, shape (n_windows,)
        A window with no variants has one distinct haplotype: H1 = H12 =
        H123 = 1 and H2/H1 = 0.

    Windows are processed in batches sized to free GPU memory. Each window's
    result depends only on its own variants, so batching never changes a value.
    """
    from ._haplotype_groups import (
        column_has_missing, window_batch_size, window_moments,
    )

    hap = _as_kernel_input(hap)
    n_hap, n_var = hap.shape
    win_start = cp.asarray(win_start, dtype=cp.int64)
    win_stop = cp.asarray(win_stop, dtype=cp.int64)
    n_windows = int(win_start.shape[0])
    out = np.empty((5, n_windows), dtype=np.float64)
    if n_windows == 0:
        return tuple(out)
    w1, w2 = hash_weights(n_var)

    col_missing = column_has_missing(hap)
    if missing_data == 'exclude':
        # A zero weight drops a site from the hash exactly: fma(x, 0, a)
        # returns a bit for bit, so the hash is the hash over the complete
        # sites alone.
        hw1 = cp.where(col_missing, 0.0, w1)
        hw2 = cp.where(col_missing, 0.0, w2)
        has_missing = cp.zeros(n_windows, dtype=cp.bool_)
    else:
        hw1, hw2 = w1, w2
        n_miss = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                                 cp.cumsum(col_missing, dtype=cp.int64)])
        has_missing = n_miss[win_stop] > n_miss[win_start]
    clean = cp.nonzero(~has_missing)[0]
    missing = cp.nonzero(has_missing)[0]
    clean_idx, missing_idx = clean.get(), missing.get()

    # Windows with no missing call: exact hash, sort and run walk.
    c_start, c_stop = win_start[clean], win_stop[clean]
    batch = _memutil.estimate_garud_window_batch(n_hap)
    for b0 in range(0, clean_idx.size, batch):
        b1 = min(b0 + batch, clean_idx.size)
        h1, h2 = window_hashes(hap, hw1, hw2, c_start[b0:b1], c_stop[b0:b1])
        s1, s2 = sort_window_hashes(h1, h2)
        del h1, h2
        stats = garud_walk(s1, s2)
        del s1, s2
        out[:, clean_idx[b0:b1]] = cp.stack(stats).get()

    # Windows with missing calls: EM frequencies.
    if missing_idx.size:
        m_start, m_stop = win_start[missing], win_stop[missing]
        n_seg = n_segments(m_start, m_stop)
        batch = window_batch_size(n_hap, n_seg)
        for b0 in range(0, missing_idx.size, batch):
            b1 = min(b0 + batch, missing_idx.size)
            sum_f2, top0, top1, top2, n_distinct = window_moments(
                hap, m_start[b0:b1], m_stop[b0:b1], w1, w2, n_seg)
            stats = garud_from_moments(sum_f2, top0, top1, top2)
            out[:, missing_idx[b0:b1]] = cp.stack([*stats, n_distinct]).get()
    return tuple(out)
