"""Haplotype frequencies from windows that contain missing calls.

A missing call hides part of a haplotype, so the haplotype's identity is
uncertain. The rule here estimates haplotype frequencies by maximum
likelihood, with the assumption that calls are missing completely at random:

1. A complete haplotype has no missing call in the window. The complete
   haplotypes are grouped by exact identity. Each group is one distinct
   haplotype, and the groups are ordered by content (their hash), never by
   row position.
2. An incomplete haplotype is compatible with a group when it has no
   mismatch against the group at its called sites. A haplotype with no
   calls is compatible with every group.
3. Expectation-maximization (EM) splits each compatible incomplete
   haplotype across its compatible groups in proportion to the current
   group frequencies, and then updates the frequencies from the split
   counts. The start is the frequencies of the complete haplotypes.
4. An incomplete haplotype with no compatible group is a distinct
   haplotype of its own (a singleton). Two incomplete haplotypes never
   join each other.
5. A window with no complete haplotype has no frequencies. The callers
   return NaN for it.

The log-likelihood, sum_g n_g log p_g + sum_h log(sum_{g compatible} p_g),
is strictly concave because every group holds at least one complete
haplotype (n_g >= 1). Thus EM has one fixed point, and the result does not
depend on row order. A haplotype with no calls does not change that fixed
point, because EM gives it to each group in proportion to p_g.

The frequency estimate has a bias toward more diversity only from the
singletons in step 4. When most haplotypes in a window are incomplete, the
complete haplotypes are few and the estimate is weak. With independent
missing calls at rate m, a haplotype is complete over L sites with
probability (1 - m)^L, so long windows need a low missing rate.

Every step runs for a batch of windows at once: one kernel counts missing
calls, one sort groups the complete rows of every window, two kernels find
the compatible groups, and one kernel runs EM with one thread block per
window, iterating to convergence on the device with no host round trip.
"""

from types import SimpleNamespace

import cupy as cp
import cupyx
import numpy as np

from . import _memutil
from ._haplotype_hash import (
    _ROW_SEGMENT, _as_kernel_input, hash_weights, n_segments, segmented_hashes,
    sort_order,
)

_THREADS = _memutil._THREADS_PER_BLOCK

# EM stops in a window when no group frequency changes by more than this
# amount in one step, or after _EM_MAX_ITER steps.
_EM_TOL = 1e-12
_EM_MAX_ITER = 100000

# The EM kernel reduces the per-step change across its block with a tree,
# which needs a power-of-two block.
_EM_THREADS = 256

# Two EM frequencies closer than this relative amount are a tie when a
# haplotype needs one group label. A different row order sums the E-step
# terms in a different order, which can change the last bits.
_LABEL_TIE_RTOL = 1e-9


# The matrix is read through a signed char pointer with byte strides, as in
# the hash kernel, so any signed integer dtype in C or F order is read in
# place. All offsets are 64-bit.
_module = cp.RawModule(code=r'''
#define EM_THREADS @EM_THREADS@

__device__ __forceinline__ bool compatible(const signed char* a,
                                           const signed char* r,
                                           long long stride1,
                                           long long lo, long long hi) {
    for (long long v = lo; v < hi; v++) {
        signed char x = a[v * stride1];
        if (x >= 0 && x != r[v * stride1]) return false;
    }
    return true;
}

// One thread per variant; adjacent threads read adjacent bytes of a C-order
// row, so the reads coalesce.
extern "C" __global__
void col_missing(const signed char* hap, long long stride0, long long stride1,
                 int n_hap, long long n_var, unsigned char* out) {
    long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_var) return;
    unsigned char m = 0;
    for (int h = 0; h < n_hap; h++) {
        if (hap[(long long)h * stride0 + v * stride1] < 0) {
            m = 1;
            break;
        }
    }
    out[v] = m;
}

// One thread per (window, row, segment); counts add into out[window, row].
extern "C" __global__
void window_row_missing(const signed char* hap, long long stride0,
                        long long stride1, const long long* starts,
                        const long long* stops, int n_hap, int n_windows,
                        long long max_seg, long long seg, int* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_w = (long long)n_hap * max_seg;
    if (t >= per_w * n_windows) return;
    int w = (int)(t / per_w);
    long long r = t % per_w;
    int h = (int)(r / max_seg);
    long long v0 = starts[w] + (r % max_seg) * seg;
    long long hi = stops[w];
    if (v0 >= hi) return;
    long long v1 = v0 + seg < hi ? v0 + seg : hi;
    const signed char* row = hap + (long long)h * stride0;
    int c = 0;
    for (long long v = v0; v < v1; v++) {
        if (row[v * stride1] < 0) c++;
    }
    if (c) atomicAdd(out + (long long)w * n_hap + h, c);
}

// One thread per incomplete row: the number of compatible groups in its
// window and the first of them.
extern "C" __global__
void compat_count(const signed char* hap, long long stride0, long long stride1,
                  const long long* starts, const long long* stops,
                  const long long* inc_w, const long long* inc_row, long long n_inc,
                  const long long* goff, const long long* reps,
                  int* n_compat, long long* first) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_inc) return;
    long long w = inc_w[k];
    const signed char* a = hap + inc_row[k] * stride0;
    int c = 0;
    long long f = -1;
    for (long long g = goff[w]; g < goff[w + 1]; g++) {
        if (compatible(a, hap + reps[g] * stride0, stride1, starts[w], stops[w])) {
            if (c == 0) f = g;
            c++;
        }
    }
    n_compat[k] = c;
    first[k] = f;
}

// One thread per row with two or more compatible groups: their ids, in
// group order, at out[off[a] .. off[a + 1]).
extern "C" __global__
void compat_fill(const signed char* hap, long long stride0, long long stride1,
                 const long long* starts, const long long* stops,
                 const long long* inc_w, const long long* inc_row,
                 const long long* amb, long long n_amb,
                 const long long* goff, const long long* reps,
                 const long long* off, long long* out) {
    long long a = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_amb) return;
    long long k = amb[a];
    long long w = inc_w[k];
    const signed char* x = hap + inc_row[k] * stride0;
    long long j = off[a];
    for (long long g = goff[w]; g < goff[w + 1]; g++) {
        if (compatible(x, hap + reps[g] * stride0, stride1, starts[w], stops[w])) {
            out[j++] = g;
        }
    }
}

// One block per window runs EM to convergence. fixed[g] holds the complete
// rows of group g plus the rows compatible with g alone. The ambiguous rows
// of the window are amb_off[w] .. amb_off[w + 1], each with its compatible
// groups at csr_g[csr_off[a] .. csr_off[a + 1]). The transpose lists, for
// each group g, the ambiguous rows compatible with it, in row order, at
// t_row[t_off[g] .. t_off[g + 1]). Each step first finds every ambiguous
// row's inverse total 1 / sum p, then each group gathers its share in a
// fixed order: no atomics, so a run repeats bit for bit.
extern "C" __global__
void em_windows(const long long* goff, const long long* amb_off,
                const long long* csr_off, const long long* csr_g,
                const long long* t_off, const long long* t_row,
                const double* n_g, const double* fixed,
                const double* n_complete, const double* total,
                double tol, int max_iter,
                double* p, double* inv_den, double* counts) {
    __shared__ double red[EM_THREADS];
    int w = blockIdx.x;
    int tid = threadIdx.x;
    long long g0 = goff[w], g1 = goff[w + 1];
    long long a0 = amb_off[w], a1 = amb_off[w + 1];
    double T = total[w];
    if (a0 == a1) {
        for (long long g = g0 + tid; g < g1; g += EM_THREADS) counts[g] = fixed[g];
        return;
    }
    for (long long g = g0 + tid; g < g1; g += EM_THREADS) {
        p[g] = n_g[g] / n_complete[w];
    }
    __syncthreads();
    for (int it = 0; it < max_iter; it++) {
        for (long long a = a0 + tid; a < a1; a += EM_THREADS) {
            double den = 0.0;
            for (long long j = csr_off[a]; j < csr_off[a + 1]; j++) den += p[csr_g[j]];
            inv_den[a] = 1.0 / den;
        }
        __syncthreads();
        double change = 0.0;
        for (long long g = g0 + tid; g < g1; g += EM_THREADS) {
            double share = 0.0;
            for (long long t = t_off[g]; t < t_off[g + 1]; t++) share += inv_den[t_row[t]];
            double q = (fixed[g] + p[g] * share) / T;
            change = fmax(change, fabs(q - p[g]));
            p[g] = q;
        }
        red[tid] = change;
        __syncthreads();
        for (int s = EM_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] = fmax(red[tid], red[tid + s]);
            __syncthreads();
        }
        double m = red[0];
        __syncthreads();
        if (m < tol) break;
    }
    for (long long g = g0 + tid; g < g1; g += EM_THREADS) counts[g] = p[g] * T;
}

// One thread per window: sum of squared frequencies, the three largest
// frequencies and the distinct count, from the group counts and singletons.
extern "C" __global__
void window_moments(const long long* goff, const double* counts,
                    const long long* n_single, const long long* n_complete,
                    int n_windows, double n,
                    double* sum_f2, double* top0, double* top1, double* top2,
                    double* n_distinct) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_windows) return;
    if (n_complete[w] == 0) {
        double nan = __longlong_as_double(0x7ff8000000000000LL);
        sum_f2[w] = nan; top0[w] = nan; top1[w] = nan; top2[w] = nan;
        n_distinct[w] = nan;
        return;
    }
    double s = 0.0, t0 = 0.0, t1 = 0.0, t2 = 0.0;
    long long ns = n_single[w];
    // Groups first, then up to three singletons (count 1 each): more
    // singletons cannot enter the top three.
    long long n_top = goff[w + 1] - goff[w] + (ns < 3 ? ns : 3);
    for (long long i = 0; i < n_top; i++) {
        long long g = goff[w] + i;
        double c = g < goff[w + 1] ? counts[g] : 1.0;
        if (c > t0) { t2 = t1; t1 = t0; t0 = c; }
        else if (c > t1) { t2 = t1; t1 = c; }
        else if (c > t2) { t2 = c; }
    }
    for (long long g = goff[w]; g < goff[w + 1]; g++) s += counts[g] * counts[g];
    s += (double)ns;
    sum_f2[w] = s / (n * n);
    top0[w] = t0 / n; top1[w] = t1 / n; top2[w] = t2 / n;
    n_distinct[w] = (double)(goff[w + 1] - goff[w] + ns);
}
'''.replace('@EM_THREADS@', str(_EM_THREADS)))

_col_missing = _module.get_function('col_missing')
_window_row_missing = _module.get_function('window_row_missing')
_compat_count = _module.get_function('compat_count')
_compat_fill = _module.get_function('compat_fill')
_em_windows = _module.get_function('em_windows')
_window_moments = _module.get_function('window_moments')


def _grid(n):
    return (max(1, (n + _THREADS - 1) // _THREADS),)


def column_has_missing(hap):
    """Boolean per variant: True where any row has a missing call."""
    hap = _as_kernel_input(hap)
    n_hap, n_var = hap.shape
    out = cp.zeros(n_var, dtype=cp.uint8)
    if n_var and n_hap:
        stride0, stride1 = hap.strides
        _col_missing(_grid(n_var), (_THREADS,),
                     (hap, np.int64(stride0), np.int64(stride1), np.int32(n_hap),
                      np.int64(n_var), out))
    return out.view(cp.bool_)


def complete_sites(x):
    """The columns of ``x`` with no missing call, as a new array."""
    return x[:, cp.nonzero(~column_has_missing(x))[0]]


def _group_windows(hap, starts, stops, w1, w2, n_seg):
    """EM haplotype groups for a batch of windows (see the module docstring).

    Group ids are global across the batch: window w holds groups
    ``goff[w] .. goff[w + 1]``, in content order. ``n_seg`` covers the
    longest window (see ``n_segments``).
    """
    n_hap = hap.shape[0]
    n_windows = int(starts.size)
    stride0, stride1 = hap.strides

    n_miss = cp.zeros((n_windows, n_hap), dtype=cp.int32)
    total_threads = n_windows * n_hap * n_seg
    if total_threads:
        _window_row_missing(
            _grid(total_threads), (_THREADS,),
            (hap, np.int64(stride0), np.int64(stride1), starts, stops,
             np.int32(n_hap), np.int32(n_windows), np.int64(n_seg),
             np.int64(_ROW_SEGMENT), n_miss))
    complete = n_miss == 0
    del n_miss

    # Group each window's complete rows by exact hash; incomplete rows get
    # infinite keys and sort last.
    h1, h2 = segmented_hashes(hap, w1, w2, starts, stops, n_seg)
    k1 = cp.where(complete, h1, cp.inf)
    k2 = cp.where(complete, h2, cp.inf)
    del h1, h2
    order = sort_order(k1, k2)
    s1 = cp.take_along_axis(k1, order, axis=1)
    s2 = cp.take_along_axis(k2, order, axis=1)
    del k1, k2
    sorted_complete = cp.take_along_axis(complete, order, axis=1)
    start = sorted_complete.copy()
    start[:, 1:] &= (s1[:, 1:] != s1[:, :-1]) | (s2[:, 1:] != s2[:, :-1])
    del s1, s2

    goff = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(start.sum(axis=1))])
    n_total_groups = int(goff[-1].get())
    local_gid = cp.cumsum(start, axis=1) - 1
    gid = local_gid + goff[:-1, None]
    n_g = cp.bincount(gid[sorted_complete], minlength=n_total_groups).astype(cp.float64)
    fw, fp = cp.nonzero(start)
    reps = order[fw, fp]
    n_complete = complete.sum(axis=1)

    # nonzero on a 2-D array returns strided views; the kernels need
    # contiguous indices.
    inc_w, inc_row = (cp.ascontiguousarray(x) for x in cp.nonzero(~complete))
    n_inc = int(inc_w.size)
    r = SimpleNamespace(
        goff=goff, n_complete=n_complete, order=order,
        sorted_complete=sorted_complete, local_gid=local_gid, inc_w=inc_w,
        inc_row=inc_row)
    if n_inc == 0:
        r.counts = n_g
        r.n_single = cp.zeros(n_windows, dtype=cp.int64)
        r.n_compat = cp.zeros(0, dtype=cp.int32)
        return r

    n_compat = cp.empty(n_inc, dtype=cp.int32)
    first = cp.empty(n_inc, dtype=cp.int64)
    _compat_count(_grid(n_inc), (_THREADS,),
                  (hap, np.int64(stride0), np.int64(stride1), starts, stops,
                   inc_w, inc_row, np.int64(n_inc), goff, reps, n_compat, first))
    one = n_compat == 1
    fixed = n_g + cp.bincount(first[one], minlength=n_total_groups)
    amb = cp.nonzero(n_compat >= 2)[0]
    n_amb = int(amb.size)
    lengths = n_compat[amb].astype(cp.int64)
    csr_off = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(lengths)])
    csr_g = cp.empty(int(csr_off[-1].get()), dtype=cp.int64)
    if n_amb:
        _compat_fill(_grid(n_amb), (_THREADS,),
                     (hap, np.int64(stride0), np.int64(stride1), starts, stops,
                      inc_w, inc_row, amb, np.int64(n_amb), goff, reps,
                      csr_off, csr_g))
    # Transpose: for each group, its ambiguous rows in row order. A stable
    # sort by group keeps the pairs of one group in row order.
    t_order = cp.argsort(csr_g)
    t_row = cp.repeat(cp.arange(n_amb, dtype=cp.int64), lengths)[t_order]
    t_off = cp.searchsorted(csr_g[t_order], cp.arange(n_total_groups + 1))
    # inc_w is sorted, so each window's ambiguous rows are one run.
    amb_off = cp.searchsorted(inc_w[amb], cp.arange(n_windows + 1))

    anchored = cp.bincount(inc_w[n_compat > 0], minlength=n_windows)
    total = (n_complete + anchored).astype(cp.float64)
    p = cp.empty(n_total_groups, dtype=cp.float64)
    counts = cp.empty(n_total_groups, dtype=cp.float64)
    _em_windows((n_windows,), (_EM_THREADS,),
                (goff, amb_off, csr_off, csr_g, t_off, t_row, n_g, fixed,
                 n_complete.astype(cp.float64), total,
                 np.float64(_EM_TOL), np.int32(_EM_MAX_ITER),
                 p, cp.empty(max(n_amb, 1), dtype=cp.float64), counts))

    r.counts = counts
    r.p = p
    r.n_single = cp.bincount(inc_w[n_compat == 0], minlength=n_windows)
    r.n_compat, r.first, r.amb, r.csr_off, r.csr_g = n_compat, first, amb, csr_off, csr_g
    return r


def window_moments(hap, starts, stops, w1, w2, n_seg):
    """Garud moments of a batch of windows with EM frequencies.

    Returns
    -------
    sum_f2, top0, top1, top2, n_distinct : cupy.ndarray, float64, (n_windows,)
        NaN for a window with no complete haplotype.
    """
    hap = _as_kernel_input(hap)
    n_windows = int(starts.size)
    r = _group_windows(hap, starts, stops, w1, w2, n_seg)
    out = tuple(cp.empty(n_windows, dtype=cp.float64) for _ in range(5))
    if n_windows:
        _window_moments(_grid(n_windows), (_THREADS,),
                        (r.goff, r.counts, r.n_single, r.n_complete,
                         np.int32(n_windows), np.float64(hap.shape[0]), *out))
    return out


def window_batch_size(n_hap, n_seg):
    """Windows per ``window_moments`` batch, sized to free GPU memory."""
    return max(1, _memutil.estimate_garud_window_batch(n_hap, memory_fraction=0.15)
               // n_seg)


def _row_labels(r, n_hap):
    """One label per (window, row) from a ``_group_windows`` result.

    In each window: the group of a complete row; the most probable group of
    a compatible incomplete row (ties go to the first group in content
    order); a unique label of its own for a singleton. The labels of a
    window run from 0 to its distinct count - 1 with no gap.
    """
    n_windows = r.order.shape[0]
    labels = cp.empty((n_windows, n_hap), dtype=cp.int64)
    cw, cj = cp.nonzero(r.sorted_complete)
    labels[cw, r.order[cw, cj]] = r.local_gid[cw, cj]
    if r.n_compat.size == 0:
        return labels
    goff = r.goff
    one = cp.nonzero(r.n_compat == 1)[0]
    labels[r.inc_w[one], r.inc_row[one]] = r.first[one] - goff[r.inc_w[one]]
    if r.amb.size:
        # Most probable group per ambiguous row: the first group, in content
        # order, whose frequency ties the row's largest one.
        seg = cp.repeat(cp.arange(r.amb.size), cp.diff(r.csr_off))
        score = r.p[r.csr_g]
        best = cp.full(r.amb.size, -1.0)
        cupyx.scatter_max(best, seg, score)
        tie = score >= best[seg] * (1.0 - _LABEL_TIE_RTOL)
        pick = cp.full(r.amb.size, goff[-1], dtype=cp.int64)
        cupyx.scatter_min(pick, seg[tie], r.csr_g[tie])
        amb_w = r.inc_w[r.amb]
        labels[amb_w, r.inc_row[r.amb]] = pick - goff[amb_w]
    # Singletons follow the groups of their window, in row order. The
    # incomplete rows are sorted by window, so a singleton's rank in its
    # window is its position minus the position of the window's first one.
    single = cp.nonzero(r.n_compat == 0)[0]
    sw = r.inc_w[single]
    rank = cp.arange(single.size) - cp.searchsorted(sw, sw, side='left')
    labels[sw, r.inc_row[single]] = goff[sw + 1] - goff[sw] + rank
    return labels


def window_labels(hap, starts, stops, w1, w2):
    """Row labels for a batch of windows (see ``_row_labels``).

    Returns
    -------
    labels : cupy.ndarray, int64, shape (n_windows, n_hap)
    n_distinct : numpy.ndarray, int64, shape (n_windows,)
    has_complete : numpy.ndarray, bool, shape (n_windows,)
        False for a window with no complete haplotype, whose labels are
        undefined.
    """
    hap = _as_kernel_input(hap)
    r = _group_windows(hap, starts, stops, w1, w2, n_segments(starts, stops))
    n_distinct = cp.diff(r.goff) + r.n_single
    n_distinct, n_complete = cp.stack([n_distinct, r.n_complete]).get()
    return _row_labels(r, hap.shape[0]), n_distinct, n_complete > 0


def haplotype_groups(hap, lo=0, hi=None, w1=None, w2=None):
    """Distinct-haplotype counts over ``hap[:, lo:hi]`` with missing calls.

    See the module docstring for the rule.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_hap, n_var), signed integer
        Allele indices, -1 (or any negative value) for missing. Read in
        place; the range is not copied.
    lo, hi : int
        Right-open variant range. ``hi`` defaults to ``n_var``.
    w1, w2 : cupy.ndarray, optional
        Hash weights for ``n_var`` variants (see ``hash_weights``).

    Returns
    -------
    None when the range holds no complete haplotype, else a tuple
    ``(counts, n_distinct)``:

    counts : cupy.ndarray, float64
        Expected count of each distinct haplotype (groups, then singletons).
        The counts sum to ``n_hap``.
    n_distinct : int
        Groups plus singletons.
    """
    hap = _as_kernel_input(hap)
    n_hap, n_var = hap.shape
    if hi is None:
        hi = n_var
    if w1 is None:
        w1, w2 = hash_weights(n_var)
    starts = cp.array([lo], dtype=cp.int64)
    stops = cp.array([hi], dtype=cp.int64)
    r = _group_windows(hap, starts, stops, w1, w2, n_segments(starts, stops))
    n_complete, n_groups, n_single = (
        int(x) for x in cp.stack([r.n_complete[0], r.goff[1], r.n_single[0]]).get())
    if n_complete == 0:
        return None
    counts = cp.concatenate([r.counts, cp.ones(n_single, dtype=cp.float64)])
    return counts, n_groups + n_single


def frequencies(hap, lo=0, hi=None, w1=None, w2=None):
    """Distinct-haplotype frequencies over ``hap[:, lo:hi]``, sorted descending.

    Returns an empty cupy array when the range holds no complete haplotype.
    """
    res = haplotype_groups(hap, lo, hi, w1, w2)
    if res is None:
        return cp.empty(0, dtype=cp.float64)
    counts, _ = res
    return cp.sort(counts)[::-1] / hap.shape[0]
