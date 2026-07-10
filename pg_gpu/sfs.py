"""
GPU-accelerated site frequency spectrum computation.

This module provides functions for computing unfolded, folded, scaled, and
joint site frequency spectra from haplotype data.
"""

from functools import lru_cache

import numpy as np
import cupy as cp
from typing import Union, Optional
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix
from .streaming_matrix import StreamingHaplotypeMatrix, _stream_sum


def _per_allele_counts(matrix, n_alleles=None):
    """Per-allele counts (n_var, K) and per-site n_valid via the fused kernel.

    K defaults to the site-local maximum allele index + 1; pass ``n_alleles``
    for a fixed/global width (e.g. to align populations for the joint SFS).
    Multiallelic-correct: each allele is counted separately. This is the sole
    counting primitive for every SFS in this module (the old collapsed
    ``dac_and_n`` wrappers were removed once all variants went per-allele).
    """
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    from ._memutil import allele_counts
    return allele_counts(matrix.haplotypes, n_alleles=n_alleles)


# ---------------------------------------------------------------------------
# Public API: Single-population SFS
# ---------------------------------------------------------------------------

def sfs(haplotype_matrix: HaplotypeMatrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include'):
    """Compute the unfolded site frequency spectrum.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'include' - per-site n_valid; bins by actual DAC
        'exclude' - only sites with no missing data

    Returns
    -------
    ndarray, int64, shape (n_chromosomes + 1,)
        Element k = number of variants with k derived alleles.
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_sum(
            haplotype_matrix,
            lambda chunk: sfs(chunk, population=population,
                              missing_data=missing_data),
        )

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    max_n = matrix.num_haplotypes

    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
        if matrix.num_variants == 0:
            return np.zeros(max_n + 1, dtype=np.int64)

    ac, n_valid = _per_allele_counts(matrix)
    # Per-allele polarised SFS: each derived allele contributes one count at its
    # own sample frequency, for 0 < count < n (both fixed classes excluded),
    # matching tskit's polarised allele_frequency_spectrum.
    derived = ac[:, 1:]
    vals = derived[(derived > 0) & (derived < n_valid[:, None])]
    if vals.size == 0:
        return np.zeros(max_n + 1, dtype=np.int64)
    s = cp.bincount(vals, minlength=max_n + 1)[:max_n + 1]
    return s.astype(cp.int64).get()


def sfs_folded(haplotype_matrix: HaplotypeMatrix,
               population: Optional[Union[str, list]] = None,
               missing_data: str = 'include'):
    """Compute the folded site frequency spectrum (minor allele counts).

    Per-allele and conforming to tskit's ``allele_frequency_spectrum(
    polarised=False)``: **every** allele (the reference/ancestral column
    included) contributes weight 1/2 to bin ``min(count, n - count)``, and the
    fixed class (bin 0) is dropped. A k-allelic site therefore contributes k
    half-weight entries, so bin values are half-integers on multiallelic data;
    on biallelic data the reference and derived alleles land in the same
    ``min`` bin, summing to the usual integer count. This cannot be built from
    the unfolded ``sfs()`` (which discards the reference column), but it is a
    single pass over the per-allele counts.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str

    Returns
    -------
    ndarray, float64, shape (n_chromosomes // 2 + 1,)
        Element k = folded weight of alleles with minor allele count k.
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_sum(
            haplotype_matrix,
            lambda chunk: sfs_folded(chunk, population=population,
                                     missing_data=missing_data),
        )

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    max_n = matrix.num_haplotypes
    length = max_n // 2 + 1

    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
        if matrix.num_variants == 0:
            return np.zeros(length, dtype=np.float64)

    ac, n_valid = _per_allele_counts(matrix)
    # Fold every allele (reference column 0 included) to bin min(count, n-count),
    # weight 1/2; drop the fixed class (foldbin == 0, i.e. count in {0, n}).
    foldbin = cp.minimum(ac, n_valid[:, None] - ac)
    mask = (ac > 0) & (foldbin > 0)
    vals = foldbin[mask].astype(cp.int32)
    if vals.size == 0:
        return np.zeros(length, dtype=np.float64)
    s = 0.5 * cp.bincount(vals, minlength=length)[:length]
    return s.astype(cp.float64).get()


def sfs_scaled(haplotype_matrix: HaplotypeMatrix,
               population: Optional[Union[str, list]] = None,
               missing_data: str = 'include'):
    """Compute the scaled unfolded site frequency spectrum.

    Scaling: element k is multiplied by k, yielding a constant expectation
    under neutrality and constant population size.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str

    Returns
    -------
    ndarray, float64, shape (n_chromosomes + 1,)
    """
    s = sfs(haplotype_matrix, population, missing_data=missing_data)
    return scale_sfs(s)


def sfs_folded_scaled(haplotype_matrix: HaplotypeMatrix,
                      population: Optional[Union[str, list]] = None,
                      missing_data: str = 'include'):
    """Compute the scaled folded site frequency spectrum.

    Scaling: element k is multiplied by k * (n - k) / n.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str

    Returns
    -------
    ndarray, float64, shape (n_chromosomes // 2 + 1,)
    """
    # sfs_folded already handles streaming / population subsetting and is pinned
    # to tskit; scaling is a fixed transform on top of it (no tskit analogue).
    if population is not None:
        n = _get_population_matrix(haplotype_matrix, population).num_haplotypes
    else:
        n = haplotype_matrix.num_haplotypes
    s = sfs_folded(haplotype_matrix, population, missing_data=missing_data)
    return scale_sfs_folded(s, n)


# ---------------------------------------------------------------------------
# Public API: Joint SFS (two populations)
# ---------------------------------------------------------------------------

def _joint_global_k(m1, m2):
    """Global allele-index width K spanning both population matrices.

    The joint SFS counts each population on a *shared* K so that allele
    column ``a`` denotes the same allele in both per-allele count matrices
    (the alignment discipline from 0001). ``K = max allele index over both
    pops + 1``.
    """
    k1 = int(m1.haplotypes.max()) if m1.num_variants > 0 else 0
    k2 = int(m2.haplotypes.max()) if m2.num_variants > 0 else 0
    return max(k1, k2, 0) + 1


def _joint_aligned_counts(haplotype_matrix, pop1, pop2):
    """Per-allele counts for both pops on a shared global K.

    Returns ``(ac1, ac2, nv1, nv2, n1, n2)`` where ``ac1``/``ac2`` are
    ``(n_var, K)`` on the same K (so allele column ``a`` is the same allele in
    both), ``nv1``/``nv2`` are per-site valid counts, and ``n1``/``n2`` the
    population sizes.
    """
    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)
    if m1.device == 'CPU':
        m1.transfer_to_gpu()
    if m2.device == 'CPU':
        m2.transfer_to_gpu()
    K = _joint_global_k(m1, m2)
    ac1, nv1 = _per_allele_counts(m1, n_alleles=K)
    ac2, nv2 = _per_allele_counts(m2, n_alleles=K)
    return ac1, ac2, nv1, nv2, m1.num_haplotypes, m2.num_haplotypes


def _joint_per_allele_counts(haplotype_matrix, pop1, pop2, missing_data):
    """Per-allele (count_A, count_B) for every derived allele passing the
    tskit cohort-polymorphic filter.

    Keeps derived columns only (ancestral column 0 excluded) and retains
    alleles with ``0 < count_A + count_B < n_valid_A + n_valid_B``
    (segregating in the A+B cohort). This is exactly tskit's sample-set AFS
    rule: all edge cells populated, only the two global-monomorphic corners
    ``(0,0)`` and ``(nA,nB)`` dropped. Returns ``(cA, cB, n1, n2)`` with
    ``cA``/``cB`` flat GPU int64 arrays.
    """
    ac1, ac2, nv1, nv2, n1, n2 = _joint_aligned_counts(haplotype_matrix,
                                                       pop1, pop2)
    dA = ac1[:, 1:]                       # derived alleles only
    dB = ac2[:, 1:]
    cohort = dA + dB
    n_cohort = (nv1 + nv2)[:, None]
    keep = (cohort > 0) & (cohort < n_cohort)
    if missing_data == 'exclude':
        complete = ((nv1 == n1) & (nv2 == n2))[:, None]
        keep = keep & complete

    cA = dA[keep].astype(cp.int64)
    cB = dB[keep].astype(cp.int64)
    return cA, cB, n1, n2


def _joint_folded_cells(haplotype_matrix, pop1, pop2, missing_data):
    """Folded joint cells ``(fA, fB)`` for tskit's ``polarised=False`` 2D AFS.

    Every allele (ancestral column 0 **included**) is folded as a unit by the
    global minor: cell ``(cA, cB)`` is paired with ``(nA-cA, nB-cB)`` and the
    smaller-total orientation is kept (ties broken by the first axis). Each
    surviving allele carries weight 1/2; global-monomorphic corners are
    dropped. Returns ``(fA, fB, n1, n2)`` (flat GPU int64), to be binned with
    weight 1/2. Unlike the unfolded rule, this folds *the whole site by one
    global minor allele* rather than each axis independently -- which is where
    tskit departs from scikit-allel's per-axis fold.
    """
    ac1, ac2, nv1, nv2, n1, n2 = _joint_aligned_counts(haplotype_matrix,
                                                       pop1, pop2)
    cohort = ac1 + ac2                    # all alleles incl. ancestral
    ncoh = (nv1 + nv2)[:, None]
    keep = (cohort > 0) & (cohort < ncoh)
    if missing_data == 'exclude':
        complete = ((nv1 == n1) & (nv2 == n2))[:, None]
        keep = keep & complete

    nv1b = nv1[:, None]
    nv2b = nv2[:, None]
    comp = ncoh - cohort
    flip = (cohort > comp) | ((cohort == comp) & (ac1 > nv1b - ac1))
    fA = cp.where(flip, nv1b - ac1, ac1)
    fB = cp.where(flip, nv2b - ac2, ac2)
    return fA[keep].astype(cp.int64), fB[keep].astype(cp.int64), n1, n2


def joint_sfs(haplotype_matrix: HaplotypeMatrix,
              pop1: Union[str, list],
              pop2: Union[str, list],
              missing_data: str = 'include'):
    """Compute the joint site frequency spectrum between two populations.

    Per-allele and conforming to tskit's ``allele_frequency_spectrum([popA,
    popB], polarised=True)``: each derived allele contributes to cell
    ``(count_A, count_B)``, retaining alleles segregating in the A+B cohort
    (``0 < count_A + count_B < nA + nB``). All edge cells are populated
    (alleles private to / fixed within one population); only the two
    global-monomorphic corners ``(0,0)`` and ``(nA,nB)`` are excluded.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop1, pop2 : str or list
        Population names or sample indices.
    missing_data : str

    Returns
    -------
    ndarray, int64, shape (n1 + 1, n2 + 1)
        Element [i, j] = number of derived alleles with i copies in pop1
        and j copies in pop2.
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_sum(
            haplotype_matrix,
            lambda chunk: joint_sfs(chunk, pop1=pop1, pop2=pop2,
                                    missing_data=missing_data),
        )

    cA, cB, n1, n2 = _joint_per_allele_counts(haplotype_matrix, pop1, pop2,
                                              missing_data)
    x = n1 + 1
    y = n2 + 1
    if cA.size == 0:
        return np.zeros((x, y), dtype=np.int64)
    flat = cA * y + cB
    s = cp.bincount(flat, minlength=x * y)
    return s[:x * y].reshape(x, y).astype(cp.int64).get()


@lru_cache(maxsize=16)
def _projection_matrix_vec(n_from, n_to):
    """Hypergeometric projection matrix from ``n_from`` to ``n_to``.

    Output shape ``(n_to + 1, n_from + 1)``; element ``[a, i]`` is the
    probability of drawing ``a`` derived alleles in a size-``n_to``
    sample without replacement from a size-``n_from`` population with
    ``i`` derived alleles. Vectorized via ``scipy.special.gammaln`` so
    it scales to ``n_from`` in the 10^5+ range without the per-cell
    big-int comb of the exact ``diversity._projection_matrix``.
    Cached on ``(n_from, n_to)`` so repeated per-chunk calls inside a
    streaming scan reuse one host-side build.
    """
    from scipy.special import gammaln
    if n_to < 0 or n_to > n_from:
        raise ValueError(
            f"need 0 <= n_to <= n_from, got n_to={n_to}, n_from={n_from}")
    if n_to == 0:
        out = np.zeros((1, n_from + 1))
        out[0, :] = 1.0  # all mass at the empty-sample bin
        return out
    k_from = np.arange(n_from + 1, dtype=np.int64)[None, :]
    k_to = np.arange(n_to + 1, dtype=np.int64)[:, None]
    valid = (k_to <= k_from) & ((n_to - k_to) <= (n_from - k_from))
    # Outside the hypergeometric support, k_from - k_to or
    # (n_from-k_from) - (n_to-k_to) is negative; clamp before gammaln
    # and zero out post-exp.
    kt = np.where(valid, k_to, 0)
    kfk = np.where(valid, k_from - k_to, 0)
    ntk = np.where(valid, n_to - k_to, 0)
    nfk = n_from - k_from
    nfk_ntk = np.where(valid, nfk - ntk, 0)
    log_P = (gammaln(k_from + 1) - gammaln(kt + 1) - gammaln(kfk + 1)
             + gammaln(nfk + 1) - gammaln(ntk + 1) - gammaln(nfk_ntk + 1)
             - (gammaln(n_from + 1) - gammaln(n_to + 1)
                - gammaln(n_from - n_to + 1)))
    # Outside the hypergeometric support, the clamped-zero arguments
    # leave a meaningless residual in log_P that can overflow ``exp``;
    # mask first, then exp, so out-of-support cells stay zero.
    log_P = np.where(valid, log_P, -np.inf)
    return np.exp(log_P)


def project_joint_sfs(haplotype_matrix: HaplotypeMatrix,
                       pop1: Union[str, list],
                       pop2: Union[str, list],
                       target_n1: int,
                       target_n2: int,
                       missing_data: str = 'include'):
    """Joint SFS projected to ``(target_n1+1, target_n2+1)`` via
    hypergeometric sampling.

    Mathematically identical to ``P1 @ joint_sfs(...) @ P2.T`` with
    hypergeometric projection matrices ``P1, P2``, but applied
    per-variant so the ``(n1+1, n2+1)`` full histogram is never
    materialized. That intermediate would be 80 GB at 100k haps per
    population; the projected output stays small regardless of source
    size. Use this whenever the source size is too large for
    ``joint_sfs`` to allocate its bincount.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix or StreamingHaplotypeMatrix
    pop1, pop2 : str or list
        Population names or explicit sample-index lists.
    target_n1, target_n2 : int
        Projection targets; each must be <= the corresponding source
        population size.
    missing_data : str

    Returns
    -------
    ndarray, float64, shape ``(target_n1 + 1, target_n2 + 1)``
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        sample_sets = haplotype_matrix.sample_sets or {}
        pop1_list = sample_sets[pop1] if isinstance(pop1, str) else pop1
        pop2_list = sample_sets[pop2] if isinstance(pop2, str) else pop2
        n1, n2 = len(pop1_list), len(pop2_list)
        if target_n1 > n1 or target_n2 > n2:
            raise ValueError(
                f"Cannot project up: target_n1={target_n1} > n1={n1} "
                f"or target_n2={target_n2} > n2={n2}")
        # Build P1, P2 once on host (cheap with gammaln) then push to
        # GPU. Per-chunk work is one gather + one small matmul.
        P1 = cp.asarray(_projection_matrix_vec(n1, target_n1))
        P2 = cp.asarray(_projection_matrix_vec(n2, target_n2))
        acc = cp.zeros((target_n1 + 1, target_n2 + 1), dtype=cp.float64)
        for _, _, chunk in haplotype_matrix.iter_gpu_chunks():
            acc += _project_joint_sfs_chunk_gpu(chunk, pop1, pop2,
                                                 P1, P2, missing_data)
        return acc.get()

    # Eager path: per-allele (count_A, count_B) then gather + matmul. This is
    # exactly P1 @ joint_sfs(...) @ P2.T -- each surviving derived allele
    # contributes P1[:, cA] outer P2[:, cB] -- so the same cohort filter as
    # joint_sfs keeps the sandwich identity.
    cA, cB, n1, n2 = _joint_per_allele_counts(haplotype_matrix, pop1, pop2,
                                              missing_data)
    if target_n1 > n1 or target_n2 > n2:
        raise ValueError(
            f"Cannot project up: target_n1={target_n1} > n1={n1} "
            f"or target_n2={target_n2} > n2={n2}")
    P1 = cp.asarray(_projection_matrix_vec(n1, target_n1))
    P2 = cp.asarray(_projection_matrix_vec(n2, target_n2))
    if cA.size == 0:
        return np.zeros((target_n1 + 1, target_n2 + 1), dtype=np.float64)
    A = P1[:, cA]
    B = P2[:, cB]
    return (A @ B.T).get()


def _project_joint_sfs_chunk_gpu(chunk_hm, pop1, pop2, P1, P2,
                                  missing_data):
    """Per-chunk projected contribution; returns a GPU array.

    Factored out so the streaming dispatch can accumulate on-device
    without round-tripping each chunk's contribution through host
    memory. Per-allele, using the same cohort filter as ``joint_sfs``.
    """
    cA, cB, _, _ = _joint_per_allele_counts(chunk_hm, pop1, pop2, missing_data)
    if cA.size == 0:
        return cp.zeros((P1.shape[0], P2.shape[0]), dtype=cp.float64)
    A = P1[:, cA]
    B = P2[:, cB]
    return A @ B.T


def joint_sfs_folded(haplotype_matrix: HaplotypeMatrix,
                     pop1: Union[str, list],
                     pop2: Union[str, list],
                     missing_data: str = 'include'):
    """Compute the folded joint site frequency spectrum.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop1, pop2 : str or list
    missing_data : str

    Per-allele and conforming to tskit's ``allele_frequency_spectrum([popA,
    popB], polarised=False)``: each allele (ancestral included) is folded as a
    unit by the global minor and contributes weight 1/2 to the kept cell. The
    output is the full ``(n1+1, n2+1)`` array (the kept region is the
    lower-total triangle; the rest is zero), with half-integer weights on
    multiallelic data. **This departs from scikit-allel's per-axis fold even on
    biallelic data** -- we pin to tskit here.

    Returns
    -------
    ndarray, float64, shape (n1 + 1, n2 + 1)
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_sum(
            haplotype_matrix,
            lambda chunk: joint_sfs_folded(chunk, pop1=pop1, pop2=pop2,
                                           missing_data=missing_data),
        )

    fA, fB, n1, n2 = _joint_folded_cells(haplotype_matrix, pop1, pop2,
                                         missing_data)
    x = n1 + 1
    y = n2 + 1
    if fA.size == 0:
        return np.zeros((x, y), dtype=np.float64)
    flat = fA * y + fB
    s = 0.5 * cp.bincount(flat, minlength=x * y)[:x * y]
    return s.reshape(x, y).astype(cp.float64).get()


def joint_sfs_scaled(haplotype_matrix: HaplotypeMatrix,
                     pop1: Union[str, list],
                     pop2: Union[str, list],
                     missing_data: str = 'include'):
    """Compute the scaled joint site frequency spectrum.

    Scaling: element [i, j] is multiplied by i * j.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop1, pop2 : str or list
    missing_data : str

    Returns
    -------
    ndarray, float64, shape (n1 + 1, n2 + 1)
    """
    s = joint_sfs(haplotype_matrix, pop1, pop2, missing_data=missing_data)
    return scale_joint_sfs(s)


def joint_sfs_folded_scaled(haplotype_matrix: HaplotypeMatrix,
                            pop1: Union[str, list],
                            pop2: Union[str, list],
                            missing_data: str = 'include'):
    """Compute the scaled folded joint site frequency spectrum.

    Scaling: element [i, j] is multiplied by i * j * (n1 - i) * (n2 - j).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop1, pop2 : str or list
    missing_data : str

    Returns
    -------
    ndarray, float64, shape (n1 + 1, n2 + 1)
    """
    # joint_sfs_folded handles streaming / subsetting and is pinned to tskit;
    # scaling is a fixed transform on top of it. n1/n2 read off the shape.
    s = joint_sfs_folded(haplotype_matrix, pop1, pop2,
                          missing_data=missing_data)
    n1, n2 = s.shape[0] - 1, s.shape[1] - 1
    return scale_joint_sfs_folded(s, n1, n2)


# ---------------------------------------------------------------------------
# Public API: Scaling and folding utilities
# ---------------------------------------------------------------------------

def scale_sfs(s):
    """Scale a site frequency spectrum by multiplying element k by k."""
    s = np.asarray(s, dtype='f8')
    k = np.arange(s.size)
    return s * k


def scale_sfs_folded(s, n):
    """Scale a folded SFS: element k multiplied by k * (n - k) / n."""
    s = np.asarray(s, dtype='f8')
    k = np.arange(s.shape[0])
    return s * k * (n - k) / n


def scale_joint_sfs(s):
    """Scale a joint SFS: element [i, j] multiplied by i * j."""
    s = np.asarray(s, dtype='f8')
    i = np.arange(s.shape[0])[:, None]
    j = np.arange(s.shape[1])[None, :]
    return (s * i) * j


def scale_joint_sfs_folded(s, n1, n2):
    """Scale a folded joint SFS: element [i,j] * i * j * (n1-i) * (n2-j)."""
    s = np.asarray(s, dtype='f8')
    i = np.arange(s.shape[0])[:, None]
    j = np.arange(s.shape[1])[None, :]
    return s * i * j * (n1 - i) * (n2 - j)


def fold_sfs(s, n):
    """Fold an unfolded SFS.

    Parameters
    ----------
    s : array_like
        Unfolded SFS.
    n : int
        Number of chromosomes.

    Returns
    -------
    ndarray
        Folded SFS.
    """
    s = np.asarray(s)

    # pad to full size if needed
    if s.shape[0] < n + 1:
        sn = np.zeros(n + 1, dtype=s.dtype)
        sn[:s.shape[0]] = s
        s = sn

    nf = (n + 1) // 2
    n_even = nf * 2
    o = s[:nf] + s[nf:n_even][::-1]
    return o


def fold_joint_sfs(s, n1, n2):
    """Fold a joint SFS.

    Parameters
    ----------
    s : array_like, shape (n1 + 1, n2 + 1)
    n1, n2 : int

    Returns
    -------
    ndarray
        Folded joint SFS.
    """
    s = np.asarray(s)

    # pad if needed
    if s.shape[0] < n1 + 1:
        sm = np.zeros((n1 + 1, s.shape[1]), dtype=s.dtype)
        sm[:s.shape[0]] = s
        s = sm
    if s.shape[1] < n2 + 1:
        sn = np.zeros((s.shape[0], n2 + 1), dtype=s.dtype)
        sn[:, :s.shape[1]] = s
        s = sn

    mf = (n1 + 1) // 2
    nf = (n2 + 1) // 2
    m_even = mf * 2
    n_even = nf * 2

    o = (s[:mf, :nf] +
         s[mf:m_even, :nf][::-1] +
         s[:mf, nf:n_even][:, ::-1] +
         s[mf:m_even, nf:n_even][::-1, ::-1])
    return o
