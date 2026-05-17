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


def _derived_allele_counts(haplotype_matrix, missing_data='include'):
    """Compute derived allele counts per variant on GPU.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    missing_data : str
        'include' - return per-site n_valid
        'exclude' - filter to complete sites

    Returns
    -------
    dac : cupy.ndarray, int64, shape (n_variants,)
        Derived allele counts.
    n : int or cupy.ndarray
        Total haplotypes (int) or per-site valid counts (array).
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes  # (n_haplotypes, n_variants)

    if missing_data == 'include':
        from ._memutil import dac_and_n
        dac, n_valid = dac_and_n(hap)
        return dac, n_valid
    elif missing_data == 'exclude':
        from ._memutil import dac_and_n
        dac, n_valid = dac_and_n(hap)
        incomplete = n_valid < hap.shape[0]
        dac[incomplete] = -1
        n = hap.shape[0]
        return dac, n
    else:
        from ._memutil import chunked_sum_int32
        n = hap.shape[0]
        dac = chunked_sum_int32(cp.maximum(hap, 0))
        return dac, n


def _allele_counts(haplotype_matrix, missing_data='include'):
    """Compute biallelic allele counts [ref, alt] per variant.

    Returns
    -------
    ac : cupy.ndarray, int64, shape (n_variants, 2)
    n : int or cupy.ndarray
    """
    dac, n = _derived_allele_counts(haplotype_matrix, missing_data)
    if isinstance(n, cp.ndarray):
        ref_counts = n - dac
    else:
        ref_counts = n - dac
    ac = cp.stack([ref_counts, dac], axis=1)
    return ac, n


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

    dac, n = _derived_allele_counts(matrix, missing_data)

    if missing_data == 'exclude':
        # filter out incomplete sites (marked as -1)
        valid = dac >= 0
        dac = dac[valid]

    if isinstance(n, cp.ndarray):
        max_n = int(cp.max(n).get()) if n.size > 0 else 0
    else:
        max_n = int(n)

    s = cp.bincount(dac.astype(cp.int32), minlength=max_n + 1)
    return s[:max_n + 1].get()


def sfs_folded(haplotype_matrix: HaplotypeMatrix,
               population: Optional[Union[str, list]] = None,
               missing_data: str = 'include'):
    """Compute the folded site frequency spectrum (minor allele counts).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str

    Returns
    -------
    ndarray, int64, shape (n_chromosomes // 2 + 1,)
        Element k = number of variants with minor allele count k.
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    ac, n = _allele_counts(matrix, missing_data)

    if missing_data == 'exclude':
        valid = ac[:, 1] >= 0  # dac >= 0
        ac = ac[valid]

    mac = cp.amin(ac, axis=1).astype(cp.int32)

    if isinstance(n, cp.ndarray):
        max_n = int(cp.max(n).get()) if n.size > 0 else 0
    else:
        max_n = int(n)

    x = max_n // 2 + 1
    s = cp.bincount(mac, minlength=x)[:x]
    return s.get()


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
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    _, n = _derived_allele_counts(matrix, missing_data)
    if isinstance(n, cp.ndarray):
        n = int(cp.max(n).get()) if n.size > 0 else 0
    s = sfs_folded(haplotype_matrix, population, missing_data=missing_data)
    return scale_sfs_folded(s, n)


# ---------------------------------------------------------------------------
# Public API: Joint SFS (two populations)
# ---------------------------------------------------------------------------

def joint_sfs(haplotype_matrix: HaplotypeMatrix,
              pop1: Union[str, list],
              pop2: Union[str, list],
              missing_data: str = 'include'):
    """Compute the joint site frequency spectrum between two populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop1, pop2 : str or list
        Population names or sample indices.
    missing_data : str

    Returns
    -------
    ndarray, int64, shape (n1 + 1, n2 + 1)
        Element [i, j] = number of variants with i derived alleles in pop1
        and j derived alleles in pop2.
    """
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_sum(
            haplotype_matrix,
            lambda chunk: joint_sfs(chunk, pop1=pop1, pop2=pop2,
                                    missing_data=missing_data),
        )

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    dac1, n1 = _derived_allele_counts(m1, missing_data)
    dac2, n2 = _derived_allele_counts(m2, missing_data)

    if missing_data == 'exclude':
        valid = (dac1 >= 0) & (dac2 >= 0)
        dac1 = dac1[valid]
        dac2 = dac2[valid]

    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0

    x = n1 + 1
    y = n2 + 1
    tmp = (dac1 * y + dac2).astype(cp.int32)
    s = cp.bincount(tmp, minlength=x * y)
    return s[:x * y].reshape(x, y).get()


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

    # Eager path: compute (dac1, dac2) once then apply the same gather.
    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)
    dac1, n1 = _derived_allele_counts(m1, missing_data)
    dac2, n2 = _derived_allele_counts(m2, missing_data)
    if missing_data == 'exclude':
        valid = (dac1 >= 0) & (dac2 >= 0)
        dac1 = dac1[valid]
        dac2 = dac2[valid]
    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0
    if target_n1 > n1 or target_n2 > n2:
        raise ValueError(
            f"Cannot project up: target_n1={target_n1} > n1={n1} "
            f"or target_n2={target_n2} > n2={n2}")
    P1 = cp.asarray(_projection_matrix_vec(n1, target_n1))
    P2 = cp.asarray(_projection_matrix_vec(n2, target_n2))
    A = P1[:, dac1.astype(cp.int64)]
    B = P2[:, dac2.astype(cp.int64)]
    return (A @ B.T).get()


def _project_joint_sfs_chunk_gpu(chunk_hm, pop1, pop2, P1, P2,
                                  missing_data):
    """Per-chunk projected contribution; returns a GPU array.

    Factored out so the streaming dispatch can accumulate on-device
    without round-tripping each chunk's contribution through host
    memory.
    """
    m1 = _get_population_matrix(chunk_hm, pop1)
    m2 = _get_population_matrix(chunk_hm, pop2)
    dac1, _ = _derived_allele_counts(m1, missing_data)
    dac2, _ = _derived_allele_counts(m2, missing_data)
    if missing_data == 'exclude':
        valid = (dac1 >= 0) & (dac2 >= 0)
        dac1 = dac1[valid]
        dac2 = dac2[valid]
    A = P1[:, dac1.astype(cp.int64)]
    B = P2[:, dac2.astype(cp.int64)]
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

    Returns
    -------
    ndarray, int64, shape (n1 // 2 + 1, n2 // 2 + 1)
    """

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    ac1, n1 = _allele_counts(m1, missing_data)
    ac2, n2 = _allele_counts(m2, missing_data)

    if missing_data == 'exclude':
        valid = (ac1[:, 1] >= 0) & (ac2[:, 1] >= 0)
        ac1 = ac1[valid]
        ac2 = ac2[valid]

    mac1 = cp.amin(ac1, axis=1).astype(cp.int32)
    mac2 = cp.amin(ac2, axis=1).astype(cp.int32)

    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0

    x = n1 // 2 + 1
    y = n2 // 2 + 1
    tmp = (mac1 * y + mac2).astype(cp.int32)
    s = cp.bincount(tmp, minlength=x * y)
    return s[:x * y].reshape(x, y).get()


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
    ndarray, float64, shape (n1 // 2 + 1, n2 // 2 + 1)
    """

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    _, n1 = _derived_allele_counts(m1, missing_data)
    _, n2 = _derived_allele_counts(m2, missing_data)

    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0

    s = joint_sfs_folded(haplotype_matrix, pop1, pop2,
                          missing_data=missing_data)
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
