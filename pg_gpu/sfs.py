"""
GPU-accelerated site frequency spectrum computation.

This module provides functions for computing unfolded, folded, scaled, and
joint site frequency spectra from haplotype data.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix


def _derived_allele_counts(haplotype_matrix, missing_data='include'):
    """Compute derived allele counts per variant on GPU.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    missing_data : str
        'include' or 'pairwise' - return per-site n_valid
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

    if missing_data in ('include', 'pairwise'):
        valid_mask = hap >= 0
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.int64)
        dac = cp.sum(cp.where(valid_mask, hap, 0), axis=0).astype(cp.int64)
        return dac, n_valid
    elif missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap_clean = cp.where(hap >= 0, hap, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.int64)
        # mask incomplete sites by setting dac to -1 (will be filtered)
        dac[~complete] = -1
        n = hap.shape[0]
        return dac, n
    else:
        n = hap.shape[0]
        hap_clean = cp.where(hap >= 0, hap, 0)
        dac = cp.sum(hap_clean, axis=0).astype(cp.int64)
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
        'include' or 'pairwise' - per-site n_valid; bins by actual DAC
        'exclude' - only sites with no missing data

    Returns
    -------
    ndarray, int64, shape (n_chromosomes + 1,)
        Element k = number of variants with k derived alleles.
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

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
        max_n = n

    s = cp.bincount(dac, minlength=max_n + 1)
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
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    ac, n = _allele_counts(matrix, missing_data)

    if missing_data == 'exclude':
        valid = ac[:, 1] >= 0  # dac >= 0
        ac = ac[valid]

    mac = cp.amin(ac, axis=1)

    if isinstance(n, cp.ndarray):
        max_n = int(cp.max(n).get()) if n.size > 0 else 0
    else:
        max_n = n

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
    md = 'include' if missing_data == 'pairwise' else missing_data

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    dac1, n1 = _derived_allele_counts(m1, md)
    dac2, n2 = _derived_allele_counts(m2, md)

    if md == 'exclude':
        valid = (dac1 >= 0) & (dac2 >= 0)
        dac1 = dac1[valid]
        dac2 = dac2[valid]

    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0

    x = n1 + 1
    y = n2 + 1
    tmp = (dac1 * y + dac2).astype(cp.int64)
    s = cp.bincount(tmp, minlength=x * y)
    return s[:x * y].reshape(x, y).get()


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
    md = 'include' if missing_data == 'pairwise' else missing_data

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    ac1, n1 = _allele_counts(m1, md)
    ac2, n2 = _allele_counts(m2, md)

    if md == 'exclude':
        valid = (ac1[:, 1] >= 0) & (ac2[:, 1] >= 0)
        ac1 = ac1[valid]
        ac2 = ac2[valid]

    mac1 = cp.amin(ac1, axis=1)
    mac2 = cp.amin(ac2, axis=1)

    if isinstance(n1, cp.ndarray):
        n1 = int(cp.max(n1).get()) if n1.size > 0 else 0
    if isinstance(n2, cp.ndarray):
        n2 = int(cp.max(n2).get()) if n2.size > 0 else 0

    x = n1 // 2 + 1
    y = n2 // 2 + 1
    tmp = (mac1 * y + mac2).astype(cp.int64)
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
    md = 'include' if missing_data == 'pairwise' else missing_data

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    _, n1 = _derived_allele_counts(m1, md)
    _, n2 = _derived_allele_counts(m2, md)

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
