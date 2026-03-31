"""
GPU-accelerated pairwise distance distribution statistics.

Computes pairwise Hamming distances and their distributional moments
(variance, skewness, kurtosis) for haploid and diploid data.
All computation stays on GPU until final scalar results.
"""

import numpy as np
import cupy as cp
from .haplotype_matrix import HaplotypeMatrix
from .genotype_matrix import GenotypeMatrix
from ._utils import get_population_matrix


def pairwise_diffs_haploid(haplotype_matrix, population=None):
    """Compute pairwise Hamming distances between haplotypes on GPU.

    Uses matrix multiply for O(n^2 * m) computation entirely on GPU.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional

    Returns
    -------
    diffs : cupy.ndarray, float64, condensed form (n_pairs,)
        Number of differing sites per pair.
    """
    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    X = cp.maximum(matrix.haplotypes, 0).astype(cp.float64)
    n_var = cp.float64(X.shape[1])

    # matches_ij = X @ X.T + (1-X) @ (1-X).T
    Xc = 1.0 - X
    matches = (X @ X.T) + (Xc @ Xc.T)
    diffs_mat = n_var - matches

    n = X.shape[0]
    idx_i, idx_j = cp.triu_indices(n, k=1)
    return diffs_mat[idx_i, idx_j]


def pairwise_diffs_diploid(genotype_matrix, population=None):
    """Compute pairwise genotype differences between diploid individuals on GPU.

    Uses indicator matrix approach for 0/1/2 genotype values.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    population : str or list, optional

    Returns
    -------
    diffs : cupy.ndarray, float64, condensed form (n_pairs,)
    """
    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    geno = cp.maximum(geno, 0)
    n_var = cp.float64(geno.shape[1])

    # indicator matrices for each genotype value
    I0 = (geno == 0).astype(cp.float64)
    I1 = (geno == 1).astype(cp.float64)
    I2 = (geno == 2).astype(cp.float64)

    matches = I0 @ I0.T + I1 @ I1.T + I2 @ I2.T
    diffs_mat = n_var - matches

    n = geno.shape[0]
    idx_i, idx_j = cp.triu_indices(n, k=1)
    return diffs_mat[idx_i, idx_j]


def dist_var(matrix, population=None):
    """Variance of pairwise distance distribution (GPU).

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    diffs = _get_diffs(matrix, population)
    if diffs.shape[0] < 2:
        return 0.0
    mean = cp.mean(diffs)
    return float((cp.sum((diffs - mean) ** 2) / (diffs.shape[0] - 1)).get())


def dist_skew(matrix, population=None):
    """Skewness of pairwise distance distribution (GPU).

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    diffs = _get_diffs(matrix, population)
    if diffs.shape[0] < 3:
        return 0.0
    n = diffs.shape[0]
    mean = cp.mean(diffs)
    m2 = cp.mean((diffs - mean) ** 2)
    m3 = cp.mean((diffs - mean) ** 3)
    # scipy skewness (bias=True): m3 / m2^1.5
    s = cp.where(m2 > 0, m3 / (m2 ** 1.5), 0.0)
    return float(s.get())


def dist_kurt(matrix, population=None):
    """Excess kurtosis of pairwise distance distribution (GPU).

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    diffs = _get_diffs(matrix, population)
    if diffs.shape[0] < 4:
        return 0.0
    mean = cp.mean(diffs)
    m2 = cp.mean((diffs - mean) ** 2)
    m4 = cp.mean((diffs - mean) ** 4)
    # excess kurtosis: m4/m2^2 - 3
    k = cp.where(m2 > 0, m4 / (m2 ** 2) - 3.0, 0.0)
    return float(k.get())


def _get_diffs(matrix, population=None):
    """Dispatch to haploid or diploid pairwise diffs (returns CuPy array)."""
    if isinstance(matrix, GenotypeMatrix):
        return pairwise_diffs_diploid(matrix, population)
    else:
        return pairwise_diffs_haploid(matrix, population)
