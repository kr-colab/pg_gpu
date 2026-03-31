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


def _extract_upper_triangle(mat):
    """Extract upper triangle of a square matrix as condensed vector."""
    n = mat.shape[0]
    idx_i, idx_j = cp.triu_indices(n, k=1)
    return mat[idx_i, idx_j]


def pairwise_diffs_haploid(haplotype_matrix, population=None):
    """Compute pairwise Hamming distances between haplotypes on GPU.

    Uses a single matrix multiply: for 0/1 data,
    diffs_ij = sum_i + sum_j - 2 * (X @ X.T)_ij.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional

    Returns
    -------
    diffs : cupy.ndarray, float64, condensed form (n_pairs,)
    """
    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    X = cp.maximum(matrix.haplotypes, 0).astype(cp.float64)
    row_sums = cp.sum(X, axis=1)
    gram = X @ X.T
    diffs_mat = row_sums[:, None] + row_sums[None, :] - 2.0 * gram

    return _extract_upper_triangle(diffs_mat)


def pairwise_diffs_diploid(genotype_matrix, population=None):
    """Compute pairwise genotype differences between diploid individuals.

    For 0/1/2 genotypes, uses indicator matrices: matches = I0.T@I0 +
    I1.T@I1 + I2.T@I2, then diffs = n_var - matches.

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

    I0 = (geno == 0).astype(cp.float64)
    I1 = (geno == 1).astype(cp.float64)
    I2 = (geno == 2).astype(cp.float64)

    matches = I0 @ I0.T + I1 @ I1.T + I2 @ I2.T
    diffs_mat = n_var - matches

    return _extract_upper_triangle(diffs_mat)


def dist_moments(matrix, population=None):
    """Compute variance, skewness, and kurtosis of pairwise distances.

    Computes the distance matrix once and derives all three moments,
    avoiding redundant matrix multiplies.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    population : str or list, optional

    Returns
    -------
    var : float
    skew : float
    kurt : float
    """
    diffs = _get_diffs(matrix, population)
    n = diffs.shape[0]

    if n < 2:
        return 0.0, 0.0, 0.0

    mean = cp.mean(diffs)
    centered = diffs - mean
    c2 = centered ** 2
    m2 = cp.mean(c2)

    var_val = float((cp.sum(c2) / (n - 1)).get())

    if n < 3 or float(m2.get()) == 0:
        return var_val, 0.0, 0.0

    m3 = cp.mean(centered ** 3)
    skew_val = float((m3 / (m2 ** 1.5)).get())

    if n < 4:
        return var_val, skew_val, 0.0

    m4 = cp.mean(centered ** 4)
    kurt_val = float((m4 / (m2 ** 2) - 3.0).get())

    return var_val, skew_val, kurt_val


def dist_var(matrix, population=None):
    """Variance of pairwise distance distribution.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    return dist_moments(matrix, population)[0]


def dist_skew(matrix, population=None):
    """Skewness of pairwise distance distribution.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    return dist_moments(matrix, population)[1]


def dist_kurt(matrix, population=None):
    """Excess kurtosis of pairwise distance distribution.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix

    Returns
    -------
    float
    """
    return dist_moments(matrix, population)[2]


def pairwise_diffs(matrix, population=None):
    """Compute pairwise Hamming distances on GPU.

    Accepts HaplotypeMatrix (0/1 data, single matrix multiply) or
    GenotypeMatrix (0/1/2 data, indicator matrix approach). Dispatches
    automatically.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    population : str or list, optional

    Returns
    -------
    diffs : cupy.ndarray, float64, condensed form (n_pairs,)
    """
    if isinstance(matrix, GenotypeMatrix):
        return pairwise_diffs_diploid(matrix, population)
    else:
        return pairwise_diffs_haploid(matrix, population)


# internal alias
_get_diffs = pairwise_diffs
