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
    """Extract upper triangle of a square matrix as condensed numpy vector."""
    n = mat.shape[0]
    idx_i, idx_j = cp.triu_indices(n, k=1)
    result = mat[idx_i, idx_j]
    return result.get() if hasattr(result, 'get') else result


def _pairwise_diffs_matrix_gpu(hap, missing_data='include'):
    """Compute full pairwise Hamming distance matrix on GPU.

    Internal helper returning the raw cupy distance matrix. Used by
    pairwise_diffs_haploid (condensed numpy output) and divergence
    two-population statistics (full cupy matrix with pop blocks).

    Accepts numpy or cupy input. When given numpy, transfers variant
    chunks to GPU on-the-fly so the full matrix never needs to reside
    on GPU at once.

    Parameters
    ----------
    hap : numpy.ndarray or cupy.ndarray, shape (n_haplotypes, n_variants)
        Haplotype data, optionally pre-filtered.
    missing_data : str
        'include' - raw counts at jointly non-missing sites
        'normalize' - per-site average (divide by jointly valid count)

    Returns
    -------
    diffs_mat : cupy.ndarray, float64, shape (n_hap, n_hap)
        Pairwise distance matrix on GPU.
    """
    from ._memutil import estimate_variant_chunk_size

    n_hap, n_var = hap.shape
    is_numpy = isinstance(hap, np.ndarray)
    chunk_size = estimate_variant_chunk_size(n_hap, bytes_per_element=8,
                                             n_intermediates=2)

    gram = cp.zeros((n_hap, n_hap), dtype=cp.float64)
    row_sums = cp.zeros(n_hap, dtype=cp.float64)
    need_valid = missing_data == 'normalize'
    joint_valid = cp.zeros((n_hap, n_hap), dtype=cp.float64) if need_valid else None

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        h_chunk = cp.asarray(hap[:, start:end]) if is_numpy else hap[:, start:end]
        x_chunk = cp.where(h_chunk >= 0, h_chunk, 0).astype(cp.float64)
        row_sums += cp.sum(x_chunk, axis=1)
        gram += x_chunk @ x_chunk.T
        if need_valid:
            v_chunk = (h_chunk >= 0).astype(cp.float64)
            joint_valid += v_chunk @ v_chunk.T
            del v_chunk
        del h_chunk, x_chunk

    diffs_mat = row_sums[:, None] + row_sums[None, :] - 2.0 * gram

    if joint_valid is not None:
        diffs_mat = cp.where(joint_valid > 0, diffs_mat / joint_valid, 0.0)

    return diffs_mat


def pairwise_diffs_haploid(haplotype_matrix, population=None,
                           missing_data='include'):
    """Compute pairwise Hamming distances between haplotypes on GPU.

    Uses a single matrix multiply: for 0/1 data,
    diffs_ij = sum_i + sum_j - 2 * (X @ X.T)_ij.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - normalize by jointly-valid sites per pair
        'exclude' - only use sites with no missing data

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
        For 'include'/'pairwise', values are per-site average differences.
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap = hap[:, complete]

    diffs_mat = _pairwise_diffs_matrix_gpu(hap, missing_data='normalize')
    return _extract_upper_triangle(diffs_mat)


def pairwise_diffs_diploid(genotype_matrix, population=None,
                           missing_data='include'):
    """Compute pairwise genotype differences between diploid individuals.

    For 0/1/2 genotypes, uses indicator matrices: matches = I0.T@I0 +
    I1.T@I1 + I2.T@I2, then diffs = n_var - matches.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' or 'pairwise' - normalize by jointly-valid sites per pair
        'exclude' - only sites with no missing data

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    if missing_data == 'exclude':
        missing_per_var = cp.sum(geno < 0, axis=0)
        complete = missing_per_var == 0
        geno = geno[:, complete]

    # 'include' mode (default): mask missing, normalize per pair
    # Chunk over variants to avoid OOM from float64 indicator matrices
    from ._memutil import estimate_variant_chunk_size
    n_ind, n_var = geno.shape
    chunk_size = estimate_variant_chunk_size(n_ind, bytes_per_element=8,
                                             n_intermediates=4)

    matches = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    joint_valid = cp.zeros((n_ind, n_ind), dtype=cp.float64)

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        g_chunk = geno[:, start:end]
        v_chunk = (g_chunk >= 0).astype(cp.float64)
        gc = cp.where(g_chunk >= 0, g_chunk, 0)
        i0 = (gc == 0).astype(cp.float64) * v_chunk
        i1 = (gc == 1).astype(cp.float64) * v_chunk
        i2 = (gc == 2).astype(cp.float64) * v_chunk
        matches += i0 @ i0.T + i1 @ i1.T + i2 @ i2.T
        joint_valid += v_chunk @ v_chunk.T
        del g_chunk, v_chunk, gc, i0, i1, i2

    diffs_mat = joint_valid - matches
    diffs_mat = cp.where(joint_valid > 0, diffs_mat / joint_valid, 0.0)

    return _extract_upper_triangle(diffs_mat)


def dist_moments(matrix, population=None, missing_data='include'):
    """Compute variance, skewness, and kurtosis of pairwise distances.

    Computes the distance matrix once and derives all three moments,
    avoiding redundant matrix multiplies.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    population : str or list, optional
    missing_data : str

    Returns
    -------
    var : float
    skew : float
    kurt : float
    """
    diffs = np.asarray(_get_diffs(matrix, population, missing_data))
    n = diffs.shape[0]

    if n < 2:
        return 0.0, 0.0, 0.0

    mean = np.mean(diffs)
    centered = diffs - mean
    c2 = centered ** 2
    m2 = np.mean(c2)

    var_val = float(np.sum(c2) / (n - 1))

    if n < 3 or m2 == 0:
        return var_val, 0.0, 0.0

    m3 = np.mean(centered ** 3)
    skew_val = float(m3 / (m2 ** 1.5))

    if n < 4:
        return var_val, skew_val, 0.0

    m4 = np.mean(centered ** 4)
    kurt_val = float(m4 / (m2 ** 2) - 3.0)

    return var_val, skew_val, kurt_val


def dist_var(matrix, population=None, missing_data='include'):
    """Variance of pairwise distance distribution."""
    return dist_moments(matrix, population, missing_data)[0]


def dist_skew(matrix, population=None, missing_data='include'):
    """Skewness of pairwise distance distribution."""
    return dist_moments(matrix, population, missing_data)[1]


def dist_kurt(matrix, population=None, missing_data='include'):
    """Excess kurtosis of pairwise distance distribution."""
    return dist_moments(matrix, population, missing_data)[2]


def pairwise_diffs(matrix, population=None, missing_data='include'):
    """Compute pairwise Hamming distances on GPU.

    Accepts HaplotypeMatrix (0/1 data, single matrix multiply) or
    GenotypeMatrix (0/1/2 data, indicator matrix approach). Dispatches
    automatically.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    population : str or list, optional
    missing_data : str

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
    """
    if isinstance(matrix, GenotypeMatrix):
        return pairwise_diffs_diploid(matrix, population, missing_data)
    else:
        return pairwise_diffs_haploid(matrix, population, missing_data)


# internal alias
_get_diffs = pairwise_diffs
