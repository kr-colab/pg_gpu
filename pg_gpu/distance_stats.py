"""
GPU-accelerated pairwise distance distribution statistics.

Computes pairwise Hamming distances and their distributional moments
(variance, skewness, kurtosis) for haploid and diploid data.
All computation stays on GPU until the final results.
"""

import numpy as np
import cupy as cp
from .haplotype_matrix import HaplotypeMatrix
from .genotype_matrix import GenotypeMatrix
from ._haplotype_groups import complete_sites
from ._utils import _eager_only, population_rows
from ._warnings import _phase_dependent, _warn_unphased


def _upper_triangle(mat):
    """The upper triangle of a square CuPy matrix as a condensed CuPy
    vector, row by row."""
    idx_i, idx_j = cp.triu_indices(mat.shape[0], k=1)
    return mat[idx_i, idx_j]


def _diffs_matrix_gpu(matrix, kind, population, missing_data):
    """Per-site differences between every two rows of ``matrix``, as a
    square CuPy matrix; NaN for two rows with no site called in both.

    ``kind`` is the class that the caller takes: its rows are haplotypes
    (allele indices) or individuals (0/1/2 genotypes), and both count
    differing values with one-hot indicators. A matrix of the other class
    raises, because its rows would be read as something they are not.
    """
    if not isinstance(matrix, kind):
        raise TypeError(
            f"expected a {kind.__name__}, got {type(matrix).__name__}")
    if population is not None:
        matrix = population_rows(matrix, population)
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    calls = matrix.genotypes if kind is GenotypeMatrix else matrix.haplotypes
    if missing_data == 'exclude':
        calls = complete_sites(calls)
    return _pairwise_diffs_matrix_gpu(calls, missing_data='normalize')


def _pairwise_diffs_matrix_gpu(hap, missing_data='include'):
    """Compute full pairwise Hamming distance matrix on GPU.

    Internal helper returning the raw cupy distance matrix. Used by
    pairwise_diffs_haploid and pairwise_diffs_diploid (condensed numpy
    output), decomposition.pairwise_distance, and the divergence
    two-population statistics (full cupy matrix with pop blocks).

    Accepts numpy or cupy input. When given numpy, transfers variant
    chunks to GPU on-the-fly so the full matrix never needs to reside
    on GPU at once.

    Multiallelic-correct Hamming: the number of DIFFERING jointly-valid sites is
    ``joint_valid - same``, where ``same`` counts sites at which the two
    haplotypes carry the same allele, summed over allele values via one-hot
    indicators (``same = sum_v I_v @ I_v.T``, ``I_v[h,s] = 1 if hap[h,s] == v``).
    This generalizes the biallelic ``sum_i + sum_j - 2*X@X.T`` gram identity
    (which silently assumes ``v^2 = v``) to any number of alleles, and reduces
    to it exactly on ``{0, 1}`` data. Missing entries (``-1``) match no allele
    value and are excluded from ``joint_valid``, so they never contribute.

    Parameters
    ----------
    hap : numpy.ndarray or cupy.ndarray, shape (n_haplotypes, n_variants)
        Haplotype data (or 0/1/2 genotypes), optionally pre-filtered.
    missing_data : str
        'include' - raw counts at jointly non-missing sites
        'normalize' - per-site average (divide by jointly valid count)
        'scale' - raw count scaled to every site (``* n_variants / joint``),
        so a pair with missing calls is not made to look close
        Under 'normalize' and 'scale' a pair with no jointly non-missing
        site has no distance and gets NaN.

    Returns
    -------
    diffs_mat : cupy.ndarray, float64, shape (n_hap, n_hap)
        Pairwise distance matrix on GPU.
    """
    from ._memutil import estimate_variant_chunk_size

    n_hap, n_var = hap.shape
    is_numpy = isinstance(hap, np.ndarray)
    chunk_size = estimate_variant_chunk_size(n_hap, bytes_per_element=8,
                                             n_intermediates=3)

    # One-hot Hamming over K allele values, K matmuls per chunk; K is the
    # number of distinct allele values, typically small.
    max_allele = int(hap.max()) if hap.size else -1
    n_alleles = max(max_allele + 1, 0)

    matches = cp.zeros((n_hap, n_hap), dtype=cp.float64)
    joint_valid = cp.zeros((n_hap, n_hap), dtype=cp.float64)

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        h_chunk = cp.asarray(hap[:, start:end]) if is_numpy else hap[:, start:end]
        v_chunk = (h_chunk >= 0).astype(cp.float64)
        joint_valid += v_chunk @ v_chunk.T
        # sites where both carry allele v (missing is -1, matches no v >= 0)
        for v in range(n_alleles):
            iv = (h_chunk == v).astype(cp.float64)
            matches += iv @ iv.T
            del iv
        del h_chunk, v_chunk

    diffs_mat = joint_valid - matches

    if missing_data in ('normalize', 'scale'):
        scale = n_var if missing_data == 'scale' else 1.0
        diffs_mat = cp.where(joint_valid > 0,
                             diffs_mat * scale / cp.maximum(joint_valid, 1.0),
                             cp.nan)
    return diffs_mat


@_eager_only
@_phase_dependent
def pairwise_diffs_haploid(haplotype_matrix, population=None,
                           missing_data='include'):
    """Compute pairwise Hamming distances between haplotypes on GPU.

    Per-site average allele differences over jointly non-missing sites,
    multiallelic-correct (one-hot indicators per allele value; see
    ``_pairwise_diffs_matrix_gpu``).

    Depends on haplotype phase; see :doc:`phasing`.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' - normalize by jointly-valid sites per pair
        'exclude' - only use sites with no missing data; every pair is
        NaN when no site is complete

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
        Per-site average differences. NaN for a pair with no site called
        in both haplotypes.
    """
    _warn_unphased(haplotype_matrix, context="pairwise_diffs_haploid",
                   alternative="use pairwise_diffs_diploid for individual-"
                   "level distances on a GenotypeMatrix")

    return cp.asnumpy(_upper_triangle(_diffs_matrix_gpu(
        haplotype_matrix, HaplotypeMatrix, population, missing_data)))


@_eager_only
def pairwise_diffs_diploid(genotype_matrix, population=None,
                           missing_data='include'):
    """Compute pairwise genotype differences between diploid individuals.

    Two genotypes differ when their 0/1/2 values differ: the one-hot count
    of ``_pairwise_diffs_matrix_gpu`` with the genotype values as alleles,
    divided by the sites both individuals call.

    Biallelic only: the input is a ``GenotypeMatrix`` of 0/1/2 alt-allele dosage,
    which is a biallelic representation by construction (multiallelic sites are
    filtered/collapsed when the matrix is built). For multiallelic data, use the
    allele-index haplotype path (``pairwise_diffs_haploid`` /
    ``divergence.pairwise_distance_matrix`` on a ``HaplotypeMatrix``), which is
    multiallelic-correct.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' - normalize by jointly-valid sites per pair
        'exclude' - only sites with no missing data; every pair is NaN
        when no site is complete

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
        NaN for a pair with no site called in both individuals.
    """
    return cp.asnumpy(_upper_triangle(_diffs_matrix_gpu(
        genotype_matrix, GenotypeMatrix, population, missing_data)))


@_eager_only
@_phase_dependent
def dist_moments(matrix, population=None, missing_data='include'):
    """Compute variance, skewness, and kurtosis of pairwise distances.

    Computes the distance matrix once and derives all three moments,
    avoiding redundant matrix multiplies. A pair with no site called in
    both samples has no distance, so the moments skip it.

    Depends on haplotype phase on a HaplotypeMatrix (a GenotypeMatrix is
    the unphased alternative); see :doc:`phasing`.

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
        A moment needs enough defined pairs (var 2, skew 3, kurt 4). With
        fewer, it is NaN when some pair has no distance, else 0.0. A
        moment with enough pairs is exactly 0.0 when every defined
        distance is the same.
    """
    _warn_unphased(matrix, context="dist_moments",
                   alternative="pass a GenotypeMatrix for individual-level "
                   "distances")
    kind = GenotypeMatrix if isinstance(matrix, GenotypeMatrix) else HaplotypeMatrix
    diffs = _diffs_matrix_gpu(matrix, kind, population, missing_data)
    n_rows = diffs.shape[0]

    # The matrix is symmetric, so each pair is there twice. The reductions
    # run over the whole matrix, without its diagonal, and the count and
    # the sums are halved; that needs no index of the upper triangle. Only
    # six scalars come to the host, in one copy.
    cp.fill_diagonal(diffs, cp.nan)
    defined = cp.isfinite(diffs)
    count = defined.sum()
    mean = cp.where(defined, diffs, 0.0).sum() / count
    centered = cp.where(defined, diffs - mean, 0.0)
    c2 = centered * centered
    count, sum_c2, sum_c3, sum_c4, low, high = cp.stack([
        count, c2.sum(), (c2 * centered).sum(), (c2 * c2).sum(),
        cp.where(defined, diffs, cp.inf).min(),
        cp.where(defined, diffs, -cp.inf).max()]).get()

    n = int(count) // 2
    fill = 0.0 if n == n_rows * (n_rows - 1) // 2 else np.nan
    if n < 2:
        return fill, fill, fill

    # Equal distances have no spread. The range says so exactly: each
    # distance is one division of two whole counts, so equal ratios are
    # equal bit for bit. The mean of equal values can round away from
    # them, which leaves centered sums that are tiny, not 0.
    same = low == high
    m2 = sum_c2 / (2 * n)
    var_val = 0.0 if same else float(sum_c2 / 2 / (n - 1))
    skew_val = fill if n < 3 else 0.0 if same else float(sum_c3 / (2 * n) / m2 ** 1.5)
    kurt_val = fill if n < 4 else 0.0 if same else float(sum_c4 / (2 * n) / m2 ** 2 - 3.0)
    return var_val, skew_val, kurt_val


@_eager_only
@_phase_dependent
def dist_var(matrix, population=None, missing_data='include'):
    """Variance of pairwise distance distribution.

    Depends on haplotype phase on a HaplotypeMatrix (a GenotypeMatrix is
    the unphased alternative); see :doc:`phasing`.
    """
    _warn_unphased(matrix, context="dist_var",
                   alternative="pass a GenotypeMatrix for individual-level "
                   "distances")
    return dist_moments(matrix, population, missing_data)[0]


@_eager_only
@_phase_dependent
def dist_skew(matrix, population=None, missing_data='include'):
    """Skewness of pairwise distance distribution.

    Depends on haplotype phase on a HaplotypeMatrix (a GenotypeMatrix is
    the unphased alternative); see :doc:`phasing`.
    """
    _warn_unphased(matrix, context="dist_skew",
                   alternative="pass a GenotypeMatrix for individual-level "
                   "distances")
    return dist_moments(matrix, population, missing_data)[1]


@_eager_only
@_phase_dependent
def dist_kurt(matrix, population=None, missing_data='include'):
    """Excess kurtosis of pairwise distance distribution.

    Depends on haplotype phase on a HaplotypeMatrix (a GenotypeMatrix is
    the unphased alternative); see :doc:`phasing`.
    """
    _warn_unphased(matrix, context="dist_kurt",
                   alternative="pass a GenotypeMatrix for individual-level "
                   "distances")
    return dist_moments(matrix, population, missing_data)[2]


@_eager_only
@_phase_dependent
def pairwise_diffs(matrix, population=None, missing_data='include'):
    """Compute pairwise Hamming distances on GPU.

    Accepts HaplotypeMatrix (allele indices) or GenotypeMatrix (0/1/2
    genotypes); both count differing values with one-hot indicators.
    Dispatches automatically.

    Depends on haplotype phase on a HaplotypeMatrix (a GenotypeMatrix is
    the unphased alternative); see :doc:`phasing`.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    population : str or list, optional
    missing_data : str

    Returns
    -------
    diffs : ndarray, float64, condensed form (n_pairs,)
        NaN for a pair with no site called in both samples.
    """
    if isinstance(matrix, GenotypeMatrix):
        return pairwise_diffs_diploid(matrix, population, missing_data)

    _warn_unphased(matrix, context="pairwise_diffs",
                   alternative="pass a GenotypeMatrix for individual-level "
                   "distances")
    return pairwise_diffs_haploid(matrix, population, missing_data)
