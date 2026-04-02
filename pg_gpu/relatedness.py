"""
GPU-accelerated relatedness and kinship statistics.

Provides GRM (Genetic Relationship Matrix) and IBS (Identity by State)
sharing computed on GPU via CuPy matrix operations.
"""

import numpy as np
import cupy as cp
from typing import Optional, Union


def grm(genotype_matrix_or_haplotype_matrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include') -> np.ndarray:
    """Compute the Genetic Relationship Matrix (GRM).

    The GRM measures genome-wide allele sharing between all pairs of
    individuals, standardized by allele frequencies. Equivalent to
    GCTA's --make-grm or PLINK2's --make-rel.

    Formula:
        A_ij = (1/M) * sum_k[(g_ik - 2*p_k)(g_jk - 2*p_k) / (2*p_k*(1-p_k))]

    where g is genotype (0/1/2), p is allele frequency, M is number of sites.

    Parameters
    ----------
    genotype_matrix_or_haplotype_matrix : GenotypeMatrix or HaplotypeMatrix
        Diploid genotypes (0/1/2) or haplotype data (auto-converted to
        diploid by pairing consecutive haplotypes).
    population : str or list, optional
        Population subset.
    missing_data : str
        'include' - use per-site allele frequencies from non-missing data.
        'exclude' - restrict to sites with no missing data.

    Returns
    -------
    ndarray, float64, shape (n_individuals, n_individuals)
        Symmetric GRM. Diagonal entries are individual inbreeding
        coefficients + 1. Off-diagonal entries are pairwise relatedness.
    """
    geno = _get_diploid_genotypes(genotype_matrix_or_haplotype_matrix,
                                   population)

    if missing_data == 'exclude':
        valid_sites = cp.all(geno >= 0, axis=0)
        geno = geno[:, valid_sites]

    n_ind, n_snps = geno.shape
    if n_snps == 0:
        return np.zeros((n_ind, n_ind), dtype=np.float64)

    # Mask missing data, compute per-site allele frequencies
    valid = (geno >= 0).astype(cp.float64)
    g_clean = cp.where(geno >= 0, geno, 0).astype(cp.float64)

    n_valid = cp.sum(valid, axis=0)
    p = cp.where(n_valid > 0, cp.sum(g_clean, axis=0) / (2.0 * n_valid), 0.0)

    # Filter monomorphic sites (variance = 0)
    poly = (p > 0) & (p < 1)
    g_clean = g_clean[:, poly]
    valid = valid[:, poly]
    p = p[poly]
    n_snps_used = int(poly.sum().get())

    if n_snps_used == 0:
        return np.zeros((n_ind, n_ind), dtype=np.float64)

    # Center: g - 2p
    centered = (g_clean - 2.0 * p) * valid

    # Scale: 1 / sqrt(2*p*(1-p))
    scale = cp.sqrt(2.0 * p * (1.0 - p))
    standardized = centered / scale

    # GRM = (1/M) * X @ X.T
    A = (standardized @ standardized.T) / n_snps_used

    return A.get()


def ibs(genotype_matrix_or_haplotype_matrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include') -> np.ndarray:
    """Compute pairwise IBS (Identity by State) proportions.

    IBS measures the proportion of alleles shared identical by state
    between all pairs of individuals. Equivalent to PLINK's --distance
    ibs matrix.

    For each site, IBS between individuals i and j is:
        IBS = (2 - |g_i - g_j|) / 2

    The matrix contains the mean IBS across all jointly-called sites.

    Parameters
    ----------
    genotype_matrix_or_haplotype_matrix : GenotypeMatrix or HaplotypeMatrix
        Diploid genotypes (0/1/2) or haplotype data.
    population : str or list, optional
        Population subset.
    missing_data : str
        'include' - use jointly non-missing sites per pair.
        'exclude' - restrict to sites with no missing data.

    Returns
    -------
    ndarray, float64, shape (n_individuals, n_individuals)
        Symmetric IBS matrix. Values in [0, 1] where 1 = identical.
        Diagonal is always 1.
    """
    geno = _get_diploid_genotypes(genotype_matrix_or_haplotype_matrix,
                                   population)

    if missing_data == 'exclude':
        valid_sites = cp.all(geno >= 0, axis=0)
        geno = geno[:, valid_sites]

    n_ind, n_snps = geno.shape
    if n_snps == 0:
        return np.eye(n_ind, dtype=np.float64)

    valid = (geno >= 0).astype(cp.float64)
    g_clean = cp.where(geno >= 0, geno, 0).astype(cp.float64)

    # IBS per site = (2 - |g_i - g_j|) / 2
    # Expand: |g_i - g_j| = g_i^2 + g_j^2 - 2*g_i*g_j (for matching)
    # Actually for IBS we need the L1 distance, not L2.
    #
    # Use indicator matrices for exact IBS counting:
    # IBS2 (share both alleles) = I0.T @ I0 + I1.T @ I1 + I2.T @ I2
    # IBS1 (share one allele) = I0.T @ I1 + I1.T @ I0 + I1.T @ I2 + I2.T @ I1
    # IBS0 (share no alleles) = I0.T @ I2 + I2.T @ I0
    # IBS_prop = (2*IBS2 + IBS1) / (2 * n_jointly_valid)

    I0 = (g_clean == 0).astype(cp.float64) * valid
    I1 = (g_clean == 1).astype(cp.float64) * valid
    I2 = (g_clean == 2).astype(cp.float64) * valid

    # Count IBS2 (identical genotypes)
    ibs2 = I0 @ I0.T + I1 @ I1.T + I2 @ I2.T

    # Count IBS1 (share exactly one allele)
    ibs1 = I0 @ I1.T + I1 @ I0.T + I1 @ I2.T + I2 @ I1.T

    # Number of jointly valid sites per pair
    n_joint = valid @ valid.T

    # IBS proportion = (2*IBS2 + IBS1) / (2 * n_joint)
    ibs_mat = cp.where(n_joint > 0,
                        (2.0 * ibs2 + ibs1) / (2.0 * n_joint),
                        0.0)

    # Diagonal should be 1 (individual is identical to itself)
    cp.fill_diagonal(ibs_mat, 1.0)

    return ibs_mat.get()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_diploid_genotypes(matrix, population=None):
    """Extract diploid genotype matrix (n_individuals, n_variants) on GPU.

    If input is HaplotypeMatrix, pairs consecutive haplotypes into diploid
    genotypes (hap[0]+hap[1], hap[2]+hap[3], ...).
    """
    from .haplotype_matrix import HaplotypeMatrix
    from .genotype_matrix import GenotypeMatrix
    from ._utils import get_population_matrix

    if population is not None:
        matrix = get_population_matrix(matrix, population)

    if hasattr(matrix, 'device') and matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if isinstance(matrix, GenotypeMatrix):
        return matrix.genotypes
    elif isinstance(matrix, HaplotypeMatrix):
        hap = matrix.haplotypes
        n_hap = hap.shape[0]
        n_ind = n_hap // 2
        # Pair consecutive haplotypes: first n_ind are allele 1,
        # next n_ind are allele 2 (pg_gpu convention)
        geno = hap[:n_ind, :].astype(cp.int8) + hap[n_ind:, :].astype(cp.int8)
        # Mark as missing (-1) if either haplotype is missing
        missing = (hap[:n_ind, :] < 0) | (hap[n_ind:, :] < 0)
        geno[missing] = -1
        return geno
    else:
        raise TypeError(f"Expected HaplotypeMatrix or GenotypeMatrix, got {type(matrix)}")
