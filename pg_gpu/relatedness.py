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
    from ._memutil import estimate_variant_chunk_size
    from .streaming_matrix import _StreamingMatrixBase
    if isinstance(genotype_matrix_or_haplotype_matrix, _StreamingMatrixBase):
        return _stream_grm(genotype_matrix_or_haplotype_matrix,
                            population=population,
                            missing_data=missing_data)

    geno, n_ind = _get_genotype_data(genotype_matrix_or_haplotype_matrix,
                                      population)
    n_var = geno.shape[1]

    # Allele frequency from genotypes, chunked to avoid large temporaries.
    dac = cp.zeros(n_var, dtype=cp.int32)
    n_valid = cp.zeros(n_var, dtype=cp.int32)
    freq_chunk = max(1, n_var // 20)
    for s in range(0, n_var, freq_chunk):
        e = min(s + freq_chunk, n_var)
        gc = geno[:, s:e]
        v = gc >= 0
        n_valid[s:e] = cp.sum(v, axis=0)
        dac[s:e] = cp.sum(cp.where(v, gc, 0), axis=0)
        del gc, v
    p = cp.where(n_valid > 0, dac.astype(cp.float64) / (2.0 * n_valid.astype(cp.float64)), 0.0)

    if missing_data == 'exclude':
        complete = n_valid == n_ind
        poly = complete & (p > 0) & (p < 1)
    else:
        poly = (p > 0) & (p < 1)

    poly_idx = cp.where(poly)[0]
    n_snps_used = len(poly_idx)
    if n_snps_used == 0:
        return np.zeros((n_ind, n_ind), dtype=np.float64)

    p_poly = p[poly_idx]
    del p, dac, n_valid, poly

    # Chunked GRM: accumulate (std_chunk @ std_chunk.T) over variant chunks.
    # Peak per-chunk: 1 float64 array (standardized genotypes) + matmul workspace.
    chunk_size = estimate_variant_chunk_size(n_ind, bytes_per_element=8,
                                             n_intermediates=3,
                                             memory_fraction=0.25)
    A = cp.zeros((n_ind, n_ind), dtype=cp.float64)

    for start in range(0, n_snps_used, chunk_size):
        end = min(start + chunk_size, n_snps_used)
        idx = poly_idx[start:end]
        p_chunk = p_poly[start:end]

        g_chunk = geno[:, idx]
        valid = (g_chunk >= 0)
        g = cp.where(valid, g_chunk, 0).astype(cp.float64)
        del g_chunk

        scale = cp.sqrt(2.0 * p_chunk * (1.0 - p_chunk))
        # std = (g - 2p) * valid / scale, computed in-place
        g -= 2.0 * p_chunk
        g *= valid
        g /= scale
        del valid, scale
        A += g @ g.T
        del g

    A /= n_snps_used
    return A.get()


def ibs(genotype_matrix_or_haplotype_matrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include') -> np.ndarray:
    """Compute pairwise IBS (Identity by State) proportions.

    IBS measures the proportion of alleles shared identical by state
    between all pairs of individuals. Equivalent to PLINK's --distance
    ibs matrix.

    For each site, IBS between individuals i and j is
    ``IBS = (2 - abs(g_i - g_j)) / 2``.

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
    from ._memutil import estimate_variant_chunk_size
    from .streaming_matrix import _StreamingMatrixBase
    if isinstance(genotype_matrix_or_haplotype_matrix, _StreamingMatrixBase):
        return _stream_ibs(genotype_matrix_or_haplotype_matrix,
                            population=population,
                            missing_data=missing_data)

    geno, n_ind = _get_genotype_data(genotype_matrix_or_haplotype_matrix,
                                      population)
    n_var = geno.shape[1]
    # Peak per-chunk: 1 float64 indicator + matmul workspace.
    chunk_size = estimate_variant_chunk_size(n_ind, bytes_per_element=8,
                                             n_intermediates=3,
                                             memory_fraction=0.25)

    ibs2 = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    ibs1 = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    n_joint = cp.zeros((n_ind, n_ind), dtype=cp.float64)

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        g_chunk = geno[:, start:end]
        valid = (g_chunk >= 0).astype(cp.float64)
        g = cp.where(g_chunk >= 0, g_chunk, 0).astype(cp.float64)
        del g_chunk

        if missing_data == 'exclude':
            site_complete = cp.all(valid > 0, axis=0)
            valid = valid[:, site_complete]
            g = g[:, site_complete]

        n_joint += valid @ valid.T

        # Compute ibs2 and ibs1 one indicator at a time to avoid
        # materializing i0, i1, i2 simultaneously.
        for gval in (0, 1, 2):
            ind = (g == gval) * valid
            ibs2 += ind @ ind.T
            if gval < 2:
                ind_next = (g == (gval + 1)) * valid
                cross = ind @ ind_next.T
                ibs1 += cross + cross.T
                del ind_next
            del ind

        del g, valid

    ibs_mat = cp.where(n_joint > 0,
                        (2.0 * ibs2 + ibs1) / (2.0 * n_joint),
                        0.0)
    cp.fill_diagonal(ibs_mat, 1.0)

    return ibs_mat.get()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_genotype_data(matrix, population=None):
    """Get diploid genotype matrix (n_individuals, n_variants) on GPU.

    Returns genotypes directly from GenotypeMatrix, or builds them from
    HaplotypeMatrix by summing paired haplotypes. Avoids the memory cost
    of round-tripping through an intermediate haplotype representation.
    """
    from .haplotype_matrix import HaplotypeMatrix
    from .genotype_matrix import GenotypeMatrix
    from ._utils import get_population_matrix

    if population is not None:
        matrix = get_population_matrix(matrix, population)

    if hasattr(matrix, 'device') and matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if isinstance(matrix, GenotypeMatrix):
        return matrix.genotypes, matrix.genotypes.shape[0]
    elif isinstance(matrix, HaplotypeMatrix):
        hap = matrix.haplotypes
        n_ind = hap.shape[0] // 2
        h1 = hap[:n_ind, :]
        h2 = hap[n_ind:, :]
        missing = (h1 < 0) | (h2 < 0)
        geno = (cp.maximum(h1, 0) + cp.maximum(h2, 0)).astype(cp.int8)
        geno[missing] = -1
        return geno, n_ind
    else:
        raise TypeError(f"Expected HaplotypeMatrix or GenotypeMatrix, got {type(matrix)}")


def _get_haplotype_data(matrix, population=None):
    """Get haplotype matrix and number of diploid individuals.

    Returns the raw haplotype matrix on GPU (no genotype conversion)
    so callers can build diploid genotypes per-chunk.
    """
    from .haplotype_matrix import HaplotypeMatrix
    from .genotype_matrix import GenotypeMatrix
    from ._utils import get_population_matrix

    if population is not None:
        matrix = get_population_matrix(matrix, population)

    if hasattr(matrix, 'device') and matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if isinstance(matrix, GenotypeMatrix):
        # Convert genotypes back to haplotype layout for uniform processing
        geno = matrix.genotypes  # (n_ind, n_var)
        n_ind = geno.shape[0]
        # Fake haplotype layout: allele1 = geno // 2, allele2 = geno - allele1
        # This is approximate for het sites but correct for IBS/GRM counting
        h1 = cp.where(geno >= 0, geno // 2, -1).astype(cp.int8)
        h2 = cp.where(geno >= 0, geno - geno // 2, -1).astype(cp.int8)
        hap = cp.concatenate([h1, h2], axis=0)
        return hap, n_ind
    elif isinstance(matrix, HaplotypeMatrix):
        n_ind = matrix.num_haplotypes // 2
        return matrix.haplotypes, n_ind
    else:
        raise TypeError(f"Expected HaplotypeMatrix or GenotypeMatrix, got {type(matrix)}")


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


# ---------------------------------------------------------------------------
# Streaming IBS: variant-axis stream + indiv-axis row-block tile
# ---------------------------------------------------------------------------
#
# The three accumulators (ibs2, ibs1, n_joint) are each (n_ind, n_ind) and
# decompose by sum over variants, so the variant axis can be streamed
# chunk-by-chunk. The output itself can exceed GPU memory (80 GB at 100k
# diploids), so the individual axis is tiled into row blocks and the
# accumulators live on host -- only one row-block's (block_size, n_ind)
# matmul output sits on the GPU at a time.

def _stream_ibs(streaming_matrix, *, population, missing_data,
                block_size=None):
    from ._memutil import estimate_indiv_block_size

    chunks = iter(streaming_matrix.iter_gpu_chunks())
    first = next(chunks, None)
    if first is None:
        # No variants at all; mirror the eager identity-on-diagonal
        # convention so callers don't have to special-case the empty
        # source.
        n_ind = _stream_n_ind(streaming_matrix, population)
        ibs_mat = np.zeros((n_ind, n_ind), dtype=np.float64)
        np.fill_diagonal(ibs_mat, 1.0)
        return ibs_mat

    geno_first, n_ind = _get_genotype_data(first[2], population)
    if block_size is None:
        # block_ibs2 + block_ibs1 + transient matmul output + slack.
        block_size = estimate_indiv_block_size(n_ind, n_intermediates=4)

    ibs2_host = np.zeros((n_ind, n_ind), dtype=np.float64)
    ibs1_host = np.zeros((n_ind, n_ind), dtype=np.float64)
    n_joint_host = np.zeros((n_ind, n_ind), dtype=np.float64)

    _accumulate_ibs_chunk(geno_first, ibs2_host, ibs1_host, n_joint_host,
                          block_size=block_size, missing_data=missing_data)
    del geno_first
    for _, _, chunk in chunks:
        geno, _ = _get_genotype_data(chunk, population)
        _accumulate_ibs_chunk(geno, ibs2_host, ibs1_host, n_joint_host,
                              block_size=block_size,
                              missing_data=missing_data)
        del geno

    ibs_mat = np.where(n_joint_host > 0,
                        (2.0 * ibs2_host + ibs1_host) / (2.0 * n_joint_host),
                        0.0)
    np.fill_diagonal(ibs_mat, 1.0)
    return ibs_mat


def _accumulate_ibs_chunk(geno, ibs2_host, ibs1_host, n_joint_host, *,
                          block_size, missing_data):
    """Add one variant-chunk's contribution to the three host accumulators.

    ``geno`` is ``(n_ind, n_var_chunk)`` on GPU. The genotype-value loop
    is outermost so each full-pop indicator ``ind_full`` is built once
    per chunk rather than once per chunk per row block; the row-block
    loop only does matmuls and host transfers.

    In the eager kernel ``ibs1 += cross + cross.T`` where
    ``cross = (g==gval) @ (g==gval+1).T``; here the row-block slice of
    ``cross`` is ``ind_curr[r0:r1] @ ind_next.T`` and the row-block
    slice of ``cross.T`` is ``ind_next[r0:r1] @ ind_curr.T``.
    """
    valid_full = (geno >= 0).astype(cp.float64)
    g = cp.where(geno >= 0, geno, 0).astype(cp.float64)
    if missing_data == 'exclude':
        site_complete = cp.all(valid_full > 0, axis=0)
        valid_full = valid_full[:, site_complete]
        g = g[:, site_complete]
        if g.shape[1] == 0:
            return

    n_ind = g.shape[0]
    for r0 in range(0, n_ind, block_size):
        r1 = min(r0 + block_size, n_ind)
        n_joint_host[r0:r1] += (valid_full[r0:r1] @ valid_full.T).get()

    # Two indicator matrices live at a time: the current g==k and the
    # next g==k+1 needed for the ibs1 cross. Reuse ``ind_next`` from
    # the previous iter as the next ``ind_curr`` so g==1 is built once.
    ind_curr = (g == 0) * valid_full
    for gval in (0, 1, 2):
        if gval < 2:
            ind_next = (g == (gval + 1)) * valid_full
        for r0 in range(0, n_ind, block_size):
            r1 = min(r0 + block_size, n_ind)
            ibs2_host[r0:r1] += (ind_curr[r0:r1] @ ind_curr.T).get()
            if gval < 2:
                ibs1_host[r0:r1] += (
                    ind_curr[r0:r1] @ ind_next.T
                ).get()
                ibs1_host[r0:r1] += (
                    ind_next[r0:r1] @ ind_curr.T
                ).get()
        del ind_curr
        if gval < 2:
            ind_curr = ind_next


def _stream_n_ind(streaming_matrix, population):
    """Best-effort individual count for the empty-source fallback. Uses
    the streaming source's diploid count (halved for haplotype-layout)
    and a population subset if requested."""
    from .streaming_matrix import StreamingHaplotypeMatrix
    is_hap = isinstance(streaming_matrix, StreamingHaplotypeMatrix)
    if population is None:
        return (streaming_matrix.num_haplotypes // 2 if is_hap
                else streaming_matrix.num_individuals)
    pop_set = streaming_matrix.sample_sets[population]
    return len(pop_set) // 2 if is_hap else len(pop_set)


# ---------------------------------------------------------------------------
# Streaming GRM: two-pass + indiv-axis row-block tile
# ---------------------------------------------------------------------------
#
# GRM standardizes each variant by its allele frequency before forming the
# (n_ind, n_ind) outer product, so the eager kernel computes p over the
# whole chromosome up front. Two-pass form for the streaming path:
# pass 1 walks chunks to accumulate per-variant DAC + n_valid into
# chromosome-wide host arrays; pass 2 walks the same chunks, slicing the
# precomputed p into chunk-local pieces, standardizing on GPU, and
# accumulating A = g @ g.T on host with row-block tiling on the
# individual axis (same scheme as the IBS streaming path).

def _stream_grm(streaming_matrix, *, population, missing_data,
                block_size=None):
    from ._memutil import estimate_indiv_block_size

    # Pass 1: per-variant DAC / n_valid into per-chunk host arrays.
    dac_parts = []
    nvalid_parts = []
    n_ind = None
    for _, _, chunk in streaming_matrix.iter_gpu_chunks():
        geno, n_ind = _get_genotype_data(chunk, population)
        valid = (geno >= 0)
        nvalid_parts.append(
            cp.sum(valid, axis=0, dtype=cp.int32).get()
        )
        dac_parts.append(
            cp.sum(cp.where(valid, geno, 0), axis=0, dtype=cp.int32).get()
        )
        del geno, valid

    if n_ind is None:
        # No chunks at all.
        n_ind = _stream_n_ind(streaming_matrix, population)
        return np.zeros((n_ind, n_ind), dtype=np.float64)

    chunk_var_counts = [p.size for p in dac_parts]
    chunk_offsets = np.cumsum([0] + chunk_var_counts)
    dac_chrom = np.concatenate(dac_parts) if dac_parts else np.zeros(0, np.int32)
    nvalid_chrom = (np.concatenate(nvalid_parts)
                    if nvalid_parts else np.zeros(0, np.int32))

    p_chrom = np.where(
        nvalid_chrom > 0,
        dac_chrom.astype(np.float64) / (2.0 * nvalid_chrom.astype(np.float64)),
        0.0,
    )
    if missing_data == 'exclude':
        poly_mask = (nvalid_chrom == n_ind) & (p_chrom > 0) & (p_chrom < 1)
    else:
        poly_mask = (p_chrom > 0) & (p_chrom < 1)
    n_snps_used = int(poly_mask.sum())
    if n_snps_used == 0:
        return np.zeros((n_ind, n_ind), dtype=np.float64)

    if block_size is None:
        # standardized g + matmul output + slack.
        block_size = estimate_indiv_block_size(n_ind, n_intermediates=3)

    A_host = np.zeros((n_ind, n_ind), dtype=np.float64)

    # Pass 2: standardize each chunk and accumulate the outer product.
    # Chunk iteration order must match pass 1 so the chunk-local slice
    # of p_chrom lines up with the chunk's variants.
    for chunk_i, (_, _, chunk) in enumerate(
            streaming_matrix.iter_gpu_chunks()):
        c_lo, c_hi = chunk_offsets[chunk_i], chunk_offsets[chunk_i + 1]
        chunk_poly = poly_mask[c_lo:c_hi]
        if not chunk_poly.any():
            continue
        chunk_p_full = p_chrom[c_lo:c_hi]

        geno, _ = _get_genotype_data(chunk, population)
        if chunk_poly.all():
            # At biobank-scale MAF distributions essentially every site
            # is polymorphic; bypass the fancy index in that case so
            # the matmul sees a contiguous column block.
            chunk_p_gpu = cp.asarray(chunk_p_full)
            g = geno
        else:
            chunk_p_gpu = cp.asarray(chunk_p_full[chunk_poly])
            chunk_idx_gpu = cp.asarray(np.where(chunk_poly)[0],
                                        dtype=cp.int32)
            g = geno[:, chunk_idx_gpu]
        valid = (g >= 0)
        g_std = cp.where(valid, g, 0).astype(cp.float64)
        del g
        scale = cp.sqrt(2.0 * chunk_p_gpu * (1.0 - chunk_p_gpu))
        g_std -= 2.0 * chunk_p_gpu
        g_std *= valid
        g_std /= scale
        del valid, scale

        for r0 in range(0, n_ind, block_size):
            r1 = min(r0 + block_size, n_ind)
            A_host[r0:r1] += (g_std[r0:r1] @ g_std.T).get()
        del g_std, geno

    A_host /= n_snps_used
    return A_host
