"""
GPU-accelerated linkage disequilibrium statistics.

This module provides an API for computing LD statistics
on GPUs with automatic missing data handling.
"""

import numpy as np
import cupy as cp
from typing import Optional, Union, Tuple, List, Dict


def dd(counts: cp.ndarray,
       populations: Optional[Union[Tuple[int, int], int]] = None,
       n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² statistic for any population configuration.

    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array:
        - Single population: shape (N, 4)
        - Two populations: shape (N, 8)
        - Multi-population: shape (N, 4*P)
    populations : tuple of int, optional
        Population indices. None for single population,
        (i, j) for between populations i and j
    n_valid : cp.ndarray, optional
        Valid sample counts per population. Shape depends on configuration:
        - Single pop: shape (N,)
        - Two pops: shape (N, 2) or tuple of (N,) arrays

    Returns
    -------
    cp.ndarray
        D² values for each locus
    """
    # Handle different input formats
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _dd_single(counts, n_valid)
        else:
            # Default to first population if counts has multiple
            return _dd_single(counts[:, :4], n_valid[:, 0] if n_valid is not None and n_valid.ndim == 2 else n_valid)

    # Two population case
    pop1, pop2 = populations
    if pop1 == pop2:
        # Within population
        start_idx = pop1 * 4
        pop_counts = counts[:, start_idx:start_idx + 4]
        pop_n_valid = None
        if n_valid is not None:
            if n_valid.ndim == 2:
                pop_n_valid = n_valid[:, pop1]
            elif isinstance(n_valid, tuple):
                pop_n_valid = n_valid[pop1]
            else:
                pop_n_valid = n_valid
        return _dd_single(pop_counts, pop_n_valid)
    else:
        # Between populations
        return _dd_between(counts, pop1, pop2, n_valid)


def dz(counts: cp.ndarray,
       populations: Optional[Tuple[int, int, int]] = None,
       n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute Dz statistic for any population configuration.

    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    populations : tuple of int, optional
        Three population indices (i, j, k) for Dz(i,j,k).
        None defaults to single population (0, 0, 0)
    n_valid : cp.ndarray, optional
        Valid sample counts per population

    Returns
    -------
    cp.ndarray
        Dz values for each locus
    """
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _dz_single(counts, n_valid)
        else:
            # Default to first population
            populations = (0, 0, 0)

    return _dz_multi(counts, populations, n_valid)


def pi2(counts: cp.ndarray,
        populations: Optional[Tuple[int, int, int, int]] = None,
        n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute π₂ statistic for any population configuration.

    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    populations : tuple of int, optional
        Four population indices (i, j, k, l) for π₂(i,j,k,l).
        None defaults to single population (0, 0, 0, 0)
    n_valid : cp.ndarray, optional
        Valid sample counts per population

    Returns
    -------
    cp.ndarray
        π₂ values for each locus
    """
    if populations is None:
        # Single population case
        if counts.shape[1] == 4:
            return _pi2_single(counts, n_valid)
        else:
            # Default to first population
            populations = (0, 0, 0, 0)

    return _pi2_multi(counts, populations, n_valid)


def dd_within(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² within a single population.

    Convenience function equivalent to dd(counts, populations=None)
    """
    return _dd_single(counts, n_valid)


def dd_between(counts: cp.ndarray,
               pop1_idx: int,
               pop2_idx: int,
               n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute D² between two populations.

    Convenience function equivalent to dd(counts, populations=(pop1_idx, pop2_idx))
    """
    return _dd_between(counts, pop1_idx, pop2_idx, n_valid)


def r(counts: cp.ndarray,
      n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute Pearson correlation coefficient r between variant pairs
    from haplotype counts.

    Parameters
    ----------
    counts : cp.ndarray, shape (N, 4)
        Haplotype counts [n11, n10, n01, n00] for each variant pair.
    n_valid : cp.ndarray, optional
        Valid sample counts per pair. Shape (N,).

    Returns
    -------
    cp.ndarray, float64, shape (N,)
        Pearson r values. NaN where computation is undefined
        (monomorphic at either locus).
    """
    c11, c10, c01, c00 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid.astype(cp.float64) if n_valid is not None else cp.sum(counts, axis=1).astype(cp.float64)

    p_A = (c11 + c10) / n
    p_B = (c11 + c01) / n
    D = (c11 * c00 - c10 * c01).astype(cp.float64) / (n * n)
    denom = p_A * (1 - p_A) * p_B * (1 - p_B)

    valid_mask = denom > 0
    result = cp.full(n.shape[0], cp.nan, dtype=cp.float64)
    result[valid_mask] = D[valid_mask] / cp.sqrt(denom[valid_mask])

    return result


def r_squared(counts: cp.ndarray,
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """
    Compute r-squared (squared Pearson correlation) between variant pairs
    from haplotype counts.

    Parameters
    ----------
    counts : cp.ndarray, shape (N, 4)
        Haplotype counts [n11, n10, n01, n00] for each variant pair.
    n_valid : cp.ndarray, optional
        Valid sample counts per pair. Shape (N,).

    Returns
    -------
    cp.ndarray, float64, shape (N,)
        r-squared values. NaN where computation is undefined.
    """
    return r(counts, n_valid) ** 2


def _prepare_segregating(mat, missing_data='include'):
    """Filter to segregating sites and return cleaned arrays.

    Returns (hap_clean, valid_mask, m) or (None, None, 0) if < 2 sites.
    """
    if hasattr(mat, 'device') and mat.device == 'CPU':
        mat.transfer_to_gpu()

    if missing_data == 'exclude':
        hap = mat.haplotypes
        missing_per_var = cp.sum(hap < 0, axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        mat = mat.get_subset(valid)

    hap = mat.haplotypes
    dac = cp.sum(cp.maximum(hap, 0).astype(cp.int32), axis=0)
    n_valid_per_site = cp.sum((hap >= 0).astype(cp.int32), axis=0)
    seg = (dac > 0) & (dac < n_valid_per_site)
    seg_idx = cp.where(seg)[0]
    if len(seg_idx) < mat.num_variants:
        mat = mat.get_subset(seg_idx)

    hap = mat.haplotypes
    m = hap.shape[1]
    if m < 2:
        return None, None, 0

    valid_mask = (hap >= 0).astype(cp.float64)
    hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)
    return hap_clean, valid_mask, m


def _tile_counts(hi, vi, hj, vj):
    """Compute 4-way haplotype counts for all pairs in a tile.

    Returns c1, c2, c3, c4 as (B_i, B_j) matrices where:
        c1 = n_AB (derived at both)
        c2 = n_Ab (derived at i, ancestral at j)
        c3 = n_aB (ancestral at i, derived at j)
        c4 = n_ab (ancestral at both)
        n  = c1+c2+c3+c4 (valid at both sites)
    """
    c1 = hi.T @ hj           # derived at both
    s12 = hi.T @ vj           # derived at i, valid at j  (= c1 + c2)
    s13 = vi.T @ hj           # valid at i, derived at j  (= c1 + c3)
    n = vi.T @ vj             # valid at both
    c2 = s12 - c1
    c3 = s13 - c1
    c4 = n - c1 - c2 - c3
    return c1, c2, c3, c4, n


def _tile_r2_naive(hi, vi, hj, vj, pi, pqi, pj, pqj):
    """Compute naive r² for a tile (frequency-based, biased)."""
    joint_n = vi.T @ vj
    joint_11 = hi.T @ hj
    p_AB = cp.where(joint_n > 0, joint_11 / joint_n, 0.0)
    D = p_AB - cp.outer(pi, pj)
    denom = cp.outer(pqi, pqj)
    return cp.where(denom > 0, (D ** 2) / denom, 0.0)


def _tile_sigma_d2(hi, vi, hj, vj):
    """Compute unbiased D²/π² (sigma_d^2) for a tile.

    Uses multinomial projection estimators (Ragsdale & Gravel 2019):
      D² = [c1(c1-1)c4(c4-1) + c2(c2-1)c3(c3-1) - 2*c1*c2*c3*c4]
           / [n(n-1)(n-2)(n-3)]
      π² = [(c1+c2)(c1+c3)(c2+c4)(c3+c4) - c1*c4*(-1+c1+3c2+3c3+c4)
            - c2*c3*(-1+3c1+c2+c3+3c4)] / [n(n-1)(n-2)(n-3)]

    Returns sigma_d2 tile and valid mask (n >= 4).
    """
    c1, c2, c3, c4, n = _tile_counts(hi, vi, hj, vj)

    # Unbiased D² numerator
    dd_num = (c1 * (c1 - 1) * c4 * (c4 - 1)
              + c2 * (c2 - 1) * c3 * (c3 - 1)
              - 2 * c1 * c2 * c3 * c4)

    # Unbiased π² numerator
    s12 = c1 + c2
    s13 = c1 + c3
    s24 = c2 + c4
    s34 = c3 + c4
    pi2_num = (s12 * s13 * s24 * s34
               - c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
               - c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4))

    valid = n >= 4
    sigma_d2 = cp.where(valid & (pi2_num != 0),
                        dd_num / pi2_num, 0.0)
    return sigma_d2, valid


def _zns_tiled(mat, missing_data='include', tile_size=512):
    """Compute ZnS without materializing the full r² matrix.

    Uses tile-based accumulation: computes r² for B×B blocks and
    sums per tile, keeping memory at O(B²) instead of O(m²).

    When missing_data='project', uses unbiased multinomial projection
    estimators (Ragsdale & Gravel 2019) computing σ_D² = D²/π²
    per pair instead of naive r².
    """
    hap_clean, valid_mask, m = _prepare_segregating(mat, missing_data)
    if m < 2:
        return 0.0

    use_projection = (missing_data == 'project')

    B = tile_size
    total = 0.0
    n_pairs = 0

    if not use_projection:
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        p = cp.where(n_valid > 0,
                     cp.sum(hap_clean, axis=0) / n_valid, 0.0)
        pq = p * (1 - p)

    for i0 in range(0, m, B):
        i1 = min(i0 + B, m)
        hi = hap_clean[:, i0:i1]
        vi = valid_mask[:, i0:i1]

        for j0 in range(i0, m, B):
            j1 = min(j0 + B, m)
            hj = hap_clean[:, j0:j1]
            vj = valid_mask[:, j0:j1]

            if use_projection:
                tile, valid = _tile_sigma_d2(hi, vi, hj, vj)
                if i0 == j0:
                    cp.fill_diagonal(tile, 0.0)
                    cp.fill_diagonal(valid, False)
                    total += float(cp.sum(tile).get())
                    n_pairs += int(cp.sum(valid).get())
                else:
                    total += 2.0 * float(cp.sum(tile).get())
                    n_pairs += 2 * int(cp.sum(valid).get())
            else:
                r2_tile = _tile_r2_naive(
                    hi, vi, hj, vj,
                    p[i0:i1], pq[i0:i1], p[j0:j1], pq[j0:j1])
                if i0 == j0:
                    cp.fill_diagonal(r2_tile, 0.0)
                    total += float(cp.sum(r2_tile).get())
                else:
                    total += 2.0 * float(cp.sum(r2_tile).get())

    if use_projection:
        return total / n_pairs if n_pairs > 0 else 0.0
    return total / (m * (m - 1))


def _zns_from_precomputed(hap_clean, valid_mask, col_start, col_end,
                          tile_size=512, use_projection=False):
    """Compute ZnS for a column range using precomputed arrays.

    This avoids creating a HaplotypeMatrix and recomputing valid_mask/hap_clean
    for each window in the windowed_analysis loop.

    Parameters
    ----------
    hap_clean : cupy.ndarray, shape (n_hap, n_variants)
        Haplotype data with missing values set to 0.
    valid_mask : cupy.ndarray, shape (n_hap, n_variants)
        1 where data is valid, 0 where missing.
    col_start, col_end : int
        Column range [col_start, col_end) to compute ZnS over.
    tile_size : int
        Tile size for accumulation.
    use_projection : bool
        If True, use unbiased multinomial projection estimators.

    Returns
    -------
    float
        ZnS value, or 0.0 if fewer than 2 segregating sites.
    """
    hc = hap_clean[:, col_start:col_end]
    vm = valid_mask[:, col_start:col_end]

    # Filter to segregating sites
    n_valid = cp.sum(vm, axis=0).astype(cp.float64)
    dac = cp.sum(hc, axis=0)
    seg = (dac > 0) & (dac < n_valid)
    seg_idx = cp.where(seg)[0]
    m = len(seg_idx)
    if m < 2:
        return 0.0

    hc = hc[:, seg_idx]
    vm = vm[:, seg_idx]

    if not use_projection:
        n_valid = n_valid[seg_idx]
        p = cp.where(n_valid > 0, cp.sum(hc, axis=0) / n_valid, 0.0)
        pq = p * (1 - p)

    B = tile_size
    total = 0.0
    n_pairs = 0

    for i0 in range(0, m, B):
        i1 = min(i0 + B, m)
        hi = hc[:, i0:i1]
        vi = vm[:, i0:i1]

        for j0 in range(i0, m, B):
            j1 = min(j0 + B, m)
            hj = hc[:, j0:j1]
            vj = vm[:, j0:j1]

            if use_projection:
                tile, valid = _tile_sigma_d2(hi, vi, hj, vj)
                if i0 == j0:
                    cp.fill_diagonal(tile, 0.0)
                    cp.fill_diagonal(valid, False)
                    total += float(cp.sum(tile).get())
                    n_pairs += int(cp.sum(valid).get())
                else:
                    total += 2.0 * float(cp.sum(tile).get())
                    n_pairs += 2 * int(cp.sum(valid).get())
            else:
                r2_tile = _tile_r2_naive(
                    hi, vi, hj, vj,
                    p[i0:i1], pq[i0:i1], p[j0:j1], pq[j0:j1])
                if i0 == j0:
                    cp.fill_diagonal(r2_tile, 0.0)
                    total += float(cp.sum(r2_tile).get())
                else:
                    total += 2.0 * float(cp.sum(r2_tile).get())

    if use_projection:
        return total / n_pairs if n_pairs > 0 else 0.0
    return total / (m * (m - 1))


def zns(r2_matrix_or_matrix, missing_data='include'):
    """Kelly's ZnS: mean pairwise r-squared across all SNP pairs.

    Parameters
    ----------
    r2_matrix_or_matrix : ndarray, HaplotypeMatrix, or GenotypeMatrix
        Square r-squared matrix, or a matrix object (dispatches to
        haploid or diploid r-squared computation automatically).
        When a HaplotypeMatrix is passed, uses tiled computation to
        avoid materializing the full m×m r² matrix.
    missing_data : str
        'include' - per-site valid data for frequency computation
        'exclude' - filter to sites with no missing data
        'project' - unbiased multinomial projection estimators, computing
          mean σ_D² = D²/π² per pair using falling-factorial corrections
          for finite sample size. Requires HaplotypeMatrix input.
          See: Ragsdale & Gravel (2019) "Unbiased estimation of linkage
          disequilibrium from unphased data", MBE 37(3):923-932.

    Returns
    -------
    float
        Mean r-squared (or mean σ_D² for 'project' mode).
    """
    from .haplotype_matrix import HaplotypeMatrix

    # Streaming path for HaplotypeMatrix: O(B²) memory instead of O(m²)
    if isinstance(r2_matrix_or_matrix, HaplotypeMatrix):
        return _zns_tiled(r2_matrix_or_matrix, missing_data)

    if missing_data == 'project':
        raise ValueError(
            "missing_data='project' requires a HaplotypeMatrix, "
            "not a pre-computed r² array")

    r2_matrix = _resolve_r2_matrix(r2_matrix_or_matrix, missing_data)

    m = r2_matrix.shape[0]
    if m < 2:
        return 0.0
    total = cp.sum(r2_matrix) - cp.trace(r2_matrix)
    return float((total / (m * (m - 1))).get())


def _build_sigma_d2_matrix(mat, missing_data='include'):
    """Build full m×m σ_D² matrix using unbiased estimators.

    Used by omega() when missing_data='project'.
    """
    hap_clean, valid_mask, m = _prepare_segregating(mat, missing_data)
    if m < 2:
        return cp.zeros((0, 0), dtype=cp.float64)

    c1, c2, c3, c4, n = _tile_counts(hap_clean, valid_mask,
                                       hap_clean, valid_mask)

    dd_num = (c1 * (c1 - 1) * c4 * (c4 - 1)
              + c2 * (c2 - 1) * c3 * (c3 - 1)
              - 2 * c1 * c2 * c3 * c4)

    s12, s13, s24, s34 = c1 + c2, c1 + c3, c2 + c4, c3 + c4
    pi2_num = (s12 * s13 * s24 * s34
               - c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
               - c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4))

    valid = (n >= 4) & (pi2_num != 0)
    result = cp.where(valid, dd_num / pi2_num, 0.0)
    cp.fill_diagonal(result, 0.0)
    return result


def omega(r2_matrix_or_matrix, missing_data='include'):
    """Kim and Nielsen's Omega: max ratio of within-partition to
    cross-partition mean LD.

    For each possible SNP partition point l, splits variants into
    [0:l) and [l:m), computes mean r-squared within each block
    and between blocks. Returns max(mean_within / mean_cross).

    Uses GPU prefix sums on the upper triangle to evaluate all
    partition points without a Python loop. Matches diploSHIC's
    convention of using upper-triangle pairs only.

    Parameters
    ----------
    r2_matrix_or_matrix : ndarray, HaplotypeMatrix, or GenotypeMatrix
        Square r-squared matrix, or a matrix object (dispatches to
        haploid or diploid r-squared computation automatically).
    missing_data : str
        'include' - per-site valid data for frequency computation
        'exclude' - filter to sites with no missing data
        'project' - unbiased multinomial projection estimators, using
          σ_D² = D²/π² instead of naive r². Requires HaplotypeMatrix input.
          See: Ragsdale & Gravel (2019) "Unbiased estimation of linkage
          disequilibrium from unphased data", MBE 37(3):923-932.

    Returns
    -------
    float
        Maximum omega value. Returns 0 if fewer than 5 SNPs.
    """
    from .haplotype_matrix import HaplotypeMatrix

    if missing_data == 'project':
        if not isinstance(r2_matrix_or_matrix, HaplotypeMatrix):
            raise ValueError(
                "missing_data='project' requires a HaplotypeMatrix")
        r2_matrix = _build_sigma_d2_matrix(r2_matrix_or_matrix)
    else:
        r2_matrix = _resolve_r2_matrix(r2_matrix_or_matrix, missing_data)

    m = r2_matrix.shape[0]
    if m < 5:
        return 0.0

    # work with upper triangle only (i < j), matching diploSHIC
    r2 = cp.triu(r2_matrix, k=1)

    # 2D prefix sums on upper triangle
    S = cp.cumsum(cp.cumsum(r2, axis=0), axis=1)

    def block_sum(r_start, r_end, c_start, c_end):
        """Sum of S[r_start:r_end, c_start:c_end] via inclusion-exclusion."""
        val = S[r_end - 1, c_end - 1]
        if r_start > 0:
            val -= S[r_start - 1, c_end - 1]
        if c_start > 0:
            val -= S[r_end - 1, c_start - 1]
        if r_start > 0 and c_start > 0:
            val += S[r_start - 1, c_start - 1]
        return val

    # partition points l = 3..m-2 (matching diploSHIC)
    l_vals = cp.arange(3, m - 1)

    # left block: upper triangle pairs (i,j) with i < j < l
    # = sum of r2[0:l, 0:l] upper triangle = block_sum(0, l, 0, l)
    left_sum = S[l_vals - 1, l_vals - 1]

    # total upper triangle sum
    total_upper = S[m - 1, m - 1]

    # cross block: pairs (i,j) with i < l and j >= l
    # = block_sum(0, l, l, m)
    cross_sum = S[l_vals - 1, m - 1] - left_sum

    # right block: pairs (i,j) with i >= l and j > i (upper triangle of right block)
    right_sum = total_upper - left_sum - cross_sum

    # pair counts (upper triangle only)
    n_left = l_vals * (l_vals - 1) // 2
    n_right = (m - l_vals) * (m - l_vals - 1) // 2
    n_cross = l_vals * (m - l_vals)

    n_within = n_left + n_right
    within_sum = left_sum + right_sum

    valid = (n_within > 0) & (n_cross > 0) & (cross_sum > 0)

    mean_within = cp.where(n_within > 0, within_sum / n_within.astype(cp.float64), 0.0)
    mean_cross = cp.where(n_cross > 0, cross_sum / n_cross.astype(cp.float64), 1.0)
    omega_vals = cp.where(valid, mean_within / mean_cross, 0.0)

    return float(cp.max(omega_vals).get())


def mu_ld(haplotype_matrix, missing_data='include'):
    """mu_LD: haplotype pattern exclusivity between left/right halves (RAiSD).

    Splits variants at midpoint and measures how exclusively haplotype
    patterns associate across halves. Elevated at sweep boundaries where
    LD structure changes abruptly.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    missing_data : str
        'include' - treat missing as wildcard in pattern matching
        'exclude' - filter to sites with no missing data

    Returns
    -------
    float
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes

    if missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        hap = hap[:, missing_per_var == 0]

    n_hap, n_var = hap.shape

    if n_var < 2:
        return 0.0

    mid = n_var // 2

    left = hap[:, :mid].get().astype(np.int8)
    right = hap[:, mid:].get().astype(np.int8)

    from .diversity import _cluster_haplotypes_with_missing
    left_labels = _cluster_haplotypes_with_missing(left)
    right_labels = _cluster_haplotypes_with_missing(right)

    # for each distinct left pattern, count how many distinct right patterns it pairs with
    left_to_right = {}
    right_to_left = {}
    for i in range(n_hap):
        ll, rl = left_labels[i], right_labels[i]
        left_to_right.setdefault(ll, set()).add(rl)
        right_to_left.setdefault(rl, set()).add(ll)

    n_left = len(left_to_right)
    n_right = len(right_to_left)

    if n_left == 0 or n_right == 0:
        return 0.0

    n_excl_left = sum(1 for v in left_to_right.values() if len(v) == 1)
    n_excl_right = sum(1 for v in right_to_left.values() if len(v) == 1)

    return float((n_excl_left / n_left + n_excl_right / n_right) / 2.0)


def _resolve_r2_matrix(r2_matrix_or_matrix, missing_data='include'):
    """Convert a matrix object to an r2 matrix, or pass through raw arrays.

    Filters to segregating sites only (excludes monomorphic variants)
    to match diploSHIC/allel convention for ZnS/Omega.
    """
    from .haplotype_matrix import HaplotypeMatrix
    from .genotype_matrix import GenotypeMatrix

    if isinstance(r2_matrix_or_matrix, (GenotypeMatrix, HaplotypeMatrix)):
        mat = r2_matrix_or_matrix
        if hasattr(mat, 'device') and mat.device == 'CPU':
            mat.transfer_to_gpu()

        # Filter missing data sites
        if missing_data == 'exclude':
            hap = mat.haplotypes if isinstance(mat, HaplotypeMatrix) else mat.genotypes
            missing_per_var = cp.sum(hap < 0, axis=0)
            valid = cp.where(missing_per_var == 0)[0]
            if isinstance(mat, HaplotypeMatrix):
                mat = mat.get_subset(valid)
            else:
                geno = mat.genotypes[:, valid]
                pos = mat.positions[valid]
                from .genotype_matrix import GenotypeMatrix as GM
                mat = GM(geno, pos)

        # Haploid: filter monomorphic sites before r^2 computation.
        # diploSHIC marks monomorphic pairs as -1 and skips them in ZnS/Omega.
        # We match this by excluding monomorphic sites entirely.
        if isinstance(mat, HaplotypeMatrix):
            hap = mat.haplotypes
            dac = cp.sum(cp.maximum(hap, 0).astype(cp.int32), axis=0)
            n_valid = cp.sum((hap >= 0).astype(cp.int32), axis=0)
            seg = (dac > 0) & (dac < n_valid)
            seg_idx = cp.where(seg)[0]
            if len(seg_idx) < mat.num_variants:
                mat = mat.get_subset(seg_idx)
            return mat.pairwise_r2().astype(cp.float64)
        else:
            return _r2_matrix_diploid(mat)
    else:
        if not isinstance(r2_matrix_or_matrix, cp.ndarray):
            return cp.asarray(r2_matrix_or_matrix, dtype=cp.float64)
        return r2_matrix_or_matrix


def _r2_matrix_diploid(genotype_matrix):
    """Compute r-squared matrix from diploid genotypes (0/1/2) on GPU.

    Uses genotype correlation: treats 0/1/2 as continuous dosage values,
    computes Pearson correlation, then squares.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix or cupy.ndarray
        If GenotypeMatrix, uses .genotypes. If array, shape (n_individuals, n_variants).

    Returns
    -------
    r2 : cupy.ndarray, float64, shape (n_variants, n_variants)
    """
    from .genotype_matrix import GenotypeMatrix

    if isinstance(genotype_matrix, GenotypeMatrix):
        if genotype_matrix.device == 'CPU':
            genotype_matrix.transfer_to_gpu()
        geno = genotype_matrix.genotypes
    else:
        geno = genotype_matrix

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    # mask missing data: compute per-site mean from valid data only
    valid_mask = (geno >= 0).astype(cp.float64)
    geno_clean = cp.where(geno >= 0, geno, 0).astype(cp.float64)
    n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)

    mean = cp.where(n_valid > 0, cp.sum(geno_clean, axis=0) / n_valid, 0.0)

    # center, zeroing out missing entries
    gn = (geno_clean - mean[None, :]) * valid_mask

    # variance per variant (using valid counts)
    var = cp.sum(gn ** 2, axis=0)

    # correlation via matrix multiply
    cov = gn.T @ gn  # (n_var, n_var)

    # normalize: r_ij = cov_ij / sqrt(var_i * var_j)
    denom = cp.sqrt(cp.outer(var, var))
    r2 = cp.where(denom > 0, (cov / denom) ** 2, 0.0)
    cp.fill_diagonal(r2, 0.0)

    return r2


# Keep old names as aliases for backward compat
r2_matrix_diploid = _r2_matrix_diploid
zns_diploid = zns
omega_diploid = omega


def compute_ld_statistics(counts: cp.ndarray,
                         statistics: List[str] = ['dd', 'dz', 'pi2'],
                         populations: Optional[Dict[str, Union[Tuple, None]]] = None,
                         n_valid: Optional[cp.ndarray] = None) -> Dict[str, cp.ndarray]:
    """
    Compute multiple LD statistics in one pass.

    Parameters
    ----------
    counts : cp.ndarray
        Haplotype counts array
    statistics : list of str
        Statistics to compute ('dd', 'dz', 'pi2')
    populations : dict, optional
        Population configurations for each statistic.
        E.g., {'dd': (0, 1), 'dz': (0, 0, 1), 'pi2': (0, 0, 1, 1)}
    n_valid : cp.ndarray, optional
        Valid sample counts per population

    Returns
    -------
    dict
        Dictionary mapping statistic names to computed values
    """
    if populations is None:
        populations = {}

    results = {}

    for stat in statistics:
        if stat == 'dd':
            pop_config = populations.get('dd', None)
            results['dd'] = dd(counts, pop_config, n_valid)
        elif stat == 'dz':
            pop_config = populations.get('dz', None)
            results['dz'] = dz(counts, pop_config, n_valid)
        elif stat == 'pi2':
            pop_config = populations.get('pi2', None)
            results['pi2'] = pi2(counts, pop_config, n_valid)
        elif stat == 'r':
            results['r'] = r(counts, n_valid)
        elif stat == 'r_squared':
            results['r_squared'] = r_squared(counts, n_valid)
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return results


# Internal implementation functions


def _get_pop_data(counts, n_valid, pop_idx):
    """Extract counts and valid sample size for one population.

    Parameters
    ----------
    counts : cp.ndarray, shape (N, 4*P)
        Concatenated haplotype counts for P populations.
    n_valid : tuple of cp.ndarray, cp.ndarray with ndim==2, or None
        Per-population valid sample counts.
    pop_idx : int
        Population index (0-based).

    Returns
    -------
    c1, c2, c3, c4, n : cp.ndarray
        Haplotype counts and total valid samples for this population.
    """
    start = pop_idx * 4
    pop_counts = counts[:, start:start+4]
    if n_valid is not None:
        if isinstance(n_valid, tuple):
            if pop_idx < len(n_valid) and n_valid[pop_idx] is not None:
                pop_n = n_valid[pop_idx]
            else:
                pop_n = cp.sum(pop_counts, axis=1)
        elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
            pop_n = n_valid[:, pop_idx]
        else:
            pop_n = n_valid
    else:
        pop_n = cp.sum(pop_counts, axis=1)
    return pop_counts[:, 0], pop_counts[:, 1], pop_counts[:, 2], pop_counts[:, 3], pop_n


def _dd_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute D² for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)

    numer = c1 * (c1 - 1) * c4 * (c4 - 1) + c2 * (c2 - 1) * c3 * (c3 - 1) - 2 * c1 * c2 * c3 * c4
    denom = n * (n - 1) * (n - 2) * (n - 3)

    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _dd_between(counts: cp.ndarray,
                pop1_idx: int,
                pop2_idx: int,
                n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute D² between two populations."""
    # Extract counts for each population
    start1 = pop1_idx * 4
    start2 = pop2_idx * 4

    c11, c12, c13, c14 = counts[:, start1], counts[:, start1+1], counts[:, start1+2], counts[:, start1+3]
    c21, c22, c23, c24 = counts[:, start2], counts[:, start2+1], counts[:, start2+2], counts[:, start2+3]

    # Get valid sample sizes
    if n_valid is not None:
        if isinstance(n_valid, tuple):
            n1 = n_valid[0] if n_valid[0] is not None else cp.sum(counts[:, start1:start1+4], axis=1)
            n2 = n_valid[1] if n_valid[1] is not None else cp.sum(counts[:, start2:start2+4], axis=1)
        elif hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
            n1 = n_valid[:, pop1_idx]
            n2 = n_valid[:, pop2_idx]
        else:
            # Assume n_valid is for between-population pairs
            n1 = n_valid
            n2 = n_valid
    else:
        n1 = cp.sum(counts[:, start1:start1+4], axis=1)
        n2 = cp.sum(counts[:, start2:start2+4], axis=1)

    D1 = c12 * c13 - c11 * c14
    D2 = c22 * c23 - c21 * c24

    numer = D1 * D2
    denom = n1 * (n1 - 1) * n2 * (n2 - 1)

    valid_mask = (n1 >= 2) & (n2 >= 2)
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _dz_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute Dz for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)

    diff = c1 * c4 - c2 * c3
    sum_34_12 = (c3 + c4) - (c1 + c2)
    sum_24_13 = (c2 + c4) - (c1 + c3)
    sum_23_14 = (c2 + c3) - (c1 + c4)

    numer = diff * sum_34_12 * sum_24_13 + diff * sum_23_14 + 2 * (c2 * c3 + c1 * c4)
    denom = n * (n - 1) * (n - 2) * (n - 3)

    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _dz_multi(counts: cp.ndarray,
              populations: Tuple[int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute Dz for multiple populations."""
    pop1, pop2, pop3 = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    if pop1 == pop2 == pop3:
        # Single population
        if n_valid is not None and isinstance(n_valid, tuple):
            # Handle tuple case
            pop_n_valid = n_valid[pop1] if pop1 < len(n_valid) and n_valid[pop1] is not None else None
        elif n_valid is not None and hasattr(n_valid, 'ndim') and n_valid.ndim == 2:
            pop_n_valid = n_valid[:, pop1]
        else:
            pop_n_valid = n_valid
        return _dz_single(counts[:, pop1*4:(pop1+1)*4], pop_n_valid)

    elif pop1 == pop2:  # Dz(i,i,j)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop3)

        numer = (
            (-c11 - c12 + c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 + c22 - c23 + c24)
        )
        denom = n2 * n1 * (n1 - 1) * (n1 - 2)

        valid_mask = (n1 >= 3) & (n2 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    elif pop1 == pop3:  # Dz(i,j,i)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop2)

        numer = (
            (-c11 + c12 - c13 + c14)
            * (-(c12 * c13) + c11 * c14)
            * (-c21 - c22 + c23 + c24)
        )
        denom = n2 * n1 * (n1 - 1) * (n1 - 2)

        valid_mask = (n1 >= 3) & (n2 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    elif pop2 == pop3:  # Dz(i,j,j)
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop2)

        numer = (-(c12 * c13) + c11 * c14) * (-c21 + c22 + c23 - c24) + (
            -(c12 * c13) + c11 * c14
        ) * (-c21 + c22 - c23 + c24) * (-c21 - c22 + c23 + c24)
        denom = n1 * (n1 - 1) * n2 * (n2 - 1)

        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    else:
        # Dz(i,j,k) all different populations
        c11, c12, c13, c14, n1 = get_pop_data(pop1)
        c21, c22, c23, c24, n2 = get_pop_data(pop2)
        c31, c32, c33, c34, n3 = get_pop_data(pop3)

        numer = -(
            (c12 * c13 - c11 * c14)
            * (c21 + c22 - c23 - c24)
            * (c31 - c32 + c33 - c34)
        )
        denom = n1 * (n1 - 1) * n2 * n3

        valid_mask = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _pi2_single(counts: cp.ndarray, n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute π₂ for single population."""
    c1, c2, c3, c4 = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    n = n_valid if n_valid is not None else cp.sum(counts, axis=1)

    s12 = c1 + c2
    s13 = c1 + c3
    s24 = c2 + c4
    s34 = c3 + c4

    term_a = s12 * s13 * s24 * s34
    term_b = c1 * c4 * (-1 + c1 + 3 * c2 + 3 * c3 + c4)
    term_c = c2 * c3 * (-1 + 3 * c1 + c2 + c3 + 3 * c4)

    numer = term_a - term_b - term_c
    denom = n * (n - 1) * (n - 2) * (n - 3)

    valid_mask = n >= 4
    result = cp.zeros_like(n, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _pi2_multi(counts: cp.ndarray,
               populations: Tuple[int, int, int, int],
               n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Compute π₂ for multiple populations."""
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    # Count how many times each population index appears
    pop_list = [i, j, k, l]
    pop_counts = {}
    for p in pop_list:
        pop_counts[p] = pop_counts.get(p, 0) + 1
    n_unique = len(pop_counts)
    max_count = max(pop_counts.values())

    if n_unique == 1:
        # All same population
        return _pi2_single(counts[:, i*4:(i+1)*4],
                          n_valid[:, i] if n_valid is not None and n_valid.ndim == 2 else n_valid)

    elif max_count == 3:
        # Three same, one different -- normalize to (single, triple, triple, triple)
        triple_pop = [p for p, c in pop_counts.items() if c == 3][0]
        single_pop = [p for p, c in pop_counts.items() if c == 1][0]
        result = _pi2_iiij(counts, (single_pop, triple_pop, triple_pop, triple_pop), n_valid)

    elif i == j and k == l:
        # pi2(i,i,k,k) -- two pairs
        c11, c12, c13, c14, n1 = get_pop_data(i)
        c21, c22, c23, c24, n2 = get_pop_data(k)

        numer1 = (c11 + c12) * (c13 + c14) * (c21 + c23) * (c22 + c24)
        numer2 = (c21 + c22) * (c23 + c24) * (c11 + c13) * (c12 + c14)

        denom = n1 * (n1 - 1) * n2 * (n2 - 1)

        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = 0.5 * (numer1[valid_mask] + numer2[valid_mask]) / denom[valid_mask]

    elif i == j and k != l:
        # pi2(i,i,k,l) type -- handles both 2 and 3 distinct populations
        result = _pi2_iikl(counts, populations, n_valid)

    elif i != j and k == l:
        # pi2(i,j,k,k) type
        result = _pi2_ijkk(counts, populations, n_valid)

    elif (i == k and j == l) or (i == l and j == k):
        # pi2(i,j,i,j) or pi2(i,j,j,i) type
        c11, c12, c13, c14, n1 = get_pop_data(i)
        c21, c22, c23, c24, n2 = get_pop_data(j)

        numer = (
            ((c12 + c14) * (c13 + c14) * (c21 + c22) * (c21 + c23)) / 4.0
            + ((c11 + c13) * (c13 + c14) * (c21 + c22) * (c22 + c24)) / 4.0
            + ((c11 + c12) * (c12 + c14) * (c21 + c23) * (c23 + c24)) / 4.0
            + ((c11 + c12) * (c11 + c13) * (c22 + c24) * (c23 + c24)) / 4.0
            + (
                -(c12 * c13 * c21)
                + c14 * c21
                - c12 * c14 * c21
                - c13 * c14 * c21
                - c14 ** 2 * c21
                - c14 * c21 ** 2
                + c13 * c22
                - c11 * c13 * c22
                - c13 ** 2 * c22
                - c11 * c14 * c22
                - c13 * c14 * c22
                - c13 * c21 * c22
                - c14 * c21 * c22
                - c13 * c22 ** 2
                + c12 * c23
                - c11 * c12 * c23
                - c12 ** 2 * c23
                - c11 * c14 * c23
                - c12 * c14 * c23
                - c12 * c21 * c23
                - c14 * c21 * c23
                - c11 * c22 * c23
                - c14 * c22 * c23
                - c12 * c23 ** 2
                + c11 * c24
                - c11 ** 2 * c24
                - c11 * c12 * c24
                - c11 * c13 * c24
                - c12 * c13 * c24
                - c12 * c21 * c24
                - c13 * c21 * c24
                - c11 * c22 * c24
                - c13 * c22 * c24
                - c11 * c23 * c24
                - c12 * c23 * c24
                - c11 * c24 ** 2
            ) / 4.0
        )

        denom = n1 * (n1 - 1) * n2 * (n2 - 1)
        valid_mask = (n1 >= 2) & (n2 >= 2)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    else:
        if n_unique == 4:
            result = _pi2_all_different(counts, populations, n_valid)
        elif n_unique == 3:
            result = _pi2_shared_pop(counts, populations, n_valid)
        else:
            result = cp.zeros_like(get_pop_data(0)[4], dtype=cp.float64)

    return result


def _pi2_iiij(counts: cp.ndarray,
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,j,j,j) configurations."""
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    # For pi2(i,j,j,j) where j==k==l and i!=j
    c11, c12, c13, c14, n1 = get_pop_data(j)  # The population that appears 3 times
    c21, c22, c23, c24, n2 = get_pop_data(i)  # The population that appears once

    # From moments _pi2_iiij formula
    numer = (
        -((c11 + c12) * c14 * (c21 + c23))
        - (c12 * (c13 + c14) * (c21 + c23))
        + ((c11 + c12) * (c12 + c14) * (c13 + c14) * (c21 + c23))
        + ((c11 + c12) * (c13 + c14) * (-2 * c22 - 2 * c24))
        + ((c11 + c12) * c14 * (c22 + c24))
        + (c12 * (c13 + c14) * (c22 + c24))
        + ((c11 + c12) * (c11 + c13) * (c13 + c14) * (c22 + c24))
    ) / 2.0

    denom = n2 * n1 * (n1 - 1) * (n1 - 2)
    valid_mask = (n1 >= 3) & (n2 >= 1)

    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _pi2_iikl(counts: cp.ndarray,
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,i,k,l) configurations."""
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    # Get all unique populations involved
    unique_pops = list(set([i, k, l]))

    if len(unique_pops) == 2:  # Cases like (0,0,0,1) -- delegate to _pi2_iiij
        pop_minor = k if k != i else l
        result = _pi2_iiij(counts, (pop_minor, i, i, i), n_valid)
    else:
        # 3 distinct populations: pi2(i,i,j,k) where i,j,k all different
        # cs1 = counts[i], cs2 = counts[k], cs3 = counts[l]
        c11, c12, c13, c14, n1 = get_pop_data(i)
        c21, c22, c23, c24, n2 = get_pop_data(k)
        c31, c32, c33, c34, n3 = get_pop_data(l)

        numer = (
            (c11 + c12)
            * (c13 + c14)
            * (c22 * (c31 + c33) + c24 * (c31 + c33) + (c21 + c23) * (c32 + c34))
        ) / 2.0

        denom = n1 * (n1 - 1) * n2 * n3
        valid_mask = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
        result = cp.zeros_like(n1, dtype=cp.float64)
        result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _pi2_ijkk(counts: cp.ndarray,
              populations: Tuple[int, int, int, int],
              n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,j,k,k) configurations."""
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    # From moments: pi2(i,j,k,k) where pop3 == pop4
    c11, c12, c13, c14, n1 = get_pop_data(k)  # pop3/pop4 (k)
    c21, c22, c23, c24, n2 = get_pop_data(i)  # pop1 (i)

    # Special case: if j == k, cs3 is the same as cs1
    if j == k:
        c31, c32, c33, c34, n3 = c11, c12, c13, c14, n1
    else:
        c31, c32, c33, c34, n3 = get_pop_data(j)  # pop2 (j)

    # From moments formula
    numer = (
        (c11 + c13)
        * (c12 + c14)
        * (c23 * (c31 + c32) + c24 * (c31 + c32) + (c21 + c22) * (c33 + c34))
    ) / 2.0

    denom = n1 * (n1 - 1) * n2 * n3
    valid_mask = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)

    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]

    return result


def _pi2_shared_pop(counts: cp.ndarray,
                    populations: Tuple[int, int, int, int],
                    n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2 with one population shared between pairs.

    Handles pi2(i,j,i,k), pi2(i,j,k,i), pi2(i,j,j,k), pi2(i,j,k,j)
    where exactly one population appears in both the first and second pair.
    """
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    # Map to canonical form: cs1 = shared pop, cs2 = other from first pair,
    # cs3 = other from second pair
    if i == k:      # pi2(i,j;i,l)
        shared, other1, other2 = i, j, l
    elif i == l:    # pi2(i,j;k,i)
        shared, other1, other2 = i, j, k
    elif j == k:    # pi2(i,j;j,l)
        shared, other1, other2 = j, i, l
    elif j == l:    # pi2(i,j;k,j)
        shared, other1, other2 = j, i, k
    else:
        n1 = get_pop_data(0)[4]
        return cp.zeros_like(n1, dtype=cp.float64)

    c11, c12, c13, c14, n1 = get_pop_data(shared)
    c21, c22, c23, c24, n2 = get_pop_data(other1)
    c31, c32, c33, c34, n3 = get_pop_data(other2)

    numer = (
        c14 ** 2 * (c21 + c22) * (c31 + c33)
        + c12 ** 2 * (c23 + c24) * (c31 + c33)
        + (-1 + c11 + c13) * (c13 * (c21 + c22) + c11 * (c23 + c24)) * (c32 + c34)
        + c14
        * (
            c11 * (c23 + c24) * (c31 + c33)
            + c21
            * (
                (-1 + c13) * c31
                + c13 * c32
                - c33
                + c13 * c33
                + c13 * c34
                + c11 * (c32 + c34)
            )
            + c22
            * (
                (-1 + c13) * c31
                + c13 * c32
                - c33
                + c13 * c33
                + c13 * c34
                + c11 * (c32 + c34)
            )
        )
        + c12
        * (
            c14 * (c21 + c22 + c23 + c24) * (c31 + c33)
            + c13
            * (c21 * (c31 + c33) + c22 * (c31 + c33) + (c23 + c24) * (c32 + c34))
            + (c23 + c24) * ((-1 + c11) * c31 - c33 + c11 * (c32 + c33 + c34))
        )
    ) / 4.0

    denom = n1 * (n1 - 1) * n2 * n3
    valid_mask = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    return result


def _pi2_all_different(counts: cp.ndarray,
                       populations: Tuple[int, int, int, int],
                       n_valid: Optional[cp.ndarray] = None) -> cp.ndarray:
    """Helper for pi2(i,j,k,l) where all 4 populations are different."""
    i, j, k, l = populations

    def get_pop_data(pop_idx):
        return _get_pop_data(counts, n_valid, pop_idx)

    c11, c12, c13, c14, n1 = get_pop_data(i)
    c21, c22, c23, c24, n2 = get_pop_data(j)
    c31, c32, c33, c34, n3 = get_pop_data(k)
    c41, c42, c43, c44, n4 = get_pop_data(l)

    numer = (
        ((c13 + c14) * (c21 + c22) * (c32 + c34) * (c41 + c43)) / 4.0
        + ((c11 + c12) * (c23 + c24) * (c32 + c34) * (c41 + c43)) / 4.0
        + ((c13 + c14) * (c21 + c22) * (c31 + c33) * (c42 + c44)) / 4.0
        + ((c11 + c12) * (c23 + c24) * (c31 + c33) * (c42 + c44)) / 4.0
    )

    denom = n1 * n2 * n3 * n4
    valid_mask = (n1 >= 1) & (n2 >= 1) & (n3 >= 1) & (n4 >= 1)
    result = cp.zeros_like(n1, dtype=cp.float64)
    result[valid_mask] = numer[valid_mask] / denom[valid_mask]
    return result


# Backward compatibility layer
def DD(counts, n_valid=None):
    """Deprecated: Use dd() instead."""
    import warnings
    warnings.warn(
        "DD() is deprecated. Use ld_statistics.dd() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return dd_within(counts, n_valid)


def DD_two_pops(counts, pop1_idx, pop2_idx, n_valid1=None, n_valid2=None):
    """Deprecated: Use dd() with populations parameter instead."""
    import warnings
    warnings.warn(
        "DD_two_pops() is deprecated. Use ld_statistics.dd(counts, populations=(pop1_idx, pop2_idx)) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Reconstruct the expected format
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return dd(counts, populations=(pop1_idx, pop2_idx), n_valid=n_valid)


def Dz_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """Deprecated: Use dz() with populations parameter instead."""
    import warnings
    warnings.warn(
        "Dz_two_pops() is deprecated. Use ld_statistics.dz(counts, populations=pop_indices) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return dz(counts, populations=pop_indices, n_valid=n_valid)


def pi2_two_pops(counts, pop_indices, n_valid1=None, n_valid2=None):
    """Deprecated: Use pi2() with populations parameter instead."""
    import warnings
    warnings.warn(
        "pi2_two_pops() is deprecated. Use ld_statistics.pi2(counts, populations=pop_indices) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if n_valid1 is not None and n_valid2 is not None:
        n_valid = (n_valid1, n_valid2)
    else:
        n_valid = None
    return pi2(counts, populations=pop_indices, n_valid=n_valid)
