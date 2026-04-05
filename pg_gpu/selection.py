"""
GPU-accelerated selection scan statistics.

This module provides efficient computation of haplotype-based selection scan
statistics including iHS, XP-EHH, nSL, XP-nSL, EHH decay, and Garud's H
statistics. All core computations are parallelized over haplotype pairs
using CuPy for GPU acceleration.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix


# ---------------------------------------------------------------------------
# Public API: Standardization utilities
# ---------------------------------------------------------------------------

def standardize(score):
    """Centre scores to zero mean and scale to unit variance.

    Parameters
    ----------
    score : array_like, float, shape (n_variants,)
        Raw scores (e.g., unstandardized IHS or nSL).

    Returns
    -------
    ndarray, float, shape (n_variants,)
        Standardized scores.
    """
    score = np.asarray(score, dtype='f8')
    return (score - np.nanmean(score)) / np.nanstd(score)


def standardize_by_allele_count(score, aac, bins=None, n_bins=None):
    """Standardize scores within allele frequency bins.

    Parameters
    ----------
    score : array_like, float, shape (n_variants,)
        Raw scores (e.g., unstandardized IHS or nSL).
    aac : array_like, int, shape (n_variants,)
        Alternate allele counts for each variant.
    bins : array_like, int, optional
        Allele count bin edges. Overrides n_bins.
    n_bins : int, optional
        Number of bins to create. Default: max(aac) // 2.

    Returns
    -------
    score_standardized : ndarray, float, shape (n_variants,)
        Standardized scores.
    bins : ndarray
        Bin edges used.
    """
    from scipy.stats import binned_statistic

    score = np.asarray(score, dtype='f8')
    aac = np.asarray(aac)

    nonan = ~np.isnan(score)
    score_nonan = score[nonan]
    aac_nonan = aac[nonan]

    if bins is None:
        if n_bins is None:
            n_bins = max(1, int(np.max(aac) // 2))
        bins = _make_similar_sized_bins(aac_nonan, n_bins)
    else:
        bins = np.asarray(bins)

    mean_score, _, _ = binned_statistic(aac_nonan, score_nonan,
                                        statistic=np.mean, bins=bins)
    std_score, _, _ = binned_statistic(aac_nonan, score_nonan,
                                       statistic=np.std, bins=bins)

    score_standardized = np.full_like(score, np.nan)
    for i in range(len(bins) - 1):
        x1 = bins[i]
        x2 = bins[i + 1]
        if i == 0:
            loc = aac < x2
        elif i == len(bins) - 2:
            loc = aac >= x1
        else:
            loc = (aac >= x1) & (aac < x2)
        m = mean_score[i]
        s = std_score[i]
        if s > 0:
            score_standardized[loc] = (score[loc] - m) / s

    return score_standardized, bins


# ---------------------------------------------------------------------------
# Public API: Garud's H statistics
# ---------------------------------------------------------------------------

def garud_h(matrix, population=None, missing_data='include'):
    """Compute Garud's H1, H12, H123, and H2/H1 statistics.

    Accepts either HaplotypeMatrix (uses haplotype frequencies) or
    GenotypeMatrix (uses diplotype frequencies) and dispatches
    automatically.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
        Haplotype or diploid genotype data.
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples.
    missing_data : str
        'include' - treat missing as wildcard (compatible with any allele).
        'exclude' - filter to sites with no missing data.
        'pairwise' - same as 'include' for haplotype pattern matching.

    Returns
    -------
    h1 : float
        Sum of squared haplotype/diplotype frequencies.
    h12 : float
        H12 statistic (top two combined).
    h123 : float
        H123 statistic (top three combined).
    h2_h1 : float
        H2/H1 ratio indicating sweep softness.
    """
    from .genotype_matrix import GenotypeMatrix

    if isinstance(matrix, GenotypeMatrix):
        return _garud_h_diploid(matrix, population)

    haplotype_matrix = matrix
    if population is not None:
        haplotype_matrix = _get_population_matrix(haplotype_matrix, population)

    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    hap = haplotype_matrix.haplotypes
    if missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        hap = hap[:, missing_per_var == 0]

    f = _distinct_haplotype_frequencies_missing(hap)

    return _garud_from_freqs(f)


def moving_garud_h(haplotype_matrix: HaplotypeMatrix,
                   size: int,
                   start: int = 0,
                   stop: Optional[int] = None,
                   step: Optional[int] = None,
                   population: Optional[Union[str, list]] = None,
                   missing_data: str = 'include'):
    """Compute Garud's H statistics in moving windows of variants.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    size : int
        Window size (number of variants).
    start : int, optional
        Starting variant index.
    stop : int, optional
        Stopping variant index.
    step : int, optional
        Step between windows. Defaults to size (non-overlapping).
    population : str or list, optional
        Population name or list of sample indices.
    missing_data : str
        'include' - treat missing as wildcard in pattern matching
        'exclude' - filter to sites with no missing data

    Returns
    -------
    h1 : ndarray, float
    h12 : ndarray, float
    h123 : ndarray, float
    h2_h1 : ndarray, float
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        hap = hap[:, missing_per_var == 0]

    n_variants = hap.shape[1]

    if stop is None:
        stop = n_variants
    if step is None:
        step = size

    has_missing = bool(cp.any(hap < 0).get())

    if has_missing:
        # Fallback: per-window wildcard matching
        results = []
        for w_start in range(start, stop - size + 1, step):
            w_end = w_start + size
            hap_window = hap[:, w_start:w_end]
            f = _distinct_haplotype_frequencies_missing(hap_window)
            results.append(_garud_from_freqs(f))
        results = np.array(results, dtype='f8')
        return results[:, 0], results[:, 1], results[:, 2], results[:, 3]

    # GPU fast path: precompute cumulative weighted sums on GPU,
    # compute per-window hashes on GPU, transfer only the small hash
    # arrays to CPU for sorting/counting.
    # float64 for prefix-sum hashing to avoid accumulation error over long ranges
    n_hap = hap.shape[0]
    h_f64 = hap.astype(cp.float64)

    rng = cp.random.RandomState(seed=42)
    w1 = rng.standard_normal(n_variants, dtype=cp.float64)
    w2 = rng.standard_normal(n_variants, dtype=cp.float64)

    hw1 = h_f64 * w1[cp.newaxis, :]
    hw2 = h_f64 * w2[cp.newaxis, :]

    cs1 = cp.zeros((n_hap, n_variants + 1), dtype=cp.float64)
    cs2 = cp.zeros((n_hap, n_variants + 1), dtype=cp.float64)
    cp.cumsum(hw1, axis=1, out=cs1[:, 1:])
    cp.cumsum(hw2, axis=1, out=cs2[:, 1:])

    windows = list(range(start, stop - size + 1, step))
    n_windows = len(windows)

    # Compute all window hashes on GPU at once: (n_windows, n_hap)
    w_starts = cp.array(windows, dtype=cp.int64)
    w_ends = w_starts + size
    # Gather prefix sums at window boundaries: (n_hap, n_windows)
    all_hash1 = cs1[:, w_ends] - cs1[:, w_starts]  # (n_hap, n_windows)
    all_hash2 = cs2[:, w_ends] - cs2[:, w_starts]

    # Transfer to CPU: (n_hap, n_windows) * 2 * 8 bytes -- small
    ah1 = all_hash1.get().T  # (n_windows, n_hap)
    ah2 = all_hash2.get().T

    results = np.empty((n_windows, 4), dtype='f8')
    for wi in range(n_windows):
        order = np.lexsort((ah2[wi], ah1[wi]))
        s1 = ah1[wi, order]
        s2 = ah2[wi, order]
        diff = (np.abs(s1[1:] - s1[:-1]) > 1e-6) | (np.abs(s2[1:] - s2[:-1]) > 1e-6)
        boundaries = np.concatenate([[True], diff])
        boundary_idx = np.where(boundaries)[0]
        counts = np.diff(np.concatenate([boundary_idx, [n_hap]]))
        freqs = np.sort(counts)[::-1].astype(np.float64) / n_hap
        results[wi] = _garud_from_freqs(freqs)

    return results[:, 0], results[:, 1], results[:, 2], results[:, 3]


def _garud_from_freqs(f):
    """Compute H1/H12/H123/H2H1 from sorted frequency array."""
    h1 = float(np.sum(f ** 2))
    h12 = float(np.sum(f[:2]) ** 2 + np.sum(f[2:] ** 2))
    h123 = float(np.sum(f[:3]) ** 2 + np.sum(f[3:] ** 2))
    h2 = h1 - float(f[0] ** 2)
    h2_h1 = h2 / h1 if h1 > 0 else 0.0
    return h1, h12, h123, h2_h1


def _garud_h_diploid(genotype_matrix, population=None):
    """Garud's H from diplotype frequencies (internal)."""
    from .diversity import diplotype_frequency_spectrum
    freqs, _ = diplotype_frequency_spectrum(genotype_matrix, population)
    return _garud_from_freqs(freqs)


# backward compat alias
garud_h_diploid = _garud_h_diploid


# ---------------------------------------------------------------------------
# Public API: nSL / XP-nSL
# ---------------------------------------------------------------------------

def nsl(haplotype_matrix: HaplotypeMatrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include'):
    """Compute the unstandardized nSL statistic for each variant.

    Compares the mean shared haplotype length around the reference (0)
    vs alternate (1) allele at each site.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or list of sample indices.
    missing_data : str
        'include' - missing extends shared suffix length (default)
        'exclude' - filter to sites with no missing data before scan

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized nSL scores: log(nSL1 / nSL0).
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        missing_per_var = matrix.count_missing(axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        if len(valid) == 0:
            return np.full(matrix.num_variants, np.nan)
        matrix = matrix.get_subset(valid)

    # pg_gpu: (n_haplotypes, n_variants) -> allel convention: (n_variants, n_haplotypes)
    hap = matrix.haplotypes.T  # now (n_variants, n_haplotypes)

    # forward scan
    nsl0_fwd, nsl1_fwd = _nsl01_scan_gpu(hap)

    # backward scan
    nsl0_rev, nsl1_rev = _nsl01_scan_gpu(hap[::-1])
    nsl0_rev = nsl0_rev[::-1]
    nsl1_rev = nsl1_rev[::-1]

    nsl0 = nsl0_fwd + nsl0_rev
    nsl1 = nsl1_fwd + nsl1_rev

    score = cp.log(nsl1 / nsl0)
    return score.get()


def xpnsl(haplotype_matrix: HaplotypeMatrix,
          pop1: Union[str, list],
          pop2: Union[str, list],
          missing_data: str = 'include'):
    """Compute the unstandardized cross-population nSL statistic.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data containing both populations.
    pop1 : str or list
        First population name or sample indices.
    pop2 : str or list
        Second population name or sample indices.
    missing_data : str
        'include' - missing extends shared suffix length (default)
        'exclude' - filter to sites with no missing data before scan

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized XP-nSL scores: log(nSL_pop1 / nSL_pop2).
    """
    if missing_data == 'exclude':
        if haplotype_matrix.device == 'CPU':
            haplotype_matrix.transfer_to_gpu()
        missing_per_var = cp.sum(haplotype_matrix.haplotypes < 0, axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        if len(valid) == 0:
            return np.full(haplotype_matrix.num_variants, np.nan)
        haplotype_matrix = haplotype_matrix.get_subset(valid)

    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    if m1.device == 'CPU':
        m1.transfer_to_gpu()
    if m2.device == 'CPU':
        m2.transfer_to_gpu()

    h1 = m1.haplotypes.T  # (n_variants, n_haplotypes1)
    h2 = m2.haplotypes.T

    # forward
    nsl1_fwd = _nsl_scan_gpu(h1)
    nsl2_fwd = _nsl_scan_gpu(h2)

    # backward
    nsl1_rev = _nsl_scan_gpu(h1[::-1])[::-1]
    nsl2_rev = _nsl_scan_gpu(h2[::-1])[::-1]

    nsl1 = nsl1_fwd + nsl1_rev
    nsl2 = nsl2_fwd + nsl2_rev

    score = cp.log(nsl1 / nsl2)
    return score.get()


# ---------------------------------------------------------------------------
# Public API: IHS / XP-EHH
# ---------------------------------------------------------------------------

def ihs(haplotype_matrix: HaplotypeMatrix,
        pos=None,
        map_pos=None,
        min_ehh: float = 0.05,
        min_maf: float = 0.05,
        include_edges: bool = False,
        gap_scale: int = 20000,
        max_gap: int = 200000,
        is_accessible=None,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include'):
    """Compute the unstandardized integrated haplotype score (iHS).

    Compares the integrated EHH for the reference vs alternate allele
    at each variant.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    pos : array_like, optional
        Variant positions. If None, uses haplotype_matrix.positions.
    map_pos : array_like, optional
        Genetic map positions. If None, uses physical positions.
    min_ehh : float, optional
        Minimum EHH below which to truncate IHH computation.
    min_maf : float, optional
        Minimum minor allele frequency to compute score.
    include_edges : bool, optional
        If True, report scores even when EHH does not decay below min_ehh.
    gap_scale : int, optional
        Rescale gaps larger than this value.
    max_gap : int, optional
        Mark gaps larger than this as invalid.
    is_accessible : array_like, optional
        Genome accessibility mask.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'include' - missing extends shared suffix length (default)
        'exclude' - filter to sites with no missing data before scan

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized iHS scores: log(IHH1 / IHH0).
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        missing_per_var = matrix.count_missing(axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        if len(valid) == 0:
            return np.full(matrix.num_variants, np.nan)
        matrix = matrix.get_subset(valid)
        if pos is not None:
            pos = np.asarray(pos)[valid.get()]

    if pos is None:
        pos = matrix.positions
    if hasattr(pos, 'get'):
        pos = pos.get()
    pos = np.asarray(pos)

    hap = matrix.haplotypes.T  # (n_variants, n_haplotypes)
    n_haplotypes = hap.shape[1]

    # Fall back to matrix's accessible mask if not provided explicitly
    if is_accessible is None and haplotype_matrix.has_accessible_mask:
        is_accessible = haplotype_matrix.accessible_mask.mask
    gaps = _compute_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)
    gaps_gpu = cp.asarray(gaps)

    # Auto-select kernel: fused kernel uses 128-bit bitmask per thread,
    # limiting it to ~256 haplotypes. Histogram approach scales better and
    # is required above that threshold.
    scan_fn = _ihh01_scan_hist_gpu if n_haplotypes > 256 else _ihh01_scan_gpu

    # forward scan
    ihh0_fwd, ihh1_fwd = scan_fn(hap, gaps_gpu, min_ehh, min_maf,
                                  include_edges)
    # backward scan
    ihh0_rev, ihh1_rev = scan_fn(
        hap[::-1], gaps_gpu[::-1],
        min_ehh, min_maf, include_edges)
    ihh0_rev = ihh0_rev[::-1]
    ihh1_rev = ihh1_rev[::-1]

    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev

    score = np.log(ihh1 / ihh0)
    return score


def xpehh(haplotype_matrix: HaplotypeMatrix,
          pop1: Union[str, list],
          pop2: Union[str, list],
          pos=None,
          map_pos=None,
          min_ehh: float = 0.05,
          include_edges: bool = False,
          gap_scale: int = 20000,
          max_gap: int = 200000,
          is_accessible=None,
          missing_data: str = 'include'):
    """Compute the unstandardized cross-population EHH (XP-EHH) statistic.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data containing both populations.
    pop1 : str or list
        First population name or sample indices.
    pop2 : str or list
        Second population name or sample indices.
    pos : array_like, optional
        Variant positions. If None, uses haplotype_matrix.positions.
    map_pos : array_like, optional
        Genetic map positions.
    min_ehh : float, optional
        Minimum EHH threshold.
    include_edges : bool, optional
        Report scores even when EHH does not decay below min_ehh.
    gap_scale : int, optional
        Rescale gaps larger than this value.
    max_gap : int, optional
        Mark gaps larger than this as invalid.
    is_accessible : array_like, optional
        Genome accessibility mask.
    missing_data : str
        'include' - missing extends shared suffix length (default)
        'exclude' - filter to sites with no missing data before scan

    Returns
    -------
    score : ndarray, float, shape (n_variants,)
        Unstandardized XP-EHH scores: log(IHH_pop1 / IHH_pop2).
    """
    if missing_data == 'exclude':
        if haplotype_matrix.device == 'CPU':
            haplotype_matrix.transfer_to_gpu()
        missing_per_var = cp.sum(haplotype_matrix.haplotypes < 0, axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        if len(valid) == 0:
            return np.full(haplotype_matrix.num_variants, np.nan)
        haplotype_matrix = haplotype_matrix.get_subset(valid)
        if pos is not None:
            pos = np.asarray(pos)[valid.get()]
    m1 = _get_population_matrix(haplotype_matrix, pop1)
    m2 = _get_population_matrix(haplotype_matrix, pop2)

    if m1.device == 'CPU':
        m1.transfer_to_gpu()
    if m2.device == 'CPU':
        m2.transfer_to_gpu()

    if pos is None:
        pos = haplotype_matrix.positions
    if hasattr(pos, 'get'):
        pos = pos.get()
    pos = np.asarray(pos)

    h1 = m1.haplotypes.T
    h2 = m2.haplotypes.T

    # Fall back to matrix's accessible mask if not provided explicitly
    if is_accessible is None and haplotype_matrix.has_accessible_mask:
        is_accessible = haplotype_matrix.accessible_mask.mask
    gaps = _compute_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)
    gaps_gpu = cp.asarray(gaps)

    # forward
    ihh1_fwd = _ihh_scan_gpu(h1, gaps_gpu, min_ehh, include_edges)
    ihh2_fwd = _ihh_scan_gpu(h2, gaps_gpu, min_ehh, include_edges)

    # backward
    ihh1_rev = _ihh_scan_gpu(h1[::-1], gaps_gpu[::-1], min_ehh,
                              include_edges)[::-1]
    ihh2_rev = _ihh_scan_gpu(h2[::-1], gaps_gpu[::-1], min_ehh,
                              include_edges)[::-1]

    ihh1 = ihh1_fwd + ihh1_rev
    ihh2 = ihh2_fwd + ihh2_rev

    score = np.log(ihh1 / ihh2)
    return score


# ---------------------------------------------------------------------------
# Public API: EHH decay
# ---------------------------------------------------------------------------

def ehh_decay(haplotype_matrix: HaplotypeMatrix,
              truncate: bool = False,
              population: Optional[Union[str, list]] = None,
              missing_data: str = 'include'):
    """Compute EHH decay from the first variant.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    truncate : bool, optional
        If True, exclude trailing zeros from the result.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'include' - missing sites skipped in prefix comparison (default)
        'exclude' - filter to sites with no missing data

    Returns
    -------
    ehh : ndarray, float, shape (n_variants,)
        EHH values at each variant position from the first.
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        missing_per_var = matrix.count_missing(axis=0)
        valid = cp.where(missing_per_var == 0)[0]
        if len(valid) < matrix.num_variants:
            matrix = matrix.get_subset(valid)

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    n_hap = hap.shape[0]
    n_variants = hap.shape[1]
    n_pairs = (n_hap * (n_hap - 1)) // 2

    # compute pairwise shared prefix lengths on GPU
    spl = _pairwise_shared_prefix_lengths_gpu(hap)

    # compute EHH from shared prefix lengths
    minlength = 0 if truncate else n_variants + 1
    b = cp.bincount(spl, minlength=minlength)
    c = cp.cumsum(b[::-1])[:-1]
    ehh = (c / n_pairs)[::-1]

    return ehh.get()


# ---------------------------------------------------------------------------
# Private helpers: haplotype frequency computation (for Garud's H)
# ---------------------------------------------------------------------------

def _distinct_haplotype_frequencies(hap):
    """Compute distinct haplotype frequencies, sorted descending.

    Uses GPU dot-product hashing with two random weight vectors for
    collision-free haplotype identification.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_haplotypes, n_variants)

    Returns
    -------
    freqs : ndarray, float64, sorted descending (CPU)
    """
    n_hap, n_var = hap.shape
    rng = cp.random.RandomState(seed=42)
    w1 = rng.standard_normal(n_var, dtype=cp.float32)
    w2 = rng.standard_normal(n_var, dtype=cp.float32)
    h_f32 = hap.astype(cp.float32)
    hash1 = h_f32 @ w1
    hash2 = h_f32 @ w2
    order = cp.lexsort(cp.stack([hash2, hash1]))
    s1 = hash1[order]
    s2 = hash2[order]
    diff = (cp.abs(s1[1:] - s1[:-1]) > 1e-3) | (cp.abs(s2[1:] - s2[:-1]) > 1e-3)
    boundaries = cp.concatenate([cp.array([True]), diff])
    boundary_idx = cp.where(boundaries)[0]
    counts = cp.diff(cp.concatenate([boundary_idx, cp.array([n_hap])]))
    freqs = cp.sort(counts)[::-1].astype(cp.float64) / n_hap
    return freqs.get()


def _distinct_haplotype_frequencies_missing(hap):
    """Compute distinct haplotype frequencies treating -1 as wildcard.

    Uses GPU dot-product hashing when no missing data is present,
    otherwise falls back to CPU wildcard matching.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_haplotypes, n_variants)

    Returns
    -------
    freqs : ndarray, float64, sorted descending (CPU)
    """
    if isinstance(hap, cp.ndarray):
        has_missing = bool(cp.any(hap < 0).get())
    else:
        has_missing = bool(np.any(hap < 0))

    if not has_missing:
        if not isinstance(hap, cp.ndarray):
            hap = cp.asarray(hap)
        return _distinct_haplotype_frequencies(hap)

    # Fallback: wildcard matching on CPU
    from .diversity import _cluster_haplotypes_with_missing
    from collections import Counter

    n_hap = hap.shape[0]
    hap_cpu = hap.get().astype(np.int8) if isinstance(hap, cp.ndarray) else hap

    labels = _cluster_haplotypes_with_missing(hap_cpu)
    counts = Counter(labels)
    freqs = np.array(sorted(counts.values(), reverse=True)) / n_hap
    return freqs


# ---------------------------------------------------------------------------
# CUDA RawKernels for SSL-based scans
# ---------------------------------------------------------------------------

_nsl01_kernel = cp.RawKernel(r'''
extern "C" __global__
void nsl01_scan_kernel(const signed char* h, int n_variants, int n_haplotypes,
                       const int* pair_j, const int* pair_k, int n_pairs,
                       double* ssl_sum_00, double* ssl_sum_11,
                       int* count_00, int* count_11) {
    // Each thread handles one haplotype pair across all variants
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int j = pair_j[pid];
    int k = pair_k[pid];
    int ssl = 0;

    for (int i = 0; i < n_variants; i++) {
        signed char a1 = h[(long long)i * n_haplotypes + j];
        signed char a2 = h[(long long)i * n_haplotypes + k];

        if (a1 < 0 || a2 < 0) {
            ssl += 1;
        } else if (a1 == a2) {
            ssl += 1;
            if (a1 == 0) {
                atomicAdd(&ssl_sum_00[i], (double)ssl);
                atomicAdd(&count_00[i], 1);
            } else {
                atomicAdd(&ssl_sum_11[i], (double)ssl);
                atomicAdd(&count_11[i], 1);
            }
        } else {
            ssl = 0;
        }
    }
}
''', 'nsl01_scan_kernel')


_nsl_kernel = cp.RawKernel(r'''
extern "C" __global__
void nsl_scan_kernel(const signed char* h, int n_variants, int n_haplotypes,
                     const int* pair_j, const int* pair_k, int n_pairs,
                     double* ssl_sum) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int j = pair_j[pid];
    int k = pair_k[pid];
    int ssl = 0;

    for (int i = 0; i < n_variants; i++) {
        signed char a1 = h[(long long)i * n_haplotypes + j];
        signed char a2 = h[(long long)i * n_haplotypes + k];

        if ((a1 != a2) && (a1 >= 0) && (a2 >= 0)) {
            ssl = 0;
        } else {
            ssl += 1;
        }
        atomicAdd(&ssl_sum[i], (double)ssl);
    }
}
''', 'nsl_scan_kernel')


_pair_index_cache = {}


def _get_pair_indices(n_haplotypes):
    """Pre-compute pair indices as contiguous int32 arrays (cached)."""
    if n_haplotypes in _pair_index_cache:
        return _pair_index_cache[n_haplotypes]
    idx_j, idx_k = cp.triu_indices(n_haplotypes, k=1)
    result = idx_j.astype(cp.int32).copy(), idx_k.astype(cp.int32).copy()
    _pair_index_cache[n_haplotypes] = result
    return result


# ---------------------------------------------------------------------------
# Private helpers: SSL-based scans for nSL
# ---------------------------------------------------------------------------

def _nsl01_scan_gpu(h):
    """Forward scan computing mean SSL for ref (0) and alt (1) allele classes.

    Uses a CUDA kernel with one thread per haplotype pair.

    Parameters
    ----------
    h : cupy.ndarray, shape (n_variants, n_haplotypes)

    Returns
    -------
    nsl0 : cupy.ndarray, float64, shape (n_variants,)
    nsl1 : cupy.ndarray, float64, shape (n_variants,)
    """
    n_variants, n_haplotypes = h.shape
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    pair_j, pair_k = _get_pair_indices(n_haplotypes)

    ssl_sum_00 = cp.zeros(n_variants, dtype=cp.float64)
    ssl_sum_11 = cp.zeros(n_variants, dtype=cp.float64)
    count_00 = cp.zeros(n_variants, dtype=cp.int32)
    count_11 = cp.zeros(n_variants, dtype=cp.int32)

    h_contig = cp.ascontiguousarray(h.astype(cp.int8))

    block = 256
    grid = (n_pairs + block - 1) // block

    _nsl01_kernel((grid,), (block,),
                  (h_contig, np.int32(n_variants), np.int32(n_haplotypes),
                   pair_j, pair_k, np.int32(n_pairs),
                   ssl_sum_00, ssl_sum_11, count_00, count_11))

    nsl0 = cp.where(count_00 > 0, ssl_sum_00 / count_00, cp.nan)
    nsl1 = cp.where(count_11 > 0, ssl_sum_11 / count_11, cp.nan)

    return nsl0, nsl1


def _nsl_scan_gpu(h):
    """Forward scan computing mean SSL across all pairs (for cross-pop stats).

    Parameters
    ----------
    h : cupy.ndarray, shape (n_variants, n_haplotypes)

    Returns
    -------
    vnsl : cupy.ndarray, float64, shape (n_variants,)
    """
    n_variants, n_haplotypes = h.shape
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    pair_j, pair_k = _get_pair_indices(n_haplotypes)

    ssl_sum = cp.zeros(n_variants, dtype=cp.float64)

    h_contig = cp.ascontiguousarray(h.astype(cp.int8))

    block = 256
    grid = (n_pairs + block - 1) // block

    _nsl_kernel((grid,), (block,),
                (h_contig, np.int32(n_variants), np.int32(n_haplotypes),
                 pair_j, pair_k, np.int32(n_pairs), ssl_sum))

    vnsl = ssl_sum / n_pairs

    return vnsl


# ---------------------------------------------------------------------------
# Private helpers: gap computation for IHS/XP-EHH
# ---------------------------------------------------------------------------

def _compute_gaps(pos, map_pos=None, gap_scale=20000, max_gap=200000,
                  is_accessible=None):
    """Compute gaps between variants for IHH integration.

    Parameters
    ----------
    pos : ndarray, int, shape (n_variants,)
        Physical positions.
    map_pos : ndarray, float, optional
        Genetic map positions. If None, uses physical positions.
    gap_scale : int
        Rescale gaps larger than this.
    max_gap : int
        Mark gaps larger than this as -1 (invalid).
    is_accessible : ndarray, bool, optional
        Genome accessibility array.

    Returns
    -------
    gaps : ndarray, float64, shape (n_variants - 1,)
    """
    if hasattr(pos, 'get'):
        pos = pos.get()
    pos = np.asarray(pos)

    if map_pos is None:
        map_pos = pos
    if hasattr(map_pos, 'get'):
        map_pos = map_pos.get()
    map_pos = np.asarray(map_pos, dtype='f8')
    gaps = np.diff(map_pos)
    physical_gaps = np.diff(pos)

    if is_accessible is not None:
        is_accessible = np.asarray(is_accessible, dtype=bool)
        accessible_gaps = np.zeros(len(gaps), dtype='f8')
        for i in range(len(gaps)):
            n_access = np.count_nonzero(is_accessible[pos[i]-1:pos[i+1]-1])
            accessible_gaps[i] = n_access
        scaling = accessible_gaps / physical_gaps
        gaps = gaps * scaling

    elif gap_scale is not None and gap_scale > 0:
        scaling = np.ones(gaps.shape, dtype='f8')
        loc_scale = physical_gaps > gap_scale
        scaling[loc_scale] = gap_scale / physical_gaps[loc_scale]
        gaps = gaps * scaling

    if max_gap is not None and max_gap > 0:
        gaps[physical_gaps > max_gap] = -1

    return gaps


# ---------------------------------------------------------------------------
# CUDA kernels for IHH computation (iHS, xpEHH)
# ---------------------------------------------------------------------------
#
# iHS uses fused kernels: one thread block per focal variant, scanning
# backward with block-level reductions. O(n_variants) memory.
#
# xpEHH uses chunked histogram kernels: one thread per pair builds SSL
# histograms, then a separate kernel integrates IHH from histograms.

_ihh01_fused_kernel = cp.RawKernel(r'''
extern "C" __global__
void ihh01_fused_kernel(
    const signed char* h,   // (n_variants, n_haplotypes), row-major
    const double* gaps,      // (n_variants - 1,)
    int n_variants, int n_haplotypes,
    const int* pair_j, const int* pair_k, int n_pairs,
    double min_ehh, double min_maf, int include_edges,
    double* out_ihh0,        // (n_variants,)
    double* out_ihh1         // (n_variants,)
) {
    int vi = blockIdx.x;     // focal variant
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // ── shared state ──────────────────────────────────────────────────
    __shared__ int    s_n0, s_n1;              // allele counts at vi
    __shared__ int    s_n00, s_n11;            // total pairs per class
    __shared__ int    s_ident_00, s_ident_11;  // identical pairs this step
    __shared__ double s_ihh0, s_ihh1;          // accumulating IHH
    __shared__ double s_ehh0_prev, s_ehh1_prev;
    __shared__ int    s_done0, s_done1;        // per-class termination
    __shared__ int    s_stop;                  // global stop

    // ── count alleles at focal variant ────────────────────────────────
    if (tid == 0) { s_n0 = 0; s_n1 = 0; }
    __syncthreads();

    for (int s = tid; s < n_haplotypes; s += nthreads) {
        signed char a = h[(long long)vi * n_haplotypes + s];
        if (a == 0)      atomicAdd(&s_n0, 1);
        else if (a == 1) atomicAdd(&s_n1, 1);
    }
    __syncthreads();

    // ── MAF check ─────────────────────────────────────────────────────
    int total = s_n0 + s_n1;
    if (total == 0 || (double)(s_n0 < s_n1 ? s_n0 : s_n1) / total < min_maf) {
        if (tid == 0) {
            out_ihh0[vi] = 0.0 / 0.0;
            out_ihh1[vi] = 0.0 / 0.0;
        }
        return;
    }

    if (tid == 0) {
        s_n00 = s_n0 * (s_n0 - 1) / 2;
        s_n11 = s_n1 * (s_n1 - 1) / 2;
        s_ihh0 = 0.0;  s_ihh1 = 0.0;
        s_ehh0_prev = 1.0;  s_ehh1_prev = 1.0;
        s_done0 = (s_n00 == 0) ? 1 : 0;
        s_done1 = (s_n11 == 0) ? 1 : 0;
        s_stop  = 0;
    }
    __syncthreads();

    // ── classify pairs and init alive bitmask ─────────────────────────
    // Each thread handles ceil(n_pairs / nthreads) consecutive pairs.
    // We use up to 4 uint32 words = 128 bits per thread as a bitmask.
    int ppthread = (n_pairs + nthreads - 1) / nthreads;
    int p_start = tid * ppthread;
    int p_end   = p_start + ppthread;
    if (p_end > n_pairs) p_end = n_pairs;
    int n_local = p_end - p_start;

    // 4 uint32 words = 128 bits; enough for ppthread <= 128
    unsigned int is_00[4]  = {0,0,0,0};
    unsigned int is_11[4]  = {0,0,0,0};
    unsigned int alive[4]  = {0,0,0,0};

    for (int p = 0; p < n_local; p++) {
        int gp = p_start + p;
        signed char a1 = h[(long long)vi * n_haplotypes + pair_j[gp]];
        signed char a2 = h[(long long)vi * n_haplotypes + pair_k[gp]];
        int w = p >> 5;            // p / 32
        unsigned int bit = 1u << (p & 31);  // p % 32
        if (a1 == 0 && a2 == 0) {
            is_00[w] |= bit;  alive[w] |= bit;
        } else if (a1 == 1 && a2 == 1) {
            is_11[w] |= bit;  alive[w] |= bit;
        }
    }

    // ── backward scan ─────────────────────────────────────────────────
    for (int d = 1; d <= vi; d++) {
        if (s_stop) break;

        int check_vi = vi - d;
        int local_00 = 0, local_11 = 0;

        for (int p = 0; p < n_local; p++) {
            int w = p >> 5;
            unsigned int bit = 1u << (p & 31);
            if (!(alive[w] & bit)) continue;

            int gp = p_start + p;
            signed char a1 = h[(long long)check_vi * n_haplotypes + pair_j[gp]];
            signed char a2 = h[(long long)check_vi * n_haplotypes + pair_k[gp]];

            if (a1 >= 0 && a2 >= 0 && a1 != a2) {
                alive[w] &= ~bit;    // pair diverged
                continue;
            }
            // still identical (match or missing)
            if (is_00[w] & bit) local_00++;
            if (is_11[w] & bit) local_11++;
        }

        // block-level reduction
        if (tid == 0) { s_ident_00 = 0; s_ident_11 = 0; }
        __syncthreads();
        if (local_00) atomicAdd(&s_ident_00, local_00);
        if (local_11) atomicAdd(&s_ident_11, local_11);
        __syncthreads();

        // block leader: EHH + trapezoidal integration
        if (tid == 0) {
            double gap = gaps[vi - d];
            if (gap < 0.0) {
                // invalid gap -> NaN
                if (!s_done0) s_ihh0 = 0.0 / 0.0;
                if (!s_done1) s_ihh1 = 0.0 / 0.0;
                s_done0 = 1; s_done1 = 1; s_stop = 1;
            } else {
                if (!s_done0) {
                    double ehh0 = (double)s_ident_00 / s_n00;
                    s_ihh0 += gap * (ehh0 + s_ehh0_prev) * 0.5;
                    s_ehh0_prev = ehh0;
                    if (ehh0 <= min_ehh) s_done0 = 1;
                }
                if (!s_done1) {
                    double ehh1 = (double)s_ident_11 / s_n11;
                    s_ihh1 += gap * (ehh1 + s_ehh1_prev) * 0.5;
                    s_ehh1_prev = ehh1;
                    if (ehh1 <= min_ehh) s_done1 = 1;
                }
                if (s_done0 && s_done1) s_stop = 1;
            }
        }
        __syncthreads();
    }

    // ── write results ─────────────────────────────────────────────────
    if (tid == 0) {
        out_ihh0[vi] = (s_done0 || include_edges) ? s_ihh0 : (0.0/0.0);
        out_ihh1[vi] = (s_done1 || include_edges) ? s_ihh1 : (0.0/0.0);
    }
}
''', 'ihh01_fused_kernel')


# ---------------------------------------------------------------------------
# Histogram-based IHH (one thread per pair, chunked)
#
# Kernels:
#   _ihh01_ssl_pervar_kernel -- per-variant allele-class histograms (iHS)
#   _ihh_ssl_kernel          -- single histogram for all pairs (xpEHH)
# ---------------------------------------------------------------------------

_ihh01_ssl_pervar_kernel = cp.RawKernel(r'''
extern "C" __global__
void ihh01_ssl_pervar_kernel(
    const signed char* h,     // (total_n_variants, n_haplotypes), full matrix
    int chunk_len,            // number of variants in this chunk
    int n_haplotypes,
    const int* pair_j, const int* pair_k, int n_pairs,
    int* hist00,              // (chunk_len, hist_size) for 0-0 pairs
    int* hist11,              // (chunk_len, hist_size) for 1-1 pairs
    int* ssl_state,           // (n_pairs,) persistent across chunks
    int hist_size,
    int chunk_start           // offset into global variant index
) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int j = pair_j[pid];
    int k = pair_k[pid];
    int ssl = ssl_state[pid];

    for (int ci = 0; ci < chunk_len; ci++) {
        int vi = chunk_start + ci;  // global variant index
        signed char h1 = h[(long long)vi * n_haplotypes + j];
        signed char h2 = h[(long long)vi * n_haplotypes + k];

        if ((h1 != h2) && (h1 >= 0) && (h2 >= 0)) {
            ssl = 0;
        } else {
            ssl += 1;
        }

        int bucket = ssl < hist_size ? ssl : hist_size - 1;

        if (h1 == 0 && h2 == 0) {
            atomicAdd(&hist00[ci * hist_size + bucket], 1);
        } else if (h1 == 1 && h2 == 1) {
            atomicAdd(&hist11[ci * hist_size + bucket], 1);
        }
    }

    ssl_state[pid] = ssl;
}
''', 'ihh01_ssl_pervar_kernel')


_ihh_ssl_kernel = cp.RawKernel(r'''
extern "C" __global__
void ihh_ssl_kernel(const signed char* h, int n_variants, int n_haplotypes,
                    const int* pair_j, const int* pair_k, int n_pairs,
                    int* hist, int* ssl_state, int hist_size) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int j = pair_j[pid];
    int k = pair_k[pid];
    int ssl = ssl_state[pid];

    for (int i = 0; i < n_variants; i++) {
        signed char a1 = h[(long long)i * n_haplotypes + j];
        signed char a2 = h[(long long)i * n_haplotypes + k];

        if ((a1 != a2) && (a1 >= 0) && (a2 >= 0)) {
            ssl = 0;
        } else {
            ssl += 1;
        }
        int bucket = ssl < hist_size ? ssl : hist_size - 1;
        atomicAdd(&hist[i * hist_size + bucket], 1);
    }

    ssl_state[pid] = ssl;
}
''', 'ihh_ssl_kernel')


_ihh_integrate_kernel = cp.RawKernel(r'''
extern "C" __global__
void ihh_integrate_kernel(const int* hist,        // (chunk_len, hist_size)
                          const double* gaps,      // (total_n_variants - 1,)
                          const int* n_pairs_arr,  // (chunk_len,)
                          int chunk_len,
                          int chunk_start,         // offset into global variant index
                          int hist_size,
                          double min_ehh,
                          int include_edges,
                          double* out_ihh) {       // (chunk_len,)
    int ci = blockDim.x * blockIdx.x + threadIdx.x;
    if (ci >= chunk_len) return;

    int n_pairs = n_pairs_arr[ci];
    int variant_idx = chunk_start + ci;  // global variant index

    if (n_pairs == 0) {
        out_ihh[ci] = 0.0 / 0.0;  // NaN
        return;
    }

    double inv_n = 1.0 / (double)n_pairs;

    // pairs with ssl=0 are no longer homozygous at this site
    int n_ident = n_pairs - hist[ci * hist_size + 0];
    double ehh_prv = (double)n_ident * inv_n;

    if (ehh_prv <= min_ehh) {
        out_ihh[ci] = 0.0;
        return;
    }

    double ihh = 0.0;
    int max_steps = variant_idx < hist_size ? variant_idx : hist_size - 1;

    for (int i = 1; i <= max_steps; i++) {
        n_ident -= hist[ci * hist_size + i];
        double ehh_cur = (double)n_ident * inv_n;

        int gap_idx = variant_idx - i;
        double gap = gaps[gap_idx];

        if (gap < 0.0) {
            out_ihh[ci] = 0.0 / 0.0;  // NaN
            return;
        }

        ihh += gap * (ehh_cur + ehh_prv) * 0.5;

        if (ehh_cur <= min_ehh) {
            out_ihh[ci] = ihh;
            return;
        }
        ehh_prv = ehh_cur;
    }

    // If we exhausted the histogram but variant_idx > hist_size,
    // the remaining bins were clamped -- all those pairs have already
    // been subtracted via the last bin. Continue with ehh_prv stable
    // (no more pairs to subtract) until we either hit min_ehh via a
    // gap or run out of variants.
    for (int i = max_steps + 1; i <= variant_idx; i++) {
        // n_ident unchanged (no more histogram bins)
        double ehh_cur = (double)n_ident * inv_n;

        int gap_idx = variant_idx - i;
        double gap = gaps[gap_idx];

        if (gap < 0.0) {
            out_ihh[ci] = 0.0 / 0.0;  // NaN
            return;
        }

        ihh += gap * (ehh_cur + ehh_prv) * 0.5;

        if (ehh_cur <= min_ehh) {
            out_ihh[ci] = ihh;
            return;
        }
        ehh_prv = ehh_cur;
    }

    if (include_edges) {
        out_ihh[ci] = ihh;
    } else {
        out_ihh[ci] = 0.0 / 0.0;  // NaN
    }
}
''', 'ihh_integrate_kernel')


def _integrate_ihh_gpu(hist, gaps, n_pairs_arr, chunk_start, min_ehh,
                       include_edges):
    """Integrate SSL histograms into IHH values on GPU.

    Parameters
    ----------
    hist : cp.ndarray, int32, shape (chunk_len, hist_size)
    gaps : cp.ndarray, float64, shape (total_n_variants - 1,)
    n_pairs_arr : cp.ndarray, int32, shape (chunk_len,)
        Number of pairs per variant (allele-class-specific or total).
    chunk_start : int
        Global variant index offset for this chunk.
    min_ehh : float
    include_edges : bool

    Returns
    -------
    ihh : np.ndarray, float64, shape (chunk_len,)
    """
    chunk_len, hist_size = hist.shape
    out_ihh = cp.empty(chunk_len, dtype=cp.float64)

    hist = cp.ascontiguousarray(hist.astype(cp.int32))
    gaps = cp.ascontiguousarray(gaps)
    n_pairs_arr = cp.ascontiguousarray(n_pairs_arr.astype(cp.int32))

    block = 256
    grid = (chunk_len + block - 1) // block

    _ihh_integrate_kernel((grid,), (block,),
                          (hist, gaps, n_pairs_arr,
                           np.int32(chunk_len), np.int32(chunk_start),
                           np.int32(hist_size), np.float64(min_ehh),
                           np.int32(int(include_edges)), out_ihh))

    return out_ihh.get()


# ---------------------------------------------------------------------------
# Private helpers: SSL-based scans for IHS
# ---------------------------------------------------------------------------

def _ihh01_scan_gpu(h, gaps, min_ehh=0.05, min_maf=0.05,
                    include_edges=False, max_ssl_cap=None):
    """Forward scan computing IHH for ref and alt allele classes.

    Uses a fused kernel: one thread block per focal variant, scanning
    backward with block-level reductions for EHH.  O(n_variants) memory.

    Parameters
    ----------
    h : cupy.ndarray, shape (n_variants, n_haplotypes)
    gaps : cupy.ndarray, float64, shape (n_variants - 1,)
    min_ehh : float
    min_maf : float
    include_edges : bool
    max_ssl_cap : int or None
        Unused (kept for API compatibility). The fused kernel does not
        use histograms.

    Returns
    -------
    ihh0 : np.ndarray, float64, shape (n_variants,)
    ihh1 : np.ndarray, float64, shape (n_variants,)
    """
    n_variants, n_haplotypes = h.shape
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    pair_j, pair_k = _get_pair_indices(n_haplotypes)
    h_contig = cp.ascontiguousarray(h.astype(cp.int8))
    gaps_contig = cp.ascontiguousarray(gaps)

    out_ihh0 = cp.empty(n_variants, dtype=cp.float64)
    out_ihh1 = cp.empty(n_variants, dtype=cp.float64)

    # One block per variant; 256 threads per block.
    # Each thread handles ceil(n_pairs/256) pairs via bitmask.
    block = 256
    _ihh01_fused_kernel(
        (n_variants,), (block,),
        (h_contig, gaps_contig,
         np.int32(n_variants), np.int32(n_haplotypes),
         pair_j, pair_k, np.int32(n_pairs),
         np.float64(min_ehh), np.float64(min_maf),
         np.int32(int(include_edges)),
         out_ihh0, out_ihh1))

    return out_ihh0.get(), out_ihh1.get()


def _ihh01_scan_hist_gpu(h, gaps, min_ehh=0.05, min_maf=0.05,
                         include_edges=False, max_ssl_cap=None):
    """Forward scan computing IHH for ref/alt allele classes via histograms.

    Uses a histogram approach with one thread per pair. Each pair
    maintains its SSL counter and contributes to per-variant allele-class
    histograms. The pair's class at each variant is determined by the
    alleles at that variant (the "focal" for iHS). Scales much better than
    the fused kernel for large haplotype counts (>~300).

    Parameters
    ----------
    h : cupy.ndarray, shape (n_variants, n_haplotypes)
    gaps : cupy.ndarray, float64, shape (n_variants - 1,)
    min_ehh, min_maf : float
    include_edges : bool
    max_ssl_cap : int or None

    Returns
    -------
    ihh0 : np.ndarray, float64, shape (n_variants,)
    ihh1 : np.ndarray, float64, shape (n_variants,)
    """
    n_variants, n_haplotypes = h.shape
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    if max_ssl_cap is None:
        max_ssl_cap = min(n_variants, 10_000)
    hist_size = min(max_ssl_cap + 1, n_variants + 1)

    pair_j, pair_k = _get_pair_indices(n_haplotypes)
    h_contig = cp.ascontiguousarray(h.astype(cp.int8))
    gaps_contig = cp.ascontiguousarray(gaps)

    # Chunk size: target ~2 GB for two histograms
    target_bytes = 2 * 1024**3
    bytes_per_variant = hist_size * 4 * 2  # two int32 histograms
    chunk_size = max(1, target_bytes // bytes_per_variant)
    chunk_size = min(chunk_size, n_variants)

    block = 256
    grid = (n_pairs + block - 1) // block

    ihh0_out = np.empty(n_variants, dtype=np.float64)
    ihh1_out = np.empty(n_variants, dtype=np.float64)
    ssl_state = cp.zeros(n_pairs, dtype=cp.int32)

    for chunk_start in range(0, n_variants, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_variants)
        c_len = chunk_end - chunk_start

        hist00 = cp.zeros((c_len, hist_size), dtype=cp.int32)
        hist11 = cp.zeros((c_len, hist_size), dtype=cp.int32)

        _ihh01_ssl_pervar_kernel((grid,), (block,),
            (h_contig, np.int32(c_len), np.int32(n_haplotypes),
             pair_j, pair_k, np.int32(n_pairs),
             hist00, hist11, ssl_state, np.int32(hist_size),
             np.int32(chunk_start)))

        # Compute per-variant pair counts for each allele class
        h_chunk_alleles = h_contig[chunk_start:chunk_end]  # (c_len, n_hap)
        n0_per_var = cp.sum(h_chunk_alleles == 0, axis=1)
        n1_per_var = cp.sum(h_chunk_alleles == 1, axis=1)
        n00 = (n0_per_var * (n0_per_var - 1) // 2).astype(cp.int32)
        n11 = (n1_per_var * (n1_per_var - 1) // 2).astype(cp.int32)

        ihh0_chunk = _integrate_ihh_gpu(hist00, gaps_contig, n00,
                                         chunk_start, min_ehh, include_edges)
        ihh1_chunk = _integrate_ihh_gpu(hist11, gaps_contig, n11,
                                         chunk_start, min_ehh, include_edges)
        del hist00, hist11

        # Apply MAF filter
        total = (n0_per_var + n1_per_var).astype(cp.float64)
        minor = cp.minimum(n0_per_var, n1_per_var).astype(cp.float64)
        maf_ok = (total > 0) & (minor / cp.maximum(total, 1) >= min_maf)
        maf_mask = maf_ok.get()
        ihh0_chunk[~maf_mask] = np.nan
        ihh1_chunk[~maf_mask] = np.nan

        ihh0_out[chunk_start:chunk_end] = ihh0_chunk
        ihh1_out[chunk_start:chunk_end] = ihh1_chunk

    return ihh0_out, ihh1_out


def _ihh_scan_gpu(h, gaps, min_ehh=0.05, include_edges=False,
                  max_ssl_cap=None):
    """Forward scan computing IHH across all pairs (for cross-pop stats).

    Uses fused kernel for <= 256 haplotypes (fast), falls back to
    chunked histogram approach for larger populations.

    Parameters
    ----------
    h : cupy.ndarray, shape (n_variants, n_haplotypes)
    gaps : cupy.ndarray, float64, shape (n_variants - 1,)
    min_ehh : float
    include_edges : bool
    max_ssl_cap : int or None
        Maximum SSL tracked in histograms. If None, caps at 50K variants
        (shared haplotype segments rarely extend further in real data).

    Returns
    -------
    vihh : np.ndarray, float64, shape (n_variants,)
    """
    n_variants, n_haplotypes = h.shape
    n_pairs = (n_haplotypes * (n_haplotypes - 1)) // 2

    # Histogram-based approach for all haplotype counts
    if max_ssl_cap is None:
        max_ssl_cap = min(n_variants, 10_000)
    hist_size = min(max_ssl_cap + 1, n_variants + 1)

    pair_j, pair_k = _get_pair_indices(n_haplotypes)

    # Determine chunk size: target ~2 GB per chunk for histogram
    target_bytes = 2 * 1024**3
    bytes_per_variant = hist_size * 4  # one histogram, int32
    chunk_size = max(1, target_bytes // bytes_per_variant)
    chunk_size = min(chunk_size, n_variants)

    h_contig = cp.ascontiguousarray(h.astype(cp.int8))
    gaps_contig = cp.ascontiguousarray(gaps)

    block_ssl = 256
    grid_ssl = (n_pairs + block_ssl - 1) // block_ssl

    vihh = np.empty(n_variants, dtype=np.float64)
    ssl_state = cp.zeros(n_pairs, dtype=cp.int32)

    for chunk_start in range(0, n_variants, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_variants)
        c_len = chunk_end - chunk_start

        hist = cp.zeros((c_len, hist_size), dtype=cp.int32)
        h_chunk = h_contig[chunk_start:chunk_end]

        _ihh_ssl_kernel((grid_ssl,), (block_ssl,),
                        (h_chunk, np.int32(c_len), np.int32(n_haplotypes),
                         pair_j, pair_k, np.int32(n_pairs), hist,
                         ssl_state, np.int32(hist_size)))

        n_pairs_arr = cp.full(c_len, n_pairs, dtype=cp.int32)
        ihh_chunk = _integrate_ihh_gpu(hist, gaps_contig, n_pairs_arr,
                                        chunk_start, min_ehh, include_edges)
        del hist

        vihh[chunk_start:chunk_end] = ihh_chunk

    return vihh


# ---------------------------------------------------------------------------
# Private helpers: pairwise shared prefix lengths (for EHH decay)
# ---------------------------------------------------------------------------

_spl_kernel = cp.RawKernel(r'''
extern "C" __global__
void shared_prefix_length_kernel(const signed char* hap,
                                  int n_haplotypes, int n_variants,
                                  const int* pair_j, const int* pair_k,
                                  int n_pairs, int* spl) {
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    if (pid >= n_pairs) return;

    int j = pair_j[pid];
    int k = pair_k[pid];

    for (int v = 0; v < n_variants; v++) {
        signed char a = hap[(long long)j * n_variants + v];
        signed char b = hap[(long long)k * n_variants + v];
        // skip missing (-1): no information, prefix continues
        if (a < 0 || b < 0) continue;
        if (a != b) {
            spl[pid] = v;
            return;
        }
    }
    spl[pid] = n_variants;
}
''', 'shared_prefix_length_kernel')


def _pairwise_shared_prefix_lengths_gpu(hap):
    """Compute shared prefix length for all haplotype pairs.

    Parameters
    ----------
    hap : cupy.ndarray, shape (n_haplotypes, n_variants)

    Returns
    -------
    spl : cupy.ndarray, int32, shape (n_pairs,)
    """
    n_hap, n_var = hap.shape
    n_pairs = (n_hap * (n_hap - 1)) // 2

    pair_j, pair_k = _get_pair_indices(n_hap)
    spl = cp.empty(n_pairs, dtype=cp.int32)

    hap_contig = cp.ascontiguousarray(hap.astype(cp.int8))

    block = 256
    grid = (n_pairs + block - 1) // block

    _spl_kernel((grid,), (block,),
                (hap_contig, np.int32(n_hap), np.int32(n_var),
                 pair_j, pair_k, np.int32(n_pairs), spl))

    return spl


# ---------------------------------------------------------------------------
# Private helpers: binning utility
# ---------------------------------------------------------------------------

def _make_similar_sized_bins(x, n):
    """Create bins with approximately equal numbers of values.

    Parameters
    ----------
    x : array_like
        Values to bin.
    n : int
        Target number of bins.

    Returns
    -------
    bins : ndarray
        Bin edges.
    """
    y = np.array(x).flatten()
    y.sort()

    bins = [y[0]]
    step = len(y) // n

    for i in range(step, len(y), step):
        v = y[i]
        if v > bins[-1]:
            bins.append(v)

    bins[-1] = y[-1]
    return np.array(bins)
