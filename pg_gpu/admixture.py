"""
GPU-accelerated admixture and Patterson F-statistics.

This module provides functions for computing Patterson's F2, F3, and D (F4)
statistics, including windowed and block-jackknife variants.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix
from .resampling import block_jackknife, _moving_nansum, _moving_nanmean


def _aligned_allele_counts(haplotype_matrix, pops):
    """Per-allele counts for a list of populations on a shared global K.

    Returns a list of ``(x, n)`` per pop, where ``x`` is ``(n_var, K)`` allele
    counts (float64) and ``n`` is ``(n_var,)`` valid haplotype counts. K is the
    max allele index over ALL listed pops + 1, so column ``a`` denotes the same
    allele in every pop (the alignment discipline from the joint SFS /
    divergence). This is the counting substrate for the tskit f-statistics.
    """
    from ._memutil import allele_counts
    mats = [_get_population_matrix(haplotype_matrix, p) for p in pops]
    for m in mats:
        if m.device == 'CPU':
            m.transfer_to_gpu()
    k = max((int(m.haplotypes.max()) for m in mats if m.haplotypes.size),
            default=0)
    n_alleles = max(k, 0) + 1
    out = []
    for m in mats:
        x, n = allele_counts(m.haplotypes, n_alleles=n_alleles)
        out.append((x.astype(cp.float64), n.astype(cp.float64)))
    return out


def _f2_terms(xa, na, xb, nb):
    """Per-variant f2(A,B), the tskit unbiased U-statistic summed over alleles.

    x* are (n_var, K) per-allele counts on a shared K; n* are (n_var,) valid
    counts. Matches tskit's ``ts.f2``. Non-estimable sites (n < 2 in either pop)
    contribute 0.
    """
    na_, nb_ = na[:, None], nb[:, None]
    num = (xa * (xa - 1) * (nb_ - xb) * (nb_ - xb - 1)
           - xa * (na_ - xa) * (nb_ - xb) * xb)
    den = na_ * (na_ - 1) * nb_ * (nb_ - 1)
    return cp.where(den > 0, num / den, 0.0).sum(axis=1)


def _f3_terms(xi, ni, xj, nj, xk, nk):
    """Per-variant f3(I; J, K), the tskit unbiased U-statistic over alleles.

    I is the target population (tskit's first sample set). Matches ``ts.f3``.
    """
    ni_, nj_, nk_ = ni[:, None], nj[:, None], nk[:, None]
    num = (xi * (xi - 1) * (nj_ - xj) * (nk_ - xk)
           - xi * (ni_ - xi) * (nj_ - xj) * xk)
    den = ni_ * (ni_ - 1) * nj_ * nk_
    return cp.where(den > 0, num / den, 0.0).sum(axis=1)


def _het_unbiased(x, n):
    """Per-variant unbiased heterozygosity: (n^2 - sum_a x_a^2) / (n(n-1)).

    Per-allele; reduces to the biallelic 2*n0*n1/(n(n-1)) form. Used as the f3
    normalization (B = heterozygosity of the target population).
    """
    sumsq = (x * x).sum(axis=1)
    return cp.where(n > 1, (n * n - sumsq) / (n * (n - 1)), 0.0)


def _f4_terms(xa, na, xb, nb, xc, nc, xd, nd):
    """Per-variant f4(A,B,C,D), the tskit unbiased U-statistic over alleles.

    Matches tskit's ``ts.f4``.
    """
    na_, nb_, nc_, nd_ = na[:, None], nb[:, None], nc[:, None], nd[:, None]
    num = (xa * xc * (nb_ - xb) * (nd_ - xd)
           - xa * xd * (nb_ - xb) * (nc_ - xc))
    den = na_ * nb_ * nc_ * nd_
    return cp.where(den > 0, num / den, 0.0).sum(axis=1)


# ---------------------------------------------------------------------------
# Public API: Per-variant F-statistics
# ---------------------------------------------------------------------------

def patterson_f2(haplotype_matrix: HaplotypeMatrix,
                 pop_a: Union[str, list],
                 pop_b: Union[str, list],
                 missing_data: str = 'include'):
    """Unbiased estimator for F2(A, B), the branch length between populations.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b : str or list
        Population names or sample indices.
    missing_data : str
        'include' - per-site n_valid for frequencies
        'exclude' - filter to sites with no missing data

    Returns
    -------
    f2 : ndarray, float64, shape (n_variants,)
        Per-variant F2 estimates.
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b])
        if haplotype_matrix.num_variants == 0:
            return np.array([])

    (xa, na), (xb, nb) = _aligned_allele_counts(haplotype_matrix,
                                                [pop_a, pop_b])
    return _f2_terms(xa, na, xb, nb).get()


def patterson_f3(haplotype_matrix: HaplotypeMatrix,
                 pop_c: Union[str, list],
                 pop_a: Union[str, list],
                 pop_b: Union[str, list],
                 missing_data: str = 'include'):
    """Unbiased estimator for F3(C; A, B), the three-population admixture test.

    A significantly negative F3 indicates that population C is admixed
    between populations A and B.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c : str or list
        Test population.
    pop_a, pop_b : str or list
        Source populations.
    missing_data : str
        'include' - per-site n_valid
        'exclude' - filter to sites with no missing data

    Returns
    -------
    T : ndarray, float64, shape (n_variants,)
        Un-normalized F3 estimates per variant (matches tskit f3(C; A, B)).
    B : ndarray, float64, shape (n_variants,)
        Unbiased heterozygosity of the target population C (f3 normalization).
    """
    return tuple(v.get() for v in
                 _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                                   missing_data))


def _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                      missing_data='include'):
    """Like patterson_f3 but returns CuPy arrays (no D2H transfer).

    T = tskit f3(C; A, B) per variant (unbiased U-statistic); B = unbiased
    heterozygosity of the target population C (the f3 normalization).
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_c, pop_a, pop_b])
        if haplotype_matrix.num_variants == 0:
            return cp.array([]), cp.array([])

    (xc, nc), (xa, na), (xb, nb) = _aligned_allele_counts(
        haplotype_matrix, [pop_c, pop_a, pop_b])

    T = _f3_terms(xc, nc, xa, na, xb, nb)   # target C is tskit's first set
    B = _het_unbiased(xc, nc)
    return T, B


def patterson_f4(haplotype_matrix: HaplotypeMatrix,
                 pop_a: Union[str, list],
                 pop_b: Union[str, list],
                 pop_c: Union[str, list],
                 pop_d: Union[str, list],
                 missing_data: str = 'include'):
    """Patterson's f4(A, B; C, D) statistic (per variant).

    The unbiased per-allele U-statistic, matching tskit's ``f4`` (site mode).
    A tree-of-populations under (A,B),(C,D) has f4 == 0; a nonzero value
    indicates gene flow.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
        Population names or sample indices.
    missing_data : str
        'include' - per-site n_valid; 'exclude' - filter complete sites.

    Returns
    -------
    f4 : ndarray, float64, shape (n_variants,)
        Per-variant f4 estimates (sum over variants gives the statistic).
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b, pop_c, pop_d])
        if haplotype_matrix.num_variants == 0:
            return np.array([])
    (xa, na), (xb, nb), (xc, nc), (xd, nd) = _aligned_allele_counts(
        haplotype_matrix, [pop_a, pop_b, pop_c, pop_d])
    return _f4_terms(xa, na, xb, nb, xc, nc, xd, nd).get()


def patterson_d(haplotype_matrix: HaplotypeMatrix,
                pop_a: Union[str, list],
                pop_b: Union[str, list],
                pop_c: Union[str, list],
                pop_d: Union[str, list],
                missing_data: str = 'include'):
    """Unbiased estimator for D(A, B; C, D), the ABBA-BABA test.

    Tests for admixture between (A or B) and (C or D).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
        Population names or sample indices.
    missing_data : str
        'include' - per-site n_valid
        'exclude' - filter to sites with no missing data

    Returns
    -------
    num : ndarray, float64, shape (n_variants,)
        Numerator (un-normalized F4 estimates).
    den : ndarray, float64, shape (n_variants,)
        Denominator.

    Notes
    -----
    D is a biallelic-SNP statistic: sites with more than two alleles across the
    four populations are excluded (num = den = 0) and a ``BiallelicOnlyWarning``
    is emitted with the dropped-site count. A multiallelic generalization is a
    possible future extension.
    """
    return tuple(v.get() for v in
                 _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                                  missing_data))


def _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                     missing_data='include'):
    """Like patterson_d but returns CuPy arrays (no D2H transfer).

    BIALLELIC-RESTRICTED: D (ABBA-BABA) is defined on biallelic SNPs; sites with
    >2 alleles across the four pops are excluded (num=den=0). On the retained
    sites the allele frequency is a proper per-allele frequency (count of the
    single alt allele / n), NOT cp.sum(hap)/n -- so a biallelic {0,2}-type site
    is handled correctly, not index-inflated. The multiallelic generalization is
    deferred (needs validation; no reference implementation to pin to).
    """
    if missing_data == 'exclude':
        haplotype_matrix = haplotype_matrix.exclude_missing_sites(
            populations=[pop_a, pop_b, pop_c, pop_d])
        if haplotype_matrix.num_variants == 0:
            return cp.array([]), cp.array([])

    (xa, na), (xb, nb), (xc, nc), (xd, nd) = _aligned_allele_counts(
        haplotype_matrix, [pop_a, pop_b, pop_c, pop_d])

    # Sites with >2 alleles across the four pops are dropped (num=den=0),
    # consistent with the package's biallelic filtering elsewhere (e.g.
    # GenotypeMatrix.from_vcf). Called once per top-level D computation
    # (patterson_d / moving_patterson_d / average_patterson_d all take the full
    # num/den arrays and window afterward), so the warning fires once per call.
    present = (xa + xb + xc + xd) > 0
    biallelic = present.sum(axis=1) <= 2
    from ._warnings import _warn_biallelic_only
    _warn_biallelic_only(int((~biallelic).sum()), context="patterson_d")

    # Alt allele = highest-index present allele (== allele 1 for a {0,1} site,
    # so this reduces to the previous behaviour on standard biallelic data);
    # D is invariant to which of the two alleles is chosen.
    k = xa.shape[1]
    alt = cp.where(present, cp.arange(k)[None, :], -1).argmax(axis=1)[:, None]

    def freq(x, n):
        alt_count = cp.take_along_axis(x, alt, axis=1)[:, 0]
        return cp.where(n > 0, alt_count / n, 0.0)

    a, b, c, d = freq(xa, na), freq(xb, nb), freq(xc, nc), freq(xd, nd)
    num = cp.where(biallelic, (a - b) * (c - d), 0.0)
    den = cp.where(biallelic,
                   (a + b - 2 * a * b) * (c + d - 2 * c * d), 0.0)
    return num, den


# ---------------------------------------------------------------------------
# Public API: Moving window variants
# ---------------------------------------------------------------------------

def moving_patterson_f3(haplotype_matrix: HaplotypeMatrix,
                        pop_c: Union[str, list],
                        pop_a: Union[str, list],
                        pop_b: Union[str, list],
                        size: int,
                        start: int = 0,
                        stop: Optional[int] = None,
                        step: Optional[int] = None,
                        normed: bool = True,
                        missing_data: str = 'include'):
    """Estimate F3(C; A, B) in moving windows.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c, pop_a, pop_b : str or list
    size : int
        Window size (number of variants).
    start, stop, step : int, optional
    normed : bool
        If True, compute normalized F3* per window.
    missing_data : str

    Returns
    -------
    f3 : ndarray, float64, shape (n_windows,)
    """
    T, B = _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                              missing_data=missing_data)

    if normed:
        T_bsum = _moving_nansum(T, size, start, stop, step)
        B_bsum = _moving_nansum(B, size, start, stop, step)
        f3 = cp.where(B_bsum != 0, T_bsum / B_bsum, cp.nan)
    else:
        f3 = _moving_nanmean(T, size, start, stop, step)

    return f3.get()


def moving_patterson_d(haplotype_matrix: HaplotypeMatrix,
                       pop_a: Union[str, list],
                       pop_b: Union[str, list],
                       pop_c: Union[str, list],
                       pop_d: Union[str, list],
                       size: int,
                       start: int = 0,
                       stop: Optional[int] = None,
                       step: Optional[int] = None,
                       missing_data: str = 'include'):
    """Estimate D(A, B; C, D) in moving windows.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
    size : int
    start, stop, step : int, optional
    missing_data : str

    Returns
    -------
    d : ndarray, float64, shape (n_windows,)
    """
    num, den = _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                                missing_data=missing_data)
    num_sum = _moving_nansum(num, size, start, stop, step)
    den_sum = _moving_nansum(den, size, start, stop, step)
    return cp.where(den_sum != 0, num_sum / den_sum, cp.nan).get()


# ---------------------------------------------------------------------------
# Public API: Block-jackknife averaged variants
# ---------------------------------------------------------------------------

def average_patterson_f3(haplotype_matrix: HaplotypeMatrix,
                         pop_c: Union[str, list],
                         pop_a: Union[str, list],
                         pop_b: Union[str, list],
                         blen: int,
                         normed: bool = True,
                         missing_data: str = 'include'):
    """Estimate F3(C; A, B) with standard error via block-jackknife.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_c, pop_a, pop_b : str or list
    blen : int
        Block size (number of variants).
    normed : bool
        If True, compute normalized F3*.
    missing_data : str

    Returns
    -------
    f3 : float
        Overall estimate.
    se : float
        Standard error.
    z : float
        Z-score.
    vb : ndarray
        Per-block values.
    vj : ndarray
        Jackknife resampled values.
    """
    T, B = _patterson_f3_gpu(haplotype_matrix, pop_c, pop_a, pop_b,
                              missing_data=missing_data)

    if normed:
        T_finite = cp.where(cp.isfinite(T), T, 0.0)
        B_finite = cp.where(cp.isfinite(B), B, 0.0)
        f3 = float((cp.sum(T_finite) / cp.sum(B_finite)).get())
        T_bsum = _moving_nansum(T, blen).get()
        B_bsum = _moving_nansum(B, blen).get()
        vb = T_bsum / B_bsum
        _, se, vj = block_jackknife(
            (T_bsum, B_bsum),
            statistic=lambda t, b: np.sum(t) / np.sum(b)
        )
    else:
        finite = cp.isfinite(T)
        f3 = float((cp.sum(cp.where(finite, T, 0.0)) / cp.sum(finite)).get())
        vb = _moving_nanmean(T, blen).get()
        _, se, vj = block_jackknife(vb, statistic=np.mean)

    z = f3 / se
    return f3, se, z, vb, vj


def average_patterson_d(haplotype_matrix: HaplotypeMatrix,
                        pop_a: Union[str, list],
                        pop_b: Union[str, list],
                        pop_c: Union[str, list],
                        pop_d: Union[str, list],
                        blen: int,
                        missing_data: str = 'include'):
    """Estimate D(A, B; C, D) with standard error via block-jackknife.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    pop_a, pop_b, pop_c, pop_d : str or list
    blen : int
        Block size (number of variants).
    missing_data : str

    Returns
    -------
    d : float
        Overall estimate.
    se : float
        Standard error.
    z : float
        Z-score.
    vb : ndarray
        Per-block values.
    vj : ndarray
        Jackknife resampled values.
    """
    num, den = _patterson_d_gpu(haplotype_matrix, pop_a, pop_b, pop_c, pop_d,
                                missing_data=missing_data)

    num_f = cp.where(cp.isfinite(num), num, 0.0)
    den_f = cp.where(cp.isfinite(den), den, 0.0)
    d_avg = float((cp.sum(num_f) / cp.sum(den_f)).get())

    num_bsum = _moving_nansum(num, blen).get()
    den_bsum = _moving_nansum(den, blen).get()
    vb = num_bsum / den_bsum

    _, se, vj = block_jackknife(
        (num_bsum, den_bsum),
        statistic=lambda n, d: np.sum(n) / np.sum(d)
    )

    z = d_avg / se
    return d_avg, se, z, vb, vj
