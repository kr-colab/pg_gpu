"""
GPU-accelerated diversity and polymorphism statistics.

This module provides efficient computation of within-population genetic diversity
metrics including nucleotide diversity (π), Watterson's theta, Tajima's D, and
related statistics. Includes the FrequencySpectrum class for SFS-based analysis,
custom weight functions, and SFS projection.
"""

import math
import numpy as np
import cupy as cp
from typing import Union, Optional, Dict, Callable
from functools import lru_cache
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix


def _apply_span_normalize(value, matrix, span_normalize):
    """Apply span normalization to a raw statistic value.

    Parameters
    ----------
    value : float or cupy scalar
        Raw statistic sum.
    matrix : HaplotypeMatrix
        Source matrix (for get_span).
    span_normalize : bool or str
        True: auto-detect best span. False: return raw value.
        String: explicit mode passed to get_span() (internal use).
    """
    if span_normalize is False:
        return float(value.get() if hasattr(value, 'get') else value)
    mode = 'auto' if span_normalize is True else span_normalize
    span = matrix.get_span(mode)
    if span > 0:
        return float(value / span)
    return float('nan')


_gpu_lookup_cache = {}


def _get_a1_inv(n_max):
    """Get cached 1/a1(n) lookup array on GPU."""
    key = ('a1_inv', n_max)
    if key in _gpu_lookup_cache:
        return _gpu_lookup_cache[key]
    a1 = np.zeros(n_max + 1, dtype=np.float64)
    for n in range(2, n_max + 1):
        a1[n] = 1.0 / np.sum(1.0 / np.arange(1, n, dtype=np.float64))
    result = cp.asarray(a1)
    _gpu_lookup_cache[key] = result
    return result


def _get_minus_eta1_norm(n_max):
    """Get cached 1/(a1-1) normalizer for minus_eta1 estimator on GPU."""
    key = ('minus_eta1', n_max)
    if key in _gpu_lookup_cache:
        return _gpu_lookup_cache[key]
    arr = np.zeros(n_max + 1, dtype=np.float64)
    for ni in range(3, n_max + 1):
        a1 = np.sum(1.0 / np.arange(1, ni, dtype=np.float64))
        arr[ni] = 1.0 / (a1 - 1.0) if a1 > 1.0 else 0.0
    result = cp.asarray(arr)
    _gpu_lookup_cache[key] = result
    return result


def _get_minus_eta1_star_norm(n_max):
    """Get cached normalizer for minus_eta1_star estimator on GPU."""
    key = ('minus_eta1_star', n_max)
    if key in _gpu_lookup_cache:
        return _gpu_lookup_cache[key]
    arr = np.zeros(n_max + 1, dtype=np.float64)
    for ni in range(4, n_max + 1):
        a1 = np.sum(1.0 / np.arange(1, ni, dtype=np.float64))
        denom = a1 - 1.0 - 1.0 / (ni - 1)
        arr[ni] = 1.0 / denom if denom > 0 else 0.0
    result = cp.asarray(arr)
    _gpu_lookup_cache[key] = result
    return result


def _achaz_alpha_beta(v1, v2, n):
    """Compute Achaz (2009) Eq. 9 alpha_n, beta_n from two per-xi weight vectors.

    O(n) time and memory — computes the bilinear form w^T @ sigma @ w
    directly using the structure of Fu (1995) sigma_ij, without
    materializing the full (n-1)x(n-1) covariance matrix.

    Our weight functions return per-xi weights w[i]. Achaz's v-vectors are
    per-u_hat weights where u_hat_i = i * xi_i. The conversion is
    v_i/sum(v_j) = w[i]/i, giving V_i = w1[i]/i - w2[i]/i.
    """
    H = cp.asarray(_harmonic_sums(n))
    an = H[n - 1]  # CuPy scalar — stays on GPU

    k = cp.arange(1, n, dtype=cp.float64)
    k_int = k.astype(cp.int64)
    V = cp.asarray((v1[1:n] - v2[1:n])) / k
    w = k * V

    alpha_n = cp.sum(k * V ** 2)

    # Precompute beta(i) for i=1..n-1 (reuse k instead of separate idx_b)
    a_b = H[k_int - 1]
    beta_vals = cp.zeros(n + 1, dtype=cp.float64)
    beta_vals[1:n] = (2.0 * n / ((n - k + 1) * (n - k))
                      * (an + 1.0 / n - a_b) - 2.0 / (n - k))

    # --- Diagonal: sum w_i^2 * sigma_{ii} ---
    # Clip indices protect against out-of-bounds; valid masks ensure
    # only correct contributions are summed.
    diag_sigma = cp.where(
        2 * k < n, beta_vals[k_int + 1],
        cp.where(2 * k == n,
                 2.0 * (an - H[k_int - 1]) / (n - k) - 1.0 / (k * k),
                 beta_vals[k_int] - 1.0 / (k * k)))
    diag_sum = cp.sum(w ** 2 * diag_sigma)

    # --- Prefix/suffix sums for off-diagonal ---
    W_prefix = cp.cumsum(w)
    W_suffix = cp.flip(cp.cumsum(cp.flip(w)))
    V_suffix = cp.flip(cp.cumsum(cp.flip(V)))

    # Case A: i < j, i+j < n => sigma = (beta(j+1) - beta(j)) / 2
    j_idx = cp.arange(2, n, dtype=cp.int64)
    dbeta = beta_vals[j_idx + 1] - beta_vals[j_idx]
    upper = cp.minimum(j_idx - 1, n - j_idx - 1)
    valid_a = upper >= 1
    psum = cp.where(valid_a,
                    W_prefix[cp.clip(upper - 1, 0, n - 2).astype(cp.int64)], 0.0)
    case_a = cp.sum(dbeta * w[j_idx - 1] * psum * valid_a)

    # Case C: i < j, i+j > n => sigma = (beta(i)-beta(i+1))/2 - 1/(i*j)
    i_idx = cp.arange(1, n - 1, dtype=cp.int64)
    dbeta_i = beta_vals[i_idx] - beta_vals[i_idx + 1]
    j_start = cp.maximum(i_idx + 1, n - i_idx + 1)
    valid_c = j_start <= n - 1
    ws = cp.where(valid_c,
                  W_suffix[cp.clip(j_start - 1, 0, n - 2).astype(cp.int64)], 0.0)
    case_c1 = cp.sum(dbeta_i * w[i_idx - 1] * ws * valid_c)
    vs = cp.where(valid_c,
                  V_suffix[cp.clip(j_start - 1, 0, n - 2).astype(cp.int64)], 0.0)
    case_c2 = cp.sum(V[i_idx - 1] * vs * valid_c)

    # Case B: i+j == n (anti-diagonal, O(n) terms)
    i_b = cp.arange(1, n, dtype=cp.int64)
    j_b = n - i_b
    valid_b = (j_b > 0) & (j_b < n) & (j_b > i_b)
    ai = H[cp.clip(i_b - 1, 0, n).astype(cp.int64)]
    aj = H[cp.clip(j_b - 1, 0, n).astype(cp.int64)]
    j_f = j_b.astype(cp.float64)
    i_f = i_b.astype(cp.float64)
    s_b = ((an - aj) / (n - j_f) + (an - ai) / (n - i_f)
           - (beta_vals[j_b] + beta_vals[cp.clip(i_b + 1, 0, n)]) / 2.0
           - 1.0 / (j_f * i_f))
    case_b = cp.sum(
        2.0 * w[i_b - 1] * w[cp.clip(j_b - 1, 0, n - 2)] * s_b * valid_b)

    # Single GPU->CPU transfer at the end
    beta_n = diag_sum + case_a + case_c1 - 2.0 * case_c2 + case_b
    return float(alpha_n), float(beta_n)


@lru_cache(maxsize=128)
def _achaz_variance_coefficients(w1_name, w2_name, n):
    """Cached Achaz (2009) Eq. 9 variance coefficients for named weight pairs.

    This is the single source of truth for all neutrality test variances.
    """
    v1 = WEIGHT_REGISTRY[w1_name](n)
    v2 = WEIGHT_REGISTRY[w2_name](n)
    return _achaz_alpha_beta(v1, v2, n)


def _achaz_variance(w1_name, w2_name, n, S):
    """Compute the Achaz (2009) null variance for a neutrality test.

    Var(T) = alpha_n * theta_est + beta_n * theta_sq_est
    where theta_est = S/a1 and theta_sq_est = S(S-1)/(a1^2+a2).
    """
    alpha_n, beta_n = _achaz_variance_coefficients(w1_name, w2_name, n)
    a1, a2 = _harmonic_a1_a2(n)
    return alpha_n * S / a1 + beta_n * S * (S - 1) / (a1 ** 2 + a2)


def _site_contribution(name, d, n_safe, seg, n_valid, n_hap, dac=None):
    """Compute per-site contribution for a theta estimator on GPU.

    This is the single source of truth for what each estimator computes.
    Both scalar (_compute_thetas) and windowed (_windowed_thetas_scatter)
    paths call this function.

    Parameters
    ----------
    name : str
        Estimator name.
    d, n_safe : cupy.ndarray, float64
        Derived allele count (float) and safe sample size per site.
    seg : cupy.ndarray, bool
        Segregating site mask.
    n_valid : cupy.ndarray, int64
        Per-site valid sample count (for watterson lookup).
    n_hap : int
        Total haplotype count (for harmonic number lookup).
    dac : cupy.ndarray, int64, optional
        Integer derived allele count (for exact comparisons like == 1).
        If None, uses d cast to int64.

    Returns
    -------
    cupy.ndarray, float64, shape (n_variants,)
        Per-site contribution (zero for non-segregating sites).
    """
    if dac is None:
        dac = d.astype(cp.int64)

    if name in ('pi', 'theta_pi'):
        return cp.where(seg, 2 * d * (n_safe - d) / (n_safe * (n_safe - 1)), 0.0)
    elif name in ('watterson', 'theta_s'):
        a1_inv = _get_a1_inv(n_hap)
        return cp.where(seg, a1_inv[n_valid], 0.0)
    elif name == 'theta_h':
        return cp.where(seg, 2 * d * d / (n_safe * (n_safe - 1)), 0.0)
    elif name == 'theta_l':
        return cp.where(seg, d / (n_safe - 1), 0.0)
    elif name == 'eta1':
        # Singletons only: dac == 1
        a1_inv = _get_a1_inv(n_hap)
        return cp.where(seg & (dac == 1), a1_inv[n_valid], 0.0)
    elif name == 'eta1_star':
        # Singletons + (n-1)-tons
        a1_inv = _get_a1_inv(n_hap)
        is_edge = (dac == 1) | (dac == n_valid - 1)
        return cp.where(seg & is_edge, a1_inv[n_valid], 0.0)
    elif name == 'minus_eta1':
        not_sing = dac >= 2
        a1m1_gpu = _get_minus_eta1_norm(n_hap)
        return cp.where(seg & not_sing, a1m1_gpu[n_valid], 0.0)
    elif name == 'minus_eta1_star':
        interior = (dac >= 2) & (dac <= n_valid - 2)
        norm_gpu = _get_minus_eta1_star_norm(n_hap)
        return cp.where(seg & interior, norm_gpu[n_valid], 0.0)
    else:
        raise ValueError(f"Unknown estimator: {name}. Use FrequencySpectrum "
                         f"for custom weight functions.")


def _prepare_dac(matrix):
    """Compute dac, n_valid, and derived quantities on GPU.

    Returns (dac, n_valid, d, n_safe, seg, n_hap) — the shared
    intermediate arrays used by all theta estimator paths.
    """
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    from ._memutil import dac_and_n
    dac, n_valid = dac_and_n(matrix.haplotypes)
    n = n_valid.astype(cp.float64)
    d = dac.astype(cp.float64)
    seg = (dac > 0) & (dac < n_valid) & (n_valid >= 2)
    n_safe = cp.maximum(n, 2.0)
    return dac, n_valid, d, n_safe, seg, matrix.num_haplotypes


def _compute_thetas(matrix, estimators=('pi', 'watterson', 'theta_h', 'theta_l')):
    """Compute multiple theta estimators via direct vectorized GPU arithmetic.

    Parameters
    ----------
    matrix : HaplotypeMatrix
        Population-subsetted, on GPU.
    estimators : tuple of str
        Estimator names.

    Returns
    -------
    dict with keys:
        'thetas': dict of estimator name -> float (raw sum)
        'S': int, number of segregating sites
        'n_harmonic_mean': int, harmonic mean of per-site sample sizes
    """
    dac, n_valid, d, n_safe, seg, n_hap = _prepare_dac(matrix)

    thetas = {}
    for name in estimators:
        val = cp.sum(_site_contribution(name, d, n_safe, seg, n_valid, n_hap, dac=dac))
        thetas[name] = float(val.get())

    S = int(cp.sum(seg).get())

    has_data = n_valid >= 2
    if cp.any(has_data):
        valid_n = n_valid[has_data].astype(cp.float64)
        n_harm = round(float(len(valid_n) / cp.sum(1.0 / valid_n).get()))
    else:
        n_harm = 0

    return {'thetas': thetas, 'S': S, 'n_harmonic_mean': n_harm}


def _compute_neutrality_test(matrix, w1_name, w2_name):
    """Compute a neutrality test statistic using the Achaz (2009) framework.

    T = (theta_w1 - theta_w2) / sqrt(alpha_n * theta_est + beta_n * theta_sq_est)

    Parameters
    ----------
    matrix : HaplotypeMatrix
        Population-subsetted, on GPU.
    w1_name, w2_name : str
        Weight vector names from WEIGHT_REGISTRY.

    Returns
    -------
    float
    """
    result = _compute_thetas(matrix, (w1_name, w2_name))
    S = result['S']
    n = result['n_harmonic_mean']
    if S < 3 or n < 3:
        return float('nan')
    var = _achaz_variance(w1_name, w2_name, n, S)
    if var <= 0:
        return float('nan')
    num = result['thetas'][w1_name] - result['thetas'][w2_name]
    return float(num / math.sqrt(var))


def _prepare_matrix(haplotype_matrix, population=None, missing_data='include'):
    """Extract population subset and apply exclude filtering."""
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()
    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
    return matrix


# ---------------------------------------------------------------------------
# SFS projection and covariance (Gutenkunst et al. 2009, Fu 1995)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _harmonic_sums(n):
    """Precompute harmonic sums H[i] = sum(1/j for j=1..i) for i=0..n."""
    H = np.zeros(n + 1)
    H[1:] = np.cumsum(1.0 / np.arange(1, n + 1))
    return H


@lru_cache(maxsize=128)
def _harmonic_a1_a2(n):
    """Return (a1, a2) harmonic number sums for sample size n.

    a1 = sum(1/i for i=1..n-1), a2 = sum(1/i^2 for i=1..n-1).
    """
    k = np.arange(1, n, dtype=np.float64)
    return float(np.sum(1.0 / k)), float(np.sum(1.0 / (k * k)))


@lru_cache(maxsize=64)
def _compute_sigma_ij_gpu(n):
    """Compute Fu (1995) sigma_ij on GPU, return CuPy array (cached)."""
    H = _harmonic_sums(n)
    an = H[n - 1]

    # beta(i, n) for i = 1..n-1 (CPU, 1D)
    idx_b = np.arange(1, n, dtype=np.float64)
    a_b = H[idx_b.astype(int) - 1]
    beta_arr = (2.0 * n / ((n - idx_b + 1) * (n - idx_b))
                * (an + 1.0 / n - a_b)
                - 2.0 / (n - idx_b))
    beta_full = np.zeros(n + 1)
    beta_full[1:n] = beta_arr

    # O(n^2) broadcast on GPU
    beta_gpu = cp.asarray(beta_full)
    H_gpu = cp.asarray(H)

    idx = cp.arange(1, n, dtype=cp.float64)
    ii = idx[:, None]
    jj = idx[None, :]
    i_hi = cp.maximum(ii, jj)
    j_lo = cp.minimum(ii, jj)
    s = i_hi + j_lo

    i_hi_int = i_hi.astype(cp.int64)
    j_lo_int = j_lo.astype(cp.int64)

    case_lt = (beta_gpu[i_hi_int + 1] - beta_gpu[i_hi_int]) / 2.0
    ai_hi = H_gpu[i_hi_int - 1]
    aj_lo = H_gpu[j_lo_int - 1]
    case_eq = ((an - ai_hi) / (n - i_hi) + (an - aj_lo) / (n - j_lo)
               - (beta_gpu[i_hi_int] + beta_gpu[j_lo_int + 1]) / 2.0
               - 1.0 / (i_hi * j_lo))
    case_gt = ((beta_gpu[j_lo_int] - beta_gpu[j_lo_int + 1]) / 2.0
               - 1.0 / (i_hi * j_lo))

    sigma = cp.where(s < n, case_lt, cp.where(s == n, case_eq, case_gt))

    # Diagonal
    i_d = idx
    i_d_int = i_d.astype(cp.int64)
    diag_vals = cp.where(
        2 * i_d < n, beta_gpu[i_d_int + 1],
        cp.where(2 * i_d == n,
                 2.0 * (an - H_gpu[i_d_int - 1]) / (n - i_d) - 1.0 / (i_d * i_d),
                 beta_gpu[i_d_int] - 1.0 / (i_d * i_d)))
    d_idx = cp.arange(n - 1)
    sigma[d_idx, d_idx] = diag_vals

    return sigma


def compute_sigma_ij(n):
    """Compute the Fu (1995) covariance matrix sigma_ij for sample size n.

    sigma_ij = Cov[xi_i, xi_j] / theta^2 for the unfolded SFS under the
    standard neutral model.

    Parameters
    ----------
    n : int
        Sample size (number of haplotypes).

    Returns
    -------
    sigma : ndarray, float64, shape (n-1, n-1)
    """
    return _compute_sigma_ij_gpu(n).get()


@lru_cache(maxsize=128)
def _projection_matrix(n_from, n_to):
    """Hypergeometric projection matrix from n_from to n_to."""
    from scipy.special import comb
    P = np.zeros((n_to + 1, n_from + 1))
    for k_from in range(n_from + 1):
        for k_to in range(max(0, k_from - (n_from - n_to)),
                          min(k_from, n_to) + 1):
            P[k_to, k_from] = (comb(k_from, k_to, exact=True)
                               * comb(n_from - k_from, n_to - k_to, exact=True)
                               / comb(n_from, n_to, exact=True))
    return P


def project_sfs(sfs, n_from, n_to):
    """Project an SFS from sample size n_from down to n_to.

    Uses hypergeometric sampling (Gutenkunst et al. 2009).
    """
    if n_to > n_from:
        raise ValueError(f"Cannot project up: n_to={n_to} > n_from={n_from}")
    if n_to == n_from:
        return sfs.copy()
    return _projection_matrix(n_from, n_to) @ sfs


# ---------------------------------------------------------------------------
# FrequencySpectrum: power-user class for SFS analysis
# ---------------------------------------------------------------------------

# Weight functions for SFS dot-product path (used by FrequencySpectrum.theta
# for custom callables; built-in names use _site_contribution instead).
def _weights_watterson(n):
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1:n] = 1.0 / a1
    return w

def _weights_pi(n):
    k = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * k * (n - k) / (n * (n - 1))
    w[0] = w[n] = 0.0
    return w

def _weights_theta_h(n):
    k = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * k ** 2 / (n * (n - 1))
    w[0] = w[n] = 0.0
    return w

def _weights_theta_l(n):
    k = np.arange(n + 1, dtype=np.float64)
    w = k / (n - 1)
    w[0] = w[n] = 0.0
    return w

def _weights_eta1(n):
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1] = 1.0 / a1
    return w

def _weights_eta1_star(n):
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1] = w[n - 1] = 1.0 / a1
    return w

def _weights_minus_eta1(n):
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[2:n] = 1.0 / (a1 - 1.0)
    return w

def _weights_minus_eta1_star(n):
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[2:n - 1] = 1.0 / (a1 - 1.0 - 1.0 / (n - 1))
    return w


WEIGHT_REGISTRY: Dict[str, Callable] = {
    'watterson': _weights_watterson, 'theta_s': _weights_watterson,
    'pi': _weights_pi, 'theta_pi': _weights_pi,
    'theta_h': _weights_theta_h,
    'theta_l': _weights_theta_l,
    'eta1': _weights_eta1,
    'eta1_star': _weights_eta1_star,
    'minus_eta1': _weights_minus_eta1,
    'minus_eta1_star': _weights_minus_eta1_star,
}


class FrequencySpectrum:
    """Site frequency spectrum with support for variable sample sizes.

    Computes derived allele counts on GPU, groups by per-site sample
    size, and provides theta estimation via weight vector dot products.
    For built-in estimators, use the scalar functions (``pi()``, etc.)
    which are faster. This class is for custom weight functions,
    SFS inspection, projection, and the general Achaz variance framework.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' (default) or 'exclude'
    n_total_sites : int, optional
        Total callable sites for invariant site correction.
    """

    def __init__(self, haplotype_matrix, population=None,
                 missing_data='include', n_total_sites=None):
        if population is not None:
            matrix = get_population_matrix(haplotype_matrix, population)
        else:
            matrix = haplotype_matrix

        if matrix.device == 'CPU':
            matrix.transfer_to_gpu()
        self._source_matrix = matrix
        n_hap = matrix.num_haplotypes
        if n_total_sites is None:
            n_total_sites = matrix.n_total_sites

        from ._memutil import dac_and_n as _dac_n
        dac, n_valid = _dac_n(matrix.haplotypes)

        if missing_data == 'exclude':
            complete = n_valid == n_hap
            dac = dac[complete]
            n_valid = n_valid[complete]

        self.sfs_by_n = {}
        if len(dac) == 0:
            self.n_max = 0
            self.n_segregating = 0
        else:
            unique_n = cp.unique(n_valid)
            self.n_max = int(unique_n[-1].get())
            for ni_gpu in unique_n:
                ni = int(ni_gpu.get())
                if ni < 2:
                    continue
                mask = n_valid == ni
                xi = cp.bincount(dac[mask], minlength=ni + 1)[:ni + 1]
                self.sfs_by_n[ni] = xi.astype(cp.float64).get()
            self.n_segregating = sum(
                int(np.sum(xi[1:n])) for n, xi in self.sfs_by_n.items())

        self.n_total_sites = n_total_sites
        if n_total_sites is not None and self.n_max > 0:
            n_invariant = n_total_sites - self.n_segregating
            if n_invariant > 0 and self.n_max in self.sfs_by_n:
                self.sfs_by_n[self.n_max][0] += n_invariant

    def theta(self, weights='pi', span_normalize=False, span=None):
        """Compute a theta estimator from the SFS.

        Parameters
        ----------
        weights : str or callable
            Name of a built-in weight function, or a callable w(n) -> array.
        span_normalize : bool
        span : float, optional
        """
        if isinstance(weights, str):
            if weights not in WEIGHT_REGISTRY:
                raise ValueError(f"Unknown weight: {weights}")
            weights_fn = WEIGHT_REGISTRY[weights]
        else:
            weights_fn = weights

        total = 0.0
        for n, xi in self.sfs_by_n.items():
            w = weights_fn(n)
            total += np.sum(xi[:len(w)] * w[:len(xi)])

        if span_normalize is not False:
            if span is not None and span > 0:
                total /= span
            elif self._source_matrix is not None:
                mode = 'auto' if span_normalize is True else span_normalize
                s = self._source_matrix.get_span(mode)
                if s > 0:
                    total /= s

        return total

    def neutrality_test(self, w1='pi', w2='watterson'):
        """Compute T = (theta1 - theta2) / sqrt(var) using Achaz (2009) Eq. 9."""
        theta1 = self.theta(w1)
        theta2 = self.theta(w2)
        numerator = theta1 - theta2

        S = self.n_segregating
        if S < 3:
            return float('nan')

        n_eff = max(self.sfs_by_n.keys(),
                    key=lambda n: np.sum(self.sfs_by_n[n]))

        w1_name = w1 if isinstance(w1, str) else None
        w2_name = w2 if isinstance(w2, str) else None

        if w1_name and w2_name:
            variance = _achaz_variance(w1_name, w2_name, n_eff, S)
        else:
            w1_fn = WEIGHT_REGISTRY[w1] if isinstance(w1, str) else w1
            w2_fn = WEIGHT_REGISTRY[w2] if isinstance(w2, str) else w2
            alpha_n, beta_n = _achaz_alpha_beta(w1_fn(n_eff), w2_fn(n_eff), n_eff)
            a1, a2 = _harmonic_a1_a2(n_eff)
            variance = alpha_n * S / a1 + beta_n * S * (S - 1) / (a1 ** 2 + a2)

        if variance <= 0:
            return float('nan')
        return numerator / math.sqrt(variance)

    def suggest_projection_n(self, retain_fraction=0.95):
        """Suggest a projection target retaining most sites."""
        if len(self.sfs_by_n) <= 1:
            return self.n_max
        sorted_ns = sorted(self.sfs_by_n.keys(), reverse=True)
        total_seg = self.n_segregating
        if total_seg == 0:
            return self.n_max
        cumulative = 0
        for ni in sorted_ns:
            cumulative += int(np.sum(self.sfs_by_n[ni][1:ni]))
            if cumulative / total_seg >= retain_fraction:
                return ni
        return sorted_ns[-1]

    def project(self, target_n):
        """Project all SFS groups to a common sample size."""
        projected = np.zeros(target_n + 1)
        for n, xi in self.sfs_by_n.items():
            if n < target_n:
                continue
            projected += project_sfs(xi, n, target_n)
        result = object.__new__(FrequencySpectrum)
        result.sfs_by_n = {target_n: projected}
        result.n_max = target_n
        result.n_segregating = int(np.sum(projected[1:target_n]))
        result.n_total_sites = self.n_total_sites
        result._source_matrix = self._source_matrix
        return result

    def sfs(self, n=None):
        """Return the SFS, optionally projected."""
        if n is not None:
            return self.project(n).sfs_by_n[n]
        if len(self.sfs_by_n) == 1:
            return list(self.sfs_by_n.values())[0]
        return self.sfs_by_n.get(self.n_max, np.array([]))

    def all_thetas(self, span_normalize=False, span=None):
        """Compute all 8 standard theta estimators."""
        return {name: self.theta(name, span_normalize=span_normalize, span=span)
                for name in ['pi', 'watterson', 'theta_h', 'theta_l',
                             'eta1', 'eta1_star', 'minus_eta1', 'minus_eta1_star']}

    def tajimas_d(self):
        """Tajima's D via Achaz (2009) general variance framework."""
        return self.neutrality_test('pi', 'watterson')

    def fay_wu_h(self, normalized=False):
        """Fay & Wu's H = pi - theta_H. Optionally normalized (H*)."""
        h = self.theta('pi') - self.theta('theta_h')
        if not normalized:
            return h
        return self.neutrality_test('pi', 'theta_h')

    def zeng_e(self):
        """Zeng's E via Achaz (2009) general variance framework."""
        return self.neutrality_test('theta_l', 'watterson')

    def all_tests(self):
        """All standard neutrality tests."""
        return {
            'tajimas_d': self.tajimas_d(),
            'fay_wu_h': self.fay_wu_h(),
            'normalized_fay_wu_h': self.fay_wu_h(normalized=True),
            'zeng_e': self.zeng_e(),
        }


# ---------------------------------------------------------------------------
# Pairwise components (power-user API)
# ---------------------------------------------------------------------------

def pi_components(haplotypes, n_total_sites=None, n_haplotypes_full=None):
    """Compute pairwise differences and comparisons across all sites.

    For advanced use cases (custom windowed aggregation, etc.).

    Parameters
    ----------
    haplotypes : cp.ndarray, shape (n_haplotypes, n_variants)
        Haplotype data with -1 for missing.
    n_total_sites : int, optional
        Total callable sites (variant + invariant). If provided, invariant
        sites contribute 0 diffs and C(n_haplotypes_full, 2) comps each.
    n_haplotypes_full : int, optional
        Full sample size (used for invariant site comps).

    Returns
    -------
    total_diffs : float
    total_comps : float
    total_missing : float
    n_sites : int
    """
    dac, n_valid_i = _dac_and_n(haplotypes)
    n_valid = n_valid_i.astype(cp.float64)
    derived = dac.astype(cp.float64)
    ancestral = n_valid - derived

    site_diffs = derived * ancestral
    site_comps = n_valid * (n_valid - 1) / 2.0

    usable = n_valid >= 2
    total_diffs = float(cp.sum(site_diffs[usable]).get())
    total_comps = float(cp.sum(site_comps[usable]).get())
    n_sites = int(cp.sum(usable).get())

    if n_total_sites is not None:
        n_full = n_haplotypes_full or haplotypes.shape[0]
        n_invariant = n_total_sites - n_sites
        if n_invariant > 0:
            total_comps += n_invariant * (n_full * (n_full - 1) / 2.0)
            n_sites += n_invariant

    n_full = n_haplotypes_full or haplotypes.shape[0]
    total_possible = (n_full * (n_full - 1) / 2.0) * n_sites
    total_missing = total_possible - total_comps

    return total_diffs, total_comps, total_missing, n_sites


def _dac_and_n(haplotypes):
    """Shared helper: derived allele counts and valid sample counts per site.

    Uses adaptive chunking from _memutil for memory safety on large matrices.

    Parameters
    ----------
    haplotypes : cupy.ndarray, int8, shape (n_hap, n_var)

    Returns
    -------
    dac : cupy.ndarray, int64, shape (n_var,)
    n_valid : cupy.ndarray, int64, shape (n_var,)
    """
    from ._memutil import dac_and_n
    return dac_and_n(haplotypes)



def pi(haplotype_matrix: HaplotypeMatrix,
       population: Optional[Union[str, list]] = None,
       span_normalize=True,
       missing_data: str = 'include') -> float:
    """
    Calculate nucleotide diversity (pi) for a population.

    Nucleotide diversity is the average number of nucleotide differences
    per site between two randomly chosen sequences from the population.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    span_normalize : bool
        ``True`` (default): auto-detect best denominator (accessible
        bases if mask set, else genomic span).
        ``False``: return raw sum.
    missing_data : str
        ``'include'`` (default) uses per-site valid data.
        ``'exclude'`` filters to sites with no missing data.

    Returns
    -------
    float
        Nucleotide diversity value
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return 0.0
    result = _compute_thetas(matrix, ('pi',))
    return _apply_span_normalize(result['thetas']['pi'], matrix, span_normalize)


def theta_w(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize=True,
            missing_data: str = 'include') -> float:
    """
    Calculate Watterson's theta for a population.

    Watterson's theta is an estimator of the population mutation rate based
    on the number of segregating sites.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    span_normalize : bool
        ``True`` (default): auto-detect best denominator.
        ``False``: return raw sum.
    missing_data : str
        ``'include'`` (default) uses per-site valid data.
        ``'exclude'`` filters to sites with no missing data.

    Returns
    -------
    float
        Watterson's theta value
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return 0.0
    result = _compute_thetas(matrix, ('watterson',))
    return _apply_span_normalize(result['thetas']['watterson'], matrix, span_normalize)


def tajimas_d(haplotype_matrix: HaplotypeMatrix,
              population: Optional[Union[str, list]] = None,
              missing_data: str = 'include') -> float:
    """
    Calculate Tajima's D statistic.

    Tajima's D tests the neutral mutation hypothesis by comparing two
    estimates of the population mutation rate.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        ``'include'`` (default) uses per-site valid data with harmonic
        mean of per-site sample sizes for variance terms.
        ``'exclude'`` filters to sites with no missing data.

    Returns
    -------
    float
        Tajima's D value
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return float("nan")
    return _compute_neutrality_test(matrix, 'pi', 'watterson')


def allele_frequency_spectrum(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            missing_data: str = 'include') -> cp.ndarray:
    """
    Calculate the allele frequency spectrum (AFS).

    The AFS is a histogram of allele frequencies across all sites.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Calculate AFS using available data per site.
        'exclude' - Only use sites with no missing data.

    Returns
    -------
    ndarray
        Array where element i contains the number of sites with i derived alleles
    """

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
        if matrix.num_variants == 0:
            return np.zeros(matrix.num_haplotypes + 1, dtype=np.int64)
        n_haplotypes = matrix.num_haplotypes
        freqs = cp.sum(matrix.haplotypes, axis=0)

    else:  # missing_data == 'include'
        max_n = matrix.num_haplotypes
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)

        sites_with_data = n_valid_per_site > 0
        if not cp.any(sites_with_data):
            return np.zeros(max_n + 1, dtype=np.int64)

        # Filter to sites with valid data and check they're biallelic
        valid_sites = cp.where(sites_with_data)[0]
        derived_at_valid = derived_counts[valid_sites]
        n_valid_at_valid = n_valid_per_site[valid_sites]

        # Check biallelic assumption: derived count should be <= n_valid
        biallelic_mask = derived_at_valid <= n_valid_at_valid
        final_derived = derived_at_valid[biallelic_mask]

        # Create AFS histogram
        # Use bincount which is more efficient than a loop
        if len(final_derived) > 0:
            # Ensure derived counts don't exceed max_n
            final_derived = cp.minimum(final_derived, max_n)
            afs = cp.bincount(final_derived, minlength=max_n + 1)
            # Ensure correct size and type
            if len(afs) < max_n + 1:
                afs_full = cp.zeros(max_n + 1, dtype=cp.int64)
                afs_full[:len(afs)] = afs
                afs = afs_full
            else:
                afs = afs[:max_n + 1].astype(cp.int64)
        else:
            return np.zeros(max_n + 1, dtype=np.int64)

        return afs.get()

    # For exclude mode, create standard histogram
    return cp.histogram(freqs, bins=cp.arange(n_haplotypes + 2))[0].get()


def segregating_sites(haplotype_matrix: HaplotypeMatrix,
                     population: Optional[Union[str, list]] = None,
                     missing_data: str = 'include') -> int:
    """
    Count the number of segregating sites.

    A site is segregating if it has more than one allele present.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Count sites as segregating based on non-missing data only.
        'exclude' - Only count sites with no missing data.

    Returns
    -------
    int
        Number of segregating sites
    """

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
        if matrix.num_variants == 0:
            return 0
        allele_counts = cp.sum(matrix.haplotypes, axis=0)
        n_haplotypes = matrix.num_haplotypes
        segregating = (allele_counts > 0) & (allele_counts < n_haplotypes)

    else:  # missing_data == 'include'
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 2

        if not cp.any(sites_with_data):
            return 0

        valid_sites = cp.where(sites_with_data)[0]
        segregating_mask = (derived_counts[valid_sites] > 0) & (derived_counts[valid_sites] < n_valid_per_site[valid_sites])
        return int(cp.sum(segregating_mask).get())

    return int(cp.sum(segregating).get())


def singleton_count(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None,
                   missing_data: str = 'include') -> int:
    """
    Count the number of singleton variants.

    A singleton is a variant present in exactly one haplotype.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Count singletons based on non-missing data only.
        'exclude' - Only count singletons at sites with no missing data.

    Returns
    -------
    int
        Number of singleton variants
    """

    # Get population subset if specified
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    # Ensure on GPU
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if missing_data == 'exclude':
        matrix = matrix.exclude_missing_sites()
        if matrix.num_variants == 0:
            return 0
        allele_counts = cp.sum(matrix.haplotypes, axis=0)

    else:  # missing_data == 'include'
        derived_counts, n_valid_per_site = _dac_and_n(matrix.haplotypes)
        sites_with_data = n_valid_per_site >= 1

        if not cp.any(sites_with_data):
            return 0

        valid_sites = cp.where(sites_with_data)[0]
        return int(cp.sum(derived_counts[valid_sites] == 1).get())

    # For exclude mode
    return int(cp.sum(allele_counts == 1).get())


def diversity_stats(haplotype_matrix: HaplotypeMatrix,
                   population: Optional[Union[str, list]] = None,
                   statistics: list = ['pi', 'theta_w', 'tajimas_d'],
                   span_normalize=True,
                   missing_data: str = 'include') -> Dict[str, float]:
    """
    Compute multiple diversity statistics at once.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    statistics : list
        List of statistics to compute
    span_normalize : bool
        ``True`` (default): auto-detect best denominator.
        ``False``: return raw sums.
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data

    Returns
    -------
    dict
        Dictionary mapping statistic names to values
    """
    # Map stat names to estimator names for batched computation
    theta_stats = {'pi': 'pi', 'theta_w': 'watterson', 'theta_h': 'theta_h',
                   'theta_l': 'theta_l'}
    # Neutrality tests: (w1, w2) weight pairs
    test_specs = {
        'tajimas_d': ('pi', 'watterson'),
        'fay_wus_h': ('pi', 'theta_h'),
        'normalized_fay_wus_h': ('pi', 'theta_h'),
        'zeng_e': ('theta_l', 'watterson'),
    }
    needs_thetas = {s for s in statistics if s in theta_stats or s in test_specs}

    results = {}

    if needs_thetas:
        matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
        if matrix.num_variants == 0:
            for s in needs_thetas:
                results[s] = 0.0 if s in theta_stats else float('nan')
        else:
            # Collect all needed estimators for a single _compute_thetas call
            estimators = set()
            for s in needs_thetas:
                if s in theta_stats:
                    estimators.add(theta_stats[s])
                elif s in test_specs:
                    estimators.update(test_specs[s])
            ct = _compute_thetas(matrix, tuple(estimators))

            for s in needs_thetas:
                if s in theta_stats:
                    results[s] = _apply_span_normalize(
                        ct['thetas'][theta_stats[s]], matrix, span_normalize)
                elif s in test_specs:
                    w1, w2 = test_specs[s]
                    S = ct['S']
                    n = ct['n_harmonic_mean']
                    if S < 3 or n < 3:
                        results[s] = float('nan')
                    elif s == 'fay_wus_h':
                        results[s] = float(ct['thetas'][w1] - ct['thetas'][w2])
                    else:
                        var = _achaz_variance(w1, w2, n, S)
                        num = ct['thetas'][w1] - ct['thetas'][w2]
                        results[s] = float(num / math.sqrt(var)) if var > 0 else float('nan')

    # Non-theta stats
    for stat in statistics:
        if stat in results:
            continue
        if stat == 'segregating_sites':
            results['segregating_sites'] = segregating_sites(haplotype_matrix, population, missing_data)
        elif stat == 'singletons':
            results['singletons'] = singleton_count(haplotype_matrix, population, missing_data)
        elif stat == 'n_variants':
            m = _prepare_matrix(haplotype_matrix, population, missing_data)
            results['n_variants'] = m.num_variants
        elif stat == 'haplotype_diversity':
            results['haplotype_diversity'] = haplotype_diversity(haplotype_matrix, population, missing_data)
        elif stat not in ('pi', 'theta_w', 'theta_h', 'theta_l', 'tajimas_d'):
            raise ValueError(f"Unknown statistic: {stat}")

    return results



def fay_wus_h(haplotype_matrix: HaplotypeMatrix,
              population: Optional[Union[str, list]] = None,
              missing_data: str = 'include') -> float:
    """
    Calculate Fay and Wu's H statistic.

    Tests for an excess of high-frequency derived alleles, which can indicate
    positive selection.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data

    Returns
    -------
    float
        Fay and Wu's H value
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return float("nan")
    result = _compute_thetas(matrix, ('pi', 'theta_h'))
    return result['thetas']['pi'] - result['thetas']['theta_h']


def haplotype_diversity(haplotype_matrix: HaplotypeMatrix,
                       population: Optional[Union[str, list]] = None,
                       missing_data: str = 'include') -> float:
    """
    Calculate haplotype diversity for a population.

    Haplotype diversity is defined as 1 - sum(p_i^2) where p_i is the
    frequency of the i-th unique haplotype in the population.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices. If None, uses all samples
    missing_data : str
        'include' - exclude haplotypes with any missing data
        'exclude' - filter to sites with no missing data

    Returns
    -------
    float
        Haplotype diversity value
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    haplotypes = matrix.haplotypes  # (n_hap, n_var)

    if missing_data == 'exclude':
        missing_per_var = cp.sum(haplotypes < 0, axis=0)
        complete = cp.where(missing_per_var == 0)[0]
        haplotypes = haplotypes[:, complete]

    n_haplotypes = haplotypes.shape[0]
    if n_haplotypes <= 1:
        return 0.0

    has_missing = bool(cp.any(haplotypes < 0).get())

    if has_missing:
        # Fallback: wildcard matching requires CPU pairwise comparison
        haplotypes_cpu = haplotypes.get() if hasattr(haplotypes, 'get') else haplotypes
        cluster_id = _cluster_haplotypes_with_missing(haplotypes_cpu)
        from collections import Counter
        counts = Counter(cluster_id)
        frequencies = np.array(list(counts.values())) / n_haplotypes
    else:
        _, counts_gpu = _count_unique_haplotypes_gpu(haplotypes)
        frequencies = (counts_gpu.astype(cp.float64) / n_haplotypes).get()

    diversity = (1.0 - np.sum(frequencies ** 2)) * n_haplotypes / (n_haplotypes - 1)
    return float(diversity)


_HASH_SEED = 42  # fixed so identical inputs produce identical groupings across calls
_HASH_TOL = 1e-3  # collision-safe for float32 dot products of 0/1 vectors at our n_var


def _count_unique_haplotypes_gpu(haplotypes):
    """Count unique haplotypes on GPU via dot-product hashing.

    Caller must guarantee the input contains no missing data (-1).

    Returns
    -------
    n_unique : int
    counts : cupy.ndarray of group sizes (unsorted)
    """
    n_haplotypes, n_var = haplotypes.shape
    rng = cp.random.RandomState(seed=_HASH_SEED)
    w1 = rng.standard_normal(n_var, dtype=cp.float32)
    w2 = rng.standard_normal(n_var, dtype=cp.float32)
    h_f32 = haplotypes.astype(cp.float32)
    hash1 = h_f32 @ w1
    hash2 = h_f32 @ w2
    order = cp.lexsort(cp.stack([hash2, hash1]))
    s1 = hash1[order]
    s2 = hash2[order]
    diff = (cp.abs(s1[1:] - s1[:-1]) > _HASH_TOL) | (cp.abs(s2[1:] - s2[:-1]) > _HASH_TOL)
    boundaries = cp.concatenate([cp.ones(1, dtype=cp.bool_), diff])
    boundary_idx = cp.where(boundaries)[0]
    tail = cp.full(1, n_haplotypes, dtype=boundary_idx.dtype)
    counts_gpu = cp.diff(cp.concatenate([boundary_idx, tail]))
    return boundary_idx.shape[0], counts_gpu


def _cluster_haplotypes_with_missing(haps):
    """Cluster haplotypes treating -1 as compatible with any allele.

    Two haplotypes are in the same cluster if they match at all positions
    where both are non-missing. Uses greedy assignment: each haplotype
    joins the first compatible cluster.

    Parameters
    ----------
    haps : ndarray, shape (n_haplotypes, n_variants)

    Returns
    -------
    labels : list of int, length n_haplotypes
    """
    n = haps.shape[0]
    has_any_missing = np.any(haps < 0)

    if not has_any_missing:
        # fast path: no missing data, use string hashing
        hap_strings = [''.join(map(str, h)) for h in haps]
        label_map = {}
        labels = []
        next_id = 0
        for s in hap_strings:
            if s not in label_map:
                label_map[s] = next_id
                next_id += 1
            labels.append(label_map[s])
        return labels

    # slow path: pairwise comparison with wildcard matching
    # representative haplotype per cluster (index into haps)
    cluster_reps = [0]
    labels = [0]

    for i in range(1, n):
        matched = False
        for c_idx, rep in enumerate(cluster_reps):
            # check if haps[i] matches haps[rep] at jointly non-missing sites
            both_valid = (haps[i] >= 0) & (haps[rep] >= 0)
            if np.all(haps[i][both_valid] == haps[rep][both_valid]):
                labels.append(c_idx)
                matched = True
                break
        if not matched:
            cluster_reps.append(i)
            labels.append(len(cluster_reps) - 1)

    return labels


_get_population_matrix = get_population_matrix


def theta_h(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize=True,
            missing_data: str = 'include') -> float:
    """Compute theta_H (homozygosity-based diversity estimator).

    theta_H = sum_i [ i^2 * S_i ] * 2 / (n*(n-1)) where S_i is the count
    of variants with derived allele count i. Used to compute Fay and Wu's H.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    span_normalize : bool
        ``True`` (default): auto-detect best denominator.
        ``False``: return raw sum.
    missing_data : str
        'include' - per-site sample sizes
        'exclude' - only sites with no missing data

    Returns
    -------
    float
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return 0.0
    result = _compute_thetas(matrix, ('theta_h',))
    return _apply_span_normalize(result['thetas']['theta_h'], matrix, span_normalize)


def theta_l(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            span_normalize=True,
            missing_data: str = 'include') -> float:
    """Compute theta_L diversity estimator.

    theta_L = sum_i(i * xi_i) / (n - 1), where xi_i is the count of
    sites with derived allele count i. Weights variants linearly by
    derived allele frequency, bridging theta_pi and theta_H.

    Reference: Zeng et al. (2006), Genetics 174: 1431-1439, Equation (8).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    span_normalize : bool
        ``True`` (default): auto-detect best denominator.
        ``False``: return raw sum.
    missing_data : str
        'include' - per-site sample sizes
        'exclude' - only sites with no missing data

    Returns
    -------
    float
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return 0.0
    result = _compute_thetas(matrix, ('theta_l',))
    return _apply_span_normalize(result['thetas']['theta_l'], matrix, span_normalize)




def normalized_fay_wus_h(haplotype_matrix: HaplotypeMatrix,
                         population: Optional[Union[str, list]] = None,
                         missing_data: str = 'include') -> float:
    """Compute normalized Fay and Wu's H (H*).

    H = theta_pi - theta_H, normalized by its standard deviation under
    the standard neutral model. The normalization allows comparison
    across samples with different numbers of segregating sites.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (11).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        ``'include'`` (default) uses per-site sample
        sizes with harmonic mean n for variance terms.
        ``'exclude'`` filters to sites with no missing data.

    Returns
    -------
    float
        Normalized H*. Negative values indicate excess high-frequency
        derived alleles (directional selection signal).
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return float("nan")
    return _compute_neutrality_test(matrix, 'pi', 'theta_h')


def zeng_e(haplotype_matrix: HaplotypeMatrix,
           population: Optional[Union[str, list]] = None,
           missing_data: str = 'include') -> float:
    """Compute Zeng's E test statistic.

    E = theta_L - theta_W, normalized by its standard deviation.

    Reference: Zeng et al. (2006), Genetics 174: 1431-1439, Equation (13).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        ``'include'`` (default) or ``'exclude'``.

    Returns
    -------
    float
    """
    matrix = _prepare_matrix(haplotype_matrix, population, missing_data)
    if matrix.num_variants == 0:
        return float('nan')
    return _compute_neutrality_test(matrix, 'theta_l', 'watterson')


def zeng_dh(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            missing_data: str = 'include') -> float:
    """Compute Zeng's DH joint test statistic.

    Combines Tajima's D and Fay & Wu's H into a single test with
    improved power to detect directional selection. Defined as the
    product D * H when both are negative, zero otherwise.

    Reference: Zeng et al. (2006), "Statistical Tests for Detecting
    Positive Selection by Utilizing High-Frequency Variants",
    Genetics 174: 1431-1439, Equation (15).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        Passed through to tajimas_d and fay_wus_h.

    Returns
    -------
    float
        DH statistic. Positive when both D and H are negative
        (consistent with a selective sweep).
    """
    D = tajimas_d(haplotype_matrix, population, missing_data=missing_data)
    H = fay_wus_h(haplotype_matrix, population, missing_data=missing_data)

    # DH is the product when both are negative (sweep signal)
    if D < 0 and H < 0:
        return float(D * H)
    else:
        return 0.0


def max_daf(haplotype_matrix: HaplotypeMatrix,
            population: Optional[Union[str, list]] = None,
            missing_data: str = 'include') -> float:
    """Maximum derived allele frequency across all variants.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' - per-site n_valid for frequency
        'exclude' - only sites with no missing data

    Returns
    -------
    float
        Maximum DAF in [0, 1].
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)

    if missing_data == 'exclude':
        complete = n_valid_i == matrix.haplotypes.shape[0]
        freqs = cp.where(complete, dac / n_valid, -1.0)
    else:
        usable = n_valid > 0
        freqs = cp.where(usable, dac / n_valid, 0.0)

    return float(cp.max(freqs).get())


def haplotype_count(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None,
                    missing_data: str = 'include') -> int:
    """Count distinct haplotypes.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' - exclude haplotypes with any missing
        'exclude' - filter to sites with no missing data

    Returns
    -------
    int
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    haplotypes = matrix.haplotypes
    excluded = missing_data == 'exclude'

    if excluded:
        haplotypes = haplotypes[:, cp.sum(haplotypes < 0, axis=0) == 0]

    if haplotypes.shape[0] <= 1:
        return haplotypes.shape[0]

    # 'exclude' already removed every site with a -1, so the remainder is clean
    has_missing = False if excluded else bool(cp.any(haplotypes < 0).get())

    if has_missing:
        # Wildcard matching requires CPU pairwise comparison
        hap_cpu = haplotypes.get().astype(np.int8)
        labels = _cluster_haplotypes_with_missing(hap_cpu)
        return len(set(labels))

    n_unique, _ = _count_unique_haplotypes_gpu(haplotypes)
    return n_unique


def daf_histogram(matrix, n_bins: int = 20,
                  population: Optional[Union[str, list]] = None,
                  missing_data: str = 'include'):
    """Normalized histogram of derived allele frequencies.

    Accepts HaplotypeMatrix or GenotypeMatrix. For diploid data,
    DAF = sum(genotypes) / (2 * n_individuals).

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
    n_bins : int
        Number of frequency bins spanning [0, 1].
    population : str or list, optional
    missing_data : str
        'include' - per-site n_valid for frequency
        'exclude' - only sites with no missing data

    Returns
    -------
    hist : ndarray, float64, shape (n_bins,)
        Normalized counts (sum to 1).
    bin_edges : ndarray, float64, shape (n_bins + 1,)
    """

    from .genotype_matrix import GenotypeMatrix

    if isinstance(matrix, GenotypeMatrix):
        return _daf_histogram_diploid(matrix, n_bins, population)

    if population is not None:
        matrix = _get_population_matrix(matrix, population)

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)

    if missing_data == 'exclude':
        complete = n_valid_i == matrix.haplotypes.shape[0]
        dafs = (dac / n_valid)[complete]
    else:
        usable = n_valid > 0
        dafs = cp.where(usable, dac / n_valid, 0.0)

    return _histogram_from_dafs(dafs, n_bins)


def diplotype_frequency_spectrum(genotype_matrix,
                                 population: Optional[Union[str, list]] = None):
    """Count distinct multi-locus genotype patterns (diplotypes).

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix
    population : str or list, optional

    Returns
    -------
    freqs : ndarray, float64, sorted descending
        Diplotype frequencies.
    n_diplotypes : int
        Number of distinct diplotypes.
    """
    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if isinstance(geno, cp.ndarray):
        geno = geno.get()

    geno = np.asarray(geno, dtype=np.int8)
    n_ind = geno.shape[0]

    # treat missing (-1) as wildcard for diplotype identity
    labels = _cluster_haplotypes_with_missing(geno)
    from collections import Counter
    counts = Counter(labels)
    freqs = np.array(sorted(counts.values(), reverse=True)) / n_ind

    return freqs, len(counts)


def _histogram_from_dafs(dafs, n_bins):
    """Shared: compute normalized histogram from DAF CuPy array."""
    bin_edges = cp.linspace(0, 1, n_bins + 1)
    hist = cp.histogram(dafs, bins=bin_edges)[0].astype(cp.float64)
    total = cp.sum(hist)
    if total > 0:
        hist = hist / total
    return hist.get(), bin_edges.get()


def _daf_histogram_diploid(genotype_matrix, n_bins=20, population=None):
    """DAF histogram from diploid genotypes (internal)."""
    if population is not None:
        pop_idx = genotype_matrix.sample_sets.get(population)
        if pop_idx is None:
            raise ValueError(f"Population {population} not found")
        geno = genotype_matrix.genotypes[pop_idx, :]
    else:
        geno = genotype_matrix.genotypes

    if not isinstance(geno, cp.ndarray):
        geno = cp.asarray(geno)

    valid_mask = geno >= 0
    geno_clean = cp.where(valid_mask, geno, 0).astype(cp.float64)
    n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
    usable = n_valid > 0
    dafs = cp.where(usable, cp.sum(geno_clean, axis=0) / (2.0 * n_valid), 0.0)

    return _histogram_from_dafs(dafs, n_bins)


# backward compat alias
daf_histogram_diploid = _daf_histogram_diploid


# Summary statistics combinations commonly used

def neutrality_tests(haplotype_matrix: HaplotypeMatrix,
                    population: Optional[Union[str, list]] = None,
                    missing_data: str = 'include') -> Dict[str, float]:
    """
    Compute common neutrality test statistics.

    Returns Tajima's D, Fay and Wu's H, and related values.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        The haplotype data
    population : str or list, optional
        Population name or list of sample indices
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data

    Returns
    -------
    dict
        Dictionary with neutrality test results
    """
    return {
        'tajimas_d': tajimas_d(haplotype_matrix, population, missing_data),
        'fay_wus_h': fay_wus_h(haplotype_matrix, population, missing_data),
        'pi': pi(haplotype_matrix, population, span_normalize=False, missing_data=missing_data),
        'theta_w': theta_w(haplotype_matrix, population, span_normalize=False, missing_data=missing_data),
        'segregating_sites': segregating_sites(haplotype_matrix, population, missing_data)
    }


def heterozygosity_expected(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            missing_data: str = 'include'):
    """
    Compute expected heterozygosity (gene diversity) per variant.

    He = 1 - sum(p_i^2) for each variant, where p_i are allele frequencies.
    For biallelic sites this simplifies to He = 2*p*(1-p).

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'exclude' - NaN at sites with any missing data
        'include' - per-site n_valid for frequency calculation

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Expected heterozygosity per variant.
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    dac_i, n_valid_i = _dac_and_n(matrix.haplotypes)
    dac = dac_i.astype(cp.float64)
    n_valid = n_valid_i.astype(cp.float64)
    n = matrix.haplotypes.shape[0]

    if missing_data == 'include':
        p = cp.where(n_valid > 0, dac / n_valid, 0.0)
        he = 2.0 * p * (1.0 - p)
        he = cp.where(n_valid >= 2, he, cp.nan)
    else:
        p = dac / n
        he = 2.0 * p * (1.0 - p)

        if missing_data == 'exclude':
            incomplete = n_valid_i < n
            he[incomplete] = cp.nan

    return he.get()


def heterozygosity_observed(haplotype_matrix: HaplotypeMatrix,
                            population: Optional[Union[str, list]] = None,
                            ploidy: int = 2,
                            missing_data: str = 'include'):
    """
    Compute observed heterozygosity per variant.

    Assumes consecutive haplotypes belong to the same individual
    (standard for diploid VCF data). A site is heterozygous in an
    individual if the two haplotypes differ.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    ploidy : int
        Ploidy level. Default 2 (diploid).
    missing_data : str
        'include' - skip missing individuals per site (default)
        'exclude' - NaN at sites with any missing data

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Observed heterozygosity per variant.
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    n_hap = hap.shape[0]

    if n_hap % ploidy != 0:
        raise ValueError(
            f"Number of haplotypes ({n_hap}) not divisible by ploidy ({ploidy})")

    n_individuals = n_hap // ploidy

    if ploidy == 2:
        h1 = hap[0::2]
        h2 = hap[1::2]

        valid = (h1 >= 0) & (h2 >= 0)
        het = (h1 != h2) & valid
        n_valid = cp.sum(valid, axis=0).astype(cp.float64)
        n_het = cp.sum(het, axis=0).astype(cp.float64)

        ho = cp.where(n_valid > 0, n_het / n_valid, cp.nan)
    else:
        n_variants = hap.shape[1]
        n_het = cp.zeros(n_variants, dtype=cp.float64)
        n_valid_ind = cp.zeros(n_variants, dtype=cp.float64)

        for ind in range(n_individuals):
            ind_haps = hap[ind * ploidy:(ind + 1) * ploidy]
            all_valid = cp.all(ind_haps >= 0, axis=0)
            all_same = cp.all(ind_haps == ind_haps[0:1], axis=0)
            n_valid_ind += all_valid.astype(cp.float64)
            n_het += (all_valid & ~all_same).astype(cp.float64)

        ho = cp.where(n_valid_ind > 0, n_het / n_valid_ind, cp.nan)

    if missing_data == 'exclude':
        has_missing = cp.any(hap < 0)
        if has_missing:
            missing_per_var = cp.sum(hap < 0, axis=0)
            ho[missing_per_var > 0] = cp.nan

    return ho.get()


def inbreeding_coefficient(haplotype_matrix: HaplotypeMatrix,
                           population: Optional[Union[str, list]] = None,
                           ploidy: int = 2,
                           missing_data: str = 'include'):
    """
    Compute Wright's inbreeding coefficient F per variant.

    F = 1 - Ho/He, where Ho is observed heterozygosity and He is
    expected heterozygosity.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    ploidy : int
        Ploidy level for observed heterozygosity computation.
    missing_data : str
        Passed to heterozygosity_expected and heterozygosity_observed.

    Returns
    -------
    ndarray, float64, shape (n_variants,)
        Inbreeding coefficient per variant. NaN where He = 0.
    """
    ho = cp.asarray(heterozygosity_observed(haplotype_matrix, population,
                                             ploidy, missing_data=missing_data))
    he = cp.asarray(heterozygosity_expected(haplotype_matrix, population,
                                             missing_data=missing_data))

    f = cp.where(he > 0, 1.0 - ho / he, cp.nan)
    return f.get()


def mu_var(haplotype_matrix: HaplotypeMatrix,
           window_length: Optional[float] = None,
           population: Optional[Union[str, list]] = None) -> float:
    """mu_VAR: SNP density statistic (RAiSD).

    Number of SNPs per base pair. Elevated near sweeps due to
    hitchhiking effects on local variant density.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    window_length : float, optional
        Window length in bp. If None, uses chrom_end - chrom_start.
    population : str or list, optional

    Returns
    -------
    float
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    n_snps = matrix.num_variants
    if window_length is None:
        window_length = matrix.chrom_end - matrix.chrom_start
    if window_length <= 0:
        return 0.0

    return float(n_snps / window_length)


def mu_sfs(haplotype_matrix: HaplotypeMatrix,
           population: Optional[Union[str, list]] = None,
           missing_data: str = 'include') -> float:
    """mu_SFS: fraction of SNPs at SFS edges (RAiSD).

    Counts singletons (DAC=1) and near-fixed variants (DAC=n-1),
    divided by total segregating sites. Elevated near selective sweeps.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    population : str or list, optional
    missing_data : str
        'include' - per-site n_valid for edge classification
        'exclude' - only sites with no missing data

    Returns
    -------
    float
    """

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    dac, n_valid = _dac_and_n(matrix.haplotypes)

    if missing_data == 'exclude':
        n = matrix.haplotypes.shape[0]
        complete = n_valid == n
        is_seg = complete & (dac > 0) & (dac < n)
        is_edge = complete & ((dac == 1) | (dac == n - 1))
    else:
        usable = n_valid >= 2
        is_seg = usable & (dac > 0) & (dac < n_valid)
        is_edge = usable & ((dac == 1) | (dac == n_valid - 1))

    n_seg = cp.sum(is_seg)
    if int(n_seg.get()) == 0:
        return 0.0

    n_edge = cp.sum(is_edge)
    return float((n_edge.astype(cp.float64) / n_seg.astype(cp.float64)).get())
