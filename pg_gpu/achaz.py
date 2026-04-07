"""
Achaz (2009) generalized theta estimation framework.

All frequency-spectrum-based theta estimators are linear combinations of
the site frequency spectrum (SFS) weighted by specific vectors. This module
computes the SFS once on GPU and derives all estimators as trivial dot
products, giving 6-24x speedup when computing multiple statistics.

References
----------
Achaz, G. (2009). Frequency Spectrum Neutrality Tests: One for All and
    All for One. Genetics, 183(1), 249-258.
Fu, Y.X. (1995). Statistical properties of segregating sites. Theoretical
    Population Biology, 48(2), 172-197.
Gutenkunst, R.N. et al. (2009). Inferring the joint demographic history
    of multiple populations from multidimensional SNP frequency data.
    PLoS Genetics, 5(10), e1000695.
"""

import numpy as np
from functools import lru_cache
from typing import Optional, Union, Callable, Dict

from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix
from .sfs import _derived_allele_counts


# ---------------------------------------------------------------------------
# Weight functions: w(k, n) for k=1..n-1
#
# Each returns a numpy array of length n+1 (indices 0..n).
# Only indices 1..n-1 are used for segregating sites.
# ---------------------------------------------------------------------------

def _weights_watterson(n):
    """Watterson's theta_S: uniform weight on SFS entries."""
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1:n] = 1.0 / a1
    return w


def _weights_pi(n):
    """Nucleotide diversity (pi): weight by heterozygosity."""
    k = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * k * (n - k) / (n * (n - 1))
    w[0] = 0.0
    w[n] = 0.0
    return w


def _weights_theta_h(n):
    """Fay & Wu's theta_H: weight by squared frequency."""
    k = np.arange(n + 1, dtype=np.float64)
    w = 2.0 * k ** 2 / (n * (n - 1))
    w[0] = 0.0
    w[n] = 0.0
    return w


def _weights_theta_l(n):
    """Zeng et al. theta_L: linear frequency weight."""
    k = np.arange(n + 1, dtype=np.float64)
    w = k / (n - 1)
    w[0] = 0.0
    w[n] = 0.0
    return w


def _weights_eta1(n):
    """Fu & Li's singleton-based theta: weight on singletons only."""
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1] = 1.0 / a1
    return w


def _weights_eta1_star(n):
    """Fu & Li's folded singleton theta: singletons + (n-1)-tons."""
    w = np.zeros(n + 1)
    # For folded spectrum: weight singletons and their complement
    a1 = np.sum(1.0 / np.arange(1, n))
    w[1] = 1.0 / a1
    w[n - 1] = 1.0 / a1
    return w


def _weights_minus_eta1(n):
    """Achaz (2008): all segregating sites except singletons."""
    w = np.zeros(n + 1)
    # S - singletons, normalized by appropriate harmonic number
    a1 = np.sum(1.0 / np.arange(1, n))
    w[2:n] = 1.0 / (a1 - 1.0)
    return w


def _weights_minus_eta1_star(n):
    """Achaz (2008): all segregating sites except singletons and (n-1)-tons."""
    w = np.zeros(n + 1)
    a1 = np.sum(1.0 / np.arange(1, n))
    w[2:n - 1] = 1.0 / (a1 - 1.0 - 1.0 / (n - 1))
    return w


# Registry of built-in weight functions
WEIGHT_REGISTRY: Dict[str, Callable] = {
    'watterson': _weights_watterson,
    'theta_s': _weights_watterson,
    'pi': _weights_pi,
    'theta_pi': _weights_pi,
    'theta_h': _weights_theta_h,
    'theta_l': _weights_theta_l,
    'eta1': _weights_eta1,
    'eta1_star': _weights_eta1_star,
    'minus_eta1': _weights_minus_eta1,
    'minus_eta1_star': _weights_minus_eta1_star,
}


# ---------------------------------------------------------------------------
# Fu (1995) sigma_ij covariance structure
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _harmonic_sums(n):
    """Precompute harmonic sums H[i] = sum(1/j for j=1..i) for i=0..n."""
    H = np.zeros(n + 1)
    H[1:] = np.cumsum(1.0 / np.arange(1, n + 1))
    return H


@lru_cache(maxsize=64)
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
        Covariance matrix indexed from i=1..n-1 (stored at [i-1, j-1]).
    """
    H = _harmonic_sums(n)
    sigma = np.zeros((n - 1, n - 1))

    def beta(i, nn):
        """Fu (1995) beta function."""
        ai = H[i - 1]
        an = H[nn - 1]
        return 2.0 * nn / ((nn - i + 1) * (nn - i)) * (an + 1.0 / nn - ai) - 2.0 / (nn - i)

    def sigma_ii(i, nn):
        """Diagonal element."""
        if 2 * i < nn:
            return beta(i + 1, nn)
        elif 2 * i == nn:
            ai = H[i - 1]
            an = H[nn - 1]
            return 2.0 * (an - ai) / (nn - i) - 1.0 / (i * i)
        else:
            return beta(i, nn) - 1.0 / (i * i)

    def sigma_ij_val(i, j, nn):
        """Off-diagonal element (i > j)."""
        if i == j:
            return sigma_ii(i, nn)
        if i < j:
            i, j = j, i
        if i + j < nn:
            return (beta(i + 1, nn) - beta(i, nn)) / 2.0
        elif i + j == nn:
            ai = H[i - 1]
            aj = H[j - 1]
            an = H[nn - 1]
            return ((an - ai) / (nn - i) + (an - aj) / (nn - j)
                    - (beta(i, nn) + beta(j + 1, nn)) / 2.0
                    - 1.0 / (i * j))
        else:
            return (beta(j, nn) - beta(j + 1, nn)) / 2.0 - 1.0 / (i * j)

    for i in range(1, n):
        for j in range(1, n):
            sigma[i - 1, j - 1] = sigma_ij_val(i, j, n)

    return sigma


# ---------------------------------------------------------------------------
# SFS Projection (Gutenkunst et al. 2009)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _projection_matrix(n_from, n_to):
    """Compute the hypergeometric projection matrix from n_from to n_to.

    P[k_to, k_from] = C(k_from, k_to) * C(n_from - k_from, n_to - k_to) / C(n_from, n_to)

    Returns
    -------
    P : ndarray, float64, shape (n_to + 1, n_from + 1)
    """
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

    Parameters
    ----------
    sfs : ndarray, shape (n_from + 1,)
        Site frequency spectrum at sample size n_from.
    n_from : int
        Current sample size.
    n_to : int
        Target sample size (must be <= n_from).

    Returns
    -------
    projected : ndarray, shape (n_to + 1,)
        Projected SFS.
    """
    if n_to > n_from:
        raise ValueError(f"Cannot project up: n_to={n_to} > n_from={n_from}")
    if n_to == n_from:
        return sfs.copy()
    P = _projection_matrix(n_from, n_to)
    return P @ sfs


# ---------------------------------------------------------------------------
# FrequencySpectrum: core class
# ---------------------------------------------------------------------------

class FrequencySpectrum:
    """Site frequency spectrum with support for variable sample sizes.

    Computes derived allele counts on GPU in a single pass, groups variants
    by per-site sample size, and provides fast theta estimation via weight
    vector dot products.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    population : str or list, optional
        Population name or sample indices.
    missing_data : str
        'include' - group variants by per-site sample size (default)
        'exclude' - only sites with no missing data, single fixed n
    n_total_sites : int, optional
        Total callable sites (variant + invariant) for the region.
        If provided, invariant sites are added to the SFS at bin 0.
        Overrides haplotype_matrix.n_total_sites if set.

    Attributes
    ----------
    sfs_by_n : dict of {int: ndarray}
        Maps sample size n to SFS vector of shape (n+1,).
    n_max : int
        Maximum sample size observed.
    n_segregating : int
        Total number of segregating sites.
    """

    def __init__(self, haplotype_matrix: HaplotypeMatrix,
                 population=None, missing_data='include',
                 n_total_sites=None):
        if population is not None:
            matrix = get_population_matrix(haplotype_matrix, population)
        else:
            matrix = haplotype_matrix

        self._source_matrix = matrix
        n_hap = matrix.num_haplotypes
        if n_total_sites is None:
            n_total_sites = matrix.n_total_sites

        # Use shared SFS allele counting (handles GPU transfer + missing data)
        dac, n_valid = _derived_allele_counts(matrix, missing_data='include')
        dac_cpu = dac.get()
        nv_cpu = n_valid.get()

        if missing_data == 'exclude':
            complete = nv_cpu == n_hap
            dac_cpu = dac_cpu[complete]
            nv_cpu = nv_cpu[complete]

        # Group by sample size, build per-group SFS
        self.sfs_by_n = {}
        self.n_max = int(np.max(nv_cpu)) if len(nv_cpu) > 0 else 0

        for ni in np.unique(nv_cpu):
            ni = int(ni)
            if ni < 2:
                continue
            mask = nv_cpu == ni
            group_dac = dac_cpu[mask].astype(np.int64)
            xi = np.bincount(group_dac, minlength=ni + 1)[:ni + 1].astype(np.float64)
            self.sfs_by_n[ni] = xi

        # Count segregating sites
        self.n_segregating = sum(
            int(np.sum(xi[1:n])) for n, xi in self.sfs_by_n.items()
        )

        # Add invariant sites if n_total_sites is provided
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
            ``True``: auto-detect best denominator from source matrix.
            ``False`` (default): return raw sum.
        span : float, optional
            Explicit span for normalization. Overrides auto-detection.
            For internal use.

        Returns
        -------
        float
            Theta estimate.
        """
        if isinstance(weights, str):
            if weights not in WEIGHT_REGISTRY:
                raise ValueError(f"Unknown weight: {weights}. "
                                 f"Available: {list(WEIGHT_REGISTRY.keys())}")
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
        """Compute a neutrality test statistic T = (theta1 - theta2) / sqrt(var).

        Uses the Achaz (2009) Eq. 9 variance formula with Fu (1995)
        sigma_ij covariance structure.

        Parameters
        ----------
        w1, w2 : str or callable
            Weight functions for the two theta estimators.

        Returns
        -------
        T : float
            Normalized test statistic.
        """
        theta1 = self.theta(w1)
        theta2 = self.theta(w2)
        numerator = theta1 - theta2

        # For variance, need effective sample size
        # Use the dominant sample size (most variants)
        n_eff = max(self.sfs_by_n.keys(),
                    key=lambda n: np.sum(self.sfs_by_n[n]))

        # Build the Omega vector (Achaz Eq. 9)
        if isinstance(w1, str):
            w1_fn = WEIGHT_REGISTRY[w1]
        else:
            w1_fn = w1
        if isinstance(w2, str):
            w2_fn = WEIGHT_REGISTRY[w2]
        else:
            w2_fn = w2

        v1 = w1_fn(n_eff)
        v2 = w2_fn(n_eff)

        # Normalize weight vectors
        s1 = np.sum(v1[1:n_eff])
        s2 = np.sum(v2[1:n_eff])
        if s1 == 0 or s2 == 0:
            return float('nan')

        # Omega_i = v1_i/sum(v1) - v2_i/sum(v2) for i=1..n-1
        Omega = np.zeros(n_eff - 1)
        for i in range(1, n_eff):
            Omega[i - 1] = v1[i] / s1 - v2[i] / s2

        # alpha_n = sum(i * Omega_i^2)
        k = np.arange(1, n_eff, dtype=np.float64)
        alpha_n = np.sum(k * Omega ** 2)

        # beta_n = sum_i sum_j (i*j * Omega_i * Omega_j * sigma_ij)
        sigma = compute_sigma_ij(n_eff)
        ij = k[:, np.newaxis] * k[np.newaxis, :]  # (n-1, n-1)
        OO = Omega[:, np.newaxis] * Omega[np.newaxis, :]  # (n-1, n-1)
        beta_n = float(np.sum(ij * OO * sigma))

        # Estimate theta and theta^2 from S (Watterson's estimator)
        S = self.n_segregating
        a1 = np.sum(1.0 / np.arange(1, n_eff))
        a2 = np.sum(1.0 / np.arange(1, n_eff) ** 2)
        theta_est = S / a1
        theta_sq_est = S * (S - 1) / (a1 ** 2 + a2)

        variance = alpha_n * theta_est + beta_n * theta_sq_est
        if variance <= 0:
            return float('nan')

        return numerator / np.sqrt(variance)

    def suggest_projection_n(self, retain_fraction=0.95):
        """Suggest a projection target that retains most sites.

        Picks the largest n such that at least ``retain_fraction`` of
        segregating sites have sample size >= n. This balances information
        retention (larger n = more resolution) against site loss (sites
        with n_valid < target are discarded by projection).

        Parameters
        ----------
        retain_fraction : float
            Fraction of segregating sites to retain (default 0.95).

        Returns
        -------
        int
            Suggested target sample size, or n_max if no missing data.
        """
        if len(self.sfs_by_n) <= 1:
            return self.n_max

        # Build cumulative site count from largest n downward
        sorted_ns = sorted(self.sfs_by_n.keys(), reverse=True)
        total_seg = self.n_segregating
        if total_seg == 0:
            return self.n_max

        cumulative = 0
        for ni in sorted_ns:
            xi = self.sfs_by_n[ni]
            cumulative += int(np.sum(xi[1:ni]))
            if cumulative / total_seg >= retain_fraction:
                return ni

        return sorted_ns[-1]

    def project(self, target_n):
        """Project all SFS groups to a common sample size.

        Parameters
        ----------
        target_n : int
            Target sample size. Must be <= min sample size in sfs_by_n.

        Returns
        -------
        FrequencySpectrum
            New FrequencySpectrum with a single SFS at target_n.
        """
        projected = np.zeros(target_n + 1)
        n_sites_dropped = 0
        for n, xi in self.sfs_by_n.items():
            if n < target_n:
                n_sites_dropped += int(np.sum(xi))
                continue
            projected += project_sfs(xi, n, target_n)

        result = object.__new__(FrequencySpectrum)
        result.sfs_by_n = {target_n: projected}
        result.n_max = target_n
        result.n_segregating = int(np.sum(projected[1:target_n]))
        result.n_total_sites = self.n_total_sites
        return result

    def sfs(self, n=None):
        """Return the SFS, optionally projected to a target sample size.

        Parameters
        ----------
        n : int, optional
            Target sample size for projection. If None, returns the SFS
            at the maximum sample size (may be inaccurate if there are
            sites with smaller sample sizes).

        Returns
        -------
        ndarray
            Site frequency spectrum.
        """
        if n is not None:
            projected = self.project(n)
            return projected.sfs_by_n[n]
        if len(self.sfs_by_n) == 1:
            return list(self.sfs_by_n.values())[0]
        # Multiple sample sizes: return at n_max (includes only those sites)
        return self.sfs_by_n.get(self.n_max, np.array([]))

    def all_thetas(self, span_normalize=False, span=None):
        """Compute all standard theta estimators at once.

        Returns
        -------
        dict
            Maps estimator names to values.
        """
        return {
            name: self.theta(name, span_normalize=span_normalize, span=span)
            for name in ['pi', 'watterson', 'theta_h', 'theta_l',
                         'eta1', 'eta1_star', 'minus_eta1', 'minus_eta1_star']
        }

    def tajimas_d(self):
        """Compute Tajima's D using the classical (1989) variance formula.

        This uses the exact Tajima (1989) variance, not the Achaz general
        form. Use neutrality_test('pi', 'watterson') for the Achaz version.
        """
        pi = self.theta('pi')
        S = float(self.n_segregating)
        if S < 3:
            return float('nan')

        n = max(self.sfs_by_n.keys(),
                key=lambda ni: np.sum(self.sfs_by_n[ni]))
        a1 = np.sum(1.0 / np.arange(1, n))
        a2 = np.sum(1.0 / np.arange(1, n) ** 2)
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - 1 / a1
        c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
        e1 = c1 / a1
        e2 = c2 / (a1 ** 2 + a2)

        tw = S / a1
        d_var = e1 * S + e2 * S * (S - 1)
        if d_var <= 0:
            return float('nan')
        return (pi - tw) / np.sqrt(d_var)

    def fay_wu_h(self, normalized=False):
        """Compute Fay & Wu's H = pi - theta_H.

        Parameters
        ----------
        normalized : bool
            If True, return H / sqrt(Var[H]) using Zeng et al. (2006).
        """
        pi = self.theta('pi')
        th = self.theta('theta_h')
        h = pi - th
        if not normalized:
            return h

        n = max(self.sfs_by_n.keys(),
                key=lambda ni: np.sum(self.sfs_by_n[ni]))
        S = float(self.n_segregating)
        a1 = np.sum(1.0 / np.arange(1, n))
        a2 = np.sum(1.0 / np.arange(1, n) ** 2)
        theta_est = S / a1
        theta_sq = S * (S - 1) / (a1 ** 2 + a2)

        # Zeng et al. (2006) Eq. 5 variance terms
        tn = theta_est * (n - 2) / (6 * (n - 1))
        tn2 = theta_sq * (
            18 * n ** 2 * (3 * n + 2) * a2
            - (88 * n ** 3 + 9 * n ** 2 - 13 * n + 6)
        ) / (9 * n * (n - 1) ** 2)
        var_h = tn + tn2
        if var_h <= 0:
            return float('nan')
        return h / np.sqrt(var_h)

    def zeng_e(self):
        """Compute Zeng's E = (theta_L - theta_W) / sqrt(Var).

        Uses the Achaz (2009) general variance framework with Fu (1995)
        covariance structure, which is equivalent to the Zeng et al.
        (2006) Eq. 14 variance but avoids hand-coded coefficient errors.
        """
        return self.neutrality_test('theta_l', 'watterson')

    def all_tests(self):
        """Compute all standard neutrality tests at once.

        Returns
        -------
        dict
            Maps test names to values.
        """
        return {
            'tajimas_d': self.tajimas_d(),
            'fay_wu_h': self.fay_wu_h(),
            'normalized_fay_wu_h': self.fay_wu_h(normalized=True),
            'zeng_e': self.zeng_e(),
        }
