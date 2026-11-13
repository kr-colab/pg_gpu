"""
Windowed analysis module for computing population genetics statistics across genomic windows.

This module provides efficient computation of statistics in sliding or non-overlapping windows
with intelligent memory management and GPU acceleration.
"""

import difflib
import inspect
import numpy as np
import cupy as cp
import pandas as pd
from typing import List, Dict, Union, Optional, Callable, Iterator, Tuple, Any
from dataclasses import dataclass, field, replace
import warnings
from tqdm import tqdm

from .accessible import _as_accessible_mask, resolve_accessible_mask
from .haplotype_matrix import HaplotypeMatrix, _bp_grid_bounds, _total_sites_share
from . import ld_statistics
from . import divergence
from . import diversity
from . import selection
from ._utils import _eager_only, population_rows
from ._haplotype_groups import column_has_missing
from ._haplotype_hash import garud_h_windows
from ._warnings import _outside_phase_guard, _phase_dependent, _warn_unphased

# Column order of the Garud H family, as garud_h_windows returns it.
_FUSED_GARUD_STATS = ('garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1',
                      'haplotype_count')

_PHASE_DEPENDENT_STATS = frozenset(
    _FUSED_GARUD_STATS
) | frozenset({
    'zns', 'omega', 'mu_ld', 'ld_decay', 'mean_r2',
    'dist_var', 'dist_skew', 'dist_kurt',
    'mean_nsl',
    'local_pca', 'local_pca_jackknife',
})


def _phase_scan_context(name, statistics):
    """Warning context naming the phase-dependent stats a scan requests, or
    None when it requests none."""
    requested = sorted(str(s) for s in set(statistics) & _PHASE_DEPENDENT_STATS)
    return f"{name} ({', '.join(requested)})" if requested else None

# Statistics that read the rows of two populations.
_TWO_POP_STATS = frozenset({'fst', 'fst_hudson', 'fst_wc', 'dxy', 'da'})

# Single-population statistics that are per-base rates: divided by the
# window's bases, rescaled for missing data, and NaN where 'exclude' leaves a
# window that had variants with none.
_RATE_STATS = frozenset({'pi', 'theta_w', 'theta_h', 'theta_l'})

# Single-population statistics that the scatter-add engine
# (_windowed_thetas_scatter) computes from per-allele counts.
_SCATTER_SINGLE_STATS = frozenset({
    'pi', 'theta_w', 'tajimas_d', 'segregating_sites', 'theta_h', 'theta_l',
    'fay_wu_h', 'singletons', 'normalized_fay_wu_h', 'zeng_e', 'zeng_dh',
    'max_daf'})

# Single-population statistics of the main fused kernel, in both the
# fused and the chunked engine: the same names, from the kernel's own sums.
_FUSED_SINGLE_STATS = _SCATTER_SINGLE_STATS

# Fused per-site stats binned into windows by reducing each window's variant
# range [win_start, win_stop) with _WindowAccumulator (prefix-sum sum/mean, a
# 2-D cumsum histogram for daf_hist), so they serve any window family --
# including unordered or overlapping regions -- without the per-(variant,
# window) membership arrays the old scatter path needed.
_PER_SITE_RANGE_STATS = frozenset({'het_expected', 'mean_nsl', 'daf_hist',
                                   'mu_sfs'})

# Fused statistics of the variant positions in each window; mu_var is the
# variant density.
_SNP_DIST_STATS = frozenset({
    'snp_dist_mean', 'snp_dist_var', 'snp_dist_min', 'snp_dist_max',
    'mu_var'})

# Fused statistics computed window by window from the window's variants.
_PAIRWISE_LD_STATS = frozenset({'zns', 'omega', 'mu_ld'})
_DIST_MOMENT_STATS = frozenset({'dist_var', 'dist_skew', 'dist_kurt'})

# Statistics that windowed_statistics_fused computes.
_FUSED_STATS = (_FUSED_SINGLE_STATS | _TWO_POP_STATS
                | frozenset(_FUSED_GARUD_STATS)
                | _PER_SITE_RANGE_STATS
                | _SNP_DIST_STATS | _PAIRWISE_LD_STATS | _DIST_MOMENT_STATS)

# Fused statistics that the per-window fallback serves under 'exclude' when a
# request holds nothing else. Its columns differ from the fused engine's with
# populations (one column per population, where the fused engine gives the
# first population only), so which engine serves a request is part of its
# output. Do not add a name here when the fallback learns to compute it.
_EXCLUDE_FALLBACK_FIRST = (frozenset({'pi', 'theta_w', 'tajimas_d',
                                      'segregating_sites', 'het_expected'})
                           | frozenset(_FUSED_GARUD_STATS) | _TWO_POP_STATS)

# Statistics that windowed_statistics_fused_chunked computes itself, in
# chunks. It hands every other fused statistic to windowed_statistics_fused,
# because those do not need the transposed matrix that the chunking bounds.
_CHUNKED_OWN_STATS = _FUSED_SINGLE_STATS | _TWO_POP_STATS

# Statistics that windowed_analysis hands to decomposition.local_pca.
_LOCAL_PCA_STATS = frozenset({'local_pca', 'local_pca_jackknife'})

# Kwargs that the 'local_pca' / 'local_pca_jackknife' dispatch consumes but
# scalar-stat paths don't accept. Filtered out before the recursive call.
_LOCAL_PCA_ONLY_KWARGS = frozenset(
    {'k', 'population', 'batch_size', 'n_blocks', 'aggregate'})


def _compute_window_bases(haplotype_matrix, win_starts, win_stops,
                          is_accessible=None):
    """Per-window base count, the denominator of per-base statistics.

    Every windowed engine uses this one rule, so they agree on each
    window. In order of precedence:

    - an accessible mask (``is_accessible``, else the matrix's): the
      accessible bases in the window;
    - ``n_total_sites`` with no mask: the window's share of it
      (``haplotype_matrix._total_sites_share``), in proportion to the
      bases it covers in the region. The shares of windows that tile the
      region add up to ``n_total_sites``, the denominator of the scalar
      function on the whole region;
    - otherwise the window length.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Source of the mask, ``n_total_sites`` and the region bounds.
    win_starts, win_stops : array_like
        Window edges, right-open (numpy, not cupy).
    is_accessible : array_like of bool or AccessibleMask, optional
        Explicit accessibility (takes precedence over the matrix's mask);
        a bool array has element 0 at position 1, as in scikit-allel.

    Returns
    -------
    numpy.ndarray, float64
        Base count per window.
    """
    ws = np.asarray(win_starts, dtype=np.float64)
    we = np.asarray(win_stops, dtype=np.float64)

    if is_accessible is not None:
        return _as_accessible_mask(is_accessible).count_accessible_windows(
            ws.astype(np.int64), we.astype(np.int64))

    if haplotype_matrix.has_accessible_mask:
        return haplotype_matrix.accessible_mask.count_accessible_windows(
            ws.astype(np.int64), we.astype(np.int64))

    if haplotype_matrix.n_total_sites is not None:
        return _total_sites_share(haplotype_matrix, ws, we)

    return we - ws


def _scatter_spans(haplotype_matrix, win_starts, win_stops, span_normalize):
    """Per-window denominators of the scatter engines, and which are zero.

    A window with no base (fully inaccessible) divides to NaN, not 0: its
    per-base rate is undefined. The spans are clamped to 1 for a safe
    division and the zero-span flags go back for that post-processing.
    """
    if span_normalize is False:
        n = len(win_starts)
        return np.ones(n), np.zeros(n, dtype=bool)
    spans = _compute_window_bases(haplotype_matrix, win_starts, win_stops)
    return np.maximum(spans, 1.0), spans <= 0


def _windowed_rescale(n_variants, n_sites_with_data):
    """Per-window num_variants / n_sites_with_data, correcting for sites a
    population's own missing calls dropped from a raw sum. 1 for a window
    with no variants, since there is nothing to rescale; NaN for a window
    that has variants but none meeting the site's own call-count gate.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n_variants == 0, 1.0,
                        np.where(n_sites_with_data > 0,
                                n_variants / n_sites_with_data, np.nan))


def _missing_data_rescale(n_variants_per_window, normalize, all_complete, count_sites):
    """Per-window rescale for a raw sum, correcting sites a population's own
    missing calls dropped from it before the span divides it. None when
    ``normalize`` is False or every site already qualifies (``all_complete``).
    ``count_sites`` is called, lazily, for the per-window count of sites
    meeting the statistic's own gate.
    """
    if normalize is False or bool(all_complete):
        return None
    return _windowed_rescale(n_variants_per_window, count_sites())


CANONICAL_WINDOW_PREFIX = (
    'chrom', 'start', 'end', 'center', 'n_variants', 'window_id')

# Frequency bins of the windowed daf_hist columns (daf_bin_0 .. daf_bin_19).
N_DAF_BINS = 20


def _init_window_results(chrom, win_starts_bp, win_stops_bp, n_variants):
    """Canonical window prefix columns, shared by every dispatch path.

    Inserts ``chrom, start, end, center, n_variants, window_id`` in that
    order. Dict insertion order drives column order in the final DataFrame.
    """
    start = np.asarray(win_starts_bp).astype(int)
    end = np.asarray(win_stops_bp).astype(int)
    n = len(start)
    return {
        'chrom': np.repeat(chrom if chrom is not None else 1, n),
        'start': start,
        'end': end,
        'center': (start + end) // 2,
        'n_variants': np.asarray(n_variants).astype(int),
        'window_id': np.arange(n, dtype=int),
    }


@dataclass
class WindowParams:
    """Parameters defining genomic windows."""
    window_type: str  # 'bp', 'snp', or 'regions'
    window_size: int
    step_size: int
    regions: Optional[pd.DataFrame] = None


@dataclass
class WindowData:
    """Data for a single genomic window."""
    chrom: Union[str, int]
    start: int
    end: int
    center: int
    matrix: HaplotypeMatrix
    n_variants: int
    window_id: int
    # Results that several statistics share, computed once per window.
    cache: Dict = field(default_factory=dict)


def _window_mean_het_expected(matrix, missing_data):
    """Per-window mean expected heterozygosity for the per-window fallback.

    The mean of the per-site He over the sites with a defined value, NaN when
    none has one -- the same value the fused engine's windows.mean returns,
    and == np.nanmean(diversity.heterozygosity_expected(window)).
    """
    he = np.asarray(diversity.heterozygosity_expected(
        matrix, missing_data=missing_data), dtype=np.float64)
    if not np.any(np.isfinite(he)):
        return np.nan
    return float(np.nanmean(he))


class StatisticsComputer:
    """Computes population genetics statistics for windows."""

    # Built-in single population statistics
    # Note: These are now created dynamically to use instance parameters

    # Built-in two population statistics
    # Note: These are now created dynamically to use instance parameters

    # LD-based statistics
    LD_STATS = {
        'ld_decay': lambda w, **kwargs: _compute_mean_r2(w.matrix, **kwargs),
        'mean_r2': lambda w, max_dist: _compute_mean_r2(w.matrix, max_dist),
    }

    def __init__(self, statistics: List[Union[str, Callable]],
                 populations: Optional[List[str]] = None,
                 custom_stat_kwargs: Optional[Dict] = None,
                 ld_bins: Optional[List[int]] = None,
                 missing_data: str = 'include',
                 span_normalize=True):
        self.statistics = statistics
        self.populations = populations or []
        self.custom_stat_kwargs = custom_stat_kwargs or {}
        self.ld_bins = ld_bins or [0, 1000, 5000, 10000, 50000]
        self.missing_data = missing_data
        self.span_normalize = span_normalize

        # Categorize statistics
        self._categorize_statistics()

    def _categorize_statistics(self):
        """Categorize statistics by type for efficient computation."""
        sn = self.span_normalize
        self.SINGLE_POP_STATS = {
            'pi': lambda w: diversity.pi(
                w.matrix, span_normalize=sn,
                missing_data=self.missing_data),
            'theta_w': lambda w: diversity.theta_w(
                w.matrix, span_normalize=sn,
                missing_data=self.missing_data),
            'tajimas_d': lambda w: diversity.tajimas_d(w.matrix, missing_data=self.missing_data),
            'theta_h': lambda w: diversity.theta_h(
                w.matrix, span_normalize=sn,
                missing_data=self.missing_data),
            'theta_l': lambda w: diversity.theta_l(
                w.matrix, span_normalize=sn,
                missing_data=self.missing_data),
            'fay_wu_h': lambda w: diversity.fay_wus_h(w.matrix, missing_data=self.missing_data),
            'max_daf': lambda w: diversity.max_daf(w.matrix, missing_data=self.missing_data),
            'normalized_fay_wu_h': lambda w: diversity.normalized_fay_wus_h(
                w.matrix, missing_data=self.missing_data),
            'zeng_e': lambda w: diversity.zeng_e(w.matrix, missing_data=self.missing_data),
            'zeng_dh': lambda w: diversity.zeng_dh(w.matrix, missing_data=self.missing_data),
            'n_variants': lambda w: w.n_variants,
            'n_singletons': lambda w: diversity.singleton_count(w.matrix, missing_data=self.missing_data),
            'segregating_sites': lambda w: diversity.segregating_sites(w.matrix, missing_data=self.missing_data),
            'het_expected': lambda w: _window_mean_het_expected(
                w.matrix, self.missing_data),
        }
        for i, name in enumerate(_FUSED_GARUD_STATS):
            self.SINGLE_POP_STATS[name] = lambda w, i=i: self._garud_h(w)[i]

        self.TWO_POP_STATS = {
            'dxy': lambda w, p1, p2: divergence.dxy(
                w.matrix, p1, p2,
                missing_data=self.missing_data,
                span_normalize=sn),
            'fst': lambda w, p1, p2: divergence.fst(w.matrix, p1, p2, missing_data=self.missing_data),
            'fst_hudson': lambda w, p1, p2: divergence.fst_hudson(w.matrix, p1, p2, missing_data=self.missing_data),
            'fst_wc': lambda w, p1, p2: divergence.fst_weir_cockerham(w.matrix, p1, p2, missing_data=self.missing_data),
            'da': lambda w, p1, p2: divergence.da(w.matrix, p1, p2,
                                                missing_data=self.missing_data,
                                                span_normalize=sn),
        }

        self.single_pop_stats = []
        self.two_pop_stats = []
        self.ld_stats = []
        self.custom_stats = []

        for stat in self.statistics:
            if isinstance(stat, str):
                if stat in self.SINGLE_POP_STATS:
                    self.single_pop_stats.append(stat)
                elif stat in self.TWO_POP_STATS:
                    self.two_pop_stats.append(stat)
                elif stat in self.LD_STATS:
                    self.ld_stats.append(stat)
                else:
                    raise ValueError(f"Unknown statistic: {stat}")
            else:
                # Custom callable
                self.custom_stats.append(stat)

    def compute(self, window: WindowData) -> Dict[str, float]:
        """Compute all requested statistics for a window."""
        results = {
            'chrom': window.chrom,
            'start': window.start,
            'end': window.end,
            'center': window.center,
            'n_variants': window.n_variants,
            'window_id': window.window_id,
        }

        # One n_variants column serves every population, so under 'exclude'
        # it counts the sites complete in all of them, as the fused engines do.
        window = replace(window, n_variants=self._n_sites(
            window.matrix, self.populations or None))
        results['n_variants'] = window.n_variants

        # Single population statistics. Each population's window is built
        # once, so statistics that share work (the Garud H family) can
        # share it through the window's cache. The windows sit in a list in
        # the order of self.populations, because a population given as a
        # list of rows cannot be a dict key.
        pop_windows = []
        if self.single_pop_stats and self.populations:
            for pop in self.populations:
                pop_matrix = population_rows(window.matrix, pop)
                pop_windows.append(replace(
                    window, matrix=pop_matrix,
                    n_variants=self._n_sites(pop_matrix), cache={}))
        for stat in self.single_pop_stats:
            if self.populations:
                # Compute for each population
                for pop, pop_window in zip(self.populations, pop_windows):
                    val = self.SINGLE_POP_STATS[stat](pop_window)
                    key = f"{stat}_{pop}"
                    self._store_result(results, key, val)
            else:
                val = self.SINGLE_POP_STATS[stat](window)
                self._store_result(results, stat, val)
        # The complete haplotypes that anchor each Garud estimate, named
        # like the Garud columns.
        if any(s in _FUSED_GARUD_STATS for s in self.single_pop_stats):
            targets = ({f"_{p}": w for p, w in zip(self.populations, pop_windows)}
                       or {"": window})
            for suffix, w in targets.items():
                results[f"n_complete_haplotypes{suffix}"] = self._garud_h(w)[-1]

        # Two population statistics
        if len(self.populations) >= 2:
            for stat in self.two_pop_stats:
                for i, pop1 in enumerate(self.populations):
                    for pop2 in self.populations[i+1:]:
                        key = f"{stat}_{pop1}_{pop2}"
                        val = self.TWO_POP_STATS[stat](window, pop1, pop2)
                        self._store_result(results, key, val)

        # LD statistics
        for stat in self.ld_stats:
            kwargs = self.custom_stat_kwargs.get(stat, {})
            # Add default bins for ld_decay if not provided
            if stat == 'ld_decay' and 'bins' not in kwargs:
                kwargs['bins'] = self.ld_bins
            results[stat] = self.LD_STATS[stat](window, **kwargs)

        # Custom statistics. The callable gets the window's matrix as it is,
        # before any 'exclude' filter, and decides what to do with it; only a
        # window whose matrix has no variant gives NaN without a call, so
        # user code never has to handle an empty matrix.
        for stat in self.custom_stats:
            if window.matrix.num_variants == 0:
                results[stat.__name__] = np.nan
                continue
            kwargs = self.custom_stat_kwargs.get(stat.__name__, {})
            results[stat.__name__] = _outside_phase_guard(stat, window, **kwargs)

        return results

    def _garud_h(self, window):
        """The Garud H family of a window, in ``_FUSED_GARUD_STATS`` order,
        followed by the number of haplotypes with no missing call.

        One call gives all six values, as the fused engine computes them,
        and the window keeps them for the other statistics.
        """
        if 'garud_h' not in window.cache:
            matrix = window.matrix
            if matrix.device == 'CPU':
                matrix.transfer_to_gpu()
            *values, n_complete = garud_h_windows(
                matrix.haplotypes, [0], [matrix.num_variants],
                missing_data=self.missing_data)
            window.cache['garud_h'] = (*(float(v[0]) for v in values),
                                       int(n_complete[0]))
        return window.cache['garud_h']

    def _n_sites(self, matrix, populations=None):
        """Sites in ``matrix``; under 'exclude', only the complete ones."""
        n = matrix.num_variants
        if self.missing_data == 'exclude':
            n -= int(matrix._site_has_missing(populations).sum())
        return n

    @staticmethod
    def _store_result(results: Dict, key: str, val):
        """Store a scalar result into the results dict."""
        results[key] = val


def _fallback_stats():
    """Names of the statistics the per-window fallback can compute."""
    computer = StatisticsComputer([])
    return frozenset(computer.SINGLE_POP_STATS) | frozenset(
        computer.TWO_POP_STATS) | frozenset(computer.LD_STATS)


class WindowIterator:
    """Iterates over genomic windows."""

    def __init__(self, haplotype_matrix: HaplotypeMatrix,
                 window_params: WindowParams, skip_empty: bool = True):
        self.matrix = haplotype_matrix
        self.params = window_params
        # The windowed fallback keeps windows with no variant, so it gives
        # the same rows as the scatter and fused engines; local PCA, which
        # cannot decompose an empty window, skips them.
        self.skip_empty = skip_empty
        self.positions = self.matrix.positions

        # Get positions as numpy array for easier manipulation
        if isinstance(self.positions, cp.ndarray):
            self.positions_np = self.positions.get()
        else:
            self.positions_np = self.positions

    def _set_window_bases(self, window_matrix, start, end):
        """Set the window's n_total_sites to its base count in the parent.

        get_subset copies the parent's n_total_sites, which counts the
        whole region; the window needs its own count, by the rule that
        every engine uses (_compute_window_bases).
        """
        if self.matrix.has_accessible_mask or self.matrix.n_total_sites is not None:
            window_matrix.n_total_sites = float(
                _compute_window_bases(self.matrix, [start], [end])[0])

    def __iter__(self) -> Iterator[WindowData]:
        """Iterate over windows based on window type."""
        if self.params.window_type == 'bp':
            return self._iter_bp_windows()
        elif self.params.window_type == 'snp':
            return self._iter_snp_windows()
        elif self.params.window_type == 'regions':
            return self._iter_region_windows()
        else:
            raise ValueError(f"Unknown window type: {self.params.window_type}")

    def _iter_bp_windows(self) -> Iterator[WindowData]:
        """Iterate over fixed base pair windows."""
        if not self._has_grid():
            return
        chrom_start, chrom_end = _bp_grid_bounds(self.matrix,
                                                 self.positions_np)
        win_starts, win_stops = _bp_window_grid(
            chrom_start, chrom_end,
            self.params.window_size, self.params.step_size)

        for window_id, (start, end) in enumerate(zip(win_starts, win_stops)):
            start, end = int(start), int(end)
            # Variants in the right-open [start, end); positions are sorted.
            lo, hi = np.searchsorted(self.positions_np, (start, end))
            if hi > lo or not self.skip_empty:
                # Extract window matrix
                window_matrix = self.matrix.get_subset(np.arange(lo, hi))
                # Set correct chromosome coordinates for span normalization
                window_matrix.chrom_start = start
                window_matrix.chrom_end = end - 1  # end is exclusive in our window definition
                self._set_window_bases(window_matrix, start, end)

                yield WindowData(
                    chrom=1,  # TODO: Handle multiple chromosomes
                    start=start,
                    end=end,
                    center=(start + end) // 2,
                    matrix=window_matrix,
                    n_variants=hi - lo,
                    window_id=window_id
                )

    def _iter_snp_windows(self) -> Iterator[WindowData]:
        """Iterate over fixed SNP count windows."""
        n_variants = len(self.positions_np)
        window_id = 0
        start_idx = 0

        while start_idx + self.params.window_size <= n_variants:
            end_idx = start_idx + self.params.window_size

            # Get positions for this window
            window_start = int(self.positions_np[start_idx])
            window_end = int(self.positions_np[end_idx - 1])
            center = (window_start + window_end) // 2

            # Extract window matrix
            variant_indices = np.arange(start_idx, end_idx)
            window_matrix = self.matrix.get_subset(variant_indices)
            # Set correct chromosome coordinates for span normalization
            window_matrix.chrom_start = window_start
            window_matrix.chrom_end = window_end
            self._set_window_bases(window_matrix, window_start, window_end + 1)

            yield WindowData(
                chrom=1,  # TODO: Handle multiple chromosomes
                start=window_start,
                end=window_end,
                center=center,
                matrix=window_matrix,
                n_variants=len(variant_indices),
                window_id=window_id
            )

            window_id += 1
            start_idx += self.params.step_size

    def _iter_region_windows(self) -> Iterator[WindowData]:
        """Iterate over custom regions."""
        if self.params.regions is None:
            raise ValueError("Regions must be provided for region window type")

        for window_id, region in self.params.regions.iterrows():
            start = int(region['start'])
            end = int(region['end'])
            center = (start + end) // 2

            # Find variants in region
            mask = (self.positions_np >= start) & (self.positions_np < end)
            variant_indices = np.where(mask)[0]

            if len(variant_indices) > 0 or not self.skip_empty:
                window_matrix = self.matrix.get_subset(variant_indices)
                window_matrix.chrom_start = start
                window_matrix.chrom_end = end - 1 # TODO: treat region bounds as inclusive.
                self._set_window_bases(window_matrix, start, end)

                yield WindowData(
                    chrom=region.get('chrom', 1),
                    start=start,
                    end=end,
                    center=center,
                    matrix=window_matrix,
                    n_variants=len(variant_indices),
                    window_id=window_id
                )

    def _has_grid(self):
        """Whether the bp grid has bounds to tile: the matrix bounds, else
        its first and last variant. A matrix with no variant and no bounds
        has none; with bounds, its windows are empty but still tiled."""
        return len(self.positions_np) > 0 or (
            self.matrix.chrom_start is not None
            and self.matrix.chrom_end is not None)

    def count_windows(self) -> int:
        """Count total number of windows."""
        if self.params.window_type == 'bp':
            if not self._has_grid():
                return 0
            chrom_start, chrom_end = _bp_grid_bounds(self.matrix,
                                                     self.positions_np)
            win_starts, _ = _bp_window_grid(
                chrom_start, chrom_end,
                self.params.window_size, self.params.step_size)
            return len(win_starts)
        elif self.params.window_type == 'snp':
            n_variants = len(self.positions_np)
            if n_variants <= self.params.window_size:
                return 1
            else:
                # Number of complete windows plus any partial window
                return ((n_variants - self.params.window_size) // self.params.step_size) + 1
        elif self.params.window_type == 'regions':
            return len(self.params.regions)


class WindowedAnalyzer:
    """Main class for windowed analysis of genomic data."""

    def __init__(self,
                 window_type: str = 'bp',
                 window_size: int = 50000,
                 step_size: Optional[int] = None,
                 statistics: List[Union[str, Callable]] = ['pi'],
                 populations: Optional[List[str]] = None,
                 regions: Optional[pd.DataFrame] = None,
                 ld_max_distance: int = 10000,
                 ld_bins: Optional[List[int]] = None,
                 gpu_memory_limit: Union[str, int] = 'auto',
                 chunk_size: Union[str, int] = 'auto',
                 n_jobs: int = 1,
                 progress_bar: bool = True,
                 custom_stat_kwargs: Optional[Dict] = None,
                 missing_data: str = 'include',
                 span_normalize=True):
        """
        Initialize windowed analyzer.

        Parameters
        ----------
        window_type : str
            Type of windows: 'bp' (base pairs), 'snp' (SNP count), or 'regions'
        window_size : int
            Size of windows (in bp or SNP count)
        step_size : int, optional
            Step between windows. If None, uses window_size (non-overlapping)
        statistics : list
            Statistics to compute. Can be strings or callable functions
        populations : list, optional
            Population names for population-specific statistics
        regions : DataFrame, optional
            Custom regions for window_type='regions'
        ld_max_distance : int
            Maximum distance for LD calculations
        ld_bins : list, optional
            Distance bins for LD decay
        gpu_memory_limit : str or int
            GPU memory limit ('auto', '8GB', or bytes)
        chunk_size : str or int
            Chunk size for processing ('auto' or number of variants)
        n_jobs : int
            Number of CPU threads
        progress_bar : bool
            Show progress bar
        custom_stat_kwargs : dict
            Keyword arguments for custom statistics
        missing_data : str
            'include' - Use all sites, calculate from available data per site
            'exclude' - Only use sites with no missing data
        span_normalize : bool
            ``True`` (default): auto-detect best denominator per window.
            ``False``: return raw sums.
        """
        self.window_params = WindowParams(
            window_type=window_type,
            window_size=window_size,
            step_size=step_size or window_size,
            regions=regions
        )

        self.statistics = statistics
        self.populations = populations
        self.ld_max_distance = ld_max_distance
        self.ld_bins = ld_bins or [0, 1000, 5000, 10000, 50000]
        self.gpu_memory_limit = gpu_memory_limit
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar
        self.missing_data = missing_data
        self.span_normalize = span_normalize

        # Initialize components
        self.stats_computer = StatisticsComputer(
            statistics, populations, custom_stat_kwargs, ld_bins=self.ld_bins,
            missing_data=missing_data, span_normalize=span_normalize
        )

    @_phase_dependent
    def compute(self, haplotype_matrix: HaplotypeMatrix) -> pd.DataFrame:
        """
        Compute windowed statistics for entire haplotype matrix.

        Depends on haplotype phase when a phase-dependent statistic is
        requested; see :doc:`phasing`.

        Parameters
        ----------
        haplotype_matrix : HaplotypeMatrix
            Input haplotype data

        Returns
        -------
        pd.DataFrame
            Results with statistics for each window
        """
        phase_ctx = _phase_scan_context("WindowedAnalyzer.compute", self.statistics)
        if phase_ctx:
            _warn_unphased(haplotype_matrix, context=phase_ctx)
        # Set population info if needed
        if self.populations and not haplotype_matrix.sample_sets:
            warnings.warn("Populations specified but haplotype_matrix has no sample_sets")

        # Create window iterator
        window_iter = WindowIterator(haplotype_matrix, self.window_params,
                                     skip_empty=False)
        total_windows = window_iter.count_windows()

        # Process windows
        results = []
        with tqdm(total=total_windows, disable=not self.progress_bar,
                 desc="Computing windows") as pbar:
            for window in window_iter:
                window_results = self.stats_computer.compute(window)
                results.append(window_results)
                pbar.update(1)

        return pd.DataFrame(results)

    @_phase_dependent
    def compute_region(self, haplotype_matrix: HaplotypeMatrix,
                      chrom: Union[str, int],
                      start: int,
                      end: int) -> pd.DataFrame:
        """
        Compute statistics for a specific genomic region.

        Depends on haplotype phase when a phase-dependent statistic is
        requested; see :doc:`phasing`.

        Parameters
        ----------
        haplotype_matrix : HaplotypeMatrix
            Input haplotype data
        chrom : str or int
            Chromosome identifier
        start : int
            Region start position (inclusive).
        end : int
            Region end position (exclusive), as in ``get_subset_from_range``.

        Returns
        -------
        pd.DataFrame
            Results for windows in the specified region
        """
        phase_ctx = _phase_scan_context("WindowedAnalyzer.compute_region", self.statistics)
        if phase_ctx:
            _warn_unphased(haplotype_matrix, context=phase_ctx)
        # Extract region from matrix
        region_matrix = haplotype_matrix.get_subset_from_range(start, end)

        # Compute statistics
        return self.compute(region_matrix)

    @_phase_dependent
    def compute_streaming(self, haplotype_matrix: HaplotypeMatrix,
                         batch_size: int = 100) -> Iterator[pd.DataFrame]:
        """
        Compute statistics in batches for memory efficiency.

        Depends on haplotype phase when a phase-dependent statistic is
        requested; see :doc:`phasing`.

        Parameters
        ----------
        haplotype_matrix : HaplotypeMatrix
            Input haplotype data
        batch_size : int
            Number of windows per batch

        Yields
        ------
        pd.DataFrame
            Batch of results
        """
        phase_ctx = _phase_scan_context("WindowedAnalyzer.compute_streaming", self.statistics)
        if phase_ctx:
            _warn_unphased(haplotype_matrix, context=phase_ctx)
        return self._iter_batches(haplotype_matrix, batch_size)

    def _iter_batches(self, haplotype_matrix, batch_size):
        # Iteration runs after compute_streaming has returned and reset the
        # phase depth, so each window runs inside its own phase-dependent
        # frame; the statistics it reaches then stay silent.
        compute_window = _phase_dependent(self.stats_computer.compute)
        window_iter = WindowIterator(haplotype_matrix, self.window_params,
                                     skip_empty=False)
        total_windows = window_iter.count_windows()

        batch_results = []
        with tqdm(total=total_windows, disable=not self.progress_bar,
                 desc="Computing windows") as pbar:
            for window in window_iter:
                window_results = compute_window(window)
                batch_results.append(window_results)
                pbar.update(1)

                if len(batch_results) >= batch_size:
                    yield pd.DataFrame(batch_results)
                    batch_results = []

            # Yield remaining results
            if batch_results:
                yield pd.DataFrame(batch_results)


def _bp_window_grid(chrom_start, chrom_end, window_size, step_size):
    """Start/stop arrays of the bp window grid.

    Starts run from chrom_start to chrom_end inclusive in steps of
    step_size; stops clip at chrom_end + 1. Both bounds are bases, so a
    variant at chrom_end is in a window and each span counts only bases
    that exist. Shared by every windowed engine so the grids agree by
    construction.
    """
    win_starts = np.arange(int(chrom_start), int(chrom_end) + 1, step_size,
                           dtype=np.float64)
    return win_starts, np.minimum(win_starts + window_size,
                                  int(chrom_end) + 1)


def _window_variant_ranges(positions, win_starts, win_stops):
    """Each window's contiguous variant-index range ``[start, stop)``.

    ``positions`` is a sorted array, so a binary search of each window
    bound gives the range endpoints.

    Returns a pair ``(start, stop)`` of CuPy int64 arrays, each the same length
    as ``win_starts`` (one entry per window): window ``w``'s range is
    ``[start[w], stop[w])``.
    """
    pos = positions if isinstance(positions, cp.ndarray) else cp.asarray(positions)
    ws = cp.asarray(win_starts, dtype=cp.float64)
    we = cp.asarray(win_stops, dtype=cp.float64)
    start = cp.searchsorted(pos, ws, side='left').astype(cp.int64)
    stop = cp.searchsorted(pos, we, side='left').astype(cp.int64)
    return start, stop


class _WindowAccumulator:
    """Per-window reductions over each window's contiguous variant range.

    Built from the variant-index ranges ``_window_variant_ranges`` returns, it
    answers per-window ``sum`` / ``mean`` / ``counts`` as prefix-sum range
    queries -- ``csum[stop] - csum[start]`` -- plus ``max`` (a host-side range
    reduction, since cupy's indexed max loses float64 precision), so a per-site
    quantity is binned into windows without materializing per-(variant, window)
    membership. Per-site consumers call these methods instead of touching the
    raw ranges, so they cannot silently no-op when windows overlap or regions
    are unordered: every window family is served because each window's range is
    found on its own.
    """

    def __init__(self, positions, win_starts, win_stops):
        self.start, self.stop = _window_variant_ranges(
            positions, win_starts, win_stops)
        self.n_windows = int(self.start.shape[0])

    def counts(self):
        """Variants per window (int64)."""
        return self.stop - self.start

    def sum(self, per_var):
        """Per-window sum of a per-variant array (float64)."""
        per_var = cp.asarray(per_var, dtype=cp.float64)
        csum = cp.concatenate([cp.zeros(1, dtype=cp.float64),
                               cp.cumsum(per_var, dtype=cp.float64)])
        return csum[self.stop] - csum[self.start]

    def sum2d(self, per_var):
        """Per-window column sums of a ``(n_var, k)`` array -> ``(n_windows, k)``.

        The 2-D cumsum form of ``sum``, for a per-site histogram (daf_hist)
        binned into windows in one pass.
        """
        per_var = cp.asarray(per_var, dtype=cp.float64)
        k = per_var.shape[1]
        csum = cp.concatenate([cp.zeros((1, k), dtype=cp.float64),
                               cp.cumsum(per_var, axis=0)])
        return csum[self.stop] - csum[self.start]

    def mean(self, per_var, valid=None):
        """Per-window mean over the valid entries, NaN where a window has none.

        ``per_var`` is ``(n_var,)`` or ``(n_var, k)`` (e.g. one nSL score per
        derived allele); a 2-D input averages over every valid ``(variant,
        item)`` entry, so the divisor is the count of valid entries, not of
        variants. ``valid`` is a mask of ``per_var``'s shape; masked-out entries
        never reach the sum, and by default it masks out the NaN (non-finite)
        entries.
        """
        per_var = cp.asarray(per_var, dtype=cp.float64)
        if valid is None:
            valid = cp.isfinite(per_var)
        valid = valid.astype(cp.float64)
        num = cp.where(valid > 0, per_var, 0.0)
        if per_var.ndim > 1:
            # Reduce each variant's items to a finite sum and count first; the
            # window mean is then two prefix-sum range queries over those.
            axes = tuple(range(1, per_var.ndim))
            num = num.sum(axis=axes)
            valid = valid.sum(axis=axes)
        total = self.sum(num)
        count = self.sum(valid)
        return cp.where(count > 0, total / count, cp.nan)

    def max(self, per_var, init=0.0):
        """Per-window maximum of a per-variant array; empty windows keep ``init``.

        Reduced on the host and returned as a numpy array, unlike the other
        reducers: max is not a prefix-sum query, and cupy's indexed max reduces
        in float32, losing the float64 precision the other engines keep.
        """
        out = np.full(self.n_windows, init, dtype=np.float64)
        start = self.start.get()
        counts = self.stop.get() - start
        total = int(counts.sum())
        if total:
            vals = cp.asarray(per_var, dtype=cp.float64).get()
            win_idx = np.repeat(np.arange(self.n_windows), counts)
            base = np.repeat(np.cumsum(counts) - counts, counts)
            var_idx = np.repeat(start, counts) + (np.arange(total) - base)
            np.maximum.at(out, win_idx, vals[var_idx])
        return out

def _windows_emptied_by_exclude(pre_positions, post_windows, win_starts, win_stops):
    pre_windows = _WindowAccumulator(pre_positions, win_starts, win_stops)
    return ((pre_windows.counts() > 0) & (post_windows.counts() == 0)).get()

def _achaz_coeffs_per_window(uniq_n, inv_idx, w1, w2):
    """(alpha, beta) per window from _achaz_variance_coefficients(w1, w2, n).

    ``uniq_n``/``inv_idx`` are ``np.unique(n_harm_w, return_inverse=True)``,
    computed once by the caller and shared across every weight pair it needs
    for this ``n_harm_w`` -- a caller requesting several neutrality tests in
    one call (e.g. tajimas_d and zeng_e) evaluates the same set of distinct
    n only once, not once per pair. NaN for n < 3.
    """
    from .diversity import _achaz_variance_coefficients

    alpha_lut = np.full(uniq_n.shape, np.nan)
    beta_lut = np.full(uniq_n.shape, np.nan)
    for i, nv in enumerate(uniq_n):
        if nv >= 3:
            alpha_lut[i], beta_lut[i] = _achaz_variance_coefficients(w1, w2, int(nv))
    return alpha_lut[inv_idx], beta_lut[inv_idx]


def _windowed_thetas_scatter(haplotype_matrix, window_size, step_size,
                              statistics, populations, missing_data,
                              span_normalize, chrom=None,
                              win_starts=None, win_stops=None):
    """Compute windowed theta estimators via scatter-add on GPU.

    Uses per-allele counts (``allele_counts``) + per-variant contributions
    (``_ac_contribution``) + per-window range sums (_WindowAccumulator) for
    per-window accumulation, so multiallelic sites are handled per-allele.
    Handles variable sample sizes per site.

    ``win_starts`` / ``win_stops`` give explicit window bounds (custom regions,
    possibly overlapping or unordered); without them the bp grid is built from
    ``window_size`` / ``step_size``. Either way each window's per-site sum is a
    prefix-sum range query, so the two window families share one code path.
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    pop = populations[0] if populations else None
    if pop is not None:
        matrix = population_rows(haplotype_matrix, pop)
    else:
        matrix = haplotype_matrix

    # Anchor the window grid before any missing-data filtering: a matrix
    # without chromosome bounds falls back to its variant hull, which
    # moves when a terminal site is incomplete, handing every statistic
    # in a mixed request its own grid.
    pos_cpu = cp.asnumpy(matrix.positions)
    chrom_start, chrom_end = _bp_grid_bounds(matrix, pos_cpu)
    if chrom_end < chrom_start:
        return pd.DataFrame()

    from .diversity import _prepare_allele_counts, _ac_contribution
    ac, n_valid, n_hap = _prepare_allele_counts(matrix)
    pos_cpu_pre_exclude = pos_cpu
    if missing_data == 'exclude':
        # This engine reads the matrix only through its per-site allele
        # counts, so 'exclude' keeps the counts of the complete sites and
        # never copies the matrix. The already-anchored window grid stays
        # even when no site is complete: empty windows render as 0/NaN rows
        # (as on the include path), so an eager all-missing region and a
        # streaming empty chunk both yield dense windows.
        complete = n_valid == n_hap
        if not bool(complete.all()):
            ac, n_valid = ac[complete], n_valid[complete]
            pos_cpu = pos_cpu[complete.get()]
        all_complete = True
    else:
        all_complete = cp.all(n_valid == n_hap)
    # Per-allele intermediates (multiallelic-correct; match the scalar diversity
    # path). mut = per-variant mutation count = (#alleles present) - 1.
    n = n_valid.astype(cp.float64)
    has_data = n_valid >= 2
    alleles_present = (ac > 0).sum(axis=1)
    mut = cp.where(has_data, cp.maximum(alleles_present - 1, 0), 0).astype(cp.float64)

    if win_starts is None:
        win_starts, win_stops = _bp_window_grid(
            chrom_start, chrom_end, window_size, step_size)
    n_windows = len(win_starts)
    # Each window's per-variant sum is a prefix-sum range query over its
    # contiguous variant range (the shared membership rule), so overlapping or
    # unordered windows accumulate a variant into every window covering it
    # without replicating it across candidate windows.
    windows = _WindowAccumulator(pos_cpu, win_starts, win_stops)
    scatter_sum = windows.sum

    # Compute requested per-variant contributions and scatter
    n_variants_per_window = windows.counts().get()
    results = _init_window_results(
        chrom, win_starts, win_stops,
        n_variants=n_variants_per_window)

    spans, zero_span = _scatter_spans(haplotype_matrix, win_starts, win_stops,
                                      span_normalize)

    stats_set = set(statistics)

    # A window that lost every site to 'exclude' is undefined (NaN) for the
    # statistics below; one with no variant to begin with stays 0.0.
    excluded_to_empty = None
    if missing_data == 'exclude' and (stats_set & (_RATE_STATS | {'max_daf'})):
        excluded_to_empty = _windows_emptied_by_exclude(
            pos_cpu_pre_exclude, windows, win_starts, win_stops)

    # Determine which theta estimators are needed as intermediates
    needs = {
        'pi': stats_set & {'pi', 'tajimas_d', 'fay_wu_h', 'normalized_fay_wu_h', 'zeng_dh'},
        'theta_h': stats_set & {'theta_h', 'fay_wu_h', 'normalized_fay_wu_h', 'zeng_dh'},
        'theta_l': stats_set & {'theta_l', 'zeng_e'},
        'watterson': stats_set & {'theta_w', 'tajimas_d', 'zeng_e', 'zeng_dh'},
    }

    # Compute per-variant contributions via _ac_contribution (single source of truth)
    raw = {}
    for est_name, dependents in needs.items():
        if dependents:
            raw[est_name] = scatter_sum(
                _ac_contribution(est_name, ac, n_valid, n_hap))

    # Without this, a site with too few calls to contribute would still
    # count toward the span, reading as invariant rather than unmeasured.
    rescale = None
    if bool(stats_set & _RATE_STATS):
        rescale = _missing_data_rescale(
            n_variants_per_window, span_normalize, all_complete,
            lambda: scatter_sum(has_data.astype(cp.float64)).get())

    if stats_set & {'segregating_sites', 'tajimas_d', 'normalized_fay_wu_h', 'zeng_e', 'zeng_dh'}:
        seg_count = scatter_sum(mut)
    derived = ac[:, 1:]
    if 'singletons' in stats_set:
        # Per-allele: derived alleles (columns >= 1) observed exactly once.
        n_sing = ((derived == 1).sum(axis=1) if derived.shape[1]
                  else cp.zeros(ac.shape[0], dtype=cp.int64))
        sing_count = scatter_sum(cp.where(has_data, n_sing, 0).astype(cp.float64))
    if 'max_daf' in stats_set:
        # Per-allele: highest derived-allele frequency at the site, reduced
        # to each window's range max (empty windows stay 0, as before).
        if derived.shape[1]:
            p_der = derived.astype(cp.float64) / cp.where(n > 0, n, 1.0)[:, None]
            dafs = cp.where(has_data, p_der.max(axis=1), 0.0)
        else:
            dafs = cp.zeros(ac.shape[0], dtype=cp.float64)
        max_daf_arr = windows.max(dafs)

    # Build output — theta estimators as per-base rates
    for est_name, out_name in [('pi', 'pi'), ('watterson', 'theta_w'),
                                ('theta_h', 'theta_h'), ('theta_l', 'theta_l')]:
        if out_name in stats_set and est_name in raw:
            vals = raw[est_name].get()
            if rescale is not None:
                vals = vals * rescale
            vals = vals / spans
            if excluded_to_empty is not None:
                vals = np.where(excluded_to_empty, np.nan, vals)
            results[out_name] = vals

    if 'segregating_sites' in stats_set:
        results['segregating_sites'] = seg_count.get()
    if 'singletons' in stats_set:
        results['singletons'] = sing_count.get()
    if 'max_daf' in stats_set:
        if excluded_to_empty is not None:
            max_daf_arr = np.where(excluded_to_empty, np.nan, max_daf_arr)
        results['max_daf'] = max_daf_arr

    # Composite stats from raw theta sums. fay_wu_h is a difference of two
    # estimators, not a rate, so it has no span-normalized form.
    if 'fay_wu_h' in stats_set:
        results['fay_wu_h'] = (raw['pi'] - raw['theta_h']).get()

    # Neutrality tests, with the effective sample size the per-window
    # harmonic mean of n_valid rather than the nominal n_hap.
    if stats_set & _ACHAZ_STATS:
        from .diversity import _harmonic_mean_n
        if bool(all_complete):
            # No missing data anywhere: every window's harmonic mean is
            # trivially n_hap, so skip the two scatter passes entirely.
            n_harm_w = np.full(n_windows, n_hap, dtype=np.int64)
        else:
            inv_valid = cp.where(has_data, 1.0 / n, 0.0)
            sum_inv_w, count_valid_w = cp.stack([
                scatter_sum(inv_valid), scatter_sum(has_data.astype(cp.float64))
            ]).get()
            n_harm_w = _harmonic_mean_n(count_valid_w, sum_inv_w)
        _achaz_test_columns(results, stats_set, seg_count.get(), n_harm_w,
                            lambda name: raw[name].get())

    # Windows with no accessible bases get NaN for every per-base rate.
    # fay_wu_h and zeng_dh have no span-normalized form, so a zero-span
    # window doesn't make them undefined.
    if zero_span.any():
        for name in ('pi', 'theta_w', 'theta_h', 'theta_l'):
            if name in results:
                results[name] = np.where(zero_span, np.nan, results[name])

    return pd.DataFrame(results)


def _windowed_twopop_scatter(haplotype_matrix, window_size, step_size,
                              statistics, populations, missing_data,
                              span_normalize, chrom=None,
                              win_starts=None, win_stops=None):
    """Compute windowed two-population stats via scatter-add on GPU.

    Same pattern as _windowed_thetas_scatter but for fst, fst_wc, dxy, da,
    including the explicit ``win_starts`` / ``win_stops`` region bounds.
    Uses per-site valid counts for correct missing data handling.
    """
    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    stats_set = set(statistics)
    pop1_name, pop2_name = populations[0], populations[1]

    if 'fst_wc' in stats_set:
        # fst_wc pairs consecutive rows of each population into individuals;
        # the other statistics here take any row list, so this check is
        # specific to fst_wc, mirroring the fused engine's equivalent.
        from ._warnings import check_paired_rows_for_populations
        check_paired_rows_for_populations(
            haplotype_matrix.sample_sets, pop1_name, pop2_name,
            "windowed fst_wc")

    mat1 = population_rows(haplotype_matrix, pop1_name)
    mat2 = population_rows(haplotype_matrix, pop2_name)

    # Anchor the window grid before any missing-data filtering, so this
    # engine tiles the same grid as every other statistic in the request.
    pos_cpu = cp.asnumpy(haplotype_matrix.positions)
    chrom_start, chrom_end = _bp_grid_bounds(haplotype_matrix, pos_cpu)
    if chrom_end < chrom_start:
        return pd.DataFrame()

    hap1 = mat1.haplotypes
    hap2 = mat2.haplotypes

    pos_cpu_pre_exclude = pos_cpu
    complete = None
    if missing_data == 'exclude':
        # This engine reads the populations only through per-site arrays,
        # so 'exclude' keeps the entries of the sites complete in both
        # populations and does not copy the rows again. The anchored grid stays
        # even when no site is complete, so empty windows render as 0/NaN
        # rows instead of vanishing (mirrors the single-pop engine).
        complete = ~(column_has_missing(hap1) | column_has_missing(hap2))
        if bool(complete.all()):
            complete = None
        else:
            pos_cpu = pos_cpu[complete.get()]

    def kept(*per_site):
        """The complete sites' entries of per-site arrays."""
        if complete is None:
            return per_site
        return tuple(v[complete] for v in per_site)

    if win_starts is None:
        win_starts, win_stops = _bp_window_grid(
            chrom_start, chrom_end, window_size, step_size)
    n_windows = len(win_starts)
    # Same shared membership rule as the single-pop engine: each window's
    # per-site sum is a prefix-sum range query over its contiguous variant
    # range, so overlapping or unordered windows need no replicated-index
    # scatter.
    windows = _WindowAccumulator(pos_cpu, win_starts, win_stops)
    scatter_sum = windows.sum
    n_variants_per_window = windows.counts().get()

    results = _init_window_results(
        chrom, win_starts, win_stops,
        n_variants=n_variants_per_window)

    spans, zero_span = _scatter_spans(haplotype_matrix, win_starts, win_stops,
                                      span_normalize)

    # A window that lost every site to 'exclude' is undefined (NaN) for
    # dxy/da; one with no variant to begin with stays 0.0.
    excluded_to_empty = None
    if missing_data == 'exclude' and (stats_set & {'dxy', 'da'}):
        excluded_to_empty = _windows_emptied_by_exclude(
            pos_cpu_pre_exclude, windows, win_starts, win_stops)

    # Compute per-site components (single pass over the data). Only needed
    # for the gamete statistics, not a pure fst_wc request. dxy and the
    # fst/da terms need different site sets (see _dxy_per_site_from_counts
    # and _site_components_from_counts), so they get separate sums even
    # though both are requested here.
    need_fst_da = stats_set & {'fst', 'fst_hudson', 'da'}
    need_dxy = 'dxy' in stats_set
    if need_fst_da or need_dxy:
        ac1, ac2, nv1, nv2 = kept(*divergence._aligned_pop_counts(hap1, hap2))
        dxy_all_complete = cp.all((nv1 > 0) & (nv2 > 0))
        da_all_complete = cp.all((nv1 > 1) & (nv2 > 1))
        if need_fst_da:
            mpd1, mpd2, between = divergence._site_components_from_counts(
                ac1, nv1, ac2, nv2)
            between_sum = scatter_sum(between)
        if need_dxy:
            dxy_per_site = divergence._dxy_per_site_from_counts(
                ac1, nv1, ac2, nv2)
            dxy_sum = scatter_sum(dxy_per_site)

    if 'fst_wc' in stats_set:
        from .divergence import _wc_site_components
        k = max(int(hap1.max()) if hap1.size else 0,
                int(hap2.max()) if hap2.size else 0, 0) + 1
        wc_a, wc_abc = kept(*_wc_site_components(hap1, hap2, k))
        wc_a_sum = scatter_sum(wc_a)
        wc_abc_sum = scatter_sum(wc_abc)

    if need_fst_da:
        within = (mpd1 + mpd2) / 2.0
        fst_num = scatter_sum(between - within)

    if 'da' in stats_set:
        pi1_sum = scatter_sum(mpd1)
        pi2_sum = scatter_sum(mpd2)

    # Without this, a site with too few calls to contribute would still
    # count toward the span, reading as invariant rather than unmeasured.
    # dxy and da need their own count, since they use different site gates.
    dxy_rescale = da_rescale = None
    if need_dxy:
        r = _missing_data_rescale(
            n_variants_per_window, span_normalize, dxy_all_complete,
            lambda: scatter_sum(((nv1 > 0) & (nv2 > 0)).astype(cp.float64)).get())
        dxy_rescale = cp.asarray(r) if r is not None else None
    if 'da' in stats_set:
        r = _missing_data_rescale(
            n_variants_per_window, span_normalize, da_all_complete,
            lambda: scatter_sum(((nv1 > 1) & (nv2 > 1)).astype(cp.float64)).get())
        da_rescale = cp.asarray(r) if r is not None else None

    # Build output — stay on GPU, single .get() per result
    spans_gpu = cp.asarray(spans)

    if stats_set & {'fst', 'fst_hudson'}:
        fst_vals = cp.where(between_sum > 0, fst_num / between_sum, cp.nan).get()
        if 'fst' in stats_set:
            results['fst'] = fst_vals
        if 'fst_hudson' in stats_set:
            results['fst_hudson'] = fst_vals

    if 'dxy' in stats_set:
        vals = dxy_sum * dxy_rescale if dxy_rescale is not None else dxy_sum
        results['dxy'] = (vals / spans_gpu).get()

    if 'da' in stats_set:
        vals = between_sum - (pi1_sum + pi2_sum) / 2.0
        if da_rescale is not None:
            vals = vals * da_rescale
        results['da'] = (vals / spans_gpu).get()

    if 'fst_wc' in stats_set:
        results['fst_wc'] = cp.where(wc_abc_sum > 0, wc_a_sum / wc_abc_sum,
                                     cp.nan).get()

    # Windows with no accessible bases get NaN for every per-base rate.
    if zero_span.any():
        for name in ('dxy', 'da'):
            if name in results:
                results[name] = np.where(zero_span, np.nan, results[name])

    if excluded_to_empty is not None:
        for name in ('dxy', 'da'):
            if name in results:
                results[name] = np.where(excluded_to_empty, np.nan, results[name])

    return pd.DataFrame(results)


# Convenience function for simple usage
def _stream_windowed_analysis(streaming_hm, *, window_size, step_size,
                              statistics, populations, missing_data,
                              span_normalize, accessible_bed, chrom,
                              **kwargs) -> pd.DataFrame:
    """Run windowed_analysis chunk-by-chunk over a StreamingHaplotypeMatrix.

    The per-chunk DataFrames are concatenated row-wise. Each chunk's windows
    are computed in isolation and ``window_size`` must divide it ``chunk_bp``.

    Sliding windows (``step_size`` != ``window_size``) and the local-PCA
    dispatch both need cross-chunk state and are not yet supported on the
    streaming path; both raise rather than silently returning wrong
    results.

    ``mean_nsl`` is computed per chunk. Unlike the per-site statistics, nSL
    integrates the shared-haplotype length in both directions across every
    variant, so a per-chunk scan is truncated at the chunk boundaries: the
    streaming result approximates the eager whole-region computation rather
    than reproducing it. Materialize the region eagerly for an exact nSL scan.
    """
    if step_size is None:
        step_size = window_size
    if step_size != window_size:
        raise NotImplementedError(
            "sliding windows (step_size != window_size) over a "
            "StreamingHaplotypeMatrix would straddle chunk boundaries; "
            "supply non-overlapping windows or materialize the region "
            "eagerly first."
        )
    chunk_bp = streaming_hm.chunk_bp
    if window_size > chunk_bp:
        raise ValueError(
            f"window_size={window_size} must divide the streaming matrix's "
            f"chunk width ({chunk_bp}); pass a smaller window_size"
        )

    if chunk_bp % window_size != 0:
        raise ValueError(
            f"window_size={window_size} must divide the streaming matrix's "
            f"chunk width ({chunk_bp}); pass a different window_size or "
            f"re-open the store with a matching chunk_bp."
        )

    if not _LOCAL_PCA_STATS.isdisjoint(statistics):
        raise NotImplementedError(
            "local_pca requires a chromosome-wide reference and is not "
            "available on the StreamingHaplotypeMatrix path; materialize "
            "the region eagerly to run it."
        )

    # As on the eager path, accessible_bed applies only when the stream has
    # no mask of its own (iter_gpu_chunks attaches that one). Parse the BED
    # once up front and hand the resolved AccessibleMask to each chunk, not
    # the path, which each chunk would re-open and re-parse.
    shared_mask = None
    if accessible_bed is not None and streaming_hm.accessible_mask is None:
        # The stream knows its contig, so a multi-contig BED needs no chrom.
        shared_mask = resolve_accessible_mask(
            accessible_bed, streaming_hm.chrom_start, streaming_hm.chrom_end,
            chrom=chrom or streaming_hm.chrom,
        )

    parts = []
    for left, right, chunk_hm in streaming_hm.iter_gpu_chunks(skip_empty = False):
        if shared_mask is not None:
            chunk_hm.set_accessible_mask(shared_mask)
        df = _windowed_analysis(
            chunk_hm,
            window_size=window_size, step_size=step_size,
            statistics=statistics, populations=populations,
            missing_data=missing_data, span_normalize=span_normalize,
            accessible_bed=None, chrom=chrom,
            **kwargs,
        )
        if df is not None and len(df):
            parts.append(df)

    if not parts:
        # Empty source (e.g. mappable region contains no variants). Return
        # an empty frame; the eager path would have raised, but for
        # streaming this is a legitimate "no data in any chunk" outcome.
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Per-chunk window_id columns restart at 0 inside each chunk; re-number
    # so concatenated results carry a globally-unique window_id matching
    # what the eager path produces.
    if "window_id" in out.columns:
        out["window_id"] = np.arange(len(out), dtype=out["window_id"].dtype)
    return out


def _check_windowed_kwargs(kwargs, statistics):
    """Raise TypeError for a keyword argument that no engine reads.

    Only WindowedAnalyzer (the fallback) and the local-PCA dispatch read
    ``**kwargs``; the scatter and fused engines drop it. Without this check,
    a misspelled name changes nothing when a fast engine serves the request.
    """
    allowed = set(inspect.signature(WindowedAnalyzer.__init__).parameters) - {'self'}
    if not _LOCAL_PCA_STATS.isdisjoint(statistics):
        allowed |= _LOCAL_PCA_ONLY_KWARGS
    unknown = sorted(set(kwargs) - allowed)
    if not unknown:
        return
    named = set(inspect.signature(windowed_analysis).parameters) - {'kwargs'}
    lines = [f"windowed_analysis() got unexpected keyword arguments {unknown}."]
    for name in unknown:
        match = difflib.get_close_matches(name, sorted(allowed | named), n=1)
        if match:
            lines.append(f"Did you mean {match[0]!r} for {name!r}?")
    if not _LOCAL_PCA_ONLY_KWARGS.isdisjoint(unknown):
        lines.append(f"{sorted(_LOCAL_PCA_ONLY_KWARGS)} apply only to the "
                     "'local_pca' and 'local_pca_jackknife' statistics.")
    raise TypeError(" ".join(lines))


@_phase_dependent
def windowed_analysis(haplotype_matrix: HaplotypeMatrix,
                     window_size: int = 50000,
                     step_size: Optional[int] = None,
                     statistics: List[str] = ['pi'],
                     populations: Optional[List[str]] = None,
                     missing_data: str = 'include',
                     span_normalize=True,
                     accessible_bed: str = None,
                     chrom: str = None,
                     min_complete_haplotypes: Optional[int] = None,
                     window_type: str = 'bp',
                     **kwargs) -> pd.DataFrame:
    """
    Convenience function for windowed analysis.

    Depends on haplotype phase when a phase-dependent statistic is requested;
    see :doc:`phasing`.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Input haplotype data
    window_size : int
        Window size in base pairs if `window_type = 'bp'` or number of variants
        if `window_type='snp'`. Not used if `window_type = 'regions'`.
    step_size : int, optional
        Step size. If None, uses window_size
    statistics : list
        Statistics to compute. ``n_variants`` is a column of every result,
        and ``n_singletons`` is another name for ``singletons``. Statistics
        that no single engine computes (for example ``ld_decay`` with
        ``zns``) raise ValueError; request them in separate calls. A custom
        callable gets NaN on a window with no variant, without a call.
    populations : list, optional
        Population names
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
    span_normalize : bool
        ``True`` (default): divide each window by its accessible bases if
        the matrix has a mask; else by its share of ``n_total_sites``, in
        proportion to the window length; else by the window length.
        ``False``: return raw sums.
    accessible_bed : str, optional
        Path to a BED file defining accessible/callable regions.
        If provided and the matrix has no mask, loads the mask.
    chrom : str, optional
        Contig of the data, used to select its intervals from
        ``accessible_bed``. Required when the BED has more than one contig
        and the matrix is eager; a streaming matrix uses its own contig.
    min_complete_haplotypes : int, optional
        With a Garud statistic or ``haplotype_count``, set those statistics
        to NaN in every window with fewer than this many haplotypes with no
        missing call (the ``n_complete_haplotypes`` column). When it is not
        given, a ``FewCompleteHaplotypesWarning`` reports when many windows
        have few complete haplotypes, where H1 and H12 come out too low.
    window_type : str
        Type of windows: 'bp' (base pairs), 'regions', or 'snp'. If 'regions',
        then provide an additional kwarg `regions` as a pandas DataFrame with the
        form `pd.DataFrame({'start': [s1, s2, ...], 'end': [e1, e2, ...]})`. For
        example, if you want the regions 1000-5000 and 8000-9000 on the chromosome,
        then `regions = pd.DataFrame({'start': [1000, 8000], 'end': [5000, 9000]})`.
        Any number of regions can be specified. Region intervals are treated as half open,
        so a window spans [start, end). If 'snp' is given, then `window_size` counts the
        number of variants.
    **kwargs
        ``regions`` (with ``window_type='regions'``) is read by whichever
        engine serves the request. The other WindowedAnalyzer parameters (``ld_bins``,
        ``custom_stat_kwargs``, ``progress_bar`` and the rest) take effect
        only when the per-window fallback serves the request; the scatter
        and fused engines accept them and do not use them. For the
        local-PCA statistics, ``k``, ``population``, ``batch_size``,
        ``n_blocks`` and ``aggregate`` go to the local-PCA functions. Any
        other name raises ``TypeError``.

    Returns
    -------
    pd.DataFrame
        Windowed statistics results. A request with a Garud statistic or
        ``haplotype_count`` also gets ``n_complete_haplotypes``, the number
        of haplotypes with no missing call in each window (suffixed by
        population where the Garud columns are).
    """
    _check_windowed_kwargs(kwargs, statistics)
    valid_wt = ('bp', 'snp', 'regions')
    if window_type not in valid_wt:
        raise ValueError(f"Unknown window_type: {window_type!r}. Valid types are {valid_wt}.")
    if (kwargs.get('regions') is not None) and (window_type != 'regions'):
        raise ValueError("`regions` requested with incompatible `window_type`. Please pass `window_type = 'regions'`.")
    if (window_type == 'regions') & (kwargs.get('regions') is None):
        raise ValueError("`regions` kwarg must be provided for `window_type='regions'`.")
    if missing_data not in ('include', 'exclude'):
        raise ValueError("Unknown value for `missing_data`. Please use 'include' or 'exclude'.")
    phase_ctx = _phase_scan_context("windowed_analysis", statistics)
    if phase_ctx:
        _warn_unphased(haplotype_matrix, context=phase_ctx)
    result = _windowed_analysis(
        haplotype_matrix, window_size=window_size, step_size=step_size,
        statistics=_engine_statistics(statistics), populations=populations,
        missing_data=missing_data, span_normalize=span_normalize,
        accessible_bed=accessible_bed, chrom=chrom, window_type=window_type, **kwargs)
    # A local-PCA request returns a LocalPCAResult; its scalar columns sit
    # in .windows.
    frame = getattr(result, 'windows', result)
    _name_singletons_as_requested(frame, statistics)
    _check_complete_haplotypes(frame, haplotype_matrix, populations,
                               min_complete_haplotypes)
    return result


def _engine_statistics(statistics):
    """The request as the engines take it.

    ``n_variants`` is dropped, since every engine returns that column, so
    naming it does not send the request to the per-window fallback.
    ``n_singletons`` is another name for ``singletons``.
    """
    return list(dict.fromkeys(
        'singletons' if s == 'n_singletons' else s
        for s in statistics if s != 'n_variants'))


def _name_singletons_as_requested(df, statistics):
    """Give the singleton columns the ``n_singletons`` name when the caller
    asked for it, keeping the ``singletons`` ones only if asked for too."""
    if 'n_singletons' not in statistics:
        return
    cols = [c for c in df.columns
            if c == 'singletons' or str(c).startswith('singletons_')]
    for col in cols:
        df['n_' + col] = df[col]
    if 'singletons' not in statistics:
        df.drop(columns=cols, inplace=True)


def _check_complete_haplotypes(df, matrix, populations, min_complete):
    """Apply ``min_complete_haplotypes``, or warn when many windows are thin.

    Each ``n_complete_haplotypes`` column covers the Garud columns with the
    same population suffix. A bare column belongs to ``populations[0]``, or
    to every row when no population is given.
    """
    from ._warnings import _warn_few_complete_haplotypes
    prefix = 'n_complete_haplotypes'
    # The engines name a population's columns f"{stat}_{pop}", so map each
    # suffix back to the population itself, which may be a list of rows.
    pop_of = {f"_{p}": p for p in populations or []}
    pop_of[""] = populations[0] if populations else None
    for col in [c for c in df.columns if c.startswith(prefix)]:
        suffix = col[len(prefix):]
        n_complete = df[col].to_numpy(dtype=np.float64)
        if min_complete is not None:
            stats = df.columns.intersection([s + suffix for s in _FUSED_GARUD_STATS])
            df.loc[n_complete < min_complete, stats] = np.nan
            continue
        pop = pop_of[suffix]
        n_rows = (len(divergence._get_population_indices(matrix, pop))
                  if pop is not None else matrix.num_haplotypes)
        _warn_few_complete_haplotypes(n_complete, n_rows,
                                      context="windowed_analysis")


# The fallback's name for a statistic that the other engines name otherwise.
_FALLBACK_NAMES = {'singletons': 'n_singletons'}


def _check_fallback_covers(requested, window_type):
    """Raise for names the per-window fallback cannot compute.

    The fallback serves every request that no single fast engine covers, and
    is the only engine for ``snp`` windows, so a name it does not know means
    either a mix that no one engine computes or an unknown name.
    """
    fallback = _fallback_stats() | set(_FALLBACK_NAMES)
    missing = sorted(s for s in requested if isinstance(s, str)
                     and s not in fallback)
    if not missing:
        return
    engines = {'scatter': _SCATTER_SINGLE_STATS | _TWO_POP_STATS,
               'fused': _FUSED_STATS}
    unknown = [s for s in missing if not any(s in e for e in engines.values())]
    if unknown:
        raise ValueError(f"Unknown statistic: {', '.join(unknown)}")
    if window_type == 'snp':
        raise NotImplementedError(
            f"{missing} not yet supported for window_type='snp'.")
    where = '; '.join(
        f"{s} ({' or '.join(n for n, e in engines.items() if s in e)} engine)"
        for s in missing)
    raise ValueError(f"No single windowed engine computes this mix; only one "
                     f"engine computes {where}. Request it separately.")


def _windowed_analysis(haplotype_matrix: HaplotypeMatrix,
                      window_size: int = 50000,
                      step_size: Optional[int] = None,
                      statistics: List[str] = ['pi'],
                      populations: Optional[List[str]] = None,
                      missing_data: str = 'include',
                      span_normalize=True,
                      accessible_bed: str = None,
                      chrom: str = None,
                      window_type: str = 'bp',
                      **kwargs) -> pd.DataFrame:
    """The body of ``windowed_analysis``, without its final checks."""
    from .streaming_matrix import StreamingHaplotypeMatrix
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        if window_type in ('regions', 'snp'):
            raise NotImplementedError(
                "window_type='regions' or 'snp' is not available "
                "on a StreamingHaplotypeMatrix."
            )
        return _stream_windowed_analysis(
            haplotype_matrix,
            window_size=window_size, step_size=step_size,
            statistics=statistics, populations=populations,
            missing_data=missing_data, span_normalize=span_normalize,
            accessible_bed=accessible_bed, chrom=chrom,
            **kwargs,
        )

    if accessible_bed is not None and not haplotype_matrix.has_accessible_mask:
        haplotype_matrix.set_accessible_mask(accessible_bed, chrom=chrom)
    if step_size is None:
        step_size = window_size

    # Local PCA dispatch: vector-valued per-window output; cannot live in the
    # scalar-stat DataFrame pipeline. Return a LocalPCAResult (with scalar
    # stats merged into .windows if requested alongside).
    # 'local_pca_jackknife' implies 'local_pca' -- the SE is meaningless
    # without the base eigendecomposition.
    if not _LOCAL_PCA_STATS.isdisjoint(statistics):
        from . import decomposition
        want_jackknife = 'local_pca_jackknife' in statistics
        local_pca_kwargs = {
            'k': kwargs.get('k', 2),
            'missing_data': missing_data,
            'population': kwargs.get('population', None),
            'batch_size': kwargs.get('batch_size', None),
            'window_size': window_size,
            'step_size': step_size,
            'window_type': window_type,
            'regions': kwargs.get('regions', None),
        }
        if want_jackknife:
            local_pca_kwargs['n_blocks'] = kwargs.get('n_blocks', 10)
            local_pca_kwargs['aggregate'] = kwargs.get('aggregate', 'mean')
            result = decomposition._local_pca_with_jackknife(
                haplotype_matrix, **local_pca_kwargs)
        else:
            result = decomposition.local_pca(haplotype_matrix, **local_pca_kwargs)
        other_stats = [s for s in statistics
                       if s not in _LOCAL_PCA_STATS]
        if other_stats:
            scalar_df = _windowed_analysis(
                haplotype_matrix,
                window_size=window_size,
                step_size=step_size,
                statistics=other_stats,
                populations=populations,
                missing_data=missing_data,
                span_normalize=span_normalize,
                accessible_bed=None,  # already applied above
                chrom=chrom,
                window_type=window_type,
                **{k: v for k, v in kwargs.items()
                   if k not in _LOCAL_PCA_ONLY_KWARGS},
            )
            # result.windows already carries the canonical prefix; drop it
            # from scalar_df and merge on window_id to avoid duplicates.
            duplicates = [c for c in CANONICAL_WINDOW_PREFIX
                          if c != 'window_id' and c in scalar_df.columns]
            result.windows = result.windows.merge(
                scalar_df.drop(columns=duplicates),
                on='window_id', how='left')
        return result

    requested = set(statistics)

    # Explicit window bounds for region windows, shared by the scatter and
    # fused engines (bp grids are built per engine from window_size/step_size,
    # so win_starts/win_stops stay None there). Each window's variant range is
    # found on its own, so the scatter engine now serves regions too, the same
    # dispatch rules as bp.
    win_starts = win_stops = None
    if window_type == 'regions':
        regions = kwargs.get('regions')
        if regions is None:
            raise ValueError("Regions must be provided for region window type")
        win_starts = np.asarray(regions['start'], dtype=np.float64)
        win_stops = np.asarray(regions['end'], dtype=np.float64)
        if len(win_starts) == 0:
            return pd.DataFrame()

    if window_type in ('bp', 'regions'):
        # Pure single-pop request
        if requested <= _SCATTER_SINGLE_STATS and len(populations or []) <= 1:
            result = _windowed_thetas_scatter(
                haplotype_matrix, window_size, step_size,
                statistics, populations, missing_data, span_normalize,
                chrom=chrom, win_starts=win_starts, win_stops=win_stops)
            if result is not None:
                return result

        # Pure two-pop request
        if requested <= _TWO_POP_STATS and len(populations or []) == 2:
            result = _windowed_twopop_scatter(
                haplotype_matrix, window_size, step_size,
                statistics, populations, missing_data, span_normalize,
                chrom=chrom, win_starts=win_starts, win_stops=win_stops)
            if result is not None:
                return result

        # Mixed single + two-pop request
        single_stats = sorted(requested & _SCATTER_SINGLE_STATS)
        twopop_stats = sorted(requested & _TWO_POP_STATS)
        if (single_stats and twopop_stats
                and requested <= (_SCATTER_SINGLE_STATS | _TWO_POP_STATS)
                and len(populations or []) == 2):
            df1 = _windowed_thetas_scatter(
                haplotype_matrix, window_size, step_size,
                single_stats, [populations[0]], missing_data, span_normalize,
                chrom=chrom, win_starts=win_starts, win_stops=win_stops)
            df2 = _windowed_twopop_scatter(
                haplotype_matrix, window_size, step_size,
                twopop_stats, populations, missing_data, span_normalize,
                chrom=chrom, win_starts=win_starts, win_stops=win_stops)
            if df1 is not None and df2 is not None:
                # Both DataFrames share the canonical prefix; copy only the
                # per-stat columns from df2 into df1.
                for col in df2.columns:
                    if col not in CANONICAL_WINDOW_PREFIX:
                        df1[col] = df2[col].values
                return df1

    # Fused CUDA kernel path for more complex stat combinations.

    # Under 'exclude' the per-window fallback serves a request made only of
    # _EXCLUDE_FALLBACK_FIRST names (Garud-only requests excepted); the
    # fused engine serves every other request it can.
    can_fuse = ((missing_data == 'include' or requested.issubset(_FUSED_GARUD_STATS)
                 or not requested <= _EXCLUDE_FALLBACK_FIRST)
                and requested <= _FUSED_STATS
                # The fused engine tiles bp grids and custom regions; snp
                # windows are variant-index ranges that only the per-window
                # fallback builds, so they must skip the fused path.
                and window_type in ('bp', 'regions')
                # A two-population statistic with anything other than exactly
                # two populations must not reach the fused engine, which
                # computes the first pair only and names the column after the
                # bare statistic. The per-window fallback computes every pair
                # and suffixes the names, the same as missing_data='exclude'.
                and (not (requested & _TWO_POP_STATS)
                     or len(populations or []) == 2))

    if can_fuse:
        if haplotype_matrix.device == 'CPU':
            haplotype_matrix.transfer_to_gpu()
        positions = haplotype_matrix.positions
        if hasattr(positions, 'get'):
            positions = positions.get()
        positions = np.asarray(positions)

        if window_type == 'regions':
            # win_starts / win_stops were extracted from the regions above.
            bp_bins = None
        else:
            chrom_start, chrom_end = _bp_grid_bounds(haplotype_matrix, positions)

            # Build window start/stop arrays (supports overlapping windows)
            win_starts, win_stops = _bp_window_grid(
                chrom_start, chrom_end, window_size, step_size)
            bp_bins = np.concatenate([win_starts, win_stops[-1:]])

        if len(win_starts) == 0:
            return pd.DataFrame()

        pop1 = populations[0] if populations and len(populations) >= 1 else None
        pop2 = populations[1] if populations and len(populations) >= 2 else None
        population = pop1  # single-pop stats use the first population

        # Chunk the fused engine when its transposed int8 copy would not fit.
        # A Garud-only request never builds that copy (it batches its own
        # windows), so it runs single-shot whatever the matrix size.
        n_hap = haplotype_matrix.num_haplotypes
        n_var = haplotype_matrix.num_variants
        transpose_bytes = n_var * n_hap  # int8
        free_mem = cp.cuda.Device().mem_info[0]
        use_chunked = (bool(requested.difference(_FUSED_GARUD_STATS))
                       and transpose_bytes * 2 > free_mem * 0.7)

        fused_fn = (windowed_statistics_fused_chunked if use_chunked
                     else windowed_statistics_fused)
        result_dict = fused_fn(
            haplotype_matrix,
            bp_bins=bp_bins,
            statistics=tuple(statistics),
            population=population,
            pop1=pop1,
            pop2=pop2,
            per_base=(span_normalize is not False),
            _win_starts=win_starts,
            _win_stops=win_stops,
            missing_data=missing_data,
            chrom=chrom,
        )
        return pd.DataFrame(result_dict)

    # Fallback: per-window Python loop, which knows some statistics by
    # another name (_FALLBACK_NAMES); its columns get the request's names.
    _check_fallback_covers(requested, window_type)
    analyzer = WindowedAnalyzer(
        window_type=window_type,
        window_size=window_size,
        step_size=step_size,
        statistics=[_FALLBACK_NAMES.get(s, s) if isinstance(s, str) else s
                    for s in statistics],
        populations=populations,
        missing_data=missing_data,
        span_normalize=span_normalize,
        **kwargs
    )
    df = analyzer.compute(haplotype_matrix)
    for ours, theirs in _FALLBACK_NAMES.items():
        if ours in requested:
            df = df.rename(columns=lambda c: ours + c[len(theirs):]
                           if str(c) == theirs or str(c).startswith(theirs + '_')
                           else c)
    # The analyzer names two-population columns "<stat>_<pop1>_<pop2>"
    # because it computes every pair. The fused and scatter engines name
    # them after the statistic alone when exactly two populations are
    # given, and a two-population request must name those columns the
    # same whichever engine serves it. With one pair the bare name is
    # unambiguous, so rename on the way out; direct analyzer use and
    # requests with three or more populations keep the suffixed form.
    if populations is not None and len(populations) == 2:
        p1, p2 = populations
        df = df.rename(columns={
            f"{stat}_{p1}_{p2}": stat
            for stat in statistics
            if isinstance(stat, str) and f"{stat}_{p1}_{p2}" in df.columns
        })
    return df


# Helper functions for built-in statistics



def _compute_mean_r2(matrix: HaplotypeMatrix, max_distance: int,
                     **kwargs) -> float:
    """Compute mean r² for variant pairs within a genomic distance."""
    r2_matrix = matrix.pairwise_r2()
    positions = matrix.positions

    pos_i, pos_j = cp.meshgrid(positions, positions, indexing='ij')
    distances = cp.abs(pos_j - pos_i)
    # Exclude undefined (monomorphic/multiallelic) pairs, which pairwise_r2
    # marks NaN, so a single such site in the window doesn't poison the mean.
    mask = (distances > 0) & (distances <= max_distance) & ~cp.isnan(r2_matrix)
    if cp.any(mask):
        return float(cp.mean(r2_matrix[mask]).get())
    return np.nan


# ---------------------------------------------------------------------------
# Fused CUDA kernel: one block per window, all stats in one pass
# ---------------------------------------------------------------------------

# Haplotype data is transposed before kernel launch so variants are the
# leading dimension (column-major for haplotype access). This ensures
# coalesced memory reads when threads iterate over haplotypes.
_FUSED_MAX_ALLELES = 8
"""Per-variant allele capacity of the fused windowed CUDA kernels.

The kernels count alleles into a fixed cnt[MAX_ALLELES] array per variant, so a
site with an allele index >= this cap cannot be represented and is filtered out
(with a MultiallelicCapWarning). Keep in sync with the ``#define MAX_ALLELES`` in
the kernel sources. Nucleotide data has at most 4 alleles, so the cap is never
hit in practice.
"""


def _warn_fused_allele_cap(n_over):
    """Emit a MultiallelicCapWarning for ``n_over`` sites dropped by the cap."""
    import warnings
    from ._warnings import MultiallelicCapWarning
    warnings.warn(
        f"{n_over} site(s) with an allele index >= {_FUSED_MAX_ALLELES} were "
        "excluded from the fused windowed statistics (kernel per-allele "
        "capacity). Nucleotide data has at most 4 alleles; the non-windowed "
        "diversity/divergence functions handle any number of alleles.",
        MultiallelicCapWarning, stacklevel=3)


# Achaz (2009) neutrality tests: the weight pair of each windowed name.
_ACHAZ_TESTS = {'tajimas_d': ('pi', 'watterson'),
                'normalized_fay_wu_h': ('pi', 'theta_h'),
                'zeng_e': ('theta_l', 'watterson')}
# Zeng's DH is combined from these two tests.
_ZENG_DH_TESTS = ('tajimas_d', 'normalized_fay_wu_h')
_ACHAZ_STATS = frozenset(_ACHAZ_TESTS) | {'zeng_dh'}


def _achaz_test_columns(results, stats, seg_count, n_harm_w, thetas):
    """Fill ``results`` with the requested Achaz neutrality tests.

    Each test divides a per-window theta difference by its standard
    deviation under the neutral model. The variance uses ``n_harm_w``, the
    per-window harmonic mean of ``n_valid`` over sites with two calls, in
    place of the nominal sample size, as the scalar
    ``_compute_neutrality_test`` does. Shared by the scatter and the fused
    engines, so they compute the tests alike.

    Parameters
    ----------
    stats : set of str
        The requested statistics.
    seg_count, n_harm_w : numpy.ndarray, shape (n_windows,)
        Segregating-site count and harmonic-mean sample size (int64).
    thetas : callable
        ``thetas(name)`` gives the per-window raw sum (numpy) of the
        estimator 'pi', 'watterson', 'theta_h' or 'theta_l'; each is summed
        from its sites' own contributions at their own ``n_valid``.
    """
    wanted = {name for name in _ACHAZ_TESTS if name in stats}
    if 'zeng_dh' in stats:
        wanted.update(_ZENG_DH_TESTS)
    if not wanted:
        return
    from .diversity import _harmonic_a1_a2_array

    S = seg_count
    a1, a2 = _harmonic_a1_a2_array(n_harm_w)
    uniq_n, inv_idx = np.unique(n_harm_w, return_inverse=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        theta_est = S / a1
        theta_sq_est = S * (S - 1) / (a1 ** 2 + a2)

    tests = {}
    for name in sorted(wanted):
        w1, w2 = _ACHAZ_TESTS[name]
        alpha, beta = _achaz_coeffs_per_window(uniq_n, inv_idx, w1, w2)
        with np.errstate(invalid='ignore', divide='ignore'):
            var = alpha * theta_est + beta * theta_sq_est
            tests[name] = np.where((var > 0) & (S >= 3) & (n_harm_w >= 3),
                                   (thetas(w1) - thetas(w2)) / np.sqrt(var),
                                   np.nan)
        if name in stats:
            results[name] = tests[name]
    if 'zeng_dh' in stats:
        results['zeng_dh'] = diversity._zeng_dh_combine(
            *(tests[name] for name in _ZENG_DH_TESTS))


# The per-window sums of _fused_windowed_kernel_v2. The order is the order of
# the kernel's output arguments: the engines pass one array per name.
_FUSED_KERNEL_SUMS = ('mpd', 'seg', 'sing', 'count', 'theta_h', 'max_daf',
                      'sum_inv', 'count_valid', 'watterson', 'theta_l')


def _fused_single_pop_columns(results, statistics, sums, per_base,
                              window_bases, excluded_to_empty):
    """Fill ``results`` with the single-population columns of a fused engine.

    ``sums`` maps each name in ``_FUSED_KERNEL_SUMS`` to its per-window
    numpy array. The single-shot engine gets them from one kernel launch and
    the chunked engine adds them up over chunks; from here on the two
    compute alike.
    """
    stats = set(statistics)
    pi, theta_h = sums['mpd'], sums['theta_h']
    watterson, theta_l = sums['watterson'], sums['theta_l']
    results['n_variants'] = sums['count'].astype(int)

    # Per-base rates: rescaled for the sites a population's own missing
    # calls dropped from the sum, then divided by the window's bases. A
    # window that lost every site to 'exclude' is undefined.
    rescale = _missing_data_rescale(sums['count'], per_base, False,
                                    lambda: sums['count_valid'])
    for name, total in (('pi', pi), ('theta_w', watterson),
                        ('theta_h', theta_h), ('theta_l', theta_l)):
        if name in stats:
            vals = total * rescale if rescale is not None else total
            vals = _rate(vals, window_bases, per_base)
            if excluded_to_empty is not None:
                vals = np.where(excluded_to_empty, np.nan, vals)
            results[name] = vals

    if 'segregating_sites' in stats:
        results['segregating_sites'] = sums['seg'].astype(int)
    if 'singletons' in stats:
        results['singletons'] = sums['sing'].astype(int)
    if 'fay_wu_h' in stats:
        # A difference of two estimators, not a rate: no span-normalized
        # form, so it never goes through _rate.
        results['fay_wu_h'] = pi - theta_h

    if stats & _ACHAZ_STATS:
        from .diversity import _harmonic_mean_n
        raw = {'pi': pi, 'watterson': watterson, 'theta_h': theta_h,
               'theta_l': theta_l}
        _achaz_test_columns(
            results, stats, sums['seg'],
            _harmonic_mean_n(sums['count_valid'], sums['sum_inv']), raw.get)

    if 'max_daf' in stats:
        max_daf = sums['max_daf']
        if excluded_to_empty is not None:
            max_daf = np.where(excluded_to_empty, np.nan, max_daf)
        results['max_daf'] = max_daf


_fused_windowed_kernel_v2 = cp.RawKernel(r'''
#define MAX_ALLELES 8   /* keep in sync with _FUSED_MAX_ALLELES (host guard) */
extern "C" __global__
void fused_windowed_stats_v2(const signed char* hap_t,
                             const long long* win_start,
                             const long long* win_stop,
                             int n_hap, int n_total_var, int n_windows,
                             const double* a1_inv_table,
                             double* out_mpd_sum,
                             double* out_seg_count,
                             double* out_sing_count,
                             double* out_var_count,
                             double* out_theta_h_sum,
                             double* out_max_daf,
                             double* out_sum_inv,
                             double* out_count_valid,
                             double* out_watterson_num,
                             double* out_theta_l_sum) {
    int wid = blockIdx.x;
    if (wid >= n_windows) return;

    int v_start = (int)win_start[wid];
    int v_stop = (int)win_stop[wid];
    int n_vars = v_stop - v_start;
    if (n_vars <= 0) {
        if (threadIdx.x == 0) {
            out_mpd_sum[wid] = 0.0;
            out_seg_count[wid] = 0.0;
            out_sing_count[wid] = 0.0;
            out_var_count[wid] = 0.0;
            out_theta_h_sum[wid] = 0.0;
            out_max_daf[wid] = 0.0;
            out_sum_inv[wid] = 0.0;
            out_count_valid[wid] = 0.0;
            out_watterson_num[wid] = 0.0;
            out_theta_l_sum[wid] = 0.0;
        }
        return;
    }

    double t_mpd = 0.0, t_seg = 0.0, t_sing = 0.0;
    double t_count = 0.0, t_theta_h = 0.0, t_max_daf = 0.0;
    double t_sum_inv = 0.0, t_count_valid = 0.0, t_watterson_num = 0.0;
    double t_theta_l = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;
        const signed char* row = hap_t + (long long)v * n_hap;
        /* Per-allele counts over non-missing calls. Sites with an allele index
           >= MAX_ALLELES are filtered out on the host before launch, so every
           allele here fits in cnt[]. */
        int cnt[MAX_ALLELES];
        for (int a = 0; a < MAX_ALLELES; a++) cnt[a] = 0;
        int nv = 0;
        for (int h = 0; h < n_hap; h++) {
            signed char al = row[h];
            if (al >= 0) { nv++; if (al < MAX_ALLELES) cnt[al]++; }
        }
        t_count += 1.0;
        if (nv < 2) continue;
        double dn = (double)nv;

        double same = 0.0;      /* sum_a cnt_a*(cnt_a-1) : same-allele pairs */
        double th = 0.0;        /* theta_h numerator: sum_{a>=1,0<c<nv} 2 c^2 */
        double tl = 0.0;        /* theta_l numerator: sum_{a>=1,0<c<nv} c */
        double mdaf = 0.0;      /* max derived-allele frequency */
        int n_present = 0;
        for (int a = 0; a < MAX_ALLELES; a++) {
            int c = cnt[a];
            if (c > 0) n_present++;
            same += (double)c * (double)(c - 1);
            if (a >= 1 && c > 0) {
                if (c == 1) t_sing += 1.0;            /* per-derived-allele singleton */
                double freq = (double)c / dn;
                if (freq > mdaf) mdaf = freq;
                if (c < nv) {
                    th += 2.0 * (double)c * (double)c;
                    tl += (double)c;
                }
            }
        }
        /* pi = 1 - sum_a C(c,2)/C(n,2); seg = mutation count (alleles present - 1) */
        t_mpd += 1.0 - same / (dn * (dn - 1.0));
        t_seg += (double)(n_present - 1);
        t_theta_h += th / (dn * (dn - 1.0));
        t_theta_l += tl / (dn - 1.0);
        if (mdaf > t_max_daf) t_max_daf = mdaf;
        t_sum_inv += 1.0 / dn;
        t_count_valid += 1.0;
        t_watterson_num += (double)(n_present - 1) * a1_inv_table[nv];
    }

    // Sum reduction for 9 accumulators (plus max_daf)
    __shared__ double smem[10 * 256];
    int tid = threadIdx.x;
    smem[tid]          = t_mpd;
    smem[256 + tid]    = t_seg;
    smem[512 + tid]    = t_sing;
    smem[768 + tid]    = t_count;
    smem[1024 + tid]   = t_theta_h;
    smem[1280 + tid]   = t_max_daf;  // will be max-reduced
    smem[1536 + tid]   = t_sum_inv;
    smem[1792 + tid]   = t_count_valid;
    smem[2048 + tid]   = t_watterson_num;
    smem[2304 + tid]   = t_theta_l;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid]          += smem[tid + s];
            smem[256 + tid]    += smem[256 + tid + s];
            smem[512 + tid]    += smem[512 + tid + s];
            smem[768 + tid]    += smem[768 + tid + s];
            smem[1024 + tid]   += smem[1024 + tid + s];
            // max for max_daf
            if (smem[1280 + tid + s] > smem[1280 + tid])
                smem[1280 + tid] = smem[1280 + tid + s];
            smem[1536 + tid]   += smem[1536 + tid + s];
            smem[1792 + tid]   += smem[1792 + tid + s];
            smem[2048 + tid]   += smem[2048 + tid + s];
            smem[2304 + tid]   += smem[2304 + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_mpd_sum[wid]     = smem[0];
        out_seg_count[wid]   = smem[256];
        out_sing_count[wid]  = smem[512];
        out_var_count[wid]   = smem[768];
        out_theta_h_sum[wid] = smem[1024];
        out_max_daf[wid]     = smem[1280];
        out_sum_inv[wid]      = smem[1536];
        out_count_valid[wid]  = smem[1792];
        out_watterson_num[wid] = smem[2048];
        out_theta_l_sum[wid]   = smem[2304];
    }
}
''', 'fused_windowed_stats_v2')


# Two-population fused kernel for FST and Dxy
_fused_windowed_twopop_kernel = cp.RawKernel(r'''
#define MAX_ALLELES 8   /* keep in sync with _FUSED_MAX_ALLELES (host guard) */

/* Tally one population at one site, pairing consecutive rows into diploid
   individuals. Counting by individual class rather than by gamete keeps the
   per-allele tallies to one update for a homozygote and two for a
   heterozygote, and both consumers derive what they need:

       gamete copies of allele a  = 2*hom[a] + het[a] + half[a]
       copies within individuals  = 2*hom[a] + het[a]

   half[] holds gametes whose partner is missing (and a trailing unpaired
   gamete), which count toward allele frequencies but form no individual.

   Sites carrying an allele index >= MAX_ALLELES are dropped on the host,
   which derives the filter from the whole matrix whenever two-population
   statistics are requested, so it covers pop1 and pop2. The bounds checks
   stay as a guard for direct kernel callers. */
__device__ inline void tally_pop(const signed char* row, int n_hap, int need_wc,
                                 int* hom, int* het, int* half,
                                 int* out_valid, int* out_nind) {
    for (int a = 0; a < MAX_ALLELES; a++) { hom[a] = 0; het[a] = 0; half[a] = 0; }
    int valid = 0, nind = 0;
    if (!need_wc) {
        /* Only gamete counts are wanted. Putting every gamete in half[] keeps
           the 2*hom + het + half sum right while touching one array per
           gamete instead of three. */
        for (int i = 0; i < n_hap; i++) {
            signed char g = row[i];
            if (g >= 0) { valid++; if (g < MAX_ALLELES) half[g]++; }
        }
        *out_valid = valid;
        *out_nind = 0;
        return;
    }
    int i = 0;
    for (; i + 1 < n_hap; i += 2) {
        signed char x = row[i];
        signed char y = row[i + 1];
        if (x >= 0 && y >= 0) {
            valid += 2;
            nind++;
            if (x == y) {
                if (x < MAX_ALLELES) hom[x]++;
            } else {
                if (x < MAX_ALLELES) het[x]++;
                if (y < MAX_ALLELES) het[y]++;
            }
        } else if (x >= 0) {
            valid++; if (x < MAX_ALLELES) half[x]++;
        } else if (y >= 0) {
            valid++; if (y < MAX_ALLELES) half[y]++;
        }
    }
    if (i < n_hap) {   /* odd count: the trailing gamete has no partner */
        signed char x = row[i];
        if (x >= 0) { valid++; if (x < MAX_ALLELES) half[x]++; }
    }
    *out_valid = valid;
    *out_nind = nind;
}

extern "C" __global__
void fused_windowed_twopop(const signed char* hap1_t,
                           const signed char* hap2_t,
                           const long long* win_start,
                           const long long* win_stop,
                           int n_hap1, int n_hap2,
                           int n_total_var, int n_windows,
                           int need_wc,
                           double* out_fst_num,
                           double* out_fst_den,
                           double* out_dxy_sum,
                           double* out_pi1_sum,
                           double* out_pi2_sum,
                           double* out_wc_a_sum,
                           double* out_wc_abc_sum,
                           double* out_dxy_valid,
                           double* out_da_valid) {
    int wid = blockIdx.x;
    if (wid >= n_windows) return;

    int v_start = (int)win_start[wid];
    int v_stop = (int)win_stop[wid];
    int n_vars = v_stop - v_start;
    if (n_vars <= 0) {
        if (threadIdx.x == 0) {
            out_fst_num[wid] = 0.0;  out_fst_den[wid] = 0.0;
            out_dxy_sum[wid] = 0.0;  out_pi1_sum[wid] = 0.0;
            out_pi2_sum[wid] = 0.0;  out_wc_a_sum[wid] = 0.0;
            out_wc_abc_sum[wid] = 0.0;
            out_dxy_valid[wid] = 0.0;  out_da_valid[wid] = 0.0;
        }
        return;
    }

    double t_fst_num = 0.0, t_fst_den = 0.0;
    double t_dxy = 0.0, t_pi1 = 0.0, t_pi2 = 0.0;
    double t_wc_a = 0.0, t_wc_abc = 0.0;
    double t_dxy_valid = 0.0, t_da_valid = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;

        // One pass per population gives three counts for each allele a:
        // hom[a] is individuals with two copies, het[a] is individuals with
        // one copy, and half[a] is lone gametes whose partner is missing.
        // Weir-Cockerham below uses hom and het. The allele frequencies use
        // all three. Without Weir-Cockerham only the frequencies are needed,
        // so the tally skips the pairing and fills half[] alone.
        int hom1[MAX_ALLELES], het1[MAX_ALLELES], half1[MAX_ALLELES];
        int hom2[MAX_ALLELES], het2[MAX_ALLELES], half2[MAX_ALLELES];
        int valid1 = 0, nind1 = 0, valid2 = 0, nind2 = 0;
        tally_pop(hap1_t + (long long)v * n_hap1, n_hap1, need_wc,
                  hom1, het1, half1, &valid1, &nind1);
        tally_pop(hap2_t + (long long)v * n_hap2, n_hap2, need_wc,
                  hom2, het2, half2, &valid2, &nind2);

        if (valid1 == 0 || valid2 == 0) continue;
        t_dxy_valid += 1.0;

        double dn1 = (double)valid1;
        double dn2 = (double)valid2;

        // Weir-Cockerham, mirroring divergence.fst_weir_cockerham (see its
        // NOTE): a per-allele one-vs-rest ANOVA over diploid individuals whose
        // components are between-population (a), between-individual-within-
        // population (b), and between-gamete-within-individual (c), giving
        // FST = sum(a) / sum(a + b + c) over alleles and sites. Absent alleles
        // contribute zero to each, so the loop can run the full width. With
        // r = 2 the scalar's (r-1)*s2/r folds to s2/2 and its nc to n_C.
        double di1 = (double)nind1;
        double di2 = (double)nind2;
        double n_tot = di1 + di2;
        double n_bar = n_tot / 2.0;
        // n_C reduces to 2*di1*di2/n_tot, so it is positive exactly when both
        // populations have a complete individual; n_bar > 1 implies n_tot > 2.
        if (need_wc && n_bar > 1.0) {
            double n_C = n_tot - (di1 * di1 + di2 * di2) / n_tot;
            if (n_C > 0.0) {
                // Divisors that do not vary across alleles, reciprocated once.
                double inv_2di1 = 0.5 / di1;
                double inv_2di2 = 0.5 / di2;
                double inv_ntot = 1.0 / n_tot;
                double inv_nbar = 1.0 / n_bar;
                double inv_nbar_m1 = 1.0 / (n_bar - 1.0);
                double nbar_over_nC = n_bar / n_C;
                double nbar_ratio = n_bar * inv_nbar_m1;
                double hb_coef = (2.0 * n_bar - 1.0) * 0.25 * inv_nbar;
                for (int a = 0; a < MAX_ALLELES; a++) {
                    double p1a = (double)(2 * hom1[a] + het1[a]) * inv_2di1;
                    double p2a = (double)(2 * hom2[a] + het2[a]) * inv_2di2;
                    double p_bar = (di1 * p1a + di2 * p2a) * inv_ntot;
                    double s2 = (di1 * (p1a - p_bar) * (p1a - p_bar) +
                                 di2 * (p2a - p_bar) * (p2a - p_bar)) * inv_nbar;
                    double hb = (double)(het1[a] + het2[a]) * inv_ntot;
                    double base = p_bar * (1.0 - p_bar) - s2 * 0.5;
                    double av = nbar_over_nC *
                        (s2 - inv_nbar_m1 * (base - hb * 0.25));
                    double bv = nbar_ratio * (base - hb_coef * hb);
                    t_wc_a += av;
                    t_wc_abc += av + bv + hb * 0.5;
                }
            }
        }

        // Per-allele within-pop mean pairwise difference (same pairs summed over
        // alleles) and between-pop difference (per-allele cross term) -- mirrors
        // divergence._site_components_from_counts / dxy, multiallelic-correct.
        double same1 = 0.0, same2 = 0.0, between_same = 0.0;
        for (int a = 0; a < MAX_ALLELES; a++) {
            double c1 = (double)(2 * hom1[a] + het1[a] + half1[a]);
            double c2 = (double)(2 * hom2[a] + het2[a] + half2[a]);
            same1 += c1 * (c1 - 1.0);
            same2 += c2 * (c2 - 1.0);
            between_same += c1 * c2;
        }
        double mpd1 = (dn1 > 1.0) ? 1.0 - same1 / (dn1 * (dn1 - 1.0)) : 0.0;
        double mpd2 = (dn2 > 1.0) ? 1.0 - same2 / (dn2 * (dn2 - 1.0)) : 0.0;
        double between = 1.0 - between_same / (dn1 * dn2);
        t_dxy += between;

        // fst/da need Hb and Hw from the same site set (both pops >= 2
        // calls), stricter than dxy alone; see _site_components_from_counts.
        bool joint_pairs = (dn1 > 1.0) && (dn2 > 1.0);
        if (joint_pairs) t_da_valid += 1.0;
        double mpd1_j = joint_pairs ? mpd1 : 0.0;
        double mpd2_j = joint_pairs ? mpd2 : 0.0;
        double between_j = joint_pairs ? between : 0.0;

        t_fst_num += between_j - (mpd1_j + mpd2_j) / 2.0;
        t_fst_den += between_j;
        t_pi1 += mpd1_j;
        t_pi2 += mpd2_j;
    }

    // Block reduction (9 values)
    __shared__ double smem[9 * 256];
    int tid = threadIdx.x;
    smem[tid]          = t_fst_num;
    smem[256 + tid]    = t_fst_den;
    smem[512 + tid]    = t_dxy;
    smem[768 + tid]    = t_pi1;
    smem[1024 + tid]   = t_pi2;
    smem[1280 + tid]   = t_wc_a;
    smem[1536 + tid]   = t_wc_abc;
    smem[1792 + tid]   = t_dxy_valid;
    smem[2048 + tid]   = t_da_valid;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid]          += smem[tid + s];
            smem[256 + tid]    += smem[256 + tid + s];
            smem[512 + tid]    += smem[512 + tid + s];
            smem[768 + tid]    += smem[768 + tid + s];
            smem[1024 + tid]   += smem[1024 + tid + s];
            smem[1280 + tid]   += smem[1280 + tid + s];
            smem[1536 + tid]   += smem[1536 + tid + s];
            smem[1792 + tid]   += smem[1792 + tid + s];
            smem[2048 + tid]   += smem[2048 + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_fst_num[wid]  = smem[0];
        out_fst_den[wid]  = smem[256];
        out_dxy_sum[wid]  = smem[512];
        out_pi1_sum[wid]  = smem[768];
        out_pi2_sum[wid]  = smem[1024];
        out_wc_a_sum[wid] = smem[1280];
        out_wc_abc_sum[wid] = smem[1536];
        out_dxy_valid[wid] = smem[1792];
        out_da_valid[wid]  = smem[2048];
    }
}
''', 'fused_windowed_twopop')


def _window_site_counts(windows, incomplete=None):
    """Sites per window: every site, or only the complete ones.

    The base count is ``windows.counts()`` -- each window's variant-range
    length. ``incomplete`` flags the sites with a missing call, for a request
    still under 'exclude' (Garud-only, see ``_FusedSites``):
    ``garud_h_windows`` skips those sites without a filtered copy, so they
    are subtracted from each range here.
    """
    base = windows.counts()
    if incomplete is None:
        return base
    n_miss = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                             cp.cumsum(incomplete, dtype=cp.int64)])
    return base - (n_miss[windows.stop] - n_miss[windows.start])


class _FusedSites:
    """The sites and rows a fused engine computes on, with no filtered copy.

    Two filters pick the sites. Under 'exclude', only sites with no missing
    call stay, and the engine's mode becomes 'include', because every row it
    reads is then complete. Completeness is judged over the rows the request
    reads: both populations when a two-population statistic is requested, so
    single-population statistics in the same request see only sites complete
    in both; else the single population; else every row. A site's
    completeness does not depend on the window, so one filter equals
    dropping each window's incomplete sites. A Garud-only request stays
    under 'exclude': ``garud_h_windows`` skips incomplete columns itself.
    The second filter drops sites with an allele index the fused kernels
    cannot count (``MultiallelicCapWarning``).

    Neither filter copies the matrix. ``keep`` holds the indices of the
    kept sites (None when every site stays), and each consumer gathers what
    it needs through it: a chunk of columns, a per-site array, the
    transposed kernel input. So an engine under 'exclude' holds no more
    than under 'include', unless a statistic reads ``matrix``.

    Attributes
    ----------
    rows : HaplotypeMatrix
        The rows of the single-population statistics, every site.
    keep : cupy.ndarray of int32, or None
        Indices of the kept sites on the variant axis.
    positions : cupy.ndarray
        Positions of the kept sites; window ranges index this axis.
    missing_data : str
        The mode the engine computes under.
    capped : bool
        Whether the allele cap dropped a site.
    """

    def __init__(self, haplotype_matrix, missing_data, statistics,
                 population=None, pop1=None, pop2=None):
        stats = set(statistics)
        two_pop = bool(_TWO_POP_STATS & stats)
        if two_pop and (pop1 is None or pop2 is None):
            raise ValueError("pop1 and pop2 required for fst/dxy/da")
        if haplotype_matrix.device == 'CPU':
            haplotype_matrix.transfer_to_gpu()
        self.rows = (haplotype_matrix if population is None
                     else population_rows(haplotype_matrix, population))
        if self.rows.device == 'CPU':
            self.rows.transfer_to_gpu()
        self.missing_data = missing_data
        self._matrix = None

        keep_mask = None
        if missing_data == 'exclude' and not stats <= set(_FUSED_GARUD_STATS):
            if two_pop:
                keep_mask = ~haplotype_matrix._site_has_missing([pop1, pop2])
            else:
                # self.rows already holds exactly the rows to judge.
                keep_mask = ~column_has_missing(self.rows.haplotypes)
            self.missing_data = 'include'

        # The two-population kernels read pop1 and pop2 rows of the caller's
        # matrix, which the single-population rows need not cover, so the
        # cap is then judged over every row.
        cap_source = haplotype_matrix if two_pop else self.rows
        overcap = cap_source.haplotypes.max(axis=0) >= _FUSED_MAX_ALLELES
        if keep_mask is not None:
            overcap &= keep_mask
        n_over = int(overcap.sum())
        self.capped = n_over > 0
        if self.capped:
            _warn_fused_allele_cap(n_over)
            keep_mask = ~overcap if keep_mask is None else keep_mask & ~overcap

        positions = self.rows.positions
        if not isinstance(positions, cp.ndarray):
            positions = cp.asarray(positions)
        self.keep = None
        if keep_mask is not None and not bool(keep_mask.all()):
            # int32 is enough: the kernels take the variant count as an int32.
            self.keep = cp.where(keep_mask)[0].astype(cp.int32)
            positions = positions[self.keep]
        self.positions = positions

    def columns(self, start=None, stop=None):
        """Selector on the unfiltered variant axis for kept sites
        ``[start, stop)``, by default all of them."""
        return slice(start, stop) if self.keep is None else self.keep[start:stop]

    def transposed(self, hap, start=None, stop=None, rows=None):
        """Kernel input for ``hap``: the kept sites ``[start, stop)`` (default
        all) of ``rows`` (default all) as a contiguous (n_sites, n_rows) int8
        array. This is the one copy the kernels need, taken straight from
        the unfiltered matrix."""
        if rows is None:
            return cp.ascontiguousarray(hap.T[self.columns(start, stop)],
                                        dtype=cp.int8)
        # Selecting sites and rows takes two gathers, and the first one
        # copies every row of its sites. So fill the block a few sites at a
        # time, sized to keep that copy at a quarter of the block.
        n_sites = stop - start
        out = cp.empty((n_sites, len(rows)), dtype=cp.int8)
        step = max(1, n_sites * len(rows) // (4 * hap.shape[0]))
        for lo in range(0, n_sites, step):
            hi = min(lo + step, n_sites)
            out[lo:hi] = hap.T[self.columns(start + lo, start + hi)][:, rows]
        return out

    def per_site(self, func):
        """``func(matrix)`` for the kept sites, where ``func`` gives one value
        (or a tuple of arrays with one row) per site and a site's value does
        not depend on the other sites. It runs on the unfiltered rows and
        the result is gathered, so the matrix is not copied. After an allele
        cap it runs on the filtered matrix, because ``func`` may size its
        work by the largest allele index."""
        if self.capped or self.keep is None:
            return func(self.matrix)
        out = func(self.rows)
        if isinstance(out, tuple):
            return tuple(v[self.keep] for v in out)
        return out[self.keep]

    def incomplete(self):
        """Per kept site, whether a row of ``rows`` has a missing call; None
        unless the engine is still under 'exclude' (a Garud-only request),
        the one case where incomplete sites stay in the window ranges."""
        if self.missing_data != 'exclude':
            return None
        flags = column_has_missing(self.rows.haplotypes)
        return flags if self.keep is None else flags[self.keep]

    @property
    def matrix(self):
        """``rows`` restricted to the kept sites. The first use copies the
        kept sites, so only statistics that need them as one matrix (nSL,
        Garud's H next to other statistics, the per-window LD and distance
        blocks) read it."""
        if self._matrix is None:
            self._matrix = (self.rows if self.keep is None
                            else self.rows._keep_sites(self.keep))
        return self._matrix


@_eager_only
@_phase_dependent
def windowed_statistics_fused(haplotype_matrix: HaplotypeMatrix,
                              bp_bins=None,
                              statistics=('pi', 'theta_w', 'tajimas_d',
                                          'segregating_sites', 'singletons'),
                              population=None,
                              pop1=None,
                              pop2=None,
                              per_base: bool = True,
                              is_accessible=None,
                              _win_starts=None,
                              _win_stops=None,
                              missing_data='include',
                              chrom=None,
                              _sites=None):
    """GPU-native windowed statistics using fused CUDA kernels.

    One kernel launch processes ALL windows in parallel. Each thread block
    handles one window, with threads cooperatively reducing over variants.
    Reads the haplotype matrix once and computes all statistics simultaneously.

    Depends on haplotype phase when a phase-dependent statistic is requested;
    see :doc:`phasing`.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    bp_bins : array_like
        Window edges in base pairs. N+1 edges define N windows.
    statistics : tuple of str
        Statistics to compute. Single-pop: 'pi', 'theta_w', 'tajimas_d',
        'segregating_sites', 'singletons', 'het_expected'. Two-pop: 'fst',
        'dxy'. Haplotype: the Garud statistics and 'haplotype_count'.
        'het_expected' is the per-window mean of per-site expected
        heterozygosity over the sites with a defined value.
    population : str or list, optional
        Population for single-pop statistics.
    pop1, pop2 : str or list, optional
        Populations for two-pop statistics.
    per_base : bool
        Normalize by window size in base pairs.
    is_accessible : array_like of bool or AccessibleMask, optional
        Accessibility for per-base normalization, in place of the matrix's
        mask; a bool array has element 0 at position 1, as in scikit-allel.
    missing_data : {'include', 'exclude'}
        'include' uses every site with the calls it has. 'exclude' keeps
        only sites with no missing call in the rows the request reads (see
        ``_FusedSites``), so ``n_variants`` counts those sites.
    chrom : str or int, optional
        Chromosome label emitted in the ``chrom`` output column.

    Returns
    -------
    dict
        Maps column names to numpy arrays of shape (n_windows,). The first
        six columns are always ``chrom, start, end, center, n_variants,
        window_id``, followed by one column per requested statistic. A
        haplotype statistic adds ``n_complete_haplotypes``, the number of
        haplotypes with no missing call in each window. Unlike
        ``windowed_analysis``, this function neither warns about windows
        with few complete haplotypes nor applies a cutoff; check that
        column yourself.
    """
    phase_ctx = _phase_scan_context("windowed_statistics_fused", statistics)
    if phase_ctx:
        _warn_unphased(haplotype_matrix, context=phase_ctx)

    # Pre-filter positions: a window exclusion emptied (NaN for max_daf)
    # looks identical to one with no variant to begin with (0.0) once
    # filtering below removes the distinction.
    orig_missing_data = missing_data
    orig_positions_gpu = haplotype_matrix.positions
    if not isinstance(orig_positions_gpu, cp.ndarray):
        orig_positions_gpu = cp.asarray(orig_positions_gpu)

    # The sites to compute on: under 'exclude' the complete ones, less any
    # the kernels' allele capacity cannot count. Window ranges index these
    # kept sites. Only the statistics that read sites.matrix copy them.
    sites = _sites if _sites is not None else _FusedSites(
        haplotype_matrix, missing_data, statistics, population, pop1, pop2)
    missing_data = sites.missing_data
    n_hap = sites.rows.haplotypes.shape[0]
    positions = sites.positions
    two_pop_requested = bool(_TWO_POP_STATS & set(statistics))
    n_total_var = int(positions.shape[0])

    # Window bounds in bp: either explicit start/stop arrays (overlapping bp
    # grids and custom regions) or contiguous bp bin edges.
    if _win_starts is not None and _win_stops is not None:
        ws_gpu = cp.asarray(_win_starts, dtype=cp.float64)
        we_gpu = cp.asarray(_win_stops, dtype=cp.float64)
    elif bp_bins is not None:
        bp_bins_gpu = cp.asarray(bp_bins, dtype=cp.float64)
        ws_gpu = bp_bins_gpu[:-1]
        we_gpu = bp_bins_gpu[1:]
    else:
        raise ValueError('bp_bins and _win_starts/_win_stops must not both be None.')

    # One membership rule for every window family: each window's contiguous
    # variant range, and the per-window reductions over it. win_start/win_stop
    # feed the fused kernels; the per-site stats use the accumulator's methods.
    windows = _WindowAccumulator(positions, ws_gpu, we_gpu)
    win_start, win_stop = windows.start, windows.stop
    n_windows = windows.n_windows

    # A window that lost every site to 'exclude' is undefined (NaN) for the
    # statistics below; one with no variant to begin with stays 0.0.
    excluded_to_empty = None
    if orig_missing_data == 'exclude' and (
            set(statistics) & (_RATE_STATS | {'dxy', 'da', 'max_daf'})):
        excluded_to_empty = _windows_emptied_by_exclude(
            orig_positions_gpu, windows, ws_gpu, we_gpu)
    ws_cpu = ws_gpu.get() if hasattr(ws_gpu, 'get') else ws_gpu
    we_cpu = we_gpu.get() if hasattr(we_gpu, 'get') else we_gpu
    # Raw variant-index span is the fallback n_variants; single-pop kernels
    # overwrite below with a missing-data-aware count.
    n_variants_raw = _window_site_counts(windows, sites.incomplete()).get()
    results = _init_window_results(
        chrom, ws_cpu, we_cpu, n_variants=n_variants_raw)

    window_bases = None
    if per_base:
        window_bases = _compute_window_bases(
            haplotype_matrix, results['start'],
            results['end'], is_accessible)

    # single-pop stats via fused kernel
    single_pop_requested = not _FUSED_SINGLE_STATS.isdisjoint(statistics)
    if single_pop_requested:
        from .diversity import _get_a1_inv

        # Only this kernel needs the transposed (n_total_var, n_hap) int8
        # copy for coalesced access; Garud-only and scatter-only requests
        # skip the extra matrix-sized allocation.
        hap = sites.transposed(sites.rows.haplotypes)
        outs = [cp.zeros(n_windows, dtype=cp.float64)
                for _ in _FUSED_KERNEL_SUMS]
        a1_inv_table = _get_a1_inv(n_hap)

        block = 256
        grid = n_windows

        _fused_windowed_kernel_v2(
            (grid,), (block,),
            (hap, win_start, win_stop,
             np.int32(n_hap), np.int32(n_total_var), np.int32(n_windows),
             a1_inv_table, *outs))

        sums = dict(zip(_FUSED_KERNEL_SUMS, cp.stack(outs).get()))
        # The later stages read the un-transposed matrix; release the copy
        # so their batches can use the memory.
        del hap

        _fused_single_pop_columns(results, statistics, sums, per_base,
                                  window_bases, excluded_to_empty)

    # two-pop stats via fused kernel
    if two_pop_requested:
        if 'fst_wc' in statistics:
            # The kernel pairs consecutive subset rows into individuals for
            # Weir-Cockerham; the gamete statistics take any row list.
            from ._warnings import check_paired_rows_for_populations
            check_paired_rows_for_populations(
                haplotype_matrix.sample_sets, pop1, pop2, "windowed fst_wc")

        # Use the original (unsubsetted) matrix for population lookup
        m1 = population_rows(haplotype_matrix, pop1)
        m2 = population_rows(haplotype_matrix, pop2)
        if m1.device == 'CPU':
            m1.transfer_to_gpu()
        if m2.device == 'CPU':
            m2.transfer_to_gpu()

        n1 = m1.haplotypes.shape[0]
        n2 = m2.haplotypes.shape[0]
        # The kept sites only, so the variant axis matches
        # positions/win_start/win_stop/n_total_var.
        hap1 = sites.transposed(m1.haplotypes)
        hap2 = sites.transposed(m2.haplotypes)

        out_fst_num = cp.zeros(n_windows, dtype=cp.float64)
        out_fst_den = cp.zeros(n_windows, dtype=cp.float64)
        out_dxy = cp.zeros(n_windows, dtype=cp.float64)
        out_pi1 = cp.zeros(n_windows, dtype=cp.float64)
        out_pi2 = cp.zeros(n_windows, dtype=cp.float64)
        out_wc_a = cp.zeros(n_windows, dtype=cp.float64)
        out_wc_abc = cp.zeros(n_windows, dtype=cp.float64)
        out_dxy_valid = cp.zeros(n_windows, dtype=cp.float64)
        out_da_valid = cp.zeros(n_windows, dtype=cp.float64)
        need_wc = 'fst_wc' in statistics

        block = 256
        grid = n_windows

        _fused_windowed_twopop_kernel(
            (grid,), (block,),
            (hap1, hap2, win_start, win_stop,
             np.int32(n1), np.int32(n2),
             np.int32(n_total_var), np.int32(n_windows),
             np.int32(need_wc),
             out_fst_num, out_fst_den, out_dxy, out_pi1, out_pi2,
             out_wc_a, out_wc_abc, out_dxy_valid, out_da_valid))

        # Post-process on GPU, single .get() per result
        if 'fst' in statistics or 'fst_hudson' in statistics:
            hudson_fst = cp.where(out_fst_den > 0,
                                  out_fst_num / out_fst_den, cp.nan).get()
            if 'fst' in statistics:
                results['fst'] = hudson_fst
            if 'fst_hudson' in statistics:
                results['fst_hudson'] = hudson_fst

        if 'fst_wc' in statistics:
            results['fst_wc'] = cp.where(out_wc_abc > 0,
                                          out_wc_a / out_wc_abc, cp.nan).get()

        # Without this, a site with too few calls to contribute would still
        # count toward the span, reading as invariant rather than unmeasured.
        # dxy and da need their own count, since they use different site gates.
        if 'dxy' in statistics:
            r = _missing_data_rescale(n_variants_raw, per_base, False,
                                      lambda: out_dxy_valid.get())
            vals = out_dxy * cp.asarray(r) if r is not None else out_dxy
            if per_base:
                wb = cp.asarray(window_bases)
                results['dxy'] = cp.where(wb > 0,
                                          vals / wb, cp.nan).get()
            else:
                results['dxy'] = vals.get()

        if 'da' in statistics:
            # da's between term is fst's (both pops >= 2 calls), not dxy's.
            da_sum = out_fst_den - (out_pi1 + out_pi2) / 2.0
            r = _missing_data_rescale(n_variants_raw, per_base, False,
                                      lambda: out_da_valid.get())
            if r is not None:
                da_sum = da_sum * cp.asarray(r)
            if per_base:
                wb = cp.asarray(window_bases)
                results['da'] = cp.where(wb > 0,
                                         da_sum / wb, cp.nan).get()
            else:
                results['da'] = da_sum.get()

        if excluded_to_empty is not None:
            for name in ('dxy', 'da'):
                if name in results:
                    results[name] = np.where(excluded_to_empty, np.nan, results[name])

    # Garud's H: exact per-window haplotype hashing, batched to GPU memory
    if any(s in statistics for s in _FUSED_GARUD_STATS):
        _compute_fused_garud_h(sites.matrix, win_start, win_stop, statistics,
                               results, missing_data=missing_data)

    # Per-site stats binned into windows. A variant belongs to window w iff
    # ws[w] <= pos < we[w] -- the same right-open variant range win_start /
    # win_stop and n_variants use -- so each is a prefix-sum range query over
    # the window's range (``windows``), correct under overlap and for unordered
    # or nested regions alike without any per-(variant, window) membership.

    # Shared per-allele counts (used by daf_hist and mu_sfs). Per-derived-allele,
    # per-site n_valid -- the same convention as diversity.daf_histogram / mu_sfs,
    # so the windowed values equal the scalar over each window (include mode).
    if any(s in statistics for s in ('daf_hist', 'mu_sfs')):
        from ._memutil import allele_counts
        ac_daf, nv_daf = sites.per_site(                    # (n_var, K), (n_var,)
            lambda m: allele_counts(m.haplotypes))
        derived = ac_daf[:, 1:]                             # (n_var, K-1)

    if 'mean_nsl' in statistics:
        from . import selection as sel
        # sites.matrix is already population-subsetted; don't re-subset. nSL
        # extends haplotypes across sites, so it needs the kept sites as one
        # matrix (the filtered copy). It is scanned once over this whole
        # (eager) matrix, so each per-site score sees the full region before being binned into windows. Over a
        # StreamingHaplotypeMatrix this runs per chunk, truncating the scan at
        # chunk boundaries (see _stream_windowed_analysis).
        # nsl() returns (n_var,) for biallelic data and (n_var, n_derived)
        # when multiallelic focal sites are present; average every finite
        # (site, allele) score in every window the site belongs to -- the
        # 2-D per-site mean the accumulator reduces over its finite entries.
        nsl_2d = cp.asarray(sel.nsl(sites.matrix)).reshape(n_total_var, -1)
        results['mean_nsl'] = windows.mean(nsl_2d).get()

    if 'het_expected' in statistics:
        # Per-window mean of per-site expected heterozygosity over the sites
        # with a defined value, NaN for a window with none -- the per-site-range
        # mean (``windows.mean``) every other per-site stat here now shares,
        # serving any window family including unordered or overlapping regions.
        # The rows are already population-subsetted; missing_data is 'include'
        # here (see _FusedSites). Reuses the scalar's per-site rule via the
        # CuPy helper, with no host round trip.
        he = sites.per_site(
            lambda m: diversity._heterozygosity_expected_sites(
                m, missing_data=missing_data))                # (n_var,) CuPy
        results['het_expected'] = windows.mean(he, valid=~cp.isnan(he)).get()

    # SNP distance stats per window
    if not _SNP_DIST_STATS.isdisjoint(statistics):
        ws_np, we_np = win_start.get(), win_stop.get()
        pos_cpu = positions.get() if hasattr(positions, 'get') else np.asarray(positions)
        sd_mean = np.full(n_windows, np.nan)
        sd_var = np.full(n_windows, np.nan)
        sd_min = np.full(n_windows, np.nan)
        sd_max = np.full(n_windows, np.nan)
        mu_var_arr = np.full(n_windows, np.nan)

        for wi in range(n_windows):
            s, e = int(ws_np[wi]), int(we_np[wi])
            n_var = e - s
            if n_var < 2:
                # No gap to report, but mu_var is still defined.
                if per_base:
                    mu_var_arr[wi] = (n_var / window_bases[wi]
                                      if window_bases[wi] > 0 else np.nan)
                else:
                    mu_var_arr[wi] = float(n_var)
                continue
            win_pos = pos_cpu[s:e]
            gaps = np.diff(win_pos).astype(np.float64)
            sd_mean[wi] = np.mean(gaps)
            sd_var[wi] = np.var(gaps)
            sd_min[wi] = np.min(gaps)
            sd_max[wi] = np.max(gaps)
            if per_base:
                mu_var_arr[wi] = (len(win_pos) / window_bases[wi]
                                  if window_bases[wi] > 0 else np.nan)
            else:
                mu_var_arr[wi] = float(len(win_pos))

        for stat, arr in [('snp_dist_mean', sd_mean), ('snp_dist_var', sd_var),
                          ('snp_dist_min', sd_min), ('snp_dist_max', sd_max),
                          ('mu_var', mu_var_arr)]:
            if stat in statistics:
                results[stat] = arr

    # DAF histogram per window. One entry per present derived allele
    # (freq = count / n_valid), binned per site into (site, daf_bin) and then
    # summed over each window's variant range, normalized per window --
    # matching diversity.daf_histogram.
    if 'daf_hist' in statistics:
        nv = cp.maximum(nv_daf.astype(cp.float64), 1.0)[:, None]
        freq = derived.astype(cp.float64) / nv                   # per (site, allele)
        dbin = diversity._daf_bin_index(freq, N_DAF_BINS)
        # One entry per segregating derived allele (present, not fixed),
        # tallied into a per-site histogram; the window histogram is the
        # range sum.
        keep = (derived > 0) & (derived < nv_daf[:, None])       # (n_var, K-1)
        site_hist = cp.zeros((n_total_var, N_DAF_BINS), dtype=cp.float64)
        kv, ka = cp.where(keep)
        cp.add.at(site_hist, (kv, dbin[kv, ka]), 1.0)
        hist_matrix = windows.sum2d(site_hist).get()            # (n_windows, N)
        row_sum = hist_matrix.sum(axis=1, keepdims=True)
        hist_matrix = np.divide(hist_matrix, row_sum,
                                out=np.zeros_like(hist_matrix),
                                where=row_sum > 0)
        for b in range(N_DAF_BINS):
            results[f'daf_bin_{b}'] = hist_matrix[:, b]

    # muSFS: fraction of segregating (site, allele) entries at the SFS edges
    # (singleton or n_valid-1), per-site n_valid -- matching diversity.mu_sfs.
    if 'mu_sfs' in statistics:
        nvc = nv_daf[:, None]
        is_seg = (derived > 0) & (derived < nvc)                 # (n_var, K-1)
        is_edge = is_seg & ((derived == 1) | (derived == nvc - 1))
        # The window index depends only on the site, so sum the per-allele
        # counts per site first, then range-sum over each window.
        seg_site = is_seg.sum(axis=1).astype(cp.float64)         # (n_var,)
        edge_site = is_edge.sum(axis=1).astype(cp.float64)
        seg_sum = windows.sum(seg_site)
        edge_sum = windows.sum(edge_site)
        # 0/0 is NaN for a window with no segregating entry
        results['mu_sfs'] = (edge_sum / seg_sum).get()

    # Per-window pairwise stats (LD, distance moments)
    perwin_stats = (_PAIRWISE_LD_STATS | _DIST_MOMENT_STATS) & set(statistics)

    if perwin_stats:
        from . import ld_statistics
        from . import distance_stats
        ws_np, we_np = win_start.get(), win_stop.get()
        # These blocks slice the kept sites by window range, and ZnS builds
        # arrays the size of the matrix, so they read the filtered copy.
        matrix = sites.matrix

        stat_arrays = {s: np.full(n_windows, np.nan) for s in perwin_stats}
        need_dist = bool(perwin_stats & _DIST_MOMENT_STATS)
        need_winmat = bool(perwin_stats - {'zns'})

        # zns/omega restrict to biallelic; the excluded multiallelic sites are
        # the same across the scan, so warn once here rather than once per
        # window (per-window omega is fed already-biallelic input below, so its
        # own restriction drops nothing and stays silent).
        from ._warnings import _warn_biallelic_only
        if {'zns', 'omega'} & stat_arrays.keys():
            bmask = matrix._biallelic_mask()
            _warn_biallelic_only(int((~bmask).sum()),
                                 context="windowed_analysis")

        # Precompute for fused ZnS path
        if 'zns' in stat_arrays:
            # Count on the 0/1 biallelic indicator so {0,2}/{1,2} codings are
            # handled.
            ind = matrix._biallelic_indicator()
            hap_clean = cp.where(ind >= 0, ind, 0).astype(cp.float64)
            valid_mask = (ind >= 0).astype(cp.float64)
            # Zeroing hap_clean (not valid_mask) forces dac==0 at multiallelic
            # sites, so the per-window segregating filter excludes them from ZnS.
            hap_clean[:, ~bmask] = 0.0

        for wi in range(n_windows):
            s, e = int(ws_np[wi]), int(we_np[wi])
            # The loop skips only empty windows. Each statistic applies its
            # own too-few-sites rule, so a small window gets the value the
            # scalar gives.
            if e == s:
                continue

            if 'zns' in stat_arrays:
                # Per-window zns defaults to the unbiased sigma_d2
                # estimator (Ragsdale & Gravel 2019), matching the new
                # default of ld_statistics.zns(estimator='auto') on
                # HaplotypeMatrix inputs.
                stat_arrays['zns'][wi] = ld_statistics._zns_from_precomputed(
                    hap_clean, valid_mask, s, e,
                    use_projection=True)

            if need_winmat:
                win_mat = HaplotypeMatrix(matrix.haplotypes[:, s:e],
                                           matrix.positions[s:e],
                                           phased=matrix.phased)
                if 'omega' in stat_arrays:
                    # Feed already-biallelic input so omega's own restriction
                    # drops nothing and stays silent; warned once for the scan
                    # above (_warn_biallelic_only is a no-op at 0).
                    stat_arrays['omega'][wi] = ld_statistics.omega(
                        win_mat.restrict_to_biallelic(),
                        missing_data=missing_data)
                if 'mu_ld' in stat_arrays:
                    stat_arrays['mu_ld'][wi] = ld_statistics.mu_ld(win_mat)
                if need_dist:
                    v, sk, ku = distance_stats.dist_moments(win_mat)
                    if 'dist_var' in stat_arrays:
                        stat_arrays['dist_var'][wi] = v
                    if 'dist_skew' in stat_arrays:
                        stat_arrays['dist_skew'][wi] = sk
                    if 'dist_kurt' in stat_arrays:
                        stat_arrays['dist_kurt'][wi] = ku

        results.update(stat_arrays)

    return results


@_eager_only
@_phase_dependent
def windowed_statistics_fused_chunked(haplotype_matrix: HaplotypeMatrix,
                                      bp_bins=None,
                                      statistics=('pi', 'theta_w', 'tajimas_d'),
                                      population=None,
                                      pop1=None,
                                      pop2=None,
                                      per_base: bool = True,
                                      is_accessible=None,
                                      _win_starts=None,
                                      _win_stops=None,
                                      missing_data='include',
                                      chrom=None):
    """Chunked fused windowed statistics for data too large for a single pass.

    Depends on haplotype phase when a phase-dependent statistic is requested;
    see :doc:`phasing`.

    Same interface and results as windowed_statistics_fused(), but splits the
    variant axis into memory-safe chunks. Each chunk is transposed and fed to
    the existing fused CUDA kernels. Partial results are accumulated across
    chunks (all kernel outputs are additive sums, except max_daf which uses
    element-wise max).
    """
    from ._memutil import estimate_fused_chunk_size, free_gpu_pool

    phase_ctx = _phase_scan_context("windowed_statistics_fused_chunked", statistics)
    if phase_ctx:
        _warn_unphased(haplotype_matrix, context=phase_ctx)

    # Pre-filter positions: a window exclusion emptied (NaN for max_daf)
    # looks identical to one with no variant to begin with (0.0) once
    # filtering below removes the distinction.
    orig_missing_data = missing_data
    orig_positions_gpu = haplotype_matrix.positions
    if not isinstance(orig_positions_gpu, cp.ndarray):
        orig_positions_gpu = cp.asarray(orig_positions_gpu)

    # The sites to compute on (see _FusedSites). Each chunk gathers its own
    # columns of the kept sites, so 'exclude' costs one chunk, not a copy of
    # the matrix.
    sites = _FusedSites(
        haplotype_matrix, missing_data, statistics, population, pop1, pop2)
    missing_data = sites.missing_data
    hap_rows = sites.rows.haplotypes
    n_hap = hap_rows.shape[0]
    positions = sites.positions
    two_pop_requested = bool(_TWO_POP_STATS & set(statistics))
    n_total_var = int(positions.shape[0])

    # Window ranges (same membership rule as non-chunked)
    if _win_starts is not None and _win_stops is not None:
        ws_gpu = cp.asarray(_win_starts, dtype=cp.float64)
        we_gpu = cp.asarray(_win_stops, dtype=cp.float64)
    elif bp_bins is not None:
        bp_bins_gpu = cp.asarray(bp_bins, dtype=cp.float64)
        ws_gpu = bp_bins_gpu[:-1]
        we_gpu = bp_bins_gpu[1:]
    else:
        raise ValueError('bp_bins and _win_starts/_win_stops must not both be None.')

    windows = _WindowAccumulator(positions, ws_gpu, we_gpu)
    win_start, win_stop = windows.start, windows.stop
    n_windows = windows.n_windows

    # A window that lost every site to 'exclude' is undefined (NaN) for the
    # statistics below; one with no variant to begin with stays 0.0.
    excluded_to_empty = None
    if orig_missing_data == 'exclude' and (
            set(statistics) & (_RATE_STATS | {'dxy', 'da', 'max_daf'})):
        excluded_to_empty = _windows_emptied_by_exclude(
            orig_positions_gpu, windows, ws_gpu, we_gpu)

    ws_cpu = ws_gpu.get() if hasattr(ws_gpu, 'get') else ws_gpu
    we_cpu = we_gpu.get() if hasattr(we_gpu, 'get') else we_gpu
    # Raw variant-index span is the fallback n_variants; single-pop kernels
    # overwrite below with a missing-data-aware count.
    n_variants_raw = _window_site_counts(windows, sites.incomplete()).get()
    results = _init_window_results(
        chrom, ws_cpu, we_cpu, n_variants=n_variants_raw)

    window_bases = None
    if per_base:
        window_bases = _compute_window_bases(
            haplotype_matrix, results['start'],
            results['end'], is_accessible)

    # Determine chunk size
    chunk_size = estimate_fused_chunk_size(n_hap)
    # Ensure chunk is at least large enough for the largest window
    max_win_variants = int((win_stop - win_start).max().get()) if n_windows > 0 else 0
    chunk_size = max(chunk_size, max_win_variants + 1)

    # ── Single-pop stats via chunked fused kernel ────────────────────────
    single_pop_requested = not _FUSED_SINGLE_STATS.isdisjoint(statistics)
    if single_pop_requested:
        from .diversity import _get_a1_inv

        # Accumulators, one per kernel sum
        acc = [cp.zeros(n_windows, dtype=cp.float64)
               for _ in _FUSED_KERNEL_SUMS]
        a1_inv_table = _get_a1_inv(n_hap)

        for c_start in range(0, n_total_var, chunk_size):
            c_end = min(c_start + chunk_size, n_total_var)

            # Find windows overlapping this chunk
            overlap = (win_start < c_end) & (win_stop > c_start)
            w_idx = cp.where(overlap)[0]
            if len(w_idx) == 0:
                continue

            # Clip window ranges to chunk boundaries
            clipped_start = cp.maximum(win_start[w_idx], c_start) - c_start
            clipped_stop = cp.minimum(win_stop[w_idx], c_end) - c_start
            n_overlap = len(w_idx)

            # Transpose chunk
            hap_chunk_t = sites.transposed(hap_rows, c_start, c_end)
            n_chunk_var = c_end - c_start

            # Per-chunk outputs
            outs = [cp.zeros(n_overlap, dtype=cp.float64)
                    for _ in _FUSED_KERNEL_SUMS]

            _fused_windowed_kernel_v2(
                (int(n_overlap),), (256,),
                (hap_chunk_t, clipped_start, clipped_stop,
                 np.int32(n_hap), np.int32(n_chunk_var), np.int32(n_overlap),
                 a1_inv_table, *outs))

            # Accumulate (all additive except max_daf)
            for name, total, out in zip(_FUSED_KERNEL_SUMS, acc, outs):
                if name == 'max_daf':
                    total[w_idx] = cp.maximum(total[w_idx], out)
                else:
                    cp.add.at(total, w_idx, out)

            del hap_chunk_t
            free_gpu_pool()

        sums = dict(zip(_FUSED_KERNEL_SUMS, cp.stack(acc).get()))
        _fused_single_pop_columns(results, statistics, sums, per_base,
                                  window_bases, excluded_to_empty)

    # ── Two-pop stats via chunked fused kernel ───────────────────────────
    if two_pop_requested:
        # The population row numbers index the original matrix, but hap_rows
        # can be the single-population subset (the dispatcher passes
        # population=pop1 alongside pop1/pop2). CuPy fancy indexing does not
        # bounds-check, so applying these rows to the subset would silently
        # read the wrong haplotypes. Read the rows from the original matrix,
        # and select the kept sites' columns.
        need_wc = 'fst_wc' in statistics

        # A population is a sample_sets name or a row list, the same as the
        # single-shot engine accepts via population_rows.
        def pop_rows_of(pop):
            if isinstance(pop, str):
                rows = haplotype_matrix.sample_sets.get(pop)
                if rows is None:
                    # Match population_rows's error for unknown names,
                    # so the chunked and single-shot engines agree.
                    raise ValueError(
                        f"Population {pop} not found in sample_sets")
            else:
                rows = list(pop)
            if not isinstance(pop, str):
                # Direct row lists never pass the sample_sets setter; apply
                # the same range and duplicate rules before the gather.
                from ._warnings import check_sample_set_rows
                check_sample_set_rows("population row list", rows,
                                      haplotype_matrix.haplotypes.shape[0])
            if need_wc:
                # The kernel pairs consecutive rows into individuals for
                # Weir-Cockerham; the gamete statistics take any row list.
                from ._warnings import check_paired_rows
                label = pop if isinstance(pop, str) else "row list"
                check_paired_rows(rows, f"windowed fst_wc({label})")
            return cp.asarray(rows, dtype=cp.int64)

        orig_hap = haplotype_matrix.haplotypes
        pop1_rows = pop_rows_of(pop1)
        pop2_rows = pop_rows_of(pop2)
        n1 = len(pop1_rows)
        n2 = len(pop2_rows)
        # Chunk size based on the larger population
        twopop_chunk = estimate_fused_chunk_size(max(n1, n2))
        twopop_chunk = max(twopop_chunk, max_win_variants + 1)

        acc_fst_num = cp.zeros(n_windows, dtype=cp.float64)
        acc_fst_den = cp.zeros(n_windows, dtype=cp.float64)
        acc_dxy = cp.zeros(n_windows, dtype=cp.float64)
        acc_pi1 = cp.zeros(n_windows, dtype=cp.float64)
        acc_pi2 = cp.zeros(n_windows, dtype=cp.float64)
        acc_wc_a = cp.zeros(n_windows, dtype=cp.float64)
        acc_wc_abc = cp.zeros(n_windows, dtype=cp.float64)
        acc_dxy_valid = cp.zeros(n_windows, dtype=cp.float64)
        acc_da_valid = cp.zeros(n_windows, dtype=cp.float64)

        for c_start in range(0, n_total_var, twopop_chunk):
            c_end = min(c_start + twopop_chunk, n_total_var)

            overlap = (win_start < c_end) & (win_stop > c_start)
            w_idx = cp.where(overlap)[0]
            if len(w_idx) == 0:
                continue

            clipped_start = cp.maximum(win_start[w_idx], c_start) - c_start
            clipped_stop = cp.minimum(win_stop[w_idx], c_end) - c_start
            n_overlap = len(w_idx)

            hap1_t = sites.transposed(orig_hap, c_start, c_end, pop1_rows)
            hap2_t = sites.transposed(orig_hap, c_start, c_end, pop2_rows)
            n_chunk_var = c_end - c_start

            out_fst_num = cp.zeros(n_overlap, dtype=cp.float64)
            out_fst_den = cp.zeros(n_overlap, dtype=cp.float64)
            out_dxy = cp.zeros(n_overlap, dtype=cp.float64)
            out_pi1 = cp.zeros(n_overlap, dtype=cp.float64)
            out_pi2 = cp.zeros(n_overlap, dtype=cp.float64)
            out_wc_a = cp.zeros(n_overlap, dtype=cp.float64)
            out_wc_abc = cp.zeros(n_overlap, dtype=cp.float64)
            out_dxy_valid = cp.zeros(n_overlap, dtype=cp.float64)
            out_da_valid = cp.zeros(n_overlap, dtype=cp.float64)

            _fused_windowed_twopop_kernel(
                (int(n_overlap),), (256,),
                (hap1_t, hap2_t, clipped_start, clipped_stop,
                 np.int32(n1), np.int32(n2),
                 np.int32(n_chunk_var), np.int32(n_overlap),
                 np.int32(need_wc),
                 out_fst_num, out_fst_den, out_dxy, out_pi1, out_pi2,
                 out_wc_a, out_wc_abc, out_dxy_valid, out_da_valid))

            cp.add.at(acc_fst_num, w_idx, out_fst_num)
            cp.add.at(acc_fst_den, w_idx, out_fst_den)
            cp.add.at(acc_dxy, w_idx, out_dxy)
            cp.add.at(acc_pi1, w_idx, out_pi1)
            cp.add.at(acc_pi2, w_idx, out_pi2)
            cp.add.at(acc_dxy_valid, w_idx, out_dxy_valid)
            cp.add.at(acc_da_valid, w_idx, out_da_valid)
            if need_wc:
                cp.add.at(acc_wc_a, w_idx, out_wc_a)
                cp.add.at(acc_wc_abc, w_idx, out_wc_abc)

            del hap1_t, hap2_t
            free_gpu_pool()

        # Post-process on GPU, single .get() per result
        if 'fst' in statistics or 'fst_hudson' in statistics:
            hudson_fst = cp.where(acc_fst_den > 0,
                                  acc_fst_num / acc_fst_den, cp.nan).get()
            if 'fst' in statistics:
                results['fst'] = hudson_fst
            if 'fst_hudson' in statistics:
                results['fst_hudson'] = hudson_fst

        if 'fst_wc' in statistics:
            results['fst_wc'] = cp.where(acc_wc_abc > 0,
                                          acc_wc_a / acc_wc_abc, cp.nan).get()

        # Without this, a site with too few calls to contribute would still
        # count toward the span, reading as invariant rather than unmeasured.
        # dxy and da need their own count, since they use different site gates.
        if 'dxy' in statistics:
            r = _missing_data_rescale(n_variants_raw, per_base, False,
                                      lambda: acc_dxy_valid.get())
            vals = acc_dxy * cp.asarray(r) if r is not None else acc_dxy
            if per_base:
                wb = cp.asarray(window_bases)
                results['dxy'] = cp.where(wb > 0,
                                          vals / wb, cp.nan).get()
            else:
                results['dxy'] = vals.get()

        if 'da' in statistics:
            # da's between term is fst's (both pops >= 2 calls), not dxy's.
            da_sum = acc_fst_den - (acc_pi1 + acc_pi2) / 2.0
            r = _missing_data_rescale(n_variants_raw, per_base, False,
                                      lambda: acc_da_valid.get())
            if r is not None:
                da_sum = da_sum * cp.asarray(r)
            if per_base:
                wb = cp.asarray(window_bases)
                results['da'] = cp.where(wb > 0,
                                         da_sum / wb, cp.nan).get()
            else:
                results['da'] = da_sum.get()

        if excluded_to_empty is not None:
            for name in ('dxy', 'da'):
                if name in results:
                    results[name] = np.where(excluded_to_empty, np.nan, results[name])

    remaining = (set(statistics) & _FUSED_STATS) - _CHUNKED_OWN_STATS
    if remaining:
        extra = windowed_statistics_fused(
            haplotype_matrix, bp_bins=bp_bins,
            statistics=tuple(remaining),
            population=population, pop1=pop1, pop2=pop2,
            per_base=per_base, is_accessible=is_accessible,
            _win_starts=_win_starts, _win_stops=_win_stops,
            missing_data=orig_missing_data, chrom=chrom, _sites=sites)
        for k, v in extra.items():
            if k not in results:
                results[k] = v

    return results


def _rate(total, window_bases, per_base):
    """A per-window total as a per-base rate when ``per_base``, else as is.

    A window with no bases has no rate, so it is NaN.
    """
    if not per_base:
        return total
    return np.where(window_bases > 0, total / window_bases, np.nan)


def _compute_fused_garud_h(matrix, win_start, win_stop, statistics, results,
                           missing_data='include'):
    """Windowed Garud's H and distinct-haplotype count into ``results``.

    A window with no variants reports one haplotype. A window with no
    complete haplotype under 'include' is NaN, so every column, the
    haplotype count too, is float64. ``n_complete_haplotypes`` holds the
    number of haplotypes with no missing call in each window.
    """
    *values, n_complete = garud_h_windows(matrix.haplotypes, win_start,
                                          win_stop, missing_data=missing_data)
    for name, column in zip(_FUSED_GARUD_STATS, values):
        if name in statistics:
            results[name] = column
    results['n_complete_haplotypes'] = n_complete


