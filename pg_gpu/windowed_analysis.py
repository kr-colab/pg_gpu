"""
Windowed analysis module for computing population genetics statistics across genomic windows.

This module provides efficient computation of statistics in sliding or non-overlapping windows
with intelligent memory management and GPU acceleration.
"""

import numpy as np
import cupy as cp
import pandas as pd
from typing import List, Dict, Union, Optional, Callable, Iterator, Tuple, Any
from dataclasses import dataclass
import warnings
from tqdm import tqdm

from .haplotype_matrix import HaplotypeMatrix
from . import ld_statistics
from . import divergence
from . import diversity


def _compute_window_bases(haplotype_matrix, win_starts, win_stops,
                          is_accessible=None):
    """Compute per-window accessible base counts for per-base normalization.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Source matrix (checked for accessible_mask attribute).
    win_starts, win_stops : array_like
        Window boundary arrays (numpy, not cupy).
    is_accessible : array_like, optional
        Explicit accessibility mask (takes precedence over matrix attribute).

    Returns
    -------
    numpy.ndarray, float64
        Accessible base count per window.
    """
    ws = np.asarray(win_starts, dtype=np.float64)
    we = np.asarray(win_stops, dtype=np.float64)

    if is_accessible is not None:
        from .accessible import AccessibleMask
        amask = AccessibleMask(np.asarray(is_accessible, dtype=bool))
        return amask.count_accessible_windows(
            ws.astype(np.int64), we.astype(np.int64))

    if haplotype_matrix.has_accessible_mask:
        return haplotype_matrix.accessible_mask.count_accessible_windows(
            ws.astype(np.int64), we.astype(np.int64))

    return we - ws


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


class MemoryManager:
    """Manages GPU memory allocation and chunking strategy."""

    def __init__(self, gpu_memory_limit: Union[str, int] = 'auto'):
        self.gpu_memory_limit = self._parse_memory_limit(gpu_memory_limit)
        self._gpu_memory_info = {}

    def _parse_memory_limit(self, limit: Union[str, int]) -> int:
        """Parse memory limit string (e.g., '8GB') to bytes."""
        if limit == 'auto':
            # Use 80% of available GPU memory
            mempool = cp.get_default_memory_pool()
            return int(cp.cuda.Device().mem_info[1] * 0.8)
        elif isinstance(limit, str):
            # Parse strings like '8GB', '512MB'
            units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
            for unit, multiplier in units.items():
                if limit.upper().endswith(unit):
                    return int(float(limit[:-len(unit)]) * multiplier)
            raise ValueError(f"Invalid memory limit format: {limit}")
        else:
            return int(limit)

    def estimate_window_memory(self, n_variants: int, n_samples: int,
                             statistics: List[str]) -> int:
        """Estimate memory required for processing a window."""
        # Base matrix memory
        matrix_memory = n_variants * n_samples * 4  # float32

        # Add overhead for statistics computation
        overhead_multiplier = 1.5 + 0.2 * len(statistics)

        # LD statistics need pairwise computations
        if any('ld' in stat.lower() for stat in statistics):
            overhead_multiplier += n_variants / 1000  # Scale with variant count

        return int(matrix_memory * overhead_multiplier)

    def determine_chunk_size(self, total_variants: int, n_samples: int,
                           window_params: WindowParams,
                           statistics: List[str]) -> int:
        """Determine optimal chunk size for processing."""
        # Estimate memory per variant
        window_memory = self.estimate_window_memory(
            window_params.window_size, n_samples, statistics
        )

        # Calculate how many windows fit in memory
        windows_per_chunk = max(1, self.gpu_memory_limit // window_memory)

        # Account for overlapping windows
        if window_params.step_size < window_params.window_size:
            overlap_factor = window_params.window_size / window_params.step_size
            chunk_variants = int(windows_per_chunk * window_params.step_size +
                               window_params.window_size)
        else:
            chunk_variants = windows_per_chunk * window_params.window_size

        # Ensure reasonable chunk size
        chunk_variants = min(chunk_variants, total_variants)
        chunk_variants = max(chunk_variants, window_params.window_size * 2)

        return chunk_variants


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
                 span_denominator: str = 'total'):
        self.statistics = statistics
        self.populations = populations or []
        self.custom_stat_kwargs = custom_stat_kwargs or {}
        self.ld_bins = ld_bins or [0, 1000, 5000, 10000, 50000]
        self.missing_data = missing_data
        self.span_denominator = span_denominator

        # Categorize statistics
        self._categorize_statistics()

    def _categorize_statistics(self):
        """Categorize statistics by type for efficient computation."""
        _is_pairwise = self.missing_data == 'pairwise'

        # In pairwise mode, pi and dxy return components for proper aggregation.
        # For other modes span_normalize controls the denominator.
        self.SINGLE_POP_STATS = {
            'pi': lambda w: diversity.pi(
                w.matrix, span_normalize=not _is_pairwise,
                missing_data=self.missing_data,
                span_denominator=self.span_denominator,
                return_components=_is_pairwise),
            'theta_w': lambda w: diversity.theta_w(
                w.matrix, span_normalize=not _is_pairwise,
                missing_data=self.missing_data,
                span_denominator=self.span_denominator,
                return_components=_is_pairwise),
            'tajimas_d': lambda w: diversity.tajimas_d(w.matrix, missing_data=self.missing_data),
            'n_variants': lambda w: w.n_variants,
            'n_singletons': lambda w: diversity.singleton_count(w.matrix, missing_data=self.missing_data),
            'segregating_sites': lambda w: diversity.segregating_sites(w.matrix, missing_data=self.missing_data),
        }

        self.TWO_POP_STATS = {
            'dxy': lambda w, p1, p2: divergence.dxy(
                w.matrix, p1, p2,
                missing_data=self.missing_data,
                span_denominator=self.span_denominator == 'total',
                return_components=_is_pairwise),
            'fst': lambda w, p1, p2: divergence.fst(w.matrix, p1, p2, missing_data=self.missing_data),
            'fst_hudson': lambda w, p1, p2: divergence.fst_hudson(w.matrix, p1, p2, missing_data=self.missing_data),
            'fst_wc': lambda w, p1, p2: divergence.fst_weir_cockerham(w.matrix, p1, p2, missing_data=self.missing_data),
            'da': lambda w, p1, p2: divergence.da(w.matrix, p1, p2,
                                                missing_data=self.missing_data,
                                                span_denominator=self.span_denominator == 'total'),
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

        # Skip if no variants
        if window.n_variants == 0:
            # Fill with NaN for all statistics
            for stat in self.statistics:
                if isinstance(stat, str):
                    results[stat] = np.nan
                else:
                    results[stat.__name__] = np.nan
            return results

        # Single population statistics
        for stat in self.single_pop_stats:
            if self.populations:
                # Compute for each population
                for pop in self.populations:
                    pop_matrix = self._get_population_matrix(window.matrix, pop)
                    val = self.SINGLE_POP_STATS[stat](
                        WindowData(window.chrom, window.start, window.end,
                                 window.center, pop_matrix, pop_matrix.num_variants,
                                 window.window_id)
                    )
                    key = f"{stat}_{pop}"
                    self._store_result(results, key, val)
            else:
                val = self.SINGLE_POP_STATS[stat](window)
                self._store_result(results, stat, val)

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

        # Custom statistics
        for stat in self.custom_stats:
            kwargs = self.custom_stat_kwargs.get(stat.__name__, {})
            results[stat.__name__] = stat(window, **kwargs)

        return results

    @staticmethod
    def _store_result(results: Dict, key: str, val):
        """Store a scalar or PairwiseResult into the results dict."""
        from .diversity import PairwiseResult
        if isinstance(val, PairwiseResult):
            results[key] = val.value
            results[f"{key}_diffs"] = val.total_diffs
            results[f"{key}_comps"] = val.total_comps
            results[f"{key}_missing"] = val.total_missing
            results[f"{key}_n_sites"] = val.n_sites
        else:
            results[key] = val

    def _get_population_matrix(self, matrix: HaplotypeMatrix,
                             pop: str) -> HaplotypeMatrix:
        """Extract population-specific haplotype matrix."""
        if pop not in matrix.sample_sets:
            raise ValueError(f"Population {pop} not found in sample_sets")

        pop_indices = matrix.sample_sets[pop]
        pop_haplotypes = matrix.haplotypes[pop_indices, :]

        return HaplotypeMatrix(
            pop_haplotypes,
            matrix.positions,
            matrix.chrom_start,
            matrix.chrom_end,
            sample_sets={'all': list(range(len(pop_indices)))},
            n_total_sites=matrix.n_total_sites,
        )


class WindowIterator:
    """Iterates over genomic windows."""

    def __init__(self, haplotype_matrix: HaplotypeMatrix,
                 window_params: WindowParams):
        self._parent_mask = haplotype_matrix.accessible_mask
        self.matrix = haplotype_matrix
        self.params = window_params
        self.positions = self.matrix.positions

        # Get positions as numpy array for easier manipulation
        if isinstance(self.positions, cp.ndarray):
            self.positions_np = self.positions.get()
        else:
            self.positions_np = self.positions

    def _attach_window_mask(self, window_matrix, start, end):
        """Set per-window n_total_sites from the parent's accessible mask."""
        if self._parent_mask is not None:
            window_matrix.n_total_sites = \
                self._parent_mask.count_accessible(start, end)

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
        chrom_start = int(self.positions_np[0])
        chrom_end = int(self.positions_np[-1])

        window_id = 0
        start = chrom_start

        while start < chrom_end:
            end = start + self.params.window_size
            center = (start + end) // 2

            # Find variants in window
            mask = (self.positions_np >= start) & (self.positions_np < end)
            variant_indices = np.where(mask)[0]

            if len(variant_indices) > 0:
                # Extract window matrix
                window_matrix = self.matrix.get_subset(variant_indices)
                # Set correct chromosome coordinates for span normalization
                window_matrix.chrom_start = start
                window_matrix.chrom_end = end - 1  # end is exclusive in our window definition
                self._attach_window_mask(window_matrix, start, end)

                yield WindowData(
                    chrom=1,  # TODO: Handle multiple chromosomes
                    start=start,
                    end=end,
                    center=center,
                    matrix=window_matrix,
                    n_variants=len(variant_indices),
                    window_id=window_id
                )

            window_id += 1
            start += self.params.step_size

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
            self._attach_window_mask(window_matrix, window_start, window_end + 1)

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
            start = region['start']
            end = region['end']
            center = (start + end) // 2

            # Find variants in region
            mask = (self.positions_np >= start) & (self.positions_np < end)
            variant_indices = np.where(mask)[0]

            if len(variant_indices) > 0:
                window_matrix = self.matrix.get_subset(variant_indices)
                self._attach_window_mask(window_matrix, start, end)

                yield WindowData(
                    chrom=region.get('chrom', 1),
                    start=start,
                    end=end,
                    center=center,
                    matrix=window_matrix,
                    n_variants=len(variant_indices),
                    window_id=window_id
                )

    def count_windows(self) -> int:
        """Count total number of windows."""
        if self.params.window_type == 'bp':
            chrom_start = int(self.positions_np[0])
            chrom_end = int(self.positions_np[-1])
            return max(1, (chrom_end - chrom_start - self.params.window_size) //
                      self.params.step_size + 1)
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
                 span_denominator: str = 'total'):
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
        span_denominator : str
            'total' - Use total genomic span (chrom_end - chrom_start)
            'sites' - Use number of sites analyzed
            'callable' - Use span from first to last site included in analysis
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
        self.span_denominator = span_denominator

        # Initialize components
        self.memory_manager = MemoryManager(gpu_memory_limit)
        self.stats_computer = StatisticsComputer(
            statistics, populations, custom_stat_kwargs, ld_bins=self.ld_bins,
            missing_data=missing_data, span_denominator=span_denominator
        )

    def compute(self, haplotype_matrix: HaplotypeMatrix) -> pd.DataFrame:
        """
        Compute windowed statistics for entire haplotype matrix.

        Parameters
        ----------
        haplotype_matrix : HaplotypeMatrix
            Input haplotype data

        Returns
        -------
        pd.DataFrame
            Results with statistics for each window
        """
        # Set population info if needed
        if self.populations and not haplotype_matrix.sample_sets:
            warnings.warn("Populations specified but haplotype_matrix has no sample_sets")

        # Create window iterator
        window_iter = WindowIterator(haplotype_matrix, self.window_params)
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

    def compute_region(self, haplotype_matrix: HaplotypeMatrix,
                      chrom: Union[str, int],
                      start: int,
                      end: int) -> pd.DataFrame:
        """
        Compute statistics for a specific genomic region.

        Parameters
        ----------
        haplotype_matrix : HaplotypeMatrix
            Input haplotype data
        chrom : str or int
            Chromosome identifier
        start : int
            Region start position
        end : int
            Region end position

        Returns
        -------
        pd.DataFrame
            Results for windows in the specified region
        """
        # Extract region from matrix
        region_matrix = haplotype_matrix.get_subset_from_range(start, end)

        # Compute statistics
        return self.compute(region_matrix)

    def compute_streaming(self, haplotype_matrix: HaplotypeMatrix,
                         batch_size: int = 100) -> Iterator[pd.DataFrame]:
        """
        Compute statistics in batches for memory efficiency.

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
        window_iter = WindowIterator(haplotype_matrix, self.window_params)
        total_windows = window_iter.count_windows()

        batch_results = []
        with tqdm(total=total_windows, disable=not self.progress_bar,
                 desc="Computing windows") as pbar:
            for window in window_iter:
                window_results = self.stats_computer.compute(window)
                batch_results.append(window_results)
                pbar.update(1)

                if len(batch_results) >= batch_size:
                    yield pd.DataFrame(batch_results)
                    batch_results = []

            # Yield remaining results
            if batch_results:
                yield pd.DataFrame(batch_results)


# Convenience function for simple usage
def windowed_analysis(haplotype_matrix: HaplotypeMatrix,
                     window_size: int = 50000,
                     step_size: Optional[int] = None,
                     statistics: List[str] = ['pi'],
                     populations: Optional[List[str]] = None,
                     missing_data: str = 'include',
                     span_denominator: str = 'total',
                     accessible_bed: str = None,
                     chrom: str = None,
                     **kwargs) -> pd.DataFrame:
    """
    Convenience function for windowed analysis.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Input haplotype data
    window_size : int
        Window size in base pairs
    step_size : int, optional
        Step size. If None, uses window_size
    statistics : list
        Statistics to compute
    populations : list, optional
        Population names
    missing_data : str
        'include' - Use all sites, calculate from available data per site
        'exclude' - Only use sites with no missing data
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
        'accessible' - Use number of accessible sites from mask
    accessible_bed : str, optional
        Path to a BED file defining accessible/callable regions.
        If provided and the matrix has no mask, loads the mask.
    **kwargs
        Additional arguments passed to WindowedAnalyzer

    Returns
    -------
    pd.DataFrame
        Windowed statistics results
    """
    if accessible_bed is not None and not haplotype_matrix.has_accessible_mask:
        haplotype_matrix.set_accessible_mask(accessible_bed, chrom=chrom)
    if step_size is None:
        step_size = window_size

    # Fast path: use fused CUDA kernels when possible.
    fused_single = {'pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                    'singletons', 'theta_h', 'fay_wu_h', 'max_daf'}
    fused_two = {'fst', 'fst_hudson', 'fst_wc', 'dxy', 'da'}
    fused_garud = {'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1',
                   'haplotype_count'}
    fused_selection = {'mean_nsl'}
    fused_diploshic = {'snp_dist_mean', 'snp_dist_var', 'snp_dist_min',
                       'snp_dist_max', 'mu_var', 'mu_sfs', 'mu_ld',
                       'daf_hist', 'zns', 'omega',
                       'dist_var', 'dist_skew', 'dist_kurt'}
    fused_all = (fused_single | fused_two | fused_garud | fused_selection
                 | fused_diploshic)
    requested = set(statistics)

    can_fuse = (missing_data in ('include', 'project')
                and requested <= fused_all)

    if can_fuse:
        if haplotype_matrix.device == 'CPU':
            haplotype_matrix.transfer_to_gpu()
        positions = haplotype_matrix.positions
        if hasattr(positions, 'get'):
            positions = positions.get()
        positions = np.asarray(positions)

        chrom_start = haplotype_matrix.chrom_start
        chrom_end = haplotype_matrix.chrom_end
        if chrom_start is None:
            chrom_start = int(positions[0])
        if chrom_end is None:
            chrom_end = int(positions[-1])
        chrom_start = int(chrom_start)
        chrom_end = int(chrom_end)

        # Build window start/stop arrays (supports overlapping windows)
        win_starts = np.arange(chrom_start, chrom_end, step_size,
                               dtype=np.float64)
        win_stops = win_starts + window_size
        # Build equivalent bp_bins for _compute_window_ranges
        bp_bins = np.concatenate([win_starts, [win_stops[-1]]])

        pop1 = populations[0] if populations and len(populations) >= 1 else None
        pop2 = populations[1] if populations and len(populations) >= 2 else None

        # Choose chunked or single-shot fused based on memory
        n_hap = haplotype_matrix.num_haplotypes
        n_var = haplotype_matrix.num_variants
        transpose_bytes = n_var * n_hap  # int8
        free_mem = cp.cuda.Device().mem_info[0]
        use_chunked = transpose_bytes * 2 > free_mem * 0.7

        fused_fn = (windowed_statistics_fused_chunked if use_chunked
                     else windowed_statistics_fused)
        result_dict = fused_fn(
            haplotype_matrix,
            bp_bins=bp_bins,
            statistics=tuple(statistics),
            pop1=pop1,
            pop2=pop2,
            per_base=(span_denominator in ('total', 'accessible')),
            _win_starts=win_starts,
            _win_stops=win_stops,
            missing_data=missing_data,
        )
        return pd.DataFrame(result_dict)

    # Fallback: per-window Python loop
    analyzer = WindowedAnalyzer(
        window_type='bp',
        window_size=window_size,
        step_size=step_size,
        statistics=statistics,
        populations=populations,
        missing_data=missing_data,
        span_denominator=span_denominator,
        **kwargs
    )
    return analyzer.compute(haplotype_matrix)


# Helper functions for built-in statistics



def _compute_mean_r2(matrix: HaplotypeMatrix, max_distance: int,
                     **kwargs) -> float:
    """Compute mean r² for variant pairs within a genomic distance."""
    r2_matrix = matrix.pairwise_r2()
    positions = matrix.positions

    pos_i, pos_j = cp.meshgrid(positions, positions, indexing='ij')
    distances = cp.abs(pos_j - pos_i)
    mask = (distances > 0) & (distances <= max_distance)
    if cp.any(mask):
        return float(cp.mean(r2_matrix[mask]).get())
    return np.nan


# ---------------------------------------------------------------------------
# Fused CUDA kernel: one block per window, all stats in one pass
# ---------------------------------------------------------------------------

# Haplotype data is transposed before kernel launch so variants are the
# leading dimension (column-major for haplotype access). This ensures
# coalesced memory reads when threads iterate over haplotypes.
_fused_windowed_kernel_v2 = cp.RawKernel(r'''
extern "C" __global__
void fused_windowed_stats_v2(const signed char* hap_t,
                             const long long* win_start,
                             const long long* win_stop,
                             int n_hap, int n_total_var, int n_windows,
                             double* out_mpd_sum,
                             double* out_seg_count,
                             double* out_sing_count,
                             double* out_var_count,
                             double* out_theta_h_sum,
                             double* out_max_daf) {
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
        }
        return;
    }

    double dn = (double)n_hap;
    double t_mpd = 0.0, t_seg = 0.0, t_sing = 0.0;
    double t_count = 0.0, t_theta_h = 0.0, t_max_daf = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;
        int dac = 0;
        const signed char* row = hap_t + (long long)v * n_hap;
        for (int h = 0; h < n_hap; h++) {
            if (row[h] > 0) dac++;
        }

        double p = (double)dac / dn;
        t_mpd += 2.0 * p * (1.0 - p) * dn / (dn - 1.0);

        int is_seg = (dac > 0 && dac < n_hap) ? 1 : 0;
        t_seg += is_seg;
        t_sing += (dac == 1 || dac == n_hap - 1) ? 1.0 : 0.0;
        t_count += 1.0;

        if (is_seg) {
            t_theta_h += 2.0 * (double)dac * (double)dac / (dn * (dn - 1.0));
        }
        if (p > t_max_daf) t_max_daf = p;
    }

    // Sum reduction for 5 accumulators
    __shared__ double smem[6 * 256];
    int tid = threadIdx.x;
    smem[tid]          = t_mpd;
    smem[256 + tid]    = t_seg;
    smem[512 + tid]    = t_sing;
    smem[768 + tid]    = t_count;
    smem[1024 + tid]   = t_theta_h;
    smem[1280 + tid]   = t_max_daf;  // will be max-reduced
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
    }
}
''', 'fused_windowed_stats_v2')


# Two-population fused kernel for FST and Dxy
_fused_windowed_twopop_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_windowed_twopop(const signed char* hap1_t,
                           const signed char* hap2_t,
                           const long long* win_start,
                           const long long* win_stop,
                           int n_hap1, int n_hap2,
                           int n_total_var, int n_windows,
                           double* out_fst_num,
                           double* out_fst_den,
                           double* out_dxy_sum,
                           double* out_pi1_sum,
                           double* out_pi2_sum,
                           double* out_wc_a_sum,
                           double* out_wc_ab_sum) {
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
            out_wc_ab_sum[wid] = 0.0;
        }
        return;
    }

    double dn1 = (double)n_hap1;
    double dn2 = (double)n_hap2;

    // Weir-Cockerham constants (haploid, r=2, constant sample sizes)
    double n_total = dn1 + dn2;
    double n_bar = n_total / 2.0;
    double n_C = (n_total - (dn1*dn1 + dn2*dn2) / n_total);  // r-1 = 1

    double t_fst_num = 0.0, t_fst_den = 0.0;
    double t_dxy = 0.0, t_pi1 = 0.0, t_pi2 = 0.0;
    double t_wc_a = 0.0, t_wc_ab = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;

        int ac1_1 = 0, ac2_1 = 0;
        const signed char* row1 = hap1_t + (long long)v * n_hap1;
        for (int h = 0; h < n_hap1; h++) {
            if (row1[h] > 0) ac1_1++;
        }
        const signed char* row2 = hap2_t + (long long)v * n_hap2;
        for (int h = 0; h < n_hap2; h++) {
            if (row2[h] > 0) ac2_1++;
        }

        double ac1_0 = dn1 - ac1_1;
        double ac2_0 = dn2 - ac2_1;
        double d_ac1_1 = (double)ac1_1;
        double d_ac2_1 = (double)ac2_1;

        // Hudson: within-pop mean pairwise difference
        double n1_pairs = dn1 * (dn1 - 1.0) / 2.0;
        double n1_same = (ac1_0 * (ac1_0 - 1.0) + d_ac1_1 * (d_ac1_1 - 1.0)) / 2.0;
        double mpd1 = (n1_pairs > 0) ? (n1_pairs - n1_same) / n1_pairs : 0.0;

        double n2_pairs = dn2 * (dn2 - 1.0) / 2.0;
        double n2_same = (ac2_0 * (ac2_0 - 1.0) + d_ac2_1 * (d_ac2_1 - 1.0)) / 2.0;
        double mpd2 = (n2_pairs > 0) ? (n2_pairs - n2_same) / n2_pairs : 0.0;

        double within = (mpd1 + mpd2) / 2.0;

        // Between-pop mean pairwise difference
        double n_between = dn1 * dn2;
        double n_between_same = ac1_0 * ac2_0 + d_ac1_1 * d_ac2_1;
        double between = (n_between > 0) ? (n_between - n_between_same) / n_between : 0.0;

        t_fst_num += between - within;
        t_fst_den += between;
        t_dxy += between;
        t_pi1 += mpd1;
        t_pi2 += mpd2;

        // Weir-Cockerham (haploid, h_bar=0, r=2)
        double p1 = d_ac1_1 / dn1;
        double p2 = d_ac2_1 / dn2;
        double p_bar = (dn1 * p1 + dn2 * p2) / n_total;
        double s2 = (dn1 * (p1 - p_bar) * (p1 - p_bar) +
                     dn2 * (p2 - p_bar) * (p2 - p_bar)) / n_bar;
        double pq = p_bar * (1.0 - p_bar);

        double a_val = 0.0, b_val = 0.0;
        if (n_bar > 1.0 && n_C > 0.0) {
            a_val = (n_bar / n_C) * (s2 - (1.0 / (n_bar - 1.0)) * (pq - s2 / 2.0));
            b_val = (n_bar / (n_bar - 1.0)) * (pq - s2 / 2.0);
        }
        t_wc_a += a_val;
        t_wc_ab += a_val + b_val;
    }

    // Block reduction (7 values)
    __shared__ double smem[7 * 256];
    int tid = threadIdx.x;
    smem[tid]          = t_fst_num;
    smem[256 + tid]    = t_fst_den;
    smem[512 + tid]    = t_dxy;
    smem[768 + tid]    = t_pi1;
    smem[1024 + tid]   = t_pi2;
    smem[1280 + tid]   = t_wc_a;
    smem[1536 + tid]   = t_wc_ab;
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
        out_wc_ab_sum[wid]= smem[1536];
    }
}
''', 'fused_windowed_twopop')


# Garud's H fused kernel: one block per window, sorts haplotype hashes
# in shared memory to count unique patterns and compute H statistics.
_fused_garud_h_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_garud_h(const double* hash1,   // (n_windows, n_hap)
                   const double* hash2,   // (n_windows, n_hap)
                   int n_hap, int n_windows,
                   double* out_h1, double* out_h12,
                   double* out_h123, double* out_h2h1,
                   double* out_n_distinct) {
    int wid = blockIdx.x;
    if (wid >= n_windows) return;
    int tid = threadIdx.x;

    // Load hashes into shared memory for sorting
    // Max 1024 haplotypes supported (shared memory limit)
    extern __shared__ double shm[];
    double* s_h1 = shm;            // n_hap doubles
    double* s_h2 = shm + n_hap;    // n_hap doubles

    // Each thread loads its element
    if (tid < n_hap) {
        s_h1[tid] = hash1[wid * n_hap + tid];
        s_h2[tid] = hash2[wid * n_hap + tid];
    }
    __syncthreads();

    // Simple odd-even sort on hash1 (secondary on hash2 for ties)
    // For n_hap <= 256 this is fast in shared memory
    for (int phase = 0; phase < n_hap; phase++) {
        int i = 2 * tid + (phase & 1);
        if (i + 1 < n_hap) {
            bool do_swap = false;
            if (s_h1[i] > s_h1[i + 1]) {
                do_swap = true;
            } else if (s_h1[i] == s_h1[i + 1] && s_h2[i] > s_h2[i + 1]) {
                do_swap = true;
            }
            if (do_swap) {
                double tmp;
                tmp = s_h1[i]; s_h1[i] = s_h1[i+1]; s_h1[i+1] = tmp;
                tmp = s_h2[i]; s_h2[i] = s_h2[i+1]; s_h2[i+1] = tmp;
            }
        }
        __syncthreads();
    }

    // Thread 0: count unique haplotypes, compute frequencies, derive H stats
    if (tid == 0) {
        // Count distinct haplotypes and collect top-3 frequencies
        double inv_n = 1.0 / (double)n_hap;
        double tol = 1e-3;

        // Walk sorted array, count runs
        // We need: sum(f_i^2), and the top 3 frequencies
        double sum_f2 = 0.0;
        double top3[3] = {0.0, 0.0, 0.0};
        int run_len = 1;
        int n_distinct = 0;

        for (int i = 1; i <= n_hap; i++) {
            bool boundary = (i == n_hap);
            if (!boundary) {
                double d1 = s_h1[i] - s_h1[i-1];
                double d2 = s_h2[i] - s_h2[i-1];
                if (d1 < 0) d1 = -d1;
                if (d2 < 0) d2 = -d2;
                boundary = (d1 > tol) || (d2 > tol);
            }
            if (boundary) {
                n_distinct++;
                double f = (double)run_len * inv_n;
                sum_f2 += f * f;
                if (f > top3[0]) {
                    top3[2] = top3[1]; top3[1] = top3[0]; top3[0] = f;
                } else if (f > top3[1]) {
                    top3[2] = top3[1]; top3[1] = f;
                } else if (f > top3[2]) {
                    top3[2] = f;
                }
                run_len = 1;
            } else {
                run_len++;
            }
        }

        double h1_val = sum_f2;
        double h12_val = (top3[0] + top3[1]) * (top3[0] + top3[1])
                       + (sum_f2 - top3[0]*top3[0] - top3[1]*top3[1]);
        double h123_val = (top3[0] + top3[1] + top3[2]) * (top3[0] + top3[1] + top3[2])
                        + (sum_f2 - top3[0]*top3[0] - top3[1]*top3[1] - top3[2]*top3[2]);
        double h2_val = h1_val - top3[0] * top3[0];
        double h2h1_val = (h1_val > 0.0) ? h2_val / h1_val : 0.0;

        out_h1[wid]   = h1_val;
        out_h12[wid]  = h12_val;
        out_h123[wid] = h123_val;
        out_h2h1[wid] = h2h1_val;
        out_n_distinct[wid] = (double)n_distinct;
    }
}
''', 'fused_garud_h')


def _compute_window_ranges(positions, bp_bins):
    """Map window edges to variant index ranges using searchsorted.

    Returns (win_start, win_stop) as CuPy int64 arrays where
    win_start[i]:win_stop[i] are the variant indices in window i.
    """
    n_windows = len(bp_bins) - 1
    win_start = cp.searchsorted(positions, bp_bins[:-1], side='left')
    win_stop = cp.searchsorted(positions, bp_bins[1:], side='left')
    return win_start.astype(cp.int64), win_stop.astype(cp.int64)


def windowed_statistics_fused(haplotype_matrix: HaplotypeMatrix,
                              bp_bins,
                              statistics=('pi', 'theta_w', 'tajimas_d',
                                          'segregating_sites', 'singletons'),
                              population=None,
                              pop1=None,
                              pop2=None,
                              per_base: bool = True,
                              is_accessible=None,
                              _win_starts=None,
                              _win_stops=None,
                              missing_data='include'):
    """GPU-native windowed statistics using fused CUDA kernels.

    One kernel launch processes ALL windows in parallel. Each thread block
    handles one window, with threads cooperatively reducing over variants.
    Reads the haplotype matrix once and computes all statistics simultaneously.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    bp_bins : array_like
        Window edges in base pairs. N+1 edges define N windows.
    statistics : tuple of str
        Statistics to compute. Single-pop: 'pi', 'theta_w', 'tajimas_d',
        'segregating_sites', 'singletons'. Two-pop: 'fst', 'dxy'.
    population : str or list, optional
        Population for single-pop statistics.
    pop1, pop2 : str or list, optional
        Populations for two-pop statistics.
    per_base : bool
        Normalize by window size in base pairs.
    is_accessible : array_like, optional
        Accessibility mask for per-base normalization.

    Returns
    -------
    dict
        Maps statistic names to numpy arrays of shape (n_windows,).
    """
    from ._utils import get_population_matrix

    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap_raw = matrix.haplotypes
    n_hap = hap_raw.shape[0]
    n_total_var = hap_raw.shape[1]
    # transpose for coalesced kernel access: (n_total_var, n_hap)
    hap = cp.ascontiguousarray(hap_raw.T.astype(cp.int8))

    positions = matrix.positions
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions)

    # Support overlapping windows via explicit start/stop arrays
    if _win_starts is not None and _win_stops is not None:
        ws_gpu = cp.asarray(_win_starts, dtype=cp.float64)
        we_gpu = cp.asarray(_win_stops, dtype=cp.float64)
        n_windows = len(ws_gpu)
        win_start = cp.searchsorted(positions, ws_gpu, side='left').astype(cp.int64)
        win_stop = cp.searchsorted(positions, we_gpu, side='left').astype(cp.int64)
    else:
        bp_bins_gpu = cp.asarray(bp_bins, dtype=cp.float64)
        n_windows = len(bp_bins_gpu) - 1
        win_start, win_stop = _compute_window_ranges(positions, bp_bins_gpu)
        ws_gpu = bp_bins_gpu[:-1]
        we_gpu = bp_bins_gpu[1:]

    results = {
        'window_start': ws_gpu.get() if hasattr(ws_gpu, 'get') else ws_gpu,
        'window_stop': we_gpu.get() if hasattr(we_gpu, 'get') else we_gpu,
    }

    if per_base:
        window_bases = _compute_window_bases(
            haplotype_matrix, results['window_start'],
            results['window_stop'], is_accessible)

    # single-pop stats via fused kernel
    single_pop_stats = {'pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                        'singletons', 'theta_h', 'fay_wu_h', 'max_daf'}
    single_pop_requested = any(s in statistics for s in single_pop_stats)
    if single_pop_requested:
        out_mpd = cp.zeros(n_windows, dtype=cp.float64)
        out_seg = cp.zeros(n_windows, dtype=cp.float64)
        out_sing = cp.zeros(n_windows, dtype=cp.float64)
        out_count = cp.zeros(n_windows, dtype=cp.float64)
        out_theta_h = cp.zeros(n_windows, dtype=cp.float64)
        out_max_daf = cp.zeros(n_windows, dtype=cp.float64)

        block = 256
        grid = n_windows

        _fused_windowed_kernel_v2(
            (grid,), (block,),
            (hap, win_start, win_stop,
             np.int32(n_hap), np.int32(n_total_var), np.int32(n_windows),
             out_mpd, out_seg, out_sing, out_count, out_theta_h, out_max_daf))

        mpd_sum = out_mpd.get()
        seg_count = out_seg.get()
        sing_count = out_sing.get()
        var_count = out_count.get()
        theta_h_sum = out_theta_h.get()
        max_daf = out_max_daf.get()

        results['n_variants'] = var_count.astype(int)

        if 'pi' in statistics:
            if per_base:
                results['pi'] = np.where(window_bases > 0,
                                         mpd_sum / window_bases, np.nan)
            else:
                results['pi'] = mpd_sum

        if 'theta_w' in statistics:
            a1 = np.sum(1.0 / np.arange(1, n_hap))
            theta_abs = seg_count / a1
            if per_base:
                results['theta_w'] = np.where(window_bases > 0,
                                              theta_abs / window_bases, np.nan)
            else:
                results['theta_w'] = theta_abs

        if 'segregating_sites' in statistics:
            results['segregating_sites'] = seg_count.astype(int)

        if 'singletons' in statistics:
            results['singletons'] = sing_count.astype(int)

        if 'tajimas_d' in statistics:
            n = n_hap
            a1 = np.sum(1.0 / np.arange(1, n))
            a2 = np.sum(1.0 / np.arange(1, n) ** 2)
            b1 = (n + 1) / (3 * (n - 1))
            b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
            c1 = b1 - 1 / a1
            c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
            e1 = c1 / a1
            e2 = c2 / (a1 ** 2 + a2)

            S = seg_count
            d_num = mpd_sum - S / a1
            d_var = e1 * S + e2 * S * (S - 1)
            d_std = np.sqrt(np.maximum(d_var, 0))
            tajd = np.where(d_std > 0, d_num / d_std, np.nan)
            tajd[S < 3] = np.nan
            results['tajimas_d'] = tajd

        if 'theta_h' in statistics:
            if per_base:
                results['theta_h'] = np.where(window_bases > 0,
                                              theta_h_sum / window_bases, np.nan)
            else:
                results['theta_h'] = theta_h_sum

        if 'fay_wu_h' in statistics:
            # H = pi - theta_H (absolute, unnormalized)
            results['fay_wu_h'] = mpd_sum - theta_h_sum

        if 'max_daf' in statistics:
            results['max_daf'] = max_daf

    # two-pop stats via fused kernel
    two_pop_stats = {'fst', 'fst_hudson', 'fst_wc', 'dxy', 'da'}
    two_pop_requested = any(s in statistics for s in two_pop_stats)
    if two_pop_requested:
        if pop1 is None or pop2 is None:
            raise ValueError("pop1 and pop2 required for fst/dxy/da")

        # Use filtered matrix so inaccessible variants are excluded
        m1 = get_population_matrix(matrix, pop1)
        m2 = get_population_matrix(matrix, pop2)
        if m1.device == 'CPU':
            m1.transfer_to_gpu()
        if m2.device == 'CPU':
            m2.transfer_to_gpu()

        n1 = m1.haplotypes.shape[0]
        n2 = m2.haplotypes.shape[0]
        hap1 = cp.ascontiguousarray(m1.haplotypes.T.astype(cp.int8))
        hap2 = cp.ascontiguousarray(m2.haplotypes.T.astype(cp.int8))

        out_fst_num = cp.zeros(n_windows, dtype=cp.float64)
        out_fst_den = cp.zeros(n_windows, dtype=cp.float64)
        out_dxy = cp.zeros(n_windows, dtype=cp.float64)
        out_pi1 = cp.zeros(n_windows, dtype=cp.float64)
        out_pi2 = cp.zeros(n_windows, dtype=cp.float64)
        out_wc_a = cp.zeros(n_windows, dtype=cp.float64)
        out_wc_ab = cp.zeros(n_windows, dtype=cp.float64)

        block = 256
        grid = n_windows

        _fused_windowed_twopop_kernel(
            (grid,), (block,),
            (hap1, hap2, win_start, win_stop,
             np.int32(n1), np.int32(n2),
             np.int32(n_total_var), np.int32(n_windows),
             out_fst_num, out_fst_den, out_dxy, out_pi1, out_pi2,
             out_wc_a, out_wc_ab))

        fst_num = out_fst_num.get()
        fst_den = out_fst_den.get()
        dxy_sum = out_dxy.get()
        pi1_sum = out_pi1.get()
        pi2_sum = out_pi2.get()
        wc_a = out_wc_a.get()
        wc_ab = out_wc_ab.get()

        if 'n_variants' not in results:
            results['n_variants'] = (win_stop - win_start).get().astype(int)

        if 'fst' in statistics or 'fst_hudson' in statistics:
            hudson_fst = np.where(fst_den > 0, fst_num / fst_den, np.nan)
            if 'fst' in statistics:
                results['fst'] = hudson_fst
            if 'fst_hudson' in statistics:
                results['fst_hudson'] = hudson_fst

        if 'fst_wc' in statistics:
            results['fst_wc'] = np.where(wc_ab > 0, wc_a / wc_ab, np.nan)

        if 'dxy' in statistics:
            if per_base:
                results['dxy'] = np.where(window_bases > 0,
                                          dxy_sum / window_bases, np.nan)
            else:
                results['dxy'] = dxy_sum

        if 'da' in statistics:
            if per_base:
                da_sum = dxy_sum - (pi1_sum + pi2_sum) / 2.0
                results['da'] = np.where(window_bases > 0,
                                         da_sum / window_bases, np.nan)
            else:
                results['da'] = dxy_sum - (pi1_sum + pi2_sum) / 2.0

    # Garud's H via fused kernel (SNP windows using prefix-sum hashing)
    garud_stats = {'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1',
                   'haplotype_count'}
    garud_requested = any(s in statistics for s in garud_stats)
    if garud_requested:
        _compute_fused_garud_h(matrix, population,
                               win_start, win_stop, n_windows, statistics,
                               results)

    # Per-site stats binned into windows via scatter_add
    bin_idx = cp.searchsorted(we_gpu, positions)
    in_range = (bin_idx >= 0) & (bin_idx < n_windows)

    # Shared DAC computation (used by daf_hist and mu_sfs)
    dac_gpu = None
    if any(s in statistics for s in ('daf_hist', 'mu_sfs')):
        dac_gpu = cp.sum(cp.maximum(matrix.haplotypes, 0).astype(cp.int32), axis=0)

    if 'mean_nsl' in statistics:
        from . import selection as sel
        nsl_gpu = cp.asarray(sel.nsl(matrix, population=population))
        valid = cp.isfinite(nsl_gpu) & in_range
        results['mean_nsl'] = _windowed_mean(nsl_gpu, bin_idx, valid, n_windows)

    # SNP distance stats per window
    snp_dist_stats = {'snp_dist_mean', 'snp_dist_var', 'snp_dist_min',
                      'snp_dist_max', 'mu_var'}
    if any(s in statistics for s in snp_dist_stats):
        ws_np, we_np = win_start.get(), win_stop.get()
        pos_cpu = positions.get() if hasattr(positions, 'get') else np.asarray(positions)
        sd_mean = np.full(n_windows, np.nan)
        sd_var = np.full(n_windows, np.nan)
        sd_min = np.full(n_windows, np.nan)
        sd_max = np.full(n_windows, np.nan)
        mu_var_arr = np.full(n_windows, np.nan)

        for wi in range(n_windows):
            s, e = int(ws_np[wi]), int(we_np[wi])
            if e - s < 2:
                if e - s == 1:
                    mu_var_arr[wi] = 1.0 / window_bases[wi] if per_base and window_bases[wi] > 0 else 1.0
                continue
            win_pos = pos_cpu[s:e]
            gaps = np.diff(win_pos).astype(np.float64)
            sd_mean[wi] = np.mean(gaps)
            sd_var[wi] = np.var(gaps)
            sd_min[wi] = np.min(gaps)
            sd_max[wi] = np.max(gaps)
            mu_var_arr[wi] = len(win_pos) / window_bases[wi] if per_base and window_bases[wi] > 0 else float(len(win_pos))

        for stat, arr in [('snp_dist_mean', sd_mean), ('snp_dist_var', sd_var),
                          ('snp_dist_min', sd_min), ('snp_dist_max', sd_max),
                          ('mu_var', mu_var_arr)]:
            if stat in statistics:
                results[stat] = arr

    # DAF histogram per window (GPU scatter)
    if 'daf_hist' in statistics:
        n_daf_bins = 20
        daf = dac_gpu.astype(cp.float64) / n_hap
        daf_bin = cp.minimum((daf * n_daf_bins).astype(cp.int32), n_daf_bins - 1)
        # Composite index: window * n_daf_bins + daf_bin
        composite = bin_idx * n_daf_bins + daf_bin
        valid_daf = in_range
        flat = _scatter_sum(cp.ones_like(composite[valid_daf], dtype=cp.float64),
                            composite[valid_daf], n_windows * n_daf_bins)
        hist_matrix = flat.get().reshape(n_windows, n_daf_bins)
        for b in range(n_daf_bins):
            results[f'daf_bin_{b}'] = hist_matrix[:, b]

    # muSFS: fraction of SNPs at SFS edges
    if 'mu_sfs' in statistics:
        is_edge = ((dac_gpu == 1) | (dac_gpu == n_hap - 1)).astype(cp.float64)
        edge_sum = _scatter_sum(is_edge[in_range], bin_idx[in_range], n_windows)
        total_count = _bin_counts(bin_idx[in_range], n_windows)
        edge_cpu = edge_sum.get()
        count_cpu = total_count.get()
        mu_sfs = np.where(count_cpu > 0, edge_cpu / count_cpu, np.nan)
        results['mu_sfs'] = mu_sfs

    # Per-window pairwise stats (LD, distance moments)
    ld_pairwise = {'zns', 'omega', 'mu_ld'}
    dist_pairwise = {'dist_var', 'dist_skew', 'dist_kurt'}
    perwin_stats = {s for s in (ld_pairwise | dist_pairwise) if s in statistics}

    if perwin_stats:
        from . import ld_statistics
        from . import distance_stats
        ws_np, we_np = win_start.get(), win_stop.get()

        stat_arrays = {s: np.full(n_windows, np.nan) for s in perwin_stats}
        need_dist = bool(perwin_stats & dist_pairwise)
        need_winmat = ('omega' in stat_arrays or 'mu_ld' in stat_arrays
                       or need_dist)

        # Precompute for fused ZnS path
        use_proj = (missing_data == 'project')
        if 'zns' in stat_arrays:
            hap = matrix.haplotypes
            hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)
            valid_mask = (hap >= 0).astype(cp.float64)

        for wi in range(n_windows):
            s, e = int(ws_np[wi]), int(we_np[wi])
            if e - s < 4:
                continue

            if 'zns' in stat_arrays:
                stat_arrays['zns'][wi] = ld_statistics._zns_from_precomputed(
                    hap_clean, valid_mask, s, e,
                    use_projection=use_proj)

            if need_winmat:
                win_mat = HaplotypeMatrix(matrix.haplotypes[:, s:e],
                                           matrix.positions[s:e])
                if 'omega' in stat_arrays:
                    stat_arrays['omega'][wi] = ld_statistics.omega(
                        win_mat, missing_data=missing_data)
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


def windowed_statistics_fused_chunked(haplotype_matrix: HaplotypeMatrix,
                                      bp_bins,
                                      statistics=('pi', 'theta_w', 'tajimas_d'),
                                      population=None,
                                      pop1=None,
                                      pop2=None,
                                      per_base: bool = True,
                                      is_accessible=None,
                                      _win_starts=None,
                                      _win_stops=None,
                                      missing_data='include'):
    """Chunked fused windowed statistics for data too large for a single pass.

    Same interface and results as windowed_statistics_fused(), but splits the
    variant axis into memory-safe chunks. Each chunk is transposed and fed to
    the existing fused CUDA kernels. Partial results are accumulated across
    chunks (all kernel outputs are additive sums, except max_daf which uses
    element-wise max).
    """
    from ._utils import get_population_matrix
    from ._memutil import estimate_fused_chunk_size, free_gpu_pool

    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap_raw = matrix.haplotypes
    n_hap = hap_raw.shape[0]
    n_total_var = hap_raw.shape[1]

    positions = matrix.positions
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions)

    # Window ranges (same setup as non-chunked)
    if _win_starts is not None and _win_stops is not None:
        ws_gpu = cp.asarray(_win_starts, dtype=cp.float64)
        we_gpu = cp.asarray(_win_stops, dtype=cp.float64)
        n_windows = len(ws_gpu)
        win_start = cp.searchsorted(positions, ws_gpu, side='left').astype(cp.int64)
        win_stop = cp.searchsorted(positions, we_gpu, side='left').astype(cp.int64)
    else:
        bp_bins_gpu = cp.asarray(bp_bins, dtype=cp.float64)
        n_windows = len(bp_bins_gpu) - 1
        win_start, win_stop = _compute_window_ranges(positions, bp_bins_gpu)
        ws_gpu = bp_bins_gpu[:-1]
        we_gpu = bp_bins_gpu[1:]

    results = {
        'window_start': ws_gpu.get() if hasattr(ws_gpu, 'get') else ws_gpu,
        'window_stop': we_gpu.get() if hasattr(we_gpu, 'get') else we_gpu,
    }

    if per_base:
        window_bases = _compute_window_bases(
            haplotype_matrix, results['window_start'],
            results['window_stop'], is_accessible)

    # Determine chunk size
    chunk_size = estimate_fused_chunk_size(n_hap)
    # Ensure chunk is at least large enough for the largest window
    max_win_variants = int((win_stop - win_start).max().get()) if n_windows > 0 else 0
    chunk_size = max(chunk_size, max_win_variants + 1)

    # ── Single-pop stats via chunked fused kernel ────────────────────────
    single_pop_stats = {'pi', 'theta_w', 'tajimas_d', 'segregating_sites',
                        'singletons', 'theta_h', 'fay_wu_h', 'max_daf'}
    single_pop_requested = any(s in statistics for s in single_pop_stats)
    if single_pop_requested:
        # Accumulators
        acc_mpd = cp.zeros(n_windows, dtype=cp.float64)
        acc_seg = cp.zeros(n_windows, dtype=cp.float64)
        acc_sing = cp.zeros(n_windows, dtype=cp.float64)
        acc_count = cp.zeros(n_windows, dtype=cp.float64)
        acc_theta_h = cp.zeros(n_windows, dtype=cp.float64)
        acc_max_daf = cp.zeros(n_windows, dtype=cp.float64)

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
            hap_chunk_t = cp.ascontiguousarray(
                hap_raw[:, c_start:c_end].T.astype(cp.int8))
            n_chunk_var = c_end - c_start

            # Per-chunk outputs
            out_mpd = cp.zeros(n_overlap, dtype=cp.float64)
            out_seg = cp.zeros(n_overlap, dtype=cp.float64)
            out_sing = cp.zeros(n_overlap, dtype=cp.float64)
            out_count = cp.zeros(n_overlap, dtype=cp.float64)
            out_theta_h = cp.zeros(n_overlap, dtype=cp.float64)
            out_max_daf = cp.zeros(n_overlap, dtype=cp.float64)

            _fused_windowed_kernel_v2(
                (int(n_overlap),), (256,),
                (hap_chunk_t, clipped_start, clipped_stop,
                 np.int32(n_hap), np.int32(n_chunk_var), np.int32(n_overlap),
                 out_mpd, out_seg, out_sing, out_count, out_theta_h,
                 out_max_daf))

            # Accumulate (all additive except max_daf)
            cp.add.at(acc_mpd, w_idx, out_mpd)
            cp.add.at(acc_seg, w_idx, out_seg)
            cp.add.at(acc_sing, w_idx, out_sing)
            cp.add.at(acc_count, w_idx, out_count)
            cp.add.at(acc_theta_h, w_idx, out_theta_h)
            acc_max_daf[w_idx] = cp.maximum(acc_max_daf[w_idx], out_max_daf)

            del hap_chunk_t
            free_gpu_pool()

        # Post-processing (identical to non-chunked)
        mpd_sum = acc_mpd.get()
        seg_count = acc_seg.get()
        sing_count = acc_sing.get()
        var_count = acc_count.get()
        theta_h_sum = acc_theta_h.get()
        max_daf_arr = acc_max_daf.get()

        results['n_variants'] = var_count.astype(int)

        if 'pi' in statistics:
            if per_base:
                results['pi'] = np.where(window_bases > 0,
                                         mpd_sum / window_bases, np.nan)
            else:
                results['pi'] = mpd_sum

        if 'theta_w' in statistics:
            a1 = np.sum(1.0 / np.arange(1, n_hap))
            theta_abs = seg_count / a1
            if per_base:
                results['theta_w'] = np.where(window_bases > 0,
                                              theta_abs / window_bases, np.nan)
            else:
                results['theta_w'] = theta_abs

        if 'segregating_sites' in statistics:
            results['segregating_sites'] = seg_count.astype(int)

        if 'singletons' in statistics:
            results['singletons'] = sing_count.astype(int)

        if 'tajimas_d' in statistics:
            n = n_hap
            a1 = np.sum(1.0 / np.arange(1, n))
            a2 = np.sum(1.0 / np.arange(1, n) ** 2)
            b1 = (n + 1) / (3 * (n - 1))
            b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
            c1 = b1 - 1 / a1
            c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
            e1 = c1 / a1
            e2 = c2 / (a1 ** 2 + a2)

            S = seg_count
            d_num = mpd_sum - S / a1
            d_var = e1 * S + e2 * S * (S - 1)
            d_std = np.sqrt(np.maximum(d_var, 0))
            tajd = np.where(d_std > 0, d_num / d_std, np.nan)
            tajd[S < 3] = np.nan
            results['tajimas_d'] = tajd

        if 'theta_h' in statistics:
            if per_base:
                results['theta_h'] = np.where(window_bases > 0,
                                              theta_h_sum / window_bases, np.nan)
            else:
                results['theta_h'] = theta_h_sum

        if 'fay_wu_h' in statistics:
            results['fay_wu_h'] = mpd_sum - theta_h_sum

        if 'max_daf' in statistics:
            results['max_daf'] = max_daf_arr

    # ── Two-pop stats via chunked fused kernel ───────────────────────────
    two_pop_stats = {'fst', 'fst_hudson', 'fst_wc', 'dxy', 'da'}
    two_pop_requested = any(s in statistics for s in two_pop_stats)
    if two_pop_requested:
        if pop1 is None or pop2 is None:
            raise ValueError("pop1 and pop2 required for fst/dxy/da")

        # Get population haplotype indices for GPU slicing
        pop1_idx = matrix.sample_sets[pop1]
        pop2_idx = matrix.sample_sets[pop2]
        n1 = len(pop1_idx)
        n2 = len(pop2_idx)
        # Chunk size based on the larger population
        twopop_chunk = estimate_fused_chunk_size(max(n1, n2))
        twopop_chunk = max(twopop_chunk, max_win_variants + 1)

        acc_fst_num = cp.zeros(n_windows, dtype=cp.float64)
        acc_fst_den = cp.zeros(n_windows, dtype=cp.float64)
        acc_dxy = cp.zeros(n_windows, dtype=cp.float64)
        acc_pi1 = cp.zeros(n_windows, dtype=cp.float64)
        acc_pi2 = cp.zeros(n_windows, dtype=cp.float64)
        acc_wc_a = cp.zeros(n_windows, dtype=cp.float64)
        acc_wc_ab = cp.zeros(n_windows, dtype=cp.float64)

        for c_start in range(0, n_total_var, twopop_chunk):
            c_end = min(c_start + twopop_chunk, n_total_var)

            overlap = (win_start < c_end) & (win_stop > c_start)
            w_idx = cp.where(overlap)[0]
            if len(w_idx) == 0:
                continue

            clipped_start = cp.maximum(win_start[w_idx], c_start) - c_start
            clipped_stop = cp.minimum(win_stop[w_idx], c_end) - c_start
            n_overlap = len(w_idx)

            hap1_t = cp.ascontiguousarray(
                hap_raw[pop1_idx, c_start:c_end].T.astype(cp.int8))
            hap2_t = cp.ascontiguousarray(
                hap_raw[pop2_idx, c_start:c_end].T.astype(cp.int8))
            n_chunk_var = c_end - c_start

            out_fst_num = cp.zeros(n_overlap, dtype=cp.float64)
            out_fst_den = cp.zeros(n_overlap, dtype=cp.float64)
            out_dxy = cp.zeros(n_overlap, dtype=cp.float64)
            out_pi1 = cp.zeros(n_overlap, dtype=cp.float64)
            out_pi2 = cp.zeros(n_overlap, dtype=cp.float64)
            out_wc_a = cp.zeros(n_overlap, dtype=cp.float64)
            out_wc_ab = cp.zeros(n_overlap, dtype=cp.float64)

            _fused_windowed_twopop_kernel(
                (int(n_overlap),), (256,),
                (hap1_t, hap2_t, clipped_start, clipped_stop,
                 np.int32(n1), np.int32(n2),
                 np.int32(n_chunk_var), np.int32(n_overlap),
                 out_fst_num, out_fst_den, out_dxy, out_pi1, out_pi2,
                 out_wc_a, out_wc_ab))

            cp.add.at(acc_fst_num, w_idx, out_fst_num)
            cp.add.at(acc_fst_den, w_idx, out_fst_den)
            cp.add.at(acc_dxy, w_idx, out_dxy)
            cp.add.at(acc_pi1, w_idx, out_pi1)
            cp.add.at(acc_pi2, w_idx, out_pi2)
            cp.add.at(acc_wc_a, w_idx, out_wc_a)
            cp.add.at(acc_wc_ab, w_idx, out_wc_ab)

            del hap1_t, hap2_t
            free_gpu_pool()

        if 'n_variants' not in results:
            results['n_variants'] = (win_stop - win_start).get().astype(int)

        fst_num = acc_fst_num.get()
        fst_den = acc_fst_den.get()
        dxy_sum = acc_dxy.get()
        pi1_sum = acc_pi1.get()
        pi2_sum = acc_pi2.get()
        wc_a = acc_wc_a.get()
        wc_ab = acc_wc_ab.get()

        if 'fst' in statistics or 'fst_hudson' in statistics:
            hudson_fst = np.where(fst_den > 0, fst_num / fst_den, np.nan)
            if 'fst' in statistics:
                results['fst'] = hudson_fst
            if 'fst_hudson' in statistics:
                results['fst_hudson'] = hudson_fst

        if 'fst_wc' in statistics:
            results['fst_wc'] = np.where(wc_ab > 0, wc_a / wc_ab, np.nan)

        if 'dxy' in statistics:
            if per_base:
                results['dxy'] = np.where(window_bases > 0,
                                          dxy_sum / window_bases, np.nan)
            else:
                results['dxy'] = dxy_sum

        if 'da' in statistics:
            if per_base:
                da_sum = dxy_sum - (pi1_sum + pi2_sum) / 2.0
                results['da'] = np.where(window_bases > 0,
                                         da_sum / window_bases, np.nan)
            else:
                results['da'] = dxy_sum - (pi1_sum + pi2_sum) / 2.0

    # Delegate Garud H / scatter-add stats / per-window LD to the
    # non-chunked function (they already handle large data or operate
    # per-window).
    garud_stats = {'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2h1',
                   'haplotype_count'}
    scatter_stats = {'mean_nsl', 'daf_hist', 'mu_sfs', 'snp_dist_mean',
                     'snp_dist_var', 'snp_dist_min', 'snp_dist_max',
                     'mu_var', 'zns', 'omega', 'mu_ld', 'dist_var',
                     'dist_skew', 'dist_kurt'}
    remaining = set(statistics) & (garud_stats | scatter_stats)
    if remaining:
        # Call the non-chunked function for just these stats;
        # they don't need the full transposed matrix.
        extra = windowed_statistics_fused(
            haplotype_matrix, bp_bins=bp_bins,
            statistics=tuple(remaining),
            population=population, pop1=pop1, pop2=pop2,
            per_base=per_base, is_accessible=is_accessible,
            _win_starts=_win_starts, _win_stops=_win_stops,
            missing_data=missing_data)
        for k, v in extra.items():
            if k not in results:
                results[k] = v

    return results


def _windowed_mean(values, bin_idx, valid_mask, n_bins):
    """Compute mean of values per window bin, returning NaN for empty bins."""
    val_sum = _scatter_sum(values[valid_mask], bin_idx[valid_mask], n_bins)
    val_count = _bin_counts(bin_idx[valid_mask], n_bins)
    sum_cpu = val_sum.get()
    count_cpu = val_count.get()
    return np.where(count_cpu > 0, sum_cpu / count_cpu, np.nan)


def _compute_fused_garud_h(haplotype_matrix, population,
                            win_start, win_stop, n_windows, statistics,
                            results):
    """Compute windowed Garud's H using prefix-sum hashing + fused GPU kernel.

    For large data, processes windows in groups to avoid allocating
    full-size prefix-sum arrays (which would be n_hap * n_var * 8 bytes).
    """
    from ._utils import get_population_matrix
    from ._memutil import free_gpu_pool

    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_hap, n_var)
    n_hap, n_var = hap.shape

    # Memory check: prefix sums need 4 arrays of (n_hap, span+1) float64
    # Use 30% of free memory as budget for prefix-sum arrays
    free_mem = cp.cuda.Device().mem_info[0]
    prefix_budget = int(free_mem * 0.3)
    # Each variant column costs n_hap * 8 * 4 bytes (hw1, hw2, cs1, cs2)
    cost_per_var = n_hap * 8 * 4
    max_span = max(1, prefix_budget // cost_per_var)

    # Check if full prefix sums fit in memory
    if n_var <= max_span:
        # Original single-pass path
        _garud_h_single_pass(hap, n_hap, n_var, win_start, win_stop,
                             n_windows, statistics, results)
    else:
        # Chunked path: process groups of windows
        _garud_h_chunked(hap, n_hap, n_var, win_start, win_stop,
                         n_windows, max_span, statistics, results)


def _garud_h_single_pass(hap, n_hap, n_var, win_start, win_stop,
                          n_windows, statistics, results):
    """Garud H via full prefix-sum hashing (fits in memory)."""
    h_f64 = hap.astype(cp.float64)
    rng = cp.random.RandomState(seed=42)
    w1 = rng.standard_normal(n_var, dtype=cp.float64)
    w2 = rng.standard_normal(n_var, dtype=cp.float64)

    hw1 = h_f64 * w1[cp.newaxis, :]
    hw2 = h_f64 * w2[cp.newaxis, :]
    cs1 = cp.zeros((n_hap, n_var + 1), dtype=cp.float64)
    cs2 = cp.zeros((n_hap, n_var + 1), dtype=cp.float64)
    cp.cumsum(hw1, axis=1, out=cs1[:, 1:])
    cp.cumsum(hw2, axis=1, out=cs2[:, 1:])

    all_h1 = (cs1[:, win_stop] - cs1[:, win_start]).T
    all_h2 = (cs2[:, win_stop] - cs2[:, win_start]).T
    all_h1 = cp.ascontiguousarray(all_h1)
    all_h2 = cp.ascontiguousarray(all_h2)

    _launch_garud_kernel(all_h1, all_h2, n_hap, n_windows, statistics, results)


def _garud_h_chunked(hap, n_hap, n_var, win_start, win_stop,
                      n_windows, max_span, statistics, results):
    """Garud H processing windows in groups to limit memory."""
    from ._memutil import free_gpu_pool

    out_h1 = np.empty(n_windows, dtype=np.float64)
    out_h12 = np.empty(n_windows, dtype=np.float64)
    out_h123 = np.empty(n_windows, dtype=np.float64)
    out_h2h1 = np.empty(n_windows, dtype=np.float64)
    out_n_distinct = np.empty(n_windows, dtype=np.float64)

    ws_cpu = win_start.get()
    we_cpu = win_stop.get()

    # Group windows by overlapping variant spans
    processed = np.zeros(n_windows, dtype=bool)
    wi = 0
    while wi < n_windows:
        # Find a group of consecutive windows that fit in max_span
        group_var_start = int(ws_cpu[wi])
        group_var_end = int(we_cpu[wi])
        group_end = wi + 1
        while group_end < n_windows:
            candidate_end = int(we_cpu[group_end])
            if candidate_end - group_var_start > max_span:
                break
            group_var_end = candidate_end
            group_end += 1

        span = group_var_end - group_var_start
        n_group = group_end - wi

        # Compute prefix sums over just this variant span
        hap_span = hap[:, group_var_start:group_var_end].astype(cp.float64)
        rng = cp.random.RandomState(seed=42)
        w1 = rng.standard_normal(span, dtype=cp.float64)
        w2 = rng.standard_normal(span, dtype=cp.float64)

        hw1 = hap_span * w1[cp.newaxis, :]
        hw2 = hap_span * w2[cp.newaxis, :]
        cs1 = cp.zeros((n_hap, span + 1), dtype=cp.float64)
        cs2 = cp.zeros((n_hap, span + 1), dtype=cp.float64)
        cp.cumsum(hw1, axis=1, out=cs1[:, 1:])
        cp.cumsum(hw2, axis=1, out=cs2[:, 1:])

        # Local window indices relative to span start
        local_ws = cp.asarray(ws_cpu[wi:group_end] - group_var_start)
        local_we = cp.asarray(we_cpu[wi:group_end] - group_var_start)

        all_h1 = (cs1[:, local_we] - cs1[:, local_ws]).T
        all_h2 = (cs2[:, local_we] - cs2[:, local_ws]).T
        all_h1 = cp.ascontiguousarray(all_h1)
        all_h2 = cp.ascontiguousarray(all_h2)

        # Launch kernel for this group
        grp_results = {}
        _launch_garud_kernel(all_h1, all_h2, n_hap, n_group,
                             statistics, grp_results)

        # Store group results
        for stat_name, out_arr in [('garud_h1', out_h1), ('garud_h12', out_h12),
                                    ('garud_h123', out_h123), ('garud_h2h1', out_h2h1),
                                    ('haplotype_count', out_n_distinct)]:
            if stat_name in grp_results:
                out_arr[wi:group_end] = grp_results[stat_name]

        del hap_span, hw1, hw2, cs1, cs2, all_h1, all_h2
        free_gpu_pool()
        wi = group_end

    if 'garud_h1' in statistics:
        results['garud_h1'] = out_h1
    if 'garud_h12' in statistics:
        results['garud_h12'] = out_h12
    if 'garud_h123' in statistics:
        results['garud_h123'] = out_h123
    if 'garud_h2h1' in statistics:
        results['garud_h2h1'] = out_h2h1
    if 'haplotype_count' in statistics:
        results['haplotype_count'] = out_n_distinct.astype(int)


def _launch_garud_kernel(all_h1, all_h2, n_hap, n_windows, statistics, results):
    """Launch the Garud H GPU kernel and store results."""
    out_h1 = cp.empty(n_windows, dtype=cp.float64)
    out_h12 = cp.empty(n_windows, dtype=cp.float64)
    out_h123 = cp.empty(n_windows, dtype=cp.float64)
    out_h2h1 = cp.empty(n_windows, dtype=cp.float64)
    out_n_distinct = cp.empty(n_windows, dtype=cp.float64)

    block = max(1, (n_hap + 1) // 2)
    block = 1 << (block - 1).bit_length()
    block = min(block, 1024)
    shm_size = 2 * n_hap * 8

    _fused_garud_h_kernel(
        (n_windows,), (block,),
        (all_h1, all_h2, np.int32(n_hap), np.int32(n_windows),
         out_h1, out_h12, out_h123, out_h2h1, out_n_distinct),
        shared_mem=shm_size)

    if 'garud_h1' in statistics:
        results['garud_h1'] = out_h1.get()
    if 'garud_h12' in statistics:
        results['garud_h12'] = out_h12.get()
    if 'garud_h123' in statistics:
        results['garud_h123'] = out_h123.get()
    if 'garud_h2h1' in statistics:
        results['garud_h2h1'] = out_h2h1.get()
    if 'haplotype_count' in statistics:
        results['haplotype_count'] = out_n_distinct.get().astype(int)


# ---------------------------------------------------------------------------
# GPU-native windowed statistics: compute once, bin everywhere
# ---------------------------------------------------------------------------

def _scatter_sum(values, bin_idx, n_bins):
    """Sum values into bins using scatter_add on GPU."""
    result = cp.zeros(n_bins, dtype=cp.float64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    cp.add.at(result, bin_idx[valid], values[valid])
    return result


def _bin_counts(bin_idx, n_bins):
    """Count variants per bin on GPU."""
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    return cp.bincount(bin_idx[valid], minlength=n_bins)


def _allele_sum_and_n(hap, has_missing=None):
    """Sum of alleles and valid count per variant, skipping missing (-1).

    Parameters
    ----------
    hap : cupy.ndarray
    has_missing : bool, optional
        If known, skip the check. If None, checks for negatives.

    Returns
    -------
    dac : cupy.ndarray  — derived allele count per variant
    n_valid : cupy.ndarray or int — valid haplotypes per variant
    """
    if has_missing is None:
        has_missing = bool(int(cp.min(hap)) < 0)
    if has_missing:
        valid_mask = hap >= 0
        dac = cp.sum(cp.where(valid_mask, hap, 0), axis=0)
        n_valid = cp.sum(valid_mask, axis=0)
    else:
        dac = cp.sum(hap, axis=0)
        n_valid = hap.shape[0]
    return dac, n_valid


def _per_variant_mpd(hap, n_hap):
    """Mean pairwise difference per variant (GPU)."""
    dac, n_valid = _allele_sum_and_n(hap)
    dac = dac.astype(cp.float64)
    if isinstance(n_valid, int):
        n = cp.float64(n_valid)
    else:
        n = n_valid.astype(cp.float64)
    usable = n > 1
    p = cp.where(usable, dac / n, 0.0)
    mpd = cp.zeros_like(dac)
    mpd[usable] = 2.0 * p[usable] * (1.0 - p[usable]) * n[usable] / (n[usable] - 1)
    return mpd


def _per_variant_is_seg(hap, n_hap_int):
    """Boolean: is variant segregating (GPU)."""
    dac, n_valid = _allele_sum_and_n(hap)
    return (dac > 0) & (dac < n_valid)


def _per_variant_is_singleton(hap, n_hap_int):
    """Boolean: is variant a singleton (GPU)."""
    dac, n_valid = _allele_sum_and_n(hap)
    return (dac == 1) | (dac == n_valid - 1)


def _per_variant_fst_hudson_components(hap1, hap2, n1, n2):
    """Per-variant Hudson FST numerator and denominator (GPU).

    Returns (num, den) as CuPy arrays. Handles missing data (-1) by
    using per-site valid counts.
    """
    valid1 = (hap1 >= 0).astype(cp.float64)
    valid2 = (hap2 >= 0).astype(cp.float64)
    ac1_1 = cp.sum(cp.where(hap1 >= 0, hap1, 0), axis=0).astype(cp.float64)
    n1_v = cp.sum(valid1, axis=0).astype(cp.float64)
    ac1_0 = n1_v - ac1_1
    ac2_1 = cp.sum(cp.where(hap2 >= 0, hap2, 0), axis=0).astype(cp.float64)
    n2_v = cp.sum(valid2, axis=0).astype(cp.float64)
    ac2_0 = n2_v - ac2_1

    n1_pairs = n1_v * (n1_v - 1) / 2
    n1_same = (ac1_0 * (ac1_0 - 1) + ac1_1 * (ac1_1 - 1)) / 2
    mpd1 = cp.where(n1_pairs > 0, (n1_pairs - n1_same) / n1_pairs, 0.0)

    n2_pairs = n2_v * (n2_v - 1) / 2
    n2_same = (ac2_0 * (ac2_0 - 1) + ac2_1 * (ac2_1 - 1)) / 2
    mpd2 = cp.where(n2_pairs > 0, (n2_pairs - n2_same) / n2_pairs, 0.0)

    within = (mpd1 + mpd2) / 2.0

    n_between = n1_v * n2_v
    n_between_same = ac1_0 * ac2_0 + ac1_1 * ac2_1
    between = cp.where(n_between > 0,
                       (n_between - n_between_same) / n_between, 0.0)

    return between - within, between


def _per_variant_dxy(hap1, hap2, n1, n2):
    """Per-variant mean pairwise difference between populations (GPU)."""
    valid1 = (hap1 >= 0).astype(cp.float64)
    valid2 = (hap2 >= 0).astype(cp.float64)
    ac1_1 = cp.sum(cp.where(hap1 >= 0, hap1, 0), axis=0).astype(cp.float64)
    n1_v = cp.sum(valid1, axis=0).astype(cp.float64)
    ac1_0 = n1_v - ac1_1
    ac2_1 = cp.sum(cp.where(hap2 >= 0, hap2, 0), axis=0).astype(cp.float64)
    n2_v = cp.sum(valid2, axis=0).astype(cp.float64)
    ac2_0 = n2_v - ac2_1

    n_pairs = n1_v * n2_v
    n_same = ac1_0 * ac2_0 + ac1_1 * ac2_1
    return cp.where(n_pairs > 0, (n_pairs - n_same) / n_pairs, 0.0)


def windowed_statistics(haplotype_matrix: HaplotypeMatrix,
                        bp_bins,
                        statistics=('pi', 'theta_w', 'tajimas_d'),
                        population=None,
                        pop1=None,
                        pop2=None,
                        per_base: bool = True,
                        is_accessible=None):
    """GPU-native windowed statistics with no Python loop over windows.

    Computes per-variant values once, then aggregates into windows using
    GPU scatter_add operations. Dramatically faster than per-window
    computation for large numbers of windows.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    bp_bins : array_like
        Window edges in base pairs. N+1 edges define N windows.
    statistics : tuple of str
        Statistics to compute. Supported:
        Single-population: 'pi', 'theta_w', 'tajimas_d', 'segregating_sites',
        'singletons', 'het_expected'
        Two-population: 'fst', 'dxy'
    population : str or list, optional
        Population for single-pop statistics.
    pop1, pop2 : str or list, optional
        Populations for two-pop statistics (fst, dxy).
    per_base : bool
        If True, normalize by window size in base pairs.
    is_accessible : array_like, optional
        Boolean accessibility mask for per-base normalization.

    Returns
    -------
    dict
        Maps statistic names to numpy arrays of shape (n_windows,).
        Also includes 'n_variants' (count per window) and 'window_start',
        'window_stop' arrays.
    """
    from ._utils import get_population_matrix

    if haplotype_matrix.device == 'CPU':
        haplotype_matrix.transfer_to_gpu()

    # get population subset for single-pop stats
    if population is not None:
        matrix = get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes  # (n_haplotypes, n_variants)
    n_hap_int = hap.shape[0]
    n_hap = cp.float64(n_hap_int)

    positions = matrix.positions
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions)

    # assign variants to windows (GPU parallel binary search)
    bp_bins = cp.asarray(bp_bins, dtype=cp.float64)
    n_windows = len(bp_bins) - 1
    bin_idx = cp.searchsorted(bp_bins, positions, side='right').astype(cp.int64) - 1
    # clamp to valid range
    bin_idx = cp.clip(bin_idx, 0, n_windows - 1)
    # mark out-of-range variants
    out_of_range = (positions < bp_bins[0]) | (positions >= bp_bins[-1])
    bin_idx[out_of_range] = -1

    variant_counts = _bin_counts(bin_idx, n_windows)

    results = {
        'window_start': bp_bins[:-1].get(),
        'window_stop': bp_bins[1:].get(),
        'n_variants': variant_counts.get().astype(int),
    }

    # window sizes for per-base normalization
    if per_base:
        window_bases = _compute_window_bases(
            haplotype_matrix, results['window_start'],
            results['window_stop'], is_accessible)
        window_bases = cp.asarray(window_bases, dtype=cp.float64)

    # Phase 1: compute per-variant values and aggregate

    # check for missing data once, share across all stats
    _has_missing = bool(int(cp.min(hap)) < 0)

    # precompute allele counts once (shared across all stats)
    dac, n_valid = _allele_sum_and_n(hap, has_missing=_has_missing)
    dac = dac.astype(cp.float64)
    if isinstance(n_valid, int):
        n_v = cp.float64(n_valid)
        usable = cp.ones(dac.shape, dtype=bool)
    else:
        n_v = n_valid.astype(cp.float64)
        usable = n_v > 1
    p = cp.where(usable, dac / n_v, 0.0)
    need_mpd = any(s in statistics for s in ('pi', 'tajimas_d'))
    need_seg = any(s in statistics for s in
                   ('theta_w', 'tajimas_d', 'segregating_sites'))

    if need_mpd:
        mpd = cp.zeros_like(dac)
        mpd[usable] = 2.0 * p[usable] * (1.0 - p[usable]) * n_v[usable] / (n_v[usable] - 1) if not isinstance(n_valid, int) else 2.0 * p[usable] * (1.0 - p[usable]) * n_hap / (n_hap - 1)
    else:
        mpd = None
    is_seg = (dac > 0) & (dac < n_v) if need_seg else None

    if 'pi' in statistics:
        pi_sum = _scatter_sum(mpd, bin_idx, n_windows)
        if per_base:
            results['pi'] = cp.where(window_bases > 0,
                                     pi_sum / window_bases, cp.nan).get()
        else:
            results['pi'] = pi_sum.get()

    if 'theta_w' in statistics:
        seg_counts = _scatter_sum(is_seg.astype(cp.float64), bin_idx, n_windows)
        n = n_hap_int
        a1 = np.sum(1.0 / np.arange(1, n))
        theta_abs = seg_counts / a1
        if per_base:
            results['theta_w'] = cp.where(window_bases > 0,
                                          theta_abs / window_bases, cp.nan).get()
        else:
            results['theta_w'] = theta_abs.get()

    if 'segregating_sites' in statistics:
        seg_vals = is_seg if is_seg is not None else (dac > 0) & (dac < n_v)
        seg_counts_out = _scatter_sum(seg_vals.astype(cp.float64), bin_idx,
                                      n_windows)
        results['segregating_sites'] = seg_counts_out.get().astype(int)

    if 'singletons' in statistics:
        is_sing = (dac == 1) | (dac == n_v - 1)
        sing_counts = _scatter_sum(is_sing.astype(cp.float64), bin_idx,
                                   n_windows)
        results['singletons'] = sing_counts.get().astype(int)

    if 'het_expected' in statistics:
        he = 2.0 * p * (1.0 - p)
        he_sum = _scatter_sum(he, bin_idx, n_windows)
        results['het_expected'] = cp.where(
            variant_counts > 0,
            he_sum / variant_counts.astype(cp.float64), cp.nan).get()

    if 'tajimas_d' in statistics:
        # aggregate mpd and seg counts into windows, then apply formula
        pi_sum_td = _scatter_sum(mpd, bin_idx, n_windows) if 'pi' not in statistics else _scatter_sum(mpd, bin_idx, n_windows)
        seg_counts_td = _scatter_sum(is_seg.astype(cp.float64), bin_idx,
                                     n_windows)

        n = n_hap_int
        a1 = np.sum(1.0 / np.arange(1, n))
        a2 = np.sum(1.0 / np.arange(1, n) ** 2)
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - 1 / a1
        c2 = b2 - (n + 2) / (a1 * n) + a2 / a1 ** 2
        e1 = c1 / a1
        e2 = c2 / (a1 ** 2 + a2)

        S = seg_counts_td.get()
        pi_w = pi_sum_td.get()

        d_num = pi_w - S / a1
        d_var = e1 * S + e2 * S * (S - 1)
        d_std = np.sqrt(np.maximum(d_var, 0))

        tajd = np.where(d_std > 0, d_num / d_std, np.nan)
        tajd[S < 3] = np.nan
        results['tajimas_d'] = tajd

    # two-population statistics
    two_pop_stats = [s for s in statistics if s in ('fst', 'dxy')]
    if two_pop_stats:
        if pop1 is None or pop2 is None:
            raise ValueError("pop1 and pop2 required for fst/dxy")

        # Use filtered matrix so inaccessible variants are excluded
        m1 = get_population_matrix(matrix, pop1)
        m2 = get_population_matrix(matrix, pop2)
        if m1.device == 'CPU':
            m1.transfer_to_gpu()
        if m2.device == 'CPU':
            m2.transfer_to_gpu()

        hap1 = m1.haplotypes
        hap2 = m2.haplotypes
        n1 = cp.float64(hap1.shape[0])
        n2 = cp.float64(hap2.shape[0])

        if 'fst' in statistics:
            fst_num, fst_den = _per_variant_fst_hudson_components(
                hap1, hap2, n1, n2)
            num_sum = _scatter_sum(fst_num, bin_idx, n_windows)
            den_sum = _scatter_sum(fst_den, bin_idx, n_windows)
            results['fst'] = cp.where(den_sum > 0,
                                      num_sum / den_sum, cp.nan).get()

        if 'dxy' in statistics:
            dxy_vals = _per_variant_dxy(hap1, hap2, n1, n2)
            dxy_sum = _scatter_sum(dxy_vals, bin_idx, n_windows)
            if per_base:
                results['dxy'] = cp.where(window_bases > 0,
                                          dxy_sum / window_bases, cp.nan).get()
            else:
                results['dxy'] = dxy_sum.get()

    return results
