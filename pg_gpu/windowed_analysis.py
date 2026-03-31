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
        'ld_decay': lambda w, **kwargs: _compute_ld_decay(w.matrix, **kwargs),
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
        # Create statistics dictionaries with current parameters
        self.SINGLE_POP_STATS = {
            'pi': lambda w: diversity.pi(w.matrix, span_normalize=True, 
                                       missing_data=self.missing_data, 
                                       span_denominator=self.span_denominator),
            'theta_w': lambda w: diversity.theta_w(w.matrix, span_normalize=True,
                                                 missing_data=self.missing_data,
                                                 span_denominator=self.span_denominator), 
            'tajimas_d': lambda w: diversity.tajimas_d(w.matrix, missing_data=self.missing_data),
            'n_variants': lambda w: w.n_variants,
            'n_singletons': lambda w: diversity.singleton_count(w.matrix, missing_data=self.missing_data),
            'segregating_sites': lambda w: diversity.segregating_sites(w.matrix, missing_data=self.missing_data),
        }
        
        self.TWO_POP_STATS = {
            'dxy': lambda w, p1, p2: divergence.dxy(w.matrix, p1, p2, 
                                                  missing_data=self.missing_data,
                                                  span_denominator=self.span_denominator == 'total'),
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
                    results[f"{stat}_{pop}"] = self.SINGLE_POP_STATS[stat](
                        WindowData(window.chrom, window.start, window.end, 
                                 window.center, pop_matrix, pop_matrix.num_variants,
                                 window.window_id)
                    )
            else:
                # Compute for all samples
                results[stat] = self.SINGLE_POP_STATS[stat](window)
        
        # Two population statistics
        if len(self.populations) >= 2:
            for stat in self.two_pop_stats:
                for i, pop1 in enumerate(self.populations):
                    for pop2 in self.populations[i+1:]:
                        key = f"{stat}_{pop1}_{pop2}"
                        results[key] = self.TWO_POP_STATS[stat](window, pop1, pop2)
        
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
            sample_sets={'all': list(range(len(pop_indices)))}
        )


class WindowIterator:
    """Iterates over genomic windows."""
    
    def __init__(self, haplotype_matrix: HaplotypeMatrix, 
                 window_params: WindowParams):
        self.matrix = haplotype_matrix
        self.params = window_params
        self.positions = haplotype_matrix.positions
        
        # Get positions as numpy array for easier manipulation
        if isinstance(self.positions, cp.ndarray):
            self.positions_np = self.positions.get()
        else:
            self.positions_np = self.positions
            
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
            'ignore' - Treat missing as reference allele (original behavior)
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
        'ignore' - Treat missing as reference allele (original behavior)
    span_denominator : str
        'total' - Use total genomic span (chrom_end - chrom_start)
        'sites' - Use number of sites analyzed
        'callable' - Use span from first to last site included in analysis
    **kwargs
        Additional arguments passed to WindowedAnalyzer
        
    Returns
    -------
    pd.DataFrame
        Windowed statistics results
    """
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



def _compute_ld_decay(matrix: HaplotypeMatrix, bins: List[int], 
                     max_distance: int = 50000) -> float:
    """Compute mean LD decay (mean r² across all distance bins)."""
    # Simplified implementation - returns single summary value
    # In practice, might want to return the full decay curve
    r2_matrix = matrix.pairwise_r2()
    positions = matrix.positions
    
    # Calculate pairwise distances
    if matrix.device == 'GPU':
        pos_i, pos_j = cp.meshgrid(positions, positions, indexing='ij')
        distances = cp.abs(pos_j - pos_i)
        
        # Get mean r² within max distance
        mask = (distances > 0) & (distances <= max_distance)
        if cp.any(mask):
            mean_r2 = float(cp.mean(r2_matrix[mask]).get())
        else:
            mean_r2 = np.nan
    else:
        # CPU version
        pos_i, pos_j = np.meshgrid(positions, positions, indexing='ij')
        distances = np.abs(pos_j - pos_i)
        
        mask = (distances > 0) & (distances <= max_distance)
        if np.any(mask):
            mean_r2 = float(np.mean(r2_matrix[mask]))
        else:
            mean_r2 = np.nan
    
    return mean_r2


def _compute_mean_r2(matrix: HaplotypeMatrix, max_distance: int) -> float:
    """Compute mean r² within a distance."""
    r2_matrix = matrix.pairwise_r2()
    positions = matrix.positions
    
    if matrix.device == 'GPU':
        pos_i, pos_j = cp.meshgrid(positions, positions, indexing='ij')
        distances = cp.abs(pos_j - pos_i)
        mask = (distances > 0) & (distances <= max_distance)
        return float(cp.mean(r2_matrix[mask]).get()) if cp.any(mask) else np.nan
    else:
        pos_i, pos_j = np.meshgrid(positions, positions, indexing='ij')
        distances = np.abs(pos_j - pos_i)
        mask = (distances > 0) & (distances <= max_distance)
        return float(np.mean(r2_matrix[mask])) if np.any(mask) else np.nan


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
                             double* out_var_count) {
    // hap_t layout: (n_total_var, n_hap) - transposed for coalesced access
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
        }
        return;
    }

    double dn = (double)n_hap;
    double thread_mpd = 0.0;
    double thread_seg = 0.0;
    double thread_sing = 0.0;
    double thread_count = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;

        // hap_t[v * n_hap + h] -- consecutive h values are contiguous in memory
        int dac = 0;
        const signed char* row = hap_t + v * n_hap;
        for (int h = 0; h < n_hap; h++) {
            if (row[h] > 0) dac++;
        }

        double p = (double)dac / dn;
        double mpd = 2.0 * p * (1.0 - p) * dn / (dn - 1.0);
        thread_mpd += mpd;
        thread_seg += (dac > 0 && dac < n_hap) ? 1.0 : 0.0;
        thread_sing += (dac == 1 || dac == n_hap - 1) ? 1.0 : 0.0;
        thread_count += 1.0;
    }

    __shared__ double smem[4 * 256];
    int tid = threadIdx.x;
    smem[tid] = thread_mpd;
    smem[256 + tid] = thread_seg;
    smem[512 + tid] = thread_sing;
    smem[768 + tid] = thread_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[256 + tid] += smem[256 + tid + s];
            smem[512 + tid] += smem[512 + tid + s];
            smem[768 + tid] += smem[768 + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_mpd_sum[wid] = smem[0];
        out_seg_count[wid] = smem[256];
        out_sing_count[wid] = smem[512];
        out_var_count[wid] = smem[768];
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
                           double* out_dxy_sum) {
    // hap1_t, hap2_t: transposed layout (n_total_var, n_hap) for coalesced reads
    int wid = blockIdx.x;
    if (wid >= n_windows) return;

    int v_start = (int)win_start[wid];
    int v_stop = (int)win_stop[wid];
    int n_vars = v_stop - v_start;
    if (n_vars <= 0) {
        if (threadIdx.x == 0) {
            out_fst_num[wid] = 0.0;
            out_fst_den[wid] = 0.0;
            out_dxy_sum[wid] = 0.0;
        }
        return;
    }

    double dn1 = (double)n_hap1;
    double dn2 = (double)n_hap2;
    double t_fst_num = 0.0;
    double t_fst_den = 0.0;
    double t_dxy = 0.0;

    for (int vi = threadIdx.x; vi < n_vars; vi += blockDim.x) {
        int v = v_start + vi;

        int ac1_1 = 0, ac2_1 = 0;
        const signed char* row1 = hap1_t + v * n_hap1;
        for (int h = 0; h < n_hap1; h++) {
            if (row1[h] > 0) ac1_1++;
        }
        const signed char* row2 = hap2_t + v * n_hap2;
        for (int h = 0; h < n_hap2; h++) {
            if (row2[h] > 0) ac2_1++;
        }

        double ac1_0 = dn1 - ac1_1;
        double ac2_0 = dn2 - ac2_1;
        double d_ac1_1 = (double)ac1_1;
        double d_ac2_1 = (double)ac2_1;

        // Within-pop mean pairwise difference
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
    }

    // Block reduction
    __shared__ double smem[3 * 256];
    int tid = threadIdx.x;
    smem[tid] = t_fst_num;
    smem[256 + tid] = t_fst_den;
    smem[512 + tid] = t_dxy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[256 + tid] += smem[256 + tid + s];
            smem[512 + tid] += smem[512 + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_fst_num[wid] = smem[0];
        out_fst_den[wid] = smem[256];
        out_dxy_sum[wid] = smem[512];
    }
}
''', 'fused_windowed_twopop')


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
                              is_accessible=None):
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

    bp_bins_gpu = cp.asarray(bp_bins, dtype=cp.float64)
    n_windows = len(bp_bins_gpu) - 1

    # map windows to variant index ranges
    win_start, win_stop = _compute_window_ranges(positions, bp_bins_gpu)

    results = {
        'window_start': bp_bins_gpu[:-1].get(),
        'window_stop': bp_bins_gpu[1:].get(),
    }

    # window sizes for normalization
    if per_base:
        if is_accessible is not None:
            is_accessible = np.asarray(is_accessible, dtype=bool)
            bp_edges = bp_bins_gpu.get()
            window_bases = np.array([
                np.count_nonzero(is_accessible[max(0, int(bp_edges[i])):int(bp_edges[i+1])])
                for i in range(n_windows)
            ], dtype=np.float64)
        else:
            window_bases = np.diff(bp_bins_gpu.get())

    # single-pop stats via fused kernel
    single_pop_requested = any(s in statistics for s in
                               ('pi', 'theta_w', 'tajimas_d',
                                'segregating_sites', 'singletons'))
    if single_pop_requested:
        out_mpd = cp.zeros(n_windows, dtype=cp.float64)
        out_seg = cp.zeros(n_windows, dtype=cp.float64)
        out_sing = cp.zeros(n_windows, dtype=cp.float64)
        out_count = cp.zeros(n_windows, dtype=cp.float64)

        block = 256
        grid = n_windows

        _fused_windowed_kernel_v2(
            (grid,), (block,),
            (hap, win_start, win_stop,
             np.int32(n_hap), np.int32(n_total_var), np.int32(n_windows),
             out_mpd, out_seg, out_sing, out_count))

        mpd_sum = out_mpd.get()
        seg_count = out_seg.get()
        sing_count = out_sing.get()
        var_count = out_count.get()

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

    # two-pop stats via fused kernel
    two_pop_requested = any(s in statistics for s in ('fst', 'dxy'))
    if two_pop_requested:
        if pop1 is None or pop2 is None:
            raise ValueError("pop1 and pop2 required for fst/dxy")

        m1 = get_population_matrix(haplotype_matrix, pop1)
        m2 = get_population_matrix(haplotype_matrix, pop2)
        if m1.device == 'CPU':
            m1.transfer_to_gpu()
        if m2.device == 'CPU':
            m2.transfer_to_gpu()

        n1 = m1.haplotypes.shape[0]
        n2 = m2.haplotypes.shape[0]
        # transpose for coalesced kernel access
        hap1 = cp.ascontiguousarray(m1.haplotypes.T.astype(cp.int8))
        hap2 = cp.ascontiguousarray(m2.haplotypes.T.astype(cp.int8))

        out_fst_num = cp.zeros(n_windows, dtype=cp.float64)
        out_fst_den = cp.zeros(n_windows, dtype=cp.float64)
        out_dxy = cp.zeros(n_windows, dtype=cp.float64)

        block = 256
        grid = n_windows

        _fused_windowed_twopop_kernel(
            (grid,), (block,),
            (hap1, hap2, win_start, win_stop,
             np.int32(n1), np.int32(n2),
             np.int32(n_total_var), np.int32(n_windows),
             out_fst_num, out_fst_den, out_dxy))

        fst_num = out_fst_num.get()
        fst_den = out_fst_den.get()
        dxy_sum = out_dxy.get()

        if 'n_variants' not in results:
            results['n_variants'] = (win_stop - win_start).get().astype(int)

        if 'fst' in statistics:
            results['fst'] = np.where(fst_den > 0, fst_num / fst_den, np.nan)

        if 'dxy' in statistics:
            if per_base:
                results['dxy'] = np.where(window_bases > 0,
                                          dxy_sum / window_bases, np.nan)
            else:
                results['dxy'] = dxy_sum

    return results


# ---------------------------------------------------------------------------
# GPU-native windowed statistics: compute once, bin everywhere
# ---------------------------------------------------------------------------

def _scatter_sum(values, bin_idx, n_bins):
    """Sum values into bins using scatter_add on GPU."""
    import cupyx
    result = cp.zeros(n_bins, dtype=cp.float64)
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    cupyx.scatter_add(result, bin_idx[valid], values[valid])
    return result


def _bin_counts(bin_idx, n_bins):
    """Count variants per bin on GPU."""
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    return cp.bincount(bin_idx[valid], minlength=n_bins)


def _allele_sum(hap, has_missing=None):
    """Sum of alleles per variant, clamping missing (-1) to 0.

    Parameters
    ----------
    hap : cupy.ndarray
    has_missing : bool, optional
        If known, skip the check. If None, checks for negatives.
    """
    if has_missing is None:
        has_missing = bool(int(cp.min(hap)) < 0)
    if has_missing:
        hap = cp.maximum(hap, 0)
    return cp.sum(hap, axis=0)


def _per_variant_mpd(hap, n_hap):
    """Mean pairwise difference per variant (GPU)."""
    dac = _allele_sum(hap).astype(cp.float64)
    p = dac / n_hap
    return 2.0 * p * (1.0 - p) * n_hap / (n_hap - 1)


def _per_variant_is_seg(hap, n_hap_int):
    """Boolean: is variant segregating (GPU)."""
    dac = _allele_sum(hap)
    return (dac > 0) & (dac < n_hap_int)


def _per_variant_is_singleton(hap, n_hap_int):
    """Boolean: is variant a singleton (GPU)."""
    dac = _allele_sum(hap)
    return (dac == 1) | (dac == n_hap_int - 1)


def _per_variant_fst_hudson_components(hap1, hap2, n1, n2):
    """Per-variant Hudson FST numerator and denominator (GPU).

    Returns (num, den) as CuPy arrays.
    """
    ac1_1 = cp.sum(hap1, axis=0).astype(cp.float64)
    ac1_0 = n1 - ac1_1
    ac2_1 = cp.sum(hap2, axis=0).astype(cp.float64)
    ac2_0 = n2 - ac2_1

    n1_pairs = n1 * (n1 - 1) / 2
    n1_same = (ac1_0 * (ac1_0 - 1) + ac1_1 * (ac1_1 - 1)) / 2
    mpd1 = cp.where(n1_pairs > 0, (n1_pairs - n1_same) / n1_pairs, 0.0)

    n2_pairs = n2 * (n2 - 1) / 2
    n2_same = (ac2_0 * (ac2_0 - 1) + ac2_1 * (ac2_1 - 1)) / 2
    mpd2 = cp.where(n2_pairs > 0, (n2_pairs - n2_same) / n2_pairs, 0.0)

    within = (mpd1 + mpd2) / 2.0

    n_between = n1 * n2
    n_between_same = ac1_0 * ac2_0 + ac1_1 * ac2_1
    between = cp.where(n_between > 0,
                       (n_between - n_between_same) / n_between, 0.0)

    return between - within, between


def _per_variant_dxy(hap1, hap2, n1, n2):
    """Per-variant mean pairwise difference between populations (GPU)."""
    ac1_1 = cp.sum(hap1, axis=0).astype(cp.float64)
    ac1_0 = n1 - ac1_1
    ac2_1 = cp.sum(hap2, axis=0).astype(cp.float64)
    ac2_0 = n2 - ac2_1

    n_pairs = n1 * n2
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
        if is_accessible is not None:
            is_accessible = np.asarray(is_accessible, dtype=bool)
            window_bases = np.zeros(n_windows)
            bp_edges = bp_bins.get()
            for i in range(n_windows):
                start = int(bp_edges[i])
                stop = int(bp_edges[i + 1])
                window_bases[i] = np.count_nonzero(
                    is_accessible[max(0, start):stop])
            window_bases = cp.asarray(window_bases, dtype=cp.float64)
        else:
            window_bases = cp.diff(bp_bins)

    # Phase 1: compute per-variant values and aggregate

    # check for missing data once, share across all stats
    _has_missing = bool(int(cp.min(hap)) < 0)

    # precompute allele counts once (shared across all stats)
    dac = _allele_sum(hap, has_missing=_has_missing).astype(cp.float64)
    p = dac / n_hap
    need_mpd = any(s in statistics for s in ('pi', 'tajimas_d'))
    need_seg = any(s in statistics for s in
                   ('theta_w', 'tajimas_d', 'segregating_sites'))

    mpd = 2.0 * p * (1.0 - p) * n_hap / (n_hap - 1) if need_mpd else None
    is_seg = (dac > 0) & (dac < n_hap_int) if need_seg else None

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
        seg_vals = is_seg if is_seg is not None else (dac > 0) & (dac < n_hap_int)
        seg_counts_out = _scatter_sum(seg_vals.astype(cp.float64), bin_idx,
                                      n_windows)
        results['segregating_sites'] = seg_counts_out.get().astype(int)

    if 'singletons' in statistics:
        is_sing = (dac == 1) | (dac == n_hap_int - 1)
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

        m1 = get_population_matrix(haplotype_matrix, pop1)
        m2 = get_population_matrix(haplotype_matrix, pop2)
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