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