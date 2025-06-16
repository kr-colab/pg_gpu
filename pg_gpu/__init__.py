# pg_gpu - GPU-accelerated population genetics statistics

from . import ld_statistics
from . import divergence
from .haplotype_matrix import HaplotypeMatrix
from .windowed_analysis import WindowedAnalyzer, windowed_analysis

__all__ = ['ld_statistics', 'divergence', 'HaplotypeMatrix', 'WindowedAnalyzer', 'windowed_analysis']

__version__ = '0.1.0'