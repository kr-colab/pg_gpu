# pg_gpu - GPU-accelerated population genetics statistics

import os

if os.environ.get('READTHEDOCS') != 'True':
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise RuntimeError(
            "pg_gpu requires a CUDA-capable GPU and CuPy installed with "
            "working CUDA drivers. No usable GPU was detected."
        ) from e

from . import ld_statistics
from . import diversity
from . import divergence
from . import windowed_analysis
from . import selection
from . import sfs
from . import admixture
from . import decomposition
from . import plotting
from . import distance_stats
from . import relatedness
from .accessible import AccessibleMask, bed_to_mask, parse_bed
from .diversity import FrequencySpectrum
from .genotype_matrix import GenotypeMatrix
from .haplotype_matrix import HaplotypeMatrix
from .windowed_analysis import WindowedAnalyzer, windowed_analysis
from .decomposition import (
    LocalPCAResult,
    local_pca,
    local_pca_jackknife,
    pc_dist,
    corners,
)

__all__ = ['ld_statistics', 'diversity', 'divergence', 'windowed_analysis', 'selection', 'sfs', 'admixture', 'decomposition', 'plotting', 'distance_stats', 'HaplotypeMatrix', 'GenotypeMatrix', 'WindowedAnalyzer', 'windowed_analysis', 'AccessibleMask', 'bed_to_mask', 'parse_bed', 'LocalPCAResult', 'local_pca', 'local_pca_jackknife', 'pc_dist', 'corners']

__version__ = '0.1.0'
