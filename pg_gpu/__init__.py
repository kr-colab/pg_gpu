# pg_gpu - GPU-accelerated population genetics statistics

import os

# Import-time CUDA check. Set PG_GPU_SKIP_CUDA_CHECK=1 to skip --
# useful for docs builds, static analysis, type checking, and other
# contexts that want to import pg_gpu without exercising the GPU. The
# Read-the-Docs builder is auto-detected via READTHEDOCS=True.
if not (os.environ.get('PG_GPU_SKIP_CUDA_CHECK')
        or os.environ.get('READTHEDOCS') == 'True'):
    try:
        import cupy as _cp  # noqa: F401  (verifying availability only)
    except ImportError as _e:
        raise ImportError(
            "pg_gpu requires CuPy, but `import cupy` failed:\n"
            f"  {_e}\n\n"
            "Install via the project's pixi environment (recommended):\n"
            "  git clone https://github.com/kr-colab/pg_gpu.git\n"
            "  cd pg_gpu && pixi install && pixi shell\n\n"
            "To import pg_gpu without CuPy (e.g. for docs builds or\n"
            "static analysis), set PG_GPU_SKIP_CUDA_CHECK=1."
        ) from _e
    try:
        if _cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA devices found")
    except Exception as _e:
        raise RuntimeError(
            "pg_gpu requires a CUDA-capable NVIDIA GPU. CuPy imported\n"
            f"successfully, but no usable CUDA device was found: {_e}\n\n"
            "Common causes:\n"
            "  - No NVIDIA driver installed; check `nvidia-smi`.\n"
            "  - CUDA toolkit version mismatch with the system driver.\n"
            "  - All GPUs are taken by another process (try setting\n"
            "    CUDA_VISIBLE_DEVICES to a free index).\n\n"
            "To import pg_gpu without a GPU (e.g. for docs builds or\n"
            "static analysis), set PG_GPU_SKIP_CUDA_CHECK=1."
        ) from _e

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
from . import resampling
from .accessible import AccessibleMask, bed_to_mask, parse_bed
from .diversity import FrequencySpectrum
from .genotype_matrix import GenotypeMatrix
from .haplotype_matrix import HaplotypeMatrix
from .windowed_analysis import WindowedAnalyzer, windowed_analysis
from .decomposition import (
    LocalPCAResult,
    LostructResult,
    local_pca,
    local_pca_jackknife,
    lostruct,
    pc_dist,
    corners,
)
from .resampling import block_jackknife, block_bootstrap
from ._memory_warning import MemoryLimitedWarning

__all__ = ['ld_statistics', 'diversity', 'divergence', 'windowed_analysis', 'selection', 'sfs', 'admixture', 'decomposition', 'plotting', 'distance_stats', 'resampling', 'HaplotypeMatrix', 'GenotypeMatrix', 'WindowedAnalyzer', 'windowed_analysis', 'AccessibleMask', 'bed_to_mask', 'parse_bed', 'LocalPCAResult', 'LostructResult', 'local_pca', 'local_pca_jackknife', 'lostruct', 'pc_dist', 'corners', 'block_jackknife', 'block_bootstrap', 'MemoryLimitedWarning']

__version__ = '0.1.0'
