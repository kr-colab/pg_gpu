# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Read the Docs and other CPU-only build environments can't install
# kvikio (which needs CUDA at build time), so ``import kvikio`` in
# ``pg_gpu.streaming_matrix`` fails before autodoc gets a chance to
# walk the module. Pre-inject a Mock under the same env-var gate that
# pg_gpu.__init__ already uses for its CuPy availability check.
if os.environ.get('READTHEDOCS') or os.environ.get('PG_GPU_SKIP_CUDA_CHECK'):
    from unittest.mock import MagicMock
    for _name in ('kvikio', 'kvikio.defaults', 'kvikio.zarr'):
        sys.modules.setdefault(_name, MagicMock())

import pg_gpu
release = pg_gpu.__version__

# Project information
project = 'pg_gpu'
copyright = '2025, Andrew Kern'
author = 'Andrew Kern'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'nbsphinx',
]

# The walkthrough needs a CUDA GPU and a multi-terabyte store that no build
# host has, so its committed outputs are the published ones. Without this,
# nbsphinx defaults to 'auto' and executes any notebook whose outputs are
# missing, turning a cleared cell into a failed docs build.
nbsphinx_execute = 'never'

# Theme
html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# Stand-ins for GPU-only deps that aren't installable on a CPU-only
# build host. The sys.modules pre-injection above covers the top-level
# ``import pg_gpu`` chain; this list covers any module autodoc imports
# afresh as it walks the API reference.
autodoc_mock_imports = ['kvikio', 'kvikio.defaults', 'kvikio.zarr']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
}

html_static_path = ['_static']
html_css_files = ['custom.css']
