# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Mock GPU dependencies for RTD builds (no GPU on build server)
autodoc_mock_imports = [
    'cupy', 'cupy.cuda', 'cupy._core',
    'cupy_backends', 'cupy_backends.cuda',
]

try:
    import pg_gpu
    release = pg_gpu.__version__
except Exception:
    release = '0.1.0'

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
]

# Theme
html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
}