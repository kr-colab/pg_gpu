# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import pg_gpu

# Project information
project = 'pg_gpu'
copyright = '2025, Andrew Kern'
author = 'Andrew Kern'
release = pg_gpu.__version__

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