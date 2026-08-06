Installation
============

For a high-level overview of what pg_gpu is and what it offers, see
:doc:`introduction`.

Requirements
------------

* A Linux x86_64 machine with a CUDA 12+ capable NVIDIA GPU
* `pixi <https://pixi.sh>`_ for the recommended environment (or ``pip``
  into an environment you already manage -- see
  `Installation into an Existing Environment`_)

Everything else (Python 3.12, CuPy, NumPy, SciPy, the matching CUDA
toolchain) is pinned and installed by ``pixi`` from ``pixi.lock``. We
recommend pixi -- not out of caprice, but because building CuPy / CUDA
extensions reproducibly is otherwise painful: pixi pulls a portable
NVIDIA toolchain into the project and removes the usual
"works-on-my-machine" tax. If you have never used pixi before, the
`installation page <https://pixi.sh/latest/#installation>`_ is a one-liner.
If you would rather not adopt pixi, pg_gpu is also a standard
pip-installable package; see `Installation into an Existing Environment`_.

Installation with Pixi
----------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/kr-colab/pg_gpu.git
   cd pg_gpu

   # Install and activate the environment
   pixi install
   pixi shell

The default environment includes CuPy with the CUDA toolkit, development
tools (pytest, ipython), and documentation tools (sphinx).

Running Your Code
~~~~~~~~~~~~~~~~~

There are three common ways to run code that uses pg_gpu:

.. code-block:: bash

   # 1. One-off script via pixi run (works from anywhere inside the repo)
   pixi run python my_script.py

   # 2. Drop into a shell with the environment already activated
   pixi shell
   python my_script.py        # now `python` is the pixi env's interpreter

   # 3. Notebooks: launch jupyter from the pixi environment
   pixi run jupyter lab

Scripts do not need to live at the repo root -- ``pixi run`` walks up to
find the nearest ``pixi.toml``, so any path inside a clone of pg_gpu
works. To use pg_gpu as a dependency in your *own* pixi project, add it
via ``pixi add --pypi pg_gpu`` from a release (or as a path/git
dependency for an in-tree clone).

Example-Specific Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few examples and integration tests pull in heavier optional
dependencies that we keep out of the default environment. The
``moments`` integration is the main case: pg_gpu accelerates the costly
LD-statistic calculations used by `moments
<https://moments.readthedocs.io/>`_, so we ship a separate pixi
environment that has both libraries installed:

.. code-block:: bash

   pixi install -e moments
   pixi run -e moments python my_script.py

See :doc:`tutorials/moments_integration` for the full
demographic-inference walk-through.

Installation into an Existing Environment
-----------------------------------------

If you already manage dependencies with conda, a virtualenv, or another
tool -- for example to call pg_gpu from a Snakemake rule or an existing
Jupyter kernel -- you can install it from PyPI with ``pip`` instead of
adopting pixi:

.. code-block:: bash

   python -m pip install pg_gpu

This pulls the full runtime stack declared in ``pyproject.toml``:
``cupy-cuda12x[ctk]``, ``kvikio`` / ``nvcomp``,
``bio2zarr``, and the usual scientific-Python libraries -- all from the
default PyPI index. The only system requirement is a Linux x86_64 machine
with an NVIDIA CUDA 12 driver; no separate conda or system-wide CUDA
toolkit is needed.

For development against a local checkout, use an editable install with the
``dev`` extra:

.. code-block:: bash

   git clone https://github.com/kr-colab/pg_gpu.git
   cd pg_gpu
   python -m pip install -e ".[dev]"

The optional extras mirror the pixi environments: ``docs`` for the
documentation toolchain and ``moments`` for the moments LD integration
(e.g. ``python -m pip install -e ".[dev,moments]"``).

Pixi remains the recommended, fully pinned environment (see above); the
pip path trades that reproducibility for fitting into an environment you
already control.

Running Tests
-------------

.. code-block:: bash

   pixi run test              # all tests (moments-dependent tests auto-skip)
   pixi run test-parallel     # parallel execution

Tests that validate against the ``moments`` library are skipped when
moments is not installed. To run the full suite including moments
validation:

.. code-block:: bash

   pixi run -e moments test   # includes moments LD validation tests
