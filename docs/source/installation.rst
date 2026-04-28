Installation
============

For a high-level overview of what pg_gpu is and what it offers, see
:doc:`introduction`.

Requirements
------------

* A CUDA 12+ capable NVIDIA GPU
* `pixi <https://pixi.sh>`_ for environment management

Everything else (Python 3.12, CuPy, NumPy, SciPy, the matching CUDA
toolchain) is pinned and installed by ``pixi`` from ``pixi.lock``. We
require pixi -- not out of caprice, but because building CuPy / CUDA
extensions reproducibly is otherwise painful: pixi pulls a portable
NVIDIA toolchain into the project and removes the usual
"works-on-my-machine" tax. If you have never used pixi before, the
`installation page <https://pixi.sh/latest/#installation>`_ is a one-liner.

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

Verify Installation
-------------------

``import pg_gpu`` performs a CUDA-availability check at import time
and raises a diagnostic error if CuPy is missing or no usable CUDA
device is detected. So the verification is just:

.. code-block:: bash

   pixi run python -c "import pg_gpu; print(pg_gpu.__version__)"

If you see the version printed, you are set up correctly. If the
import fails, the error message will name the underlying cause -- the
most common one being a mismatch between the system CUDA driver and
the toolkit pixi installed; ``nvidia-smi`` should show a driver
version that supports CUDA 12.

For docs builds, static analysis, type checking, or any other context
where you want to import ``pg_gpu`` without exercising the GPU, set
``PG_GPU_SKIP_CUDA_CHECK=1`` to bypass the check. (The
``READTHEDOCS=True`` environment variable is auto-detected.)

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
