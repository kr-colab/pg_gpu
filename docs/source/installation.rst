Installation
============

Requirements
------------

* Python 3.12+
* CUDA 12+ capable GPU
* CuPy (GPU computing)
* NumPy, SciPy, pandas

Installation with Pixi
----------------------

pg_gpu uses `pixi <https://pixi.sh>`_ for environment management.

.. code-block:: bash

   # Clone repository
   git clone https://github.com/kr-colab/pg_gpu.git
   cd pg_gpu

   # Install and activate the environment
   pixi install
   pixi shell

The default environment includes CUDA/CuPy, development tools (pytest, ipython),
and documentation tools (sphinx).

Moments Integration
~~~~~~~~~~~~~~~~~~~

The `moments <https://moments.readthedocs.io/>`_ demographic inference library
is available in a separate pixi environment to keep the default install
lightweight:

.. code-block:: bash

   pixi install -e moments
   pixi run -e moments python my_script.py

See :doc:`moments_integration` for usage details.

Verify Installation
-------------------

.. code-block:: python

   import pg_gpu
   print(pg_gpu.__version__)

   # Check GPU availability
   import cupy as cp
   print(f"GPU available: {cp.cuda.is_available()}")
   print(f"GPU device: {cp.cuda.Device().name}")

Running Tests
-------------

.. code-block:: bash

   pixi run test              # all tests
   pixi run test-parallel     # parallel execution
