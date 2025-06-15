Installation
============

Requirements
------------

* Python 3.8+
* CUDA-capable GPU
* CuPy (GPU computing)
* NumPy
* moments (population genetics)

Conda Installation
------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/your-org/pg_gpu.git
   cd pg_gpu
   
   # Create conda environment
   conda env create -f environment.yml
   conda activate pg_gpu
   
   # Install in development mode
   pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import pg_gpu
   print(pg_gpu.__version__)
   
   # Check GPU availability
   import cupy as cp
   print(f"GPU available: {cp.cuda.is_available()}")
   print(f"GPU device: {cp.cuda.Device().name}")