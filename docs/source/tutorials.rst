Tutorials
=========

These tutorials are longer narrative walk-throughs, each tied to a
self-contained script under ``examples/``. They pair simulated data
(so you can reproduce the figures end-to-end without needing your own
VCF) with explanation of *why* the recipe is structured the way it
is, so they double as templates you can adapt to your own analyses.

For short, copy-paste-ready code snippets that show one feature at a
time, see :doc:`examples` instead.

.. toctree::
   :maxdepth: 1
   :caption: Available tutorials

   tutorials/sweep_tajimas_d_bootstrap
   tutorials/admixture_detection
   tutorials/accessibility_mask
   tutorials/local_pca
   tutorials/ld_blocks
   tutorials/moments_integration

Each tutorial follows the same structure:

* The packaged script and one-line invocations to run it.
* A short narrative of what the script demonstrates and why.
* For tutorials that have a meaningful Python API beyond a single
  function call (e.g. lostruct), an inline code block of the same
  pipeline outside the script.

For end-to-end command-line workflows on real data (VCF to zarr,
genome scans), see :doc:`workflows`.
