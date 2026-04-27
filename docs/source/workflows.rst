Workflows
=========

The ``utils/`` directory ships standalone command-line scripts for
common end-to-end workflows. They are thin wrappers around the
library API, so reading their source is also a good way to see how the
pieces fit together for a real analysis.

Run any of them via ``pixi run`` to pick up the GPU-enabled
environment:

.. code-block:: bash

   pixi run python utils/<script>.py --help

vcf_to_zarr.py -- VCF to VCZ Conversion
---------------------------------------

Converts a bgzipped, indexed VCF (or BCF) into a `bio2zarr
<https://sgkit-dev.github.io/bio2zarr/>`_ VCZ store. Subsequent loads
via ``HaplotypeMatrix.from_zarr`` are dramatically faster than parsing
the VCF on every run, which is the usual reason to do this step once
up front.

.. code-block:: bash

   pixi run python utils/vcf_to_zarr.py input.vcf.gz output.zarr
   pixi run python utils/vcf_to_zarr.py input.vcf.gz output.zarr --workers 8

Arguments:

* ``vcf`` -- path to bgzipped/indexed VCF (``.vcf.gz``) or BCF.
* ``zarr`` -- output zarr store path.
* ``--workers N`` -- number of worker processes (default: all CPUs).

genome_scan.py -- End-to-End Genome Scan
----------------------------------------

A complete chromosome-scan workflow: load VCF or zarr data, optionally
assign populations, compute windowed diversity / divergence / Garud's H
plus scalar summaries on the GPU, and write a multi-panel scan figure
(PDF). Two populations trigger automatic computation of divergence
statistics (F\ :sub:`ST`, d\ :sub:`xy`) and the joint SFS; a single
population produces a diversity-only scan.

.. code-block:: bash

   # Single-population scan from a VCF, restricted to one chromosome
   pixi run python utils/genome_scan.py data.vcf.gz --region 3R

   # Two-population scan from a zarr store, with population assignments
   pixi run python utils/genome_scan.py data.zarr \
       --region 3R --pop-file pops.tsv

   # With an accessibility mask and custom windows
   pixi run python utils/genome_scan.py data.vcf.gz \
       --region 3R --pop-file pops.tsv \
       --accessible-bed mask.bed \
       --window-size 50000 --step-size 10000

The ``--pop-file`` argument expects a tab-delimited file with columns
``sample`` and ``pop`` -- one row per VCF sample.

Arguments:

* ``input`` -- VCF (``.vcf``, ``.vcf.gz``, ``.bcf``) or zarr store.
* ``--region`` -- chromosome or ``chrom:start-end``.
* ``--pop-file`` -- tab-delimited (sample, pop). Optional.
* ``--accessible-bed`` -- BED file of accessible regions. Optional.
* ``--window-size`` -- bp window size (default: 100,000).
* ``--step-size`` -- bp step between windows (default: 10,000).
* ``-o``, ``--output`` -- output figure path (default: ``genome_scan.pdf``).

Adding Your Own Workflow
------------------------

The scripts in ``utils/`` are deliberately short -- under ~250 lines
each -- and follow the same shape: parse arguments, build a
``HaplotypeMatrix``, optionally attach a mask and population
assignments, hand the matrix to ``windowed_analysis`` (or a scalar
statistics function), and write the result. Forking one of them is
usually the fastest way to assemble a custom scan.
