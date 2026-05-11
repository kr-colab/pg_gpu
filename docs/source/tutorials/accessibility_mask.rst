Accessibility Masks and Windowed Diversity
==========================================

Packaged script: ``examples/accessibility_mask.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/accessibility_mask.py
   pixi run python examples/accessibility_mask.py --window 20_000

Background
----------

Per-base summary statistics like :math:`\pi` divide a sum of
pairwise differences (numerator) by a count of bases (denominator).
On real sequencing data, only a fraction of bases are reliably
*callable*: repetitive regions are masked, low-coverage stretches
fail filtering, ambiguous mapping yields gaps. If a windowed scan
divides by the *total* base count (callable + uncallable), the
denominator includes positions where variation could never be
observed and per-bp diversity looks artificially low in
low-callability regions. This is a frequent confounder for genome
scans -- selection signatures and accessibility deserts produce the
same downward dip.

An *accessibility mask* fixes this by restricting both the variants
visible to the analysis and the denominator of the per-bp
normalization to the callable subset. ``set_accessible_mask`` does
both in one call: it filters the matrix's variants to those falling
in accessible regions, and it tells span normalization to use the
accessible-base count (rather than the genomic span) as the
denominator. Both ``windowed_analysis`` and the scalar diversity /
divergence functions read this through automatically.

What the script does
--------------------

This tutorial sets up a controlled scenario where the "truth" is
known, then verifies that the mask recovers it:

1. Simulate a 1 Mb chromosome under ``msprime`` with a 200 kb central
   block of 100x lower mutation rate -- a stand-in for a low-callability
   region. The genealogy is identical inside and outside the block, so
   the per-site pattern of coalescence is the same; only the *probability of
   observing* a mutation differs. Outside the block, per-bp
   diversity is :math:`4 N_e \mu_{\text{high}}`.
2. Compute windowed :math:`\pi` twice from the same simulation:

   * once with no accessibility mask, and
   * once with the low-mutation block flagged inaccessible via an
     in-memory ``numpy`` boolean array passed to
     ``set_accessible_mask``.

3. Plot both traces on the same axes with the excluded region shaded.

Without the mask, :math:`\pi` drops misleadingly across the
low-callability block -- mutations are simply rarer, so :math:`\pi`/bp
looks lower even though the genealogy is unchanged. With the mask:

* Fully-inaccessible windows return ``NaN`` and render as visual
  gaps -- the scan is silent where it shouldn't speak rather than
  reporting a spurious value.
* Partially-inaccessible windows divide their numerator by the
  accessible-base count, so the flanking :math:`\pi` sits at the
  expected :math:`4 N_e \mu_{\text{high}}` value.

Why it's useful as a template
-----------------------------

This example is applicable to any region where the
*probability of observing variation* differs from the surrounding
genome (repeat masks, low-coverage regions, hard-to-map duplications,
centromeres) biases per-bp summary statistics unless the denominator
is corrected. ``set_accessible_mask`` accepts:

* A BED file path. The path is parsed, the BED 0-based half-open
  intervals are converted to a 1-based mask aligned to the matrix
  positions, and the BED's full extent (not just the matrix range)
  is honored.
* A NumPy boolean array, useful for simulated data or for masks
  built programmatically at runtime.
* A pre-built ``AccessibleMask`` instance, useful for sharing one
  mask across multiple matrices.

For the underlying machinery -- mask construction, span
normalization rules, and the ``n_callable_sites`` /
``n_segregating_sites`` / ``n_invariant_sites`` decomposition --
see :doc:`../missing_data`.
