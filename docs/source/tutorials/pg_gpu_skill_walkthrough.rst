End-to-End Walkthrough: VCF → QC → π → PCA → Divergence → Selection → LD
=========================================================================

This tutorial is the canonical full-pipeline tour of pg_gpu. Each
section starts with a natural-language request a user might make of the
Claude Code skill (``.claude/SKILL.md``); the code beneath is what the
skill runs against pg_gpu's API. The example below was verified
end-to-end on the Ag1000G Ag3.0 X chromosome (1,470 *Anopheles gambiae*
diploids); every documented stat family runs without modification.

Background
----------

A full population-genomics analysis touches many sub-systems: bulk VCF
parsing, on-disk columnar storage, GPU-resident haplotype matrices,
streaming for regions larger than GPU memory, per-population
statistics, and figure generation. The Claude Code skill ties these
together in a single mental model so users can stay in
natural-language and pg_gpu reaches for the right API for each step.

This tutorial collapses the skill into a linear walk-through so the
behaviour is reproducible without an LLM in the loop.

Data
----

* **Source VCF**: ``ag1000g.X.vcf.gz`` (326 GB, 1,470 diploids, ~7.6 M
  variants on the X chromosome).
* **Accessibility mask**: ``agp3.is_accessible.txt.npz`` keyed
  per-chromosome (``access_X`` for this tutorial). ~67% of X is
  callable.
* **Metadata**: ``ag1kgp3.gamb.ck_short_sorted_rdl.csv`` (sample-level:
  ``country``, ``taxon``, ``rdl_gt`` etc.).

The first step converts the VCF to VCZ (one-time, ~hours on a fresh
VCF; ``vcf_to_zarr`` is idempotent). Every subsequent run loads from
the VCZ in seconds.

1. Load (VCZ if available, otherwise convert)
---------------------------------------------

``HaplotypeMatrix.vcf_to_zarr`` is a thin GPU-aware wrapper over
bio2zarr; ``from_zarr(streaming='auto')`` returns either an eager
``HaplotypeMatrix`` (if the materialized matrix fits in <50% of free
GPU memory) or a ``StreamingHaplotypeMatrix`` (otherwise — same API,
chunk-by-chunk execution).

.. code-block:: python

   import os, time
   from pg_gpu import HaplotypeMatrix

   VCF_X  = '/path/to/ag1000g.X.vcf.gz'
   ZARR_X = '/path/to/ag1000g.X.zarr'

   if not os.path.exists(ZARR_X):
       HaplotypeMatrix.vcf_to_zarr(VCF_X, ZARR_X, worker_processes=24)

   hm = HaplotypeMatrix.from_zarr(ZARR_X, region='X:1-10000000',
                                  streaming='auto')
   print(type(hm).__name__)   # 'StreamingHaplotypeMatrix' for 10 Mb

10 Mb of X with 1,470 diploids materializes to ~29 GB, which falls back
to streaming on any non-trivially busy GPU. The stat APIs below accept
either type.

2. QC field inspection — direct zarr (CPU-only, no memory cap)
--------------------------------------------------------------

``fields=`` on ``HaplotypeMatrix.from_zarr`` requires
``streaming='never'`` (a hard library constraint), which caps the
loadable region. For QC inspection over a large region, skip pg_gpu's
loader and read the QC arrays directly via ``zarr.open`` — they're
plain numpy datasets keyed ``variant_<TAG>`` / ``call_<TAG>``.

.. code-block:: python

   import zarr, numpy as np, pandas as pd

   z = zarr.open(ZARR_X, mode='r')
   # Find variant index range for the region.
   ci   = list(z['contig_id'][:]).index('X')
   mask = ((z['variant_contig'][:]   == ci) &
           (z['variant_position'][:] >= 1)  &
           (z['variant_position'][:] <= 10_000_000))
   i0, i1 = int(np.where(mask)[0].min()), int(np.where(mask)[0].max()) + 1

   pos     = z['variant_position'][i0:i1]
   mq      = z['variant_MQ'][i0:i1]
   quality = z['variant_quality'][i0:i1]      # NOTE: lowercase 'quality', not 'QUAL'
   dp      = z['call_DP'][i0:i1]              # (n_var, n_samples)

Reads 7.6 M variants × 6 QC columns in ~36 seconds, no GPU involved.
The per-site DataFrame feeds into bivariate hexbin matrices,
constant/all-NaN flagging, and histogram panels — all standard
numpy/pandas.

3. Windowed π in 10 kb windows, three missingness modes
-------------------------------------------------------

``windowed_analysis`` accepts a ``missing_data=`` mode:

* ``'include'`` — per-site ``n_valid``; divide by total window length.
* ``'exclude'`` — drop sites with any missing call.
* ``'pairwise'`` — pixy-style; count only called pairs per site,
  denominator is the total of these pair counts.

.. code-block:: python

   from pg_gpu import windowed_analysis

   results = {}
   for mode in ['include', 'exclude', 'pairwise']:
       results[mode] = windowed_analysis(
           hm, window_size=10_000, step_size=10_000,
           statistics=['pi'], missing_data=mode,
       )

In a dataset with ~75% per-site missingness (typical for Ag1000G),
``'pairwise'`` ≈ ``'include'`` to within a few percent and is the
strictly correct estimator under MCAR; ``'exclude'`` is brutal and can
return zero-site windows.

4. Accessibility-mask normalization
-----------------------------------

``AccessibleMask`` flips windowed π's span normalization from
"per window length" to "per callable base in window". On a 36%-callable
region the un-masked estimate underestimates per-callable-base π by
roughly 3×.

.. code-block:: python

   from pg_gpu.accessible import AccessibleMask
   import numpy as np

   m       = np.load('agp3.is_accessible.txt.npz')
   arr_X   = m['access_X']
   amask   = AccessibleMask(arr_X, offset=1)
   hm.set_accessible_mask(amask, chrom='X')

   df_masked = windowed_analysis(
       hm, window_size=10_000, step_size=10_000,
       statistics=['pi'], missing_data='pairwise',
   )

5. PCA with metadata coloring
-----------------------------

PCA needs unlinked biallelic SNPs with reasonable minor allele
frequency. Standard recipe:

.. code-block:: python

   from pg_gpu import decomposition
   import numpy as np, pandas as pd

   # Load eagerly so apply_biallelic_filter / locate_unlinked work.
   hm_pca = (HaplotypeMatrix
             .from_zarr(ZARR_X, region='X:1-2000000', streaming='never')
             .apply_biallelic_filter())

   # MAF > 0.05 (subsumes "segregating only").
   haps    = hm_pca.haplotypes
   import cupy as cp; xp = cp if type(haps).__module__.startswith('cupy') else np
   af      = xp.where((haps >= 0).sum(0) > 0,
                      (haps == 1).sum(0) / xp.maximum((haps >= 0).sum(0), 1),
                      0.0)
   maf     = xp.minimum(af, 1.0 - af)
   keep    = xp.where(maf > 0.05)[0]
   hm_pca  = hm_pca.get_subset(keep.get() if hasattr(keep, 'get') else keep)

   unlinked = hm_pca.locate_unlinked(size=100, step=20, threshold=0.1)
   hm_pca   = hm_pca.get_subset(np.where(
       unlinked.get() if hasattr(unlinked, 'get') else unlinked)[0])

   coords, explained = decomposition.randomized_pca(hm_pca, n_components=10)

Merge the per-haplotype coords down to per-sample by averaging
consecutive pairs (bio2zarr orders haplotypes as
``sample0_hap0, sample0_hap1, sample1_hap0, …``), then join to the
metadata DataFrame on ``sample_id``.

6. Per-population diversity, SFS, divergence
--------------------------------------------

Set up populations from the metadata, then run the per-pop stats in
single GPU passes.

.. code-block:: python

   from pg_gpu import diversity, divergence, sfs as pg_sfs

   # country -> haplotype indices (sample i -> [2i, 2i+1])
   meta = pd.read_csv('ag1kgp3.gamb.ck_short_sorted_rdl.csv')
   s2i  = {s: i for i, s in enumerate(sample_ids)}
   c2h  = {c: [j for s in g['sample_id'] if s in s2i for j in (2*s2i[s], 2*s2i[s]+1)]
           for c, g in meta.groupby('country')}
   top4 = sorted(c2h, key=lambda k: -len(c2h[k]))[:4]

   hm_stats = (HaplotypeMatrix.from_zarr(ZARR_X, region='X:1-2000000',
                                         streaming='never')
               .apply_biallelic_filter())
   hm_stats.sample_sets = {c: c2h[c] for c in top4}

   # Per-population: pi, theta_w, theta_h, theta_l, tajimas_d.
   per_pop = {p: diversity.diversity_stats(
                   hm_stats, population=p,
                   statistics=['pi','theta_w','theta_h','theta_l','tajimas_d'])
              for p in top4}

   # SFS (unfolded + folded) per population.
   sfs_un = {p: pg_sfs.sfs(hm_stats, population=p)        for p in top4}
   sfs_fo = {p: pg_sfs.sfs_folded(hm_stats, population=p) for p in top4}

   # Pairwise FST + dxy.
   from itertools import combinations
   for p1, p2 in combinations(top4, 2):
       fst = divergence.fst(hm_stats, p1, p2)   # Hudson FST by default
       dxy = divergence.dxy(hm_stats, p1, p2)

   # Windowed FST and dxy for the most-diverged pair.
   wdf = windowed_analysis(
       hm_stats, window_size=10_000, step_size=10_000,
       statistics=['fst','dxy'], populations=[p1, p2],
       missing_data='pairwise',
   )

7. Selection scans (iHS + Garud's H)
------------------------------------

.. code-block:: python

   from pg_gpu import selection

   ihs = selection.ihs(hm_stats)
   h1, h12, h123, h2_h1 = selection.garud_h(hm_stats)

Both expect phased biallelic data. ``hm_stats`` is already biallelic,
and the unphased zarr is loaded as 'phased' for these purposes — a
documented compromise for Ag1000G-style cohorts.

8. 2-locus LD stats (moments-LD)
--------------------------------

For demographic inference you want the moments-LD vectors per bp bin:
``D²`` (``DD``), ``Dz``, ``π_2``. ``pg_gpu.moments_ld.compute_ld_statistics``
is a GPU drop-in for ``moments.LD.Parsing.compute_ld_statistics``.

The cost is :math:`O(n_{\text{var}}^2)` per pop per bin — on a busy
GPU, restrict to a smaller region and two populations.

.. code-block:: python

   from pg_gpu import moments_ld
   import numpy as np

   hm_ld   = (HaplotypeMatrix
              .from_zarr(ZARR_X, region='X:1-500000', streaming='never')
              .apply_biallelic_filter())
   ld_pops = list(hm_stats.sample_sets)[:2]
   hm_ld.sample_sets = {p: hm_stats.sample_sets[p] for p in ld_pops}

   bp_bins  = np.logspace(2, 5, 10).astype(int)
   ld_stats = moments_ld.compute_ld_statistics(
       haplotype_matrix=hm_ld,
       pop_assignment={p: hm_ld.sample_sets[p] for p in ld_pops},
       pops=ld_pops, bp_bins=bp_bins, ac_filter=True, report=False,
   )

   # ld_stats['stats'] = [ld_names, het_names]
   # ld_stats['sums']  = [bin_array, ..., het_array]  (last has different shape)
   ld_names     = list(ld_stats['stats'][0])
   ld_arrays    = [np.asarray(a) for a in ld_stats['sums']
                   if np.asarray(a).ndim == 1 and len(np.asarray(a)) == len(ld_names)]
   sums_per_bin = np.stack(ld_arrays)        # (n_bins, n_stats)

Plot ``DD_i_i``, ``Dz_i_i_i``, ``pi2_i_i_i_i`` for each population's
within-pop curve. Pass the same ``ld_stats`` dict to a ``moments.LD``
demographic fitter.

9. Block-jackknife confidence interval
--------------------------------------

.. code-block:: python

   from pg_gpu import resampling, diversity

   pop = max(hm_stats.sample_sets, key=lambda k: len(hm_stats.sample_sets[k]))
   pi_est, pi_se, pi_ci = resampling.block_jackknife(
       hm_stats,
       statistic=lambda m: diversity.pi(m, population=pop),
       block_size=200_000,
   )

200 kb blocks are a reasonable default for taxa with ~kb-scale LD
decay; tune from the LD-decay panel in §8.

Reproduce
---------

Paste each section into a fresh Python session with pg_gpu installed
and a converted VCZ on disk. The whole sequence completes in ~30
minutes on a single A100 GPU.

The Claude Code skill that drives this workflow conversationally lives
at ``.claude/SKILL.md`` (added separately).
