Biobank-Scale Streaming from VCZ
================================

For stores too large to fit on the GPU eagerly -- biobank-scale VCZ
(bio2zarr) datasets with tens to hundreds of thousands of haplotypes
-- ``HaplotypeMatrix.from_zarr`` (and ``GenotypeMatrix.from_zarr``)
returns a *streaming matrix* that walks the chromosome chunk by
chunk. The streaming object plugs into every scalar / windowed /
pair-bin / pairwise kernel that already accepts an eager matrix; GPU
peak memory scales with one chunk, not the chromosome length.

Background
----------

The eager path materializes the full ``(n_haplotypes, n_variants)``
haplotype matrix on the GPU before any statistic runs. At biobank
sample counts the matrix alone is dozens of gigabytes per chromosome
-- before any working memory for the kernel itself -- and on a 100k-
diploid store an A100 80 GB will OOM at the load step. Streaming
solves this by:

* Opening the zarr store as a ``ZarrGenotypeSource`` and probing the
  on-disk chunking + codec.
* Returning a ``StreamingHaplotypeMatrix`` (or
  ``StreamingGenotypeMatrix``) instead of an eager
  ``HaplotypeMatrix``.
* Iterating that object yields per-chunk *eager* matrices on the
  GPU; the next chunk's host read overlaps the current chunk's
  compute through a producer thread.
* Per-chunk reducible statistics (windowed diversity / divergence /
  Garud H, SFS, joint SFS, moments-LD pair bins) are *dispatched
  through the existing function names* -- you do not write a chunk
  loop yourself.
* Cross-window or pairwise kernels (``pairwise_r2``,
  per-individual-pair distance, custom recipes that need every
  variant simultaneously) get a sub-region eager matrix via
  ``streaming_matrix.materialize(region=..., sample_subset=...)``
  and run on that.

The store layout that gets the on-GPU codec decode path is
*bio2zarr-style sample chunking* (small enough chunks on the sample
axis that each chunk fits in a kvikio + nvCOMP decompression buffer).
A store whose sample axis is one chunk falls back to the host fetcher
with a one-time ``BadlyChunkedWarning``; everything else still works.

When ``HaplotypeMatrix.from_zarr`` returns a streaming object
--------------------------------------------------------------

``streaming='auto'`` (the default) picks based on whether the eager
matrix fits in roughly half the free GPU memory. ``streaming='always'``
forces it; ``streaming='never'`` forces eager. The streaming object
is a drop-in for the eager class for every kernel below -- the
calling code is the same.

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, windowed_analysis, sfs
   from pg_gpu.relatedness import ibs, grm

   stream = HaplotypeMatrix.from_zarr(
       "biobank/chr15.vcz",
       streaming="always",
       chunk_bp=500_000,         # genomic chunk size (default auto)
       prefetch=1,               # read-ahead depth (0 disables)
       pop_file="biobank/chr15.pops.tsv",   # optional sample -> pop TSV
   )

   # Per-window diversity + divergence: one streaming pass per call.
   df = windowed_analysis(stream, window_size=100_000, step_size=100_000,
                          statistics=["pi", "theta_w", "tajimas_d",
                                       "fst", "dxy"],
                          populations=["AFR", "EUR"])

   # Marginal + joint SFS: per-chunk reduction.
   sfs_afr = sfs.sfs(stream, population="AFR")
   joint   = sfs.joint_sfs(stream, pop1=list(stream.sample_sets["AFR"][:200]),
                                   pop2=list(stream.sample_sets["EUR"][:200]))

   # moments-LD bin sums with tail-buffer cross-chunk pair handling.
   ld = stream.compute_ld_statistics_gpu_two_pops(
       bp_bins=[0, 1_000, 10_000, 100_000], pop1="AFR", pop2="EUR",
   )

   # Pairwise relatedness: variant axis streamed, (n_ind, n_ind)
   # accumulators on host so the output can exceed GPU memory.
   ibs_mat = ibs(stream)
   grm_mat = grm(stream)

The same code on an eager ``HaplotypeMatrix`` (small store) runs
unchanged -- the only difference is which class ``from_zarr`` returns.

Sub-region eager builds for pairwise kernels
--------------------------------------------

A pairwise-r² heatmap, a Garud's H hash table, or a per-individual-pair
distance over an entire chromosome would not fit eagerly at biobank
scale, but a single 1 Mb sub-region with a haplotype subsample does:

.. code-block:: python

   # Pull a 1 Mb region restricted to 5,000 haplotypes from one pop.
   region = (50_000_000, 51_000_000)
   subsample = list(stream.sample_sets["AFR"][:5_000])
   eager = stream.materialize(region=region, sample_subset=subsample)

   # Run any pairwise kernel on the eager view.
   r2 = eager.pairwise_r2()

``materialize(sample_subset=...)`` routes the read through
``slice_subsample_gpu`` when the streaming source uses the kvikio
backend -- the (variants × subsample × 2) call_genotype block is
decompressed directly on the GPU rather than going through zarr's
sync host-side ``oindex``. At biobank scale this drops a 5 Mb / 10 k-
haplotype probe from ~170 s to ~3 s.

Converting a scikit-allel store
-------------------------------

Empirical biobank data often arrives in scikit-allel zarr layout
(``calldata/GT``, ``variants/POS``, ``samples``) -- e.g. the
Ag1000G ``AgamP3.phased.zarr`` release. ``pg_gpu.zarr_io.allel_zarr_to_vcz``
streams the source in variant blocks (so the conversion does not
materialize the full matrix) and writes a VCZ store with
bio2zarr-style sample chunking the streaming reader can consume:

.. code-block:: python

   from pg_gpu.zarr_io import allel_zarr_to_vcz

   allel_zarr_to_vcz(
       "AgamP3.phased.zarr",          # scikit-allel layout
       "AgamP3.phased.3R.vcz",        # output VCZ
       contig="3R",                    # required for grouped allel stores
       region="3R:1_000_000-2_000_000",  # optional sub-range
       variant_chunk=10_000, sample_chunk=1_000,
   )

The resulting store is what ``HaplotypeMatrix.from_zarr`` opens for
streaming.

Storage and host RAM footprints
-------------------------------

* A 100 k-diploid / 11.6 M-variant chr15 store at zstd-compressed
  bio2zarr chunking lives on disk at ~8 GB; the same chromosome
  uncompressed is ~2.3 TB. Streaming reads the compressed bytes from
  disk and decompresses one chunk at a time into ~12 GB GPU memory.
* The producer thread holds one extra decompressed chunk in host RAM
  (~12 GB for the parameters above), so peak host = ``(prefetch + 1)
  * chunk_bytes``. ``prefetch=0`` disables the read-ahead and reads
  serially.
* The ``(n_ind, n_ind)`` outputs of ``ibs`` and ``grm`` live on host
  as numpy arrays. At 50 k diploids that is ~20 GB of host RAM for
  ``grm``; ``ibs`` carries three such accumulators (~60 GB). The
  ``block_size=`` kwarg controls how many rows of the output are
  resident on the GPU at a time.

End-to-end paper example
------------------------

A complete biobank-scale scan -- per-window diversity + divergence
at three scales, marginal + joint SFS, Garud's H per pop, moments-LD
decay across probe regions, and a pairwise-r² heatmap of a 1 Mb
sub-region -- ships as
``06_simulated_genome_scan/scripts/genome_scan_ooa.py`` in the
companion paper-analysis repository. On chr15 from ``stdpopsim``
``OutOfAfrica_2T12`` simulated at 50 k diploids per population (200 k
haplotypes, 11.6 M variants, 7.9 GB on disk) the script completes
in ~16 minutes on a single A100 80 GB.

When streaming is not what you want
-----------------------------------

* The data already fits on the GPU eagerly. ``streaming='auto'``
  picks eager automatically; the eager path is faster per call
  because there is no per-chunk dispatch overhead.
* The kernel needs every variant simultaneously and the chromosome
  cannot be partitioned (e.g. a chromosome-wide pairwise r² heatmap
  on the full sample axis). Use ``materialize(region=...)`` to scope
  it to a tractable sub-region.
* The store is in scikit-allel layout, not VCZ. Convert it first
  with ``allel_zarr_to_vcz``; the streaming reader is VCZ-only.
