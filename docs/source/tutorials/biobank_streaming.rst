Biobank-Scale Streaming from VCZ
================================

When a VCZ (bio2zarr) dataset has too many haplotypes to fit on the
GPU all at once -- tens to hundreds of thousands --
``HaplotypeMatrix.from_zarr`` (and ``GenotypeMatrix.from_zarr``)
gives you a *streaming matrix* that reads the chromosome one chunk
at a time. The streaming object works with every per-window, SFS,
LD, and pairwise relatedness function in pg_gpu with no change to
your code; GPU memory use scales with one chunk, not the whole
chromosome.

Background
----------

Normally pg_gpu loads the entire ``(n_haplotypes, n_variants)``
genotype matrix onto the GPU before any statistic runs. At biobank
sample sizes that matrix is dozens of gigabytes per chromosome --
and that is before pg_gpu allocates any working memory for the
statistic itself. On a store with 100,000 diploids an A100 80 GB
will run out of memory just trying to load the data.

Streaming avoids that by reading the chromosome in genomic chunks
(e.g. 500 kb at a time). Each chunk is decompressed onto the GPU,
the statistic is computed for that chunk, the result is added to a
running total, and the chunk is freed before the next one is read.
The next chunk is read on a background thread while the current
chunk is being processed, so disk I/O and GPU compute overlap.

Statistics that combine naturally across chunks --

* per-window diversity and divergence,
* the site frequency spectrum,
* the joint SFS,
* Garud's H,
* moments-LD pair bins (``DD``, ``Dz``, ``pi2``, derived
  :math:`\sigma_D^2`),
* identity by state (``ibs``) and the genetic relationship matrix
  (``grm``)

-- are handled automatically: you call the same function as on a
small in-memory matrix and pg_gpu takes care of the chunk-by-chunk
accumulation.

For statistics that need every variant in scope at the same time
(a pairwise-:math:`r^2` heatmap, a Garud's H hash table, a per-
individual-pair distance matrix), call
``streaming_matrix.materialize(region=..., sample_subset=...)`` to
read a smaller piece -- a sub-region of the chromosome, restricted
to a subset of haplotypes -- into a regular in-memory
``HaplotypeMatrix`` and run the statistic on that.

The streaming reader needs a VCZ store with bio2zarr-style chunking
on the sample axis (the standard layout produced by
``bio2zarr explode``). A store that puts the entire sample axis in
a single chunk still works, but falls back to a slower read path
with a one-time ``BadlyChunkedWarning``.

When you get a streaming object back
------------------------------------

``HaplotypeMatrix.from_zarr`` chooses between in-memory and
streaming based on the ``streaming`` argument:

* ``streaming='auto'`` (default): in-memory if the data fits in
  roughly half of the free GPU memory; streaming otherwise.
* ``streaming='always'``: always streaming.
* ``streaming='never'``: always in-memory (raises if the data does
  not fit).

Either way, the object you get back accepts the same pg_gpu
functions:

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

   # Per-window diversity + divergence, one streaming pass per call.
   df = windowed_analysis(stream, window_size=100_000, step_size=100_000,
                          statistics=["pi", "theta_w", "tajimas_d",
                                       "fst", "dxy"],
                          populations=["AFR", "EUR"])

   # Marginal and joint SFS.
   sfs_afr = sfs.sfs(stream, population="AFR")
   joint   = sfs.joint_sfs(stream, pop1=list(stream.sample_sets["AFR"][:200]),
                                   pop2=list(stream.sample_sets["EUR"][:200]))

   # moments-LD pair-bin statistics. Pairs that fall on opposite
   # sides of a chunk boundary are handled correctly via a tail
   # buffer that carries the last max(bp_bins) of variants forward.
   ld = stream.compute_ld_statistics_gpu_two_pops(
       bp_bins=[0, 1_000, 10_000, 100_000], pop1="AFR", pop2="EUR",
   )

   # Pairwise relatedness. The variant axis is streamed; the
   # (n_indiv, n_indiv) output lives in CPU memory so the result
   # itself can be larger than the GPU can hold.
   ibs_mat = ibs(stream)
   grm_mat = grm(stream)

The same code on a small in-memory ``HaplotypeMatrix`` runs
unchanged -- the only difference is which kind of object
``from_zarr`` returns.

Pulling a sub-region into memory for pairwise statistics
--------------------------------------------------------

Some statistics -- a pairwise-:math:`r^2` heatmap, Garud's H, per-
individual distance matrices -- need every variant present at the
same time. The full chromosome at biobank scale will not fit in
GPU memory for those, but a single 1 Mb sub-region with a
haplotype subsample easily will:

.. code-block:: python

   # Pull a 1 Mb region restricted to 5,000 haplotypes from one pop.
   region = (50_000_000, 51_000_000)
   subsample = list(stream.sample_sets["AFR"][:5_000])
   region_hm = stream.materialize(region=region, sample_subset=subsample)

   # Run any pairwise statistic on the regular in-memory matrix.
   r2 = region_hm.pairwise_r2()

``materialize(sample_subset=...)`` reads the (variants × subsample)
genotype block directly onto the GPU when the store is set up for
it (zstd / blosc / lz4 / deflate compressed bio2zarr chunks decoded
through kvikio + nvCOMP). At biobank scale this is roughly 60×
faster than the equivalent CPU-side path: a 5 Mb / 10,000-haplotype
block drops from ~170 seconds to ~3 seconds.

Converting a scikit-allel store
-------------------------------

Older biobank releases (e.g. the Ag1000G ``AgamP3.phased.zarr``)
ship in scikit-allel zarr layout (``calldata/GT``, ``variants/POS``,
``samples``) rather than VCZ. ``pg_gpu.zarr_io.allel_zarr_to_vcz``
reads the source in variant blocks -- so the conversion itself
does not need to hold the full matrix in memory -- and writes a
VCZ store the streaming reader can consume:

.. code-block:: python

   from pg_gpu.zarr_io import allel_zarr_to_vcz

   allel_zarr_to_vcz(
       "AgamP3.phased.zarr",                    # scikit-allel layout
       "AgamP3.phased.3R.vcz",                  # output VCZ
       contig="3R",                              # required for grouped allel stores
       region="3R:1_000_000-2_000_000",          # optional sub-range
       variant_chunk=10_000, sample_chunk=1_000,
   )

The resulting store is then opened the normal way through
``HaplotypeMatrix.from_zarr``.

Disk, CPU memory, and GPU memory
--------------------------------

* A 100,000-diploid / 11.6 M-variant chr15 store at zstd-compressed
  bio2zarr chunking takes about 8 GB on disk; the same chromosome
  uncompressed would be about 2.3 TB. Streaming reads the
  compressed bytes from disk and decompresses one chunk at a time
  into roughly 12 GB of GPU memory.
* The background read-ahead holds one extra decompressed chunk in
  CPU memory (~12 GB for the parameters above), so peak CPU memory
  use is ``(prefetch + 1) × chunk_size``. Set ``prefetch=0`` to
  read each chunk only when it is needed.
* The ``(n_indiv, n_indiv)`` output of ``ibs`` and ``grm`` lives in
  CPU memory as a numpy array. At 50,000 diploids that is about
  20 GB for ``grm``; ``ibs`` keeps three such matrices and needs
  ~60 GB. The ``block_size`` argument controls how many rows of
  the output sit on the GPU at any one time.

Worked example
--------------

A complete biobank-scale scan -- per-window diversity and
divergence at three window sizes (10 kb, 100 kb, 1 Mb), marginal
and joint SFS, Garud's H per population, moments-LD decay across
probe regions, and a pairwise-:math:`r^2` heatmap of a 1 Mb
sub-region -- ships as
``06_simulated_genome_scan/scripts/genome_scan_ooa.py`` in the
companion paper-analysis repository. On chr15 from ``stdpopsim``'s
``OutOfAfrica_2T12`` simulated at 50,000 diploids per population
(200,000 haplotypes, 11.6 M variants, 7.9 GB on disk) the script
completes in about 16 minutes on a single A100 80 GB.

When streaming is not the right choice
--------------------------------------

* Your data already fits on the GPU in one piece. ``streaming='auto'``
  will pick the in-memory path automatically; that path is faster
  per call because there is no per-chunk overhead.
* Your statistic needs every variant at once and the whole
  chromosome is too big to load. Use ``materialize(region=...)`` to
  read a smaller sub-region instead.
* Your store is in scikit-allel layout rather than VCZ. Convert it
  first with ``allel_zarr_to_vcz``; the streaming reader only
  accepts VCZ.
