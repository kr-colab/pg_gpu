Biobank-Scale Streaming from VCZ
================================

TL;DR
-----

If your genotype matrix is too big to fit on the GPU (tens of
thousands of haplotypes upward), keep using ``pg_gpu`` the way you
always have -- just pass ``streaming='always'`` (or let the default
``streaming='auto'`` pick) to ``HaplotypeMatrix.from_zarr``. Every
per-window, per-site, LD, and pairwise relatedness function in the
library dispatches transparently on the streaming object, walking
the chromosome chunk by chunk on the GPU. GPU memory use is bounded
by one chunk's size (e.g. roughly 12 GB for chr15 at 100k diploids
and 500 kb chunks) regardless of how long the chromosome is.

The only constraint is that your data must be a VCZ store. If
you currently have a VCF or an older scikit-allel zarr, see
*Data format: VCZ vs VCF* and *Converting a scikit-allel store*
below for the one-time conversion.

Data format: VCZ vs VCF
-----------------------

The streaming path reads VCZ stores, not VCFs. A VCZ store is
bio2zarr's Zarr-on-disk encoding of a VCF: the genotype matrix
``call_genotype`` is laid out as small chunks of samples by
variants, each chunk compressed independently on disk. That layout
is what lets the reader pull a single chunk into GPU memory
without paying for the rest of the chromosome.

VCF text doesn't work for streaming -- the format isn't sliceable
chunk-wise and VCF parsing in htslib is single-threaded, so
loading a very large VCF takes hours and the parsed matrix still
has to fit in host memory.

Convert a VCF to VCZ once via the ``bio2zarr`` CLI (``vcf2zarr``):

.. code-block:: bash

   # One-time intermediate followed by the final VCZ encode.
   vcf2zarr explode biobank.vcf.gz biobank.icf
   vcf2zarr encode  biobank.icf     biobank.vcz

Both steps are parallelized; the encode step is what produces the
sample-by-variant chunking the streaming reader needs (see
*Required Zarr format* below).

How chunk-by-chunk accumulation works
-------------------------------------

For a streaming-friendly statistic -- per-window diversity, the
SFS, moments-LD bins, IBS, GRM -- ``pg_gpu``:

1. Reads one genomic chunk (default 500 kb) into GPU memory.
2. Runs the statistic on that chunk.
3. Adds the chunk's contribution to a running total kept either
   on the host or in a small on-GPU accumulator.
4. Frees the chunk and reads the next.
5. While the GPU is computing on chunk N, the next chunk is being
   read on a background thread (overlapped I/O).

The chromosome-wide answer is the running total at the end of the
walk. A concrete one-statistic example -- nucleotide diversity per
100 kb window on a chr15 store with 100,000 diploids:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, windowed_analysis

   stream = HaplotypeMatrix.from_zarr("biobank/chr15.vcz",
                                       streaming="always")
   pi = windowed_analysis(stream, window_size=100_000,
                          step_size=100_000, statistics=["pi"])
   # pi is a pandas.DataFrame, one row per 100 kb window across
   # chr15 (~1,000 rows). Peak GPU memory while this runs: ~12 GB
   # for the chunk that's currently on the device -- a single-shot
   # load of the same chromosome would have wanted ~2.3 TB.

Statistics that combine across chunks naturally:

* per-window diversity and divergence (``windowed_analysis``),
* the site frequency spectrum -- ``sfs.sfs``, ``sfs.joint_sfs``,
  ``sfs.sfs_folded``, ``sfs.joint_sfs_folded``,
* the projected joint SFS at sample sizes where the full histogram
  does not fit (``sfs.project_joint_sfs``),
* moments-LD pair bins (``DD``, ``Dz``, ``pi2``, derived
  :math:`\sigma_D^2`),
* identity by state (``relatedness.ibs``) and the genetic
  relationship matrix (``relatedness.grm``).

Statistics that need every variant present at the same time --
pairwise :math:`r^2` heatmaps, Garud's H hash tables -- cannot be
streamed end to end. The *Pulling a sub-region into memory*
section below shows how to pull a smaller piece into a regular
fully loaded matrix and run those.

streaming='auto' vs 'always' vs 'never'
---------------------------------------

``HaplotypeMatrix.from_zarr`` chooses between a single-shot load
and a streaming view based on the ``streaming`` argument:

* ``streaming='auto'`` (default): a single-shot load if the data
  fits in roughly half of free GPU memory; streaming otherwise.
* ``streaming='always'``: always streaming.
* ``streaming='never'``: always single-shot. Raises ``MemoryError``
  if the data does not fit.

Either way, the object you get back accepts the same ``pg_gpu``
functions:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix, windowed_analysis, sfs
   from pg_gpu.relatedness import ibs, grm

   stream = HaplotypeMatrix.from_zarr(
       "biobank/chr15.vcz",
       streaming="always",
       chunk_bp=500_000,                   # genomic chunk size
       pop_file="biobank/chr15.pops.tsv",  # optional sample -> pop
   )

   # Per-window diversity + divergence in a single streaming pass.
   df = windowed_analysis(stream, window_size=100_000, step_size=100_000,
                          statistics=["pi", "theta_w", "tajimas_d",
                                       "fst", "dxy"],
                          populations=["AFR", "EUR"])

   # Marginal SFS uses the full panel (1D output, fits anywhere).
   sfs_afr = sfs.sfs(stream, population="AFR")

   # Joint SFS: the full (n1+1, n2+1) histogram is 80 GB at 100k
   # haps per population, so use project_joint_sfs instead. It
   # applies a hypergeometric projection per variant and
   # accumulates straight into the small target grid -- every
   # variant from every haplotype contributes, no subsampling.
   joint_projected = sfs.project_joint_sfs(
       stream, pop1="AFR", pop2="EUR",
       target_n1=200, target_n2=200,
   )

   # moments-LD pair-bin statistics. Pairs that straddle a chunk
   # boundary are kept correct by a tail buffer that carries the
   # last max(bp_bins) of variants of one chunk forward and pairs
   # them with the start of the next.
   ld = stream.compute_ld_statistics_gpu_two_pops(
       bp_bins=[0, 1_000, 10_000, 100_000], pop1="AFR", pop2="EUR",
   )

   # Pairwise relatedness. The variant axis is streamed; the
   # (n_indiv, n_indiv) output lives in CPU memory so the result
   # itself can be larger than the GPU can hold.
   ibs_mat = ibs(stream)
   grm_mat = grm(stream)

The same code run on a fully loaded ``HaplotypeMatrix`` is
unchanged -- the only difference is which kind of object
``from_zarr`` returns.

The ``pop_file`` kwarg accepts a path to a TSV, a dict mapping
sample to population, a numpy array of labels (one per sample), or
the name of a zarr key in the store that holds a 1-D population
array. See ``HaplotypeMatrix.from_zarr`` for the full list.

Pulling a sub-region into memory for kernels that need every variant at once
----------------------------------------------------------------------------

A pairwise-:math:`r^2` heatmap or a Garud's H hash table needs
every variant present at the same time. The full chromosome at
biobank scale will not fit in GPU memory for those, but a single
1 Mb sub-region with a haplotype subsample easily will:

.. code-block:: python

   # Pull a 1 Mb region restricted to 5,000 haplotypes from one pop.
   region = (50_000_000, 51_000_000)
   subsample = list(stream.sample_sets["AFR"][:5_000])
   region_hm = stream.materialize(region=region, sample_subset=subsample)

   # Run any pairwise statistic on the fully loaded matrix.
   r2 = region_hm.pairwise_r2()

``materialize(sample_subset=...)`` reads the (variants × subsample)
genotype block directly onto the GPU when the store is set up for
it (zstd / blosc / lz4 / deflate compressed bio2zarr chunks decoded
through kvikio + nvCOMP). At biobank-scale sample sizes this is
roughly 60× faster than the CPU-side path: a 5 Mb / 10,000-haplotype
block drops from ~170 seconds to ~3 seconds.

Required Zarr format
--------------------

The streaming reader needs a VCZ store -- a bio2zarr-encoded zarr
group with ``call_genotype``, ``variant_position``, and
``sample_id``. The on-GPU codec decode (kvikio + nvCOMP) is enabled
when:

* The ``call_genotype`` codec is one of zstd, blosc, lz4, deflate.
  bio2zarr defaults to zstd, so this is usually free; other codecs
  surface as a ``ValueError`` at construction time so the caller
  can re-encode.
* The ``call_genotype`` chunks are not whole-sample-axis. A store
  written with ``sample_chunk`` equal to the full sample axis still
  works but warns once (``BadlyChunkedWarning``) and falls back to
  the host-buffer fetcher because the kvikio path gives no speedup
  at that chunking.

bio2zarr's defaults produce a store the streaming reader accepts;
the warnings only fire when somebody re-encoded a VCZ store with
non-standard arguments or converted from another zarr layout with
``sample_chunk`` too large.

Converting a scikit-allel store
-------------------------------

Older data releases (e.g. the Ag1000G ``AgamP3.phased.zarr``) ship
in scikit-allel zarr layout (``calldata/GT``, ``variants/POS``,
``samples``) rather than VCZ. ``pg_gpu.zarr_io.allel_zarr_to_vcz``
reads the source in variant blocks -- so the conversion itself
does not need to hold the full matrix in memory -- and writes a
VCZ store the streaming reader can consume:

.. code-block:: python

   from pg_gpu.zarr_io import allel_zarr_to_vcz

   allel_zarr_to_vcz(
       "AgamP3.phased.zarr",                # scikit-allel layout
       "AgamP3.phased.3R.vcz",              # output VCZ
       contig="3R",                         # required for grouped allel stores
       region="3R:1_000_000-2_000_000",     # optional sub-range
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
  use is ``(prefetch + 1) * chunk_size``. ``prefetch=1`` is the
  default (one chunk read in parallel with the previous chunk's
  GPU work); set ``prefetch=0`` to read each chunk only when
  needed.
* The ``(n_indiv, n_indiv)`` output of ``ibs`` and ``grm`` lives in
  CPU memory as a numpy array. At 50,000 diploids that is about
  20 GB for ``grm``; ``ibs`` keeps three such matrices and needs
  ~60 GB. The ``block_size`` argument controls how many rows of the
  output sit on the GPU at any one time.

Worked example
--------------

A worked end-to-end scan -- per-window diversity and divergence at
three window sizes (10 kb, 100 kb, 1 Mb), marginal SFS,
:math:`200 \times 200`-projected joint SFS, moments-LD decay across
probe regions, and a pairwise-:math:`r^2` heatmap of a 1 Mb
sub-region -- ships as ``examples/biobank_streaming_scan.py`` in
this repository. It uses the streaming reader for the
per-chunk-reducible kernels and ``materialize`` for the
pairwise-:math:`r^2` heatmap; on chr15 from ``stdpopsim``'s
``OutOfAfrica_2T12`` at 50,000 diploids per population (200,000
haplotypes, 11.6 M variants, 7.9 GB on disk) it completes in about
16 minutes on a single A100 80 GB.

When streaming is not the right choice
--------------------------------------

* Your data already fits on the GPU in one shot. ``streaming='auto'``
  picks the single-shot path automatically; that path is faster per
  call because there is no per-chunk overhead.
* Your statistic needs every variant at once and the whole
  chromosome is too big to load. Use ``materialize(region=...)`` to
  read a smaller sub-region instead.
* Your store is in scikit-allel layout rather than VCZ. Convert it
  first with ``allel_zarr_to_vcz``; the streaming reader only
  accepts VCZ.
