"""Chunk-streamed view over a VCZ store, used by HaplotypeMatrix.from_zarr
at biobank scale.

A ``StreamingHaplotypeMatrix`` does not materialize the full genotype
matrix; it iterates the store in genomic chunks via ``iter_gpu_chunks()``.
Each chunk arrives as an eager ``HaplotypeMatrix`` on the GPU covering
one slice of the chromosome. Compute-side overlap with the next chunk's
read happens through a ``ChunkFetcher`` -- ``HostChunkFetcher`` runs a
producer thread that calls ``ZarrGenotypeSource.slice_region`` while the
GPU is working on the current chunk, with one chunk of read-ahead by
default.

Streaming-aware kernels (``windowed_analysis``, ``sfs.sfs``,
``sfs.joint_sfs``) dispatch on ``isinstance(hm, StreamingHaplotypeMatrix)``
at the top of the call and run themselves chunk-by-chunk. Kernels
that have not been adapted yet (Garud, pairwise statistics) raise on
this class; ``.materialize(region=...)`` returns an eager
``HaplotypeMatrix`` over a sub-region for those.
"""

import queue
import threading
import time
import warnings
from abc import ABC, abstractmethod

import cupy as cp
import kvikio
import kvikio.defaults
import numpy as np
import zarr
from kvikio.zarr import GDSStore

from ._gpu_genotype_prep import build_genotype_matrix, build_haplotype_matrix
from ._warnings import BadlyChunkedWarning, KvikioUnavailableWarning


def _pad_to(arr, shape):
    """Zero-pad ``arr`` to the given (potentially larger) shape."""
    out = np.zeros(shape, dtype=arr.dtype)
    out[tuple(slice(0, n) for n in arr.shape)] = arr
    return out


def _stream_sum(streaming_hm, kernel_fn):
    """Run ``kernel_fn`` on every chunk and sum the per-chunk numpy results.

    Used by SFS-style kernels whose chunk results compose by addition --
    each chunk contributes its own bincount or joint-bincount, and the
    chromosome-wide answer is their sum. Same-shape chunk results take a
    fast path (in-place add); the slow path only fires on the edge case
    where two chunks have different shapes (e.g. one chunk's population
    has strictly more valid samples after masking than another, giving
    it an extra bin along that axis).
    """
    total = None
    for _, _, chunk in streaming_hm.iter_gpu_chunks():
        # Preserve the kernel's own dtype: integer bincounts (sfs, joint_sfs)
        # stay int64; the folded SFS carries half-integer weights, so it must
        # accumulate in float (a forced int64 cast would truncate per chunk).
        s = np.asarray(kernel_fn(chunk))
        if total is None:
            total = s.copy()
            continue
        if s.shape == total.shape:
            total += s
            continue
        shape = tuple(max(a, b) for a, b in zip(total.shape, s.shape))
        total = _pad_to(total, shape)
        total += _pad_to(s, shape)
    return total


class ChunkFetcher(ABC):
    """Pluggable producer that yields per-chunk genotype blocks.

    Implementations are responsible for whatever async / parallelism
    strategy they want; the consumer (``StreamingHaplotypeMatrix``)
    only consumes the yielded tuples. Errors raised inside the
    producer must be forwarded to the consumer with their traceback
    preserved.
    """

    @abstractmethod
    def iter_chunks(self, chunks, prefetch):
        """Yield ``(ci, left, right, gt, pos, t_read_s)`` per chunk.

        Parameters
        ----------
        chunks : list of (int, int)
            Half-open ``(left, right)`` bp intervals.
        prefetch : int
            Read-ahead depth. ``0`` means serial; >=1 means a worker
            thread reads ahead of the consumer.
        """


#: Codecs the kvikio + nvCOMP path can decode on the GPU. Stores using
#: anything else fall back to the host fetcher; on-disk pg_gpu / bio2zarr
#: stores default to zstd, so the common case is GPU-decodable.
_NVCOMP_SUPPORTED_CODECS = frozenset({"zstd", "blosc", "lz4", "deflate"})


def _pick_chunk_fetcher(source, *, backend):
    """Pick a fetcher for ``source``. Errors and warnings reflect what the
    caller asked for: ``backend='kvikio'`` raises on an incompatible
    store, while ``backend='auto'`` warns once and falls back."""
    if backend == "host":
        return HostChunkFetcher(source)
    if backend == "kvikio":
        return KvikioChunkFetcher(source)

    if not _kvikio_can_decode(source):
        # A zarr v2 store decodes through CPU numcodecs, and an
        # uncompressed or unsupported codec has no nvCOMP GPU decoder --
        # either way kvikio cannot engage, so use the host fetcher. Hint
        # on the v2 case (the common, fixable one) when the store is big
        # enough that the GPU path would have mattered; small fixtures
        # would just get noise.
        if source.zarr_format != 3 and _store_large_enough_to_warn(source):
            warnings.warn(
                f"{source.path}: zarr v{source.zarr_format} store; the "
                f"kvikio GPU decoder needs zarr v3 (a v2 compressor decodes "
                f"on the CPU). Falling back to the host fetcher. Re-encode "
                f"as zarr v3 with HaplotypeMatrix.vcf_to_zarr(..., "
                f"zarr_format=3) to enable kvikio.",
                KvikioUnavailableWarning, stacklevel=4,
            )
        return HostChunkFetcher(source)
    chunks = source.chunks  # already cached on the source from __init__
    whole_sample_axis = (chunks is not None and len(chunks) >= 2
                         and chunks[1] >= source.num_diploids)
    if whole_sample_axis:
        # Whole-sample-axis chunking defeats the kvikio fetcher's win on
        # full-haplotype reads: zarr's async pipeline already saturates
        # ~3 cores per chunk decode, so a GPU codec on a single chunk
        # buys nothing. Only warn when the store is large enough that
        # the user would actually see the speedup of a rechunk -- on
        # small test fixtures the warning is noise.
        if _store_large_enough_to_warn(source):
            warnings.warn(
                f"{source.path}: call_genotype.chunks={chunks} spans the "
                f"full sample axis ({source.num_diploids} diploids); the "
                f"kvikio fetcher gives no speedup at this chunking. "
                f"Falling back to the host fetcher. Re-encode with "
                f"HaplotypeMatrix.vcf_to_zarr(..., zarr_format=3), whose "
                f"bio2zarr sample chunking splits the sample axis, to "
                f"enable kvikio.",
                BadlyChunkedWarning, stacklevel=4,
            )
        return HostChunkFetcher(source)
    return KvikioChunkFetcher(source)


def _kvikio_can_decode(source):
    """True when kvikio's nvCOMP GPU decoder can read this store: a zarr
    v3 store (a v2 store decodes through CPU numcodecs) whose
    call_genotype codec nvCOMP supports."""
    return (source.zarr_format == 3
            and source.codec in _NVCOMP_SUPPORTED_CODECS)


def _store_large_enough_to_warn(source):
    """A kvikio fallback is only worth a warning on a store big enough
    that the GPU decode path would have mattered; on small fixtures the
    warning is just noise."""
    warn_threshold_bytes = 1 << 30  # 1 GiB of int8 footprint
    eager_bytes = int(source.num_variants) * int(source.num_haplotypes)
    return eager_bytes >= warn_threshold_bytes


def _iter_chunks_with_prefetch(chunks, prefetch, slice_fn, *,
                                thread_name="chunk-prefetch"):
    """Shared producer-thread iteration for ``ChunkFetcher`` implementations.

    ``slice_fn(left, right)`` returns ``(gt, pos)`` -- the host fetcher
    passes a numpy-returning callable, the kvikio fetcher passes a
    cupy-returning one, and everything else (bounded queue, error
    propagation, daemon-thread cleanup) is identical.

    Yields ``(ci, left, right, gt, pos, t_read_s)`` tuples. ``prefetch
    <= 0`` reads serially in the consumer thread; ``prefetch >= 1``
    runs a daemon producer that fills a bounded queue with ``prefetch``
    chunks of read-ahead, so the next chunk's read is in flight while
    the consumer is computing on the current chunk.
    """
    if prefetch <= 0:
        for ci, (left, right) in enumerate(chunks):
            t0 = time.perf_counter()
            gt, pos = slice_fn(left, right)
            yield ci, left, right, gt, pos, time.perf_counter() - t0
        return

    # bounded queue keeps host RAM at (prefetch + 1) * chunk_bytes;
    # _END is a private sentinel; ("ERR", exc) forwards producer-side
    # exceptions to the consumer's call stack with the traceback intact.
    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    _END = object()

    def producer():
        try:
            for ci, (left, right) in enumerate(chunks):
                if stop.is_set():
                    return
                t0 = time.perf_counter()
                gt, pos = slice_fn(left, right)
                t_read = time.perf_counter() - t0
                if stop.is_set():
                    return
                q.put((ci, left, right, gt, pos, t_read))
        except BaseException as e:
            q.put(("ERR", e))
            return
        q.put(_END)

    t = threading.Thread(target=producer, daemon=True, name=thread_name)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                raise item[1]
            yield item
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        t.join(timeout=5)


class HostChunkFetcher(ChunkFetcher):
    """Read chunks through the source's host-buffer ``slice_region``.

    ``prefetch >= 1`` spawns a daemon producer thread that fills a
    bounded queue with the next chunk while the consumer is computing
    on the current chunk. A GPU-codec-decode fetcher (kvikio +
    nvidia-nvcomp) is a drop-in replacement for the same ABC when
    the store's codec is GPU-decodable.
    """

    def __init__(self, source):
        self._source = source

    def iter_chunks(self, chunks, prefetch):
        yield from _iter_chunks_with_prefetch(
            chunks, prefetch,
            self._source.slice_region,
            thread_name="zarr-prefetch",
        )


class KvikioChunkFetcher(ChunkFetcher):
    """Read chunks via ``kvikio.zarr.GDSStore`` with on-GPU codec decode.

    Hardware / software requirements (the kvikio path falls back to
    ``HostChunkFetcher`` if these are not met):

    * NVIDIA driver and a CUDA-capable GPU; pg_gpu already requires
      both.
    * ``kvikio`` and ``nvidia-nvcomp`` Python packages in the same
      environment as ``pg_gpu`` (the project's pixi env ships these
      by default).
    * The VCZ store's ``call_genotype`` chunks compressed with a
      codec in nvCOMP's GPU-decode list -- currently zstd, blosc,
      lz4, or deflate. bio2zarr defaults to zstd. Other codecs
      surface as ``ValueError`` on construction so the caller can
      re-encode.
    * Sample-axis chunking smaller than the full sample axis. A
      store with one whole-sample-axis chunk per variant chunk is
      still readable but gets no kvikio speedup, so ``backend='auto'``
      warns and falls back to the host path.

    No special filesystem support is needed. The fetcher opens the
    store through ``kvikio.zarr.GDSStore`` and switches zarr 3 to its
    GPU buffer prototype so nvCOMP decodes the codec directly on the
    device -- ``build_haplotype_matrix`` then skips the
    host-to-device copy in ``cp.asarray(gt)`` because ``gt`` is
    already a cupy array. The win is in that on-GPU decode, not in
    GPU Direct Storage.

    By default this fetcher sets ``kvikio.defaults.compat_mode=ON``
    so kvikio reads bytes through posix into bounce buffers rather
    than trying to use GPU Direct Storage (which would be slower
    on hosts without ``/etc/cufile.json`` configured because of
    repeated failed cufile probes). Pass ``compat_mode=AUTO`` if
    the storage path is known to be GDS-capable.

    Construction probes the store's ``call_genotype`` codec and
    raises ``ValueError`` if it's not in the nvCOMP-supported list,
    rather than silently returning incorrect data through a CPU
    fallback. On close the zarr buffer prototype is reset so
    subsequent eager calls in the same process get numpy buffers
    back.
    """

    def __init__(self, source, *, compat_mode="ON", num_threads=8):
        if compat_mode not in ("ON", "AUTO", "OFF"):
            raise ValueError(
                f"compat_mode must be 'ON', 'AUTO', or 'OFF'; "
                f"got {compat_mode!r}"
            )
        if source.zarr_format != 3:
            raise ValueError(
                f"kvikio GPU decode requires a zarr v3 store; "
                f"{source.path} is zarr v{source.zarr_format}, whose "
                f"numcodecs compressor decodes on the CPU. Re-encode as "
                f"zarr v3 with HaplotypeMatrix.vcf_to_zarr(..., "
                f"zarr_format=3) to use the kvikio backend."
            )
        if source.codec not in _NVCOMP_SUPPORTED_CODECS:
            raise ValueError(
                f"KvikioChunkFetcher requires a call_genotype codec the "
                f"nvCOMP GPU decoder supports "
                f"({sorted(_NVCOMP_SUPPORTED_CODECS)}); {source.path} has "
                f"codec {source.codec!r}"
            )
        self._source = source
        # GDSStore is a separate handle from source._store: the codec
        # pipeline is cached on the zarr group, and we need a group
        # opened with the GPU buffer prototype active so nvCOMP runs
        # the decoder on the device. Reusing source._store would
        # pick up the wrong codec pipeline.
        self._gds_store = GDSStore(source.path)
        mode_const = {"ON": kvikio.CompatMode.ON,
                      "AUTO": kvikio.CompatMode.AUTO,
                      "OFF": kvikio.CompatMode.OFF}[compat_mode]
        # kvikio.defaults.set is process-global; this writes the
        # config every time a new fetcher is constructed, so the last
        # fetcher wins if multiple are alive at once. In practice only
        # one streaming matrix is iterated at a time on biobank-scale
        # work, so this is acceptable.
        kvikio.defaults.set({"compat_mode": mode_const,
                             "num_threads": int(num_threads)})
        self._gpu_buffer_active = False

    def iter_chunks(self, chunks, prefetch):
        # zarr.config.enable_gpu is process-global; we flip it on for
        # the duration of the iteration and reset on every exit path
        # so a downstream eager call gets numpy buffers back. The
        # call_genotype handle is opened once here (with the GPU
        # buffer active) and reused for every chunk read.
        self._enable_gpu_buffer()
        try:
            cg = zarr.open_group(self._gds_store, mode="r")["call_genotype"]
            yield from _iter_chunks_with_prefetch(
                chunks, prefetch,
                lambda left, right: self._source.slice_region_gpu(
                    left, right, cg=cg),
                thread_name="kvikio-prefetch",
            )
        finally:
            self._reset_gpu_buffer()

    def close(self):
        """Reset the zarr buffer prototype.

        Called automatically at the end of ``iter_chunks``; exposed for
        tests and for callers that build a fetcher without immediately
        iterating it. There is no ``__del__`` -- Python's GC timing is
        nondeterministic, so cleanup is via try/finally inside
        ``iter_chunks`` and explicit ``close()`` for callers that
        construct without iterating.
        """
        self._reset_gpu_buffer()

    def _enable_gpu_buffer(self):
        zarr.config.enable_gpu()
        self._gpu_buffer_active = True

    def _reset_gpu_buffer(self):
        if not self._gpu_buffer_active:
            return
        zarr.config.set({
            "buffer": "zarr.core.buffer.cpu.Buffer",
            "ndbuffer": "zarr.core.buffer.cpu.NDBuffer",
        })
        self._gpu_buffer_active = False


class _StreamingMatrixBase:
    """Shared chunk-iteration machinery for streaming matrix classes.

    Subclasses provide the per-chunk ``_build_chunk`` (which decides
    HaplotypeMatrix vs GenotypeMatrix layout), the sample-axis size
    that drives the default 'all' sample set, and the data-property
    raise text (since the eager classes' bulk-data attribute names
    differ: ``.haplotypes`` vs ``.genotypes``).

    Direct array access is not supported -- the whole point of these
    classes is that the matrix is too big to materialize. Use
    ``iter_gpu_chunks()`` to walk per-chunk eager matrices, pass the
    object to a streaming-aware kernel, or call
    ``.materialize(region=...)`` for a sub-region eager build.

    Parameters
    ----------
    source : ZarrGenotypeSource
    fetcher : ChunkFetcher
    chunk_bp : int
        Genomic span per chunk, in bp.
    prefetch : int
        Read-ahead depth handed to the fetcher.
    align_bp : int, optional
        Chunk boundaries are snapped to multiples of this so a
        windowed kernel can guarantee windows never straddle a chunk
        boundary. Defaults to ``chunk_bp`` (single window per chunk).
    """

    def __init__(self, source, fetcher, chunk_bp, prefetch, *,
                 align_bp=None, accessible_mask=None, n_total_sites=None):
        self._source = source
        self._fetcher = fetcher
        self._chunk_bp = int(chunk_bp)
        self._prefetch = int(prefetch)
        self._align_bp = int(align_bp) if align_bp is not None else self._chunk_bp
        self._chunks = list(
            source.iter_chunks(self._chunk_bp, self._align_bp)
        )
        # Mirror the eager classes' idiom: store the explicit value
        # (or None) and let the property fall back to a default 'all'
        # set when no pop file resolved at source construction. The pop
        # file resolves to haplotype columns; _sets_in_row_space puts them
        # in this stream's own row space. Validate once here -- every
        # chunk shares the stream's row space, so chunks carry the sets
        # without re-checking.
        sets = source.pop_cols
        if sets:
            sets = self._sets_in_row_space(sets)
        # None means no pop file; {} means a pop file matched nothing.
        # Keeping the difference mirrors the eager loaders.
        self.sample_sets = sets
        # Span-normalization metadata, mirroring the eager HaplotypeMatrix.
        # None when no accessible BED was supplied at load; see get_span.
        self.accessible_mask = accessible_mask
        self.n_total_sites = n_total_sites
        # Emit the multiallelic->missing BiallelicOnlyWarning at most once per
        # streaming load, on the first chunk that recodes any site.
        self._biallelic_warned = False

    @property
    def num_variants(self):
        return self._source.num_variants

    @property
    def chrom(self):
        return self._source.chrom

    @property
    def align_bp(self):
        """Chunk-boundary alignment in bp.

        Every chunk has a width that is a multiple of this, so a window
        whose size also divides ``align_bp`` is guaranteed to fit inside
        a single chunk. Streaming-aware kernels read this property to
        validate that the caller's ``window_size`` divides it.
        """
        return self._align_bp

    @property
    def chrom_start(self):
        """Chunk-grid origin. Not the first variant position --
        per-chunk windows are anchored to the chunk grid, so reporting
        the variant-based origin would be misleading for callers
        building a comparable eager matrix."""
        return self._chunks[0][0] if self._chunks else 0

    @property
    def chrom_end(self):
        """Chunk-grid right edge (exclusive). Not the last variant
        position; per-chunk windows are uniform width within their
        chunk, so the right edge here is the chunk grid's exclusive
        upper bound rather than HaplotypeMatrix's last-inclusive
        convention."""
        return self._chunks[-1][1] if self._chunks else 0

    def get_span(self, mode='auto'):
        """Genomic span for normalization, mirroring HaplotypeMatrix.get_span.

        Spans use the *variant-position* bounds (``mappable_lo``/``mappable_hi``)
        rather than the chunk-grid ``chrom_start``/``chrom_end``, so the value
        matches an eager matrix built from the same store. ``mappable_hi`` is one
        past the last variant, so the raw span is ``mappable_hi - mappable_lo``
        and the accessible count is over the half-open ``[lo, hi)``.

        'auto' priority: accessible-mask count > n_total_sites > raw per-base span.

        ``per_variant``/``sites`` and ``callable`` are computed over the
        accessible-filtered variant set (as the eager path does through its
        mask-filtered variant view), so they agree with eager under a mask;
        ``per_base``/``total`` is the raw variant span, mask-independent.
        """
        lo = self._source.mappable_lo
        hi = self._source.mappable_hi          # one past the last variant
        if mode == 'auto':
            if self.accessible_mask is not None:
                return self.accessible_mask.count_accessible(lo, hi)
            if self.n_total_sites is not None:
                return self.n_total_sites
            return hi - lo
        if mode == 'accessible':
            if self.accessible_mask is None:
                raise ValueError("mode='accessible' requires an accessible mask")
            return self.accessible_mask.count_accessible(lo, hi)
        if mode in ('per_base', 'total'):
            return hi - lo
        if mode in ('per_variant', 'sites', 'callable'):
            pos = self._source.site_pos
            if self.accessible_mask is not None:
                pos = pos[self.accessible_mask.is_accessible_at(pos)]
            if mode == 'callable':
                return int(pos[-1] - pos[0] + 1) if pos.size else 0
            return int(pos.size)
        raise ValueError(f"Invalid span mode: {mode}")

    @property
    def sample_sets(self):
        """Population -> sample-axis indices. Falls back to a single
        'all' set when no pop file was resolved."""
        if self._sample_sets is None:
            return {"all": list(range(self._sample_axis_size()))}
        return self._sample_sets

    @sample_sets.setter
    def sample_sets(self, value):
        """
        Set the sample sets.

        Each value must name rows in this stream's own row space --
        haplotype rows on a haplotype stream, individual rows on a
        genotype stream -- as a list, tuple, or one-dimensional integer
        array, without duplicates. Every chunk carries these sets, so an
        invalid assignment would put garbage rows in front of each
        chunk's statistics. The dict is stored as given, not copied;
        mutating it afterwards is not re-validated.
        """
        if value is None:
            self._sample_sets = None
            return
        if not isinstance(value, dict):
            raise ValueError("sample_sets must be a dictionary")
        from ._warnings import check_sample_set_rows
        n_rows = self._sample_axis_size()
        for key, rows in value.items():
            check_sample_set_rows(f"sample_sets[{key!r}]", rows, n_rows)
        self._sample_sets = value

    def iter_gpu_chunks(self):
        """Yield ``(left, right, eager_matrix)`` tuples covering the source.

        Each yielded eager matrix lives on the GPU and represents one
        genomic chunk. Empty chunks (regions with no variants, e.g. an
        acrocentric arm) are skipped -- callers see only chunks with
        at least one variant.
        """
        for ci, left, right, gt, pos, t_read in self._fetcher.iter_chunks(
                self._chunks, self._prefetch):
            if gt.shape[0] == 0:
                continue
            m = self._build_chunk(
                gt, pos,
                # chrom_end is the chunk's exclusive right edge so the
                # last window in each chunk does not get clipped to the
                # last variant position the way an eager matrix would.
                chrom_start=int(left), chrom_end=int(right),
                sample_sets=self._sample_sets,
            )
            # A load-time accessible mask filters variants to accessible sites,
            # exactly as it does on an eager matrix -- attach it here, at the
            # single chunk-production point, so every streaming consumer sees
            # accessible-filtered chunks by construction.
            if self.accessible_mask is not None:
                m.accessible_mask = self.accessible_mask
            yield int(left), int(right), m

    def materialize(self, *, region=None, sample_subset=None):
        """Build an eager matrix over a sub-region.

        Pairwise / cross-window kernels (``pairwise_r2``, ``grm``,
        ``ibs``, ``locate_unlinked``, the r^2 heatmap path) can't be
        evaluated chunk-by-chunk because they need every (variant,
        variant) pair simultaneously. Pull the slice you want into one
        device-resident eager matrix and run those kernels on it.

        Parameters
        ----------
        region : tuple of int, optional
            ``(left, right)`` bp interval to materialize. ``right`` is
            exclusive. ``None`` materializes the full mappable range,
            which on a biobank-scale store will OOM.
        sample_subset : sequence of int, optional
            Rows to keep, in this stream's own row space -- the same
            space ``sample_sets`` uses, so
            ``materialize(sample_subset=stream.sample_sets[pop])`` works
            on either class. ``None`` keeps everything. The subset obeys
            the same rules as a ``sample_sets`` value: rows within the
            matrix, no duplicates.

            On a haplotype stream rows are gametes: sample ``i`` is rows
            ``2i`` and ``2i + 1``, and asking for one gamete of a pair
            gives back rows that sit side by side without being one
            sample -- do not feed such a matrix to anything that works
            per individual (``grm``, ``ibs``, ``fst_weir_cockerham``,
            ``heterozygosity_observed``,
            ``GenotypeMatrix.from_haplotype_matrix``). On a genotype
            stream rows are individuals, so whole samples are the only
            thing it is possible to ask for.

            The stream's ``sample_sets`` are renumbered into the subset:
            each population keeps the members the subset retains, and a
            population with none left is dropped.
        """
        if region is None:
            left, right = self.chrom_start, self.chrom_end
        else:
            left, right = int(region[0]), int(region[1])

        if sample_subset is None:
            gt, pos = self._source.slice_region(left, right)
        else:
            # The subset names rows in this stream's own row space and
            # obeys the same rules as a sample_sets value; the store read
            # below wants haplotype columns, which _subset_hap_columns
            # supplies per class.
            from ._warnings import check_sample_set_rows
            sample_subset = list(sample_subset)
            check_sample_set_rows("sample_subset", sample_subset,
                                  self._sample_axis_size())
            hap_cols = self._subset_hap_columns(sample_subset)
            # If the fetcher uses kvikio, decompress through it: at
            # biobank-scale sample subsets the host-side oindex codec
            # pipeline that ``slice_subsample`` would use is minutes per
            # probe; the kvikio + nvCOMP path is seconds. Otherwise
            # fall back to the host stage with the gather on GPU.
            if isinstance(self._fetcher, KvikioChunkFetcher):
                gm_gpu, pos = self._read_subsample_via_kvikio(
                    left, right, hap_cols
                )
            else:
                gm_gpu, pos = self._source.slice_subsample(
                    left, right, hap_cols, to_gpu=True
                )
            n_var, n_hap = gm_gpu.shape
            if n_hap % 2 != 0:
                raise ValueError(
                    f"materialize(sample_subset=...) requires an even "
                    f"count to round-trip through (n_dip, 2) layout; "
                    f"got {n_hap}."
                )
            n_dip_sub = n_hap // 2
            # Canonical row order puts a sample's two gametes adjacent, so the
            # haplotype axis is already the flattened (n_dip', 2) layout and
            # the split is a view rather than a copy.
            gt = gm_gpu.reshape(n_var, n_dip_sub, 2)

        # sample_subset renumbers the rows, so full-matrix sample_sets no
        # longer apply. Keep each population's surviving members, renumbered
        # into the subset; a population with no member left is dropped. The
        # renumbering is per-class: the subset is haplotype columns, and the
        # genotype stream's sets hold individual rows.
        sets = self._sample_sets
        if sample_subset is not None and sets:
            sets = {p: rows for p, rows in
                    self._renumber_sets(sets, sample_subset).items()
                    if len(rows)}

        return self._build_chunk(
            gt, pos,
            chrom_start=left, chrom_end=right,
            sample_sets=sets,
        )

    def _renumber_sets(self, sets, sample_subset):
        """Sets renumbered into the subset's row space; rows the subset
        does not retain are dropped. The subset and the sets share this
        stream's row space, so one mapping serves both classes."""
        new_row = {int(r): i for i, r in enumerate(sample_subset)}
        return {p: [new_row[int(r)] for r in rows if int(r) in new_row]
                for p, rows in sets.items()}

    def _subset_hap_columns(self, sample_subset):
        """The store's haplotype columns for a subset given in this
        stream's row space. Host ints for a haplotype stream (a cupy
        subset arrives as device scalars); the genotype stream expands
        each individual to its two gametes."""
        return [int(i) for i in sample_subset]

    def _read_subsample_via_kvikio(self, left, right, sample_subset):
        """Open the fetcher's GDSStore-backed ``call_genotype`` array
        with the GPU buffer prototype active and route
        ``slice_subsample_gpu`` through it. The buffer prototype is
        reset on every exit path so subsequent eager work gets numpy
        buffers back."""
        self._fetcher._enable_gpu_buffer()
        try:
            cg = zarr.open_group(
                self._fetcher._gds_store, mode="r"
            )["call_genotype"]
            return self._source.slice_subsample_gpu(
                left, right, sample_subset, cg=cg,
            )
        finally:
            self._fetcher._reset_gpu_buffer()

    def _sample_axis_size(self):  # pragma: no cover -- abstract
        raise NotImplementedError

    def _build_chunk(self, gt, pos, **kwargs):  # pragma: no cover -- abstract
        raise NotImplementedError

    def _sets_in_row_space(self, pop_cols):
        """Pop-file sets in this stream's row space.

        The pop file resolves to haplotype columns, which already are the
        rows of a haplotype stream; the genotype stream overrides this to
        map gametes to individuals.
        """
        return pop_cols

    def __repr__(self):
        return (
            f"{type(self).__name__}(num_variants={self.num_variants}, "
            f"{self._repr_sample_axis()}, "
            f"chrom={self.chrom!r}, n_chunks={len(self._chunks)}, "
            f"chunk_bp={self._chunk_bp}, prefetch={self._prefetch})"
        )

    def _repr_sample_axis(self):  # pragma: no cover -- abstract
        raise NotImplementedError


class StreamingHaplotypeMatrix(_StreamingMatrixBase):
    """Chunked view over a ``ZarrGenotypeSource``, yielding per-chunk
    ``HaplotypeMatrix`` instances.

    Returned by ``HaplotypeMatrix.from_zarr`` when the requested
    matrix does not fit eagerly on the GPU. See ``_StreamingMatrixBase``
    for the iteration / materialize / sample_sets contract.

    Every statistic call walks the region from disk again, so N calls
    cost N full passes; batch work into one pass where possible, or
    ``materialize`` a sub-region for repeated calls on the same data.
    """

    @property
    def num_haplotypes(self):
        return self._source.num_haplotypes

    @property
    def haplotypes(self):
        raise NotImplementedError(
            "StreamingHaplotypeMatrix has no materialized .haplotypes "
            "array; the matrix is too big to fit eagerly, which is why "
            "from_zarr returned this class instead of HaplotypeMatrix. "
            "For grm or any other kernel that needs every variant in "
            "scope at once, call .materialize(region=(lo, hi)) to get "
            "an eager HaplotypeMatrix over a sub-region. For per-window "
            "streaming stats, ibs, and the LD pair-bin statistics, "
            "pass this object directly to the kernel."
        )

    def _sample_axis_size(self):
        return self.num_haplotypes

    def _build_chunk(self, gt, pos, **kwargs):
        # The stream's setter validated these sets in the stream's row
        # space, which every chunk shares, so skip the eager setter's
        # re-validation -- np.unique per population per chunk adds up to
        # minutes over a biobank-scale walk.
        sets = kwargs.pop('sample_sets', None)
        m = build_haplotype_matrix(gt, pos, **kwargs)
        if sets is not None:
            m._sample_sets = sets
        return m

    def _repr_sample_axis(self):
        return f"num_haplotypes={self.num_haplotypes}"

    def compute_ld_statistics_gpu_single_pop(self, bp_bins, raw=False,
                                              chunk_size='auto'):
        from .haplotype_matrix import _stream_ld_single_pop
        return _stream_ld_single_pop(
            self, bp_bins=bp_bins, raw=raw, chunk_size=chunk_size,
        )

    def compute_ld_statistics_gpu_two_pops(self, bp_bins, pop1, pop2,
                                            raw=False, chunk_size='auto'):
        from .haplotype_matrix import _stream_ld_two_pops
        return _stream_ld_two_pops(
            self, bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw,
            chunk_size=chunk_size,
        )


class StreamingGenotypeMatrix(_StreamingMatrixBase):
    """Chunked view over a ``ZarrGenotypeSource``, yielding per-chunk
    ``GenotypeMatrix`` (dosage-coded ``(n_indiv, n_var)``) instances.

    Returned by ``GenotypeMatrix.from_zarr`` when the requested matrix
    does not fit eagerly on the GPU. Sample sets index the diploid
    axis (``0..num_individuals-1``), not the haplotype axis. ``grm`` and
    ``ibs`` accept this class directly, streaming variant chunks; kernels
    that need every variant in scope at once call
    ``.materialize(region=(lo, hi))`` for an eager sub-region.
    """

    @property
    def num_individuals(self):
        return self._source.num_diploids

    @property
    def genotypes(self):
        raise NotImplementedError(
            "StreamingGenotypeMatrix has no materialized .genotypes "
            "array; the matrix is too big to fit eagerly, which is why "
            "from_zarr returned this class instead of GenotypeMatrix. "
            "For per-window streaming stats, grm, and ibs, pass this "
            "object directly to the kernel. For a kernel that needs "
            "every variant in scope at once, call "
            ".materialize(region=(lo, hi)) to get an eager "
            "GenotypeMatrix over a sub-region."
        )

    def _sample_axis_size(self):
        return self.num_individuals

    def _sets_in_row_space(self, pop_cols):
        # The pop file resolves to haplotype columns, but this stream's
        # rows are individuals. Both of a sample's gametes floor-divide to
        # its individual, so the unique halves are the individual rows.
        return {p: np.unique(np.asarray(cols) // 2)
                for p, cols in pop_cols.items()}

    def _subset_hap_columns(self, sample_subset):
        # This stream's rows are individuals; the store's columns are
        # gametes. Sample i owns columns 2i and 2i + 1, and the pair
        # lands adjacently, so the (n_dip, 2) reshape below recovers
        # exactly the requested individuals in order.
        return [c for i in sample_subset for c in (2 * int(i), 2 * int(i) + 1)]

    def _build_chunk(self, gt, pos, **kwargs):
        # Same skip as the haplotype stream: the sets were validated once
        # at construction, in this stream's own (individual) row space.
        sets = kwargs.pop('sample_sets', None)
        m = build_genotype_matrix(gt, pos, **kwargs)
        if sets is not None:
            m._sample_sets = sets
        if m._n_multiallelic_recoded and not self._biallelic_warned:
            from ._warnings import _warn_biallelic_only
            # The count covers only the chunk that trips the latch; a full
            # region count would need a whole extra pass over the store.
            _warn_biallelic_only(
                m._n_multiallelic_recoded,
                context="GenotypeMatrix.from_zarr (streaming; count is for "
                        "the first affected chunk, later chunks recode "
                        "silently)",
                action="recoded to all-missing rows")
            self._biallelic_warned = True
        return m

    def _repr_sample_axis(self):
        return f"num_individuals={self.num_individuals}"
