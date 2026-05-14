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

Kernels accept this class via dispatch in a follow-up; until then,
calling a kernel directly raises a clear error. Use ``iter_gpu_chunks``
to walk chunks manually if you need to.
"""

import queue
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from ._gpu_genotype_prep import build_haplotype_matrix


def _pad_to(arr, shape):
    """Zero-pad ``arr`` to the given (potentially larger) shape."""
    out = np.zeros(shape, dtype=arr.dtype)
    out[tuple(slice(0, n) for n in arr.shape)] = arr
    return out


def _stream_sum(streaming_hm, kernel_fn):
    """Run ``kernel_fn`` on every chunk and sum the per-chunk numpy results.

    Used by SFS-style kernels whose chunk results compose by addition --
    each chunk contributes its own bincount or joint-bincount, and the
    chromosome-wide answer is their sum. The padding step covers the
    edge case where two chunks produce arrays of different shapes (e.g.
    when one chunk has a population with strictly more valid samples
    after masking than another, giving it one more bin along that
    axis).
    """
    total = None
    for _, _, chunk in streaming_hm.iter_gpu_chunks():
        s = np.asarray(kernel_fn(chunk), dtype=np.int64)
        if total is None:
            total = s.copy()
            continue
        if s.shape != total.shape:
            shape = tuple(max(a, b) for a, b in zip(total.shape, s.shape))
            total = _pad_to(total, shape)
            s = _pad_to(s, shape)
        total += s
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


class HostChunkFetcher(ChunkFetcher):
    """Read chunks through the source's host-buffer ``slice_region``.

    ``prefetch >= 1`` spawns a daemon producer thread that fills a
    bounded queue with the next chunk while the consumer is computing
    on the current chunk. This is the right baseline for any local
    zarr store; the kvikio fetcher in a follow-up swaps the same
    interface for GPU-side codec decode.
    """

    def __init__(self, source):
        self._source = source

    def iter_chunks(self, chunks, prefetch):
        if prefetch <= 0:
            yield from self._iter_serial(chunks)
            return
        yield from self._iter_with_producer(chunks, prefetch)

    def _iter_serial(self, chunks):
        for ci, (left, right) in enumerate(chunks):
            t0 = time.perf_counter()
            gt, pos = self._source.slice_region(left, right)
            yield ci, left, right, gt, pos, time.perf_counter() - t0

    def _iter_with_producer(self, chunks, prefetch):
        # Use a bounded queue so the producer blocks once it is `prefetch`
        # chunks ahead, preventing unbounded host RAM growth on slow
        # consumers. _END is a private sentinel; ("ERR", exc) forwards
        # producer-side exceptions to the consumer's call stack.
        q = queue.Queue(maxsize=prefetch)
        stop = threading.Event()
        _END = object()

        def producer():
            try:
                for ci, (left, right) in enumerate(chunks):
                    if stop.is_set():
                        return
                    t0 = time.perf_counter()
                    gt, pos = self._source.slice_region(left, right)
                    t_read = time.perf_counter() - t0
                    if stop.is_set():
                        return
                    q.put((ci, left, right, gt, pos, t_read))
            except BaseException as e:
                q.put(("ERR", e))
                return
            q.put(_END)

        t = threading.Thread(target=producer, daemon=True,
                             name="zarr-prefetch")
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
            # drain so the producer's last put doesn't block forever
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5)


class StreamingHaplotypeMatrix:
    """Chunked view over a ``ZarrGenotypeSource``.

    Used by ``HaplotypeMatrix.from_zarr`` when the requested matrix
    does not fit eagerly on the GPU. Kernels that consume this class
    do so by iterating ``iter_gpu_chunks``; each yielded
    ``HaplotypeMatrix`` is an eager device-resident chunk covering a
    single genomic interval.

    Direct array access (``.haplotypes``, ``.positions``) is not
    supported -- the whole point of this class is that the matrix is
    too big to materialize. The dispatch in the kernels uses the
    iterator instead.

    Parameters
    ----------
    source : ZarrGenotypeSource
    fetcher : ChunkFetcher
    chunk_bp : int
        Genomic span per chunk, in bp.
    prefetch : int
        Read-ahead depth handed to the fetcher.
    align_bp : int, optional
        Chunk boundaries are snapped to multiples of this so a windowed
        kernel can guarantee windows never straddle a chunk boundary.
        Defaults to ``chunk_bp`` (single window per chunk).
    """

    def __init__(self, source, fetcher, chunk_bp, prefetch, *,
                 align_bp=None):
        self._source = source
        self._fetcher = fetcher
        self._chunk_bp = int(chunk_bp)
        self._prefetch = int(prefetch)
        self._align_bp = int(align_bp) if align_bp is not None else self._chunk_bp
        self._chunks = list(
            source.iter_chunks(self._chunk_bp, self._align_bp)
        )
        # _sample_sets follows the same idiom as HaplotypeMatrix: store the
        # explicit value (or None) and let the property fall back to a
        # default 'all' set when no pop file resolved at source construction.
        self._sample_sets = source.pop_cols

    @property
    def num_variants(self):
        return self._source.num_variants

    @property
    def num_haplotypes(self):
        return self._source.num_haplotypes

    @property
    def chrom(self):
        return self._source.chrom

    @property
    def chrom_start(self):
        """Leading edge of the analyzed region.

        Returns the chunk grid origin (typically ``0``) rather than the
        first variant position -- streaming's per-chunk windows are
        anchored to that grid, so reporting the variant-based origin
        would be misleading for callers building a comparable eager
        matrix.
        """
        return self._chunks[0][0] if self._chunks else 0

    @property
    def chrom_end(self):
        """Trailing edge of the analyzed region (chunk-grid right edge).

        Reported as the chunk grid's exclusive upper bound rather than
        the last-variant-position-inclusive form HaplotypeMatrix uses
        elsewhere. The streaming path runs windowed_analysis once per
        chunk and concatenates; using the chunk grid's right edge for
        each per-chunk chrom_end keeps all interior windows at uniform
        width. A caller building a comparable eager matrix should set
        ``eager.chrom_end = stream.chrom_end`` for the same effect.
        """
        return self._chunks[-1][1] if self._chunks else 0

    @property
    def sample_sets(self):
        """Population -> hap-axis indices. Falls back to a single 'all' set
        when no pop file was resolved."""
        if self._sample_sets is None:
            return {"all": list(range(self.num_haplotypes))}
        return self._sample_sets

    @sample_sets.setter
    def sample_sets(self, value):
        self._sample_sets = value

    @property
    def haplotypes(self):
        raise NotImplementedError(
            "StreamingHaplotypeMatrix has no materialized .haplotypes "
            "array; the matrix is too big to fit eagerly, which is why "
            "from_zarr returned this class instead of HaplotypeMatrix. "
            "For pairwise / cross-window statistics over a sub-region, "
            "call .materialize(region=(lo, hi)) to get an eager "
            "HaplotypeMatrix over that slice. For per-window streaming "
            "stats, iterate .iter_gpu_chunks() directly or pass this "
            "object to a streaming-aware kernel like windowed_analysis."
        )

    def materialize(self, *, region=None, sample_subset=None):
        """Build an eager ``HaplotypeMatrix`` over a sub-region.

        Pairwise kernels (``pairwise_r2``, the r^2 heatmap path,
        ``locate_unlinked``) can't be evaluated chunk-by-chunk because
        they need every (variant, variant) pair simultaneously. Use
        this to pull the slice you want into one device-resident
        HaplotypeMatrix and run those kernels on it.

        Parameters
        ----------
        region : tuple of int, optional
            ``(left, right)`` bp interval to materialize.
            ``right`` is exclusive. ``None`` materializes the full
            mappable range, which on a biobank-scale store will OOM.
        sample_subset : sequence of int, optional
            Haplotype-axis indices to keep. ``None`` keeps every
            haplotype.

        Returns
        -------
        HaplotypeMatrix
            Eager, device-resident, ready for any pg_gpu kernel.
        """
        from ._gpu_genotype_prep import build_haplotype_matrix

        if region is None:
            left, right = self.chrom_start, self.chrom_end
        else:
            left, right = int(region[0]), int(region[1])

        if sample_subset is None:
            gt, pos = self._source.slice_region(left, right)
        else:
            # slice_subsample returns a 2-D (n_var, n_hap_subset) int8
            # block; HaplotypeMatrix wants (n_hap, n_var). Reshape on
            # the GPU via build_haplotype_matrix's same prep path by
            # promoting the 2-D subsample to a (n_var, n_dip', 2)
            # block. The subsample's ploidy ordering follows the
            # convention build_haplotype_matrix produces: haps
            # 0..n_dip-1 = ploidy 0, n_dip..2*n_dip-1 = ploidy 1.
            import numpy as _np
            gm, pos = self._source.slice_subsample(left, right, sample_subset)
            n_var, n_hap = gm.shape
            # round n_hap to even for the (n_dip, 2) reshape -- if the
            # caller picked an odd subset count we error out rather
            # than silently splitting a diploid in half.
            if n_hap % 2 != 0:
                raise ValueError(
                    f"materialize(sample_subset=...) requires an even "
                    f"count to round-trip through (n_dip, 2) layout; "
                    f"got {n_hap}."
                )
            n_dip_sub = n_hap // 2
            gt = _np.empty((n_var, n_dip_sub, 2), dtype=gm.dtype)
            gt[:, :, 0] = gm[:, :n_dip_sub]
            gt[:, :, 1] = gm[:, n_dip_sub:]

        return build_haplotype_matrix(
            gt, pos,
            chrom_start=left, chrom_end=right,
            sample_sets=self._sample_sets,
        )

    def iter_gpu_chunks(self):
        """Yield ``(left, right, HaplotypeMatrix)`` tuples covering the source.

        Each yielded HaplotypeMatrix lives on the GPU and represents one
        genomic chunk's variants on the full haplotype axis. Empty
        chunks (regions with no variants, e.g. an acrocentric arm) are
        skipped -- callers see only chunks with at least one variant.
        """
        for ci, left, right, gt, pos, t_read in self._fetcher.iter_chunks(
                self._chunks, self._prefetch):
            if gt.shape[0] == 0:
                continue
            hm = build_haplotype_matrix(
                gt, pos,
                # chrom_end is the chunk's exclusive right edge so the
                # last window in each chunk does not get clipped to the
                # last variant position the way an eager matrix would.
                chrom_start=int(left), chrom_end=int(right),
                sample_sets=self._sample_sets,
            )
            yield int(left), int(right), hm

    def __repr__(self):
        return (
            f"StreamingHaplotypeMatrix(num_variants={self.num_variants}, "
            f"num_haplotypes={self.num_haplotypes}, "
            f"chrom={self.chrom!r}, n_chunks={len(self._chunks)}, "
            f"chunk_bp={self._chunk_bp}, prefetch={self._prefetch})"
        )
