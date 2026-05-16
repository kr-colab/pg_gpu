"""Lazy view over a VCZ zarr store for chunked / subset reads.

``ZarrGenotypeSource`` opens a VCZ store and exposes the read patterns the
streaming HaplotypeMatrix path needs: ``slice_region`` for the full-haplotype
windowed-scan chunks, ``slice_subsample`` for the sample-subset reads (LD
probes, joint SFS subsamples), and ``iter_chunks`` for generating the
genomic intervals a streaming loop walks. No genotype data is materialized
at construction -- each slice call reads exactly the variants and samples
the caller asked for.

Currently supports VCZ (bio2zarr) layout only. Flat / grouped scikit-allel
layouts are detected and rejected with a clear conversion hint; callers
that want streaming on those layouts should re-encode via
``HaplotypeMatrix.vcf_to_zarr`` or ``HaplotypeMatrix.to_zarr(format='vcz')``.
"""

import warnings

import cupy as cp
import numpy as np
import zarr

from .zarr_io import _parse_region, detect_zarr_layout


#: Maximum contiguous-run count before ``slice_subsample_gpu`` falls
#: back to the host-staged path. Each run reads one zarr slab, so a
#: subset scattered across many runs spends more time on kvikio
#: dispatch than the host oindex path would.
_SLICE_SUBSAMPLE_GPU_MAX_RUNS = 64


class ZarrGenotypeSource:
    """Open a VCZ store and expose chunked + subset read primitives.

    Parameters
    ----------
    path : str
        Filesystem path to a VCZ zarr store.
    region : str, optional
        ``'chrom:start-end'`` to restrict the source to a single
        sub-region. The source is single-contig regardless: a
        multi-contig store without ``region`` raises.
    pop_file : str, optional
        Tab-delimited file mapping ``sample`` -> ``pop`` (same format as
        ``HaplotypeMatrix.load_pop_file``). Default ``None`` looks for
        ``<path>.pops.tsv`` next to the store and uses it if present;
        emits a one-line stderr note so the auto-load isn't invisible.
        Pass ``pop_file=False`` to disable the auto-load entirely.
    contig_id : str, optional
        Pick a contig by name when ``region`` is not given and the store
        has multiple contigs. Ignored otherwise.

    Attributes
    ----------
    num_variants : int
        Variants inside ``region`` (or the whole store), after contig
        selection.
    num_diploids : int
        Sample-axis length.
    num_haplotypes : int
        ``2 * num_diploids``.
    site_pos : ndarray
        Variant positions (host), length ``num_variants``.
    chrom : str
        Selected contig name.
    chunks : tuple
        ``call_genotype`` chunk shape.
    pop_cols : dict or None
        ``{pop_name: ndarray[hap_axis_indices]}`` when a pop file was
        resolved, else None.
    """

    def __init__(self, path, *, region=None, pop_file=None, contig_id=None):
        self.path = str(path)
        self._store = zarr.open_group(self.path, mode="r")

        layout = detect_zarr_layout(self._store)
        if layout != "vcz":
            raise ValueError(
                f"ZarrGenotypeSource currently supports VCZ layout only; "
                f"got {layout!r}. Re-encode via HaplotypeMatrix.vcf_to_zarr "
                f"or HaplotypeMatrix.to_zarr(format='vcz')."
            )

        contig_ids = list(np.array(self._store["contig_id"]))
        all_contigs = np.array(self._store["variant_contig"])
        all_pos = np.array(self._store["variant_position"])

        if region is not None:
            chrom, start, end = _parse_region(region)
            if chrom not in contig_ids:
                raise ValueError(
                    f"Contig {chrom!r} not found in store. "
                    f"Available: {contig_ids}"
                )
            contig_idx = contig_ids.index(chrom)
            mask = (all_contigs == contig_idx) & (all_pos >= start) & (all_pos < end)
        else:
            unique_contigs = np.unique(all_contigs)
            if len(unique_contigs) > 1 and contig_id is None:
                names = [contig_ids[i] for i in unique_contigs]
                raise ValueError(
                    f"Store contains {len(unique_contigs)} contigs: "
                    f"{names}. Pass region='chrom:start-end' or "
                    f"contig_id=... to pick one."
                )
            if contig_id is not None:
                if contig_id not in contig_ids:
                    raise ValueError(
                        f"Contig {contig_id!r} not found in store. "
                        f"Available: {contig_ids}"
                    )
                contig_idx = contig_ids.index(contig_id)
            else:
                contig_idx = int(unique_contigs[0])
            chrom = contig_ids[contig_idx]
            mask = all_contigs == contig_idx

        self._zarr_var_indices = np.where(mask)[0]
        self.site_pos = all_pos[mask]
        self.chrom = chrom

        cg = self._store["call_genotype"]
        self.chunks = tuple(cg.chunks)
        self.num_diploids = int(cg.shape[1])
        self.num_haplotypes = 2 * self.num_diploids
        self.num_variants = int(self.site_pos.size)

        self.pop_cols = self._resolve_pop_file(pop_file)

    @property
    def mappable_lo(self):
        """Position of the first variant in the source (or 0 if empty)."""
        return int(self.site_pos[0]) if self.num_variants else 0

    @property
    def mappable_hi(self):
        """One past the position of the last variant in the source."""
        return int(self.site_pos[-1]) + 1 if self.num_variants else 0

    def slice_region(self, left, right):
        """Read every haplotype for variants in ``[left, right)``.

        Returns
        -------
        gt : ndarray, shape (n_var, num_diploids, 2), dtype int8
            Raw call_genotype block, ``-1`` for multiallelic rows.
        pos : ndarray, shape (n_var,), dtype int64
            Variant positions.
        """
        lo, hi = self._site_index_range(left, right)
        if hi <= lo:
            return (np.empty((0, self.num_diploids, 2), np.int8),
                    np.empty(0, np.int64))
        zlo, zhi = self._zarr_row_range(lo, hi)
        gt = np.asarray(self._store["call_genotype"][zlo:zhi])
        pos = self.site_pos[lo:hi].copy()
        return gt, pos

    def slice_region_gpu(self, left, right, *, cg):
        """Like ``slice_region`` but reads through a caller-provided
        ``cg`` -- the ``call_genotype`` zarr array on a group opened
        from a ``kvikio.zarr.GDSStore`` with the GPU buffer prototype
        active. The chunk lands on the GPU via the active buffer
        prototype.

        ``cg`` is passed in (not opened here) because zarr caches the
        codec pipeline on the group at open time; the caller opens
        the group once after toggling ``zarr.config.enable_gpu()``
        and reuses the resulting array across every chunk in the
        iteration. Per-call opens would cost ~ms per chunk and add
        nothing.
        """
        lo, hi = self._site_index_range(left, right)
        if hi <= lo:
            return (cp.empty((0, self.num_diploids, 2), cp.int8),
                    np.empty(0, np.int64))
        zlo, zhi = self._zarr_row_range(lo, hi)
        gt = cg[zlo:zhi]
        pos = self.site_pos[lo:hi].copy()
        return gt, pos

    def _zarr_row_range(self, lo, hi):
        """Map an in-source row range ``[lo, hi)`` to the underlying
        zarr ``call_genotype`` row range, accounting for the region
        filter applied at construction."""
        zlo = int(self._zarr_var_indices[lo])
        zhi = int(self._zarr_var_indices[hi - 1]) + 1
        return zlo, zhi

    def slice_subsample_gpu(self, left, right, hap_cols, *, cg):
        """Like ``slice_subsample(to_gpu=True)`` but reads chunks
        through the caller-provided ``cg`` -- a ``call_genotype``
        zarr array on a group opened from a ``kvikio.zarr.GDSStore``
        with the GPU buffer prototype active. Decompression happens
        on the GPU (nvCOMP) rather than via the synchronous host-side
        ``oindex`` codec pipeline, which at biobank-scale sample
        subsets is minutes per call.

        Sample-axis subsetting works by grouping the requested
        diploids into contiguous runs (typically one or two for a
        population-subset read), reading each run as a contiguous
        ``cg[zlo:zhi, rlo:rhi, :]`` slab that kvikio decompresses
        directly to GPU memory, then doing one GPU fancy-index to
        assemble the ``(n_var, len(hap_cols))`` result. Pathologically
        scattered subsets (every dip its own run) fall back to
        ``slice_subsample(to_gpu=True)``.

        Returns
        -------
        gm : cupy.ndarray, shape (n_var, len(hap_cols)), dtype int8
        pos : ndarray, shape (n_var,), dtype int64
        """
        lo, hi = self._site_index_range(left, right)
        hap_cols = np.asarray(hap_cols, dtype=np.int64)
        if hi <= lo:
            return (cp.empty((0, len(hap_cols)), cp.int8),
                    np.empty(0, np.int64))
        zlo, zhi = self._zarr_row_range(lo, hi)

        is_p1 = hap_cols >= self.num_diploids
        dip_idx = np.where(is_p1, hap_cols - self.num_diploids, hap_cols)
        ploidy = is_p1.astype(np.int64)
        unique_dips, inv = np.unique(dip_idx, return_inverse=True)

        # Group sorted unique_dips into contiguous diploid-axis runs.
        # Two for a typical two-population subset, one for single-pop.
        breaks = np.where(np.diff(unique_dips) != 1)[0]
        run_lo = np.concatenate([unique_dips[:1], unique_dips[breaks + 1]])
        run_hi = np.concatenate([unique_dips[breaks] + 1, unique_dips[-1:] + 1])

        # Fall back if the subset is so scattered that per-run slab
        # reads would no longer amortize. The pair of contiguous runs
        # typical of LD probes has len(run_lo) == 2.
        if len(run_lo) > _SLICE_SUBSAMPLE_GPU_MAX_RUNS:
            return self.slice_subsample(left, right, hap_cols, to_gpu=True)

        # One contiguous slab per run via kvikio's GPU-buffer codec
        # pipeline, then concatenate. ``slab_offset`` maps each entry
        # in ``unique_dips`` to its position in the concatenated block.
        slabs = []
        slab_offset = np.zeros(unique_dips.size, dtype=np.int64)
        cum = 0
        for rlo, rhi in zip(run_lo, run_hi):
            slabs.append(cg[zlo:zhi, int(rlo):int(rhi), :])
            in_run = (unique_dips >= rlo) & (unique_dips < rhi)
            slab_offset[in_run] = unique_dips[in_run] - rlo + cum
            cum += int(rhi - rlo)
        block_gpu = (slabs[0] if len(slabs) == 1
                     else cp.concatenate(slabs, axis=1))
        del slabs

        # Final GPU fancy-index: for each requested hap_col, pick the
        # right (dip, ploidy) cell.
        gm_gpu = block_gpu[:,
                            cp.asarray(slab_offset[inv], dtype=cp.int64),
                            cp.asarray(ploidy, dtype=cp.int64)]
        del block_gpu
        pos = self.site_pos[lo:hi].copy()
        return gm_gpu, pos

    def slice_subsample(self, left, right, hap_cols, *, to_gpu=False):
        """Read variants in ``[left, right)`` restricted to ``hap_cols``.

        ``hap_cols`` is an iterable of haplotype-axis indices in
        ``[0, 2 * num_diploids)``. Uses zarr's ``oindex`` for the
        sample-axis subset; with bio2zarr-style sample chunking only
        the chunks containing the requested diploids are decompressed.

        Parameters
        ----------
        to_gpu : bool
            When ``True`` the returned ``gm`` is a cupy array on the
            GPU. The ploidy gather is done on the GPU regardless --
            this kwarg only controls whether ``gm`` gets downloaded
            back to host before returning. Callers that immediately
            upload the result (e.g. ``materialize``) should pass
            ``to_gpu=True`` to avoid the round-trip.

        Returns
        -------
        gm : ndarray (host) or cupy.ndarray (device), shape
             ``(n_var, len(hap_cols))``, dtype ``int8``
            ``-1`` for multiallelic rows.
        pos : ndarray, shape (n_var,), dtype int64
            Variant positions.
        """
        lo, hi = self._site_index_range(left, right)
        hap_cols = np.asarray(hap_cols, dtype=np.int64)
        if hi <= lo:
            empty = (cp.empty((0, len(hap_cols)), cp.int8) if to_gpu
                     else np.empty((0, len(hap_cols)), np.int8))
            return empty, np.empty(0, np.int64)
        zlo, zhi = self._zarr_row_range(lo, hi)

        # The haplotype axis indexes a flat (n_dip * 2) layout where hap j
        # = (ploidy j // n_dip, dip j % n_dip). VCZ stores (n_var, n_dip, 2)
        # so we have to translate each requested hap index into (dip, ploidy)
        # and feed unique dips through oindex (zarr's oindex on a 3-D axis
        # does not de-duplicate, and requesting the same dip twice would
        # double-decompress its sample chunk).
        is_p1 = hap_cols >= self.num_diploids
        dip_idx = np.where(is_p1, hap_cols - self.num_diploids, hap_cols)
        ploidy = is_p1.astype(np.int64)
        unique_dips, inv = np.unique(dip_idx, return_inverse=True)

        block_host = np.asarray(
            self._store["call_genotype"].oindex[zlo:zhi, unique_dips, :]
        )
        # Ploidy gather on the GPU: at biobank-scale sample subsets
        # (~10 k samples in ``inv`` over hundreds of thousands of
        # variants) the equivalent ``block[:, inv, ploidy]`` host
        # fancy-index is single-threaded numpy and takes tens of
        # seconds per Mb of variants; the parallel cupy gather is
        # sub-second.
        block_gpu = cp.asarray(block_host)
        del block_host
        gm_gpu = block_gpu[:, cp.asarray(inv, dtype=cp.int64),
                            cp.asarray(ploidy, dtype=cp.int64)]
        del block_gpu
        pos = self.site_pos[lo:hi].copy()
        if to_gpu:
            return gm_gpu, pos
        return cp.asnumpy(gm_gpu), pos

    def iter_chunks(self, chunk_bp, align_bp=None, start=None):
        """Yield ``(left, right)`` genomic intervals tiling the source.

        Intervals are sized as multiples of ``align_bp`` (defaults to
        ``chunk_bp``) so a caller running window-based stats can
        guarantee windows never straddle a chunk boundary. Empty
        regions at the start of the contig (e.g. acrocentric arm) are
        yielded as well -- the caller can detect and skip them via the
        cheap ``gt.shape[0] == 0`` check.
        """
        if align_bp is None:
            align_bp = chunk_bp
        windows_per_chunk = max(1, chunk_bp // align_bp)
        step = windows_per_chunk * align_bp
        end = self.mappable_hi
        s = 0 if start is None else int(start)
        if end == 0:
            return
        while s < end:
            yield s, min(s + step, end)
            s += step

    def _site_index_range(self, left, right):
        """Half-open ``[lo, hi)`` row range covering positions in [left, right)."""
        lo = int(np.searchsorted(self.site_pos, left, side="left"))
        hi = int(np.searchsorted(self.site_pos, right, side="left"))
        return lo, hi

    def _resolve_pop_file(self, pop_file):
        from .zarr_io import normalize_pop_input

        # Defer the sample_id lookup until we actually need it -- some
        # test fixtures write minimal VCZ stores without a sample_id
        # array, and a missing companion .pops.tsv should not need it.
        if "sample_id" in self._store:
            sample_ids = list(np.array(self._store["sample_id"]))
        else:
            sample_ids = []
        pop_map = normalize_pop_input(
            pop_file, zarr_path=self.path,
            sample_names=sample_ids,
            zarr_store=self._store,
            announce_prefix="ZarrGenotypeSource",
        )
        if pop_map is None:
            return None

        idx_by_name = {str(s): i for i, s in enumerate(sample_ids)}
        pop_to_dips = {}
        for sample, pop in pop_map.items():
            i = idx_by_name.get(str(sample))
            if i is None:
                warnings.warn(
                    f"pop_file sample {sample!r} not in store; skipping",
                    stacklevel=2,
                )
                continue
            pop_to_dips.setdefault(pop, []).append(i)

        # Map diploid indices to haplotype-axis indices: dip i lives at
        # haps i (ploidy 0) and i + num_diploids (ploidy 1). Matches the
        # layout build_haplotype_matrix produces and load_pop_file expects.
        n_dip = self.num_diploids
        pop_cols = {}
        for pop, dips in pop_to_dips.items():
            dips = np.asarray(sorted(dips), dtype=np.int64)
            pop_cols[pop] = np.concatenate([dips, dips + n_dip])
        return pop_cols
