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

import numpy as np
import zarr

from .zarr_io import _parse_region, detect_zarr_layout


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

    def slice_region_gpu(self, left, right, *, gds_store):
        """Like ``slice_region`` but reads through ``gds_store``
        (a ``kvikio.zarr.GDSStore`` opened on the same path) so the
        chunk lands on the GPU via the zarr GPU buffer prototype.

        Caller is responsible for activating
        ``zarr.config.enable_gpu()`` before iterating chunks and
        restoring the CPU buffer prototype afterwards; this method
        does not toggle the global config so it can run inside a
        producer thread without racing the consumer's eager calls.
        """
        import cupy as _cp
        lo, hi = self._site_index_range(left, right)
        if hi <= lo:
            return (_cp.empty((0, self.num_diploids, 2), _cp.int8),
                    np.empty(0, np.int64))
        zlo, zhi = self._zarr_row_range(lo, hi)
        # Open a fresh group on each call -- the group caches the
        # codec pipeline against the active zarr.config snapshot, so
        # reopening picks up the GPU buffer prototype the consumer
        # enabled outside this call.
        import zarr
        cg = zarr.open_group(gds_store, mode="r")["call_genotype"]
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

    def slice_subsample(self, left, right, hap_cols):
        """Read variants in ``[left, right)`` restricted to ``hap_cols``.

        ``hap_cols`` is an iterable of haplotype-axis indices in
        ``[0, 2 * num_diploids)``. Uses zarr's ``oindex`` for the
        sample-axis subset; with bio2zarr-style sample chunking only
        the chunks containing the requested diploids are decompressed.

        Returns
        -------
        gm : ndarray, shape (n_var, len(hap_cols)), dtype int8
            ``-1`` for multiallelic rows.
        pos : ndarray, shape (n_var,), dtype int64
            Variant positions.
        """
        lo, hi = self._site_index_range(left, right)
        hap_cols = np.asarray(hap_cols, dtype=np.int64)
        if hi <= lo:
            return (np.empty((0, len(hap_cols)), np.int8),
                    np.empty(0, np.int64))
        zlo = int(self._zarr_var_indices[lo])
        zhi = int(self._zarr_var_indices[hi - 1]) + 1

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

        block = np.asarray(
            self._store["call_genotype"].oindex[zlo:zhi, unique_dips, :]
        )
        gm = block[:, inv, ploidy]
        pos = self.site_pos[lo:hi].copy()
        return gm, pos

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
        from .zarr_io import resolve_pop_file_path
        pop_file = resolve_pop_file_path(
            self.path, pop_file,
            announce_prefix="ZarrGenotypeSource",
        )
        if pop_file is None:
            return None

        # Resolve sample names to diploid indices via the store's sample_id.
        sample_ids = list(np.array(self._store["sample_id"]))
        idx_by_name = {str(s): i for i, s in enumerate(sample_ids)}

        pop_to_dips = {}
        with open(pop_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2 or parts[0] == "sample":
                    continue
                sample, pop = parts[0], parts[1]
                i = idx_by_name.get(sample)
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
