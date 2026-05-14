import cupy as cp
import numpy as np
import allel
import tskit
from collections import Counter, OrderedDict

from .accessible import AccessibleMask, bed_to_mask, resolve_accessible_mask


#: Fraction of free GPU memory the eager matrix is allowed to consume
#: before ``streaming='auto'`` falls back to streaming. 0.5 leaves room
#: for the haplotype matrix plus the working memory each statistic kernel
#: needs (windowed scatter buffers, pairwise outputs, etc.).
STREAMING_AUTO_EAGER_FRACTION = 0.5


def _decide_streaming_mode(zarr_path, region, streaming, pop_file,
                           free_gpu_bytes=None,
                           fraction=STREAMING_AUTO_EAGER_FRACTION):
    """Pick ``'eager'`` vs ``'streaming'`` based on the projected matrix size.

    Returns ``(mode, source)`` where ``source`` is the
    ``ZarrGenotypeSource`` opened for the size probe and reused
    downstream (when ``mode == 'streaming'``), or ``None`` when no
    source was opened (scikit-allel layout, missing store, or the
    eager-fits branch where the caller's eager path opens its own
    store via ``read_genotypes``).

    Scikit-allel layouts always return ``('eager', None)``: the
    streaming source is VCZ-only, so the size check would refuse a
    large allel store with nowhere to fall back to.

    ``free_gpu_bytes`` is exposed for tests; production callers leave
    it None to let ``cp.cuda.Device().mem_info`` provide it.
    """
    import zarr
    from .zarr_io import detect_zarr_layout
    try:
        store = zarr.open_group(zarr_path, mode="r")
        layout = detect_zarr_layout(store)
    except (FileNotFoundError, KeyError, ValueError):
        # Store path is missing, missing required arrays, or has an
        # unrecognized layout. Defer to the eager path so its richer
        # error messages (read_genotypes, layout-specific) surface.
        return "eager", None
    if layout != "vcz":
        return "eager", None

    from .zarr_source import ZarrGenotypeSource
    source = ZarrGenotypeSource(zarr_path, region=region,
                                pop_file=False)
    eager_bytes = int(source.num_variants) * int(source.num_haplotypes)

    if free_gpu_bytes is None:
        import cupy as _cp
        free_gpu_bytes = int(_cp.cuda.Device().mem_info[0])

    if eager_bytes <= fraction * free_gpu_bytes:
        return "eager", None
    if streaming == "never":
        raise MemoryError(
            f"streaming='never' but the eager matrix would be "
            f"{eager_bytes / 1e9:.1f} GB on a device with "
            f"{free_gpu_bytes / 1e9:.1f} GB free (limit: "
            f"{fraction:.0%}). Use streaming='auto' or 'always' to "
            f"return a StreamingHaplotypeMatrix instead."
        )
    return "streaming", source


def _resolve_companion_pop_file(zarr_path, pop_file):
    """Wrapper around ``zarr_io.resolve_pop_file_path`` tagged for the
    ``HaplotypeMatrix.from_zarr`` auto-load announce message."""
    from .zarr_io import resolve_pop_file_path
    return resolve_pop_file_path(zarr_path, pop_file,
                                 announce_prefix="HaplotypeMatrix.from_zarr")


class HaplotypeMatrix:
    """Haplotype matrix for population genetics analysis.

    Stores phased haplotype data with variant positions and population
    labels. Supports GPU-accelerated computation of diversity, divergence,
    selection, and LD statistics.

    Parameters
    ----------
    genotypes : ndarray, shape (n_haplotypes, n_variants)
        Haplotype data (0/1 with -1 for missing).
    positions : ndarray, shape (n_variants,)
        Variant positions.
    chrom_start, chrom_end : int, optional
        Chromosome boundaries for span normalization.
    sample_sets : dict, optional
        Maps population names to lists of haplotype indices.
    n_total_sites : int, optional
        Total callable sites for span normalization.
    samples : list, optional
        Diploid sample names (from VCF).
    accessible_mask : AccessibleMask, optional
        Genome accessibility mask.
    """
    def __init__(self,
                 genotypes,
                 positions,
                 chrom_start: int = None,
                 chrom_end: int = None,
                 sample_sets: dict = None,
                 n_total_sites: int = None,
                 samples: list = None,
                 accessible_mask=None,
                ):
        if genotypes.size == 0:
            raise ValueError("genotypes cannot be empty")
        if positions.size == 0:
            raise ValueError("positions cannot be empty")
        if not isinstance(genotypes, (np.ndarray, cp.ndarray)):
            raise ValueError("genotypes must be a numpy or cupy array")
        if not isinstance(positions, (np.ndarray, cp.ndarray)):
            raise ValueError("positions must be a numpy or cupy array")

        if isinstance(genotypes, cp.ndarray):
            self._device = 'GPU'
            if isinstance(positions, np.ndarray):
                positions = cp.array(positions)
        else:
            self._device = 'CPU'
            if isinstance(positions, cp.ndarray):
                positions = positions.get()

        self._haplotypes = genotypes
        self._positions = positions
        self._accessible_idx = None
        self._hap_filtered = None
        self._pos_filtered = None
        self._accessible_mask = None
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self._sample_sets = sample_sets
        self.n_total_sites = n_total_sites
        self.samples = samples  # diploid sample names from VCF

        if accessible_mask is not None and not isinstance(accessible_mask, AccessibleMask):
            accessible_mask = resolve_accessible_mask(
                accessible_mask, chrom_start, chrom_end)
        self.accessible_mask = accessible_mask
        if self._accessible_mask is not None and self.n_total_sites is None:
            if chrom_start is not None and chrom_end is not None:
                self.n_total_sites = self._accessible_mask.count_accessible(
                    chrom_start, chrom_end + 1)
            else:
                self.n_total_sites = self._accessible_mask.total_accessible

    @property
    def haplotypes(self):
        if self._accessible_idx is None:
            return self._haplotypes
        if self._hap_filtered is None:
            self._hap_filtered = self._haplotypes[:, self._accessible_idx]
        return self._hap_filtered

    @haplotypes.setter
    def haplotypes(self, value):
        self._haplotypes = value
        self._hap_filtered = None

    @property
    def positions(self):
        if self._accessible_idx is None:
            return self._positions
        if self._pos_filtered is None:
            self._pos_filtered = self._positions[self._accessible_idx]
        return self._pos_filtered

    @positions.setter
    def positions(self, value):
        self._positions = value
        self._pos_filtered = None

    @property
    def accessible_mask(self):
        return self._accessible_mask

    @accessible_mask.setter
    def accessible_mask(self, mask):
        self._accessible_mask = mask
        if mask is not None:
            pos = self._positions.get() if isinstance(self._positions, cp.ndarray) \
                else np.asarray(self._positions)
            keep = mask.is_accessible_at(pos.astype(int))
            if keep.all():
                self._accessible_idx = None
            else:
                xp = cp if self._device == 'GPU' else np
                self._accessible_idx = xp.asarray(np.where(keep)[0])
        else:
            self._accessible_idx = None
        self._hap_filtered = None
        self._pos_filtered = None

    @property
    def device(self):
        """Returns the current device (CPU or GPU)."""
        return self._device

    @property
    def sample_sets(self):
        """
        Defines groups of haplotypes that belong to populations.

        Returns:
            dict: A dictionary mapping population names to lists of haplotype indices.
                  If _sample_sets was not specified at construction, returns a default
                  dictionary with a single key 'all' containing all haplotype indices.
        """
        if self._sample_sets is None:
            # All haplotypes belong to a single population labeled "all"
            return {"all": list(range(self.haplotypes.shape[0]))}
        return self._sample_sets

    @sample_sets.setter
    def sample_sets(self, sample_sets: dict):
        """
        Set the sample sets.
        """
        if not isinstance(sample_sets, dict):
            raise ValueError("sample_sets must be a dictionary")
        # check that the values are lists
        for key, value in sample_sets.items():
            if not isinstance(value, list):
                raise ValueError("values in sample_sets must be lists")
        self._sample_sets = sample_sets

    @property
    def has_accessible_mask(self):
        """Whether an accessible site mask is attached."""
        return self.accessible_mask is not None

    def set_accessible_mask(self, mask_or_path, chrom=None):
        """Attach an accessible site mask (non-destructive).

        The mask filters which variants are visible through the haplotypes
        and positions properties. Original data is preserved and a different
        mask can be applied later. Returns self for chaining.

        Parameters
        ----------
        mask_or_path : str, path-like, numpy.ndarray, or AccessibleMask
            If a string/path, treated as a BED file path defining accessible
            regions. If an ndarray, wrapped as an AccessibleMask with offset
            equal to chrom_start. If already an AccessibleMask, assigned
            directly.
        chrom : str, optional
            Chromosome name to extract from the BED file. Required when
            mask_or_path is a BED file path.

        Returns
        -------
        self
            For method chaining.
        """
        # The property setter handles _accessible_idx and cache invalidation
        self.accessible_mask = resolve_accessible_mask(
            mask_or_path, self.chrom_start, self.chrom_end, chrom)
        # n_total_sites is the BED-accessible-base count within the
        # analysis range [chrom_start, chrom_end]. The mask itself may
        # span more (we widen it to avoid losing bases inside the range
        # when chrom_end < BED max), but only in-range bases count toward
        # the per-base normalization denominator.
        if self.chrom_start is not None and self.chrom_end is not None:
            self.n_total_sites = self.accessible_mask.count_accessible(
                self.chrom_start, self.chrom_end + 1)
        else:
            self.n_total_sites = self.accessible_mask.total_accessible
        return self

    def remove_accessible_mask(self):
        """Remove the accessible mask, restoring all original variants.

        Returns self for chaining.
        """
        self.accessible_mask = None  # setter clears _accessible_idx and caches
        self.n_total_sites = None
        return self

    @property
    def has_invariant_info(self):
        """Whether invariant site information is available for span normalization."""
        return self.n_total_sites is not None

    @property
    def n_callable_sites(self):
        """Total callable sites in the analysis universe.

        Alias for ``n_total_sites``. When an accessible mask is set, this is
        the BED span (mask.total_accessible). When no mask is set but the
        matrix was loaded with ``include_invariant=True``, this is the matrix
        length at construction. Otherwise None.

        This is the denominator for per-base span normalization, and
        ``n_callable_sites = n_segregating_sites + n_invariant_sites``.
        """
        return self.n_total_sites

    @property
    def n_segregating_sites(self):
        """Number of polymorphic sites in the matrix.

        Counts sites where 0 < derived_count < n_valid (i.e. polymorphic
        among observed haplotypes), with at least 2 valid samples.
        """
        xp = cp if self.device == 'GPU' else np
        hap = self.haplotypes
        valid_mask = hap >= 0
        hap_clean = xp.where(valid_mask, hap, 0)
        derived_counts = xp.sum(hap_clean, axis=0)
        n_valid = xp.sum(valid_mask, axis=0)
        is_variant = (derived_counts > 0) & (derived_counts < n_valid) & (n_valid >= 2)
        return int(xp.sum(is_variant))

    @property
    def n_invariant_sites(self):
        """Number of invariant sites in the callable span, or None if unknown.

        Computed as ``n_callable_sites - n_segregating_sites``. The matrix may
        physically contain fewer than ``n_callable_sites`` rows (e.g. for a
        variants-only VCF the matrix has only the polymorphic rows and the
        rest are implied invariants from the accessible-mask span); in that
        case the returned count includes those implied invariants.

        Note that ``num_variants`` (matrix row count) is generally not equal
        to ``n_segregating_sites`` and not equal to ``n_callable_sites`` --
        it is just the physical matrix length.
        """
        if self.n_total_sites is None:
            return None
        return self.n_total_sites - self.n_segregating_sites

    def transfer_to_gpu(self):
        """Transfer data from CPU to GPU."""
        if self.device == 'CPU':
            self._haplotypes = cp.asarray(self._haplotypes)
            self._positions = cp.asarray(self._positions)
            if self._accessible_idx is not None:
                self._accessible_idx = cp.asarray(self._accessible_idx)
            self._hap_filtered = None
            self._pos_filtered = None
            self._device = 'GPU'

    def transfer_to_cpu(self):
        """Transfer data from GPU to CPU."""
        if self.device == 'GPU':
            self._haplotypes = np.asarray(self._haplotypes.get())
            self._positions = np.asarray(self._positions.get())
            if self._accessible_idx is not None:
                self._accessible_idx = np.asarray(self._accessible_idx.get())
            self._hap_filtered = None
            self._pos_filtered = None
            self._device = 'CPU'

    @classmethod
    def from_vcf(cls, path: str, region: str = None,
                 samples: list = None, include_invariant: bool = False,
                 accessible_bed: str = None):
        """Construct a HaplotypeMatrix from a VCF file.

        Parameters
        ----------
        path : str
            Path to VCF/BCF file (optionally gzipped + tabix-indexed).
        region : str, optional
            Genomic region to load, e.g. 'chr1:1000000-2000000'.
            Requires the VCF to be bgzipped and tabix-indexed.
        samples : list of str, optional
            Subset of samples to load. If None, loads all samples.
        include_invariant : bool
            If True, set n_total_sites from the loaded variant count.
        accessible_bed : str, optional
            Path to a BED file defining accessible/callable regions.

        Returns
        -------
        HaplotypeMatrix
            Phased haplotype data with sample names stored.
        """
        from ._biobank_warning import _maybe_biobank_warn
        _maybe_biobank_warn(path, region=region)
        vcf = allel.read_vcf(path, region=region, samples=samples)
        if vcf is None:
            raise ValueError(f"No variants found in {path}"
                             + (f" for region {region}" if region else ""))

        genotypes = allel.GenotypeArray(vcf['calldata/GT'])
        num_variants, num_samples, ploidy = genotypes.shape
        assert ploidy == 2

        haplotypes = np.empty((num_variants, 2 * num_samples), dtype=genotypes.dtype)
        haplotypes[:, :num_samples] = genotypes[:, :, 0]
        haplotypes[:, num_samples:] = genotypes[:, :, 1]
        haplotypes = haplotypes.T

        positions = np.array(vcf['variants/POS'])
        sample_names = list(vcf['samples'])

        # When a region is given, use its bounds (1-based inclusive) so the
        # analysis range matches what the user asked for, not just the
        # span between the first and last observed variants. This matters
        # for span normalization with an accessible mask: the denominator
        # is BED accessible bases within [chrom_start, chrom_end].
        chrom_start = int(positions[0])
        chrom_end = int(positions[-1])
        chrom = None
        if region is not None:
            chrom_part, _, range_part = region.partition(':')
            chrom = chrom_part or None
            if range_part:
                start_str, _, end_str = range_part.partition('-')
                if start_str:
                    chrom_start = int(start_str.replace(',', ''))
                if end_str:
                    chrom_end = int(end_str.replace(',', ''))
        elif 'variants/CHROM' in vcf:
            chrom = vcf['variants/CHROM'][0]

        n_total_sites = num_variants if include_invariant else None
        hm = cls(haplotypes, positions, chrom_start, chrom_end,
                 n_total_sites=n_total_sites, samples=sample_names)
        if accessible_bed is not None:
            hm.set_accessible_mask(accessible_bed, chrom=chrom)
        return hm

    @classmethod
    def from_zarr(cls, path: str, region: str = None,
                  accessible_bed: str = None,
                  pop_file=None,
                  streaming: str = "auto",
                  chunk_bp: int = 1_500_000,
                  prefetch: int = 1,
                  backend: str = "auto"):
        """Construct a HaplotypeMatrix from a Zarr store.

        Supports both VCZ (bio2zarr) and scikit-allel zarr layouts.
        Layout is auto-detected. For multi-chromosome stores, region
        must be specified.

        Parameters
        ----------
        path : str
            Path to Zarr store directory.
        region : str, optional
            Genomic region 'chrom:start-end' to load a subset.
        accessible_bed : str, optional
            Path to a BED file defining accessible/callable regions.
        pop_file : str or False, optional
            Tab-delimited file mapping ``sample`` -> ``pop`` in the
            format ``HaplotypeMatrix.load_pop_file`` expects. Default
            (``None``) looks for ``<path>.pops.tsv`` next to the
            store and loads it if present, announcing the auto-load
            to stderr. Pass ``False`` to disable the auto-load.
        streaming : {'auto', 'always', 'never'}, optional
            Whether to return a ``StreamingHaplotypeMatrix`` that
            iterates the store chunk-by-chunk through the GPU
            (suitable for biobank-scale stores that do not fit
            eagerly on the device). ``'always'`` forces streaming;
            ``'never'`` forces eager (and raises ``MemoryError`` if
            the matrix would not fit in free GPU memory). ``'auto'``
            (default) checks the projected eager footprint against
            free GPU memory and picks streaming when the eager
            matrix would consume more than half the device.
            Scikit-allel layouts always route to eager because the
            streaming source is VCZ-only.
        chunk_bp : int, optional
            Genomic chunk size in bp for the streaming path. Ignored
            on the eager path.
        prefetch : int, optional
            Read-ahead depth for the streaming path. Ignored on the
            eager path.
        backend : {'auto', 'host', 'kvikio'}, optional
            Streaming chunk-fetch backend. ``'kvikio'`` decodes the
            store's codec on the GPU via ``kvikio + nvCOMP``; only
            works when ``call_genotype``'s codec is in the
            nvCOMP-supported list (zstd, blosc, lz4, deflate) and
            the sample-axis chunking is bio2zarr-shaped (sample chunk
            smaller than the full sample axis). ``'host'`` is the
            host-buffer fallback. ``'auto'`` (default) picks
            ``'kvikio'`` when both conditions hold and warns +
            ``'host'`` when chunks are whole-sample-axis (the kvikio
            path gives no speedup at that chunking).

        Returns
        -------
        HaplotypeMatrix or StreamingHaplotypeMatrix
        """
        if streaming not in ("auto", "always", "never"):
            raise ValueError(
                f"streaming must be 'auto', 'always', or 'never'; "
                f"got {streaming!r}"
            )
        if backend not in ("auto", "host", "kvikio"):
            raise ValueError(
                f"backend must be 'auto', 'host', or 'kvikio'; "
                f"got {backend!r}"
            )

        if streaming == "always":
            return cls._build_streaming(
                path, region=region, pop_file=pop_file,
                chunk_bp=chunk_bp, prefetch=prefetch,
                backend=backend,
            )

        # 'auto' and 'never' both want eager when the matrix fits; 'auto'
        # falls back to streaming when it doesn't, 'never' raises. The
        # decision needs the matrix's projected size, which is only
        # available via ZarrGenotypeSource (VCZ-only). Scikit-allel
        # stores always route to eager because there is no streaming
        # source for that layout.
        choice, source = _decide_streaming_mode(path, region=region,
                                                streaming=streaming,
                                                pop_file=pop_file)
        if choice == "streaming":
            return cls._build_streaming(
                path, region=region, pop_file=pop_file,
                chunk_bp=chunk_bp, prefetch=prefetch,
                source=source, backend=backend,
            )
        return cls._build_eager(path, region=region,
                                accessible_bed=accessible_bed,
                                pop_file=pop_file)

    @classmethod
    def _build_eager(cls, path, *, region, accessible_bed, pop_file):
        from .zarr_io import read_genotypes
        from ._gpu_genotype_prep import build_haplotype_matrix

        data = read_genotypes(path, region)
        gt = data['gt']
        positions = data['positions']
        sample_names = data['samples']

        if gt.shape[2] != 2:
            raise ValueError(f"expected ploidy 2; got {gt.shape[2]}")

        hm = build_haplotype_matrix(
            gt, positions,
            chrom_start=int(positions[0]),
            chrom_end=int(positions[-1]),
            samples=list(sample_names) if sample_names is not None else None,
        )

        chrom = region.split(':')[0] if region else None
        if accessible_bed is not None:
            hm.set_accessible_mask(accessible_bed, chrom=chrom)

        resolved_pop = _resolve_companion_pop_file(path, pop_file)
        if resolved_pop is not None:
            hm.load_pop_file(resolved_pop)

        return hm

    @classmethod
    def _build_streaming(cls, path, *, region, pop_file, chunk_bp, prefetch,
                         source=None, backend="auto"):
        from .streaming_matrix import (
            HostChunkFetcher, KvikioChunkFetcher, StreamingHaplotypeMatrix,
            _pick_chunk_fetcher,
        )
        from .zarr_source import ZarrGenotypeSource

        if source is None:
            source = ZarrGenotypeSource(path, region=region, pop_file=pop_file)
        else:
            # _decide_streaming_mode opens the source with pop_file=False
            # to keep its size probe cheap; resolve the caller's pop_file
            # now without re-opening the zarr store.
            source.pop_cols = source._resolve_pop_file(pop_file)
        fetcher = _pick_chunk_fetcher(source, backend=backend)
        return StreamingHaplotypeMatrix(
            source, fetcher,
            chunk_bp=chunk_bp, prefetch=prefetch,
        )

    def to_zarr(self, zarr_path: str, format: str = 'vcz',
                contig_name: str = None):
        """Save haplotype data to Zarr format.

        Parameters
        ----------
        zarr_path : str
            Output Zarr store path.
        format : str
            ``'vcz'`` (default) for bio2zarr-compatible VCZ layout,
            ``'scikit-allel'`` for legacy layout.
        contig_name : str, optional
            Chromosome/contig name to store in VCZ format.
        """
        from .zarr_io import write_vcz, write_allel

        hap = self.haplotypes if isinstance(self.haplotypes, np.ndarray) \
            else self.haplotypes.get()
        pos = self.positions if isinstance(self.positions, np.ndarray) \
            else self.positions.get()
        gt = self._haplotypes_to_gt(hap)

        if format == 'vcz':
            write_vcz(zarr_path, gt, pos, self.samples,
                      contig_name=contig_name)
        elif format == 'scikit-allel':
            write_allel(zarr_path, gt, pos, self.samples)
        else:
            raise ValueError(
                f"Unknown format: {format!r}. Use 'vcz' or 'scikit-allel'."
            )

    @staticmethod
    def vcf_to_zarr(vcf_paths, zarr_path, worker_processes=None,
                    icf_path=None, max_memory='4GB', show_progress=True):
        """Convert VCF file(s) to VCZ-format Zarr store using bio2zarr.

        Parameters
        ----------
        vcf_paths : str or list of str
            Path(s) to VCF/BCF files (bgzipped + indexed).
        zarr_path : str
            Output Zarr store path.
        worker_processes : int, optional
            Number of worker processes. Defaults to os.cpu_count().
        icf_path : str, optional
            Directory for intermediate columnar format files.
            Defaults to ``<zarr_path>.icf_tmp`` (same filesystem as output).
            ICF files can be 1-3x the VCF size.
        max_memory : str
            Maximum memory for encoding step. Default '4GB'.
        show_progress : bool
            Show progress bars. Default True.
        """
        from .zarr_io import vcf_to_zarr
        vcf_to_zarr(vcf_paths, zarr_path,
                     worker_processes=worker_processes,
                     icf_path=icf_path, max_memory=max_memory,
                     show_progress=show_progress)

    @staticmethod
    def _haplotypes_to_gt(hap):
        """Convert (n_hap, n_var) haplotype matrix back to (n_var, n_samples, 2) GT array."""
        n_hap, n_var = hap.shape
        n_samples = n_hap // 2
        gt = np.empty((n_var, n_samples, 2), dtype=hap.dtype)
        gt[:, :, 0] = hap[:n_samples, :].T
        gt[:, :, 1] = hap[n_samples:, :].T
        return gt

    def load_pop_file(self, pop_file: str, pops: list = None):
        """Load population assignments from a tab-delimited file.

        Sets sample_sets from a file mapping sample names to populations.
        Requires that sample names were stored during from_vcf().

        Parameters
        ----------
        pop_file : str
            Tab-delimited file with columns: sample, pop.
            Header line starting with 'sample' is skipped.
        pops : list of str, optional
            Populations to include. If None, includes all found populations.
        """
        if self.samples is None:
            raise ValueError("No sample names stored. Use from_vcf() to load data.")

        n_samples = len(self.samples)
        pop_map = {}
        with open(pop_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] != 'sample':
                    pop_map[parts[0]] = parts[1]

        # Build sample_sets
        found_pops = set(pop_map.values())
        if pops is None:
            pops = sorted(found_pops)

        pop_sets = {p: [] for p in pops}
        for i, name in enumerate(self.samples):
            pop = pop_map.get(name)
            if pop in pop_sets:
                pop_sets[pop].append(i)
                pop_sets[pop].append(i + n_samples)

        self.sample_sets = pop_sets

    @classmethod
    def from_ts(cls, ts: tskit.TreeSequence, device: str = 'CPU',
                include_invariant: bool = False,
                accessible_bed: str = None,
                chrom: str = None) -> 'HaplotypeMatrix':
        """
        Create a HaplotypeMatrix from a tskit.TreeSequence.

        Args:
            ts: A tskit.TreeSequence object
            device: 'CPU' or 'GPU'
            include_invariant: If True, set n_total_sites from the sequence
                length so that calculations can account for
                invariant sites analytically (no extra rows stored).
            accessible_bed: Path to a BED file defining accessible regions.
            chrom: Chromosome name for BED file filtering.

        Returns:
            HaplotypeMatrix: A new HaplotypeMatrix instance

        Notes:
            Populations declared in the tree sequence (with a name in
            metadata) are automatically registered in ``sample_sets``:
            each pop name maps to the haplotype row indices of its
            samples. The no-demography case (``sim_ancestry(samples=N)``
            with a single unnamed population) leaves ``sample_sets``
            unset. Users who need custom groupings can overwrite
            ``hm.sample_sets`` after construction.
        """
        # Convert ts to haplotype matrix
        haplotypes = ts.genotype_matrix().T
        positions = ts.tables.sites.position
        # tskit uses 0-based exclusive ends (positions in [0, sequence_length))
        # while pg_gpu treats chrom_end as 1-based inclusive. Subtract 1 so
        # span = chrom_end - chrom_start + 1 == sequence_length.
        chrom_start = 0
        chrom_end = int(ts.sequence_length) - 1
        if device == 'GPU':
            # Convert to CuPy arrays
            haplotypes = cp.array(haplotypes)
            positions = cp.array(positions)

        n_total_sites = int(ts.sequence_length) if include_invariant else None
        hm = cls(haplotypes, positions, chrom_start, chrom_end,
                 n_total_sites=n_total_sites)
        if accessible_bed is not None:
            hm.set_accessible_mask(accessible_bed, chrom=chrom)

        sample_sets = {}
        sample_idx = {s: i for i, s in enumerate(ts.samples())}
        for pop in ts.populations():
            name = pop.metadata.get("name") if pop.metadata else None
            if name is None:
                continue
            nodes = ts.samples(population=pop.id)
            if len(nodes):
                sample_sets[name] = [sample_idx[s] for s in nodes]
        if sample_sets:
            hm.sample_sets = sample_sets

        return hm

    def get_matrix(self) -> cp.ndarray:
        """
        Returns the haplotype matrix.

        Returns:
            cp.ndarray: The array representing the haplotype/genotype matrix.
        """
        return self.haplotypes

    def get_positions(self) -> cp.ndarray:
        """
        Returns the variant positions.

        Returns:
            cp.ndarray: The array of positions.
        """
        return self.positions

    @property
    def shape(self):
        """
        Returns the shape of the haplotype matrix.

        Returns:
            tuple: A tuple representing the dimensions (variants, samples)
                   of the haplotype matrix.
        """
        return self.haplotypes.shape

    @property
    def num_variants(self):
        """
        Returns the number of variants in the haplotype matrix.
        """
        return self.haplotypes.shape[1]

    @property
    def num_haplotypes(self):
        """
        Returns the number of haplotypes in the haplotype matrix.
        """
        return self.haplotypes.shape[0]

    def __repr__(self):
        first_pos = self.positions[0] if self.positions.size > 0 else None
        last_pos = self.positions[-1] if self.positions.size > 0 else None
        return (f"HaplotypeMatrix(shape={self.shape}, "
                f"first_position={first_pos}, last_position={last_pos})")

    def get_subset(self, positions) -> "HaplotypeMatrix":
        """
        Get a subset of the haplotype matrix based on the provided positions.

        Parameters:
            positions: A one-dimensional array of indices to select from the haplotype matrix.
                       This can be either a NumPy array or a CuPy array.

        Returns:
            HaplotypeMatrix: A new instance containing the subset of the haplotype matrix.
        """
        # Ensure positions is one-dimensional
        if positions.ndim != 1:
            raise ValueError("Positions must be a one-dimensional array.")

        # Convert positions to match the device of the haplotype matrix.
        if self.device == 'CPU' and isinstance(positions, cp.ndarray):
            positions = cp.asnumpy(positions)
        elif self.device == 'GPU' and isinstance(positions, np.ndarray):
            positions = cp.array(positions)

        # Validate that positions are valid indices.
        # Ensure positions are valid indices and convert to integer type
        positions = cp.asarray(positions, dtype=np.int64) if self.device == 'GPU' else np.asarray(positions, dtype=np.int64)

        # Handle empty positions array
        if len(positions) == 0:
            # Create empty subset maintaining the same structure
            # Need to create arrays that have non-zero size to satisfy constructor
            if self.device == 'GPU':
                empty_haplotypes = cp.empty((self.haplotypes.shape[0], 0), dtype=self.haplotypes.dtype)
                empty_positions = cp.array([], dtype=self.positions.dtype)
            else:
                empty_haplotypes = np.empty((self.haplotypes.shape[0], 0), dtype=self.haplotypes.dtype)
                empty_positions = np.array([], dtype=self.positions.dtype)

            # For empty subsets, bypass constructor validation
            result = object.__new__(HaplotypeMatrix)
            result._haplotypes = empty_haplotypes
            result._positions = empty_positions
            result._accessible_idx = None
            result._hap_filtered = None
            result._pos_filtered = None
            result.chrom_start = self.chrom_start
            result.chrom_end = self.chrom_end
            result._sample_sets = self._sample_sets
            result._device = self._device
            result.n_total_sites = self.n_total_sites
            result.accessible_mask = None
            result.samples = self.samples
            return result

        if not (positions >= 0).all() or not (positions < self.haplotypes.shape[1]).all():
            raise ValueError("Positions must be valid indices within the haplotype matrix.")

        subset_haplotypes = self.haplotypes[:, positions]
        subset_positions = self.positions[positions]

        # Create and return a new instance, maintaining the device state and sample sets.
        # Don't propagate accessible_mask -- child data is already filtered.
        return HaplotypeMatrix(
            subset_haplotypes,
            subset_positions,
            sample_sets=self._sample_sets,
            n_total_sites=self.n_total_sites,
        )

    def get_subset_from_range(self, low: int, high: int) -> "HaplotypeMatrix":
        """
        Get a subset of the haplotype matrix based on a range of positions.

        Parameters:
            low (int): The lower bound of the range (inclusive).
            high (int): The upper bound of the range (exclusive).

        Returns:
            HaplotypeMatrix: A new instance containing the subset of the haplotype matrix.
        """
        # Validate range
        if low < 0 or high > self.positions.size or low >= high:
            raise ValueError("Invalid range specified")

        # Check device and find indices of positions within the specified range
        positions = cp.asarray(self.positions) if self.device == 'GPU' else np.asarray(self.positions)
        indices = cp.where((positions >= low) & (positions < high))[0] if self.device == 'GPU' else np.where((positions >= low) & (positions < high))[0]

        # Create the subset of haplotypes based on the found indices
        sliced_mask = None
        if self.accessible_mask is not None:
            sliced_mask = self.accessible_mask.slice(low, high)
        return HaplotypeMatrix(
            self.haplotypes[:, indices],
            self.positions[indices],
            chrom_start=low,
            chrom_end=high,
            sample_sets=self._sample_sets,
            accessible_mask=sliced_mask,
        )

    def apply_biallelic_filter(self) -> "HaplotypeMatrix":
        """
        Apply biallelic filter to remove variants that are not strictly biallelic.

        This filter matches the behavior of moments' get_genotypes function, which uses
        is_biallelic_01() to remove variants that:
        1. Have more than 2 alleles present in the data
        2. Don't have both reference (0) and alternate (1) alleles present

        This is the actual filtering that moments does by default, not an AC filter.

        Returns:
            HaplotypeMatrix: A new HaplotypeMatrix instance with filtered variants.

        Note:
            This replicates moments' is_biallelic_01() filtering behavior.
        """
        if self.device == 'GPU':
            xp = cp
        else:
            xp = np

        # For biallelic filtering, we need to check across ALL haplotypes
        # Count alleles for each variant across all samples
        n_variants = self.num_variants

        # Count occurrences of each allele value (ignoring missing = -1)
        alt_count = xp.sum(self.haplotypes == 1, axis=0)
        ref_count = xp.sum(self.haplotypes == 0, axis=0)
        multiallelic_count = xp.sum(self.haplotypes >= 2, axis=0)

        # A variant is biallelic if:
        # 1. No multiallelic alleles (2+) are present
        # 2. Both reference (0) and alternate (1) alleles are present
        # Missing data (-1) is ignored — a site with only 0, 1, and -1 is biallelic
        is_biallelic = (multiallelic_count == 0) & (ref_count > 0) & (alt_count > 0)

        keep_mask = is_biallelic

        # Get indices of variants to keep
        keep_indices = xp.where(keep_mask)[0]

        # Create filtered HaplotypeMatrix
        filtered_haplotypes = self.haplotypes[:, keep_indices]
        filtered_positions = self.positions[keep_indices]

        # Update chromosome boundaries if needed
        if len(keep_indices) > 0:
            new_chrom_start = int(filtered_positions[0].get()) if self.device == 'GPU' else int(filtered_positions[0])
            new_chrom_end = int(filtered_positions[-1].get()) if self.device == 'GPU' else int(filtered_positions[-1])
        else:
            new_chrom_start = self.chrom_start
            new_chrom_end = self.chrom_end

        # Create new instance with same sample sets
        filtered_matrix = HaplotypeMatrix(
            filtered_haplotypes,
            filtered_positions,
            chrom_start=new_chrom_start,
            chrom_end=new_chrom_end,
            sample_sets=self._sample_sets,
            n_total_sites=self.n_total_sites,
            accessible_mask=self.accessible_mask,
        )

        return filtered_matrix

    ####### Missing data methods #######
    def is_missing(self, axis=None):
        """
        Detect missing calls (-1 values).

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute. If None, returns boolean array same shape as haplotypes.
            If 0, returns missing per variant. If 1, returns missing per sample.

        Returns
        -------
        missing : array
            Boolean array indicating missing data
        """
        if self.device == 'GPU':
            missing = self.haplotypes < 0
            if axis is not None:
                return cp.any(missing, axis=axis)
            return missing
        else:
            missing = self.haplotypes < 0
            if axis is not None:
                return np.any(missing, axis=axis)
            return missing

    def is_called(self, axis=None):
        """
        Detect valid (non-missing) calls.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute. If None, returns boolean array same shape as haplotypes.
            If 0, returns called per variant. If 1, returns called per sample.

        Returns
        -------
        called : array
            Boolean array indicating valid data
        """
        return ~self.is_missing(axis=axis)

    def count_missing(self, axis=None):
        """
        Count missing calls per variant or sample.

        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, returns total count.
            If 0, returns count per variant. If 1, returns count per sample.

        Returns
        -------
        count : int or array
            Count of missing calls
        """
        missing = self.is_missing()
        if self.device == 'GPU':
            return cp.sum(missing, axis=axis)
        else:
            return np.sum(missing, axis=axis)

    def count_called(self, axis=None):
        """
        Count valid calls per variant or sample.

        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, returns total count.
            If 0, returns count per variant. If 1, returns count per sample.

        Returns
        -------
        count : int or array
            Count of valid calls
        """
        called = self.is_called()
        if self.device == 'GPU':
            return cp.sum(called, axis=axis)
        else:
            return np.sum(called, axis=axis)

    def get_span(self, mode='auto'):
        """
        Get the genomic span for normalization calculations.

        Parameters
        ----------
        mode : str
            'auto' - Use best available: accessible mask > n_total_sites
            > total genomic span > callable span
            'accessible' - Use accessible base count from mask (error if
            no mask set)
            'per_base' - Use total genomic span (chrom_end - chrom_start)
            'per_variant' - Use number of variant sites
            'callable' - Use span from first to last variant position
            'total' - Alias for 'per_base' (backward compatibility)
            'sites' - Alias for 'per_variant' (backward compatibility)

        Returns
        -------
        span : int
            The span to use for normalization
        """
        # chrom_start and chrom_end are 1-based inclusive throughout pg_gpu;
        # count_accessible() is half-open, so pass end + 1 to include
        # chrom_end itself, and the inclusive count is end - start + 1.
        # The denominator is BED accessible bases within the analysis range
        # [chrom_start, chrom_end], matching scikit-allel's
        # sequence_diversity(is_accessible=..., start=..., stop=...).
        if mode == 'auto':
            if self.accessible_mask is not None:
                start = self.chrom_start if self.chrom_start is not None else 0
                end = self.chrom_end if self.chrom_end is not None else start
                return self.accessible_mask.count_accessible(start, end + 1)
            if self.n_total_sites is not None:
                return self.n_total_sites
            if self.chrom_start is not None and self.chrom_end is not None:
                return self.chrom_end - self.chrom_start + 1
            mode = 'callable'

        if mode == 'accessible':
            if self.accessible_mask is not None:
                start = self.chrom_start if self.chrom_start is not None else 0
                end = self.chrom_end if self.chrom_end is not None else start
                return self.accessible_mask.count_accessible(start, end + 1)
            raise ValueError(
                "mode='accessible' requires an accessible mask. "
                "Use set_accessible_mask() first.")

        if mode in ('per_base', 'total'):
            if self.chrom_start is not None and self.chrom_end is not None:
                return self.chrom_end - self.chrom_start + 1
            mode = 'callable'

        if mode in ('per_variant', 'sites'):
            return self.num_variants

        if mode == 'callable':
            if self.device == 'GPU':
                if len(self.positions) > 0:
                    return int((cp.max(self.positions) - cp.min(self.positions)).get()) + 1
                else:
                    return 0
            else:
                if len(self.positions) > 0:
                    return int(np.max(self.positions) - np.min(self.positions)) + 1
                else:
                    return 0

        raise ValueError(f"Invalid span mode: {mode}")

    def exclude_missing_sites(self, populations=None):
        """Return a new matrix with only sites that have no missing data.

        Parameters
        ----------
        populations : list of (str or list), optional
            If given, only require completeness within these populations.
            Each element is a population name (looked up in sample_sets)
            or a list of sample indices.

        Returns
        -------
        HaplotypeMatrix
            Filtered matrix. Returns self if no missing data.

        Raises
        ------
        ValueError
            If no sites remain after filtering.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        hap = self.haplotypes
        if populations is not None:
            idx = []
            for pop in populations:
                if isinstance(pop, str):
                    idx.extend(self.sample_sets[pop])
                else:
                    idx.extend(list(pop))
            has_missing = cp.any(hap[idx] < 0, axis=0)
        else:
            has_missing = cp.any(hap < 0, axis=0)

        valid = cp.where(~has_missing)[0]
        if len(valid) == hap.shape[1]:
            return self
        if len(valid) == 0:
            return self.get_subset(valid)
        return self.get_subset(valid)

    def filter_variants_by_missing(self, max_missing_freq=0.1):
        """
        Return a new HaplotypeMatrix with variants filtered by missing data frequency.

        Parameters
        ----------
        max_missing_freq : float
            Maximum allowed frequency of missing data per variant

        Returns
        -------
        filtered : HaplotypeMatrix
            New HaplotypeMatrix with filtered variants
        """
        missing_freq = self.count_missing(axis=0) / self.num_haplotypes
        if self.device == 'GPU':
            valid_mask = missing_freq <= max_missing_freq
            valid_indices = cp.where(valid_mask)[0]
            return self.get_subset(valid_indices)
        else:
            valid_mask = missing_freq <= max_missing_freq
            valid_indices = np.where(valid_mask)[0]
            return self.get_subset(valid_indices)

    def summarize_missing_data(self):
        """
        Get summary statistics about missing data patterns.

        Returns
        -------
        summary : dict
            Dictionary with missing data statistics
        """
        total_missing = self.count_missing()
        total_calls = self.num_haplotypes * self.num_variants
        missing_per_variant = self.count_missing(axis=0)
        missing_per_sample = self.count_missing(axis=1)

        if self.device == 'GPU':
            return {
                'total_missing_calls': int(total_missing.get()),
                'total_calls': total_calls,
                'missing_freq_overall': float((total_missing / total_calls).get()),
                'variants_with_no_missing': int(cp.sum(missing_per_variant == 0).get()),
                'samples_with_no_missing': int(cp.sum(missing_per_sample == 0).get()),
                'max_missing_per_variant': int(cp.max(missing_per_variant).get()),
                'max_missing_per_sample': int(cp.max(missing_per_sample).get())
            }
        else:
            return {
                'total_missing_calls': int(total_missing),
                'total_calls': total_calls,
                'missing_freq_overall': float(total_missing / total_calls),
                'variants_with_no_missing': int(np.sum(missing_per_variant == 0)),
                'samples_with_no_missing': int(np.sum(missing_per_sample == 0)),
                'max_missing_per_variant': int(np.max(missing_per_variant)),
                'max_missing_per_sample': int(np.max(missing_per_sample))
            }

    ####### some polymorphism statistics #######
    def allele_frequency_spectrum(self) -> cp.ndarray:
        """
        Calculate the allele frequency spectrum for a haplotype matrix.

        Note: This method is deprecated. Use diversity.allele_frequency_spectrum() instead.
        """
        from . import diversity
        return diversity.allele_frequency_spectrum(self)

    def diversity(self, span_normalize=True) -> float:
        """
        Calculate the nucleotide diversity (π) for the haplotype matrix.

        Note: This method is deprecated. Use diversity.pi() instead.

        Parameters:
            span_normalize (bool, optional): If True, the result is normalized by the span of the haplotype matrix. Defaults to True.

        Returns:
            float: The nucleotide diversity (π) for the haplotype matrix.
        """
        from . import diversity
        return diversity.pi(self, span_normalize=span_normalize)

    def watersons_theta(self, span_normalize=True) -> float:
        """
        Calculate Waterson's theta for the haplotype matrix.

        Note: This method is deprecated. Use diversity.theta_w() instead.
        """
        from . import diversity
        return diversity.theta_w(self, span_normalize=span_normalize)

    def Tajimas_D(self) -> float:
        """
        Calculate Tajima's D for the haplotype matrix.

        Note: This method is deprecated. Use diversity.tajimas_d() instead.
        """
        from . import diversity
        return diversity.tajimas_d(self)


    def _pairwise_ld_core(self, hap_clean=None, valid_mask=None):
        """Shared computation for pairwise LD methods.

        Computes allele frequencies, joint frequencies, and D matrix from
        haplotype data, handling missing values.

        Parameters
        ----------
        hap_clean : cupy.ndarray, optional
            Pre-cleaned haplotype submatrix (missing set to 0). If None,
            uses self.haplotypes.
        valid_mask : cupy.ndarray, optional
            Validity mask (1 where not missing). Must be provided iff
            hap_clean is provided.

        Returns
        -------
        D : cupy.ndarray, shape (m, m)
            Pairwise D = p_AB - p_A*p_B.
        p : cupy.ndarray, shape (m,)
            Per-site allele frequencies.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        if hap_clean is None:
            hap = self.haplotypes
            valid_mask = (hap >= 0).astype(cp.float64)
            hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)

        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        p = cp.where(n_valid > 0, cp.sum(hap_clean, axis=0) / n_valid, 0.0)

        joint_n = valid_mask.T @ valid_mask
        joint_11 = hap_clean.T @ hap_clean
        p_AB = cp.where(joint_n > 0, joint_11 / joint_n, 0.0)

        D = p_AB - cp.outer(p, p)
        return D, p

    def pairwise_LD_v(self) -> cp.ndarray:
        """Pairwise linkage disequilibrium (D statistic) via matrix multiply."""
        D, _ = self._pairwise_ld_core()
        cp.fill_diagonal(D, 0)
        return D

    def pairwise_r2(self, estimator: str = 'r2') -> cp.ndarray:
        """Pairwise r-squared for all variant pairs.

        Parameters
        ----------
        estimator : str
            ``'r2'`` (default) -- naive haplotype r² from
            frequency-based formula. ``'rogers_huff'`` -- Rogers-Huff
            r² computed on diploid 0/1/2 dosages obtained by pairing
            adjacent haplotypes (sample 0 = haplotypes 0,1; sample 1
            = haplotypes 2,3; ...). Matches
            :func:`scikit-allel.rogers_huff_r` ** 2.

        Returns
        -------
        cupy.ndarray, float64, shape (n_variants, n_variants)
        """
        if estimator == 'rogers_huff':
            from .ld_statistics import _rogers_huff_pairwise_r
            r_full = _rogers_huff_pairwise_r(self)
            r2 = r_full ** 2
            r2 = cp.where(cp.isnan(r2), 0.0, r2)
            cp.fill_diagonal(r2, 0)
            return r2
        if estimator != 'r2':
            raise ValueError(
                f"Unknown estimator: {estimator!r} "
                f"(expected 'r2' or 'rogers_huff')")
        D, p = self._pairwise_ld_core()
        denom_squared = cp.outer(p * (1 - p), p * (1 - p))
        r2 = cp.where(denom_squared > 0, (D ** 2) / denom_squared, 0)
        cp.fill_diagonal(r2, 0)
        return r2

    def locate_unlinked(self, size=100, step=20, threshold=0.1):
        """Locate variants in approximate linkage equilibrium.

        Uses a sliding window approach to identify variants whose r-squared
        with all other variants in the window is below the threshold.

        Parameters
        ----------
        size : int
            Window size (number of variants).
        step : int
            Number of variants to advance between windows.
        threshold : float
            Maximum r-squared value to consider variants unlinked.

        Returns
        -------
        ndarray, bool, shape (n_variants,)
            True for variants in approximate linkage equilibrium.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        m = self.num_variants
        hap = self.haplotypes
        valid_mask = (hap >= 0).astype(cp.float64)
        hap_clean = cp.where(hap >= 0, hap, 0).astype(cp.float64)

        # pruning state kept on CPU to avoid per-scalar GPU transfers
        loc = np.ones(m, dtype=bool)

        for w_start in range(0, m, step):
            w_end = min(w_start + size, m)

            active = loc[w_start:w_end]
            if np.sum(active) <= 1:
                continue

            active_idx = np.where(active)[0] + w_start
            active_idx_gpu = cp.asarray(active_idx)

            D, p_w = self._pairwise_ld_core(
                hap_clean[:, active_idx_gpu],
                valid_mask[:, active_idx_gpu],
            )
            denom = cp.outer(p_w * (1 - p_w), p_w * (1 - p_w))
            r2_mat = cp.where(denom > 0, (D ** 2) / denom, 0.0)
            cp.fill_diagonal(r2_mat, 0.0)

            # prune on CPU to avoid per-scalar GPU transfers
            r2_mat_cpu = r2_mat.get()
            n_active = len(active_idx)
            for i in range(n_active):
                if not loc[active_idx[i]]:
                    continue
                for j in range(i + 1, n_active):
                    if not loc[active_idx[j]]:
                        continue
                    if r2_mat_cpu[i, j] > threshold:
                        loc[active_idx[j]] = False

        return loc

    def windowed_r_squared(self, bp_bins, percentile=50, pop=None,
                           estimator: str = 'r2'):
        """Compute percentiles of r-squared in genomic distance bins.

        Parameters
        ----------
        bp_bins : array_like
            Bin edges for genomic distances in base pairs.
        percentile : float or array_like
            Percentile(s) to compute within each bin.
        pop : str, optional
            Population key to use.
        estimator : str
            ``'r2'`` (default) -- naive haplotype r² from frequency
            counts. ``'rogers_huff'`` -- Rogers-Huff r² on diploid
            0/1/2 dosages obtained by pairing adjacent haplotypes;
            matches :func:`scikit-allel.rogers_huff_r` ** 2.

        Returns
        -------
        result : ndarray, shape (n_bins,) or (n_bins, n_percentiles)
            Percentile(s) of r-squared per bin.
        counts : ndarray, int, shape (n_bins,)
            Number of variant pairs per bin.
        """
        if self.device == 'CPU':
            self.transfer_to_gpu()

        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)

        m = self.num_variants

        if estimator == 'rogers_huff':
            if pop is not None:
                raise NotImplementedError(
                    "windowed_r_squared(estimator='rogers_huff', pop=...) "
                    "is not yet implemented; pass the full matrix or "
                    "subset to the desired haplotypes first.")
            from pg_gpu.ld_statistics import _rogers_huff_pairwise_r
            r_full = _rogers_huff_pairwise_r(self)
            iu = cp.triu_indices(m, k=1)
            r2_vals = (r_full[iu]) ** 2
        elif estimator == 'r2':
            # compute counts and r² via tally
            counts_arr, n_valid = self.tally_gpu_haplotypes(pop=pop)
            from pg_gpu import ld_statistics
            r2_vals = ld_statistics.r_squared(counts_arr, n_valid=n_valid)
        else:
            raise ValueError(
                f"Unknown estimator: {estimator!r} "
                f"(expected 'r2' or 'rogers_huff')")

        # pair distances
        idx_i, idx_j = cp.triu_indices(m, k=1)
        distances = pos[idx_j] - pos[idx_i]

        bp_bins_cp = cp.array(bp_bins)
        n_bins = len(bp_bins) - 1
        bin_inds = cp.digitize(distances, bp_bins_cp) - 1

        valid_mask = (bin_inds >= 0) & (bin_inds < n_bins) & ~cp.isnan(r2_vals)

        # transfer to CPU for percentile computation
        r2_cpu = r2_vals[valid_mask].get()
        bins_cpu = bin_inds[valid_mask].get()

        percentile = np.atleast_1d(percentile)
        result = np.full((n_bins, len(percentile)), np.nan)
        pair_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = bins_cpu == i
            pair_counts[i] = int(np.sum(mask))
            if pair_counts[i] > 0:
                for p_idx, pct in enumerate(percentile):
                    result[i, p_idx] = np.percentile(r2_cpu[mask], pct)

        if result.shape[1] == 1:
            result = result[:, 0]

        return result, pair_counts


    def tally_gpu_haplotypes(self, pop=None):
        """
        GPU implementation of computing pairwise haplotype tallies.
        Automatically detects and handles missing data if present.

        Parameters:
            pop (str, optional): Population key from sample_sets to use. If None, uses all samples.

        Returns:
            tuple: (counts, n_valid) where:
                - counts: Array of shape (#pairs, 4) containing [n11, n10, n01, n00] for each variant pair
                - n_valid: Array of shape (#pairs,) containing the number of valid haplotypes for each pair
                          or None if no missing data is present
        """
        # Ensure data is on the GPU
        if self.device == 'CPU':
            self.transfer_to_gpu()

        # Get the appropriate subset of haplotypes
        if pop is not None:
            if self._sample_sets is None:
                raise ValueError("sample_sets must be defined to use pop parameter")
            if pop not in self._sample_sets:
                raise KeyError(f"Population key {pop} must exist in sample_sets")
            X = self.haplotypes[self._sample_sets[pop], :]
        else:
            X = self.haplotypes

        # Check if there's any missing data
        has_missing = cp.any(X == -1)

        if has_missing:
            # Use the missing data implementation
            return self._tally_gpu_haplotypes_with_missing_impl(X)
        else:
            # Use the faster non-missing implementation
            m = X.shape[1]  # number of variants

            # Count ones per variant
            ones_per_variant = cp.sum(X, axis=0)

            # Compute n11 matrix
            n11_mat = X.T @ X

            # Get indices for upper triangle
            idx_i, idx_j = cp.triu_indices(m, k=1)

            # Compute counts
            n11_pairs = n11_mat[idx_i, idx_j]
            n10_pairs = ones_per_variant[idx_i] - n11_pairs
            n01_pairs = ones_per_variant[idx_j] - n11_pairs
            n00_pairs = X.shape[0] - (n11_pairs + n10_pairs + n01_pairs)

            # Stack all results
            counts = cp.stack([n11_pairs, n10_pairs, n01_pairs, n00_pairs], axis=1)

            return counts, None

    def _tally_gpu_haplotypes_with_missing_impl(self, X):
        """
        Internal implementation of computing pairwise haplotype tallies with missing data support.

        For each variant pair, only counts haplotypes where both variants are non-missing.
        Missing data is encoded as -1 in the haplotype matrix.

        Parameters:
            X (cp.ndarray): Haplotype matrix to process

        Returns:
            tuple: (counts, n_valid) where:
                - counts: Array of shape (#pairs, 4) containing [n11, n10, n01, n00] for each variant pair
                - n_valid: Array of shape (#pairs,) containing the number of valid haplotypes for each pair
        """

        m = X.shape[1]  # number of variants
        n_haps = X.shape[0]  # number of haplotypes

        # Create missing mask for each variant (True where data is missing)
        missing_mask = (X == -1)

        # Get indices for upper triangle
        idx_i, idx_j = cp.triu_indices(m, k=1)
        n_pairs = len(idx_i)

        # Initialize arrays for results
        n11_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n10_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n01_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n00_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        n_valid = cp.zeros(n_pairs, dtype=cp.int32)

        # Process pairs (this could be optimized with custom kernels)
        for pair_idx in range(n_pairs):
            i = idx_i[pair_idx]
            j = idx_j[pair_idx]

            # Create valid mask for this pair (where both variants are non-missing)
            valid_mask = ~(missing_mask[:, i] | missing_mask[:, j])
            n_valid[pair_idx] = cp.sum(valid_mask)

            if n_valid[pair_idx] > 0:
                # Extract valid haplotypes for this pair
                valid_haps_i = X[valid_mask, i]
                valid_haps_j = X[valid_mask, j]

                # Count haplotype combinations
                n11_pairs[pair_idx] = cp.sum((valid_haps_i == 1) & (valid_haps_j == 1))
                n10_pairs[pair_idx] = cp.sum((valid_haps_i == 1) & (valid_haps_j == 0))
                n01_pairs[pair_idx] = cp.sum((valid_haps_i == 0) & (valid_haps_j == 1))
                n00_pairs[pair_idx] = cp.sum((valid_haps_i == 0) & (valid_haps_j == 0))

        # Stack all results
        counts = cp.stack([n11_pairs, n10_pairs, n01_pairs, n00_pairs], axis=1)

        return counts, n_valid

    def tally_gpu_haplotypes_two_pops_with_missing(self, pop1: str, pop2: str):
        """
        GPU implementation of computing pairwise haplotype tallies for two populations with missing data support.

        For each variant pair, only counts haplotypes where both variants are non-missing in both populations.
        Missing data is encoded as -1 in the haplotype matrix.

        Parameters:
            pop1 (str): First population key from sample_sets
            pop2 (str): Second population key from sample_sets

        Returns:
            tuple: (counts, n_valid1, n_valid2) where:
                - counts: Array of shape (#pairs, 8) containing counts for both populations
                  [n11_1, n10_1, n01_1, n00_1, n11_2, n10_2, n01_2, n00_2]
                - n_valid1: Array of shape (#pairs,) with valid haplotypes for pop1
                - n_valid2: Array of shape (#pairs,) with valid haplotypes for pop2
        """
        import cupy as cp

        if self.device == 'CPU':
            self.transfer_to_gpu()

        # Check populations
        if self._sample_sets is None:
            raise ValueError("sample_sets must be defined to use this function")
        if pop1 not in self._sample_sets or pop2 not in self._sample_sets:
            raise KeyError(f"Population keys {pop1} and {pop2} must exist in sample_sets")

        # Get indices for each population
        idx1 = self._sample_sets[pop1]
        idx2 = self._sample_sets[pop2]

        # Extract submatrices for each population
        X1 = self.haplotypes[idx1, :]
        X2 = self.haplotypes[idx2, :]
        m = self.num_variants

        # Create missing masks for each population
        missing_mask1 = (X1 == -1)
        missing_mask2 = (X2 == -1)

        # Get indices for upper triangle
        idx_i, idx_j = cp.triu_indices(m, k=1)
        n_pairs = len(idx_i)

        # Initialize arrays for results
        counts = cp.zeros((n_pairs, 8), dtype=cp.int32)
        n_valid1 = cp.zeros(n_pairs, dtype=cp.int32)
        n_valid2 = cp.zeros(n_pairs, dtype=cp.int32)

        # Process pairs (this could be optimized with custom kernels)
        for pair_idx in range(n_pairs):
            i = idx_i[pair_idx]
            j = idx_j[pair_idx]

            # Create valid masks for each population
            valid_mask1 = ~(missing_mask1[:, i] | missing_mask1[:, j])
            valid_mask2 = ~(missing_mask2[:, i] | missing_mask2[:, j])
            n_valid1[pair_idx] = cp.sum(valid_mask1)
            n_valid2[pair_idx] = cp.sum(valid_mask2)

            # Population 1 counts
            if n_valid1[pair_idx] > 0:
                valid_haps1_i = X1[valid_mask1, i]
                valid_haps1_j = X1[valid_mask1, j]
                counts[pair_idx, 0] = cp.sum((valid_haps1_i == 1) & (valid_haps1_j == 1))  # n11
                counts[pair_idx, 1] = cp.sum((valid_haps1_i == 1) & (valid_haps1_j == 0))  # n10
                counts[pair_idx, 2] = cp.sum((valid_haps1_i == 0) & (valid_haps1_j == 1))  # n01
                counts[pair_idx, 3] = cp.sum((valid_haps1_i == 0) & (valid_haps1_j == 0))  # n00

            # Population 2 counts
            if n_valid2[pair_idx] > 0:
                valid_haps2_i = X2[valid_mask2, i]
                valid_haps2_j = X2[valid_mask2, j]
                counts[pair_idx, 4] = cp.sum((valid_haps2_i == 1) & (valid_haps2_j == 1))  # n11
                counts[pair_idx, 5] = cp.sum((valid_haps2_i == 1) & (valid_haps2_j == 0))  # n10
                counts[pair_idx, 6] = cp.sum((valid_haps2_i == 0) & (valid_haps2_j == 1))  # n01
                counts[pair_idx, 7] = cp.sum((valid_haps2_i == 0) & (valid_haps2_j == 0))  # n00

        return counts, n_valid1, n_valid2

    def tally_gpu_haplotypes_two_pops(self, pop1: str, pop2: str):
        """
        GPU version of tallying haplotype counts between all pairs of variants for two populations.
        Automatically detects and handles missing data if present.

        Returns:
            tuple: (counts, n_valid1, n_valid2) where:
                - counts: Array of shape (#pairs, 8) containing counts for both populations
                - n_valid1: Array of shape (#pairs,) with valid haplotypes for pop1 (or None if no missing data)
                - n_valid2: Array of shape (#pairs,) with valid haplotypes for pop2 (or None if no missing data)
        """
        import cupy as cp

        if self.device == 'CPU':
            self.transfer_to_gpu()

        # Check populations
        if self._sample_sets is None:
            raise ValueError("sample_sets must be defined to use this function")
        if pop1 not in self._sample_sets or pop2 not in self._sample_sets:
            raise KeyError(f"Population keys {pop1} and {pop2} must exist in sample_sets")

        # Get indices for each population
        idx1 = self._sample_sets[pop1]
        idx2 = self._sample_sets[pop2]

        # Extract submatrices for each population
        X1 = self.haplotypes[idx1, :]
        X2 = self.haplotypes[idx2, :]

        # Check if there's any missing data
        has_missing = cp.any(self.haplotypes == -1)

        if has_missing:
            # Use the missing data implementation
            return self.tally_gpu_haplotypes_two_pops_with_missing(pop1, pop2)
        else:
            # Use the faster non-missing implementation
            n1 = len(idx1)
            n2 = len(idx2)
            m = self.num_variants

            # Count ones per variant for each population
            ones_per_variant1 = cp.sum(X1, axis=0)
            ones_per_variant2 = cp.sum(X2, axis=0)

            # Compute n11 matrices for each population
            n11_mat1 = X1.T @ X1
            n11_mat2 = X2.T @ X2

            # Get indices for upper triangle only
            idx_i, idx_j = cp.triu_indices(m, k=1)

            # Compute counts for population 1
            n11_pairs1 = n11_mat1[idx_i, idx_j]
            n10_pairs1 = ones_per_variant1[idx_i] - n11_pairs1
            n01_pairs1 = ones_per_variant1[idx_j] - n11_pairs1
            n00_pairs1 = n1 - (n11_pairs1 + n10_pairs1 + n01_pairs1)

            # Compute counts for population 2
            n11_pairs2 = n11_mat2[idx_i, idx_j]
            n10_pairs2 = ones_per_variant2[idx_i] - n11_pairs2
            n01_pairs2 = ones_per_variant2[idx_j] - n11_pairs2
            n00_pairs2 = n2 - (n11_pairs2 + n10_pairs2 + n01_pairs2)

            # Stack all results
            counts = cp.stack([
                n11_pairs1, n10_pairs1, n01_pairs1, n00_pairs1,
                n11_pairs2, n10_pairs2, n01_pairs2, n00_pairs2
            ], axis=1)

            return counts, None, None

    # TODO: this is not correct
    def compute_ld_statistics_gpu_single_pop(
        self,
        bp_bins,
        raw=False,
        ac_filter=True,
        chunk_size='auto'
    ):
        """
        GPU-based LD statistics computation for a single population.

        Computes DD, Dz, and pi2 statistics for variant pairs binned by distance.
        Only processes pairs within max(bp_bins) distance for memory efficiency.

        Parameters
        ----------
        bp_bins : array-like
            Array of bin boundaries in base pairs. Pairs are binned by distance
            into intervals [bp_bins[i], bp_bins[i+1]).
        raw : bool, optional
            If True, return raw sums of statistics across pairs in each bin.
            If False (default), return means.
        ac_filter : bool, optional
            If True (default), apply biallelic filtering before computation.
        chunk_size : int or 'auto', optional
            Number of pairs to process per chunk. If 'auto' (default),
            automatically estimates optimal size based on available GPU memory.
            Can specify an integer for manual control.

        Returns
        -------
        dict
            Dictionary mapping (bin_start, bin_end) tuples to tuples of statistics.
            Each bin contains (DD, Dz, pi2) values.

        Examples
        --------
        >>> bp_bins = [0, 10000, 50000, 100000]
        >>> stats = hm.compute_ld_statistics_gpu_single_pop(bp_bins)
        >>> stats[(0.0, 10000.0)]  # (DD, Dz, pi2) for first bin
        """
        if ac_filter:
            filtered_self = self.apply_biallelic_filter()
            return filtered_self.compute_ld_statistics_gpu_single_pop(
                bp_bins=bp_bins, raw=raw, ac_filter=False, chunk_size=chunk_size
            )
        if self.device == 'CPU':
            self.transfer_to_gpu()

        bp_bins_arr = np.array(bp_bins)
        max_dist = float(bp_bins_arr[-1])
        n_bins = len(bp_bins_arr) - 1
        bp_bins_cp = cp.array(bp_bins_arr)
        if chunk_size == 'auto':
            chunk_size = _estimate_ld_chunk_size(self.num_haplotypes)

        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)
        bin_sums = cp.zeros((n_bins, 3), dtype=cp.float64)
        bin_counts = cp.zeros(n_bins, dtype=cp.float64)
        _accumulate_pair_bins(
            self.haplotypes, pos, bp_bins_cp, n_bins,
            max_dist, int(chunk_size), n_tail=0,
            bin_sums=bin_sums, bin_counts=bin_counts,
            pop1_indices=None, pop2_indices=None,
        )
        return _format_ld_single_pop(bp_bins_arr, bin_sums, bin_counts, raw)


    def compute_ld_statistics_gpu_two_pops(
        self,
        bp_bins,
        pop1: str,
        pop2: str,
        raw=False,
        ac_filter=True,
        chunk_size='auto'
    ):
        """
        GPU-based LD statistics computation for two populations.

        Computes DD, Dz, and pi2 statistics for variant pairs binned by distance.
        Only processes pairs within max(bp_bins) distance for memory efficiency.

        Parameters
        ----------
        bp_bins : array-like
            Array of bin boundaries in base pairs. Pairs are binned by distance
            into intervals [bp_bins[i], bp_bins[i+1]).
        pop1 : str
            Name of first population (must exist in sample_sets)
        pop2 : str
            Name of second population (must exist in sample_sets)
        raw : bool, optional
            If True, return raw sums of statistics across pairs in each bin.
            If False (default), return means.
        ac_filter : bool, optional
            If True (default), apply biallelic filtering before computation.
        chunk_size : int or 'auto', optional
            Number of pairs to process per chunk. If 'auto' (default),
            automatically estimates optimal size based on available GPU memory.
            Can specify an integer for manual control.

        Returns
        -------
        dict
            Dictionary mapping (bin_start, bin_end) tuples to OrderedDict of statistics.
            Each bin contains 15 statistics:
            - DD_0_0, DD_0_1, DD_1_1 (D squared)
            - Dz_0_0_0, Dz_0_0_1, Dz_0_1_1, Dz_1_0_0, Dz_1_0_1, Dz_1_1_1
            - pi2_0_0_0_0, pi2_0_0_0_1, pi2_0_0_1_1, pi2_0_1_0_1, pi2_0_1_1_1, pi2_1_1_1_1

        Examples
        --------
        >>> bp_bins = [0, 10000, 50000, 100000]
        >>> stats = hm.compute_ld_statistics_gpu_two_pops(bp_bins, 'pop1', 'pop2')
        >>> stats[(0.0, 10000.0)]['DD_0_0']  # D^2 for pop1 in first bin
        """
        if ac_filter:
            filtered_self = self.apply_biallelic_filter()
            return filtered_self.compute_ld_statistics_gpu_two_pops(
                bp_bins=bp_bins, pop1=pop1, pop2=pop2, raw=raw,
                ac_filter=False, chunk_size=chunk_size
            )
        if self.device == 'CPU':
            self.transfer_to_gpu()

        bp_bins_arr = np.array(bp_bins)
        max_dist = float(bp_bins_arr[-1])
        n_bins = len(bp_bins_arr) - 1
        bp_bins_cp = cp.array(bp_bins_arr)
        pop1_indices = self._sample_sets[pop1]
        pop2_indices = self._sample_sets[pop2]
        if chunk_size == 'auto':
            chunk_size = _estimate_ld_chunk_size(
                max(len(pop1_indices), len(pop2_indices))
            )

        pos = self.positions
        if not isinstance(pos, cp.ndarray):
            pos = cp.array(pos)
        bin_sums = cp.zeros((n_bins, 15), dtype=cp.float64)
        bin_counts = cp.zeros(n_bins, dtype=cp.float64)
        _accumulate_pair_bins(
            self.haplotypes, pos, bp_bins_cp, n_bins,
            max_dist, int(chunk_size), n_tail=0,
            bin_sums=bin_sums, bin_counts=bin_counts,
            pop1_indices=pop1_indices, pop2_indices=pop2_indices,
        )
        return _format_ld_two_pops(bp_bins_arr, bin_sums, bin_counts, raw)


# =============================================================================
# LD pair-bin accumulator: shared by eager and streaming paths
# =============================================================================
#
# Per-chunk bin sums are sum-reducible: pair counts (n11/n10/n01/n00) at
# a variant pair are independent of every other pair, and the per-bin
# DD / Dz / pi2 numerators are polynomials in those counts, so totals
# decompose by pair set. The streaming path takes advantage of this by
# carrying a tail of the last ``max_bp_dist`` of variants from the
# previous chunk; pairs whose both endpoints fall inside that tail are
# masked out, since they were summed on the previous chunk's pass.

def _new_tail(stitched_haps, stitched_pos, max_bp_dist):
    """Carry forward variants within ``max_bp_dist`` of the right edge
    for the next chunk's stitch. Tail size is bounded by
    ``max_bp_dist`` worth of variants regardless of chunk width."""
    if stitched_pos.size == 0:
        return None, None
    cutoff = stitched_pos[-1] - max_bp_dist
    keep = stitched_pos > cutoff
    return stitched_haps[:, keep], stitched_pos[keep]


def _stitch_with_tail(chunk_haps, chunk_pos, tail_haps, tail_pos):
    """Return ``(stitched_haps, stitched_pos, n_tail)`` for the pair
    iteration. ``n_tail`` is the number of leading variants drawn from
    the previous chunk's tail; the pair mask keys off this offset."""
    if tail_haps is None or tail_haps.shape[1] == 0:
        return chunk_haps, chunk_pos, 0
    stitched_haps = cp.concatenate([tail_haps, chunk_haps], axis=1)
    stitched_pos = cp.concatenate([tail_pos, chunk_pos])
    return stitched_haps, stitched_pos, tail_haps.shape[1]


def _missing_flag(haps, pop_indices):
    """Whether the (optionally pop-subset) haplotype matrix contains any
    missing data. Cached once per chunk so ``compute_counts_for_pairs``
    can skip its per-batch full-matrix reduction."""
    if pop_indices is None:
        return bool(cp.any(haps == -1))
    if isinstance(pop_indices, list):
        pop_indices = cp.array(pop_indices, dtype=cp.int32)
    return bool(cp.any(haps[pop_indices] == -1))


def _accumulate_pair_bins(
    haps, pos, bp_bins_cp, n_bins, max_dist, chunk_size, n_tail,
    bin_sums, bin_counts, *, pop1_indices, pop2_indices,
):
    """Walk pair batches on ``haps`` / ``pos``, drop already-counted
    (tail, tail) pairs, and scatter-add per-pair statistics into
    ``bin_sums`` / ``bin_counts``. ``pop2_indices=None`` selects the
    single-population path; otherwise both pops feed the two-population
    batch kernel."""
    from pg_gpu import ld_statistics
    two_pop = pop2_indices is not None
    has_miss_p1 = _missing_flag(haps, pop1_indices)
    has_miss_p2 = _missing_flag(haps, pop2_indices) if two_pop else False
    for chunk_idx_i, chunk_idx_j in _iter_pairs_within_distance(
            pos, max_dist, chunk_size):
        if n_tail > 0:
            keep = ~((chunk_idx_i < n_tail) & (chunk_idx_j < n_tail))
            chunk_idx_i = chunk_idx_i[keep]
            chunk_idx_j = chunk_idx_j[keep]
            if chunk_idx_i.size == 0:
                continue
        distances = pos[chunk_idx_j] - pos[chunk_idx_i]
        chunk_bin_inds = cp.digitize(distances, bp_bins_cp) - 1
        del distances

        if two_pop:
            counts1, n_valid1 = _compute_counts_for_pairs(
                haps, chunk_idx_i, chunk_idx_j, pop1_indices,
                has_missing=has_miss_p1,
            )
            counts2, n_valid2 = _compute_counts_for_pairs(
                haps, chunk_idx_i, chunk_idx_j, pop2_indices,
                has_missing=has_miss_p2,
            )
            chunk_stats = _compute_two_pop_statistics_batch(
                counts1, counts2, n_valid1, n_valid2, ld_statistics
            )
        else:
            counts, n_valid = _compute_counts_for_pairs(
                haps, chunk_idx_i, chunk_idx_j, pop1_indices,
                has_missing=has_miss_p1,
            )
            chunk_stats = _compute_single_pop_statistics_batch(
                counts, n_valid, ld_statistics
            )

        valid_mask = (chunk_bin_inds >= 0) & (chunk_bin_inds < n_bins)
        valid_bin_inds = chunk_bin_inds[valid_mask]
        valid_stats = chunk_stats[valid_mask]
        for s in range(chunk_stats.shape[1]):
            cp.add.at(bin_sums[:, s], valid_bin_inds, valid_stats[:, s])
        cp.add.at(
            bin_counts, valid_bin_inds,
            cp.ones(len(valid_bin_inds), dtype=cp.float64),
        )


def _stream_ld_single_pop(streaming_hm, *, bp_bins, raw, ac_filter,
                          chunk_size):
    """Chunk-streamed dispatch for ``compute_ld_statistics_gpu_single_pop``."""
    bp_bins_arr = np.array(bp_bins)
    max_dist = float(bp_bins_arr[-1])
    n_bins = len(bp_bins_arr) - 1
    bp_bins_cp = cp.array(bp_bins_arr)
    if chunk_size == 'auto':
        chunk_size_int = _estimate_ld_chunk_size(streaming_hm.num_haplotypes)
    else:
        chunk_size_int = int(chunk_size)
    bin_sums = cp.zeros((n_bins, 3), dtype=cp.float64)
    bin_counts = cp.zeros(n_bins, dtype=cp.float64)

    tail_haps, tail_pos = None, None
    for _, _, chunk_hm in streaming_hm.iter_gpu_chunks():
        if ac_filter:
            chunk_hm = chunk_hm.apply_biallelic_filter()
        chunk_haps = chunk_hm.haplotypes
        chunk_pos = chunk_hm.positions
        if not isinstance(chunk_pos, cp.ndarray):
            chunk_pos = cp.array(chunk_pos)
        if chunk_haps.shape[1] == 0:
            continue
        stitched_haps, stitched_pos, n_tail = _stitch_with_tail(
            chunk_haps, chunk_pos, tail_haps, tail_pos
        )
        _accumulate_pair_bins(
            stitched_haps, stitched_pos, bp_bins_cp, n_bins,
            max_dist, chunk_size_int, n_tail,
            bin_sums=bin_sums, bin_counts=bin_counts,
            pop1_indices=None, pop2_indices=None,
        )
        tail_haps, tail_pos = _new_tail(stitched_haps, stitched_pos, max_dist)
        del stitched_haps, stitched_pos, chunk_haps, chunk_pos

    return _format_ld_single_pop(bp_bins_arr, bin_sums, bin_counts, raw)


def _stream_ld_two_pops(streaming_hm, *, bp_bins, pop1, pop2, raw,
                        ac_filter, chunk_size):
    """Chunk-streamed dispatch for ``compute_ld_statistics_gpu_two_pops``."""
    bp_bins_arr = np.array(bp_bins)
    max_dist = float(bp_bins_arr[-1])
    n_bins = len(bp_bins_arr) - 1
    bp_bins_cp = cp.array(bp_bins_arr)
    pop1_indices = streaming_hm.sample_sets[pop1]
    pop2_indices = streaming_hm.sample_sets[pop2]
    if chunk_size == 'auto':
        chunk_size_int = _estimate_ld_chunk_size(
            max(len(pop1_indices), len(pop2_indices))
        )
    else:
        chunk_size_int = int(chunk_size)
    bin_sums = cp.zeros((n_bins, 15), dtype=cp.float64)
    bin_counts = cp.zeros(n_bins, dtype=cp.float64)

    tail_haps, tail_pos = None, None
    for _, _, chunk_hm in streaming_hm.iter_gpu_chunks():
        if ac_filter:
            chunk_hm = chunk_hm.apply_biallelic_filter()
        chunk_haps = chunk_hm.haplotypes
        chunk_pos = chunk_hm.positions
        if not isinstance(chunk_pos, cp.ndarray):
            chunk_pos = cp.array(chunk_pos)
        if chunk_haps.shape[1] == 0:
            continue
        stitched_haps, stitched_pos, n_tail = _stitch_with_tail(
            chunk_haps, chunk_pos, tail_haps, tail_pos
        )
        _accumulate_pair_bins(
            stitched_haps, stitched_pos, bp_bins_cp, n_bins,
            max_dist, chunk_size_int, n_tail,
            bin_sums=bin_sums, bin_counts=bin_counts,
            pop1_indices=pop1_indices, pop2_indices=pop2_indices,
        )
        tail_haps, tail_pos = _new_tail(stitched_haps, stitched_pos, max_dist)
        del stitched_haps, stitched_pos, chunk_haps, chunk_pos

    return _format_ld_two_pops(bp_bins_arr, bin_sums, bin_counts, raw)


def _format_ld_single_pop(bp_bins_arr, bin_sums, bin_counts, raw):
    # One device->host transfer for each accumulator; per-bin format
    # then reads from host memory rather than syncing per element.
    sums = cp.asnumpy(bin_sums)
    counts = cp.asnumpy(bin_counts)
    out = {}
    for i in range(len(bp_bins_arr) - 1):
        key = (float(bp_bins_arr[i]), float(bp_bins_arr[i + 1]))
        if raw:
            out[key] = (float(sums[i, 0]), float(sums[i, 1]), float(sums[i, 2]))
        elif counts[i] > 0:
            inv = 1.0 / float(counts[i])
            out[key] = (
                float(sums[i, 0]) * inv,
                float(sums[i, 1]) * inv,
                float(sums[i, 2]) * inv,
            )
        else:
            out[key] = (0.0, 0.0, 0.0)
    return out


def _format_ld_two_pops(bp_bins_arr, bin_sums, bin_counts, raw):
    sums = cp.asnumpy(bin_sums)
    counts = cp.asnumpy(bin_counts)
    names = _ld_names(2)
    out = {}
    for i in range(len(bp_bins_arr) - 1):
        key = (float(bp_bins_arr[i]), float(bp_bins_arr[i + 1]))
        if raw:
            row = sums[i]
        elif counts[i] > 0:
            row = sums[i] / float(counts[i])
        else:
            row = np.zeros(len(names))
        out[key] = OrderedDict(
            (name, float(row[j])) for j, name in enumerate(names)
        )
    return out


# =============================================================================
# Re-exports from ld_pipeline for backward compatibility
# =============================================================================
from .ld_pipeline import (  # noqa: E402, F401
    estimate_ld_chunk_size as _estimate_ld_chunk_size,
    iter_pairs_within_distance as _iter_pairs_within_distance,
    compute_counts_for_pairs as _compute_counts_for_pairs,
    compute_genotype_counts_for_pairs as _compute_genotype_counts_for_pairs,
    compute_two_pop_statistics_batch as _compute_two_pop_statistics_batch,
    compute_single_pop_statistics_batch as _compute_single_pop_statistics_batch,
    ld_names as _ld_names,
    het_names as _het_names,
    generate_stat_specs as _generate_stat_specs,
    PopData as _PopData,
)

