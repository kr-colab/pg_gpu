"""
Diploid genotype matrix for population genetics analysis.

Stores genotype data as alt allele counts (0/1/2) per individual per variant.
Provides conversion to/from HaplotypeMatrix.
"""

import numpy as np
import cupy as cp
from typing import Optional

from .accessible import AccessibleMask, resolve_accessible_mask


def _biallelic_and_alt(ac):
    """Per-site biallelic mask and alt-allele index from an (n_var, K) allele-count
    matrix (any array namespace).

    Biallelic = at most two distinct present alleles. Allele 0 is the reference;
    ``alt`` is the highest-index present allele > 0, which the 0/1/2 dosage counts --
    code-independent, so {0,2} and reference-absent {1,2} sites work, while {0,1}
    stays bit-identical (including a site fixed for allele 1). A site with no alt
    allele present (all reference / all missing) gets the out-of-range sentinel ``K``
    so its dosage counts to 0 and never matches the -1 missing code.
    """
    xp = cp if isinstance(ac, cp.ndarray) else np
    present = ac > 0
    biallelic = present.sum(axis=1) <= 2
    K = ac.shape[1]
    if K <= 1:
        return biallelic, xp.full(ac.shape[0], K)
    alt_present = present[:, 1:]                                  # alleles 1..K-1
    has_alt = alt_present.any(axis=1)
    highest_alt = K - 1 - alt_present[:, ::-1].argmax(axis=1)     # highest present > 0
    alt = xp.where(has_alt, highest_alt, K)
    return biallelic, alt


class GenotypeMatrix:
    """Diploid genotype matrix with values 0 (hom ref), 1 (het), 2 (hom alt).

    Shape: (n_individuals, n_variants). Missing data encoded as -1.

    Biallelic by construction: values are alt-allele dosage (0/1/2), so this
    structure cannot represent multiallelic genotypes. ``from_vcf`` and
    ``from_haplotype_matrix`` keep sites with at most two distinct present
    alleles -- so {0,1}, {0,2}, and reference-absent {1,2} are all kept, with the
    dosage counting the alt allele -- and drop sites with three or more distinct
    present alleles, emitting a ``BiallelicOnlyWarning`` with the dropped-site
    count. For
    multiallelic data use the allele-index ``HaplotypeMatrix``.

    Parameters
    ----------
    genotypes : ndarray, int8, shape (n_individuals, n_variants)
        Alt allele count per individual per variant.
    positions : ndarray, shape (n_variants,)
        Variant positions.
    chrom_start, chrom_end : int, optional
        Chromosome boundaries.
    sample_sets : dict, optional
        Maps population names to lists of individual indices.
    """

    def __init__(self, genotypes, positions, chrom_start=None, chrom_end=None,
                 sample_sets=None, n_total_sites=None, samples=None,
                 accessible_mask=None, fields=None):
        if genotypes.size == 0:
            raise ValueError("genotypes cannot be empty")
        if positions.size == 0:
            raise ValueError("positions cannot be empty")

        if isinstance(genotypes, cp.ndarray):
            self._device = 'GPU'
            if isinstance(positions, np.ndarray):
                positions = cp.array(positions)
        else:
            self._device = 'CPU'
            if isinstance(positions, cp.ndarray):
                positions = positions.get()

        self._genotypes = genotypes
        self._positions = positions
        self._accessible_idx = None
        self._geno_filtered = None
        self._pos_filtered = None
        self._accessible_mask = None
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self._sample_sets = sample_sets
        self.n_total_sites = n_total_sites
        self.samples = samples
        # See HaplotypeMatrix.fields for the shape contract.
        self.fields = dict(fields) if fields else {}

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
    def genotypes(self):
        if self._accessible_idx is None:
            return self._genotypes
        if self._geno_filtered is None:
            self._geno_filtered = self._genotypes[:, self._accessible_idx]
        return self._geno_filtered

    @genotypes.setter
    def genotypes(self, value):
        self._genotypes = value
        self._geno_filtered = None

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
        self._geno_filtered = None
        self._pos_filtered = None

    @property
    def device(self):
        return self._device

    @property
    def sample_sets(self):
        if self._sample_sets is None:
            return {"all": list(range(self.genotypes.shape[0]))}
        return self._sample_sets

    @sample_sets.setter
    def sample_sets(self, sample_sets):
        self._sample_sets = sample_sets

    @property
    def shape(self):
        return self.genotypes.shape

    @property
    def num_variants(self):
        return self.genotypes.shape[1]

    @property
    def num_individuals(self):
        return self.genotypes.shape[0]

    @property
    def has_accessible_mask(self):
        """Whether an accessible site mask is attached."""
        return self.accessible_mask is not None

    def set_accessible_mask(self, mask_or_path, chrom=None):
        """Attach an accessible site mask (non-destructive).

        Returns self for chaining.

        Parameters
        ----------
        mask_or_path : str, path-like, numpy.ndarray, or AccessibleMask
            BED file path, boolean array, or AccessibleMask instance.
        chrom : str, optional
            Chromosome name (required for BED file input).
        """
        self.accessible_mask = resolve_accessible_mask(
            mask_or_path, self.chrom_start, self.chrom_end, chrom)
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
        self.accessible_mask = None
        self.n_total_sites = None
        return self

    @property
    def has_invariant_info(self):
        """Whether invariant site information is available for span normalization."""
        return self.n_total_sites is not None

    @property
    def n_callable_sites(self):
        """Total callable sites in the analysis universe.

        Alias for ``n_total_sites``. See ``HaplotypeMatrix.n_callable_sites``
        for full semantics.
        """
        return self.n_total_sites

    @property
    def n_segregating_sites(self):
        """Number of polymorphic sites in the matrix.

        Counts sites where 0 < alt_count < 2 * n_valid (polymorphic among
        observed diploid genotypes), with at least 2 valid samples.
        """
        xp = cp if self.device == 'GPU' else np
        geno = self.genotypes
        valid_mask = geno >= 0
        geno_clean = xp.where(valid_mask, geno, 0)
        alt_counts = xp.sum(geno_clean, axis=0)
        n_valid = xp.sum(valid_mask, axis=0)
        max_alt = 2 * n_valid
        is_variant = (alt_counts > 0) & (alt_counts < max_alt) & (n_valid >= 2)
        return int(xp.sum(is_variant))

    @property
    def n_invariant_sites(self):
        """Number of invariant sites in the callable span, or None if unknown.

        Computed as ``n_callable_sites - n_segregating_sites``. See
        ``HaplotypeMatrix.n_invariant_sites`` for full semantics.
        """
        if self.n_total_sites is None:
            return None
        return self.n_total_sites - self.n_segregating_sites

    def __repr__(self):
        return (f"GenotypeMatrix(shape={self.shape}, "
                f"first_position={self.positions[0]}, "
                f"last_position={self.positions[-1]})")

    def transfer_to_gpu(self):
        if self._device == 'CPU':
            self._genotypes = cp.asarray(self._genotypes)
            self._positions = cp.asarray(self._positions)
            if self._accessible_idx is not None:
                self._accessible_idx = cp.asarray(self._accessible_idx)
            self._geno_filtered = None
            self._pos_filtered = None
            self._device = 'GPU'

    def transfer_to_cpu(self):
        if self._device == 'GPU':
            self._genotypes = np.asarray(self._genotypes.get())
            self._positions = np.asarray(self._positions.get())
            if self._accessible_idx is not None:
                self._accessible_idx = np.asarray(self._accessible_idx.get())
            self._geno_filtered = None
            self._pos_filtered = None
            self._device = 'CPU'

    @classmethod
    def from_haplotype_matrix(cls, hap_matrix):
        """Convert a HaplotypeMatrix to a GenotypeMatrix.

        Pairs consecutive haplotypes (0,1), (2,3), ... as diploid individuals.
        Each genotype is the count of the site's alt allele in the pair (0, 1, or
        2); a pair with any missing haplotype is -1. Only biallelic sites (at most
        two distinct present alleles) are kept -- sites with three or more
        distinct present alleles cannot be a 0/1/2 dosage and are dropped with a
        BiallelicOnlyWarning.

        Parameters
        ----------
        hap_matrix : HaplotypeMatrix
            Haploid data. Must have even number of haplotypes.

        Returns
        -------
        GenotypeMatrix
        """
        n_hap = hap_matrix.haplotypes.shape[0]
        if n_hap % 2 != 0:
            raise ValueError(
                f"Need even number of haplotypes for diploid conversion, got {n_hap}")

        if hap_matrix.device == 'CPU':
            hap_matrix.transfer_to_gpu()
        hap = hap_matrix.haplotypes
        positions = hap_matrix.positions

        # Biallelic = at most two distinct present alleles; the dosage counts the
        # alt (highest present allele > 0), so {0,2} and reference-absent {1,2}
        # sites are handled correctly and {0,1} is unchanged. Sites with >= 3
        # alleles cannot be a dosage and are dropped with a warning; positions
        # shares hap's array namespace, so the same mask indexes both.
        from ._memutil import allele_counts
        ac, _ = allele_counts(hap)
        biallelic, alt = _biallelic_and_alt(ac)
        n_dropped = int((~biallelic).sum())
        if n_dropped:
            from ._warnings import _warn_biallelic_only
            _warn_biallelic_only(
                n_dropped, context="GenotypeMatrix.from_haplotype_matrix")
            hap = hap[:, biallelic]
            positions = positions[biallelic]
            alt = alt[biallelic]

        h1 = hap[0::2]  # even indices
        h2 = hap[1::2]  # odd indices
        n_ind = h1.shape[0]
        n_var = h1.shape[1]

        # Dosage = per-individual count of the alt allele (0/1/2); a pair with any
        # missing haplotype -> -1. Chunk over variants to bound GPU memory.
        geno = cp.empty((n_ind, n_var), dtype=cp.int8)
        free_mem = cp.cuda.Device().mem_info[0]
        chunk = max(1, int(free_mem * 0.3 / (n_ind * 4)))

        for vs in range(0, n_var, chunk):
            ve = min(vs + chunk, n_var)
            c1 = h1[:, vs:ve]
            c2 = h2[:, vs:ve]
            a = alt[vs:ve][None, :]
            missing = (c1 < 0) | (c2 < 0)
            geno[:, vs:ve] = ((c1 == a).astype(cp.int8) + (c2 == a).astype(cp.int8))
            geno[:, vs:ve][missing] = -1

        # remap sample_sets: haplotype indices -> individual indices
        new_sample_sets = None
        if hap_matrix._sample_sets is not None:
            new_sample_sets = {}
            for name, indices in hap_matrix._sample_sets.items():
                # map haplotype indices to individual indices
                ind_indices = sorted(set(i // 2 for i in indices))
                new_sample_sets[name] = ind_indices

        return cls(geno, positions, hap_matrix.chrom_start,
                   hap_matrix.chrom_end, sample_sets=new_sample_sets,
                   n_total_sites=hap_matrix.n_total_sites,
                   accessible_mask=hap_matrix.accessible_mask)

    def to_haplotype_matrix(self):
        """Convert back to HaplotypeMatrix (expand diploid to haploid).

        Each individual becomes two consecutive haplotypes.
        Het sites (genotype=1) are assigned as (0,1).

        Returns
        -------
        HaplotypeMatrix
        """
        from .haplotype_matrix import HaplotypeMatrix

        geno = self.genotypes
        xp = cp if isinstance(geno, cp.ndarray) else np

        n_ind, n_var = geno.shape
        hap = xp.zeros((n_ind * 2, n_var), dtype=xp.int8)

        missing = geno < 0
        g = xp.maximum(geno, 0)

        # haplotype 1: 1 if genotype >= 1
        hap[0::2] = xp.where(g >= 1, 1, 0).astype(xp.int8)
        # haplotype 2: 1 if genotype >= 2
        hap[1::2] = xp.where(g >= 2, 1, 0).astype(xp.int8)

        # propagate missing
        hap[0::2][missing] = -1
        hap[1::2][missing] = -1

        return HaplotypeMatrix(hap, self.positions, self.chrom_start,
                               self.chrom_end,
                               n_total_sites=self.n_total_sites,
                               accessible_mask=self.accessible_mask)

    @classmethod
    def from_vcf(cls, path, include_invariant=False, accessible_bed=None,
                 fields=None):
        """Construct from a VCF file.

        Parameters
        ----------
        path : str
            Path to VCF file.
        include_invariant : bool
            If True, include invariant sites and set n_total_sites.
        accessible_bed : str, optional
            Path to a BED file defining accessible/callable regions.
        fields : list of str, optional
            VCF FORMAT/INFO tags to load (e.g. ``['GQ', 'DP', 'MQ']``); see
            ``HaplotypeMatrix.from_vcf`` for the shape contract. Arrays are
            sliced down to the biallelic-only variant set this constructor
            keeps, so per-variant fields end up shape ``(n_kept,)`` and
            per-genotype fields ``(n_kept, n_samples)``.

        Returns
        -------
        GenotypeMatrix
        """
        import allel
        from .haplotype_matrix import (
            _build_read_vcf_fields, _classify_vcf_qc_tags,
            _resolve_qc_fields_vcf,
        )
        from ._warnings import _maybe_memory_warn
        _maybe_memory_warn(path)
        if fields:
            tag_to_path, unknown_tags = _classify_vcf_qc_tags(path, fields)
            read_fields = _build_read_vcf_fields(tag_to_path.values())
        else:
            tag_to_path, unknown_tags = {}, []
            read_fields = None
        callset = allel.read_vcf(path, fields=read_fields)
        gt = callset['calldata/GT']  # (n_variants, n_samples, 2)
        pos = callset['variants/POS']
        samples = list(callset['samples'])

        from ._warnings import check_diploid_encoding, _warn_biallelic_only
        check_diploid_encoding(gt, sample_names=samples, source=f"VCF '{path}'")

        # Biallelic = at most two distinct present alleles; the dosage counts the
        # alt (highest present allele > 0), so {0,2} and reference-absent {1,2}
        # sites are handled correctly and {0,1} is unchanged. Only sites with
        # three or more distinct present alleles are dropped (with a warning);
        # monomorphic and all-missing sites are retained, matching
        # from_haplotype_matrix. count_alleles ignores missing (-1), giving an
        # (n_variants, n_alleles) count matrix.
        ac = np.asarray(allel.GenotypeArray(gt).count_alleles())
        is_biallelic, alt = _biallelic_and_alt(ac)
        _warn_biallelic_only(int(np.sum(~is_biallelic)),
                             context="GenotypeMatrix.from_vcf")
        gt = gt[is_biallelic]
        pos = pos[is_biallelic]
        alt = alt[is_biallelic]

        qc_fields = (_resolve_qc_fields_vcf(callset, tag_to_path, unknown_tags)
                     if fields else {})
        # The biallelic filter trims the variant axis; QC arrays must be
        # sliced consistently or they'd no longer align with the genotype
        # matrix.
        for tag, arr in qc_fields.items():
            qc_fields[tag] = arr[is_biallelic]

        # Dosage = per-genotype count of the alt allele (0/1/2); a call with any
        # missing allele (-1) -> -1.
        geno = (gt == alt[:, None, None]).sum(axis=2).astype(np.int8)
        missing = np.any(gt < 0, axis=2)
        geno[missing] = -1

        # transpose to (n_individuals, n_variants)
        geno = geno.T

        n_total_sites = geno.shape[1] if include_invariant else None
        gm = cls(geno, pos, chrom_start=pos[0], chrom_end=pos[-1],
                 n_total_sites=n_total_sites, samples=samples,
                 fields=qc_fields)
        if accessible_bed is not None:
            chrom = None
            if 'variants/CHROM' in callset:
                chrom = callset['variants/CHROM'][0]
            gm.set_accessible_mask(accessible_bed, chrom=chrom)
        return gm

    @classmethod
    def from_zarr(cls, path, region=None, accessible_bed=None,
                  include_invariant=False,
                  pop_assignment=None,
                  streaming: str = "auto",
                  chunk_bp: int = 1_500_000,
                  prefetch: int = 1,
                  backend: str = "auto",
                  fields: list = None):
        """Construct a GenotypeMatrix from a Zarr store.

        Identical interface to ``HaplotypeMatrix.from_zarr``;
        see that method's docstring for the meaning of every kwarg
        (``streaming``, ``pop_assignment`` flexibility, ``chunk_bp``,
        ``prefetch``, ``backend``, ``fields``). The only difference is the
        returned type: this returns ``GenotypeMatrix`` (or
        ``StreamingGenotypeMatrix`` on the streaming path) where
        each row is a (n_indiv, ploidy) genotype call, versus
        ``HaplotypeMatrix`` which presents the same data as a
        (n_haplotypes, n_variants) phased matrix.

        Parameters
        ----------
        include_invariant : bool
            If True, set ``n_total_sites`` from the loaded variant
            count. Unique to this entry point (the haplotype matrix
            does not need it).
        path, region, accessible_bed, pop_assignment, streaming, chunk_bp, prefetch, backend, fields
            See ``HaplotypeMatrix.from_zarr``.

        Returns
        -------
        GenotypeMatrix or StreamingGenotypeMatrix
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
            if fields:
                raise NotImplementedError(
                    "fields= is not supported on the streaming "
                    "(streaming='always') path yet. Load eagerly to "
                    "access VCF FORMAT/INFO arrays.")
            return cls._build_streaming(
                path, region=region, pop_assignment=pop_assignment,
                chunk_bp=chunk_bp, prefetch=prefetch,
                backend=backend, accessible_bed=accessible_bed,
            )

        # 'auto' and 'never' both want eager when the matrix fits;
        # 'auto' falls back to streaming when it doesn't, 'never'
        # raises. The decision needs the matrix's projected size,
        # which is only available via ZarrGenotypeSource (VCZ-only).
        # Scikit-allel stores always route to eager.
        from .haplotype_matrix import _decide_streaming_mode
        choice, source = _decide_streaming_mode(path, region=region,
                                                streaming=streaming,
                                                pop_assignment=pop_assignment)
        if choice == "streaming":
            if fields:
                raise NotImplementedError(
                    "fields= is not supported on the streaming path; "
                    "matrix would not fit at streaming='auto'. Pass "
                    "streaming='never' to force eager (and fit-check), "
                    "or load without fields=.")
            return cls._build_streaming(
                path, region=region, pop_assignment=pop_assignment,
                chunk_bp=chunk_bp, prefetch=prefetch,
                backend=backend, source=source, accessible_bed=accessible_bed,
            )
        return cls._build_eager(path, region=region,
                                accessible_bed=accessible_bed,
                                include_invariant=include_invariant,
                                pop_assignment=pop_assignment,
                                fields=fields)

    @classmethod
    def _build_eager(cls, path, *, region, accessible_bed,
                     include_invariant, pop_assignment, fields=None):
        from .zarr_io import read_genotypes, normalize_pop_input, read_qc_fields
        from ._gpu_genotype_prep import build_genotype_matrix

        data = read_genotypes(path, region)
        gt = data['gt']  # (n_variants, n_samples, ploidy)
        positions = data['positions']
        samples = data['samples']

        from ._warnings import check_diploid_encoding
        check_diploid_encoding(gt, sample_names=samples,
                               source=f"zarr store '{path}'")

        n_total_sites = gt.shape[0] if include_invariant else None
        chrom = region.split(':')[0] if region else None

        gm = build_genotype_matrix(
            gt, positions,
            chrom_start=int(positions[0]),
            chrom_end=int(positions[-1]),
            n_total_sites=n_total_sites,
            samples=list(samples) if samples else None,
        )
        from ._warnings import _warn_biallelic_only
        _warn_biallelic_only(gm._n_multiallelic_recoded,
                             context="GenotypeMatrix.from_zarr")
        if fields:
            gm.fields = read_qc_fields(
                path, fields,
                variant_indices=data.get('variant_indices'),
                region=region,
            )
        if accessible_bed is not None:
            gm.set_accessible_mask(accessible_bed, chrom=chrom)

        try:
            import zarr
            store = zarr.open_group(path, mode="r")
        except Exception:
            store = None
        pop_map = normalize_pop_input(
            pop_assignment, zarr_path=path,
            sample_names=gm.samples or [],
            zarr_store=store,
            announce_prefix="GenotypeMatrix.from_zarr",
        )
        if pop_map is not None:
            gm.load_pop_file(pop_map)

        return gm

    @classmethod
    def _build_streaming(cls, path, *, region, pop_assignment, chunk_bp,
                         prefetch, backend="auto", source=None,
                         accessible_bed=None):
        from .streaming_matrix import (
            StreamingGenotypeMatrix, _pick_chunk_fetcher,
        )
        from .zarr_source import ZarrGenotypeSource

        if source is None:
            source = ZarrGenotypeSource(path, region=region,
                                        pop_assignment=pop_assignment)
        else:
            source.pop_cols = source._resolve_pop_assignment(pop_assignment)
        fetcher = _pick_chunk_fetcher(source, backend=backend)

        # Resolve the accessible BED once over the source's variant-position
        # bounds so every chunk is filtered to accessible variants, matching
        # the eager path's mask-filtered .genotypes view (grm/ibs read that
        # view, so streaming and eager see the same variant set).
        from .accessible import resolve_streaming_accessible_mask
        accessible_mask = resolve_streaming_accessible_mask(
            accessible_bed, source, region)

        return StreamingGenotypeMatrix(
            source, fetcher,
            chunk_bp=chunk_bp, prefetch=prefetch,
            accessible_mask=accessible_mask,
        )

    def filter(self, variants=None, genotypes=None,
               drop_all_missing: bool = True) -> "GenotypeMatrix":
        """Return a new GenotypeMatrix with quality filters applied.

        Parameters
        ----------
        variants : array-like of bool, shape (n_variants,), optional
            Per-variant keep mask; variants where ``False`` are dropped
            from every array on the returned matrix.
        genotypes : array-like of bool, shape (n_variants, n_samples), optional
            Per-genotype keep mask; genotypes where ``False`` are set to
            ``-1`` (missing). ``fields`` entries keep their original
            per-genotype QC values.
        drop_all_missing : bool, default True
            After applying both masks, drop variants where every
            individual's call is ``-1``.

        Returns
        -------
        GenotypeMatrix
            Fresh matrix; underlying arrays are new allocations.
            ``samples``, ``sample_sets``, ``chrom_start`` / ``chrom_end``
            are preserved. See ``HaplotypeMatrix.filter`` for the
            accessibility-mask interaction (this method behaves the
            same way).
        """
        xp = cp if self._device == 'GPU' else np
        geno_src = self._genotypes
        pos_src = self._positions
        n_indiv, n_var = geno_src.shape

        if variants is not None:
            variants_arr = xp.asarray(variants).astype(bool, copy=False)
            if variants_arr.shape != (n_var,):
                raise ValueError(
                    f"variants mask shape mismatch: expected ({n_var},), "
                    f"got {tuple(variants_arr.shape)}")
        else:
            variants_arr = None

        if genotypes is not None:
            genotypes_arr = xp.asarray(genotypes).astype(bool, copy=False)
            if genotypes_arr.shape != (n_var, n_indiv):
                raise ValueError(
                    f"genotypes mask shape mismatch: expected "
                    f"({n_var}, {n_indiv}), got "
                    f"{tuple(genotypes_arr.shape)}")
        else:
            genotypes_arr = None

        if genotypes_arr is not None:
            # User mask is (n_var, n_samples); genotype matrix lays
            # samples along axis 0, so transpose to broadcast.
            geno = xp.where(genotypes_arr.T, geno_src, np.int8(-1))
        else:
            geno = geno_src

        keep_v = xp.ones(n_var, dtype=bool)
        if variants_arr is not None:
            keep_v &= variants_arr
        if drop_all_missing:
            keep_v &= ~(geno == -1).all(axis=0)
        keep_idx = xp.where(keep_v)[0]

        if int(keep_idx.size) == 0:
            if self._device == 'GPU':
                empty_geno = cp.empty((n_indiv, 0), dtype=geno_src.dtype)
                empty_pos = cp.array([], dtype=pos_src.dtype)
            else:
                empty_geno = np.empty((n_indiv, 0), dtype=geno_src.dtype)
                empty_pos = np.array([], dtype=pos_src.dtype)
            result = object.__new__(GenotypeMatrix)
            result._genotypes = empty_geno
            result._positions = empty_pos
            result._accessible_idx = None
            result._geno_filtered = None
            result._pos_filtered = None
            result._accessible_mask = None
            result.chrom_start = self.chrom_start
            result.chrom_end = self.chrom_end
            result._sample_sets = self._sample_sets
            result._device = self._device
            result.n_total_sites = None
            result.samples = self.samples
            result.fields = {tag: arr[:0] for tag, arr in self.fields.items()}
            return result

        new_geno = geno[:, keep_idx]
        new_pos = pos_src[keep_idx]
        keep_idx_np = keep_idx.get() if hasattr(keep_idx, 'get') else keep_idx
        new_fields = {tag: arr[keep_idx_np] for tag, arr in self.fields.items()}

        return GenotypeMatrix(
            new_geno, new_pos,
            chrom_start=self.chrom_start, chrom_end=self.chrom_end,
            sample_sets=self._sample_sets,
            samples=self.samples,
            fields=new_fields,
        )

    def to_zarr(self, zarr_path, format='vcz', contig_name=None):
        """Save genotype data to Zarr format.

        Parameters
        ----------
        zarr_path : str
            Output Zarr store path.
        format : str
            ``'vcz'`` (default) or ``'scikit-allel'``.
        contig_name : str, optional
            Chromosome name for VCZ format.
        """
        from .zarr_io import write_vcz, write_allel

        geno = self.genotypes if isinstance(self.genotypes, np.ndarray) \
            else self.genotypes.get()
        pos = self.positions if isinstance(self.positions, np.ndarray) \
            else self.positions.get()

        # Expand (n_individuals, n_variants) to (n_variants, n_samples, 2)
        n_ind, n_var = geno.shape
        g = geno.T  # (n_var, n_ind)
        gt = np.zeros((n_var, n_ind, 2), dtype=np.int8)
        gt[:, :, 0] = (g >= 1).view(np.int8)
        gt[:, :, 1] = (g >= 2).view(np.int8)
        missing = g < 0
        gt[missing] = -1

        if format == 'vcz':
            write_vcz(zarr_path, gt, pos, self.samples,
                      contig_name=contig_name, fields=self.fields)
        elif format == 'scikit-allel':
            if self.fields:
                raise NotImplementedError(
                    "Writing fields= round-trip is only supported for "
                    "format='vcz'; the scikit-allel writer has not been "
                    "extended yet.")
            write_allel(zarr_path, gt, pos, self.samples)
        else:
            raise ValueError(
                f"Unknown format: {format!r}. Use 'vcz' or 'scikit-allel'."
            )

    def load_pop_file(self, pop_assignment, pops=None):
        """Load population assignments from a tab-delimited file or
        an already-resolved sample->population mapping.

        Parameters
        ----------
        pop_assignment : str or dict
            Either a path to a tab-delimited file with columns
            ``sample\tpop``, or a dict mapping sample names to
            population labels.
        pops : list of str, optional
            Populations to include. If None, includes all found.
        """
        if self.samples is None:
            raise ValueError("No sample names stored. Use from_vcf() to load data.")

        if isinstance(pop_assignment, dict):
            pop_map = {str(k): str(v) for k, v in pop_assignment.items() if v}
        else:
            pop_map = {}
            with open(pop_assignment) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] != 'sample':
                        pop_map[parts[0]] = parts[1]

        if pops is None:
            pops = sorted(set(pop_map.values()))

        pop_sets = {p: [] for p in pops}
        for i, name in enumerate(self.samples):
            pop = pop_map.get(name)
            if pop in pop_sets:
                pop_sets[pop].append(i)

        self.sample_sets = pop_sets

    def apply_biallelic_filter(self):
        """Drop monomorphic (non-segregating) sites.

        The matrix is already biallelic by construction (0/1/2 dosage), so this
        does not classify alleles. It keeps only sites still segregating among the
        non-missing individuals: the alt allele is present but not fixed
        (0 < total alt dosage < 2 * n_called) and at least two individuals are
        called.

        Returns
        -------
        GenotypeMatrix
        """
        xp = cp if self._device == 'GPU' else np
        geno = self.genotypes
        valid = geno >= 0
        geno_clean = xp.where(valid, geno, 0)
        alt_counts = xp.sum(geno_clean, axis=0)
        n_valid = xp.sum(valid, axis=0)
        max_alt = 2 * n_valid
        keep = (alt_counts > 0) & (alt_counts < max_alt) & (n_valid >= 2)

        if self._device == 'GPU':
            keep_idx = cp.where(keep)[0]
            new_geno = self.genotypes[:, keep_idx]
            new_pos = self.positions[keep_idx]
        else:
            keep_np = keep if isinstance(keep, np.ndarray) else keep.get()
            new_geno = self.genotypes[:, keep_np]
            new_pos = self.positions[keep_np]

        return GenotypeMatrix(new_geno, new_pos,
                              self.chrom_start, self.chrom_end,
                              sample_sets=self._sample_sets,
                              n_total_sites=self.n_total_sites,
                              samples=self.samples,
                              accessible_mask=self.accessible_mask)
