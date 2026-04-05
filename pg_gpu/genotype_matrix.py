"""
Diploid genotype matrix for population genetics analysis.

Stores genotype data as alt allele counts (0/1/2) per individual per variant.
Provides conversion to/from HaplotypeMatrix.
"""

import numpy as np
import cupy as cp
from typing import Optional

from .accessible import AccessibleMask, resolve_accessible_mask


class GenotypeMatrix:
    """Diploid genotype matrix with values 0 (hom ref), 1 (het), 2 (hom alt).

    Shape: (n_individuals, n_variants). Missing data encoded as -1.

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
                 accessible_mask=None):
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

        if accessible_mask is not None and not isinstance(accessible_mask, AccessibleMask):
            accessible_mask = resolve_accessible_mask(
                accessible_mask, chrom_start, chrom_end)
        self.accessible_mask = accessible_mask
        if self._accessible_mask is not None and self.n_total_sites is None:
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
        """Whether invariant site information is available for pairwise mode."""
        return self.n_total_sites is not None

    @property
    def n_invariant_sites(self):
        """Number of invariant sites, or None if unknown."""
        if self.n_total_sites is None:
            return None
        xp = cp if self.device == 'GPU' else np
        geno = self.genotypes
        valid_mask = geno >= 0
        geno_clean = xp.where(valid_mask, geno, 0)
        alt_counts = xp.sum(geno_clean, axis=0)
        n_valid = xp.sum(valid_mask, axis=0)
        # max possible alt count per site (diploid: 2 * n_valid)
        max_alt = 2 * n_valid
        is_variant = (alt_counts > 0) & (alt_counts < max_alt) & (n_valid >= 2)
        n_variant = int(xp.sum(is_variant))
        return self.n_total_sites - n_variant

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
        Genotype = sum of paired haplotypes (0, 1, or 2).

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

        hap = hap_matrix.haplotypes
        xp = cp if isinstance(hap, cp.ndarray) else np

        h1 = hap[0::2]  # even indices
        h2 = hap[1::2]  # odd indices
        n_ind = h1.shape[0]
        n_var = h1.shape[1]

        # Chunk over variants to avoid OOM from boolean intermediates
        geno = xp.empty((n_ind, n_var), dtype=xp.int8)
        if xp is cp:
            free_mem = cp.cuda.Device().mem_info[0]
            # Each variant needs ~4 * n_ind bytes for intermediates
            chunk = max(1, int(free_mem * 0.3 / (n_ind * 4)))
        else:
            chunk = n_var

        for vs in range(0, n_var, chunk):
            ve = min(vs + chunk, n_var)
            c1 = h1[:, vs:ve]
            c2 = h2[:, vs:ve]
            missing = (c1 < 0) | (c2 < 0)
            geno[:, vs:ve] = (xp.maximum(c1, 0) + xp.maximum(c2, 0)).astype(xp.int8)
            geno[:, vs:ve][missing] = -1

        # remap sample_sets: haplotype indices -> individual indices
        new_sample_sets = None
        if hap_matrix._sample_sets is not None:
            new_sample_sets = {}
            for name, indices in hap_matrix._sample_sets.items():
                # map haplotype indices to individual indices
                ind_indices = sorted(set(i // 2 for i in indices))
                new_sample_sets[name] = ind_indices

        return cls(geno, hap_matrix.positions, hap_matrix.chrom_start,
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
    def from_vcf(cls, path, include_invariant=False, accessible_bed=None):
        """Construct from a VCF file.

        Parameters
        ----------
        path : str
            Path to VCF file.
        include_invariant : bool
            If True, include invariant sites and set n_total_sites.
        accessible_bed : str, optional
            Path to a BED file defining accessible/callable regions.

        Returns
        -------
        GenotypeMatrix
        """
        import allel
        callset = allel.read_vcf(path)
        gt = callset['calldata/GT']  # (n_variants, n_samples, 2)
        pos = callset['variants/POS']
        samples = list(callset['samples'])

        # Filter to biallelic sites (max allele index <= 1)
        is_biallelic = np.all(gt <= 1, axis=(1, 2)) | np.all(gt < 0, axis=(1, 2))
        gt_array = allel.GenotypeArray(gt)
        ac = gt_array.count_alleles()
        is_biallelic = ac.is_biallelic_01()
        gt = gt[is_biallelic]
        pos = pos[is_biallelic]

        # sum alleles to get alt count (0/1/2)
        geno = np.sum(gt, axis=2).astype(np.int8)  # (n_variants, n_samples)
        # handle missing (-1 in either allele)
        missing = np.any(gt < 0, axis=2)
        geno[missing] = -1

        # transpose to (n_individuals, n_variants)
        geno = geno.T

        n_total_sites = geno.shape[1] if include_invariant else None
        gm = cls(geno, pos, chrom_start=pos[0], chrom_end=pos[-1],
                 n_total_sites=n_total_sites, samples=samples)
        if accessible_bed is not None:
            chrom = None
            if 'variants/CHROM' in callset:
                chrom = callset['variants/CHROM'][0]
            gm.set_accessible_mask(accessible_bed, chrom=chrom)
        return gm

    def load_pop_file(self, pop_file, pops=None):
        """Load population assignments from a tab-delimited file.

        Parameters
        ----------
        pop_file : str
            Tab-delimited file with columns: sample, pop.
        pops : list of str, optional
            Populations to include. If None, includes all found.
        """
        if self.samples is None:
            raise ValueError("No sample names stored. Use from_vcf() to load data.")

        pop_map = {}
        with open(pop_file) as f:
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
        """Filter to biallelic variant sites.

        Keeps variants where both ref and alt alleles are present
        among non-missing individuals.

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
