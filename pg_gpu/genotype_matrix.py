"""
Diploid genotype matrix for population genetics analysis.

Stores genotype data as alt allele counts (0/1/2) per individual per variant.
Provides conversion to/from HaplotypeMatrix.
"""

import numpy as np
import cupy as cp
from typing import Optional


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
                 sample_sets=None):
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

        self.genotypes = genotypes
        self.positions = positions
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self._sample_sets = sample_sets

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

    def __repr__(self):
        return (f"GenotypeMatrix(shape={self.shape}, "
                f"first_position={self.positions[0]}, "
                f"last_position={self.positions[-1]})")

    def transfer_to_gpu(self):
        if self._device == 'CPU':
            self.genotypes = cp.asarray(self.genotypes)
            self.positions = cp.asarray(self.positions)
            self._device = 'GPU'

    def transfer_to_cpu(self):
        if self._device == 'GPU':
            self.genotypes = np.asarray(self.genotypes.get())
            self.positions = np.asarray(self.positions.get())
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

        # pair consecutive haplotypes
        h1 = hap[0::2]  # even indices
        h2 = hap[1::2]  # odd indices

        # handle missing: if either haplotype is missing, genotype is missing
        if xp is cp:
            missing = (h1 < 0) | (h2 < 0)
            geno = (xp.maximum(h1, 0) + xp.maximum(h2, 0)).astype(xp.int8)
            geno[missing] = -1
        else:
            missing = (h1 < 0) | (h2 < 0)
            geno = (np.maximum(h1, 0) + np.maximum(h2, 0)).astype(np.int8)
            geno[missing] = -1

        # remap sample_sets: haplotype indices -> individual indices
        new_sample_sets = None
        if hap_matrix._sample_sets is not None:
            new_sample_sets = {}
            for name, indices in hap_matrix._sample_sets.items():
                # map haplotype indices to individual indices
                ind_indices = sorted(set(i // 2 for i in indices))
                new_sample_sets[name] = ind_indices

        return cls(geno, hap_matrix.positions, hap_matrix.chrom_start,
                   hap_matrix.chrom_end, sample_sets=new_sample_sets)

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
                               self.chrom_end)

    @classmethod
    def from_vcf(cls, path):
        """Construct from a VCF file.

        Parameters
        ----------
        path : str
            Path to VCF file.

        Returns
        -------
        GenotypeMatrix
        """
        import allel
        callset = allel.read_vcf(path)
        gt = callset['calldata/GT']  # (n_variants, n_samples, 2)
        pos = callset['variants/POS']

        # sum alleles to get alt count (0/1/2)
        geno = np.sum(gt, axis=2).astype(np.int8)  # (n_variants, n_samples)
        # handle missing (-1 in either allele)
        missing = np.any(gt < 0, axis=2)
        geno[missing] = -1

        # transpose to (n_individuals, n_variants)
        geno = geno.T

        return cls(geno, pos, chrom_start=pos[0], chrom_end=pos[-1])
