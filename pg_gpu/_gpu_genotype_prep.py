"""GPU-side preparation of raw VCZ-shape genotype blocks for HaplotypeMatrix
and GenotypeMatrix.

The VCZ ``call_genotype`` array is ``(n_var, n_dip, 2)`` int8. The two matrix
classes consume it differently:

* ``HaplotypeMatrix`` wants ``(n_hap, n_var)`` int8 with haps
  ``0..n_dip-1`` carrying ploidy 0 and haps ``n_dip..2*n_dip-1`` carrying
  ploidy 1. ``build_haplotype_matrix`` does the ploidy interleave +
  transpose on the GPU.
* ``GenotypeMatrix`` wants ``(n_indiv, n_var)`` int8 dosages (0/1/2 with
  ``-1`` for missing). ``build_genotype_matrix`` counts each site's alt
  allele on the GPU and propagates missing.

Both helpers take the same raw input shape and exist to keep the
host-side numpy reshape (which on a 1 Mb / 200k-haplotype block is
~60 s of single-threaded strided int8 copy and a 56 GB allocation)
off the per-chunk hot path. cupy's tiled transpose / fused add runs in
seconds with a fixed memory footprint.

Missing cells (gt = -1) are preserved by both helpers -- downstream kernels
handle them via the ``'include'`` / ``'exclude'`` missing-data modes.
``build_haplotype_matrix`` keeps raw allele indices (the per-allele haplotype
statistics are multiallelic-capable). ``build_genotype_matrix`` needs a 0/1/2
dosage, so it classifies each site by the shared biallelic definition and
recodes any site with three or more distinct alleles present in the sample to a
fully-missing row (no valid dosage exists), matching the eager from_vcf loader.
"""

import cupy as cp


def build_haplotype_matrix(gt, pos, *,
                           chrom_start=None, chrom_end=None,
                           sample_sets=None, n_total_sites=None,
                           samples=None, accessible_mask=None):
    """Build a HaplotypeMatrix from a raw VCZ-style genotype block.

    Parameters
    ----------
    gt : ndarray, shape (n_var, n_dip, 2)
        Raw call_genotype block of allele indices (0 = reference, 1.. =
        alternate), ``-1`` missing. Indices are preserved as-is (the
        per-allele haplotype statistics are multiallelic-capable). Host or
        device.
    pos : ndarray, shape (n_var,)
        Variant positions. Host or device.

    The remaining kwargs are forwarded to ``HaplotypeMatrix.__init__``.

    Returns
    -------
    HaplotypeMatrix
        With haplotypes on the GPU in ``(n_hap, n_var)`` layout.
    """
    from .haplotype_matrix import HaplotypeMatrix

    if gt.ndim != 3 or gt.shape[2] != 2:
        raise ValueError(
            f"gt must have shape (n_var, n_dip, 2); got {gt.shape}"
        )
    if gt.shape[0] != pos.shape[0]:
        raise ValueError(
            f"gt and pos disagree on n_var: gt={gt.shape[0]}, pos={pos.shape[0]}"
        )

    n_var, n_dip, _ = gt.shape
    gt_gpu = cp.asarray(gt)

    # transpose (n_var, n_dip, 2) -> (2, n_dip, n_var) puts ploidy outermost,
    # then the reshape concatenates: hap[0..n_dip-1] = ploidy 0 samples,
    # hap[n_dip..2*n_dip-1] = ploidy 1 samples. This matches the layout
    # HaplotypeMatrix.load_pop_file builds, so sample_sets indices line up
    # without a permutation.
    haps = cp.ascontiguousarray(
        gt_gpu.transpose(2, 1, 0).reshape(2 * n_dip, n_var)
    )
    del gt_gpu

    positions = cp.asarray(pos)

    return HaplotypeMatrix(
        haps, positions,
        chrom_start=chrom_start, chrom_end=chrom_end,
        sample_sets=sample_sets, n_total_sites=n_total_sites,
        samples=samples, accessible_mask=accessible_mask,
    )


def build_genotype_matrix(gt, pos, *,
                          chrom_start=None, chrom_end=None,
                          sample_sets=None, n_total_sites=None,
                          samples=None, accessible_mask=None):
    """Build a GenotypeMatrix from a raw VCZ-style genotype block.

    Parameters
    ----------
    gt : ndarray, shape (n_var, n_dip, 2)
        Raw call_genotype block of allele indices (0 = reference, 1.. =
        alternate), ``-1`` missing. Host or device.
    pos : ndarray, shape (n_var,)
        Variant positions. Host or device.

    The remaining kwargs are forwarded to ``GenotypeMatrix.__init__``.

    Returns
    -------
    GenotypeMatrix
        With genotypes on the GPU in ``(n_indiv, n_var)`` int8 layout.
        Each cell is ``0/1/2`` (count of the site's alt allele) or ``-1``
        when either ploidy on that variant was missing. A site with three or
        more distinct alleles present in the sample cannot be a 0/1/2 dosage,
        so its whole row is set to ``-1`` (present but fully missing), keeping
        the row/chunk alignment the streaming path relies on. The number of
        such recoded sites is
        stashed on the result as ``_n_multiallelic_recoded`` for the caller
        to surface as a BiallelicOnlyWarning (once per load).
    """
    from .genotype_matrix import GenotypeMatrix, _biallelic_and_alt
    from ._memutil import allele_counts

    if gt.ndim != 3 or gt.shape[2] != 2:
        raise ValueError(
            f"gt must have shape (n_var, n_dip, 2); got {gt.shape}"
        )
    if gt.shape[0] != pos.shape[0]:
        raise ValueError(
            f"gt and pos disagree on n_var: gt={gt.shape[0]}, pos={pos.shape[0]}"
        )

    gt_gpu = cp.asarray(gt)
    n_var, n_dip, _ = gt_gpu.shape

    # Biallelic = at most two distinct present alleles; dosage counts the chosen
    # alt (highest present allele > 0). Matches from_vcf / from_haplotype_matrix,
    # so all three GenotypeMatrix loaders agree. allele_counts wants an
    # (n_hap, n_var) layout; a site sits entirely on the variant axis of this
    # chunk, so per-chunk counts are complete.
    haps = gt_gpu.reshape(n_var, 2 * n_dip).T
    ac, _ = allele_counts(haps)
    biallelic, alt = _biallelic_and_alt(ac)

    missing = (gt_gpu < 0).any(axis=2)
    geno = (gt_gpu == alt[:, None, None]).sum(axis=2).astype(cp.int8)
    geno = cp.where(missing, cp.int8(-1), geno)
    # A site with three or more distinct present alleles has no valid 0/1/2
    # dosage -> the whole row is missing.
    n_multiallelic = int((~biallelic).sum())
    if n_multiallelic:
        geno[~biallelic, :] = cp.int8(-1)

    # transpose to the (n_indiv, n_var) layout GenotypeMatrix kernels
    # expect; cupy's tiled transpose handles the strided write
    # efficiently.
    geno = cp.ascontiguousarray(geno.T)
    del gt_gpu

    positions = cp.asarray(pos)

    gm = GenotypeMatrix(
        geno, positions,
        chrom_start=chrom_start, chrom_end=chrom_end,
        sample_sets=sample_sets, n_total_sites=n_total_sites,
        samples=samples, accessible_mask=accessible_mask,
    )
    gm._n_multiallelic_recoded = n_multiallelic
    return gm
