"""GPU-side preparation of raw VCZ-shape genotype blocks for HaplotypeMatrix
and GenotypeMatrix.

The VCZ ``call_genotype`` array is ``(n_var, n_dip, 2)`` int8. The two matrix
classes consume it differently:

* ``HaplotypeMatrix`` wants ``(n_hap, n_var)`` int8 with haps
  ``0..n_dip-1`` carrying ploidy 0 and haps ``n_dip..2*n_dip-1`` carrying
  ploidy 1. ``build_haplotype_matrix`` does the ploidy interleave +
  transpose on the GPU.
* ``GenotypeMatrix`` wants ``(n_indiv, n_var)`` int8 dosages (0/1/2 with
  ``-1`` for missing). ``build_genotype_matrix`` sums the two ploidies on
  the GPU and propagates missing.

Both helpers take the same raw input shape and exist to keep the
host-side numpy reshape (which on a 1 Mb / 200k-haplotype block is
~60 s of single-threaded strided int8 copy and a 56 GB allocation)
off the per-chunk hot path. cupy's tiled transpose / fused add runs in
seconds with a fixed memory footprint.

Missing / multiallelic cells (gt = -1) are preserved by both helpers --
downstream kernels handle them via the ``'include'`` / ``'exclude'``
missing-data modes, and dropping them at build time would silently
change semantics for callers that opted into 'include'.
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
        Raw call_genotype block. ``-1`` marks multiallelic / missing
        cells; rows are preserved. Host or device.
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
        Raw call_genotype block. ``-1`` marks multiallelic / missing
        cells; a row is treated as missing for a given diploid if
        either ploidy is ``-1``. Host or device.
    pos : ndarray, shape (n_var,)
        Variant positions. Host or device.

    The remaining kwargs are forwarded to ``GenotypeMatrix.__init__``.

    Returns
    -------
    GenotypeMatrix
        With genotypes on the GPU in ``(n_indiv, n_var)`` int8 layout.
        Each cell is ``0/1/2`` (dosage) or ``-1`` when either ploidy
        on that variant was missing in the input.
    """
    from .genotype_matrix import GenotypeMatrix

    if gt.ndim != 3 or gt.shape[2] != 2:
        raise ValueError(
            f"gt must have shape (n_var, n_dip, 2); got {gt.shape}"
        )
    if gt.shape[0] != pos.shape[0]:
        raise ValueError(
            f"gt and pos disagree on n_var: gt={gt.shape[0]}, pos={pos.shape[0]}"
        )

    gt_gpu = cp.asarray(gt)
    # Dosage = sum of ploidies, clamping -1 to 0 first so the sum
    # makes sense before we paint missing back in.
    missing = (gt_gpu < 0).any(axis=2)
    geno = cp.maximum(gt_gpu, 0).sum(axis=2).astype(cp.int8)
    geno = cp.where(missing, cp.int8(-1), geno)
    # transpose to the (n_indiv, n_var) layout GenotypeMatrix kernels
    # expect; cupy's tiled transpose handles the strided write
    # efficiently.
    geno = cp.ascontiguousarray(geno.T)
    del gt_gpu

    positions = cp.asarray(pos)

    return GenotypeMatrix(
        geno, positions,
        chrom_start=chrom_start, chrom_end=chrom_end,
        sample_sets=sample_sets, n_total_sites=n_total_sites,
        samples=samples, accessible_mask=accessible_mask,
    )
