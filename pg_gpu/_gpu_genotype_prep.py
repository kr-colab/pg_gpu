"""GPU-side preparation of raw VCZ-shape genotype blocks for HaplotypeMatrix.

The VCZ ``call_genotype`` array is ``(n_var, n_dip, 2)`` int8. HaplotypeMatrix
expects ``(n_hap, n_var)`` int8 with haps ``0..n_dip-1`` carrying ploidy 0 and
haps ``n_dip..2*n_dip-1`` carrying ploidy 1. The host-side numpy version of
that transform allocates a fresh ``2 * chunk_bytes`` of host memory and runs
a single-threaded strided int8 copy; on a 1 Mb / 200k-haplotype block that is
~60 s of wall and a 56 GB allocation. The GPU version uses cupy's tiled
transpose kernel after a single PCIe upload and runs in seconds with a fixed
memory footprint.

Missing / multiallelic rows (gt = -1) are preserved -- the downstream pg_gpu
kernels handle them via their ``'include'`` / ``'exclude'`` missing-data
modes, so dropping them here would silently change semantics for callers
that opted into 'include'.
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
