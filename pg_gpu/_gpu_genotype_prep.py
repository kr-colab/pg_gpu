"""GPU-side preparation of raw VCZ-shape genotype blocks for HaplotypeMatrix
and GenotypeMatrix.

The VCZ ``call_genotype`` array is ``(n_var, n_samples, ploidy)`` int8. The
two matrix classes consume it differently:

* ``HaplotypeMatrix`` wants ``(n_hap, n_var)`` int8 in the canonical row
  order: for a diploid store, hap ``2i`` carries sample i's first copy and
  hap ``2i+1`` its second; for a haploid store, hap ``i`` is sample i.
  ``build_haplotype_matrix`` does that interleave + transpose on the GPU.
* ``GenotypeMatrix`` wants ``(n_indiv, n_var)`` int8 dosages (0/1/2 with
  ``-1`` for missing). ``build_genotype_matrix`` counts each site's alt
  allele on the GPU and propagates missing; it takes a diploid block only.

Both helpers exist to keep the
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


# Exchange allele code 0 with the ancestral allele's code at each site. A
# negative value is a missing call or ploidy padding and stays as it is.
_SWAP_ANCESTRAL = cp.RawKernel(r'''
extern "C" __global__
void swap_ancestral(signed char* haps, const signed char* code,
                    const long long n_var, const long long size) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    signed char a = code[idx % n_var];
    if (a <= 0) return;
    signed char h = haps[idx];
    if (h == 0) haps[idx] = a;
    else if (h == a) haps[idx] = 0;
}
''', 'swap_ancestral')


def swap_ancestral(haps, codes):
    """Make allele code 0 the ancestral allele, in place.

    Parameters
    ----------
    haps : cupy.ndarray, int8, shape (n_hap, n_var), C-contiguous
        Allele codes; negative is missing.
    codes : array-like, int8, shape (n_var,)
        The code of the ancestral allele at each site. At a site with code
        ``a > 0``, codes 0 and ``a`` change places, so a multiallelic site
        keeps its other codes. A site with code 0 or -1 (unknown) does not
        change.
    """
    if haps.dtype != cp.int8 or not haps.flags.c_contiguous:
        raise ValueError("haps must be a C-contiguous int8 array")
    n_var = haps.shape[1]
    codes = cp.asarray(codes)
    if codes.shape != (n_var,):
        raise ValueError(
            f"one ancestral code per variant is needed: {n_var} variants, "
            f"codes of shape {codes.shape}")
    # A code outside int8 would wrap in the cast below and name some other
    # allele, with no error.
    if codes.dtype.kind not in 'iu' or (
            codes.size and not (-128 <= int(codes.min()) and int(codes.max()) <= 127)):
        raise ValueError(
            "ancestral codes must be integers that fit int8 (-128 to 127)")
    codes = cp.ascontiguousarray(codes.astype(cp.int8, copy=False))
    size = haps.size
    if size:
        _SWAP_ANCESTRAL(((size + 255) // 256,), (256,),
                        (haps, codes, cp.int64(n_var), cp.int64(size)))
    return haps


def build_haplotype_matrix(gt, pos, *,
                           phased,
                           chrom_start=None, chrom_end=None,
                           sample_sets=None, n_total_sites=None,
                           samples=None, accessible_mask=None,
                           allow_empty = False, ancestral_codes=None):
    """Build a HaplotypeMatrix from a raw VCZ-style genotype block.

    Parameters
    ----------
    gt : ndarray, shape (n_var, n_samples, ploidy)
        Raw call_genotype block of allele indices (0 = reference, 1.. =
        alternate), ``-1`` missing. Indices are preserved as-is (the
        per-allele haplotype statistics are multiallelic-capable). Host or
        device. The ploidy axis is 1 or 2, and the matrix takes its
        ``ploidy`` from it.
    pos : ndarray, shape (n_var,)
        Variant positions. Host or device.
    ancestral_codes : array-like, int8, shape (n_var,), optional
        The code of the ancestral allele at each site; the block is
        recoded with ``swap_ancestral`` so that code 0 is ancestral.

    The remaining kwargs are forwarded to ``HaplotypeMatrix.__init__``.

    Returns
    -------
    HaplotypeMatrix
        With haplotypes on the GPU in ``(n_hap, n_var)`` layout.
    """
    from .haplotype_matrix import HaplotypeMatrix

    if gt.ndim != 3 or gt.shape[2] not in (1, 2):
        raise ValueError(
            f"gt must have shape (n_var, n_samples, ploidy) with ploidy 1 "
            f"or 2; got {gt.shape}"
        )
    if gt.shape[0] != pos.shape[0]:
        raise ValueError(
            f"gt and pos disagree on n_var: gt={gt.shape[0]}, pos={pos.shape[0]}"
        )

    n_var, n_samples, ploidy = gt.shape
    gt_gpu = cp.asarray(gt)

    # The ploidy axis is last and varies fastest, so reshaping to
    # (n_var, ploidy * n_samples) interleaves the gametes of each sample;
    # the transpose then yields the canonical row order, sample i at rows
    # ploidy * i onward (2i and 2i+1 for a diploid).
    haps = cp.ascontiguousarray(
        gt_gpu.reshape(n_var, ploidy * n_samples).T
    )
    del gt_gpu
    if ancestral_codes is not None:
        swap_ancestral(haps, ancestral_codes)

    positions = cp.asarray(pos)

    return HaplotypeMatrix(
        haps, positions,
        chrom_start=chrom_start, chrom_end=chrom_end,
        sample_sets=sample_sets, n_total_sites=n_total_sites,
        samples=samples, accessible_mask=accessible_mask,
        allow_empty=allow_empty, phased=phased, ploidy=ploidy,
    )


def build_genotype_matrix(gt, pos, *,
                          chrom_start=None, chrom_end=None,
                          sample_sets=None, n_total_sites=None,
                          samples=None, accessible_mask=None,
                          allow_empty = False):
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
        the row/chunk alignment the streaming path relies on. The number
        of such recoded sites is stashed on the result as
        ``_n_multiallelic_recoded`` for the caller to surface as a
        BiallelicOnlyWarning (once per load).
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
        allow_empty=allow_empty,
    )
    gm._n_multiallelic_recoded = n_multiallelic
    return gm
