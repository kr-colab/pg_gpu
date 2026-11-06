"""
GPU-accelerated relatedness and kinship statistics.

Provides GRM (Genetic Relationship Matrix) and IBS (Identity by State)
sharing computed on GPU via CuPy matrix operations.
"""

import numpy as np
import cupy as cp
from typing import Optional, Union


def genetic_relatedness(haplotype_matrix, sample_sets=None, indexes=None, *,
                        centre: bool = True, polarised: bool = False,
                        proportion: bool = True, span_normalize=True,
                        missing_data: str = 'include'):
    """Site-mode genetic relatedness between sample sets.

    Matches ``tskit.TreeSequence.genetic_relatedness`` (site mode): the
    centered per-allele allele-frequency covariance. For sample sets ``i`` and
    ``j`` it is the sum over sites and over every allele ``a`` of
    ``(p_i(a) - pbar(a)) * (p_j(a) - pbar(a))``, where ``p_k(a)`` is allele
    ``a``'s frequency in set ``k`` and ``pbar(a)`` the unweighted mean of
    ``p_k(a)`` across the sample sets. There is no variance standardization.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Sample (haplotype) data. Samples are the unit; individuals or
        populations are expressed by grouping samples into ``sample_sets``.
    sample_sets : list of sequences of int, optional
        Disjoint groups of haplotype (row) indices. Default is one set per
        haplotype, giving the sample-by-sample matrix.
    indexes : list of (int, int), optional
        Pairs of sample-set indices to return. Default returns the full
        symmetric ``(D, D)`` matrix.
    centre : bool
        Center each allele frequency by the across-set mean (True, the default
        and the standard relatedness) or use the raw frequencies (False, the
        noncentred form).
    polarised : bool
        Sum over all alleles including the reference (False, the default;
        note tskit defaults to True) or over derived alleles only (True).
    proportion : bool
        Divide by the number of segregating sites among the analyzed samples
        (the per-site sum of ``num_alleles - 1``), giving a relatedness
        proportion (True, the default, matching tskit). When True,
        ``span_normalize`` has no effect (it cancels between the covariance
        and the segregating-site count). NaN when no site is segregating.
    span_normalize : bool or str
        True (default) divides by the analysis span (see ``get_span``); False
        returns the raw sum. Like tskit's ``span_normalise``, except that when
        the matrix carries an accessible mask the span is the accessible-base
        count over the analyzed region rather than the raw sequence length. The
        eager and streaming paths use the same span source (an ``accessible_bed``
        supplied at load).
    missing_data : str
        'include' (default) uses per-site, per-set valid counts; 'exclude'
        restricts to sites with no missing data among the analyzed samples.

    Returns
    -------
    ndarray, float64
        The ``(D, D)`` matrix, or a 1-D array of values for the requested
        ``indexes``. NaN for a pair of sets with no column called in both.
        0 when no column is called at all and ``proportion=False``.
        ``proportion=True`` gives NaN whenever no site segregates.
    """
    from ._memutil import estimate_variant_chunk_size
    from .streaming_matrix import StreamingHaplotypeMatrix
    from .haplotype_matrix import HaplotypeMatrix

    # This list-of-row-lists parameter never passes the sample_sets setter,
    # so apply the same range and duplicate rules here. Without them an
    # invalid row was silently dropped from the tally -- wrong membership
    # rather than an error. Gate on the accepted input types so anything
    # else still reaches the dispatch below and its TypeError.
    if sample_sets is not None and isinstance(
            haplotype_matrix, (StreamingHaplotypeMatrix, HaplotypeMatrix)):
        from ._warnings import check_sample_set_rows
        for k, rows in enumerate(sample_sets):
            check_sample_set_rows(f"sample_sets[{k}]", list(rows),
                                  haplotype_matrix.num_haplotypes)

    # Only the haplotype stream has the row semantics this statistic
    # needs; a genotype stream must fall through to the TypeError below,
    # the same way grm and ibs reject the wrong stream class.
    if isinstance(haplotype_matrix, StreamingHaplotypeMatrix):
        return _stream_genetic_relatedness(
            haplotype_matrix, sample_sets=sample_sets, indexes=indexes,
            centre=centre, polarised=polarised, proportion=proportion,
            span_normalize=span_normalize, missing_data=missing_data)
    if not isinstance(haplotype_matrix, HaplotypeMatrix):
        raise TypeError(
            "genetic_relatedness requires a HaplotypeMatrix (samples are "
            f"haplotypes); got {type(haplotype_matrix)}")

    if getattr(haplotype_matrix, 'device', 'GPU') == 'CPU':
        haplotype_matrix.transfer_to_gpu()
    hap = haplotype_matrix.haplotypes
    n_hap = hap.shape[0]
    D = n_hap if sample_sets is None else len(sample_sets)

    # missing_data='exclude': keep only sites called in every analyzed sample.
    if missing_data == 'exclude':
        if sample_sets is None:
            covered = cp.ones(n_hap, dtype=bool)
        else:
            covered = cp.zeros(n_hap, dtype=bool)
            for members in sample_sets:
                covered[cp.asarray(members)] = True
        complete = cp.all(hap[covered] >= 0, axis=0)
        hap = hap[:, complete]

    R = cp.zeros((D, D), dtype=cp.float64)
    # Count of (site, allele) columns where both sets of a pair have a call,
    # the per-pair denominator the n_cols rescaling below needs.
    J = cp.zeros((D, D), dtype=cp.float64)
    # Total (site, allele) columns summed into R, for that rescaling.
    n_cols = 0
    n_complete_cols = 0
    # Segregating sites among the analyzed samples: per-site sum of
    # (present alleles - 1), accumulated on device (only when needed).
    seg_cols = 0
    seg_sites = cp.zeros((), dtype=cp.float64)
    if hap.shape[1] > 0:
        K = int(hap.max()) + 1
        # cols + valid_mask (as float) + the two matmul outputs, per chunk.
        chunk = estimate_variant_chunk_size(n_hap, bytes_per_element=8,
                                            n_intermediates=4,
                                            memory_fraction=0.25)
        for start in range(0, hap.shape[1], chunk):
            end = min(start + chunk, hap.shape[1])
            cols, col_site, col_allele, valid_mask = _relatedness_columns(
                hap[:, start:end], sample_sets, K, centre=centre)
            if proportion and col_site.size:
                # col_site is site-major sorted; distinct sites = jumps + 1.
                seg_cols += col_site.size
                seg_sites += 1 + (cp.diff(col_site) != 0).sum()
            if polarised:
                keep = col_allele != 0
                cols = cols[:, keep]
                valid_mask = valid_mask[:, keep]
            n_cols += cols.shape[1]
            R += cols @ cols.T
            # A chunk where every set has a call contributes the same count
            # to every pair, so it is tracked as a scalar instead of a
            # matmul; the validity matmul that remains runs in float32
            # (exact for any realistic chunk size).
            if bool(valid_mask.all()):
                n_complete_cols += cols.shape[1]
            else:
                vm = valid_mask.astype(cp.float32)
                J += (vm @ vm.T).astype(cp.float64)

    if n_cols > 0:
        J += n_complete_cols
        # Project each pair's partial sum (over its own J jointly-called
        # columns) up to the full n_cols, so a pair's value does not shrink
        # just because it called fewer of the analyzed columns.
        R = cp.where(J > 0, R * n_cols / cp.maximum(J, 1.0), cp.nan)
    # else: no column at all -- R stays the all-zero matrix (a sum over no
    # sites); with proportion=True this also means no segregating site, so
    # the denom <= 0 branch below turns it into NaN.

    if proportion:
        # Divide by segregating sites; span_normalise cancels, so it is ignored.
        denom = float(seg_cols - seg_sites)
        if denom > 0:
            R = R / denom
        else:
            R = cp.full_like(R, cp.nan)
    elif span_normalize is not False:
        mode = 'auto' if span_normalize is True else span_normalize
        span = haplotype_matrix.get_span(mode)
        # NaN, not the raw sum, when the span is unmeasurable.
        R = R / span if span > 0 else cp.full_like(R, cp.nan)

    if indexes is None:
        return R.get()
    idx = np.asarray(indexes, dtype=np.intp)
    return R[cp.asarray(idx[:, 0]), cp.asarray(idx[:, 1])].get()


def _stream_genetic_relatedness(streaming_matrix, *, sample_sets, indexes,
                                centre, polarised, proportion, span_normalize,
                                missing_data, block_size=None):
    """Streaming genetic_relatedness: single pass over variant chunks.

    The relatedness is additive over (site, allele) columns and each column's
    centering is local to its site, so one pass over the chunks suffices. The
    (D, D) output is accumulated on host with the set (row) axis tiled into
    blocks, the same scheme as the streaming GRM/IBS, so it holds when D is
    large (the per-haplotype default). Span normalization uses the streaming
    matrix's get_span (accessible mask / n_total_sites / raw span), matching
    the eager path.
    """
    from ._memutil import estimate_indiv_block_size

    n_hap = streaming_matrix.num_haplotypes
    D = n_hap if sample_sets is None else len(sample_sets)

    covered = None
    if missing_data == 'exclude':
        if sample_sets is None:
            covered = cp.ones(n_hap, dtype=bool)
        else:
            covered = cp.zeros(n_hap, dtype=bool)
            for members in sample_sets:
                covered[cp.asarray(members)] = True

    # D is known from the sample sets without peeking a chunk, so an empty
    # source falls through to this zero matrix (relatedness of nothing is 0).
    R = np.zeros((D, D), dtype=np.float64)
    # Count of (site, allele) columns where both sets of a pair have a call,
    # the per-pair denominator the n_cols rescaling below needs.
    J = np.zeros((D, D), dtype=np.float64)
    # Total (site, allele) columns summed into R, for that rescaling.
    n_cols = 0
    n_complete_cols = 0
    seg_cols = 0
    seg_sites = cp.zeros((), dtype=cp.float64)
    if block_size is None:
        # cols + valid_mask (as float) + the two matmul outputs, per block.
        block_size = estimate_indiv_block_size(D, n_intermediates=4)

    # iter_gpu_chunks already attaches any load-time accessible mask, so the
    # chunks here are variant-filtered to accessible sites (matching eager).
    for _, _, chunk in streaming_matrix.iter_gpu_chunks():
        hap = chunk.haplotypes
        if covered is not None:
            complete = cp.all(hap[covered] >= 0, axis=0)
            hap = hap[:, complete]
        if hap.shape[1] == 0:
            continue
        K = int(hap.max()) + 1
        cols, col_site, col_allele, valid_mask = _relatedness_columns(
            hap, sample_sets, K, centre=centre)
        if proportion and col_site.size:
            seg_cols += col_site.size
            seg_sites += 1 + (cp.diff(col_site) != 0).sum()
        if polarised:
            keep = col_allele != 0
            cols = cols[:, keep]
            valid_mask = valid_mask[:, keep]
        n_cols += cols.shape[1]
        # A chunk where every set has a call contributes the same count to
        # every pair, so it is tracked as a scalar instead of a matmul; the
        # validity matmul that remains runs in float32 (exact for any
        # realistic chunk size).
        all_valid = bool(valid_mask.all())
        if all_valid:
            n_complete_cols += cols.shape[1]
        else:
            vm = valid_mask.astype(cp.float32)
        for r0 in range(0, D, block_size):
            r1 = min(r0 + block_size, D)
            R[r0:r1] += (cols[r0:r1] @ cols.T).get()
            if not all_valid:
                J[r0:r1] += (vm[r0:r1] @ vm.T).get()

    if n_cols > 0:
        J += n_complete_cols
        # Project each pair's partial sum up to the full n_cols; see the
        # eager path's identical step.
        R = np.where(J > 0, R * n_cols / np.maximum(J, 1.0), np.nan)
    # else: no column at all -- R stays the all-zero matrix (a sum over no
    # sites); with proportion=True this also means no segregating site, so
    # the denom <= 0 branch below turns it into NaN.

    if proportion:
        denom = seg_cols - float(seg_sites.get())
        if denom > 0:
            R = R / denom
        else:
            R = np.full_like(R, np.nan)
    elif span_normalize is not False:
        # Same span source as the eager path: get_span('auto') prefers the
        # accessible mask / n_total_sites (set at load via accessible_bed) over
        # the raw per-base span, and uses variant-position bounds so it matches
        # the eager matrix built from the same store.
        mode = 'auto' if span_normalize is True else span_normalize
        span = streaming_matrix.get_span(mode)
        # NaN, not the raw sum, when the span is unmeasurable.
        R = R / span if span > 0 else np.full_like(R, np.nan)

    if indexes is None:
        return R
    idx = np.asarray(indexes, dtype=np.intp)
    return R[idx[:, 0], idx[:, 1]]


def grm(genotype_matrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include') -> np.ndarray:
    """Compute the Genetic Relationship Matrix (GRM).

    The GRM measures genome-wide allele sharing between all pairs of
    individuals, standardized by allele frequencies. Equivalent to
    GCTA's --make-grm or PLINK2's --make-rel.

    Formula:
        A_ij = (1/M) * sum_k[(g_ik - 2*p_k)(g_jk - 2*p_k) / (2*p_k*(1-p_k))]

    where g is genotype (0/1/2), p is allele frequency, M is number of sites.

    This is a diploid, biallelic estimator, so it takes a GenotypeMatrix (or
    a streaming genotype matrix), which is diploid and biallelic by
    construction. For relatedness between haplotypes or arbitrary sample sets,
    or on multiallelic data, use ``genetic_relatedness``.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix or StreamingGenotypeMatrix
        Diploid genotypes (0/1/2).
    population : str or list, optional
        Population subset.
    missing_data : str
        'include' - use per-site allele frequencies from non-missing data.
        'exclude' - restrict to sites with no missing data.

    Returns
    -------
    ndarray, float64, shape (n_individuals, n_individuals)
        Symmetric GRM. Diagonal entries are individual inbreeding
        coefficients + 1. Off-diagonal entries are pairwise relatedness.
        NaN for a pair with no polymorphic SNP called in both, and for the
        whole matrix when no SNP is polymorphic.
    """
    from ._memutil import estimate_variant_chunk_size
    from .streaming_matrix import _StreamingMatrixBase, StreamingHaplotypeMatrix
    from .haplotype_matrix import HaplotypeMatrix
    if isinstance(genotype_matrix, (HaplotypeMatrix, StreamingHaplotypeMatrix)):
        raise TypeError(
            "grm is the diploid biallelic GCTA GRM and requires a "
            "GenotypeMatrix. For haplotype or sample-set relatedness (and "
            "multiallelic data) use genetic_relatedness; to run the GCTA GRM "
            "on haplotype data convert it with "
            "GenotypeMatrix.from_haplotype_matrix first.")
    if isinstance(genotype_matrix, _StreamingMatrixBase):
        return _stream_grm(genotype_matrix,
                            population=population,
                            missing_data=missing_data)

    geno, n_ind = _get_genotype_data(genotype_matrix, population)
    n_var = geno.shape[1]

    # Allele frequency from genotypes, chunked to avoid large temporaries.
    dac = cp.zeros(n_var, dtype=cp.int32)
    n_valid = cp.zeros(n_var, dtype=cp.int32)
    freq_chunk = max(1, n_var // 20)
    for s in range(0, n_var, freq_chunk):
        e = min(s + freq_chunk, n_var)
        gc = geno[:, s:e]
        v = gc >= 0
        n_valid[s:e] = cp.sum(v, axis=0)
        dac[s:e] = cp.sum(cp.where(v, gc, 0), axis=0)
        del gc, v
    p = cp.where(n_valid > 0, dac.astype(cp.float64) / (2.0 * n_valid.astype(cp.float64)), 0.0)

    if missing_data == 'exclude':
        complete = n_valid == n_ind
        poly = complete & (p > 0) & (p < 1)
    else:
        poly = (p > 0) & (p < 1)

    poly_idx = cp.where(poly)[0]
    n_snps_used = len(poly_idx)
    if n_snps_used == 0:
        return np.full((n_ind, n_ind), np.nan, dtype=np.float64)

    p_poly = p[poly_idx]
    del p, dac, n_valid, poly

    # Chunked GRM: accumulate (std_chunk @ std_chunk.T) over variant chunks,
    # plus the count of jointly-called SNPs per pair (N), so a pair is
    # normalized by its own called-SNP count rather than the dataset total.
    # A complete chunk is tracked as a scalar instead of a matmul; the
    # validity matmul only runs for a chunk with a missing call, in float32
    # (exact for any realistic chunk size).
    # Peak per-chunk: 1 float64 array (standardized genotypes) + matmul workspace.
    chunk_size = estimate_variant_chunk_size(n_ind, bytes_per_element=8,
                                             n_intermediates=4,
                                             memory_fraction=0.25)
    A = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    N = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    n_complete = 0

    for start in range(0, n_snps_used, chunk_size):
        end = min(start + chunk_size, n_snps_used)
        idx = poly_idx[start:end]
        p_chunk = p_poly[start:end]

        g_chunk = geno[:, idx]
        valid = (g_chunk >= 0)
        all_valid = bool(valid.all())
        g = cp.where(valid, g_chunk, 0).astype(cp.float64)
        del g_chunk

        scale = cp.sqrt(2.0 * p_chunk * (1.0 - p_chunk))
        # std = (g - 2p) * valid / scale, computed in-place
        g -= 2.0 * p_chunk
        g *= valid
        g /= scale
        A += g @ g.T
        del g
        if all_valid:
            n_complete += end - start
        else:
            vf = valid.astype(cp.float32)
            N += (vf @ vf.T).astype(cp.float64)
            del vf
        del valid, scale

    N += n_complete
    A = cp.where(N > 0, A / cp.maximum(N, 1.0), cp.nan)
    return A.get()


def ibs(genotype_matrix,
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include') -> np.ndarray:
    """Compute pairwise IBS (Identity by State) proportions.

    IBS measures the proportion of alleles shared identical by state between
    all pairs of individuals. Equivalent to PLINK's --distance ibs matrix. For
    each site the shared count is (2 - abs(g_i - g_j)) and the matrix is the
    summed shared count over jointly-called sites divided by twice that site
    count.

    This is a diploid, biallelic estimator (the shared count is the diploid
    max-matching of two genotypes), so it takes a GenotypeMatrix (or a streaming
    genotype matrix), which is diploid and biallelic by construction. For the
    general allele-sharing quantity between arbitrary sample sets (haplotypes,
    individuals, or populations), and on multiallelic data, use
    ``genetic_relatedness`` with ``centre=False``: the noncentred relatedness is
    the allele-match probability between sample sets, the sample-set
    generalization of allele sharing.

    Parameters
    ----------
    genotype_matrix : GenotypeMatrix or StreamingGenotypeMatrix
        Diploid genotypes (0/1/2).
    population : str or list, optional
        Population subset.
    missing_data : str
        'include' - use jointly non-missing sites per pair.
        'exclude' - restrict to sites with no missing data; every pair is
        NaN when no site is complete.

    Returns
    -------
    ndarray, float64, shape (n_individuals, n_individuals)
        Symmetric IBS matrix. Values in [0, 1] where 1 = identical.
        Diagonal is always 1. NaN for a pair of individuals with no site
        called in both, which has no IBS proportion.
    """
    from ._memutil import estimate_variant_chunk_size
    from .streaming_matrix import _StreamingMatrixBase, StreamingHaplotypeMatrix
    from .haplotype_matrix import HaplotypeMatrix
    if isinstance(genotype_matrix, (HaplotypeMatrix, StreamingHaplotypeMatrix)):
        raise TypeError(
            "ibs is the diploid biallelic PLINK IBS and requires a "
            "GenotypeMatrix. For allele sharing between haplotypes or arbitrary "
            "sample sets (and multiallelic data) use genetic_relatedness with "
            "centre=False; to run the diploid IBS on haplotype data convert it "
            "with GenotypeMatrix.from_haplotype_matrix first.")
    if isinstance(genotype_matrix, _StreamingMatrixBase):
        return _stream_ibs(genotype_matrix,
                            population=population,
                            missing_data=missing_data)

    geno, n_ind = _get_genotype_data(genotype_matrix, population)
    n_var = geno.shape[1]
    # Peak per-chunk: 1 float64 indicator + matmul workspace.
    chunk_size = estimate_variant_chunk_size(n_ind, bytes_per_element=8,
                                             n_intermediates=4,
                                             memory_fraction=0.25)

    ibs2 = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    ibs1 = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    n_joint = cp.zeros((n_ind, n_ind), dtype=cp.float64)
    n_complete = 0

    for start in range(0, n_var, chunk_size):
        end = min(start + chunk_size, n_var)
        g_chunk = geno[:, start:end]
        valid_b = (g_chunk >= 0)
        g = cp.where(valid_b, g_chunk, 0).astype(cp.float64)
        del g_chunk

        if missing_data == 'exclude':
            site_complete = cp.all(valid_b, axis=0)
            valid_b = valid_b[:, site_complete]
            g = g[:, site_complete]

        # A complete chunk (always true under 'exclude') is tracked as a
        # scalar instead of a matmul; the validity matmul that remains runs
        # in float32 (exact for any realistic chunk size).
        if bool(valid_b.all()):
            n_complete += g.shape[1]
        else:
            vf = valid_b.astype(cp.float32)
            n_joint += (vf @ vf.T).astype(cp.float64)
            del vf
        valid = valid_b.astype(cp.float64)
        del valid_b

        # ibs2 (both alleles shared) and ibs1 (one shared) via genotype-state
        # indicators, one at a time to avoid materializing all three together.
        ind_curr = (g == 0) * valid
        for gval in (0, 1, 2):
            if gval < 2:
                ind_next = (g == (gval + 1)) * valid
            ibs2 += ind_curr @ ind_curr.T
            if gval < 2:
                cross = ind_curr @ ind_next.T
                ibs1 += cross + cross.T
                ind_curr = ind_next

        del g, valid

    n_joint += n_complete
    ibs_mat = cp.where(n_joint > 0,
                        (2.0 * ibs2 + ibs1) / (2.0 * n_joint),
                        cp.nan)
    cp.fill_diagonal(ibs_mat, 1.0)
    return ibs_mat.get()


# ---------------------------------------------------------------------------
# genetic_relatedness: per-set, per-allele centered frequency columns
# ---------------------------------------------------------------------------

def _relatedness_columns(hap, sample_sets, n_alleles, centre=True):
    """Per-set, per-allele frequency columns for genetic_relatedness.

    Builds a matrix whose row ``k`` is sample set ``k``'s per-allele frequency.
    Writing ``p_k(a)`` for the frequency of allele ``a`` in set ``k`` and
    ``pbar(a)`` for the mean of ``p_k(a)`` over the sets that have a call at
    that site, with ``centre`` the columns are ``p_k(a) - pbar(a)`` and the
    relatedness of a set pair ``(i, j)`` is the sum over every allele and site
    of ``(p_i(a) - pbar(a)) * (p_j(a) - pbar(a))``, i.e. ``(C @ C.T)[i, j]``
    (the product is formed in a later step); without ``centre`` the columns
    are the raw ``p_k(a)`` and ``C @ C.T`` is the noncentred form. All alleles
    including the reference are represented, and there is no variance
    standardization; a caller wanting derived alleles only drops the
    ``col_allele == 0`` columns.

    A set with no call at a site has an undefined ``p_k(a)``, so it is
    excluded from the ``pbar(a)`` mean and its own column entry is forced to
    0 (contributing nothing to any product), rather than entering either as a
    fake 0 frequency; ``valid_mask`` names which sets are defined at each
    column so a caller can tell, per pair, whether any column counted at all.

    Each present ``(site, allele)`` pair among the analyzed samples is one
    column; columns are ordered site-major, allele-minor. "Present" is relative
    to the samples that appear in some set, so passing a subset of samples
    restricts the alleles considered.

    Parameters
    ----------
    hap : cupy.ndarray, int8, shape (n_hap, n_var)
        Allele indices (0 reference, 1.. alternate), -1 missing.
    sample_sets : list of 1D index arrays, or None
        Disjoint groups of haplotype indices. None means one set per haplotype
        (the tskit-native singleton grouping, ``D = n_hap``).
    n_alleles : int
        Global allele-index width ``K`` so allele ``a`` aligns across sets.
    centre : bool
        Subtract the across-set mean ``pbar(a)`` from each column (True), or
        leave the raw per-set frequencies (False, the noncentred form).

    Returns
    -------
    cols : cupy.ndarray, float64, shape (D, n_cols)
    col_site : cupy.ndarray, int64, shape (n_cols,)
        Site index each column belongs to.
    col_allele : cupy.ndarray, int64, shape (n_cols,)
        Allele index each column represents (0 is the reference).
    valid_mask : cupy.ndarray, bool, shape (D, n_cols)
        Whether set k has a call at that column's site.

    Notes
    -----
    Per-set frequencies use per-site valid counts, so a missing entry lowers only
    its own set's denominator. The exact missing-data policy (a set with no valid
    sample at a site, and the include/exclude modes) is applied by the caller; on
    complete data every set's denominator is its size.
    """
    n_hap, n_var = hap.shape
    K = int(n_alleles)
    valid = hap >= 0

    if sample_sets is None:
        M = None
        D = n_hap
        covered = valid                                  # every haplotype is a set
    else:
        D = len(sample_sets)
        M = cp.zeros((D, n_hap), dtype=cp.float64)
        for k, members in enumerate(sample_sets):
            M[k, cp.asarray(members)] = 1.0
        covered = valid & (M.sum(axis=0) > 0)[:, None]

    # Present (site, allele) pairs among the analyzed samples -> one argwhere sync.
    pooled_ac = cp.zeros((n_var, K), dtype=cp.int64)
    for a in range(K):
        pooled_ac[:, a] = (covered & (hap == a)).sum(axis=0, dtype=cp.int64)
    pairs = cp.argwhere(pooled_ac > 0)
    col_site = pairs[:, 0]
    col_allele = pairs[:, 1]

    # Per-set count and valid count for each present column.
    hs = hap[:, col_site]                                # (n_hap, n_cols)
    carries = ((hs == col_allele[None, :]) & valid[:, col_site]).astype(cp.float64)
    valid_cols = valid[:, col_site].astype(cp.float64)
    if M is None:
        cnt = carries                                    # (D, n_cols), D == n_hap
        nvalid = valid_cols
    else:
        cnt = M @ carries                                # (D, n_cols)
        nvalid = M @ valid_cols

    valid_mask = nvalid > 0
    p = cnt / cp.maximum(nvalid, 1.0)                    # per-set frequency, safe where invalid
    if centre:
        pbar = cp.nanmean(cp.where(valid_mask, p, cp.nan), axis=0, keepdims=True)
        cols = cp.where(valid_mask, p - pbar, 0.0)
    else:
        cols = cp.where(valid_mask, p, 0.0)
    return cols, col_site, col_allele, valid_mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_genotype_data(matrix, population=None):
    """Get the diploid dosage matrix (n_individuals, n_variants) on GPU.

    Callers reach here only with a GenotypeMatrix: ``grm`` and ``ibs`` reject
    haplotype input up front, so haplotypes become dosages through the one
    conversion in ``GenotypeMatrix.from_haplotype_matrix``, which counts the
    alt allele and drops sites carrying more than two.
    """
    from .genotype_matrix import GenotypeMatrix
    from ._utils import population_rows

    if population is not None:
        matrix = population_rows(matrix, population)

    if hasattr(matrix, 'device') and matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    if isinstance(matrix, GenotypeMatrix):
        return matrix.genotypes, matrix.genotypes.shape[0]
    raise TypeError(f"Expected GenotypeMatrix, got {type(matrix)}")  # pragma: no cover


def _stream_ibs(streaming_matrix, *, population, missing_data,
                block_size=None):
    from ._memutil import estimate_indiv_block_size

    chunks = iter(streaming_matrix.iter_gpu_chunks())
    first = next(chunks, None)
    if first is None:
        # No variants at all, so no pair shares a called site; mirror the
        # eager result so callers don't have to special-case the empty
        # source.
        n_ind = _stream_n_ind(streaming_matrix, population)
        ibs_mat = np.full((n_ind, n_ind), np.nan)
        np.fill_diagonal(ibs_mat, 1.0)
        return ibs_mat

    geno_first, n_ind = _get_genotype_data(first[2], population)
    if block_size is None:
        # block_ibs2 + block_ibs1 + transient matmul output + slack.
        block_size = estimate_indiv_block_size(n_ind, n_intermediates=4)

    ibs2_host = np.zeros((n_ind, n_ind), dtype=np.float64)
    ibs1_host = np.zeros((n_ind, n_ind), dtype=np.float64)
    n_joint_host = np.zeros((n_ind, n_ind), dtype=np.float64)

    _accumulate_ibs_chunk(geno_first, ibs2_host, ibs1_host, n_joint_host,
                          block_size=block_size, missing_data=missing_data)
    del geno_first
    for _, _, chunk in chunks:
        geno, _ = _get_genotype_data(chunk, population)
        _accumulate_ibs_chunk(geno, ibs2_host, ibs1_host, n_joint_host,
                              block_size=block_size, missing_data=missing_data)
        del geno

    ibs_mat = np.divide(2.0 * ibs2_host + ibs1_host, 2.0 * n_joint_host,
                        out=np.full_like(n_joint_host, np.nan),
                        where=n_joint_host > 0)
    np.fill_diagonal(ibs_mat, 1.0)
    return ibs_mat


def _accumulate_ibs_chunk(geno, ibs2_host, ibs1_host, n_joint_host, *,
                          block_size, missing_data):
    """Add one variant-chunk's contribution to the host accumulators.

    ``geno`` is (n_ind, n_var_chunk) on GPU. The genotype-state indicators are
    built once per chunk; the row-block loop tiles the individual axis so only
    one block's (block_size, n_ind) matmul output sits on the GPU at a time
    before it is added to the host accumulators.
    """
    valid_b = (geno >= 0)
    g = cp.where(valid_b, geno, 0).astype(cp.float64)
    if missing_data == 'exclude':
        site_complete = cp.all(valid_b, axis=0)
        valid_b = valid_b[:, site_complete]
        g = g[:, site_complete]
        if g.shape[1] == 0:
            return

    n_ind = g.shape[0]
    # A complete chunk (always true under 'exclude') is tracked as a scalar
    # instead of a matmul; the validity matmul that remains runs in float32
    # (exact for any realistic chunk size).
    if bool(valid_b.all()):
        n_joint_host += g.shape[1]
    else:
        vf = valid_b.astype(cp.float32)
        for r0 in range(0, n_ind, block_size):
            r1 = min(r0 + block_size, n_ind)
            n_joint_host[r0:r1] += (vf[r0:r1] @ vf.T).get()
        del vf

    valid_full = valid_b.astype(cp.float64)
    del valid_b
    ind_curr = (g == 0) * valid_full
    for gval in (0, 1, 2):
        if gval < 2:
            ind_next = (g == (gval + 1)) * valid_full
        for r0 in range(0, n_ind, block_size):
            r1 = min(r0 + block_size, n_ind)
            ibs2_host[r0:r1] += (ind_curr[r0:r1] @ ind_curr.T).get()
            if gval < 2:
                ibs1_host[r0:r1] += (ind_curr[r0:r1] @ ind_next.T).get()
                ibs1_host[r0:r1] += (ind_next[r0:r1] @ ind_curr.T).get()
        del ind_curr
        if gval < 2:
            ind_curr = ind_next


def _stream_n_ind(streaming_matrix, population):
    """Best-effort individual count for the empty-source fallback. Uses
    the streaming source's diploid count (halved for haplotype-layout)
    and a population subset if requested."""
    from .streaming_matrix import StreamingHaplotypeMatrix
    is_hap = isinstance(streaming_matrix, StreamingHaplotypeMatrix)
    if population is None:
        return (streaming_matrix.num_haplotypes // 2 if is_hap
                else streaming_matrix.num_individuals)
    pop_set = streaming_matrix.sample_sets[population]
    return len(pop_set) // 2 if is_hap else len(pop_set)


# ---------------------------------------------------------------------------
# Streaming GRM: two-pass + indiv-axis row-block tile
# ---------------------------------------------------------------------------
#
# GRM standardizes each variant by its allele frequency before forming the
# (n_ind, n_ind) outer product, so the eager kernel computes p over the
# whole chromosome up front. Two-pass form for the streaming path:
# pass 1 walks chunks to accumulate per-variant DAC + n_valid into
# chromosome-wide host arrays; pass 2 walks the same chunks, slicing the
# precomputed p into chunk-local pieces, standardizing on GPU, and
# accumulating A = g @ g.T on host with row-block tiling on the
# individual axis (same scheme as the IBS streaming path).

def _stream_grm(streaming_matrix, *, population, missing_data,
                block_size=None):
    from ._memutil import estimate_indiv_block_size

    # Pass 1: per-variant DAC / n_valid into per-chunk host arrays.
    dac_parts = []
    nvalid_parts = []
    n_ind = None
    for _, _, chunk in streaming_matrix.iter_gpu_chunks():
        geno, n_ind = _get_genotype_data(chunk, population)
        valid = (geno >= 0)
        nvalid_parts.append(
            cp.sum(valid, axis=0, dtype=cp.int32).get()
        )
        dac_parts.append(
            cp.sum(cp.where(valid, geno, 0), axis=0, dtype=cp.int32).get()
        )
        del geno, valid

    if n_ind is None:
        # No chunks at all.
        n_ind = _stream_n_ind(streaming_matrix, population)
        return np.full((n_ind, n_ind), np.nan, dtype=np.float64)

    chunk_var_counts = [p.size for p in dac_parts]
    chunk_offsets = np.cumsum([0] + chunk_var_counts)
    dac_chrom = np.concatenate(dac_parts) if dac_parts else np.zeros(0, np.int32)
    nvalid_chrom = (np.concatenate(nvalid_parts)
                    if nvalid_parts else np.zeros(0, np.int32))

    p_chrom = np.where(
        nvalid_chrom > 0,
        dac_chrom.astype(np.float64) / (2.0 * nvalid_chrom.astype(np.float64)),
        0.0,
    )
    if missing_data == 'exclude':
        poly_mask = (nvalid_chrom == n_ind) & (p_chrom > 0) & (p_chrom < 1)
    else:
        poly_mask = (p_chrom > 0) & (p_chrom < 1)
    n_snps_used = int(poly_mask.sum())
    if n_snps_used == 0:
        return np.full((n_ind, n_ind), np.nan, dtype=np.float64)

    if block_size is None:
        # standardized g + validity float + matmul output + slack.
        block_size = estimate_indiv_block_size(n_ind, n_intermediates=4)

    A_host = np.zeros((n_ind, n_ind), dtype=np.float64)
    # Count of jointly-called polymorphic SNPs per pair, so a pair is
    # normalized by its own called-SNP count rather than the dataset total.
    N_host = np.zeros((n_ind, n_ind), dtype=np.float64)

    # Pass 2: standardize each chunk and accumulate the outer product.
    # Chunk iteration order must match pass 1 so the chunk-local slice
    # of p_chrom lines up with the chunk's variants.
    for chunk_i, (_, _, chunk) in enumerate(
            streaming_matrix.iter_gpu_chunks()):
        c_lo, c_hi = chunk_offsets[chunk_i], chunk_offsets[chunk_i + 1]
        chunk_poly = poly_mask[c_lo:c_hi]
        if not chunk_poly.any():
            continue
        chunk_p_full = p_chrom[c_lo:c_hi]

        geno, _ = _get_genotype_data(chunk, population)
        if chunk_poly.all():
            # At biobank-scale MAF distributions essentially every site
            # is polymorphic; bypass the fancy index in that case so
            # the matmul sees a contiguous column block.
            chunk_p_gpu = cp.asarray(chunk_p_full)
            g = geno
        else:
            chunk_p_gpu = cp.asarray(chunk_p_full[chunk_poly])
            chunk_idx_gpu = cp.asarray(np.where(chunk_poly)[0],
                                        dtype=cp.int32)
            g = geno[:, chunk_idx_gpu]
        valid = (g >= 0)
        all_valid = bool(valid.all())
        n_cols = g.shape[1]
        g_std = cp.where(valid, g, 0).astype(cp.float64)
        del g
        scale = cp.sqrt(2.0 * chunk_p_gpu * (1.0 - chunk_p_gpu))
        g_std -= 2.0 * chunk_p_gpu
        g_std *= valid
        g_std /= scale
        del scale
        # A complete chunk is tracked as a scalar instead of a matmul; the
        # validity matmul that remains runs in float32 (exact for any
        # realistic chunk size).
        if not all_valid:
            vf = valid.astype(cp.float32)
        del valid

        for r0 in range(0, n_ind, block_size):
            r1 = min(r0 + block_size, n_ind)
            A_host[r0:r1] += (g_std[r0:r1] @ g_std.T).get()
            if all_valid:
                N_host[r0:r1] += n_cols
            else:
                N_host[r0:r1] += (vf[r0:r1] @ vf.T).get()
        del g_std, geno
        if not all_valid:
            del vf

    A_host = np.where(N_host > 0, A_host / np.maximum(N_host, 1.0), np.nan)
    return A_host
