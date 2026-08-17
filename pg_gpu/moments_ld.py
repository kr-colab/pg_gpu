"""
Integration layer: GPU-accelerated LD statistics for moments inference.

Drop-in replacement for moments.LD.Parsing.compute_ld_statistics() using
pg_gpu's GPU kernels. Output format is identical to moments, so downstream
inference (bootstrap_data, optimize_log_lbfgsb, Godambe) works unchanged.

Supports 1-4 populations.

Usage:
    from pg_gpu.moments_ld import compute_ld_statistics

    ld_stats = compute_ld_statistics(
        "data.vcf", rec_map_file="map.txt", pop_file="pops.txt",
        pops=["pop0", "pop1"], r_bins=r_bins,
    )
    mv = moments.LD.Parsing.bootstrap_data({0: ld_stats})
"""

import numpy as np
import cupy as cp

from .haplotype_matrix import HaplotypeMatrix
from .ld_pipeline import (
    iter_pairs_within_distance as _iter_pairs_within_distance,
    compute_counts_for_pairs as _compute_counts_for_pairs,
    compute_genotype_counts_for_pairs as _compute_genotype_counts_for_pairs,
    compute_two_pop_statistics_batch as _compute_two_pop_statistics_batch,
    estimate_ld_chunk_size as _estimate_ld_chunk_size,
    ld_names as _ld_names,
    het_names as _het_names,
    generate_stat_specs as _generate_stat_specs,
)
from .genotype_kernels import compute_multi_pop_statistics_batch_geno
from .haplotype_kernels import compute_multi_pop_statistics_batch_hap
from .genotype_matrix import GenotypeMatrix
from . import ld_statistics


def compute_ld_statistics(
    vcf_file=None, rec_map_file=None, pop_file=None, pops=None,
    r_bins=None, bp_bins=None, use_genotypes=True,
    report=True, haplotype_matrix=None,
    genotype_matrix=None, accessible_bed=None,
    pop_assignment=None,
):
    """GPU LD statistics in the moments.LD.Parsing.compute_ld_statistics output format.

    Accepts the same arguments as the moments version, but restricts to biallelic
    segregating sites with arbitrary allele coding (two present alleles): {0,2}
    and reference-absent {1,2} sites are kept and recoded, whereas moments'
    is_biallelic_01 drops them. Results match moments only when the input is
    already {0,1}-coded. The example below mirrors the moments call::

        # moments (CPU):
        import moments.LD
        ld_stats = moments.LD.Parsing.compute_ld_statistics(
            vcf_file="data.vcf.gz",
            rec_map_file="rec_map.txt",
            pop_file="pops.txt",
            pops=["popA", "popB"],
            r_bins=[0, 1e-6, 2e-6, 5e-6],
        )

        # pg_gpu (GPU, same call signature):
        from pg_gpu.moments_ld import compute_ld_statistics
        ld_stats = compute_ld_statistics(
            vcf_file="data.vcf.gz",
            rec_map_file="rec_map.txt",
            pop_file="pops.txt",
            pops=["popA", "popB"],
            r_bins=[0, 1e-6, 2e-6, 5e-6],
        )

    The returned dict has the same structure (keys 'bins', 'sums', 'stats',
    'pops') and can be passed directly to moments inference functions.

    .. important::
       ``use_genotypes`` defaults to ``True``, matching
       ``moments.LD.Parsing.compute_ld_statistics``. This is deliberate:
       the two functions compute *different estimators*
       (``True`` = unphased 9-way genotype counts, ``False`` = phased
       4-way haplotype counts), which agree in expectation but give
       different values on finite data. Matching the moments default means
       a bare import swap reproduces moments exactly. If your data are
       phased and you want the haplotype estimator, you must pass
       ``use_genotypes=False`` here **and** to moments -- setting it on
       only one side silently compares two different estimators (a
       discrepancy that looks like a bug but is not).

    Parameters
    ----------
    vcf_file : str, optional
        Path to VCF file. Not needed if haplotype_matrix/genotype_matrix provided.
    rec_map_file : str, optional
        Recombination map (tab-delimited: pos, Map(cM)). Required with r_bins.
    pop_file, pop_assignment : str, dict, numpy.ndarray, list, or False, optional
        Sample-to-population assignment in any form
        ``normalize_pop_input`` accepts (path / sample-to-pop dict /
        labels array / zarr key name / ``False`` to disable). The
        two kwargs are aliases; ``pop_file`` is the
        moments-compatible spelling, ``pop_assignment`` is the
        spelling the rest of pg_gpu uses. Passing both at once is
        an error.
    pops : list of str
        Population names (1-4). Defaults to ['pop0', 'pop1'].
    r_bins : array-like, optional
        Recombination rate bin edges (Morgans).
    bp_bins : array-like, optional
        Base-pair distance bin edges (alternative to r_bins).
    use_genotypes : bool, default True
        Selects the estimator, and defaults to ``True`` to match
        ``moments.LD.Parsing.compute_ld_statistics``. If True, use diploid
        genotype counts (9-way), the correct estimator for unphased data.
        If False, use haplotype counts (4-way), which requires phased data.
        The two estimators differ on finite data; keep this flag identical
        on both the pg_gpu and moments sides of any comparison.
    report : bool
        Print progress.
    haplotype_matrix : HaplotypeMatrix, optional
        Pre-loaded HaplotypeMatrix (skips VCF loading and GPU transfer).
    genotype_matrix : GenotypeMatrix, optional
        Pre-loaded GenotypeMatrix (skips VCF loading and GPU transfer).
    accessible_bed : str, optional
        Path to a BED file defining accessible/callable regions. Variants
        at inaccessible positions are removed before computing statistics.

    Returns
    -------
    dict with keys 'bins', 'sums', 'stats', 'pops' (moments format).
    """
    if pop_file is not None and pop_assignment is not None:
        raise TypeError(
            "pass only one of pop_file or pop_assignment; they are "
            "aliases for the same argument."
        )
    if pop_assignment is not None:
        pop_file = pop_assignment

    if pops is None:
        pops = ['pop0', 'pop1']
    num_pops = len(pops)
    if num_pops < 1 or num_pops > 4:
        raise ValueError("1-4 populations supported")
    if r_bins is None and bp_bins is None:
        raise ValueError("Either r_bins or bp_bins must be provided")

    if use_genotypes:
        # Genotype (diploid) path
        if genotype_matrix is not None:
            gm = genotype_matrix
            if gm.device != 'GPU':
                gm.transfer_to_gpu()
        elif haplotype_matrix is not None:
            gm = GenotypeMatrix.from_haplotype_matrix(haplotype_matrix)
            if gm.device != 'GPU':
                gm.transfer_to_gpu()
        else:
            if vcf_file is None:
                raise ValueError("vcf_file or genotype_matrix required")
            if pop_file is None:
                raise ValueError("pop_file is required when loading from VCF")
            if report:
                print(f"Loading {vcf_file} (genotypes) ...")
            gm = GenotypeMatrix.from_vcf(vcf_file)
            gm.load_pop_file(pop_file, pops=pops)
            if accessible_bed is not None and not gm.has_accessible_mask:
                gm.set_accessible_mask(accessible_bed)
            gm.transfer_to_gpu()
        mat = gm
        if report:
            print(f"  {gm.num_individuals} individuals, {gm.num_variants:,} variants")
    else:
        # Haplotype (phased) path
        if haplotype_matrix is not None:
            hm = haplotype_matrix
            if not isinstance(hm.haplotypes, cp.ndarray):
                hm.transfer_to_gpu()
        else:
            if vcf_file is None:
                raise ValueError("vcf_file or haplotype_matrix is required")
            if pop_file is None:
                raise ValueError("pop_file is required when loading from VCF")
            if report:
                print(f"Loading {vcf_file} ...")
            hm = HaplotypeMatrix.from_vcf(vcf_file)
            hm.load_pop_file(pop_file, pops=pops)
            if accessible_bed is not None and not hm.has_accessible_mask:
                hm.set_accessible_mask(accessible_bed)
            hm.transfer_to_gpu()
        mat = hm
        if report:
            print(f"  {hm.num_haplotypes} hap, {hm.num_variants:,} variants")

    # Determine bins and distance metric for pair binning
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file required with r_bins")
        bins = np.asarray(r_bins, dtype=np.float64)
        pos_cpu = mat.positions.get() if hasattr(mat.positions, 'get') else np.asarray(mat.positions)
        gen_dists = _interpolate_genetic_distances(pos_cpu, rec_map_file)
        gen_dists_gpu = cp.asarray(gen_dists)
        max_bp_dist = _max_bp_for_r_dist(pos_cpu, gen_dists, float(bins[-1]))
    else:
        bins = np.asarray(bp_bins, dtype=np.float64)
        gen_dists_gpu = None
        max_bp_dist = float(bins[-1])

    n_bins = len(bins) - 1
    if report:
        print(f"  Computing LD ({n_bins} bins, {num_pops} pops) ...")

    ld_stat_names = _ld_names(num_pops)
    het_stat_names = _het_names(num_pops)

    if use_genotypes:
        ld_sums = _compute_ld_sums(mat, pops, bins, gen_dists_gpu, max_bp_dist,
                                    use_genotypes=True)
        het = _compute_heterozygosity(mat, pops, use_genotypes=True)
    else:
        ld_sums = _compute_ld_sums(mat, pops, bins, gen_dists_gpu, max_bp_dist)
        het = _compute_heterozygosity(mat, pops)

    if report:
        print("  Done.")

    bin_tuples = [(float(bins[i]), float(bins[i + 1])) for i in range(n_bins)]
    sums_list = [ld_sums[i] for i in range(n_bins)]
    sums_list.append(np.array([het[h] for h in het_stat_names]))

    return {'bins': bin_tuples, 'sums': sums_list,
            'stats': (ld_stat_names, het_stat_names), 'pops': pops}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _interpolate_genetic_distances(positions, rec_map_file):
    """Interpolate per-variant genetic map positions (Morgans) from map file."""
    map_pos, map_vals = [], []
    with open(rec_map_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    map_pos.append(float(parts[0]))
                    map_vals.append(float(parts[1]) / 100.0)  # cM -> Morgans
                except ValueError:
                    continue
    return np.interp(positions, np.array(map_pos), np.array(map_vals))


def _max_bp_for_r_dist(positions, gen_dists, max_r):
    """Conservative max physical distance for a given recombination distance.

    Uses minimum local recombination rate with safety margin. Caps at
    chromosome length to avoid generating excessive pairs.
    """
    chrom_len = float(positions[-1] - positions[0])
    if len(positions) < 2:
        return chrom_len

    bp_diffs = np.diff(positions).astype(np.float64)
    r_diffs = np.diff(gen_dists)
    valid = bp_diffs > 0
    if not np.any(valid):
        return chrom_len

    rates = r_diffs[valid] / bp_diffs[valid]
    min_rate = np.min(rates[rates > 0]) if np.any(rates > 0) else max_r / chrom_len
    result = max_r / min_rate * 1.1
    return min(result, chrom_len)


def _compute_ld_sums(mat, pops, bins, gen_dists_gpu, max_bp_dist,
                     use_genotypes=False):
    """Compute LD statistic sums per bin on GPU for N populations.

    Handles both haplotype (4-way) and genotype (9-way) count modes.
    """
    num_pops = len(pops)

    if use_genotypes:
        # Filter to variants biallelic across the union of specified populations
        # (matches moments' behavior in _count_types_sparse lines 545-547)
        # Sum per-pop to avoid materializing the full union genotype matrix
        geno = mat.genotypes
        xp = cp if isinstance(geno, cp.ndarray) else np
        alt_sum = xp.zeros(mat.num_variants, dtype=xp.int64)
        n_valid_filter = xp.zeros(mat.num_variants, dtype=xp.int64)
        seen = set()
        for pop in pops:
            for idx in mat.sample_sets[pop]:
                if idx in seen:
                    continue
                seen.add(idx)
                row = geno[idx, :]
                v = row >= 0
                alt_sum += xp.where(v, row, 0).astype(xp.int64)
                n_valid_filter += v.astype(xp.int64)
        max_alt = 2 * n_valid_filter
        keep = (alt_sum > 0) & (alt_sum < max_alt) & (n_valid_filter >= 2)
        keep_idx = xp.where(keep)[0]
        pos = mat.positions[keep_idx]
        data_matrix = mat.genotypes[:, keep_idx]
        count_fn = _compute_genotype_counts_for_pairs
        stat_fn = compute_multi_pop_statistics_batch_geno
    else:
        # Keep exactly-two-present (biallelic + segregating) sites, any coding;
        # the 0/1 indicator recodes {0,2}/{1,2} so the tally counts the alt.
        from ._memutil import allele_counts
        ac, _ = allele_counts(mat.haplotypes)
        keep = (ac > 0).sum(axis=1) == 2
        keep_idx = cp.where(keep)[0]
        pos = mat.positions[keep_idx]
        data_matrix = mat._biallelic_indicator()[:, keep_idx]
        count_fn = _compute_counts_for_pairs
        stat_fn = None  # handled by 2-pop fast path or multi-pop

    if not isinstance(pos, cp.ndarray):
        pos = cp.array(pos)

    n_bins = len(bins) - 1
    bins_gpu = cp.asarray(bins)
    pop_indices = [mat.sample_sets[p] for p in pops]
    max_samp = max(len(pi) for pi in pop_indices)
    chunk_size = _estimate_ld_chunk_size(max_samp, num_pops=num_pops)

    ld_stat_names = _ld_names(num_pops)
    n_ld = len(ld_stat_names)

    # Genetic-distance lookup: filter once outside the loop so fancy-indexing
    # on the keep-mask doesn't repeat per chunk.
    if gen_dists_gpu is not None:
        gen_dists_lookup = gen_dists_gpu[keep_idx]
    else:
        gen_dists_lookup = None

    bin_sums = cp.zeros((n_bins, n_ld), dtype=cp.float64)
    stat_specs = _generate_stat_specs(num_pops) if (num_pops != 2 or use_genotypes) else None

    for ci, cj in _iter_pairs_within_distance(pos, max_bp_dist, chunk_size):
        if gen_dists_lookup is not None:
            distances = cp.abs(gen_dists_lookup[cj] - gen_dists_lookup[ci])
        else:
            distances = pos[cj] - pos[ci]
        cb = cp.digitize(distances, bins_gpu) - 1
        del distances

        counts_list = []
        n_valid_list = []
        for pidx in pop_indices:
            c, nv = count_fn(data_matrix, ci, cj, pidx)
            counts_list.append(c)
            n_valid_list.append(nv)

        if not use_genotypes and num_pops == 2:
            stats = _compute_two_pop_statistics_batch(
                counts_list[0], counts_list[1],
                n_valid_list[0], n_valid_list[1], ld_statistics)
        elif use_genotypes:
            stats = compute_multi_pop_statistics_batch_geno(
                counts_list, n_valid_list, None, stat_specs)
        else:
            stats = compute_multi_pop_statistics_batch_hap(
                counts_list, n_valid_list, ld_statistics, stat_specs)

        valid = (cb >= 0) & (cb < n_bins)
        vb = cb[valid]
        vs = stats[valid]
        flat_idx = vb[:, None] * n_ld + cp.arange(n_ld)[None, :]
        cp.add.at(bin_sums.ravel(), flat_idx.ravel(), vs.ravel())

        del counts_list, n_valid_list, stats, cb

    return bin_sums.get()


def _compute_heterozygosity(mat, pops, use_genotypes=False):
    """Compute H_i_j statistics on GPU for N populations (moments convention).

    Works with both haplotype and genotype data by converting to allele
    counts with the haploid sample size convention.
    """
    num_pops = len(pops)

    alt_counts = []
    ref_counts = []
    hap_sizes = []
    for pop in pops:
        pidx = mat.sample_sets[pop]
        if use_genotypes:
            if isinstance(pidx, list):
                pidx = cp.array(pidx, dtype=cp.int32)
            pop_data = mat.genotypes[pidx, :]
            valid = pop_data >= 0
            alt = cp.sum(cp.where(valid, pop_data, 0).astype(cp.int32), axis=0).astype(cp.float64)
            n_hap = 2.0 * cp.sum(valid, axis=0).astype(cp.float64)
        else:
            alt = cp.sum(cp.maximum(mat.haplotypes[pidx, :], 0).astype(cp.int32), axis=0).astype(cp.float64)
            n_hap = cp.float64(len(pidx)) * cp.ones_like(alt)
        alt_counts.append(alt)
        ref_counts.append(n_hap - alt)
        hap_sizes.append(n_hap)

    # A site where a population has no genotyped individual (n_hap == 0, which
    # missing data allows even after the union-biallelic filter) makes the
    # haploid-size denominator zero. The alt/ref counts are also zero there, so
    # the site's heterozygosity contribution is zero; clamp the denominator to
    # avoid 0/0 = NaN poisoning the per-site sum.
    result = {}
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            if ii == jj:
                denom = cp.maximum(hap_sizes[ii] * (hap_sizes[ii] - 1), 1.0)
                val = float(cp.sum(
                    2.0 * ref_counts[ii] * alt_counts[ii] / denom
                ).get())
            else:
                denom = cp.maximum(hap_sizes[ii] * hap_sizes[jj], 1.0)
                val = float(cp.sum(
                    (ref_counts[ii] * alt_counts[jj] + alt_counts[ii] * ref_counts[jj])
                    / denom
                ).get())
            result[f"H_{ii}_{jj}"] = val

    return result


