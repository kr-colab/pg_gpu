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
from .haplotype_matrix import (
    _generate_pairs_within_distance,
    _compute_counts_for_pairs,
    _compute_two_pop_statistics_batch,
    _compute_multi_pop_statistics_batch,
    _estimate_ld_chunk_size,
    _ld_names,
    _het_names,
    _generate_stat_specs,
)
from . import ld_statistics


def compute_ld_statistics(
    vcf_file=None, rec_map_file=None, pop_file=None, pops=None,
    r_bins=None, bp_bins=None, use_genotypes=False,
    report=True, ac_filter=True, haplotype_matrix=None,
):
    """GPU-accelerated multi-population LD statistics, moments-compatible.

    Parameters
    ----------
    vcf_file : str, optional
        Path to VCF file. Not needed if haplotype_matrix is provided.
    rec_map_file : str, optional
        Recombination map (tab-delimited: pos, Map(cM)). Required with r_bins.
    pop_file : str, optional
        Population file (tab-delimited: sample, pop). Not needed if
        haplotype_matrix already has sample_sets configured.
    pops : list of str
        Population names (1-4). Defaults to ['pop0', 'pop1'].
    r_bins : array-like, optional
        Recombination rate bin edges (Morgans).
    bp_bins : array-like, optional
        Base-pair distance bin edges (alternative to r_bins).
    report : bool
        Print progress.
    ac_filter : bool
        Apply biallelic filter.
    haplotype_matrix : HaplotypeMatrix, optional
        Pre-loaded HaplotypeMatrix (skips VCF loading and GPU transfer).

    Returns
    -------
    dict with keys 'bins', 'sums', 'stats', 'pops' (moments format).
    """
    if pops is None:
        pops = ['pop0', 'pop1']
    num_pops = len(pops)
    if num_pops < 1 or num_pops > 4:
        raise ValueError("1-4 populations supported")
    if r_bins is None and bp_bins is None:
        raise ValueError("Either r_bins or bp_bins must be provided")

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
        if ac_filter:
            hm = hm.apply_biallelic_filter()
        hm.transfer_to_gpu()

    if report:
        print(f"  {hm.num_haplotypes} hap, {hm.num_variants:,} variants")

    # Determine bins and distance metric for pair binning
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file required with r_bins")
        bins = np.asarray(r_bins, dtype=np.float64)
        pos_cpu = hm.positions.get() if hasattr(hm.positions, 'get') else np.asarray(hm.positions)
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

    ld_sums = _compute_ld_sums(hm, pops, bins, gen_dists_gpu, max_bp_dist)
    het = _compute_heterozygosity(hm, pops)

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


def _compute_ld_sums(hm, pops, bins, gen_dists_gpu, max_bp_dist):
    """Compute LD statistic sums per bin on GPU for N populations."""
    num_pops = len(pops)
    pos = hm.positions
    if not isinstance(pos, cp.ndarray):
        pos = cp.array(pos)

    n_bins = len(bins) - 1
    bins_gpu = cp.asarray(bins)
    pop_indices = [hm.sample_sets[p] for p in pops]
    max_hap = max(len(pi) for pi in pop_indices)
    chunk_size = _estimate_ld_chunk_size(max_hap, num_pops=num_pops)

    ld_stat_names = _ld_names(num_pops)
    n_ld = len(ld_stat_names)

    idx_i, idx_j = _generate_pairs_within_distance(pos, max_bp_dist)
    if len(idx_i) == 0:
        return np.zeros((n_bins, n_ld), dtype=np.float64)

    # Bin by recombination or physical distance
    if gen_dists_gpu is not None:
        distances = cp.abs(gen_dists_gpu[idx_j] - gen_dists_gpu[idx_i])
    else:
        distances = pos[idx_j] - pos[idx_i]
    bin_inds = cp.digitize(distances, bins_gpu) - 1
    del distances

    bin_sums = cp.zeros((n_bins, n_ld), dtype=cp.float64)
    total_pairs = len(idx_i)

    stat_specs = _generate_stat_specs(num_pops) if num_pops != 2 else None

    for chunk_start in range(0, total_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pairs)
        ci = idx_i[chunk_start:chunk_end]
        cj = idx_j[chunk_start:chunk_end]
        cb = bin_inds[chunk_start:chunk_end]

        counts_list = []
        n_valid_list = []
        for pidx in pop_indices:
            c, nv = _compute_counts_for_pairs(hm.haplotypes, ci, cj, pidx)
            counts_list.append(c)
            n_valid_list.append(nv)

        if num_pops == 2:
            stats = _compute_two_pop_statistics_batch(
                counts_list[0], counts_list[1],
                n_valid_list[0], n_valid_list[1], ld_statistics)
        else:
            stats = _compute_multi_pop_statistics_batch(
                counts_list, n_valid_list, ld_statistics, stat_specs)

        valid = (cb >= 0) & (cb < n_bins)
        vb = cb[valid]
        vs = stats[valid]
        # Vectorized scatter-add: single kernel launch for all stats
        flat_idx = vb[:, None] * n_ld + cp.arange(n_ld)[None, :]
        cp.add.at(bin_sums.ravel(), flat_idx.ravel(), vs.ravel())

        del counts_list, n_valid_list, stats

    del idx_i, idx_j, bin_inds
    return bin_sums.get()


def _compute_heterozygosity(hm, pops):
    """Compute H_i_j statistics on GPU for N populations (moments convention)."""
    hap = hm.haplotypes
    num_pops = len(pops)

    # Pre-compute alt/ref counts and sample sizes per population
    alt_counts = []
    ref_counts = []
    pop_sizes = []
    for pop in pops:
        pidx = hm.sample_sets[pop]
        alt = cp.sum(cp.maximum(hap[pidx, :], 0).astype(cp.int32), axis=0).astype(cp.float64)
        n = cp.float64(len(pidx))
        alt_counts.append(alt)
        ref_counts.append(n - alt)
        pop_sizes.append(n)

    result = {}
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            if ii == jj:
                # Within-population heterozygosity
                val = float(cp.sum(
                    2.0 * ref_counts[ii] * alt_counts[ii]
                    / (pop_sizes[ii] * (pop_sizes[ii] - 1))
                ).get())
            else:
                # Between-population heterozygosity
                val = float(cp.sum(
                    (ref_counts[ii] * alt_counts[jj] + alt_counts[ii] * ref_counts[jj])
                    / (pop_sizes[ii] * pop_sizes[jj])
                ).get())
            result[f"H_{ii}_{jj}"] = val

    return result
