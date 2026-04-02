"""
Integration layer: GPU-accelerated LD statistics for moments inference.

Drop-in replacement for moments.LD.Parsing.compute_ld_statistics() using
pg_gpu's GPU kernels. Output format is identical to moments, so downstream
inference (bootstrap_data, optimize_log_lbfgsb, Godambe) works unchanged.

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
import allel

from .haplotype_matrix import HaplotypeMatrix
from .haplotype_matrix import (
    _generate_pairs_within_distance,
    _compute_counts_for_pairs,
    _compute_two_pop_statistics_batch,
    _estimate_ld_chunk_size,
)
from . import ld_statistics

_LD_NAMES = [
    'DD_0_0', 'DD_0_1', 'DD_1_1',
    'Dz_0_0_0', 'Dz_0_0_1', 'Dz_0_1_1',
    'Dz_1_0_0', 'Dz_1_0_1', 'Dz_1_1_1',
    'pi2_0_0_0_0', 'pi2_0_0_0_1', 'pi2_0_0_1_1',
    'pi2_0_1_0_1', 'pi2_0_1_1_1', 'pi2_1_1_1_1',
]
_HET_NAMES = ['H_0_0', 'H_0_1', 'H_1_1']
_N_LD = len(_LD_NAMES)


def compute_ld_statistics(
    vcf_file, rec_map_file=None, pop_file=None, pops=None,
    r_bins=None, bp_bins=None, use_genotypes=False,
    report=True, ac_filter=True,
):
    """GPU-accelerated two-population LD statistics, moments-compatible.

    Parameters
    ----------
    vcf_file : str
        Path to VCF file.
    rec_map_file : str, optional
        Recombination map (tab-delimited: pos, Map(cM)). Required with r_bins.
    pop_file : str
        Population file (tab-delimited: sample, pop).
    pops : list of str
        Two population names. Defaults to ['pop0', 'pop1'].
    r_bins : array-like, optional
        Recombination rate bin edges (Morgans).
    bp_bins : array-like, optional
        Base-pair distance bin edges (alternative to r_bins).
    report : bool
        Print progress.
    ac_filter : bool
        Apply biallelic filter.

    Returns
    -------
    dict with keys 'bins', 'sums', 'stats', 'pops' (moments format).
    """
    if pops is None:
        pops = ['pop0', 'pop1']
    if len(pops) != 2:
        raise ValueError("Only two-population LD is supported")
    if r_bins is None and bp_bins is None:
        raise ValueError("Either r_bins or bp_bins must be provided")
    if pop_file is None:
        raise ValueError("pop_file is required")

    if report:
        print(f"Loading {vcf_file} ...")
    hm = HaplotypeMatrix.from_vcf(vcf_file)

    # Parse populations (reads VCF sample names for index mapping)
    samples = allel.read_vcf(vcf_file, fields=['samples'])['samples']
    _assign_populations(hm, samples, pop_file, pops)

    if ac_filter:
        hm = hm.apply_biallelic_filter()
    hm.transfer_to_gpu()

    pop1, pop2 = pops
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
        print(f"  Computing LD ({n_bins} bins) ...")

    ld_sums = _compute_ld_sums(hm, pop1, pop2, bins, gen_dists_gpu, max_bp_dist)
    het = _compute_heterozygosity(hm, pop1, pop2)

    if report:
        print("  Done.")

    bin_tuples = [(float(bins[i]), float(bins[i + 1])) for i in range(n_bins)]
    sums_list = [ld_sums[i] for i in range(n_bins)]
    sums_list.append(np.array([het['H_0_0'], het['H_0_1'], het['H_1_1']]))

    return {'bins': bin_tuples, 'sums': sums_list,
            'stats': (_LD_NAMES, _HET_NAMES), 'pops': pops}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _assign_populations(hm, samples, pop_file, pops):
    """Set sample_sets from population file and VCF sample order."""
    n_samples = len(samples)
    pop_map = {}
    with open(pop_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != 'sample':
                pop_map[parts[0]] = parts[1]

    pop_sets = {p: [] for p in pops}
    for i, name in enumerate(samples):
        pop = pop_map.get(name)
        if pop in pop_sets:
            pop_sets[pop].append(i)
            pop_sets[pop].append(i + n_samples)
    hm.sample_sets = pop_sets


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


def _compute_ld_sums(hm, pop1, pop2, bins, gen_dists_gpu, max_bp_dist):
    """Compute LD statistic sums per bin on GPU.

    Bins by recombination distance when gen_dists_gpu is provided,
    otherwise by physical distance.
    """
    pos = hm.positions
    if not isinstance(pos, cp.ndarray):
        pos = cp.array(pos)

    n_bins = len(bins) - 1
    bins_gpu = cp.asarray(bins)
    pop1_idx = hm.sample_sets[pop1]
    pop2_idx = hm.sample_sets[pop2]
    chunk_size = _estimate_ld_chunk_size(max(len(pop1_idx), len(pop2_idx)))

    idx_i, idx_j = _generate_pairs_within_distance(pos, max_bp_dist)
    if len(idx_i) == 0:
        return np.zeros((n_bins, _N_LD), dtype=np.float64)

    # Bin by recombination or physical distance
    if gen_dists_gpu is not None:
        distances = cp.abs(gen_dists_gpu[idx_j] - gen_dists_gpu[idx_i])
    else:
        distances = pos[idx_j] - pos[idx_i]
    bin_inds = cp.digitize(distances, bins_gpu) - 1
    del distances

    bin_sums = cp.zeros((n_bins, _N_LD), dtype=cp.float64)
    total_pairs = len(idx_i)

    for chunk_start in range(0, total_pairs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pairs)
        ci = idx_i[chunk_start:chunk_end]
        cj = idx_j[chunk_start:chunk_end]
        cb = bin_inds[chunk_start:chunk_end]

        c1, nv1 = _compute_counts_for_pairs(hm.haplotypes, ci, cj, pop1_idx)
        c2, nv2 = _compute_counts_for_pairs(hm.haplotypes, ci, cj, pop2_idx)
        stats = _compute_two_pop_statistics_batch(c1, c2, nv1, nv2, ld_statistics)

        valid = (cb >= 0) & (cb < n_bins)
        vb = cb[valid]
        vs = stats[valid]
        for k in range(_N_LD):
            cp.add.at(bin_sums[:, k], vb, vs[:, k])

        del c1, c2, nv1, nv2, stats

    del idx_i, idx_j, bin_inds
    return bin_sums.get()


def _compute_heterozygosity(hm, pop1, pop2):
    """Compute H_0_0, H_0_1, H_1_1 on GPU (moments convention)."""
    hap = hm.haplotypes
    p1 = hm.sample_sets[pop1]
    p2 = hm.sample_sets[pop2]

    alt1 = cp.sum(cp.maximum(hap[p1, :], 0).astype(cp.int32), axis=0).astype(cp.float64)
    n1 = cp.float64(len(p1))
    ref1 = n1 - alt1

    alt2 = cp.sum(cp.maximum(hap[p2, :], 0).astype(cp.int32), axis=0).astype(cp.float64)
    n2 = cp.float64(len(p2))
    ref2 = n2 - alt2

    H_0_0 = float(cp.sum(2.0 * ref1 * alt1 / (n1 * (n1 - 1))).get())
    H_1_1 = float(cp.sum(2.0 * ref2 * alt2 / (n2 * (n2 - 1))).get())
    H_0_1 = float(cp.sum((ref1 * alt2 + alt1 * ref2) / (n1 * n2)).get())

    return {'H_0_0': H_0_0, 'H_0_1': H_0_1, 'H_1_1': H_1_1}
