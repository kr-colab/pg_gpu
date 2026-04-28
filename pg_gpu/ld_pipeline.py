"""
LD pipeline utilities: pair generation, counting, stat specs, and batch dispatch.

These functions are used by moments_ld.py and the fused kernel modules to
orchestrate the multi-population LD computation pipeline.
"""
import numpy as np
import cupy as cp


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


def estimate_ld_chunk_size(n_haplotypes_per_pop, available_memory_bytes=None,
                           num_pops=2):
    """
    Estimate optimal chunk size for LD computation based on GPU memory.

    Memory per pair (N pairs, H haplotypes per pop, P populations):
    - hap_i, hap_j arrays (P pops): 4 * H * P * N bytes
    - counts arrays: 32 * P * N bytes
    - statistics: 120 * P * N bytes
    - Overhead (~3x): accounts for intermediates, fragmentation

    Formula: bytes_per_pair = (4*H*P + 150*P) * 3

    Parameters
    ----------
    n_haplotypes_per_pop : int
        Number of haplotypes in the larger population
    available_memory_bytes : int, optional
        Available GPU memory in bytes. If None, queries the GPU.
    num_pops : int
        Number of populations (default 2)

    Returns
    -------
    int
        Recommended chunk size (number of pairs per iteration)
    """
    if available_memory_bytes is None:
        available_memory_bytes = int(cp.cuda.Device().mem_info[0] * 0.5)

    bytes_per_pair = (4 * n_haplotypes_per_pop * num_pops + 150 * num_pops) * 3
    chunk_size = available_memory_bytes // bytes_per_pair

    return max(100_000, min(chunk_size, 10_000_000))


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def iter_pairs_within_distance(positions, max_dist, chunk_size):
    """
    Yield GPU chunks of variant pair indices (i, j) with pos[j] - pos[i] <= max_dist.

    Fully GPU-native: no CPU materialization of the full pair list. Variants
    are partitioned into contiguous anchor blocks whose cumulative pair count
    is close to ``chunk_size``; each block's pairs are emitted as a single
    chunk. Overshoot above ``chunk_size`` per chunk is bounded by
    ``max(pairs_per_variant) ~= density * max_dist``.

    Parameters
    ----------
    positions : cp.ndarray
        Sorted variant positions on GPU.
    max_dist : float
        Maximum distance between variants to include.
    chunk_size : int
        Target pairs per chunk (actual chunk size may slightly exceed this).

    Yields
    ------
    idx_i, idx_j : cp.ndarray[int32]
        First and second variant index of each pair in the chunk.
    """
    n = len(positions)
    if n < 2:
        return

    upper_bounds = positions + max_dist
    j_max = cp.searchsorted(positions, upper_bounds, side='right')
    variant_indices = cp.arange(n, dtype=cp.int64)
    pairs_per_variant = cp.maximum(0, j_max - variant_indices - 1)
    cumulative = cp.cumsum(pairs_per_variant)
    total_pairs = int(cumulative[-1].get())
    if total_pairs == 0:
        return

    chunk_size = max(1, int(chunk_size))
    n_chunks = (total_pairs + chunk_size - 1) // chunk_size
    targets = cp.minimum(
        cp.arange(1, n_chunks + 1, dtype=cp.int64) * chunk_size,
        total_pairs)
    # Exclusive upper variant for each chunk: smallest v with cumulative[v] >= target.
    v_his = (cp.searchsorted(cumulative, targets, side='left') + 1).get()
    v_his = np.minimum(v_his, n)

    v_lo = 0
    for v_hi in v_his:
        v_hi = int(v_hi)
        if v_hi <= v_lo:
            continue
        counts_slice = pairs_per_variant[v_lo:v_hi]
        cum_local = cp.cumsum(counts_slice)
        n_pairs_chunk = int(cum_local[-1].get())
        if n_pairs_chunk == 0:
            v_lo = v_hi
            continue
        flat = cp.arange(n_pairs_chunk, dtype=cp.int64)
        var_pos = cp.searchsorted(cum_local, flat, side='right')
        # Exclusive prefix-sum: inclusive cumsum minus the addend at each index.
        exclusive = cum_local - counts_slice
        within_var = flat - exclusive[var_pos]
        idx_i = (v_lo + var_pos).astype(cp.int32)
        idx_j = idx_i + 1 + within_var.astype(cp.int32)
        yield idx_i, idx_j
        v_lo = v_hi


# ---------------------------------------------------------------------------
# Haplotype pair counting (4-way)
# ---------------------------------------------------------------------------


def compute_counts_for_pairs(haplotypes, idx_i, idx_j, pop_indices=None):
    """
    Compute haplotype counts [n11, n10, n01, n00] for specific pairs.

    Parameters
    ----------
    haplotypes : cp.ndarray
        Shape (n_haplotypes, n_variants), values 0, 1, or -1 (missing)
    idx_i, idx_j : cp.ndarray
        Pair indices, shape (n_pairs,)
    pop_indices : list or cp.ndarray, optional
        Indices of samples to include (for population-specific counts)

    Returns
    -------
    counts : cp.ndarray
        Shape (n_pairs, 4), columns [n11, n10, n01, n00]
    n_valid : cp.ndarray or None
        Shape (n_pairs,), valid sample counts per pair. None if no missing data.
    """
    if pop_indices is not None:
        if isinstance(pop_indices, list):
            pop_indices = cp.array(pop_indices, dtype=cp.int32)
        haplotypes = haplotypes[pop_indices, :]

    hap_i = haplotypes[:, idx_i]
    hap_j = haplotypes[:, idx_j]

    has_missing = cp.any(haplotypes == -1)

    if has_missing:
        valid_mask = (hap_i >= 0) & (hap_j >= 0)
        n_valid = cp.sum(valid_mask, axis=0, dtype=cp.int32)
        n11 = cp.sum((hap_i == 1) & (hap_j == 1) & valid_mask, axis=0, dtype=cp.int32)
        n10 = cp.sum((hap_i == 1) & (hap_j == 0) & valid_mask, axis=0, dtype=cp.int32)
        n01 = cp.sum((hap_i == 0) & (hap_j == 1) & valid_mask, axis=0, dtype=cp.int32)
        n00 = cp.sum((hap_i == 0) & (hap_j == 0) & valid_mask, axis=0, dtype=cp.int32)
    else:
        n_valid = None
        n11 = cp.sum((hap_i == 1) & (hap_j == 1), axis=0, dtype=cp.int32)
        n10 = cp.sum((hap_i == 1) & (hap_j == 0), axis=0, dtype=cp.int32)
        n01 = cp.sum((hap_i == 0) & (hap_j == 1), axis=0, dtype=cp.int32)
        n00 = cp.sum((hap_i == 0) & (hap_j == 0), axis=0, dtype=cp.int32)

    counts = cp.stack([n11, n10, n01, n00], axis=1)
    return counts, n_valid


# ---------------------------------------------------------------------------
# Genotype pair counting (9-way)
# ---------------------------------------------------------------------------


_GENO_COUNTS_KERN = cp.RawKernel(r'''
extern "C" __global__
void k(const signed char* __restrict__ g,
       const int n_indiv, const int n_var,
       const int* __restrict__ idx_i, const int* __restrict__ idx_j,
       int* __restrict__ counts_out,
       int* __restrict__ n_valid_out, const int n_pairs){
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=n_pairs)return;
    int i=idx_i[t], j=idx_j[t];
    int c00=0,c01=0,c02=0,c10=0,c11=0,c12=0,c20=0,c21=0,c22=0;
    int valid=0;
    for(int k=0;k<n_indiv;++k){
        signed char gi=g[k*n_var+i];
        signed char gj=g[k*n_var+j];
        if(gi>=0 && gj>=0){
            int combo=gi*3+gj;
            if(combo==0)++c00;
            else if(combo==1)++c01;
            else if(combo==2)++c02;
            else if(combo==3)++c10;
            else if(combo==4)++c11;
            else if(combo==5)++c12;
            else if(combo==6)++c20;
            else if(combo==7)++c21;
            else if(combo==8)++c22;
            ++valid;
        }
    }
    int*row=counts_out+t*9;
    row[0]=c00; row[1]=c01; row[2]=c02;
    row[3]=c10; row[4]=c11; row[5]=c12;
    row[6]=c20; row[7]=c21; row[8]=c22;
    n_valid_out[t]=valid;
}''', "k", options=("-std=c++11",))


def compute_genotype_counts_for_pairs(genotypes, idx_i, idx_j, pop_indices=None):
    """Compute 9-way genotype counts for variant pairs.

    For each pair (locus_i, locus_j), counts all 3x3 combinations of
    genotype values (0/1/2) at the two loci.

    Parameters
    ----------
    genotypes : cp.ndarray
        Shape (n_individuals, n_variants), values 0, 1, 2, or -1 (missing)
    idx_i, idx_j : cp.ndarray
        Pair indices, shape (n_pairs,)
    pop_indices : list or cp.ndarray, optional
        Indices of individuals to include

    Returns
    -------
    counts : cp.ndarray
        Shape (n_pairs, 9), ordering: (n00, n01, n02, n10, n11, n12, n20, n21, n22)
    n_valid : cp.ndarray or None
        Shape (n_pairs,), valid individual counts per pair. None if no missing data.
    """
    if pop_indices is not None:
        if isinstance(pop_indices, list):
            pop_indices = cp.array(pop_indices, dtype=cp.int32)
        genotypes = genotypes[pop_indices, :]

    g = cp.ascontiguousarray(genotypes, dtype=cp.int8)
    n_indiv, n_var = g.shape
    idx_i_c = cp.ascontiguousarray(idx_i, dtype=cp.int32)
    idx_j_c = cp.ascontiguousarray(idx_j, dtype=cp.int32)
    n_pairs = int(idx_i_c.shape[0])

    has_missing = bool(cp.any(g < 0))

    counts = cp.empty((n_pairs, 9), dtype=cp.int32)
    n_valid = cp.empty(n_pairs, dtype=cp.int32)
    block = 256
    grid = (n_pairs + block - 1) // block
    _GENO_COUNTS_KERN(
        (grid,), (block,),
        (g, n_indiv, n_var, idx_i_c, idx_j_c, counts, n_valid, n_pairs))

    return counts, (n_valid if has_missing else None)


# ---------------------------------------------------------------------------
# 2-population fast path (haplotype only)
# ---------------------------------------------------------------------------


def compute_two_pop_statistics_batch(counts_pop1, counts_pop2,
                                     n_valid1, n_valid2, ld_statistics):
    """
    Compute all 15 two-population LD statistics for a batch of pairs.

    This is the optimized fast path for exactly 2 populations, using the
    ld_statistics module's dd/dz/pi2 functions directly.

    Parameters
    ----------
    counts_pop1, counts_pop2 : cp.ndarray
        Shape (n_pairs, 4), haplotype counts per population
    n_valid1, n_valid2 : cp.ndarray or None
        Valid sample counts per pair per population
    ld_statistics : module
        The ld_statistics module

    Returns
    -------
    statistics : cp.ndarray
        Shape (n_pairs, 15)
    """
    counts_between = cp.concatenate([counts_pop1, counts_pop2], axis=1)
    n_valid_between = (n_valid1, n_valid2) if n_valid1 is not None else None

    DD_0_0 = ld_statistics.dd(counts_pop1, n_valid=n_valid1)
    DD_0_1 = ld_statistics.dd(counts_between, populations=(0, 1), n_valid=n_valid_between)
    DD_1_1 = ld_statistics.dd(counts_pop2, n_valid=n_valid2)

    Dz_0_0_0 = ld_statistics.dz(counts_pop1, n_valid=n_valid1)
    Dz_0_0_1 = 0.5 * (
        ld_statistics.dz(counts_between, populations=(0, 0, 1), n_valid=n_valid_between)
        + ld_statistics.dz(counts_between, populations=(0, 1, 0), n_valid=n_valid_between))
    Dz_0_1_1 = ld_statistics.dz(counts_between, populations=(0, 1, 1), n_valid=n_valid_between)
    Dz_1_0_0 = ld_statistics.dz(counts_between, populations=(1, 0, 0), n_valid=n_valid_between)
    Dz_1_0_1 = 0.5 * (
        ld_statistics.dz(counts_between, populations=(1, 0, 1), n_valid=n_valid_between)
        + ld_statistics.dz(counts_between, populations=(1, 1, 0), n_valid=n_valid_between))
    Dz_1_1_1 = ld_statistics.dz(counts_pop2, n_valid=n_valid2)

    pi2_0_0_0_0 = ld_statistics.pi2(counts_pop1, n_valid=n_valid1)
    pi2_0_0_0_1 = ld_statistics.pi2(counts_between, populations=(0, 0, 0, 1), n_valid=n_valid_between)
    pi2_0_0_1_1 = ld_statistics.pi2(counts_between, populations=(0, 0, 1, 1), n_valid=n_valid_between)
    pi2_0_1_0_1 = ld_statistics.pi2(counts_between, populations=(0, 1, 0, 1), n_valid=n_valid_between)
    pi2_0_1_1_1 = ld_statistics.pi2(counts_between, populations=(0, 1, 1, 1), n_valid=n_valid_between)
    pi2_1_1_1_1 = ld_statistics.pi2(counts_pop2, n_valid=n_valid2)

    return cp.stack([
        DD_0_0, DD_0_1, DD_1_1,
        Dz_0_0_0, Dz_0_0_1, Dz_0_1_1, Dz_1_0_0, Dz_1_0_1, Dz_1_1_1,
        pi2_0_0_0_0, pi2_0_0_0_1, pi2_0_0_1_1, pi2_0_1_0_1, pi2_0_1_1_1, pi2_1_1_1_1
    ], axis=1)


# ---------------------------------------------------------------------------
# Single-population batch (used by HaplotypeMatrix methods)
# ---------------------------------------------------------------------------


def compute_single_pop_statistics_batch(counts, n_valid, ld_statistics):
    """
    Compute single-population LD statistics (DD, Dz, pi2) for a batch of pairs.

    Parameters
    ----------
    counts : cp.ndarray
        Shape (n_pairs, 4), haplotype counts [n11, n10, n01, n00]
    n_valid : cp.ndarray or None
    ld_statistics : module

    Returns
    -------
    statistics : cp.ndarray
        Shape (n_pairs, 3), columns [DD, Dz, pi2]
    """
    DD = ld_statistics.dd(counts, n_valid=n_valid)
    Dz = ld_statistics.dz(counts, n_valid=n_valid)
    pi2 = ld_statistics.pi2(counts, n_valid=n_valid)
    return cp.stack([DD, Dz, pi2], axis=1)


# ---------------------------------------------------------------------------
# Stat name and spec generation (moments-compatible)
# ---------------------------------------------------------------------------


def ld_names(num_pops):
    """Generate LD statistic names matching moments.LD.Util.ld_names()."""
    names = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            names.append(f"DD_{ii}_{jj}")
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj, num_pops):
                names.append(f"Dz_{ii}_{jj}_{kk}")
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            for kk in range(ii, num_pops):
                for ll in range(kk, num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    names.append(f"pi2_{ii}_{jj}_{kk}_{ll}")
    return names


def het_names(num_pops):
    """Generate heterozygosity statistic names matching moments.LD.Util.het_names()."""
    names = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            names.append(f"H_{ii}_{jj}")
    return names


def generate_stat_specs(num_pops):
    """Generate computation specs for each LD statistic.

    Each spec is (stat_name, [(weight, stat_type, pop_indices), ...])
    encoding the exact averaging logic from moments' _call_sgc().

    Returns
    -------
    list of (str, list of (float, str, tuple))
    """
    specs = []
    names = ld_names(num_pops)

    for name in names:
        parts = name.split("_")
        stat_type = parts[0]
        pop_nums = tuple(int(p) for p in parts[1:])

        if stat_type == "DD":
            specs.append((name, [(1.0, 'dd', pop_nums)]))

        elif stat_type == "Dz":
            ii, jj, kk = pop_nums
            if jj == kk:
                specs.append((name, [(1.0, 'dz', (ii, jj, kk))]))
            else:
                specs.append((name, [
                    (0.5, 'dz', (ii, jj, kk)),
                    (0.5, 'dz', (ii, kk, jj)),
                ]))

        elif stat_type == "pi2":
            ii, jj, kk, ll = pop_nums
            if ii == jj:
                if kk == ll:
                    if ii == kk:
                        specs.append((name, [(1.0, 'pi2', (ii, jj, kk, ll))]))
                    else:
                        specs.append((name, [
                            (0.5, 'pi2', (ii, jj, kk, ll)),
                            (0.5, 'pi2', (kk, ll, ii, jj)),
                        ]))
                else:
                    specs.append((name, [
                        (0.25, 'pi2', (ii, jj, kk, ll)),
                        (0.25, 'pi2', (ii, jj, ll, kk)),
                        (0.25, 'pi2', (kk, ll, ii, jj)),
                        (0.25, 'pi2', (ll, kk, ii, jj)),
                    ]))
            else:
                if kk == ll:
                    specs.append((name, [
                        (0.25, 'pi2', (ii, jj, kk, ll)),
                        (0.25, 'pi2', (jj, ii, kk, ll)),
                        (0.25, 'pi2', (kk, ll, ii, jj)),
                        (0.25, 'pi2', (kk, ll, jj, ii)),
                    ]))
                else:
                    specs.append((name, [
                        (0.125, 'pi2', (ii, jj, kk, ll)),
                        (0.125, 'pi2', (ii, jj, ll, kk)),
                        (0.125, 'pi2', (jj, ii, kk, ll)),
                        (0.125, 'pi2', (jj, ii, ll, kk)),
                        (0.125, 'pi2', (kk, ll, ii, jj)),
                        (0.125, 'pi2', (ll, kk, ii, jj)),
                        (0.125, 'pi2', (kk, ll, jj, ii)),
                        (0.125, 'pi2', (ll, kk, jj, ii)),
                    ]))

    return specs


# ---------------------------------------------------------------------------
# Per-population precomputation (haplotype)
# ---------------------------------------------------------------------------


class PopData:
    """Precomputed per-population arrays for haplotype batch LD computation."""
    __slots__ = ('c1', 'c2', 'c3', 'c4', 'n', 'D', 'pA', 'qA', 'pB', 'qB')

    def __init__(self, counts, n_valid):
        self.c1 = counts[:, 0].astype(cp.float64)
        self.c2 = counts[:, 1].astype(cp.float64)
        self.c3 = counts[:, 2].astype(cp.float64)
        self.c4 = counts[:, 3].astype(cp.float64)
        if n_valid is not None:
            self.n = n_valid.astype(cp.float64)
        else:
            self.n = self.c1 + self.c2 + self.c3 + self.c4
        self.D = self.c2 * self.c3 - self.c1 * self.c4
        self.pA = self.c1 + self.c2
        self.qA = self.c3 + self.c4
        self.pB = self.c1 + self.c3
        self.qB = self.c2 + self.c4
