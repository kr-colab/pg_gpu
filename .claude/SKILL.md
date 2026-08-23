---
name: pg_gpu
description: >
  GPU-accelerated population genetics analysis using the pg_gpu library
  (https://github.com/kr-colab/pg_gpu). Use this skill whenever the user
  wants to: load a VCF or zarr file for population genetics analysis,
  calculate diversity statistics (pi, Tajima's D, theta), run selection
  scans (iHS, XP-EHH, nSL), compute divergence/FST/dxy between populations,
  calculate windowed statistics across chromosomes, handle missing data,
  compute the site frequency spectrum, run PCA or dimensionality reduction,
  plot windowed stats across chromosomes, panel multiple chromosomes, or run
  any pg_gpu function. Always use this skill when the user references @vcf,
  population genetics stats, or pg_gpu commands.
---

# pg_gpu Skill

GPU-accelerated population genetics with `pg_gpu`. This skill covers the
full workflow: loading data, filtering, handling missing data, computing
statistics, windowed analyses, multi-chromosome panels, and visualization.

---

## Environment

```python
# Standard imports for all sessions
from pg_gpu import (
    HaplotypeMatrix, GenotypeMatrix,
    diversity, divergence, selection,
    ld_statistics, decomposition, sfs,
    windowed_analysis, resampling, admixture
)
from pg_gpu import plotting as pg_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

pg_gpu requires a CUDA-capable NVIDIA GPU. `import pg_gpu` runs a hard
CUDA check at module load time and raises `ImportError` / `RuntimeError`
if CuPy is missing or no usable device is found — there is no CPU
fallback for the statistics. Loaders return matrices already resident
on the GPU and every stat function operates on device arrays; `.get()`
moves a result to numpy when you need it for pandas / seaborn /
matplotlib. The `PG_GPU_SKIP_CUDA_CHECK=1` env var exists only to let
docs builds and static analysis import the package on a CPU box; do not
set it for analysis runs.

---

## Loading Data

### From VCF

```python
# Basic load
h = HaplotypeMatrix.from_vcf("data.vcf.gz")

# Specific region (requires .tbi index)
h = HaplotypeMatrix.from_vcf("data.vcf.gz", region="chr1:1000000-2000000")

# Subset of samples
h = HaplotypeMatrix.from_vcf("data.vcf.gz", samples=["ind1", "ind2"])

# With accessibility mask (for proper span normalization)
h = HaplotypeMatrix.from_vcf("data.vcf.gz", accessible_bed="access.bed")
```

**Diploid VCFs**: loaded as 2×n_samples haplotypes (treated as phased).
For unphased/diploid analyses use `GenotypeMatrix.from_vcf()` instead.

### From Zarr (preferred for large datasets)

> **Before converting a large VCF**, check whether a large fraction of
> sites is completely missing across all samples or falls outside your
> project's accessibility mask. If so, run a one-time `bcftools view -e
> 'INFO/AN==0' --regions-file accessible.bed` first — drops fully-missing
> and inaccessible sites, can shrink a 326 GB VCF to ~50–100 GB, and
> makes subsequent loads / QC inspection fit in GPU memory.

```python
# Convert once. worker_processes scales near-linearly on big VCFs; bump
# to 16-24+ when many cores are free and the source is hundreds of GB.
HaplotypeMatrix.vcf_to_zarr("data.vcf.gz", "data.zarr", worker_processes=8)

# Load (much faster than VCF for repeated access). On a GPU machine,
# always pass streaming='auto' unless you've confirmed the materialized
# matrix fits in <50% of free GPU memory -- see the next section.
h = HaplotypeMatrix.from_zarr("data.zarr", streaming='auto')
h = HaplotypeMatrix.from_zarr("data.zarr", region="chr1:1-5000000",
                              streaming='auto')
```

### Loading on GPU: streaming, kvikio, and device selection

`from_zarr` and `from_vcf` materialize the haplotype matrix on the GPU
by default. The full-fit threshold is the project constant
`STREAMING_AUTO_EAGER_FRACTION = 0.5` — pg_gpu's `streaming='auto'`
materializes eagerly when the projected `n_variants × n_haplotypes ×
2 bytes` footprint fits in under 50 % of free GPU memory and falls
back to a `StreamingHaplotypeMatrix` otherwise. `streaming='never'`
raises `MemoryError` past that cap; `streaming='always'` always returns
a streaming matrix.

```python
# from_zarr knobs you actually need:
#   streaming: {'auto', 'always', 'never'}
#   chunk_bp:  genomic chunk size for the streaming path (default 1_500_000)
#   prefetch:  read-ahead depth (0 = serial, 1 = overlap one chunk; default 1)
#   backend:   chunk-fetch backend, see below
h = HaplotypeMatrix.from_zarr(
    "data.zarr",
    region="X:1-10000000",
    streaming='auto',
    chunk_bp=1_500_000,
    prefetch=1,
    backend='auto',
)
print(type(h).__name__)   # 'HaplotypeMatrix' or 'StreamingHaplotypeMatrix'
```

Size estimate before loading: `n_variants × n_haplotypes × 2 bytes`.
A 10 Mb region of *A. gambiae* X with 1,470 diploids is ~29 GB; an
80 GB A100 has plenty if free, but on a shared box other jobs eat
that fast. `StreamingHaplotypeMatrix` runs the same `diversity.*`,
`sfs.sfs` / `sfs.joint_sfs`, `selection.*`, and `windowed_analysis`
kernels chunk-by-chunk on the GPU; results are identical, throughput
is within a small constant of the eager path.

> **`fields=` only works on the eager (non-streaming) path.** Passing
> `fields=` with `streaming='always'` raises `NotImplementedError`, and
> the `streaming='auto'` path raises with the same message when the
> matrix would not fit eagerly. For QC-field inspection on regions
> bigger than 50 % of free GPU memory, skip the loader and read the
> arrays directly:
>
> ```python
> import zarr
> store = zarr.open("data.zarr", mode="r")
> dp = store["call_DP"][:]      # shape (n_var, n_samples)
> mq = store["variant_MQ"][:]   # shape (n_var,)
> pos = store["variant_position"][:]  # for region filtering
> ```
>
> These are plain zarr arrays at `variant_<TAG>` and `call_<TAG>` paths
> with no GPU involvement and no memory cap.

#### The `backend` kwarg — kvikio vs host

`backend='auto'` (default) picks the chunk-fetch path based on what's
on the store. The two paths:

- **`'kvikio'`** — uses `kvikio.zarr.GDSStore` + `zarr.config.enable_gpu()`
  so nvCOMP decompresses each `call_genotype` chunk directly on the GPU.
  Skips the host-to-device copy entirely; the win is in the on-device
  decode, not in GPU Direct Storage (kvikio defaults to
  `compat_mode=ON`, reading bytes through posix into bounce buffers).
  Requires: a `call_genotype` codec the GPU decoder supports
  (`zstd`, `blosc`, `lz4`, or `deflate` — bio2zarr defaults to `zstd`)
  **and** sample-axis chunking that is smaller than the full sample
  axis. bio2zarr's `sample_chunk ~= 1000` is the canonical shape.
- **`'host'`** — reads each chunk through the source's host-buffer
  pipeline, then copies to the GPU. The fallback when the kvikio
  preconditions aren't met. Still uses the same prefetch thread, so
  the next chunk's read overlaps the current chunk's compute.

`'auto'` picks `'kvikio'` when both codec and chunking are supported,
falls back to `'host'` otherwise, and warns
(`BadlyChunkedWarning`) on whole-sample-axis chunked stores ≥1 GiB so
you know to rewrite. Explicit `backend='kvikio'` on an unsupported
store raises `ValueError`; explicit `backend='host'` skips the codec
probe entirely.

Re-encoding for kvikio (one-time, after which loads are dramatically
faster on biobank-scale stores):

```python
# bio2zarr-shaped chunking unlocks the kvikio fast path. The defaults
# in HaplotypeMatrix.vcf_to_zarr already produce bio2zarr layout; an
# older sgkit/scikit-allel store may need a re-encode via:
HaplotypeMatrix.vcf_to_zarr("data.vcf.gz", "data.zarr", worker_processes=8)
```

`prefetch=1` (default) runs the chunk reads on a daemon thread so the
next chunk is in flight while you compute on the current one. Set
`prefetch=0` only when debugging chunked dispatch — there is no
correctness difference, just no overlap.

#### Eager-only kernels: `materialize(region=..., sample_subset=...)`

Pairwise / cross-window statistics need every variant in scope at the
same time and **cannot** run chunk-by-chunk: `h.pairwise_r2`,
`h.locate_unlinked`, `h.windowed_r_squared`'s full r² heatmap form,
`relatedness.grm`, and `relatedness.ibs`. Calling these on a
`StreamingHaplotypeMatrix` raises. The escape hatch is `.materialize`:

```python
# Pull a 5 Mb sub-region into one eager HaplotypeMatrix on the GPU.
h_eager = h.materialize(region=(1_000_000, 6_000_000))
r2 = h_eager.pairwise_r2()
unlinked = h_eager.locate_unlinked(size=100, step=20, threshold=0.1)

# With a sample-axis subset — at biobank scale this is the difference
# between minutes (host oindex codec) and seconds: the streaming
# matrix routes the subset read through kvikio so each diploid-axis
# run decompresses directly on the GPU.
hap_cols = h.sample_sets['target_pop']
h_pop = h.materialize(region=(1_000_000, 6_000_000),
                      sample_subset=hap_cols)
```

`materialize(region=None)` materializes the whole mappable range and
will OOM on biobank-scale stores; always pass a region you can fit.

#### Pin to a specific GPU on shared machines

CuPy lands on device 0 by default. If that GPU is saturated, the load
OOMs even when other GPUs are idle. Check `nvidia-smi`, then:

```bash
CUDA_VISIBLE_DEVICES=0 python my_script.py
```
```python
# or in Python, BEFORE importing pg_gpu / cupy:
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

#### MemoryLimitedWarning on `from_vcf`

Separate check, host-RAM not GPU. pg_gpu emits this `UserWarning` when
the source VCF is >10 GB and the region is >5 Mb (or unspecified), or
the file has >5,000 samples. htslib's VCF parse is single-threaded and
the full genotype block must fit in host RAM. Either accept the slow
load or convert once with `vcf_to_zarr` and load via `from_zarr` after
that. Silence with:

```python
import warnings
from pg_gpu import MemoryLimitedWarning
warnings.filterwarnings("ignore", category=MemoryLimitedWarning)
```

### Population assignments

```python
# From file (tab-delimited: sample\tpop, header optional)
h.load_pop_file("pops.txt")           # loads all pops
h.load_pop_file("pops.txt", pops=["pop1", "pop2"])  # subset

# Manual
h.sample_sets = {"pop1": [0,1,2,3], "pop2": [4,5,6,7]}

# Inspect
print(h.sample_sets)
print(h.num_variants, h.num_haplotypes)
```

---

## Accessibility Mask

A boolean array marking which genomic positions can be called. Used as
the **denominator** for span-normalized stats (pi, dxy, theta_w):
`h.set_accessible_mask(...)` (or `accessible_bed=` on a loader) sets
`h.n_total_sites` to the count of accessible bases inside the matrix's
`[chrom_start, chrom_end]` range, and `span_normalize=True` (the
default on diversity stats) divides by `n_total_sites` automatically.

```python
from pg_gpu.accessible import AccessibleMask, bed_to_mask
import numpy as np

# Option A: load a saved boolean array (.npz / .npy). Print the keys
# first -- many public masks (e.g. Ag1000G Ag3.0) are keyed per
# chromosome, e.g. ['access_2L', 'access_2R', 'access_3L', ...]. Pick the
# entry that matches your data's chromosome; don't blindly grab the first.
m = np.load("agp3.is_accessible.txt.npz")
print(m.files)                                      # ['access_2L', 'access_2R', ...]
arr = m["access_3L"]                                # explicit per-chrom key
amask = AccessibleMask(arr, offset=1)               # 1-based; arr[0] -> position 1

# Option B: from a BED file. bed_to_mask returns an AccessibleMask already;
# do not wrap it in AccessibleMask(...) again. Signature:
#   bed_to_mask(path, chrom, length, offset=0) -> AccessibleMask
# `length` is the size of the boolean array to allocate (typically the
# chromosome length); `offset` is the 1-based coordinate of mask[0].
amask = bed_to_mask("access.bed", chrom="3L", length=L, offset=1)

# Quick queries
print(amask.total_accessible)             # total True positions
print(amask.count_accessible(start, end)) # callable bases in [start, end)
print(amask.is_accessible_at(1_234_567))  # single position lookup

# Attach to a HaplotypeMatrix — sets n_total_sites to the count of accessible
# bases in the matrix's [chrom_start, chrom_end] range. Does NOT filter
# variants out; the mask only changes the denominator for span-normalized
# stats and pairwise comparisons.
h.set_accessible_mask(amask)              # accepts AccessibleMask, ndarray, or BED path
# When mask_or_path is a BED file path, chrom is REQUIRED to pick the
# right contig out of the BED. Passing chrom with a ready AccessibleMask
# is allowed but ignored.
h.set_accessible_mask("access.bed", chrom='3L')

# Loaders accept it directly
h = HaplotypeMatrix.from_vcf("data.vcf.gz", region="3L:1-2000000",
                              accessible_bed="access.bed")
h = HaplotypeMatrix.from_zarr("data.zarr", accessible_bed="access.bed")

# Remove if you want to recompute
h.remove_accessible_mask()
```

Once a mask is attached, `h.n_total_sites` reflects the accessible footprint
in the matrix's range, and `span_normalize=True` (the default) automatically
divides by `n_total_sites` instead of by `chrom_end - chrom_start`. So
`diversity.pi(h)` becomes per-callable-base estimation without any extra flag.

---

## Missing Data

**Two implemented modes** (all stat functions accept `missing_data=`):

- **`'include'`** (default) — keep every site, use per-site valid sample
  counts (`n_valid`). Unbiased under MCAR.
- **`'exclude'`** — drop entire sites with any missing call before
  computing.

Unknown values for `missing_data=` silently fall through to `'include'`
(`diversity._prepare_matrix` only branches on `'exclude'`); pass one of
the two documented values or the kernel will quietly use the default.
A pixy-style `'pairwise'` mode was discussed in `docs/source/missing_data.rst`
but is not implemented anywhere in pg_gpu — do not pass it.

`'include'` automatically uses `n_total_sites` for the span-normalization
denominator when an accessibility mask is attached, so `pi` becomes
per-callable-base estimation without any second flag.

### Inspect missing data

```python
summary = h.summarize_missing_data()
print(summary)

# Per-site and per-sample counts
miss_per_site   = h.count_missing(axis=0)   # shape (n_variants,)
miss_per_sample = h.count_missing(axis=1)   # shape (n_haplotypes,)
```

### Filter missing data

```python
# Option 1: exclude all sites with ANY missing data
h_clean = h.exclude_missing_sites()
# Population-aware (only require completeness within these pops)
h_clean = h.exclude_missing_sites(populations=["pop1", "pop2"])

# Option 2: filter by per-site missing frequency threshold
h_filtered = h.filter_variants_by_missing(max_missing_freq=0.1)  # ≤10% missing
h_filtered = h.filter_variants_by_missing(max_missing_freq=0.2)  # ≤20% missing

# Option 3: keep missing, use 'include' mode (default, recommended)
pi = diversity.pi(h)                          # include mode
pi = diversity.pi(h, missing_data='exclude')  # exclude mode

# Biallelic-only filter (required for iHS, LD stats)
h_bi = h.apply_biallelic_filter()
```

### Demonstrate missing data options (test/compare)

```python
def compare_missing_strategies(h, window_size=10000):
    """Show effect of different missing data strategies on pi."""
    results = {}

    # Strategy 1: include (default, per-site n)
    results['include'] = diversity.pi(h, missing_data='include')

    # Strategy 2: exclude sites
    results['exclude'] = diversity.pi(h, missing_data='exclude')

    # Strategy 3: filter variants (10% threshold) then compute
    h10 = h.filter_variants_by_missing(max_missing_freq=0.1)
    results['filter_10pct'] = diversity.pi(h10)

    # Strategy 4: filter variants (20% threshold)
    h20 = h.filter_variants_by_missing(max_missing_freq=0.2)
    results['filter_20pct'] = diversity.pi(h20)

    # Report
    print("Missing data strategy comparison:")
    print(f"  Total variants: {h.num_variants}")
    print(f"  After filter 10%: {h10.num_variants}")
    print(f"  After filter 20%: {h20.num_variants}")
    for k, v in results.items():
        print(f"  pi ({k}): {v:.6f}")
    return results
```

---

## Diversity Statistics

```python
from pg_gpu import diversity, sfs

# Whole-chromosome
pi_val  = diversity.pi(h)
theta   = diversity.theta_w(h)
tajd    = diversity.tajimas_d(h)
he      = diversity.heterozygosity_expected(h)
# heterozygosity_observed takes a HaplotypeMatrix, not a GenotypeMatrix. It
# pairs rows into individuals: sample i is rows 2i and 2i+1, which is what
# every loader produces.
ho      = diversity.heterozygosity_observed(h)

# Per-population
pi_pop1 = diversity.pi(h, population='pop1')
tajd_p1 = diversity.tajimas_d(h, population='pop1')

# Batch (single GPU pass for all)
stats = diversity.diversity_stats(h, population='pop1',
    statistics=['pi', 'theta_w', 'theta_h', 'theta_l', 'tajimas_d'])

# SFS
afs = sfs.sfs(h)

# Neutrality test suite
nt = diversity.neutrality_tests(h, population='pop1')
```

---

## Windowed Statistics

**Namespace note.** `pg_gpu.windowed_analysis` at attribute access is the
**convenience function** (not a submodule), because `__init__.py` re-exports
the function under the same name. Use the function directly, or
`from pg_gpu.windowed_analysis import ...` to pull lower-level pieces from the
submodule.

```python
from pg_gpu import windowed_analysis

# All-in-one — single GPU pass, returns pandas DataFrame. Accepts both
# eager HaplotypeMatrix and StreamingHaplotypeMatrix (the streaming path
# tiles automatically over the source's chunks).
df = windowed_analysis(
    h, window_size=10000, step_size=5000,
    statistics=['pi', 'theta_w', 'tajimas_d', 'fst', 'dxy'],
    populations=['pop1', 'pop2'],  # required for fst/dxy
    missing_data='include',        # 'include' or 'exclude'
    span_normalize=True,           # bool; True auto-detects best denominator
)
# Columns: chrom, start, end, center, n_variants, window_id, then one
# column per requested statistic with the bare stat name ('pi', 'fst',
# 'dxy', ...). The fused kernels emit bare names regardless of how
# populations= is set — don't expect 'pi_pop1' / 'fst_pop1_pop2' suffixes
# here. (Those only appear in the slow per-window WindowedAnalyzer.compute
# path, which fires for non-fused stats like LD-bin stats.) If
# accessible_bed= is passed (or the matrix already has a mask attached),
# span_normalize uses the per-window accessible-base count as the
# denominator.

# More control: the class form (eager path; per-region calls, configured
# missing-data and span-normalization once and reused across calls).
from pg_gpu.windowed_analysis import WindowedAnalyzer
wa = WindowedAnalyzer(
    window_type='bp', window_size=10000, step_size=5000,
    statistics=['pi', 'tajimas_d'], populations=['pop1'],
    missing_data='include',
)
df = wa.compute(h)
# also: wa.compute_region(h, chrom, start, end) for a sub-region call.
# For a StreamingHaplotypeMatrix, use the convenience function above —
# it dispatches on the streaming type and tiles over the source chunks.

# Lower-level fused — bin edges instead of size/step. Returns a dict
# mapping column names to numpy arrays of shape (n_windows,). The first
# six columns are always chrom, start, end, center, n_variants, window_id.
from pg_gpu.windowed_analysis import windowed_statistics_fused
fused = windowed_statistics_fused(
    h, bp_bins=np.arange(0, 1_000_001, 10_000),
    statistics=('pi', 'theta_w', 'tajimas_d'),
)
```

Supported fused stats: single-pop `pi`, `theta_w`, `tajimas_d`,
`segregating_sites`, `singletons`; two-pop `fst`, `fst_hudson`, `fst_wc`,
`dxy`, `da`; selection `garud_h1`, `garud_h12`, `garud_h123`, `garud_h2h1`,
`mean_nsl`.

Quick plot of a windowed scan:

```python
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df['center'], df['pi'], lw=0.6)
ax.set_xlabel('position (bp)')
ax.set_ylabel(r'$\pi$')
ax.margins(x=0)
plt.tight_layout()
```

For two-pop scans (FST + dxy) put them on a twin axis with
`ax2 = ax.twinx()`; for multi-chromosome panels, build a per-chrom
DataFrame, concatenate with a `chrom` column, and facet with
`seaborn.FacetGrid` or hand-rolled subplots.

---

## Divergence Statistics

```python
from pg_gpu import divergence

fst   = divergence.fst(h, 'pop1', 'pop2')        # Hudson FST (default)
dxy   = divergence.dxy(h, 'pop1', 'pop2')
da    = divergence.da(h, 'pop1', 'pop2')
pbs   = divergence.pbs(h, 'pop1', 'pop2', 'pop3', window_size=50)

# All allele-count divergence stats in one call (fst, dxy, da, ...)
div_stats = divergence.divergence_stats(h, 'pop1', 'pop2')

# Distance-based stats (snn, dxy_min, gmin, dd1, dd2, dd_rank1, dd_rank2)
# share one pairwise-distance computation internally — call this when you
# want multiple distance-based stats without redundant GPU work.
dist_stats = divergence.distance_based_stats(h, 'pop1', 'pop2')

# Raw matrices (only if you need them directly): returns the 3-tuple
# (between, within1, within2). distance_based_stats computes these
# internally, so do not pass them back in — there's no API to inject
# pre-computed matrices.
dist_between, dist_within1, dist_within2 = (
    divergence.pairwise_distance_matrix(h, 'pop1', 'pop2')
)
```

---

## Selection Scans

```python
from pg_gpu import selection

# Requires phased, biallelic data
h_bi = h.apply_biallelic_filter()

ihs     = selection.ihs(h_bi)
ihs_std = selection.standardize(ihs)          # standardize raw scores
nsl     = selection.nsl(h_bi)

# Allele-count-binned standardization (recommended for iHS / nSL):
# bin scores by the per-variant alt allele count and z-score within bins.
# h_bi.haplotypes is a cupy array on the GPU; missing cells (-1) need to
# be masked out of the count.
import cupy as cp
hap = h_bi.haplotypes  # (n_hap, n_var), values in {0, 1, -1}
aac = cp.sum(cp.where(hap > 0, 1, 0), axis=0).get()  # alt allele count per variant
ihs_std_binned, bins = selection.standardize_by_allele_count(ihs, aac)

# Cross-population
xpehh = selection.xpehh(h_bi, 'pop1', 'pop2')

# Garud's H (haplotype homozygosity)
h1, h12, h123, h2_h1 = selection.garud_h(h_bi)
```

---

## Site Frequency Spectrum

```python
from pg_gpu import sfs as pg_sfs

# Unfolded / folded
s       = pg_sfs.sfs(h)
s_fold  = pg_sfs.sfs_folded(h)

# Scaled (divide by harmonic number)
s_sc    = pg_sfs.sfs_scaled(h)

# Joint SFS (two populations)
jsfs    = pg_sfs.joint_sfs(h, 'pop1', 'pop2')
jsfs_f  = pg_sfs.joint_sfs_folded(h, 'pop1', 'pop2')
```

---

## PCA / Dimensionality Reduction

```python
from pg_gpu import decomposition

# Standard PCA (returns PCs as numpy array, shape n_samples × n_components)
coords, explained = decomposition.pca(h, n_components=10)
# coords shape: (n_individuals, n_components)
# For HaplotypeMatrix: n_individuals = n_haplotypes

# Randomized PCA (faster for large datasets)
coords, explained = decomposition.randomized_pca(h, n_components=20)

# PCoA (distance-based)
dist = decomposition.pairwise_distance(h)
coords_pcoa = decomposition.pcoa(dist, n_components=10)

# Plot first 2 PCs with population labels. Build a per-haplotype pop label
# array from h.sample_sets: each pop maps to its haplotype-axis indices.
pop_label = np.empty(h.num_haplotypes, dtype=object)
for pop, idx in h.sample_sets.items():
    pop_label[np.asarray(idx)] = pop
df_pcs = pd.DataFrame(coords[:, :2], columns=['PC1', 'PC2'])
df_pcs['pop'] = pop_label
sns.scatterplot(df_pcs, x='PC1', y='PC2', hue='pop', s=12)
```

---

## LD Statistics

```python
from pg_gpu import ld_statistics

# Pairwise r² matrix
r2 = h.pairwise_r2()

# LD pruning
unlinked_mask = h.locate_unlinked(size=100, step=20, threshold=0.1)
h_pruned = h.get_subset(np.where(unlinked_mask)[0])

# ZnS (mean pairwise r²)
zns = ld_statistics.zns(h)

# Windowed r²
r2_vals, pair_counts = h.windowed_r_squared(
    bp_bins=np.logspace(2, 5, 20).astype(int)
)
```

---

## Block Jackknife / Bootstrap

`resampling.block_jackknife` and `resampling.block_bootstrap` operate
on **pre-binned per-block values** that the caller computes first.
Neither takes a HaplotypeMatrix or a `block_size=` kwarg — the typical
pattern is to compute a windowed statistic (the windows are the blocks)
and pass the resulting array through.

```python
from pg_gpu import resampling, windowed_analysis

# Per-block values: one windowed estimate per block. Pick a block size
# large enough to be approximately independent (LD scale, e.g. 100 kb).
df = windowed_analysis(h, window_size=100_000, statistics=['pi'])
pi_per_block = df['pi'].to_numpy()

# Block jackknife: returns (estimate, se, per_iter)
pi_est, pi_se, _ = resampling.block_jackknife(
    pi_per_block,
    statistic=lambda v: float(np.mean(v)),
)

# Block bootstrap: returns (estimate, se, replicates)
td_per_block = windowed_analysis(
    h, window_size=100_000, statistics=['tajimas_d']
)['tajimas_d'].to_numpy()
td_est, td_se, td_reps = resampling.block_bootstrap(
    td_per_block,
    statistic=lambda v: float(np.mean(v)),
    n_replicates=1000,
)
# 95% CI via percentiles:
lo, hi = np.quantile(td_reps, [0.025, 0.975])
```

For ratio-of-sums statistics (Patterson F3, Patterson D, Hudson FST,
etc.) pass the numerator and denominator as a **tuple** so the same
resampled block indices are applied to both — required for correctness:

```python
T, B = admixture.patterson_f3(h, 'C', 'A', 'B')
# Bin per-variant T and B into block-sized chunks. Trivial bp-window
# binning works for genome-wide F3; for fine-grained block sizes use
# np.add.reduceat with block boundaries from h.positions.
n_per_block = 1000  # variants per block
n_blocks = T.size // n_per_block
T_blocks = T[:n_blocks * n_per_block].reshape(n_blocks, n_per_block).sum(1)
B_blocks = B[:n_blocks * n_per_block].reshape(n_blocks, n_per_block).sum(1)

f3, f3_se, _ = resampling.block_jackknife(
    (T_blocks, B_blocks),
    statistic=lambda t, b: float(t.sum() / b.sum()),
)
```

Notes:
- `block_jackknife(values, statistic, *, weights=None)` — supply `weights` (a length-`n` array of block sizes, e.g. SNPs per block) for the Busing et al. (1999) weighted delete-`m` formulation when blocks have unequal sizes.
- `block_bootstrap(values, statistic, *, n_replicates=1000, rng=None)` — pass an integer seed or a `numpy.random.Generator` to `rng` for reproducibility.
- For per-population block-jackknife on Patterson statistics specifically, `admixture.average_patterson_f3(h, pop_c, pop_a, pop_b, blen=...)` and `average_patterson_d(...)` wrap the binning + jackknife internally and return the estimate, SE, Z-score, and supporting arrays directly.

---

## Quick Reference: Common Workflows

- **Multi-chromosome scan**: loop over `region=f"{chrom}:1-{chrom_len}"`,
  call `windowed_analysis(h, ...)`, append `chrom` column, concatenate
  the DataFrames, facet plot with `sns.FacetGrid(df, row='chrom')`.
- **PCA with population labels**: `coords, _ = decomposition.pca(h, ...);
  df = pd.DataFrame(coords[:, :2], columns=['PC1', 'PC2']); df['pop'] = ...;
  sns.scatterplot(df, x='PC1', y='PC2', hue='pop')`.
- **Windowed FST + dxy on one figure**: call `windowed_analysis(h, ...,
  statistics=['fst', 'dxy'], populations=['pop1', 'pop2'])` once, then
  plot `df['fst']` against `ax` and `df['dxy']` against `ax.twinx()` —
  the fused two-pop kernels emit bare column names, no pop-pair suffix.

---

## Common Pitfalls

1. **Shape**: `HaplotypeMatrix.shape` is `(n_haplotypes, n_variants)` — rows are haplotypes, columns are variants.
2. **`windowed_analysis` is a function, not a module** at attribute access. `windowed_analysis.windowed_statistics(...)` raises AttributeError; call `windowed_analysis(h, ...)` directly, or `from pg_gpu.windowed_analysis import windowed_statistics_fused`.
3. **`patterson_f3` / `patterson_d` return per-variant arrays, not scalars.** `T, B = admixture.patterson_f3(h, pop_c, pop_a, pop_b)` returns two `(n_variants,)` numpy arrays; the scalar F3 is `T.sum() / B.sum()`. Same shape contract for `num, den = admixture.patterson_d(...)`: scalar is `num.sum() / den.sum()`. For block-jackknife SE, prefer `admixture.average_patterson_f3(..., blen=N)` / `average_patterson_d(..., blen=N)` — they return `(estimate, se, z, vb, vj)` and bin internally. With `missing_data='exclude'` the arrays carry `0` placeholders for masked sites and sums still come out right; the explicit `nansum` form is unnecessary.
4. **Matrices live on the GPU.** Loaders (`from_vcf`, `from_zarr`) return GPU-resident matrices; all stat functions also auto-`transfer_to_gpu()` on the eager classes if a caller hand-constructed one from numpy. You almost never need to call the transfer methods yourself. To hand a result to pandas / seaborn / matplotlib, call `.get()` on the returned cupy array — that's the device-to-host step. `StreamingHaplotypeMatrix` has no `transfer_to_*` methods (it manages chunk-by-chunk residency); pass it straight to the stat functions.
5. **Selection scans need biallelic data**: always run `h.apply_biallelic_filter()` first.
6. **`span_normalize`**: default `True`; requires `chrom_start` / `chrom_end`, an `accessible_bed`, or `h.set_accessible_mask(...)` for meaningful per-base estimates.
7. **Streaming dispatch**: `streaming='never'` raises `MemoryError` when the matrix would exceed 50 % of free GPU memory. Use `streaming='auto'` for unknown-size loads — `windowed_analysis`, `diversity.*`, `sfs.sfs` / `joint_sfs`, and `selection.*` dispatch on the streaming matrix automatically. Pairwise / cross-window kernels (`pairwise_r2`, `locate_unlinked`, the full-r² heatmap form of `windowed_r_squared`, `grm`, `ibs`) cannot run streaming — call `.materialize(region=(lo, hi))` to pull a sub-region into an eager matrix for those.
8. **`fields=` is eager-only**: passing `fields=['DP', 'GQ', ...]` raises `NotImplementedError` whenever the load takes the streaming path (always under `streaming='always'`; under `streaming='auto'` only when the matrix would not fit eagerly). For larger regions, skip the loader and read `call_<TAG>` / `variant_<TAG>` directly with `zarr.open(...)`.
9. **Shared GPUs**: if CuPy lands on a saturated device the load OOMs even when other GPUs are free. Set `CUDA_VISIBLE_DEVICES=N` before importing pg_gpu.
