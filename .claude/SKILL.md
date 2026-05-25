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

pg_gpu requires a CUDA GPU. On CPU-only machines, arrays stay on CPU
(`transfer_to_gpu()` / `transfer_to_cpu()` move data). Always check
`h.device` if results look wrong.

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
> makes subsequent loads/QC inspection fit in GPU memory. Full recipe
> See the upstream skill bundle for the full recipe.

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

### Loading on GPU: streaming and device selection

`from_zarr` and `from_vcf` materialize the haplotype matrix on the GPU
by default. pg_gpu refuses up front if the matrix would exceed 50% of
free GPU memory (raises `MemoryError` with a clear remediation hint);
beyond the cap, you have to pick a streaming strategy.

> **`fields=` only works with `streaming='never'`.** This is a hard
> library constraint — `fields=` raises `NotImplementedError` on the
> streaming path. In practice this means QC-field inspection
> (`fields=['DP', 'GQ', ...]`) through pg_gpu's loader is only feasible
> on regions whose full haplotype matrix fits in <50% of free GPU
> memory. For larger regions, **skip pg_gpu's loader and read the QC
> arrays directly with `zarr.open()`** — the QC fields are plain
> numpy/zarr datasets at `variant_<TAG>` / `call_<TAG>` keys, no GPU
> involved, no memory cap, ~36s for 10 Mb on Ag1000G. Recipe in the upstream skill bundle.

```python
# 'never'  -> always materialize; raise MemoryError if too big.
# 'auto'   -> materialize if it fits, else return StreamingHaplotypeMatrix.
# 'always' -> always return StreamingHaplotypeMatrix.
h = HaplotypeMatrix.from_zarr(VCZ, region='X:1-10000000', streaming='auto')
print(type(h).__name__)   # 'HaplotypeMatrix' or 'StreamingHaplotypeMatrix'
```

Size estimate before loading: `(n_variants × n_haplotypes × 2 bytes)`.
A 10 Mb region of *A. gambiae* X with 1,470 diploids is ~29 GB; an 80 GB
A100 has plenty if free, but on a shared box other jobs eat that fast.
`StreamingHaplotypeMatrix` runs the same statistical functions
(`diversity.*`, `windowed_analysis`, `selection.*`, …) chunk-by-chunk
over the GPU; results are identical, throughput a few× slower than the
in-memory path.

**Pin to a specific GPU on shared machines.** Check `nvidia-smi` for
free memory; CuPy picks index 0 by default. If that one's saturated:

```bash
CUDA_VISIBLE_DEVICES=0 python my_script.py
# or in Python, BEFORE importing pg_gpu/cupy:
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

**MemoryLimitedWarning on `from_vcf`** (separate check, host-RAM not
GPU): pg_gpu emits this UserWarning when the source VCF is >10 GB and
the region is >5 Mb (or unspecified), or the file has >5,000 samples.
htslib's VCF parse is single-threaded and the full genotype matrix
must fit in host RAM. Either accept the slow load, or do the one-time
`vcf_to_zarr` conversion. Silence with:

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
the **denominator** for span-normalized stats (pi, dxy, theta_w) and as
the canonical `n_total_sites` for `missing_data='pairwise'`.

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

# Option B: from a BED file
amask = AccessibleMask(bed_to_mask("access.bed", chrom_length=L), offset=1)

# Quick queries
print(amask.total_accessible)             # total True positions
print(amask.count_accessible(start, end)) # callable bases in [start, end)
print(amask.is_accessible_at(1_234_567))  # single position lookup

# Attach to a HaplotypeMatrix — sets n_total_sites to the count of accessible
# bases in the matrix's [chrom_start, chrom_end] range. Does NOT filter
# variants out; the mask only changes the denominator for span-normalized
# stats and pairwise comparisons.
h.set_accessible_mask(amask)              # accepts AccessibleMask or path/.bed/.npz
# Optional: per-chromosome data → pass chrom explicitly
h.set_accessible_mask(amask, chrom='3L')

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

- **`'include'`** (default) — per-site `n_valid`, unbiased under MCAR.
- **`'exclude'`** — drop entire sites with any missing call.

The `docs/source/missing_data.rst` page documents a `'pairwise'` (pixy-style)
mode, but it is **not implemented in the current code**: `diversity.pi`
silently accepts unknown `missing_data=` values and falls through to
`'include'`. Verify modes you rely on against the source for the function
in question. The legacy `'ignore'` mode has also been removed.

`include` automatically uses `n_total_sites` for the span-normalization
denominator when an accessibility mask is attached, so `pi` becomes
per-callable-base automatically — no second flag needed.

See the upstream skill bundle for the full table of what each mode does per stat.

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
from pg_gpu import diversity

# Whole-chromosome
pi_val  = diversity.pi(h)
theta   = diversity.theta_w(h)
tajd    = diversity.tajimas_d(h)
he      = diversity.heterozygosity_expected(h)
ho      = diversity.heterozygosity_observed(h)   # requires GenotypeMatrix

# Per-population
pi_pop1 = diversity.pi(h, population='pop1')
tajd_p1 = diversity.tajimas_d(h, population='pop1')

# Batch (single GPU pass for all)
stats = diversity.diversity_stats(h, population='pop1',
    statistics=['pi', 'theta_w', 'theta_h', 'theta_l', 'tajimas_d'])

# SFS
afs = diversity.allele_frequency_spectrum(h)

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

# All-in-one — single GPU pass, returns pandas DataFrame
df = windowed_analysis(
    h, window_size=10000, step_size=5000,
    statistics=['pi', 'theta_w', 'tajimas_d', 'fst', 'dxy'],
    populations=['pop1', 'pop2'],   # required for fst/dxy
    missing_data='include',         # 'include' | 'exclude' | 'pairwise'
    span_denominator='total',       # 'total' | 'sites' | 'callable'
)
# df columns: window_start, window_stop, n_variants, pi, theta_w, ...,
#             fst_pop1_pop2, dxy_pop1_pop2.
# In pairwise mode: also _diffs/_comps/_missing component columns.

# More control: the class form (streaming, region-restricted)
from pg_gpu.windowed_analysis import WindowedAnalyzer
wa = WindowedAnalyzer(
    window_type='bp', window_size=10000, step_size=5000,
    statistics=['pi','tajimas_d'], populations=['pop1'],
    missing_data='include',
)
df = wa.compute(h)
# also: wa.compute_region(h, chrom, start, end), wa.compute_streaming(h)

# Lower-level fused — bin edges instead of size/step, returns dict
from pg_gpu.windowed_analysis import windowed_statistics_fused
fused = windowed_statistics_fused(
    h, bp_bins=np.arange(0, 1_000_001, 10_000),
    statistics=('pi','theta_w','tajimas_d'),
)
```

Supported fused stats: single-pop `pi`, `theta_w`, `tajimas_d`,
`segregating_sites`, `singletons`; two-pop `fst`, `fst_hudson`, `fst_wc`,
`dxy`, `da`; selection `garud_h1`, `garud_h12`, `garud_h123`, `garud_h2h1`,
`mean_nsl`.

For plotting windowed results, see the upstream skill bundle.

---

## Divergence Statistics

```python
from pg_gpu import divergence

fst   = divergence.fst(h, 'pop1', 'pop2')        # Hudson FST (default)
dxy   = divergence.dxy(h, 'pop1', 'pop2')
da    = divergence.da(h, 'pop1', 'pop2')
pbs   = divergence.pbs(h, 'pop1', 'pop2', 'pop3', window_size=50)

# All divergence stats in one call
div_stats = divergence.divergence_stats(h, 'pop1', 'pop2')

# Distance-based stats (pre-compute matrix once for efficiency)
dm = divergence.pairwise_distance_matrix(h, 'pop1', 'pop2')
all_dist = divergence.distance_based_stats(h, 'pop1', 'pop2',
                                           distance_matrices=dm)
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

# Allele-count-binned standardization (recommended for iHS/nSL)
dac = np.sum(np.maximum(h_bi.haplotypes if h_bi.device == 'CPU'
              else h_bi.haplotypes.get(), 0), axis=0)
ihs_std_binned, bins = selection.standardize_by_allele_count(ihs, dac)

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

# Plot first 3 PCs — see the upstream skill bundle for examples
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

```python
from pg_gpu import resampling

# Block jackknife CI on pi
pi_est, pi_se, pi_ci = resampling.block_jackknife(
    h, statistic=lambda m: diversity.pi(m),
    block_size=100000
)

# Bootstrap
samples = resampling.block_bootstrap(
    h, statistic=lambda m: diversity.tajimas_d(m),
    n_bootstrap=200, block_size=50000
)
```

---

## Quick Reference: Common Workflows

For full code, see the upstream skill bundle:
- **Multi-chromosome panel**: load per-chrom → compute stats → combine → panel plot
- **PCA with population labels**: PCA → DataFrame → seaborn scatterplot with hue
- **Windowed FST + dxy**: fused windowed → DataFrame → dual-axis plot

---

## Common Pitfalls

1. **Shape**: `HaplotypeMatrix.shape` is `(n_haplotypes, n_variants)` — rows are haplotypes, columns are variants.
2. **`windowed_analysis` is a function, not a module** at attribute access. `windowed_analysis.windowed_statistics(...)` raises AttributeError; call `windowed_analysis(h, ...)` directly, or `from pg_gpu.windowed_analysis import windowed_statistics_fused`.
3. **`patterson_f3` / `patterson_d` return per-variant arrays, not scalars**: `T, B = admixture.patterson_f3(...)` and the scalar is `np.nansum(T)/np.nansum(B)`. Same shape for `patterson_d` `(num, den)`. Use `average_patterson_*(..., blen=...)` for block-jackknife SE (returns 5-tuple).
4. **GPU transfer**: on an eager `HaplotypeMatrix` call `h.transfer_to_gpu()` before heavy computation on GPU machines; `h.transfer_to_cpu()` before pandas/seaborn (which expect numpy). **`StreamingHaplotypeMatrix` has neither method** — it manages its own chunk-by-chunk GPU residency. Calling `transfer_to_gpu()` on a streaming matrix raises `AttributeError`; just pass it straight to the stat functions.
5. **Selection scans need biallelic data**: always run `h.apply_biallelic_filter()` first.
6. **span_normalize**: default `True`; requires `chrom_start`/`chrom_end`, an `accessible_bed`, or `h.set_accessible_mask(...)` for meaningful per-base estimates.
7. **CuPy arrays**: `.get()` converts CuPy array → numpy. E.g. `coords_np = coords.get()` if needed.
8. **`streaming='never'` raises `MemoryError` past 50% of free GPU memory.** Use `streaming='auto'` for unknown-size loads on GPU machines (returns a `StreamingHaplotypeMatrix` when the materialized matrix won't fit). All stat functions accept both types. See *Loading on GPU* above. **But `fields=` is the exception**: it requires `streaming='never'` and raises `NotImplementedError` on the streaming path — shrink the region until the matrix fits before loading QC fields.
9. **Shared GPUs**: if CuPy lands on a saturated device the load OOMs even when other GPUs are free. Set `CUDA_VISIBLE_DEVICES=N` before importing pg_gpu.
