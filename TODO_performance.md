# Performance TODO

Issues identified from full-arm Ag1000G benchmarking (3R: 10.9M variants x 2940 haplotypes).

## FIXED: xpehh regression

Capped histogram size at 50K variants (was n_variants). Now 1.7x faster
than allel at full-arm scale (279s vs 485s, was 4085s).

## FIXED: GenotypeMatrix construction OOM

Chunked diploid conversion over variant axis. Now builds in 4.8s at
full-arm scale.

## OOM: grm, ibs, pairwise_diffs, dist_var at full-arm scale

These functions create float64 intermediates of the full genotype/haplotype
matrix (n_ind x n_var in float64 = ~16GB). Each needs chunked computation
over the variant axis.

- `relatedness.grm()`: casts to float64 for frequency centering
- `relatedness.ibs()`: creates float64 difference matrix
- `distance_stats.pairwise_diffs_haploid()`: float64 for hamming distance
- `distance_stats.dist_var()`: calls pairwise_diffs

## OOM: randomized_pca at full-arm scale

`_prepare_matrix()` creates float64 intermediates of shape (n_hap, n_var).
For 2940 x 10.9M that's ~256GB in float64.

- File: `pg_gpu/decomposition.py`, `_prepare_matrix()` line ~28
- Fix: chunked scaling or use _DeferredPCA path more aggressively

## Minor: SFS/allele_frequency_spectrum slower than allel (0.4-0.5x)

allel's histogram is a simple numpy bincount, hard to beat on GPU for this
operation. Low priority since SFS is already sub-second.

- Files: `pg_gpu/sfs.py`, `pg_gpu/diversity.py`
