# Performance TODO

Issues identified from full-arm Ag1000G benchmarking (3R: 10.9M variants x 2940 haplotypes).

## Critical: xpehh regression (0.1x vs allel)

`selection.xpehh` uses `_ihh_scan_gpu` which sizes histograms to `n_variants` (10.9M).
This makes each chunk only ~46 variants, resulting in 237K kernel launches.
`ihs` avoids this via a binary `_ihh01_scan_gpu` path. xpehh needs the same
optimization or a capped histogram size.

- File: `pg_gpu/selection.py`, `_ihh_scan_gpu()` line ~1466
- Impact: 4085s vs 485s (allel), 8x slower

## OOM: GenotypeMatrix construction at full-arm scale

`GenotypeMatrix.from_haplotype_matrix()` creates boolean intermediates
(h1 < 0) | (h2 < 0) that OOM when haplotype matrix is already on GPU.
Blocks: grm, ibs, pairwise_diffs, dist_var.

- File: `pg_gpu/genotype_matrix.py`, `from_haplotype_matrix()` line ~249
- Fix: chunk the diploid conversion over the variant axis

## OOM: randomized_pca at full-arm scale

`_prepare_matrix()` creates float64 intermediates of shape (n_hap, n_var).
For 2940 x 10.9M that's ~256GB in float64.

- File: `pg_gpu/decomposition.py`, `_prepare_matrix()` line ~28
- Fix: chunked scaling or use _DeferredPCA path more aggressively

## Minor: SFS/allele_frequency_spectrum slower than allel (0.4-0.5x)

allel's histogram is a simple numpy bincount, hard to beat on GPU for this
operation. Low priority since SFS is already sub-second.

- Files: `pg_gpu/sfs.py`, `pg_gpu/diversity.py`
