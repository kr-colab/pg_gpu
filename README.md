# pg_gpu

GPU-accelerated population genetics statistics using CuPy.

[![Documentation Status](https://readthedocs.org/projects/pg-gpu/badge/?version=latest)](https://pg-gpu.readthedocs.io/en/latest/?badge=latest)

## Installation

pg_gpu uses [pixi](https://pixi.sh) for environment management.
Requires an NVIDIA GPU.

```bash
pixi install
pixi shell
```

## Documentation

Full documentation is available at [https://pg-gpu.readthedocs.io/](https://pg-gpu.readthedocs.io/).

For an interactive walkthrough of the library, see the [examples/pg_gpu_tour.ipynb](examples/pg_gpu_tour.ipynb) notebook.

## Quick Start

```python
from pg_gpu import HaplotypeMatrix, diversity, divergence, selection, sfs

# Load from VCF
hm = HaplotypeMatrix.from_vcf("data.vcf.gz", region="chr1:1-1000000")
hm.load_pop_file("populations.txt")
hm.transfer_to_gpu()

# Diversity statistics
diversity.pi(hm, population="pop1")
diversity.tajimas_d(hm, population="pop1")
diversity.theta_w(hm, population="pop1")

# Divergence
divergence.fst_hudson(hm, "pop1", "pop2")
divergence.dxy(hm, "pop1", "pop2")

# Site frequency spectra
sfs.sfs(hm, population="pop1")
sfs.joint_sfs(hm, "pop1", "pop2")

# Selection scans
selection.ihs(hm, map_pos=genetic_map)
selection.nsl(hm)
```

## Windowed Statistics

Fused CUDA kernels compute multiple statistics per window in a single pass:

```python
from pg_gpu import windowed_analysis

results = windowed_analysis(hm, statistics=["pi", "theta_w", "tajimas_d"],
                            window_size=50000, step_size=10000)
```

## Achaz Framework

Compute all theta estimators and neutrality tests from a single SFS pass using the
[Achaz (2009)](https://doi.org/10.1534/genetics.109.104042) generalized framework:

```python
from pg_gpu import FrequencySpectrum

# Build SFS (one GPU pass)
fs = FrequencySpectrum(hm, population="pop1")

# All standard estimators as weight vector dot products
fs.theta("pi")          # nucleotide diversity
fs.theta("watterson")   # Watterson's theta
fs.theta("theta_h")     # Fay & Wu's theta_H
fs.theta("theta_l")     # Zeng's theta_L
fs.theta("eta1")        # singleton-based theta (Fu & Li)

# Neutrality tests
fs.tajimas_d()          # Tajima's D
fs.fay_wu_h()           # Fay & Wu's H

# All at once (8 thetas + 4 tests)
fs.all_thetas()
fs.all_tests()

# SFS projection for missing data handling
fs_projected = fs.project(target_n=50)

# Custom weight vector
fs.theta(lambda n: my_weights(n))

# Batch computation (uses Achaz internally)
diversity.diversity_stats_fast(hm, population="pop1")
```

## Loading Data

```python
# From VCF (supports region queries and sample subsetting)
hm = HaplotypeMatrix.from_vcf("data.vcf.gz", region="chr1:1-1000000")

# From tree sequences (msprime, SLiM)
hm = HaplotypeMatrix.from_ts(ts)

# From Zarr (fast reloading)
hm.to_zarr("data.zarr")
hm = HaplotypeMatrix.from_zarr("data.zarr")

# Population labels
hm.load_pop_file("populations.txt")  # tab-delimited: sample\tpopulation
```

## Statistics

29 statistics validated against scikit-allel at machine precision.

| Category | Functions |
|----------|-----------|
| Diversity | `pi`, `theta_w`, `theta_h`, `theta_l`, `tajimas_d`, `fay_wus_h`, `normalized_fay_wus_h`, `zeng_e`, `zeng_dh`, `segregating_sites`, `singleton_count` |
| Divergence | `fst_hudson`, `fst_weir_cockerham`, `fst_nei`, `dxy`, `da`, `pbs` |
| SFS | `sfs`, `sfs_folded`, `joint_sfs`, `joint_sfs_folded` (plus scaled variants) |
| Selection | `ihs`, `nsl`, `xpehh`, `xpnsl`, `garud_h`, `moving_garud_h`, `ehh_decay` |
| LD | `dd`, `dz`, `pi2`, `pairwise_r2` |
| Admixture | `patterson_f2`, `patterson_f3`, `patterson_d` |
| Decomposition | `pca`, `randomized_pca`, `pcoa`, `pairwise_distance` |
| Relatedness | `grm`, `ibs` |

All functions support three missing data modes: `'include'` (default), `'exclude'`, and `'pairwise'`.

## Development

```bash
# Run tests
pixi run pytest tests/

# Run full validation against scikit-allel
pixi run python tests/validate_against_allel.py

# Lint
pixi run -e lint ruff check pg_gpu/
```
