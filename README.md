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

## Quick Start

```python
from pg_gpu import HaplotypeMatrix, diversity, divergence, selection, sfs

# Load from VCF
hm = HaplotypeMatrix.from_vcf("data.vcf.gz", region="chr1:1-1000000")
hm.load_pop_file("populations.txt")

# Diversity
diversity.pi(hm, population="pop1")
diversity.tajimas_d(hm, population="pop1")

# Divergence
divergence.fst_hudson(hm, "pop1", "pop2")
divergence.dxy(hm, "pop1", "pop2")

# Selection scans
selection.ihs(hm)
selection.nsl(hm)

# Windowed statistics (fused CUDA kernels)
from pg_gpu import windowed_analysis
results = windowed_analysis(hm, statistics=["pi", "theta_w", "tajimas_d"],
                            window_size=50000)
```

## Documentation

Full documentation at [https://pg-gpu.readthedocs.io/](https://pg-gpu.readthedocs.io/).

Interactive walkthrough: [examples/pg_gpu_tour.ipynb](examples/pg_gpu_tour.ipynb).

## Statistics

| Category | Functions |
|----------|-----------|
| Diversity | `pi`, `theta_w`, `theta_h`, `theta_l`, `tajimas_d`, `fay_wus_h`, `normalized_fay_wus_h`, `zeng_e`, `zeng_dh`, `segregating_sites`, `singleton_count`, `haplotype_diversity`, `haplotype_count`, `heterozygosity_expected`, `heterozygosity_observed`, `inbreeding_coefficient`, `allele_frequency_spectrum`, `max_daf`, `daf_histogram`, `diplotype_frequency_spectrum`, `diversity_stats` |
| Divergence | `fst_hudson`, `fst_weir_cockerham`, `fst_nei`, `dxy`, `da`, `pbs`, `pairwise_fst` |
| Distance-based two-pop | `snn`, `dxy_min`, `gmin`, `dd`, `dd_rank`, `zx` |
| Distance moments | `pairwise_diffs`, `dist_var`, `dist_skew`, `dist_kurt`, `dist_moments` |
| Selection scans | `ihs`, `nsl`, `xpehh`, `xpnsl`, `garud_h`, `moving_garud_h`, `ehh_decay` |
| LD | `r`, `r_squared`, `dd` (LD), `dz`, `pi2`, `zns`, `omega`, `mu_ld` |
| SFS | `sfs`, `sfs_folded`, `sfs_scaled`, `sfs_folded_scaled`, `joint_sfs`, `joint_sfs_folded`, `joint_sfs_scaled`, `joint_sfs_folded_scaled`, `project_joint_sfs`, `fold_sfs`, `fold_joint_sfs` |
| Admixture / F-stats | `patterson_f2`, `patterson_f3`, `patterson_d`, `moving_patterson_f3`, `moving_patterson_d`, `average_patterson_f3`, `average_patterson_d` |
| Resampling | `block_jackknife`, `block_bootstrap` |
| Decomposition | `pca`, `randomized_pca`, `pairwise_distance`, `pcoa`, `local_pca`, `local_pca_jackknife`, `pc_dist`, `corners` |
| Relatedness | `grm`, `ibs` |
| Windowed pipeline | `windowed_analysis` — fused GPU windowing for any of the above |
| Biobank-scale streaming | `HaplotypeMatrix.from_zarr(streaming='always')` walks VCZ stores chunk by chunk; every per-window / SFS / LD / pairwise relatedness statistic dispatches transparently. See [tutorials/biobank_streaming](https://pg-gpu.readthedocs.io/en/latest/tutorials/biobank_streaming.html). |

## Development

```bash
pixi run pytest tests/
pixi run -e lint ruff check pg_gpu/
```
