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
| Diversity | `pi`, `theta_w`, `theta_h`, `theta_l`, `tajimas_d`, `fay_wus_h`, `zeng_e`, `segregating_sites`, `singleton_count` |
| Divergence | `fst_hudson`, `fst_weir_cockerham`, `fst_nei`, `dxy`, `da`, `pbs`, `snn`, `gmin`, `dd`, `dd_rank`, `zx` |
| SFS | `sfs`, `sfs_folded`, `joint_sfs`, `joint_sfs_folded` |
| Selection | `ihs`, `nsl`, `xpehh`, `xpnsl`, `garud_h`, `ehh_decay` |
| LD | `dd`, `dz`, `pi2`, `r_squared`, `zns`, `omega` |
| Admixture | `patterson_f2`, `patterson_f3`, `patterson_d` |
| Decomposition | `pca`, `randomized_pca`, `pairwise_distance` |
| Relatedness | `grm`, `ibs` |

## Development

```bash
pixi run pytest tests/
pixi run -e lint ruff check pg_gpu/
```
