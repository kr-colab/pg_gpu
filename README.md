# pg_gpu

GPU-accelerated population genetics statistics using CuPy.

[![Documentation Status](https://readthedocs.org/projects/pg-gpu/badge/?version=latest)](https://pg-gpu.readthedocs.io/en/latest/?badge=latest)

## Installation

the conda environment is specified in `environment.yml`.

```bash
conda env create -f environment.yml
```

## Documentation

Full documentation is available at [https://pg-gpu.readthedocs.io/](https://pg-gpu.readthedocs.io/)

## Quick Start

```python
from pg_gpu import HaplotypeMatrix, ld_statistics

# Load data
h = HaplotypeMatrix.from_vcf("data.vcf")

# Compute LD statistics
result = h.tally_gpu_haplotypes()
if isinstance(result, tuple):
    counts, n_valid = result
else:
    counts, n_valid = result, None
dd_vals = ld_statistics.dd(counts, n_valid=n_valid)
```

## Development

tests can be run with `pytest`.

```bash
pytest tests/
```



## Usage

```python
import msprime
import numpy as np
from pg_gpu.haplotype_matrix import HaplotypeMatrix

# do a simulation
ts = msprime.sim_ancestry(
    samples=10,
    sequence_length=1e5,
    recombination_rate=1e-8,
    population_size=10000,
    ploidy=2,
    discrete_genome=False
    )
ts = msprime.sim_mutations(ts, rate=1e-8, model="binary")

# create a haplotype matrix
h = HaplotypeMatrix.from_ts(ts)

# compute pi, compare to tskit
print(ts.diversity(mode="site"))
print(h.diversity(span_normalize=True))

# compute pairwise LD
D = h.pairwise_LD_v()
```


