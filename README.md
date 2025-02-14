# pg_gpu

A package for computing population genetics statistics using CuPy.

## Installation

the conda environment is specified in `environment.yml`.

```bash
conda env create -f environment.yml
```

## Development

tests can be run with `pytest`.

```bash
pytest tests/test_haplotype_matrix.py
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


