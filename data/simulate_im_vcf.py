import msprime
import demes
import os

# set up simulation parameters
L = 1e6
u = r = 1.5e-8
n = 10

g = demes.load("data/demes_mod.yaml")
demog = msprime.Demography.from_demes(g)

trees = msprime.sim_ancestry(
    {"deme0": n, "deme1": n},
    demography=demog,
    sequence_length=L,
    recombination_rate=r,
    random_seed=321,
)

trees = msprime.sim_mutations(trees, rate=u, random_seed=123)

with open("data/im-parsing-example.vcf", "w+") as fout:
    trees.write_vcf(fout)