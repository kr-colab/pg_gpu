import msprime

# simulate a vcf file with msprime

ts = msprime.sim_ancestry(
    samples=10,
    population_size=1000,
    recombination_rate=1e-8,
    sequence_length=10_000_000,
    random_seed=42
)

ts = msprime.sim_mutations(ts, rate=1e-8)

ts.dump("data/test2.trees")
# write to vcf
with open("data/test2.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file)
