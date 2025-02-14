import msprime

# simulate a vcf file with msprime

ts = msprime.sim_ancestry(
    samples=10,
    population_size=1000,
    recombination_rate=1e-8,
    sequence_length=1_000_000,
    random_seed=42
)

ts = msprime.sim_mutations(ts, rate=1e-8)

# write to vcf
with open("data/test.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file)
