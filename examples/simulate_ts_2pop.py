"""
Simulate nucleotide data from a Jukes-Cantor model with 2 populations and a split.
Save compressed tree sequence.
"""
import argparse
import os
import msprime
import tszip

parser = argparse.ArgumentParser()
parser.add_argument("--popsize", type=int, default=10000)
parser.add_argument("--mu", type=float, default=1e-8)
parser.add_argument("--samples", type=int, default=50, help="Samples per population")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--r", type=float, default=1e-8)
parser.add_argument("--seqlen", type=float, default=1e6)
parser.add_argument("--split_time", type=int, default=1000,
                    help="Population split time in generations")
parser.add_argument("--migration_rate", type=float, default=0.0,
                    help="Symmetric migration rate between populations")
parser.add_argument("--output", type=str, default="trees_2pop.tsz")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

demography = msprime.Demography()
demography.add_population(name="pop0", initial_size=args.popsize)
demography.add_population(name="pop1", initial_size=args.popsize)
demography.add_population(name="ancestral", initial_size=args.popsize)
demography.add_population_split(time=args.split_time, derived=["pop0", "pop1"], ancestral="ancestral")
if args.migration_rate > 0:
    demography.set_symmetric_migration_rate(["pop0", "pop1"], args.migration_rate)

ts = msprime.sim_ancestry(
    samples={"pop0": args.samples, "pop1": args.samples},
    demography=demography,
    recombination_rate=args.r,
    sequence_length=args.seqlen,
    random_seed=args.seed,
)

ts = msprime.sim_mutations(
    ts,
    rate=args.mu,
    model=msprime.JC69(state_independent=True),
    keep=False,
    random_seed=args.seed + 1,
)

tszip.compress(ts, args.output)
print(f"Saved: {args.output} | samples: {args.samples}/pop | trees: {ts.num_trees} | split: {args.split_time} gen")

