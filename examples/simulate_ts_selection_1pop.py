"""
Simulate nucleotide data from a Jukes-Cantor model around a sweep.
Save compressed tree sequence.
"""
import argparse
import os
import msprime
import tszip

parser = argparse.ArgumentParser()
parser.add_argument("--popsize", type=int, default=10000)
parser.add_argument("--mu", type=float, default=1e-8)
parser.add_argument("--s", type=float, default=0.01)
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--r", type=float, default=1e-8)
parser.add_argument("--seqlen", type=float, default=1e6)
parser.add_argument("--fixation_time", type=int, default=None,
                    help="Generations since sweep ended (sweep age). Default: auto from popsize.")
parser.add_argument("--output", type=str, default="trees.tsz")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None


if args.s > 0:
    fixation_time = args.fixation_time if args.fixation_time is not None else max(1, int(0.05 * args.popsize))  # 5% of Ne


if args.s == 0:
    ts = msprime.sim_ancestry(
        args.samples,
        recombination_rate=args.r,
        sequence_length=args.seqlen,
        population_size=args.popsize,
        random_seed=args.seed,
    )
else:
    ts = msprime.sim_ancestry(
        args.samples,
        recombination_rate=args.r,
        sequence_length=args.seqlen,
        population_size=args.popsize,
        random_seed=args.seed,
        model=[
            msprime.StandardCoalescent(duration=fixation_time),
            msprime.SweepGenicSelection(
                s=args.s,
                position=0.5 * args.seqlen,
                start_frequency=1 / args.popsize / 2,
                end_frequency=1 - 1 / args.popsize / 2,
                dt=1 / (100 * args.popsize),
            ),
            msprime.StandardCoalescent(),
        ],
    )

ts = msprime.sim_mutations(
    ts,
    rate=args.mu,
    model=msprime.JC69(state_independent=True),
    keep=False,
    random_seed=args.seed + 1,
)

tszip.compress(ts, args.output)
print(f"Saved: {args.output}")


