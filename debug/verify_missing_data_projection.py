#!/usr/bin/env python
"""
Verify theta estimation under missing data with and without SFS projection.

Simulates under a standard neutral model with known theta = 4*N*mu*L,
injects missing data at varying rates, and compares:
  1. FrequencySpectrum with missing_data='include' (group-by-n approach)
  2. FrequencySpectrum with .project(target_n) (hypergeometric projection)
  3. FrequencySpectrum with missing_data='exclude' (drop incomplete sites)

For each approach we examine bias (E[theta_hat] vs true theta) and
variance across replicates.

Usage:
    pixi run python debug/verify_missing_data_projection.py
"""

import numpy as np
import msprime
from pg_gpu import HaplotypeMatrix
from pg_gpu.achaz import FrequencySpectrum


# ── Simulation parameters ───────────────────────────────────────────────
N = 10_000           # diploid effective population size
MU = 1e-8            # per-site per-generation mutation rate
L = 200_000          # sequence length
N_HAP = 100          # number of haploid chromosomes (50 diploids)
N_REPS = 200         # replicates per missing rate
MISSING_RATES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40]
ESTIMATORS = ['pi', 'watterson', 'theta_h', 'theta_l']

THETA_TRUE = 4 * N * MU * L  # expected total theta (unnormalized)


def simulate_haplotypes(seed):
    """Simulate haplotypes under standard neutral model."""
    ts = msprime.sim_ancestry(
        samples=N_HAP // 2,
        sequence_length=L,
        recombination_rate=1e-8,
        population_size=N,
        random_seed=seed,
        ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    return ts.genotype_matrix().T, ts.tables.sites.position


def inject_missing(haplotypes, rate, rng):
    """Randomly set entries to -1 (missing) at the given rate."""
    if rate == 0:
        return haplotypes.copy()
    hap = haplotypes.copy()
    mask = rng.random(hap.shape) < rate
    hap[mask] = -1
    return hap


def estimate_thetas(hm, method, projection_n=None):
    """Compute theta estimates using the specified method."""
    if method == 'include':
        fs = FrequencySpectrum(hm, missing_data='include')
    elif method == 'exclude':
        fs = FrequencySpectrum(hm, missing_data='exclude')
    elif method == 'project':
        fs = FrequencySpectrum(hm, missing_data='include')
        if projection_n is not None and fs.n_max >= projection_n:
            fs = fs.project(projection_n)
        else:
            return {name: np.nan for name in ESTIMATORS}
    else:
        raise ValueError(f"Unknown method: {method}")

    return {name: fs.theta(name) for name in ESTIMATORS}


def main():
    print(f"Simulation: N={N}, mu={MU}, L={L:,}, n={N_HAP} haplotypes")
    print(f"True theta = 4*N*mu*L = {THETA_TRUE:.2f}")
    print(f"Replicates: {N_REPS}")
    print()

    # Determine projection target: use the minimum n that all sites
    # will have even at the highest missing rate.
    # At 40% missing, expected n_valid per site ~ 0.6 * 100 = 60.
    # Use a conservative target: 50
    projection_n = 50

    print(f"{'Missing':>8s} {'Method':<10s}", end="")
    for est in ESTIMATORS:
        print(f" {'E['+est+']':>14s} {'bias%':>7s} {'SD':>10s}", end="")
    print()
    print("-" * (20 + len(ESTIMATORS) * 32))

    for miss_rate in MISSING_RATES:
        results = {
            'include': {e: [] for e in ESTIMATORS},
            'exclude': {e: [] for e in ESTIMATORS},
            'project': {e: [] for e in ESTIMATORS},
        }

        rng = np.random.default_rng(42)

        for rep in range(N_REPS):
            seed = rep + 1
            hap_clean, positions = simulate_haplotypes(seed)

            if hap_clean.shape[1] < 2:
                continue

            hap_missing = inject_missing(hap_clean, miss_rate, rng)
            positions_np = np.array(positions, dtype=np.float64)
            hm = HaplotypeMatrix(hap_missing, positions_np)

            for method in ['include', 'exclude', 'project']:
                proj_n = projection_n if method == 'project' else None
                try:
                    thetas = estimate_thetas(hm, method, projection_n=proj_n)
                    for est in ESTIMATORS:
                        v = thetas[est]
                        if np.isfinite(v):
                            results[method][est].append(v)
                except Exception:
                    pass

        # Print results for this missing rate
        for method in ['include', 'exclude', 'project']:
            label = f"{miss_rate:.0%}" if method == 'include' else ""
            print(f"{label:>8s} {method:<10s}", end="")
            for est in ESTIMATORS:
                vals = results[method][est]
                if len(vals) > 5:
                    mean = np.mean(vals)
                    sd = np.std(vals)
                    bias_pct = 100 * (mean - THETA_TRUE) / THETA_TRUE
                    print(f" {mean:>14.2f} {bias_pct:>+6.1f}% {sd:>10.2f}", end="")
                else:
                    print(f" {'---':>14s} {'---':>7s} {'---':>10s}", end="")
            print()
        print()

    print(f"\nNotes:")
    print(f"  - True theta = {THETA_TRUE:.2f}")
    print(f"  - 'include': groups variants by per-site sample size, applies n-specific weights")
    print(f"  - 'exclude': drops sites with any missing data, uses fixed n={N_HAP}")
    print(f"  - 'project': projects all sites to n={projection_n} via hypergeometric sampling")
    print(f"  - bias% = (E[theta_hat] - theta_true) / theta_true * 100")


if __name__ == "__main__":
    main()
