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


def make_figure():
    """Generate publication-quality figure showing bias and variance
    of theta estimators under missing data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("Running simulations for figure...", flush=True)

    N = 10_000
    mu = 1e-8
    L = 200_000
    n_hap = 100
    n_reps = 200
    missing_rates = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
    theta_true = 4 * N * mu * L
    projection_n = 50
    estimators = ['pi', 'watterson', 'theta_h', 'theta_l']
    methods = ['include', 'project', 'exclude']
    method_labels = {
        'include': 'Group by $n_i$ (default)',
        'project': f'Project to $n={projection_n}$',
        'exclude': 'Exclude incomplete sites',
    }
    method_colors = {
        'include': '#2ecc71',
        'project': '#3498db',
        'exclude': '#e74c3c',
    }
    est_labels = {
        'pi': r'$\hat{\theta}_\pi$',
        'watterson': r'$\hat{\theta}_W$',
        'theta_h': r'$\hat{\theta}_H$',
        'theta_l': r'$\hat{\theta}_L$',
    }

    # Collect results: {method: {estimator: {rate: [values]}}}
    all_results = {m: {e: {r: [] for r in missing_rates}
                       for e in estimators} for m in methods}

    rng = np.random.default_rng(42)

    for rep in range(n_reps):
        if (rep + 1) % 50 == 0:
            print(f"  rep {rep + 1}/{n_reps}", flush=True)
        seed = rep + 1
        hap_clean, positions = simulate_haplotypes(seed)
        if hap_clean.shape[1] < 2:
            continue
        positions_np = np.array(positions, dtype=np.float64)

        for rate in missing_rates:
            hap_miss = inject_missing(hap_clean, rate, rng)
            hm = HaplotypeMatrix(hap_miss, positions_np)

            for method in methods:
                proj_n = projection_n if method == 'project' else None
                try:
                    thetas = estimate_thetas(hm, method, projection_n=proj_n)
                    for est in estimators:
                        v = thetas[est]
                        if np.isfinite(v):
                            all_results[method][est][rate].append(v)
                except Exception:
                    pass

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    rates_pct = [r * 100 for r in missing_rates]

    for idx, est in enumerate(estimators):
        ax = fig.add_subplot(gs[idx])

        for method in methods:
            means = []
            ci_lo = []
            ci_hi = []
            for rate in missing_rates:
                vals = all_results[method][est][rate]
                if len(vals) > 5:
                    m = np.mean(vals)
                    se = np.std(vals) / np.sqrt(len(vals))
                    means.append(m)
                    ci_lo.append(m - 1.96 * se)
                    ci_hi.append(m + 1.96 * se)
                else:
                    means.append(np.nan)
                    ci_lo.append(np.nan)
                    ci_hi.append(np.nan)

            means = np.array(means)
            ci_lo = np.array(ci_lo)
            ci_hi = np.array(ci_hi)
            valid = np.isfinite(means)

            ax.plot(np.array(rates_pct)[valid], means[valid],
                    'o-', color=method_colors[method],
                    label=method_labels[method], markersize=5, linewidth=1.5)
            ax.fill_between(np.array(rates_pct)[valid],
                            ci_lo[valid], ci_hi[valid],
                            alpha=0.15, color=method_colors[method])

        ax.axhline(theta_true, color='black', linestyle='--',
                    linewidth=1, alpha=0.7, label=r'True $\theta$')
        ax.set_xlabel('Missing data rate (%)')
        ax.set_ylabel(r'$E[\hat{\theta}]$')
        ax.set_title(est_labels[est], fontsize=14)
        ax.set_ylim(0, theta_true * 1.5)

        if idx == 0:
            ax.legend(fontsize=8, loc='lower left')

    fig.suptitle(
        r'Theta estimation under missing data ($\theta_{true}$ = '
        f'{theta_true:.0f}, n = {n_hap}, '
        f'{n_reps} replicates)\n'
        'Shaded regions: 95% CI of the mean',
        fontsize=13, y=1.01)

    outpath = 'debug/verify_missing_data_projection.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {outpath}")

    # ── Also make a bias + variance panel ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Relative bias
    ax = axes2[0]
    for method in methods:
        biases = []
        for rate in missing_rates:
            # Average bias across all four estimators
            bias_per_est = []
            for est in estimators:
                vals = all_results[method][est][rate]
                if len(vals) > 5:
                    bias_per_est.append(
                        (np.mean(vals) - theta_true) / theta_true * 100)
            if bias_per_est:
                biases.append(np.mean(bias_per_est))
            else:
                biases.append(np.nan)

        biases = np.array(biases)
        valid = np.isfinite(biases)
        ax.plot(np.array(rates_pct)[valid], biases[valid],
                'o-', color=method_colors[method],
                label=method_labels[method], markersize=6, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Missing data rate (%)', fontsize=12)
    ax.set_ylabel('Mean relative bias (%)', fontsize=12)
    ax.set_title('Bias (averaged across 4 estimators)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-110, 20)

    # Panel B: Coefficient of variation
    ax = axes2[1]
    for method in methods:
        cvs = []
        for rate in missing_rates:
            cv_per_est = []
            for est in estimators:
                vals = all_results[method][est][rate]
                if len(vals) > 5:
                    cv_per_est.append(np.std(vals) / np.mean(vals) * 100)
            if cv_per_est:
                cvs.append(np.mean(cv_per_est))
            else:
                cvs.append(np.nan)

        cvs = np.array(cvs)
        valid = np.isfinite(cvs)
        ax.plot(np.array(rates_pct)[valid], cvs[valid],
                's-', color=method_colors[method],
                label=method_labels[method], markersize=6, linewidth=2)

    ax.set_xlabel('Missing data rate (%)', fontsize=12)
    ax.set_ylabel('Coefficient of variation (%)', fontsize=12)
    ax.set_title('Precision (averaged across 4 estimators)', fontsize=13)
    ax.legend(fontsize=10)

    fig2.suptitle(
        r'Missing data handling strategies for $\theta$ estimation'
        f'\n(Standard neutral model, $\\theta$ = {theta_true:.0f}, '
        f'n = {n_hap}, {n_reps} replicates)',
        fontsize=13, y=1.03)

    outpath2 = 'debug/verify_missing_data_bias_variance.png'
    fig2.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {outpath2}")


if __name__ == "__main__":
    main()
    print()
    make_figure()
