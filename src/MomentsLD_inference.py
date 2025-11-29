"""
MomentsLD demographic parameter inference using linkage disequilibrium.

This module provides functions to:
1. Load and aggregate LD statistics from multiple simulation windows
2. Compute expected LD under demographic models using Demes graphs
3. Optimize demographic parameters via composite Gaussian likelihood
4. Generate comparison plots between empirical and theoretical LD
"""

import json
import logging
import pickle
import importlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import moments
import nlopt
import numdifftools as nd

# Default settings
DEFAULT_R_BINS = np.array(
    [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
)
JITTER = 1e-12  # numerical stability for matrix inversion
CONVERGENCE_TOL = 1e-8


# =============================================================================
# Data Loading and Preparation
# =============================================================================


def load_sampled_params(sim_dir, required=True):
    """Load sampled parameters from simulation directory."""
    pkl_file = sim_dir / "sampled_params.pkl"
    if not pkl_file.exists():
        if required:
            raise FileNotFoundError(f"sampled_params.pkl missing in {pkl_file.parent}")
        logging.info(
            "sampled_params.pkl not found in %s; continuing without true parameters",
            pkl_file.parent,
        )
        return None

    with pkl_file.open("rb") as f:
        return pickle.load(f)


def load_config(config_path):
    """Load configuration from JSON file."""
    with config_path.open("r") as f:
        return json.load(f)


def aggregate_ld_statistics(ld_root):
    """
    Aggregate LD statistics from multiple windows into means and covariances.

    Args:
        ld_root: Path to MomentsLD directory containing LD_stats/ subdirectory

    Returns:
        Dictionary with 'means' and 'varcovs' for empirical LD statistics

    Creates:
        - means.varcovs.pkl: Aggregated statistics
        - bootstrap_sets.pkl: Bootstrap data for variance estimation
    """
    means_file = ld_root / "means.varcovs.pkl"
    boots_file = ld_root / "bootstrap_sets.pkl"

    # Return cached results if available
    if means_file.exists() and boots_file.exists():
        with means_file.open("rb") as f:
            return pickle.load(f)

    # Load individual LD statistics files
    ld_stats_dir = ld_root / "LD_stats"
    ld_files = list(ld_stats_dir.glob("LD_stats_window_*.pkl"))

    if not ld_files:
        raise RuntimeError(f"No LD statistics files found in {ld_stats_dir}")

    ld_stats = {}
    for pkl_file in ld_files:
        window_id = int(pkl_file.stem.split("_")[-1])
        with pkl_file.open("rb") as f:
            ld_stats[window_id] = pickle.load(f)

    # Aggregate using moments.LD
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)
    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(ld_stats)

    # Save results
    with means_file.open("wb") as f:
        pickle.dump(mv, f)
    with boots_file.open("wb") as f:
        pickle.dump(bootstrap_sets, f)

    logging.info(f"Aggregated {len(ld_stats)} LD windows → {ld_root}")
    return mv


# =============================================================================
# Theoretical LD Computation
# =============================================================================


def compute_theoretical_ld(params, param_names, demographic_model, r_bins, populations):
    """
    Compute expected LD statistics under a demographic model.

    Args:
        params: Parameter values in log10 space
        param_names: List of parameter names
        demographic_model: Function that creates Demes graph from parameters
        r_bins: Recombination rate bin edges
        populations: List of population names to sample from

    Returns:
        moments.LD.LDstats: Expected σD² statistics
    """
    # Convert to absolute scale and create parameter dictionary
    param_values = 10 ** np.array(params)
    param_dict = dict(zip(param_names, param_values))

    # Create demographic graph
    graph = demographic_model(param_dict)

    # Find reference population size for scaling
    ref_size = (
        param_dict.get("N0")
        or param_dict.get("N_ANC")
        or next((v for k, v in param_dict.items() if k.startswith("N")), 1.0)
    )

    # Compute LD using Simpson-like integration across r-bins
    rho_edges = 4.0 * ref_size * np.array(r_bins)
    ld_edges = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_edges)

    rho_mids = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    ld_mids = moments.Demes.LD(graph, sampled_demes=populations, rho=rho_mids)

    # Simpson's rule weighted average
    ld_bins = [
        (ld_edges[i] + ld_edges[i + 1] + 4 * ld_mids[i]) / 6.0
        for i in range(len(rho_mids))
    ]
    ld_bins.append(ld_edges[-1])  # Add final edge value

    # Convert to σD² format
    ld_stats = moments.LD.LDstats(
        ld_bins, num_pops=ld_edges.num_pops, pop_ids=ld_edges.pop_ids
    )
    return moments.LD.Inference.sigmaD2(ld_stats)


def prepare_data_for_comparison(theoretical_ld, empirical_data, normalization=0):
    """
    Prepare theoretical and empirical data for likelihood comparison.

    Args:
        theoretical_ld: Output from compute_theoretical_ld()
        empirical_data: Dictionary with 'means' and 'varcovs' keys
        normalization: LD normalization scheme (0 = no normalization)

    Returns:
        Tuple of (theory_arrays, empirical_means, empirical_covariances)
    """
    # Process theoretical predictions
    theory_processed = moments.LD.LDstats(
        theoretical_ld[:],
        num_pops=theoretical_ld.num_pops,
        pop_ids=theoretical_ld.pop_ids,
    )
    theory_processed = moments.LD.Inference.remove_normalized_lds(
        theory_processed, normalization=normalization
    )
    theory_arrays = [
        np.array(pred) for pred in theory_processed[:-1]
    ]  # Remove heterozygosity

    # Process empirical data
    emp_means = [np.array(x) for x in empirical_data["means"]]
    emp_covars = [np.array(x) for x in empirical_data["varcovs"]]

    # Remove normalized statistics
    emp_means, emp_covars = moments.LD.Inference.remove_normalized_data(
        emp_means,
        emp_covars,
        normalization=normalization,
        num_pops=theoretical_ld.num_pops,
    )

    # Remove heterozygosity statistics
    emp_means = emp_means[:-1]
    emp_covars = emp_covars[:-1]

    return theory_arrays, emp_means, emp_covars
    """
    Writes: <ld_root>/empirical_vs_theoretical_comparison.pdf
    """
    pdf = ld_root / "empirical_vs_theoretical_comparison.pdf"
    if pdf.exists():
        return

    demo_mod = importlib.import_module("simulation")
    if config["demographic_model"] == "drosophila_three_epoch":
        demo_function = getattr(demo_module, "drosophila_three_epoch")
    else:
        demo_function = getattr(demo_module, config["demographic_model"] + "_model")
    graph = demo_func(sampled_params)

    # pops to compare come from the config’s num_samples keys
    sampled_demes = list(cfg["num_samples"].keys())

    y = moments.Demes.LD(
        graph, sampled_demes=sampled_demes, rho=4 * sampled_params["N0"] * r_vec
    )
    # Simpson-like binning
    y = moments.LD.LDstats(
        [(yl + yr) / 2 for yl, yr in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    if cfg["demographic_model"] == "bottleneck":
        stats_to_plot = [["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]]
        labels = [[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]]
        rows = 2
    else:
        stats_to_plot = [
            ["DD_0_0"],
            ["DD_0_1"],
            ["DD_1_1"],
            ["Dz_0_0_0"],
            ["Dz_0_1_1"],
            ["Dz_1_1_1"],
            ["pi2_0_0_1_1"],
            ["pi2_0_1_0_1"],
            ["pi2_1_1_1_1"],
        ]
        labels = [
            [r"$D_0^2$"],
            [r"$D_0 D_1$"],
            [r"$D_1^2$"],
            [r"$Dz_{0,0,0}$"],
            [r"$Dz_{0,1,1}$"],
            [r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$"],
            [r"$\pi_{2;0,1,0,1}$"],
            [r"$\pi_{2;1,1,1,1}$"],
        ]
        rows = 3

    try:
        # Ensure r_vec length matches the data dimensions
        empirical_means = mv["means"][:-1]
        empirical_varcovs = mv["varcovs"][:-1]

        # If there's a dimension mismatch, truncate r_vec to match the data
        if hasattr(empirical_means, "__len__") and len(r_vec) != len(empirical_means):
            logging.warning(
                "r_vec length (%d) doesn't match empirical data length (%d), truncating r_vec",
                len(r_vec),
                len(empirical_means),
            )
            r_vec = r_vec[: len(empirical_means)]

        fig = moments.LD.Plotting.plot_ld_curves_comp(
            y,
            empirical_means,
            empirical_varcovs,
            rs=r_vec,
            stats_to_plot=stats_to_plot,
            labels=labels,
            rows=rows,
            plot_vcs=True,
            show=False,
            fig_size=(6, 4),
        )
        fig.savefig(pdf, dpi=300)
        plt.close(fig)
        logging.info("Written comparison PDF → %s", pdf)
    except (IndexError, ValueError) as e:
        logging.warning("Plotting failed due to data structure mismatch: %s", e)
        logging.warning(
            "Skipping PDF generation - this may indicate insufficient LD statistics data"
        )
        # Create empty file to satisfy Snakemake dependencies
        pdf.touch()
    except Exception as e:
        logging.error("Unexpected error in plotting: %s", e)
        # Create empty file to satisfy Snakemake dependencies
        pdf.touch()
        raise


# ─── Core LD Analysis Functions ─────────────────────────────────────────────

# =============================================================================
# Likelihood Computation
# =============================================================================


def compute_composite_likelihood(
    empirical_means, empirical_covariances, theoretical_predictions
):
    """
    Compute composite Gaussian log-likelihood.

    Args:
        empirical_means: List of empirical mean vectors for each r-bin
        empirical_covariances: List of covariance matrices for each r-bin
        theoretical_predictions: List of model prediction vectors for each r-bin

    Returns:
        Log-likelihood value
    """
    total_loglik = 0.0

    for obs, cov, pred in zip(
        empirical_means, empirical_covariances, theoretical_predictions
    ):
        if len(obs) == 0:
            continue

        residual = obs - pred
        cov_matrix = np.array(cov)

        if cov_matrix.ndim == 2 and cov_matrix.size > 1:
            # Add small jitter for numerical stability
            cov_matrix += np.eye(cov_matrix.shape[0]) * JITTER

            try:
                cov_inv = np.linalg.inv(cov_matrix)
                total_loglik -= 0.5 * float(residual @ cov_inv @ residual)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                cov_inv = np.linalg.pinv(cov_matrix)
                total_loglik -= 0.5 * float(residual @ cov_inv @ residual)
        else:
            # Scalar variance case
            total_loglik -= 0.5 * float(residual @ residual)

    return total_loglik


def objective_function(
    log_params,
    param_names,
    demographic_model,
    r_bins,
    empirical_data,
    populations,
    normalization=0,
):
    """
    Objective function for optimization: computes log-likelihood for given parameters.

    Args:
        log_params: Parameters in log10 space
        param_names: List of parameter names
        demographic_model: Function creating Demes graph
        r_bins: Recombination rate bins
        empirical_data: Dictionary with empirical LD statistics
        populations: Population names to sample
        normalization: LD normalization scheme

    Returns:
        Composite log-likelihood
    """
    try:
        # Compute theoretical LD
        theoretical_ld = compute_theoretical_ld(
            log_params, param_names, demographic_model, r_bins, populations
        )

        # Prepare data for comparison
        theory_arrays, emp_means, emp_covars = prepare_data_for_comparison(
            theoretical_ld, empirical_data, normalization
        )

        # Compute likelihood
        return compute_composite_likelihood(emp_means, emp_covars, theory_arrays)

    except Exception as e:
        logging.warning(f"Error in objective function: {e}")
        return -np.inf


# =============================================================================
# Parameter Optimization
# =============================================================================


def handle_fixed_parameters(config, sampled_params, param_names):
    """
    Parse configuration to determine which parameters should be fixed.

    Args:
        config: Configuration dictionary
        sampled_params: Dictionary of sampled parameter values (optional)
        param_names: List of all parameter names

    Returns:
        List where each element is either None (free parameter) or a float (fixed value)
    """
    fixed_values = [None] * len(param_names)
    fixed_config = config.get("fixed_parameters", {})

    for i, param_name in enumerate(param_names):
        if param_name not in fixed_config:
            continue

        fixed_spec = fixed_config[param_name]

        if isinstance(fixed_spec, (int, float)):
            fixed_values[i] = float(fixed_spec)
        elif isinstance(fixed_spec, str) and fixed_spec.lower() in ["sampled", "true"]:
            if sampled_params is None or param_name not in sampled_params:
                logging.warning(
                    "Config requested %s be fixed to '%s', but sampled_params are unavailable. "
                    "Leaving this parameter free instead.",
                    param_name,
                    fixed_spec,
                )
                continue  # leave fixed_values[i] = None → parameter remains free
            fixed_values[i] = float(sampled_params[param_name])
        else:
            raise ValueError(
                f"Invalid fixed parameter specification for {param_name}: {fixed_spec}"
            )

    return fixed_values


def create_free_parameter_vectors(
    full_params, bounds_lower, bounds_upper, fixed_values
):
    """
    Extract free parameters and their bounds from full parameter specifications.

    Returns:
        Tuple of (free_params, free_lower_bounds, free_upper_bounds, expand_function)
    """
    if fixed_values is None or all(v is None for v in fixed_values):
        # All parameters are free
        return full_params, bounds_lower, bounds_upper, lambda x: x

    # Extract only free parameters
    free_indices = [i for i, fixed_val in enumerate(fixed_values) if fixed_val is None]
    free_params = full_params[free_indices]
    free_lower = bounds_lower[free_indices]
    free_upper = bounds_upper[free_indices]

    def expand_to_full(free_values):
        """Reconstruct full parameter vector from free parameters."""
        full = np.zeros(len(fixed_values))
        free_idx = 0
        for i, fixed_val in enumerate(fixed_values):
            if fixed_val is None:
                full[i] = free_values[free_idx]
                free_idx += 1
            else:
                full[i] = fixed_val
        return full

    return free_params, free_lower, free_upper, expand_to_full


def optimize_parameters(
    start_values,
    lower_bounds,
    upper_bounds,
    param_names,
    demographic_model,
    r_bins,
    empirical_data,
    populations,
    normalization=0,
    tolerance=CONVERGENCE_TOL,
    verbose=True,
    fixed_values=None,
):
    """
    Optimize demographic parameters using L-BFGS algorithm.

    Args:
        start_values: Initial parameter values (absolute scale)
        lower_bounds: Parameter lower bounds (absolute scale)
        upper_bounds: Parameter upper bounds (absolute scale)
        param_names: List of parameter names
        demographic_model: Function creating Demes graph from parameters
        r_bins: Recombination rate bins
        empirical_data: Dictionary with empirical LD statistics
        populations: Population names to sample
        normalization: LD normalization scheme
        tolerance: Convergence tolerance
        verbose: Print optimization progress
        fixed_values: List of fixed parameter values (None = free)

    Returns:
        Tuple of (optimal_parameters, max_log_likelihood, status_code)
    """
    # Convert to log10 space for optimization
    start_log10 = np.log10(np.maximum(start_values, 1e-300))
    bounds_lower_log10 = np.log10(np.maximum(lower_bounds, 1e-300))
    bounds_upper_log10 = np.log10(upper_bounds)

    # Handle fixed parameters
    fixed_log10 = (
        None
        if fixed_values is None
        else [None if v is None else np.log10(max(v, 1e-300)) for v in fixed_values]
    )

    free_start, free_lower, free_upper, expand_function = create_free_parameter_vectors(
        start_log10, bounds_lower_log10, bounds_upper_log10, fixed_log10
    )

    # Track best solution found
    best_likelihood = -np.inf
    best_params = None

    # Create objective function for optimization
    def nlopt_objective(free_params, gradient):
        nonlocal best_likelihood, best_params

        # Expand to full parameter set
        full_params_log10 = expand_function(free_params)

        # Compute likelihood
        likelihood = objective_function(
            full_params_log10,
            param_names,
            demographic_model,
            r_bins,
            empirical_data,
            populations,
            normalization,
        )

        # Track best solution
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_params = np.array(free_params)

        # Compute numerical gradient if requested
        if gradient.size > 0:
            gradient_func = nd.Gradient(
                lambda p: objective_function(
                    expand_function(p),
                    param_names,
                    demographic_model,
                    r_bins,
                    empirical_data,
                    populations,
                    normalization,
                ),
                step=1e-4,
            )
            gradient[:] = gradient_func(free_params)

        if verbose:
            param_values = 10**full_params_log10
            param_str = ", ".join(
                [f"{name}={val:.3g}" for name, val in zip(param_names, param_values)]
            )
            print(f"LL = {likelihood:.6f} | {param_str}")

        return likelihood

    # Set up and run optimization
    optimizer = nlopt.opt(nlopt.LD_LBFGS, len(free_start))
    optimizer.set_lower_bounds(free_lower)
    optimizer.set_upper_bounds(free_upper)
    optimizer.set_max_objective(nlopt_objective)
    optimizer.set_ftol_rel(tolerance)

    try:
        optimal_free = optimizer.optimize(free_start)
        status = optimizer.last_optimize_result()
        max_likelihood = optimizer.last_optimum_value()
    except Exception as e:
        logging.warning(f"Optimization failed: {e}. Using best solution found.")
        if best_params is not None:
            optimal_free = best_params
            max_likelihood = best_likelihood
            status = -1
        else:
            optimal_free = free_start
            max_likelihood = best_likelihood
            status = -1

    # Use tracked best if better
    if best_params is not None and best_likelihood > max_likelihood:
        optimal_free = best_params
        max_likelihood = best_likelihood

    # Convert back to absolute scale
    optimal_full_log10 = expand_function(optimal_free)
    optimal_params = 10**optimal_full_log10

    return optimal_params, max_likelihood, status


# =============================================================================
# Visualization
# =============================================================================


def create_comparison_plot(config, sampled_params, empirical_data, r_bins, output_path):
    """
    Create comparison plot between empirical and theoretical LD curves.

    Args:
        config: Configuration dictionary with demographic model info
        sampled_params: Dictionary of sampled parameter values
        empirical_data: Dictionary with empirical LD statistics
        r_bins: Recombination rate bins
        output_path: Path where to save the PDF
    """
    if output_path.exists():
        return

    try:
        # Load demographic model
        demo_module = importlib.import_module("simulation")
        if config["demographic_model"] == "drosophila_three_epoch":
            demo_function = getattr(demo_module, "drosophila_three_epoch")
        else:
            demo_function = getattr(demo_module, config["demographic_model"] + "_model")

        # Get populations to sample
        populations = list(config["num_samples"].keys())

        # Compute theoretical LD under sampled parameters
        log_params = [
            np.log10(sampled_params[name]) for name in config["priors"].keys()
        ]
        param_names = list(config["priors"].keys())
        theoretical_ld = compute_theoretical_ld(
            log_params, param_names, demo_function, r_bins, populations
        )

        # Extract empirical data (excluding heterozygosity like in the docs example)
        emp_means = empirical_data["means"][:-1]
        emp_covars = empirical_data["varcovs"][:-1]

        # Create plot - handle dimension mismatch gracefully
        r_vec_plot = r_bins
        theory_for_plot = theoretical_ld

        # Check if we need to truncate due to dimension mismatch
        if len(emp_means) < len(r_bins):
            r_vec_plot = r_bins[: len(emp_means)]
            logging.warning(
                f"Truncating r_bins from {len(r_bins)} to {len(emp_means)} for plotting"
            )

            # Recompute theoretical LD with truncated r_bins to match dimensions
            theory_for_plot = compute_theoretical_ld(
                [np.log10(sampled_params[name]) for name in config["priors"].keys()],
                list(config["priors"].keys()),
                demo_function,
                r_vec_plot,
                populations,
            )

        # Define statistics to plot based on model type
        if config["demographic_model"] == "bottleneck":
            stats_to_plot = [["DD_0_0"], ["Dz_0_0_0"], ["pi2_0_0_0_0"]]
            labels = [[r"$D_0^2$"], [r"$Dz_{0,0,0}$"], [r"$\pi_{2;0,0,0,0}$"]]
            rows = 2
        else:
            stats_to_plot = [
                ["DD_0_0"],
                ["DD_0_1"],
                ["DD_1_1"],
                ["Dz_0_0_0"],
                ["Dz_0_1_1"],
                ["Dz_1_1_1"],
                ["pi2_0_0_1_1"],
                ["pi2_0_1_0_1"],
                ["pi2_1_1_1_1"],
            ]
            labels = [
                [r"$D_0^2$"],
                [r"$D_0 D_1$"],
                [r"$D_1^2$"],
                [r"$Dz_{0,0,0}$"],
                [r"$Dz_{0,1,1}$"],
                [r"$Dz_{1,1,1}$"],
                [r"$\pi_{2;0,0,1,1}$"],
                [r"$\pi_{2;0,1,0,1}$"],
                [r"$\pi_{2;1,1,1,1}$"],
            ]
            rows = 3

        fig = moments.LD.Plotting.plot_ld_curves_comp(
            theory_for_plot,
            emp_means,
            emp_covars,
            rs=r_vec_plot,
            stats_to_plot=stats_to_plot,
            labels=labels,
            rows=rows,
            plot_vcs=True,
            show=False,
            fig_size=(6, 4),
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        logging.info(f"Comparison plot saved → {output_path}")

    except Exception as e:
        logging.warning(f"Plot generation failed: {e}")
        logging.warning("Creating empty file to satisfy dependencies")
        output_path.touch()


# =============================================================================
# Main Interface
# =============================================================================


def run_momentsld_inference(
    config, empirical_data, results_dir, r_bins, sampled_params=None
):
    """
    Main function to run MomentsLD demographic parameter inference.

    This function:
    1. Sets up the optimization problem from configuration
    2. Handles any fixed parameters
    3. Runs L-BFGS optimization to find best-fit parameters
    4. Saves results and creates comparison plots

    Args:
        config: Configuration dictionary with priors, demographic model, etc.
        empirical_data: Dictionary with aggregated LD statistics
        results_dir: Directory where results will be saved
        r_bins: Recombination rate bin edges
        sampled_params: Optional true parameter values for fixing parameters
    """
    results_file = results_dir / "best_fit.pkl"
    if results_file.exists():
        logging.info("Results already exist - skipping optimization")
        return

    # Load demographic model function
    demo_module = importlib.import_module("simulation")
    if config["demographic_model"] == "drosophila_three_epoch":
        demo_function = getattr(demo_module, "drosophila_three_epoch")
    else:
        demo_function = getattr(demo_module, config["demographic_model"] + "_model")

    # Extract parameter setup from configuration
    priors = config["priors"]
    param_names = list(priors.keys())
    lower_bounds = np.array([prior[0] for prior in priors.values()])
    upper_bounds = np.array([prior[1] for prior in priors.values()])
    start_values = np.sqrt(lower_bounds * upper_bounds)  # geometric mean starting point

    # Get populations to sample
    populations = list(config.get("num_samples", {}).keys())
    if not populations:
        raise ValueError(
            "Configuration must specify 'num_samples' to determine populations"
        )

    # Parse optimization settings
    normalization = config.get("ld_normalization", 0)
    tolerance = config.get("ld_rtol", CONVERGENCE_TOL)
    verbose = config.get("ld_verbose", True)

    # Handle fixed parameters
    fixed_values = handle_fixed_parameters(config, sampled_params, param_names)

    # Run optimization
    logging.info("Starting MomentsLD parameter optimization...")
    optimal_params, max_likelihood, status = optimize_parameters(
        start_values=start_values,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        param_names=param_names,
        demographic_model=demo_function,
        r_bins=r_bins,
        empirical_data=empirical_data,
        populations=populations,
        normalization=normalization,
        tolerance=tolerance,
        verbose=verbose,
        fixed_values=fixed_values,
    )

    # Save results
    results = {
        "best_params": dict(zip(param_names, optimal_params)),
        "best_lls": max_likelihood,
        "status": status,
    }

    with results_file.open("wb") as f:
        pickle.dump(results, f)

    logging.info(
        f"Optimization completed: LL = {max_likelihood:.6f}, status = {status}"
    )
    logging.info(f"Results saved → {results_file}")

    # Create comparison plot if sampled parameters are available
    if sampled_params is not None:
        plot_file = results_dir / "empirical_vs_theoretical_comparison.pdf"
        create_comparison_plot(
            config, sampled_params, empirical_data, r_bins, plot_file
        )

# =============================================================================
# High-level helper for Snakemake: run MomentsLD from LD_stats directory
# =============================================================================

from typing import Optional, Union

def run_momentsld_from_ld_dir(
    ld_dir: Union[str, Path],
    config_path: Union[str, Path],
    results_dir: Union[str, Path],
    sim_dir: Optional[Union[str, Path]] = None,
    r_per_bp: Optional[float] = None,
    num_rbins: Optional[int] = None,
):
    """
    High-level convenience wrapper to run MomentsLD inference directly
    from a directory of LD_stats_window_*.pkl files.

    Parameters
    ----------
    ld_dir
        Directory that contains files named LD_stats_window_*.pkl
        (e.g. GPU/LD_stats).
    config_path
        Path to JSON config describing priors, demographic_model,
        and num_samples.
    results_dir
        Directory where best_fit.pkl and comparison PDF will be written.
    sim_dir
        Optional directory containing sampled_params.pkl (true parameters)
        from the simulations. If present, used for fixing parameters and
        plotting.
    r_per_bp
        Per-base recombination rate. If provided with num_rbins, we build
        an r-bin array [0, ..., r_per_bp] with num_rbins bins.
        If None, we fall back to DEFAULT_R_BINS.
    num_rbins
        Number of r-bins. Only used if r_per_bp is not None.

    Side Effects
    ------------
    - Creates results_dir if it doesn't exist.
    - Writes:
        - results_dir / "best_fit.pkl"
        - results_dir / "empirical_vs_theoretical_comparison.pdf" (if possible)
    """
    ld_dir = Path(ld_dir)
    config_path = Path(config_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config ----
    config = load_config(config_path)

    # ---- Load & aggregate LD statistics from LD_stats_window_*.pkl ----
    ld_files = sorted(ld_dir.glob("LD_stats_window_*.pkl"))
    if not ld_files:
        raise RuntimeError(f"No LD_stats_window_*.pkl files found in {ld_dir}")

    ld_stats = {}
    for pkl_file in ld_files:
        # window index is the last underscore-separated token
        try:
            window_id = int(pkl_file.stem.split("_")[-1])
        except ValueError:
            # Fall back to sequential indexing if parsing fails
            window_id = len(ld_stats)
        with pkl_file.open("rb") as f:
            ld_stats[window_id] = pickle.load(f)

    # moments.LD expects a dict: {window_id: LDstats}
    mv = moments.LD.Parsing.bootstrap_data(ld_stats)

    empirical_data = {
        "means": mv["means"],
        "varcovs": mv["varcovs"],
    }

    # ---- Build r-bin edges ----
    if r_per_bp is None or num_rbins is None:
        # Use the default r-bins if user didn't specify binning.
        r_bins = DEFAULT_R_BINS
        logging.info(
            "Using DEFAULT_R_BINS with %d bins for MomentsLD inference",
            len(DEFAULT_R_BINS) - 1,
        )
    else:
        # Simple linear binning from 0 to r_per_bp
        r_bins = np.linspace(0, float(r_per_bp), int(num_rbins) + 1)
        logging.info(
            "Using linear r-bins from 0 to %g with %d bins",
            r_per_bp,
            num_rbins,
        )

    # ---- Load sampled (true) parameters if available ----
    sampled_params = None
    if sim_dir is not None:
        sim_dir = Path(sim_dir)
        try:
            sampled_params = load_sampled_params(sim_dir, required=False)
        except FileNotFoundError:
            sampled_params = None

    # ---- Run core MomentsLD inference ----
    run_momentsld_inference(
        config=config,
        empirical_data=empirical_data,
        results_dir=results_dir,
        r_bins=r_bins,
        sampled_params=sampled_params,
    )

    # If sampled_params was provided, run_momentsld_inference will also
    # produce the comparison PDF via create_comparison_plot().
