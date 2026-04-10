#!/usr/bin/env python
"""
An example of demographic inference from LD statistics using pg_gpu + moments.
This script simulates replicate 1Mb regions under a three population model with
recent admixture using msprime, then computes LD statistics with pg_gpu,
then fits the demographic model using the `Demes` inference engine from moments.LD.

Usage:
    pixi run -e moments python examples/moments_3pop_integration_demo.py

Requires the 'moments' pixi environment (pixi install -e moments).
"""

import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import msprime
import demes
import demesdraw
import moments
import moments.LD
import logging
import sys
from collections import OrderedDict
from math import ceil

import pg_gpu.moments_ld

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
for handler in logger.handlers:
    handler.setFormatter(formatter)


# ── Settings ───────────────────────────────────────────────────

NUM_REPS = 200
SEQ_LEN = 1_000_000
MUT_RATE = 1.5e-8
REC_RATE = 1.5e-8
SAMPLE_SIZE = 10  # diploids per population
SAMPLE_POPS = ["deme0", "deme1", "deme2"]
DATA_DIR = "examples/data/moments_3pop_integration_demo"
R_BINS = np.array([0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4])
OVERWRITE = False  # don't use the cache and overwrite intermediate output


# ── Model definition and simulation ────────────────────────────

def simulate_data(demographic_model, vcf_dir):
    """
    Simulate replicate regions with msprime and write VCFs, alongside
    samples-to-deme map and flat recombination map. Return paths to
    VCF files, recombination map, and samples file.
    """
    os.makedirs(vcf_dir, exist_ok=True)
    vcf_paths = [os.path.join(vcf_dir, f"rep_{i}.vcf.gz") for i in range(NUM_REPS)]
    map_path = os.path.join(vcf_dir, "flat_map.txt")
    samples_path = os.path.join(vcf_dir, "samples.txt")

    tree_sequences = msprime.sim_ancestry(
        {pop: SAMPLE_SIZE for pop in SAMPLE_POPS},
        demography=msprime.Demography.from_demes(demographic_model),
        sequence_length=SEQ_LEN,
        recombination_rate=REC_RATE,
        num_replicates=NUM_REPS,
        random_seed=1024,
    )
    for i, (ts, vcf) in enumerate(zip(tree_sequences, vcf_paths)):
        ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=i * 10 + 1)
        population_names = [pop.metadata["name"] for pop in ts.populations()]
        individual_populations = [population_names[ind.population] for ind in ts.individuals()]
        individual_names = [f"{pop}{i}" for i, pop in enumerate(individual_populations)]
        ts.write_vcf(gzip.open(vcf, "wt"), individual_names=individual_names, position_transform="legacy")

    # write samples file
    with open(samples_path, "w") as handle:
        handle.write("sample\tpop\n")
        for name, pop in zip(individual_names, individual_populations):
            handle.write(f"{name}\t{pop}\n")

    # write flat recombination map
    with open(map_path, "w") as handle:
        handle.write("pos\tMap(cM)\n")
        handle.write("0\t0\n")
        handle.write(f"{SEQ_LEN}\t{REC_RATE * SEQ_LEN * 100}\n")

    return vcf_paths, map_path, samples_path


generative_model = """
# YAML of the generative model to simulate and subsequently fit.
# See the demes docs for details on the specification:
# https://popsim-consortium.github.io/demes-docs/latest/introduction.html
time_units: generations
generation_time: 1
demes:
    - name: anc
      epochs:
        - {end_time: 15000.0, start_size: 10000.0}
    - name: trunk
      ancestors: [anc]
      epochs:
        - {end_time: 5000.0, start_size: 20000.0}
    - name: deme0
      ancestors: [anc]
      epochs:
        - {end_time: 0, start_size: 5000.0, end_size: 50000.0}
    - name: deme1
      ancestors: [trunk]
      epochs:
        - {end_time: 0, start_size: 20000.0}
    - name: deme2
      ancestors: [trunk]
      epochs:
        - {end_time: 0, start_size: 20000.0}
migrations:
    - {source: deme0, dest: deme2, rate: 0.0001}
"""


moments_optimization_options = """
# YAML mapping parameters to slots in the demes model.
# See the moments.Demes documentation for more details:
# https://momentsld.github.io/moments/extensions/demes.html
parameters:
    - name: N_anc
      description: Size of ancestral population
      values:
        - demes:
            anc:
                epochs:
                    0: start_size
    - name: N_trunk
      description: Size of ancestor of deme1 and deme2
      values:
        - demes:
            trunk:
                epochs:
                    0: start_size
    - name: N_deme0_start
      description: Historical size of deme0
      values:
        - demes:
            deme0:
                epochs:
                    0: start_size
    - name: N_deme0_end
      description: Contemporary size of deme0
      values:
        - demes:
            deme0:
                epochs:
                    0: end_size
    - name: N_deme1
      description: Size of deme1
      values:
        - demes:
            deme1:
                epochs:
                    0: start_size
    - name: N_deme2
      description: Size of deme2
      values:
        - demes:
            deme2:
                epochs:
                    0: start_size
    - name: T_trunk
      description: Time ago where deme0 and trunk merge
      values:
        - demes:
            anc:
                epochs:
                    0: end_time
    - name: T_recent
      description: Time ago where deme1 and deme2 merge
      values:
        - demes:
            trunk:
                epochs:
                    0: end_time
    - name: M_deme0_deme2
      description: Migration from deme0 to deme2 forwards in time
      values:
        - migrations:
            0: rate
constraints:
    - params: [T_trunk, T_recent]
      constraint: greater_than
"""


# ── Usage ──────────────────────────────────────────────────────

if __name__ == "__main__":

    logger.info(f"Output will be saved to {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    true_yaml_path = os.path.join(DATA_DIR, "true_model.yaml")
    logger.info(f"Writing generative model to {true_yaml_path}:\n{generative_model.strip()}")
    with open(true_yaml_path, "w") as handle:
        handle.write(generative_model)

    true_model_path = os.path.join(DATA_DIR, "true_model.png")
    logger.info(f"Plotting generative model at {true_model_path}")
    fig, axs = plt.subplots(1, figsize=(4, 4), constrained_layout=True)
    demesdraw.tubes(demes.load(true_yaml_path))
    plt.savefig(true_model_path)
    plt.close(fig)


    logger.info("Simulating data and calculating statistics")
    cache_path = os.path.join(DATA_DIR, "ld_stats_cache_pg_gpu.pkl")
    vcf_path = os.path.join(DATA_DIR, "data")
    if not os.path.exists(cache_path) or OVERWRITE:

        logger.info("Simulating chunks of sequence")
        vcf_paths, map_path, samples_path = simulate_data(demes.load(true_yaml_path), vcf_path)

        logger.info("Calculating statistics across chunks with pg_gpu")
        ld_stats = {
            # NB: this is where pg_gpu fits into the workflow, by providing a
            # *drop-in replacement* for the moments.LD function of the same name
            vcf: pg_gpu.moments_ld.compute_ld_statistics(
                vcf, rec_map_file=map_path, pop_file=samples_path,
                pops=SAMPLE_POPS, r_bins=R_BINS, report=False,
            ) for vcf in vcf_paths
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(ld_stats, handle)
    with open(cache_path, "rb") as handle:
        ld_stats = pickle.load(handle)


    logger.info("Bootstrapping and averaging chunks")
    bootstrap_mean_variance = moments.LD.Parsing.bootstrap_data(ld_stats)
    bootstrap_means, bootstrap_varcovs, bootstrap_stats = moments.LD.Inference.remove_normalized_data(
        bootstrap_mean_variance["means"],
        bootstrap_mean_variance["varcovs"],
        normalization=0,
        num_pops=len(SAMPLE_POPS),
        statistics=bootstrap_mean_variance["stats"],
    )


    logger.info("Optimizing model")
    fitted_yaml_path = os.path.join(DATA_DIR, "fitted_model.yaml")
    optim_options_path = os.path.join(DATA_DIR, "options.yaml")
    if not os.path.exists(fitted_yaml_path) or OVERWRITE:

        logger.info(f"Writing optimization options to {optim_options_path}")
        with open(optim_options_path, "w") as handle:
            handle.write(moments_optimization_options)

        logger.info(f"Fitting and writing the optimized model to {fitted_yaml_path}")
        fitted_parameters = moments.Demes.Inference.optimize_LD(
            true_yaml_path,
            optim_options_path,
            bootstrap_means,
            bootstrap_varcovs,
            pop_ids=SAMPLE_POPS,
            perturb=0,
            rs=R_BINS,
            normalization=SAMPLE_POPS[0],
            method="powell",
            output=fitted_yaml_path,
            # DEBUG:
            verbose=True,
        )


    logger.info("Calculating standard errors via Godambe information matrix")
    summary_path = os.path.join(DATA_DIR, "fitted_parameters.txt")
    if not os.path.exists(summary_path) or OVERWRITE:
        bootstraps = moments.LD.Parsing.get_bootstrap_sets(ld_stats)
        std_errors = moments.Demes.Inference.uncerts_LD(
            fitted_yaml_path,
            optim_options_path,
            bootstrap_means,
            bootstrap_varcovs,
            bootstraps=bootstraps,
            statistics=bootstrap_stats,
            pop_ids=SAMPLE_POPS,
            rs=R_BINS,
            normalization=SAMPLE_POPS[0],
            method="GIM",
            output=summary_path,
        )
    with open(summary_path) as handle:
        logger.info(f"Wrote estimates to {summary_path}:\n{handle.read().strip()}")


    plot_path = os.path.join(DATA_DIR, "fitted_vs_observed.png")
    logger.info(f"Plotting fitted vs observed statistics at {plot_path}")
    fitted = moments.Demes.Inference.compute_bin_stats(
        demes.load(fitted_yaml_path),
        SAMPLE_POPS,
        rs=R_BINS,
    )
    fitted = moments.LD.Inference.sigmaD2(fitted)
    fitted = moments.LD.Inference.remove_normalized_lds(fitted, normalization=0)
    ld_stat_names, het_stat_names = bootstrap_stats
    cols = 4
    rows = ceil(len(ld_stat_names) / cols)
    fig = moments.LD.Plotting.plot_ld_curves_comp(
        fitted,
        bootstrap_means,
        bootstrap_varcovs,
        stats_to_plot=[[x] for x in ld_stat_names],
        statistics=bootstrap_stats,
        rs=R_BINS, binned_data=True, plot_vcs=True,
        rows=rows, cols=cols, fig_size=(cols*2.25, rows*2),
    )
    plt.savefig(plot_path)


