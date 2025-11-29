############################
# IM MODEL LD WORKFLOW
############################

configfile: "config.yaml"

import numpy as np

# Pull parameters from config
IM = config["im_model"]

NUM_REPS   = IM["num_reps"]
L          = IM["L"]
MU         = IM["mu"]
R_PER_BP   = IM["r_per_bp"]
N_PER_POP  = IM["n_per_pop"]
NUM_RBINS  = IM["num_rbins"]

SIM_DIR           = IM["sim_dir"]
TRAD_LD_DIR       = IM["trad_ld_dir"]
TRAD_RESULTS_DIR  = IM["trad_results_dir"]
GPU_LD_DIR        = IM["gpu_ld_dir"]
GPU_RESULTS_DIR   = IM["gpu_results_dir"]

REPS = list(range(NUM_REPS))

# Final outputs that we care about
TRAD_MEANS       = f"{TRAD_RESULTS_DIR}/means.varcovs.traditional.{NUM_REPS}_reps.bp"
TRAD_BOOT        = f"{TRAD_RESULTS_DIR}/bootstrap_sets.traditional.{NUM_REPS}_reps.bp"
TRAD_PLOT        = f"{TRAD_RESULTS_DIR}/comparison_traditional.pdf"

GPU_MEANS        = f"{GPU_RESULTS_DIR}/means.varcovs.gpu.{NUM_REPS}_reps.bp"
GPU_BOOT         = f"{GPU_RESULTS_DIR}/bootstrap_sets.gpu.{NUM_REPS}_reps.bp"
GPU_PLOT         = f"{GPU_RESULTS_DIR}/comparison_gpu.pdf"


###################
# RULES
###################

rule all:
    input:
        TRAD_MEANS,
        TRAD_BOOT,
        TRAD_PLOT,
        GPU_MEANS,
        GPU_BOOT,
        GPU_PLOT


# 1) SIMULATION (once, but declares per-rep trees + flat_map.txt)
#    ALSO depends on config.yaml so changing config reruns this rule
rule simulate_im_model:
    input:
        cfg = "config.yaml"
    output:
        flat_map = f"{SIM_DIR}/flat_map.txt",
        trees    = expand(f"{SIM_DIR}/window_{{rep}}.trees", rep=REPS)
    shell:
        """
        python snakemake_scripts/simulate_im_model.py \
          --sim-dir {SIM_DIR} \
          --num-reps {NUM_REPS} \
          --L {L} \
          --mu {MU} \
          --r-per-bp {R_PER_BP} \
          --n-per-pop {N_PER_POP}
        """


# 2) FILTERED SITE SET PER REPLICATE
rule filter_sites:
    input:
        tree   = f"{SIM_DIR}/window_{{rep}}.trees",
        recmap = f"{SIM_DIR}/flat_map.txt"
    output:
        ts_filt      = f"{SIM_DIR}/window_{{rep}}.filtered.trees",
        vcf_filt_gz  = f"{SIM_DIR}/split_mig.filtered.{{rep}}.vcf.gz",
        kept_sites   = f"{SIM_DIR}/kept_sites_rep{{rep}}.txt",
        sha1         = f"{SIM_DIR}/kept_sites_rep{{rep}}.sha1",
    shell:
        """
        python snakemake_scripts/filter_sites.py \
          --sim-dir {SIM_DIR} \
          --rep {wildcards.rep}
        """


# 3) TRADITIONAL LD PER REPLICATE (parallelizable)
rule traditional_ld_rep:
    input:
        ts_filt     = f"{SIM_DIR}/window_{{rep}}.filtered.trees",
        vcf_filt_gz = f"{SIM_DIR}/split_mig.filtered.{{rep}}.vcf.gz",
        kept_sites  = f"{SIM_DIR}/kept_sites_rep{{rep}}.txt",
        sha1        = f"{SIM_DIR}/kept_sites_rep{{rep}}.sha1",
        recmap      = f"{SIM_DIR}/flat_map.txt"
    output:
        ld_pkl      = f"{TRAD_LD_DIR}/LD_stats_window_{{rep}}.pkl"
    shell:
        """
        python snakemake_scripts/traditional_ld_rep.py \
          --sim-dir {SIM_DIR} \
          --trad-ld-dir {TRAD_LD_DIR} \
          --rep {wildcards.rep} \
          --num-rbins {NUM_RBINS}
        """


# 4) GPU LD PER REPLICATE (parallelizable, 1 GPU per job)
rule gpu_ld_rep:
    input:
        ts_filt    = f"{SIM_DIR}/window_{{rep}}.filtered.trees",
        kept_sites = f"{SIM_DIR}/kept_sites_rep{{rep}}.txt",
        sha1       = f"{SIM_DIR}/kept_sites_rep{{rep}}.sha1"
    resources:
        gpu = 1
    output:
        ld_pkl     = f"{GPU_LD_DIR}/LD_stats_window_{{rep}}.pkl"
    wildcard_constraints:
        rep = r"\d+"
    shell:
        """
        python snakemake_scripts/gpu_ld_rep.py \
          --sim-dir {SIM_DIR} \
          --gpu-ld-dir {GPU_LD_DIR} \
          --rep {wildcards.rep} \
          --r-per-bp {R_PER_BP} \
          --num-rbins {NUM_RBINS}
        """


# 5) BOOTSTRAP + PLOT FOR TRADITIONAL
rule traditional_bootstrap:
    input:
        ld_pkls = expand(f"{TRAD_LD_DIR}/LD_stats_window_{{rep}}.pkl", rep=REPS)
    output:
        TRAD_MEANS,
        TRAD_BOOT,
        TRAD_PLOT
    shell:
        """
        python snakemake_scripts/traditional_bootstrap.py \
          --trad-ld-dir {TRAD_LD_DIR} \
          --trad-results-dir {TRAD_RESULTS_DIR} \
          --num-reps {NUM_REPS} \
          --num-rbins {NUM_RBINS}
        """


# 6) BOOTSTRAP + PLOT FOR GPU
rule gpu_bootstrap:
    input:
        ld_pkls = expand(f"{GPU_LD_DIR}/LD_stats_window_{{rep}}.pkl", rep=REPS)
    output:
        GPU_MEANS,
        GPU_BOOT,
        GPU_PLOT
    shell:
        """
        python snakemake_scripts/gpu_bootstrap.py \
          --gpu-ld-dir {GPU_LD_DIR} \
          --gpu-results-dir {GPU_RESULTS_DIR} \
          --num-reps {NUM_REPS} \
          --num-rbins {NUM_RBINS}
        """
