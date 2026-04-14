#!/usr/bin/env Rscript
#
# Time lostruct::eigen_windows + lostruct::pc_dist on a single input
# dataset. Developer-local only; invoked by examples/local_pca_benchmark.py.
#
# Usage:
#   Rscript examples/local_pca_benchmark.R <input.json> <output.json>
#
# Input JSON (produced by the Python orchestrator):
#   hap          integer matrix, rows = samples, cols = variants
#   window_size  SNP-count window size
#   k            number of PCs
#
# Output JSON:
#   n_windows
#   eigen_windows_s  wall-clock seconds for lostruct::eigen_windows
#   pc_dist_s        wall-clock seconds for lostruct::pc_dist (normalize="L1")
#   total_s          eigen_windows_s + pc_dist_s

suppressPackageStartupMessages({
    library(lostruct)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    stop("usage: Rscript local_pca_benchmark.R <input.json> <output.json>")
}
in_path <- args[1]
out_path <- args[2]

payload <- fromJSON(in_path)
hap <- as.matrix(payload$hap)
window_size <- as.integer(payload$window_size)
k <- as.integer(payload$k)

# lostruct wants variants as rows, samples as columns.
variant_major <- t(hap)

t0 <- Sys.time()
eigs <- eigen_windows(variant_major, k = k, win = window_size)
t1 <- Sys.time()
pd <- pc_dist(eigs, npc = k, normalize = "L1")
t2 <- Sys.time()

eigen_windows_s <- as.numeric(difftime(t1, t0, units = "secs"))
pc_dist_s <- as.numeric(difftime(t2, t1, units = "secs"))

write_json(list(
    n_windows = nrow(eigs),
    eigen_windows_s = eigen_windows_s,
    pc_dist_s = pc_dist_s,
    total_s = eigen_windows_s + pc_dist_s
), out_path, digits = 10, auto_unbox = TRUE)
