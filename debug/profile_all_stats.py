"""Profile all pg_gpu statistics to find the slowest ones."""

import time
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity, divergence, selection, sfs, ld_statistics
from pg_gpu import admixture, decomposition, relatedness


def make_data(n_haps=200, n_snps=50000, seed=42):
    rng = np.random.default_rng(seed)
    founders = rng.integers(0, 2, size=(5, n_snps), dtype=np.int8)
    assignments = rng.integers(0, 5, size=n_haps)
    haps = founders[assignments].copy()
    mutations = rng.random(size=(n_haps, n_snps)) < 0.02
    haps ^= mutations.astype(np.int8)
    positions = np.arange(n_snps) * 100
    hm = HaplotypeMatrix(haps, positions, positions[0], positions[-1])
    hm.transfer_to_gpu()
    return hm


def bench(name, fn, n_reps=3):
    try:
        fn()
        cp.cuda.Device(0).synchronize()
    except Exception as e:
        print(f"  {name:.<45} ERROR: {e}")
        return None

    times = []
    for _ in range(n_reps):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    median_ms = np.median(times) * 1000
    print(f"  {name:.<45} {median_ms:>10.2f} ms")
    return median_ms


def main():
    print("Generating data: 200 haplotypes x 50,000 variants")
    hm = make_data(200, 50000)

    pop1 = list(range(50))
    pop2 = list(range(50, 100))
    pop3 = list(range(50, 75))

    results = {}

    # --- Diversity ---
    print("\n=== DIVERSITY ===")
    for name, fn in [
        ("pi", lambda: diversity.pi(hm)),
        ("theta_w", lambda: diversity.theta_w(hm)),
        ("theta_h", lambda: diversity.theta_h(hm)),
        ("theta_l", lambda: diversity.theta_l(hm)),
        ("tajimas_d", lambda: diversity.tajimas_d(hm)),
        ("fay_wus_h", lambda: diversity.fay_wus_h(hm)),
        ("normalized_fay_wus_h", lambda: diversity.normalized_fay_wus_h(hm)),
        ("zeng_e", lambda: diversity.zeng_e(hm)),
        ("zeng_dh", lambda: diversity.zeng_dh(hm)),
        ("segregating_sites", lambda: diversity.segregating_sites(hm)),
        ("singleton_count", lambda: diversity.singleton_count(hm)),
        ("allele_frequency_spectrum", lambda: diversity.allele_frequency_spectrum(hm)),
        ("haplotype_diversity", lambda: diversity.haplotype_diversity(hm)),
        ("haplotype_count", lambda: diversity.haplotype_count(hm)),
        ("max_daf", lambda: diversity.max_daf(hm)),
        ("daf_histogram", lambda: diversity.daf_histogram(hm)),
        ("heterozygosity_expected", lambda: diversity.heterozygosity_expected(hm)),
        ("heterozygosity_observed", lambda: diversity.heterozygosity_observed(hm)),
        ("inbreeding_coefficient", lambda: diversity.inbreeding_coefficient(hm)),
        ("diversity_stats", lambda: diversity.diversity_stats(hm)),
        ("diversity_stats_fast", lambda: diversity.diversity_stats_fast(hm)),
        ("neutrality_tests", lambda: diversity.neutrality_tests(hm)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Divergence ---
    print("\n=== DIVERGENCE ===")
    for name, fn in [
        ("fst_hudson", lambda: divergence.fst_hudson(hm, pop1, pop2)),
        ("fst_weir_cockerham", lambda: divergence.fst_weir_cockerham(hm, pop1, pop2)),
        ("fst_nei", lambda: divergence.fst_nei(hm, pop1, pop2)),
        ("dxy", lambda: divergence.dxy(hm, pop1, pop2)),
        ("da", lambda: divergence.da(hm, pop1, pop2)),
        ("pbs", lambda: divergence.pbs(hm, pop1, pop2, pop3)),
        ("snn", lambda: divergence.snn(hm, pop1, pop2)),
        ("dxy_min", lambda: divergence.dxy_min(hm, pop1, pop2)),
        ("gmin", lambda: divergence.gmin(hm, pop1, pop2)),
        ("dd", lambda: divergence.dd(hm, pop1, pop2)),
        ("dd_rank", lambda: divergence.dd_rank(hm, pop1, pop2)),
        ("zx", lambda: divergence.zx(hm, pop1, pop2)),
        ("divergence_stats", lambda: divergence.divergence_stats(hm, pop1, pop2)),
        ("distance_based_stats", lambda: divergence.distance_based_stats(hm, pop1, pop2)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Selection ---
    print("\n=== SELECTION ===")
    for name, fn in [
        ("garud_h", lambda: selection.garud_h(hm)),
        ("nsl", lambda: selection.nsl(hm)),
        ("ihs", lambda: selection.ihs(hm)),
        ("xpehh", lambda: selection.xpehh(hm, pop1, pop2)),
        ("xpnsl", lambda: selection.xpnsl(hm, pop1, pop2)),
        ("ehh_decay", lambda: selection.ehh_decay(hm)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- SFS ---
    print("\n=== SFS ===")
    for name, fn in [
        ("sfs", lambda: sfs.sfs(hm)),
        ("sfs_folded", lambda: sfs.sfs_folded(hm)),
        ("joint_sfs", lambda: sfs.joint_sfs(hm, pop1, pop2)),
        ("joint_sfs_folded", lambda: sfs.joint_sfs_folded(hm, pop1, pop2)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- LD ---
    print("\n=== LD STATISTICS ===")
    for name, fn in [
        ("r_squared", lambda: ld_statistics.r_squared(hm)),
        ("zns", lambda: ld_statistics.zns(hm)),
        ("omega", lambda: ld_statistics.omega(hm)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Admixture ---
    print("\n=== ADMIXTURE ===")
    for name, fn in [
        ("patterson_f2", lambda: admixture.patterson_f2(hm, pop1, pop2)),
        ("patterson_f3", lambda: admixture.patterson_f3(hm, pop3, pop1, pop2)),
        ("patterson_d", lambda: admixture.patterson_d(hm, pop1, pop2, pop3, list(range(75, 100)))),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Decomposition ---
    print("\n=== DECOMPOSITION ===")
    for name, fn in [
        ("pca", lambda: decomposition.pca(hm, n_components=10)),
        ("randomized_pca", lambda: decomposition.randomized_pca(hm, n_components=10)),
        ("pairwise_distance", lambda: decomposition.pairwise_distance(hm)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Relatedness ---
    print("\n=== RELATEDNESS ===")
    for name, fn in [
        ("grm", lambda: relatedness.grm(hm)),
        ("ibs", lambda: relatedness.ibs(hm)),
    ]:
        r = bench(name, fn)
        if r is not None:
            results[name] = r

    # --- Summary: top 15 slowest ---
    print("\n" + "=" * 60)
    print("TOP 15 SLOWEST STATISTICS")
    print("=" * 60)
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, ms) in enumerate(ranked[:15], 1):
        print(f"  {i:>2}. {name:.<45} {ms:>10.2f} ms")


if __name__ == "__main__":
    main()
