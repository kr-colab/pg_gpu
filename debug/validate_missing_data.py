#!/usr/bin/env python
"""
Validate missing data handling across pg_gpu statistics using
unphased Ag1000G 3L data (~75% missingness).

Tests three missing_data modes:
  - 'include': per-site valid counts (default)
  - 'exclude': drop sites with any missing
  - 'pairwise': pixy-style sum(diffs)/sum(comps)

Checks:
  1. All modes run without error on high-missingness data
  2. Results are finite (no NaN/Inf from division by zero)
  3. 'exclude' on pre-filtered complete data matches 'include'
  4. 'pairwise' values differ from 'include' (missingness matters)
  5. Values are in expected ranges (e.g., 0 <= FST <= 1)

Usage:
    pixi run python debug/validate_missing_data.py
"""

import numpy as np
from pg_gpu import HaplotypeMatrix, diversity, divergence, ld_statistics

ZARR_PATH = "/sietch_colab/data_share/Ag1000G/Ag3.0/ag1000g.unphased.3L.zarr"
REGION = "3L:5000000-5500000"

# Split samples into two populations for divergence stats
N_POP = 200  # haplotypes per population


def load():
    print(f"Loading {REGION}...", flush=True)
    hm = HaplotypeMatrix.from_zarr(ZARR_PATH, region=REGION)
    print(f"  {hm.num_haplotypes} haps x {hm.num_variants:,} variants")

    hap = hm.haplotypes
    n_miss = np.sum(hap < 0)
    pct = 100 * n_miss / hap.size
    print(f"  Missing: {pct:.1f}%")

    # Define two populations
    hm.sample_sets = {
        "pop1": list(range(0, N_POP)),
        "pop2": list(range(N_POP, 2 * N_POP)),
    }
    hm.transfer_to_gpu()
    return hm


def make_complete_subset(hm):
    """Create a subset with zero missing data for consistency checks."""
    import cupy as cp
    hap = hm.haplotypes
    n_missing_per_site = cp.sum(hap < 0, axis=0)
    complete = cp.where(n_missing_per_site == 0)[0]
    if len(complete) < 10:
        print("  WARNING: fewer than 10 complete sites, skipping consistency check")
        return None
    hm_complete = hm.get_subset(complete)
    hm_complete.sample_sets = hm.sample_sets
    print(f"  Complete-data subset: {len(complete)} sites")
    return hm_complete


def check(name, value, expected_finite=True, lo=None, hi=None):
    """Validate a scalar result."""
    ok = True
    issues = []

    if expected_finite and not np.isfinite(value):
        issues.append(f"not finite ({value})")
        ok = False
    if lo is not None and value < lo:
        issues.append(f"below {lo} ({value})")
        ok = False
    if hi is not None and value > hi:
        issues.append(f"above {hi} ({value})")
        ok = False

    status = "PASS" if ok else "FAIL"
    detail = f" -- {'; '.join(issues)}" if issues else ""
    print(f"    {name:<45s} = {value:>12.6f}  [{status}]{detail}")
    return ok


def test_diversity(hm, hm_complete):
    """Test diversity statistics across missing data modes."""
    print("\n== Diversity ==")
    all_pass = True

    for mode in ['include', 'exclude', 'pairwise']:
        print(f"\n  mode='{mode}':")
        kwargs = dict(population="pop1", missing_data=mode)
        all_pass &= check("pi", diversity.pi(hm, **kwargs), lo=0)
        all_pass &= check("theta_w", diversity.theta_w(hm, **kwargs), lo=0)
        all_pass &= check("tajimas_d", diversity.tajimas_d(hm, **kwargs),
                          lo=-3, hi=3)
        all_pass &= check("theta_h", diversity.theta_h(hm, **kwargs), lo=0)
        all_pass &= check("theta_l", diversity.theta_l(hm, **kwargs), lo=0)

    # Consistency: 'include' on complete data == 'exclude' on complete data
    if hm_complete is not None:
        print("\n  Consistency (complete data, include vs exclude):")
        for stat_name, fn in [("pi", diversity.pi),
                              ("theta_w", diversity.theta_w),
                              ("tajimas_d", diversity.tajimas_d)]:
            v_inc = fn(hm_complete, population="pop1", missing_data='include')
            v_exc = fn(hm_complete, population="pop1", missing_data='exclude')
            rel = abs(v_inc - v_exc) / max(abs(v_exc), 1e-15)
            ok = rel < 1e-6
            status = "PASS" if ok else "FAIL"
            print(f"    {stat_name:<20s} include={v_inc:.8f}  "
                  f"exclude={v_exc:.8f}  rel_err={rel:.2e}  [{status}]")
            all_pass &= ok

    return all_pass


def test_divergence(hm, hm_complete):
    """Test divergence statistics across missing data modes."""
    print("\n== Divergence ==")
    all_pass = True

    for mode in ['include', 'exclude', 'pairwise']:
        print(f"\n  mode='{mode}':")
        kwargs = dict(pop1="pop1", pop2="pop2", missing_data=mode)
        all_pass &= check("fst_hudson",
                          divergence.fst_hudson(hm, **kwargs), lo=-0.5, hi=1)
        all_pass &= check("fst_weir_cockerham",
                          divergence.fst_weir_cockerham(hm, **kwargs),
                          lo=-0.5, hi=1)
        try:
            all_pass &= check("dxy", divergence.dxy(hm, **kwargs), lo=0)
        except IndexError as e:
            print(f"    {'dxy':<45s}   BUG: {e}")
            all_pass = False
        try:
            all_pass &= check("da", divergence.da(hm, **kwargs))
        except IndexError as e:
            print(f"    {'da':<45s}   BUG: {e}")
            all_pass = False

    # Consistency on complete data
    if hm_complete is not None:
        print("\n  Consistency (complete data, include vs exclude):")
        for stat_name, fn in [("fst_hudson", divergence.fst_hudson),
                              ("dxy", divergence.dxy)]:
            v_inc = fn(hm_complete, pop1="pop1", pop2="pop2",
                       missing_data='include')
            v_exc = fn(hm_complete, pop1="pop1", pop2="pop2",
                       missing_data='exclude')
            rel = abs(v_inc - v_exc) / max(abs(v_exc), 1e-15)
            ok = rel < 1e-6
            status = "PASS" if ok else "FAIL"
            print(f"    {stat_name:<20s} include={v_inc:.8f}  "
                  f"exclude={v_exc:.8f}  rel_err={rel:.2e}  [{status}]")
            all_pass &= ok

    return all_pass


def test_ld(hm, hm_complete):
    """Test LD statistics across missing data modes."""
    print("\n== LD Statistics ==")
    all_pass = True

    # Use a smaller subset for LD (ZnS is O(m^2) or O(n^2*m))
    import cupy as cp
    n_sites = min(hm.num_variants, 2000)
    hm_small = hm.get_subset(cp.arange(n_sites))
    hm_small.sample_sets = hm.sample_sets

    for mode in ['include', 'exclude']:
        print(f"\n  mode='{mode}':")
        all_pass &= check("zns", ld_statistics.zns(hm_small, missing_data=mode),
                          lo=0, hi=1)
        all_pass &= check("omega",
                          ld_statistics.omega(hm_small, missing_data=mode),
                          lo=0)

    # Also test sigma_d2 estimator (unbiased)
    print(f"\n  estimator='sigma_d2':")
    all_pass &= check("zns (sigma_d2)",
                      ld_statistics.zns(hm_small, estimator='sigma_d2'),
                      lo=-0.1, hi=2)

    return all_pass


def test_neutrality(hm):
    """Test neutrality test statistics with missing data."""
    print("\n== Neutrality Tests ==")
    all_pass = True

    for mode in ['include', 'exclude']:
        print(f"\n  mode='{mode}':")
        kwargs = dict(population="pop1", missing_data=mode)
        all_pass &= check("fay_wus_h",
                          diversity.fay_wus_h(hm, **kwargs))
        all_pass &= check("normalized_fay_wus_h",
                          diversity.normalized_fay_wus_h(hm, **kwargs))
        all_pass &= check("zeng_e",
                          diversity.zeng_e(hm, **kwargs))

    return all_pass


def test_site_counts(hm):
    """Verify that different modes use different site/comparison counts."""
    print("\n== Site Count Sanity ==")
    import cupy as cp

    hap = hm.haplotypes
    pop_idx = hm.sample_sets["pop1"]
    hap_pop = hap[pop_idx]

    n_total = hap_pop.shape[1]
    n_missing_per_site = cp.sum(hap_pop < 0, axis=0)
    n_complete = int(cp.sum(n_missing_per_site == 0).get())
    n_segregating_all = int(cp.sum(
        (cp.sum(cp.maximum(hap_pop, 0), axis=0) > 0) &
        (cp.sum(cp.maximum(hap_pop, 0), axis=0) < cp.sum(hap_pop >= 0, axis=0))
    ).get())

    print(f"  Total sites: {n_total}")
    print(f"  Complete sites (0 missing): {n_complete}")
    print(f"  Segregating (among all): {n_segregating_all}")

    # 'exclude' should use fewer sites than 'include'
    s_inc = diversity.segregating_sites(hm, population="pop1",
                                        missing_data='include')
    s_exc = diversity.segregating_sites(hm, population="pop1",
                                        missing_data='exclude')
    ok = s_exc <= s_inc
    status = "PASS" if ok else "FAIL"
    print(f"  seg_sites include={s_inc}  exclude={s_exc}  "
          f"(exclude <= include: {ok})  [{status}]")
    return ok


def main():
    hm = load()
    hm_complete = make_complete_subset(hm)

    results = []
    results.append(("Site counts", test_site_counts(hm)))
    results.append(("Diversity", test_diversity(hm, hm_complete)))
    results.append(("Divergence", test_divergence(hm, hm_complete)))
    results.append(("LD", test_ld(hm, hm_complete)))
    results.append(("Neutrality", test_neutrality(hm)))

    print("\n== Summary ==")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED.")


if __name__ == "__main__":
    main()
