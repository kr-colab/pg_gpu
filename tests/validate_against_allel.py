"""
Cross-validation of pg_gpu statistics against scikit-allel reference implementations.

This script is NOT part of the pytest suite. Run it directly:

    pixi run python tests/validate_against_allel.py

It loads real Ag1000G data (100 A. gambiae, X chromosome 8-12 Mb) and compares
every statistic that has a scikit-allel equivalent. Results are printed as a
table with PASS/FAIL status.
"""

import sys
import time
import numpy as np
import allel

from pg_gpu import HaplotypeMatrix, diversity, divergence, sfs, selection, decomposition
from pg_gpu import admixture

# ── Configuration ──────────────────────────────────────────────────────────
VCF_PATH = "examples/data/gamb.X.8-12Mb.n100.derived.vcf.gz"
RTOL = 1e-4          # relative tolerance for floating-point comparisons
ATOL = 1e-8          # absolute tolerance
N_POP1 = 50          # first 50 diploid individuals -> pop1 (100 haplotypes)
N_POP2 = 50          # next 50 -> pop2 (100 haplotypes)

# Selection scan sub-region (keep tractable)
SEL_START, SEL_END = 9_950_000, 10_000_000


# ── Helpers ────────────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.rows = []
        self.timings = []  # (category, name, allel_sec, pg_sec)

    def add(self, category, name, ref, pg, *, is_array=False, skip_msg=None,
            t_allel=None, t_pg=None):
        if t_allel is not None and t_pg is not None:
            self.timings.append((category, name, t_allel, t_pg))
        if skip_msg:
            self.rows.append((category, name, None, None, "SKIP", skip_msg))
            return
        if is_array:
            ref = np.asarray(ref, dtype=float)
            pg = np.asarray(pg, dtype=float)
            valid = np.isfinite(ref) & np.isfinite(pg)
            if valid.sum() == 0:
                self.rows.append((category, name, "all-nan", "all-nan", "SKIP", ""))
                return
            r, p = ref[valid], pg[valid]
            maxerr = np.max(np.abs(r - p) / (np.abs(r) + ATOL))
            ok = np.allclose(r, p, rtol=RTOL, atol=ATOL)
            self.rows.append((category, name,
                              f"array({valid.sum()})", f"maxrel={maxerr:.2e}",
                              "PASS" if ok else "FAIL", ""))
        else:
            ref_f, pg_f = float(ref), float(pg)
            ok = np.isclose(ref_f, pg_f, rtol=RTOL, atol=ATOL)
            self.rows.append((category, name, f"{ref_f:.8g}", f"{pg_f:.8g}",
                              "PASS" if ok else "FAIL", ""))

    def print(self):
        n_pass = sum(1 for r in self.rows if r[4] == "PASS")
        n_fail = sum(1 for r in self.rows if r[4] == "FAIL")
        n_skip = sum(1 for r in self.rows if r[4] == "SKIP")

        print(f"\n{'Category':<14} {'Statistic':<28} {'allel':>16} {'pg_gpu':>16} {'Status':>6}")
        print("-" * 84)
        for cat, name, ref_s, pg_s, status, msg in self.rows:
            extra = f"  ({msg})" if msg else ""
            ref_disp = ref_s if ref_s is not None else ""
            pg_disp = pg_s if pg_s is not None else ""
            print(f"{cat:<14} {name:<28} {ref_disp:>16} {pg_disp:>16} {status:>6}{extra}")
        print("-" * 84)
        print(f"Total: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")

        if self.timings:
            print(f"\n{'Category':<14} {'Statistic':<28} {'allel (s)':>10} {'pg_gpu (s)':>10} {'speedup':>8}")
            print("-" * 74)
            for cat, name, ta, tp in self.timings:
                speedup = ta / tp if tp > 0 else float('inf')
                print(f"{cat:<14} {name:<28} {ta:>10.4f} {tp:>10.4f} {speedup:>7.1f}x")
            print("-" * 74)

        return n_fail


# ── Load data ──────────────────────────────────────────────────────────────
def main():
    res = Results()
    t0 = time.time()

    print(f"Loading {VCF_PATH} ...")
    hm = HaplotypeMatrix.from_vcf(VCF_PATH)
    n_hap = hm.num_haplotypes
    n_var = hm.num_variants
    positions = np.asarray(hm.positions)
    print(f"  {n_hap} haplotypes x {n_var:,} variants, loaded in {time.time()-t0:.1f}s")

    # Define populations (haplotype indices)
    pop1_idx = list(range(0, N_POP1 * 2))
    pop2_idx = list(range(N_POP1 * 2, (N_POP1 + N_POP2) * 2))
    hm.sample_sets = {"pop1": pop1_idx, "pop2": pop2_idx}
    hm.transfer_to_gpu()

    # Build allel objects
    hap_np = np.asarray(hm.haplotypes.get()).T  # (n_variants, n_haplotypes)
    h_allel = allel.HaplotypeArray(hap_np)
    ac_all = h_allel.count_alleles()
    ac1 = h_allel.count_alleles(subpop=pop1_idx)
    ac2 = h_allel.count_alleles(subpop=pop2_idx)

    # ── DIVERSITY ──────────────────────────────────────────────────────────
    print("Computing diversity statistics ...")

    # pi (absolute, no span normalization)
    t = time.time()
    mpd_allel = allel.mean_pairwise_difference(ac_all, fill=0)
    pi_allel = np.sum(mpd_allel)
    ta = time.time() - t
    t = time.time()
    pi_pg = diversity.pi(hm, span_normalize=False)
    tp = time.time() - t
    res.add("diversity", "pi (absolute)", pi_allel, pi_pg, t_allel=ta, t_pg=tp)

    # pi per-population
    for label, ac_pop, pop_name in [("pop1", ac1, "pop1"), ("pop2", ac2, "pop2")]:
        mpd_pop = allel.mean_pairwise_difference(ac_pop, fill=0)
        pi_pop_allel = np.sum(mpd_pop) / n_var
        pi_pop_pg = divergence.pi_within_population(hm, pop_name)
        res.add("diversity", f"pi_within ({label})", pi_pop_allel, pi_pop_pg)

    # theta_w (absolute)
    t = time.time()
    n_allel = int(ac_all.sum(axis=1).max())
    S_allel = int(ac_all.count_segregating())
    a1 = np.sum(1.0 / np.arange(1, n_allel))
    tw_allel = S_allel / a1
    ta = time.time() - t
    t = time.time()
    tw_pg = diversity.theta_w(hm, span_normalize=False)
    tp = time.time() - t
    res.add("diversity", "theta_w (absolute)", tw_allel, tw_pg,
            t_allel=ta, t_pg=tp)

    # tajima_d
    t = time.time()
    td_allel = allel.tajima_d(ac_all)
    ta = time.time() - t
    t = time.time()
    td_pg = diversity.tajimas_d(hm)
    tp = time.time() - t
    res.add("diversity", "tajima_d", td_allel, td_pg, t_allel=ta, t_pg=tp)

    # segregating sites
    S_pg = diversity.segregating_sites(hm)
    res.add("diversity", "segregating_sites", S_allel, S_pg)

    # singletons (manual from allel)
    dac_all = hap_np.sum(axis=1)
    sing_allel = int(np.sum(dac_all == 1))
    sing_pg = diversity.singleton_count(hm)
    res.add("diversity", "singleton_count", sing_allel, sing_pg)

    # haplotype_diversity (whole sample)
    t = time.time()
    hd_allel = allel.haplotype_diversity(h_allel)
    ta = time.time() - t
    t = time.time()
    hd_pg = diversity.haplotype_diversity(hm)
    tp = time.time() - t
    res.add("diversity", "haplotype_diversity", hd_allel, hd_pg,
            t_allel=ta, t_pg=tp)

    # ── SFS ────────────────────────────────────────────────────────────────
    print("Computing SFS ...")

    # Unfolded SFS (all samples)
    t = time.time()
    sfs_allel = allel.sfs(dac_all, n=n_hap)
    ta = time.time() - t
    t = time.time()
    sfs_pg = sfs.sfs(hm)
    tp = time.time() - t
    sfs_pg_np = np.asarray(sfs_pg)
    min_len = min(len(sfs_allel), len(sfs_pg_np))
    res.add("sfs", "sfs (unfolded)", sfs_allel[:min_len], sfs_pg_np[:min_len],
            is_array=True, t_allel=ta, t_pg=tp)

    # Folded SFS
    sfs_f_allel = allel.sfs_folded(ac_all)
    sfs_f_pg = sfs.sfs_folded(hm)
    sfs_f_pg_np = np.asarray(sfs_f_pg)
    min_len = min(len(sfs_f_allel), len(sfs_f_pg_np))
    res.add("sfs", "sfs (folded)", sfs_f_allel[:min_len], sfs_f_pg_np[:min_len],
            is_array=True)

    # Joint SFS
    t = time.time()
    jsfs_allel = allel.joint_sfs(ac1[:, 1], ac2[:, 1],
                                  n1=len(pop1_idx), n2=len(pop2_idx))
    ta = time.time() - t
    t = time.time()
    jsfs_pg = sfs.joint_sfs(hm, pop1="pop1", pop2="pop2")
    tp = time.time() - t
    jsfs_pg_np = np.asarray(jsfs_pg)
    s0 = min(jsfs_allel.shape[0], jsfs_pg_np.shape[0])
    s1 = min(jsfs_allel.shape[1], jsfs_pg_np.shape[1])
    res.add("sfs", "joint_sfs", jsfs_allel[:s0, :s1], jsfs_pg_np[:s0, :s1],
            is_array=True, t_allel=ta, t_pg=tp)

    # Joint SFS folded
    jsfs_f_allel = allel.joint_sfs_folded(ac1, ac2)
    jsfs_f_pg = sfs.joint_sfs_folded(hm, pop1="pop1", pop2="pop2")
    jsfs_f_pg_np = np.asarray(jsfs_f_pg)
    s0 = min(jsfs_f_allel.shape[0], jsfs_f_pg_np.shape[0])
    s1 = min(jsfs_f_allel.shape[1], jsfs_f_pg_np.shape[1])
    res.add("sfs", "joint_sfs_folded", jsfs_f_allel[:s0, :s1],
            jsfs_f_pg_np[:s0, :s1], is_array=True)

    # ── DIVERGENCE ─────────────────────────────────────────────────────────
    print("Computing divergence statistics ...")

    # Dxy (per-site average)
    t = time.time()
    mpd_between = allel.mean_pairwise_difference_between(ac1, ac2, fill=0)
    dxy_allel = np.sum(mpd_between) / n_var
    ta = time.time() - t
    t = time.time()
    dxy_pg = divergence.dxy(hm, "pop1", "pop2")
    tp = time.time() - t
    res.add("divergence", "dxy", dxy_allel, dxy_pg, t_allel=ta, t_pg=tp)

    # Da = Dxy - (pi1+pi2)/2
    mpd1 = allel.mean_pairwise_difference(ac1, fill=0)
    mpd2 = allel.mean_pairwise_difference(ac2, fill=0)
    pi1_a = np.sum(mpd1) / n_var
    pi2_a = np.sum(mpd2) / n_var
    da_allel = dxy_allel - (pi1_a + pi2_a) / 2
    da_pg = divergence.da(hm, "pop1", "pop2")
    res.add("divergence", "da", da_allel, da_pg)

    # Hudson FST
    t = time.time()
    num_h, den_h = allel.hudson_fst(ac1, ac2)
    fst_h_allel = np.nansum(num_h) / np.nansum(den_h)
    ta = time.time() - t
    t = time.time()
    fst_h_pg = divergence.fst_hudson(hm, "pop1", "pop2")
    tp = time.time() - t
    res.add("divergence", "fst_hudson", fst_h_allel, fst_h_pg,
            t_allel=ta, t_pg=tp)

    # Weir-Cockerham FST (diploid-based in allel, haploid in pg_gpu)
    g_allel = h_allel.to_genotypes(ploidy=2)
    dip_pop1 = list(range(N_POP1))
    dip_pop2 = list(range(N_POP1, N_POP1 + N_POP2))
    t = time.time()
    a_wc, b_wc, c_wc = allel.weir_cockerham_fst(g_allel, [dip_pop1, dip_pop2])
    fst_wc_allel = np.nansum(a_wc) / (np.nansum(a_wc) + np.nansum(b_wc) + np.nansum(c_wc))
    ta = time.time() - t
    t = time.time()
    fst_wc_pg = divergence.fst_weir_cockerham(hm, "pop1", "pop2")
    tp = time.time() - t
    res.add("divergence", "fst_wc (haploid vs diploid)",
            fst_wc_allel, fst_wc_pg,
            skip_msg="expected: allel uses diploid h_bar, pg_gpu uses h_bar=0",
            t_allel=ta, t_pg=tp)

    # Patterson FST
    num_p, den_p = allel.patterson_fst(ac1, ac2)
    fst_p_allel = np.nansum(num_p) / np.nansum(den_p)
    res.add("divergence", "patterson_fst (info)",
            fst_p_allel, fst_p_allel, skip_msg="no pg_gpu equivalent")

    # ── ADMIXTURE (Patterson F-statistics) ─────────────────────────────────
    print("Computing Patterson F-statistics ...")

    # For F3/D we need 3-4 populations; split pop1 in half
    pop_a_idx = pop1_idx[:50]
    pop_b_idx = pop1_idx[50:]
    pop_c_idx = pop2_idx[:50]
    pop_d_idx = pop2_idx[50:]
    hm.sample_sets["pop_a"] = pop_a_idx
    hm.sample_sets["pop_b"] = pop_b_idx
    hm.sample_sets["pop_c"] = pop_c_idx
    hm.sample_sets["pop_d"] = pop_d_idx

    ac_a = h_allel.count_alleles(subpop=pop_a_idx)
    ac_b = h_allel.count_alleles(subpop=pop_b_idx)
    ac_c = h_allel.count_alleles(subpop=pop_c_idx)
    ac_d = h_allel.count_alleles(subpop=pop_d_idx)

    # Patterson F2
    t = time.time()
    f2_allel = allel.patterson_f2(ac_a, ac_b)
    ta = time.time() - t
    t = time.time()
    f2_pg = admixture.patterson_f2(hm, "pop_a", "pop_b")
    tp = time.time() - t
    f2_pg_np = np.asarray(f2_pg)
    res.add("admixture", "patterson_f2 (per-site)", f2_allel, f2_pg_np,
            is_array=True, t_allel=ta, t_pg=tp)

    # Patterson F3: F3(C; A, B)
    t = time.time()
    T_allel, B_allel = allel.patterson_f3(ac_c, ac_a, ac_b)
    f3_allel = np.nansum(T_allel) / np.nansum(B_allel)
    ta = time.time() - t
    t = time.time()
    T_pg, B_pg = admixture.patterson_f3(hm, "pop_c", "pop_a", "pop_b")
    f3_pg = np.nansum(T_pg) / np.nansum(B_pg)
    tp = time.time() - t
    res.add("admixture", "patterson_f3 (normalized)", f3_allel, f3_pg,
            t_allel=ta, t_pg=tp)

    # Patterson D: D(A, B; C, D)
    t = time.time()
    num_d_allel, den_d_allel = allel.patterson_d(ac_a, ac_b, ac_c, ac_d)
    d_allel = np.nansum(num_d_allel) / np.nansum(den_d_allel)
    ta = time.time() - t
    t = time.time()
    num_d_pg, den_d_pg = admixture.patterson_d(hm, "pop_a", "pop_b", "pop_c", "pop_d")
    d_pg = np.nansum(num_d_pg) / np.nansum(den_d_pg)
    tp = time.time() - t
    res.add("admixture", "patterson_d (normalized)", d_allel, d_pg,
            t_allel=ta, t_pg=tp)

    # ── SELECTION SCANS ────────────────────────────────────────────────────
    print("Computing selection scan statistics ...")

    # Subset to smaller region for selection scans
    import cupy as cp
    pos_gpu = cp.asarray(positions)
    sel_idx = cp.where((pos_gpu >= SEL_START) & (pos_gpu < SEL_END))[0]
    h_sel = hm.get_subset(sel_idx)
    pos_sel = h_sel.positions.get() if hasattr(h_sel.positions, 'get') else np.asarray(h_sel.positions)
    hap_sel_np = h_sel.haplotypes.get().T if hasattr(h_sel.haplotypes, 'get') else np.asarray(h_sel.haplotypes).T
    h_sel_allel = allel.HaplotypeArray(hap_sel_np)
    n_sel = h_sel.num_variants
    print(f"  Selection sub-region: {SEL_START/1e6:.1f}-{SEL_END/1e6:.1f} Mb, "
          f"{n_sel:,} variants")

    # Garud's H
    t = time.time()
    h1_a, h12_a, h123_a, h2h1_a = allel.garud_h(h_sel_allel)
    ta = time.time() - t
    t = time.time()
    h_result = selection.garud_h(h_sel)
    tp = time.time() - t
    res.add("selection", "garud_h H1", h1_a, h_result[0], t_allel=ta, t_pg=tp)
    res.add("selection", "garud_h H12", h12_a, h_result[1])
    res.add("selection", "garud_h H123", h123_a, h_result[2])
    res.add("selection", "garud_h H2/H1", h2h1_a, h_result[3])

    # nSL
    t = time.time()
    nsl_allel = allel.nsl(h_sel_allel)
    ta = time.time() - t
    t = time.time()
    nsl_pg = selection.nsl(h_sel)
    tp = time.time() - t
    res.add("selection", "nsl (per-site)", nsl_allel, nsl_pg,
            is_array=True, t_allel=ta, t_pg=tp)

    # iHS
    t = time.time()
    ihs_allel = allel.ihs(h_sel_allel, pos_sel, include_edges=False)
    ta = time.time() - t
    t = time.time()
    ihs_pg = selection.ihs(h_sel, include_edges=False)
    tp = time.time() - t
    res.add("selection", "ihs (per-site)", ihs_allel, ihs_pg,
            is_array=True, t_allel=ta, t_pg=tp)

    # Cross-population nSL
    h_sel_pop1 = allel.HaplotypeArray(hap_sel_np[:, pop1_idx])
    h_sel_pop2 = allel.HaplotypeArray(hap_sel_np[:, pop2_idx])
    t = time.time()
    xpnsl_allel = allel.xpnsl(h_sel_pop1, h_sel_pop2)
    ta = time.time() - t
    h_sel.sample_sets = {"pop1": pop1_idx, "pop2": pop2_idx}
    t = time.time()
    xpnsl_pg = selection.xpnsl(h_sel, "pop1", "pop2")
    tp = time.time() - t
    res.add("selection", "xpnsl (per-site)", xpnsl_allel, xpnsl_pg,
            is_array=True, t_allel=ta, t_pg=tp)

    # Cross-population EHH
    t = time.time()
    xpehh_allel = allel.xpehh(h_sel_pop1, h_sel_pop2, pos_sel,
                               include_edges=False)
    ta = time.time() - t
    t = time.time()
    xpehh_pg = selection.xpehh(h_sel, "pop1", "pop2", include_edges=False)
    tp = time.time() - t
    res.add("selection", "xpehh (per-site)", xpehh_allel, xpehh_pg,
            is_array=True, t_allel=ta, t_pg=tp)

    # EHH decay
    t = time.time()
    ehh_allel = allel.ehh_decay(h_sel_allel)
    ta = time.time() - t
    t = time.time()
    ehh_pg = selection.ehh_decay(h_sel)
    tp = time.time() - t
    res.add("selection", "ehh_decay", ehh_allel, ehh_pg,
            is_array=True, t_allel=ta, t_pg=tp)

    # Moving Garud's H
    win_size = 400
    t = time.time()
    mgh_allel = allel.moving_garud_h(h_sel_allel, size=win_size)
    ta = time.time() - t
    h1_mg_a = mgh_allel[0]
    h12_mg_a = mgh_allel[1]
    t = time.time()
    h1_mg_pg, h12_mg_pg, _, _ = selection.moving_garud_h(h_sel, size=win_size)
    tp = time.time() - t
    res.add("selection", "moving_garud_h H1", h1_mg_a, h1_mg_pg,
            is_array=True, t_allel=ta, t_pg=tp)
    res.add("selection", "moving_garud_h H12", h12_mg_a, h12_mg_pg,
            is_array=True)

    # ── PRINT RESULTS ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_fail = res.print()
    print(f"\nCompleted in {elapsed:.1f}s")
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
