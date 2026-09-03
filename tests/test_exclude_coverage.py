"""missing_data='exclude' across the statistics modules.

``exclude`` drops every site with a missing call before computing, so a
statistic run with ``missing_data='exclude'`` must equal the same statistic
run (default ``'include'``) on a matrix hand-restricted to the complete
sites. One fixture with a handful of missing calls drives the whole sweep.
"""
import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import (HaplotypeMatrix, GenotypeMatrix, diversity, sfs,
                    divergence, selection, distance_stats, relatedness,
                    admixture)


def _host(a):
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def _two_pop_hm(seed=5, n_per_pop=8, seq_length=200_000):
    """Two named populations, biallelic, with a missing call at a fifth of
    the sites (the rest stay complete, so the exclude subset is well fed)."""
    demography = msprime.Demography()
    demography.add_population(name="p1", initial_size=10_000)
    demography.add_population(name="p2", initial_size=10_000)
    demography.add_population(name="anc", initial_size=10_000)
    demography.add_population_split(time=5000, derived=["p1", "p2"],
                                    ancestral="anc")
    ts = msprime.sim_ancestry(samples={"p1": n_per_pop, "p2": n_per_pop},
                              demography=demography, sequence_length=seq_length,
                              recombination_rate=1e-8, random_seed=seed,
                              ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=seed, model="binary")
    hm = HaplotypeMatrix.from_ts(ts)
    hap = _host(hm.haplotypes).copy()
    pos = _host(hm.positions)
    rng = np.random.default_rng(seed)
    dirty = rng.choice(hap.shape[1], size=hap.shape[1] // 5, replace=False)
    for j in dirty:
        hap[rng.integers(hap.shape[0]), j] = -1
    out = HaplotypeMatrix(hap, pos, hm.chrom_start, hm.chrom_end)
    out.sample_sets = dict(hm.sample_sets)
    return out


def _complete_hm(hm):
    """The independently-built complete-site subset: sites with no -1."""
    hap = _host(hm.haplotypes)
    keep = np.where((hap >= 0).all(axis=0))[0]
    sub = HaplotypeMatrix(hap[:, keep], _host(hm.positions)[keep],
                          hm.chrom_start, hm.chrom_end)
    sub.sample_sets = dict(hm.sample_sets)
    return sub


def _complete_hm_over(hm, pops):
    """Complete-site subset counting only the rows of ``pops`` -- a two/four
    population statistic drops a site when its own populations are incomplete,
    ignoring missingness in populations it never reads."""
    hap = _host(hm.haplotypes)
    rows = sorted(r for p in pops for r in hm.sample_sets[p])
    keep = np.where((hap[rows] >= 0).all(axis=0))[0]
    sub = HaplotypeMatrix(hap[:, keep], _host(hm.positions)[keep],
                          hm.chrom_start, hm.chrom_end)
    sub.sample_sets = dict(hm.sample_sets)
    return sub


def _gm_with_missing(hm):
    # pairwise_diffs_diploid is called with population=None below, so the
    # sample_sets from_haplotype_matrix would carry are not read.
    return GenotypeMatrix.from_haplotype_matrix(hm)


def _complete_gm(gm):
    g = _host(gm.genotypes)
    keep = np.where((g >= 0).all(axis=0))[0]
    sub = GenotypeMatrix(g[:, keep], _host(gm.positions)[keep])
    sub.sample_sets = dict(gm.sample_sets)
    return sub


def _agree(a, b):
    np.testing.assert_allclose(_host(a), _host(b), rtol=1e-9, atol=1e-12,
                               equal_nan=True)


@pytest.fixture(scope="module")
def hm():
    return _two_pop_hm()


@pytest.fixture(scope="module")
def hm_complete(hm):
    return _complete_hm(hm)


@pytest.fixture(scope="module")
def hm4():
    return _four_pop_hm()


# ── single-population diversity ─────────────────────────────────────────
# Each entry is (name, f) where f(matrix, missing_data) issues the call.
SINGLE_POP = [
    ("pi", lambda m, md: diversity.pi(m, missing_data=md)),
    ("theta_w", lambda m, md: diversity.theta_w(m, missing_data=md)),
    ("tajimas_d", lambda m, md: diversity.tajimas_d(m, missing_data=md)),
    ("max_daf", lambda m, md: diversity.max_daf(m, missing_data=md)),
    ("haplotype_diversity",
     lambda m, md: diversity.haplotype_diversity(m, missing_data=md)),
    ("mu_sfs", lambda m, md: diversity.mu_sfs(m, missing_data=md)),
    ("sfs", lambda m, md: sfs.sfs(m, missing_data=md)),
    ("sfs_folded", lambda m, md: sfs.sfs_folded(m, missing_data=md)),
]


@pytest.mark.parametrize("name,fn", SINGLE_POP, ids=[n for n, _ in SINGLE_POP])
def test_single_pop_exclude_matches_complete_subset(name, fn, hm, hm_complete):
    _agree(fn(hm, "exclude"), fn(hm_complete, "include"))


# heterozygosity_expected / observed return a per-site array and keep every
# segregating site, so ``exclude`` nan-masks the incomplete sites rather than
# dropping them (dropping would break the per-site alignment). The non-nan
# entries must still equal the ``include`` values at those sites.
@pytest.mark.parametrize("fn", [diversity.heterozygosity_expected,
                                diversity.heterozygosity_observed],
                         ids=["heterozygosity_expected",
                              "heterozygosity_observed"])
def test_per_site_heterozygosity_exclude_masks_incomplete(fn, hm):
    # The output is one value per site over every site, so exclude nan-masks
    # exactly the incomplete sites and leaves the rest equal to include.
    hap = _host(hm.haplotypes)
    incomplete = ~(hap >= 0).all(axis=0)
    inc = _host(fn(hm, missing_data="include"))
    exc = _host(fn(hm, missing_data="exclude"))
    assert np.isnan(inc).sum() == 0
    assert incomplete.sum() > 0
    np.testing.assert_array_equal(np.isnan(exc), incomplete)
    np.testing.assert_allclose(exc[~incomplete], inc[~incomplete],
                               rtol=1e-9, atol=1e-12)


# ── two-population statistics ───────────────────────────────────────────
def test_dxy_exclude(hm, hm_complete):
    _agree(divergence.dxy(hm, "p1", "p2", missing_data="exclude"),
           divergence.dxy(hm_complete, "p1", "p2"))


def test_fst_hudson_exclude(hm, hm_complete):
    _agree(divergence.fst_hudson(hm, "p1", "p2", missing_data="exclude"),
           divergence.fst_hudson(hm_complete, "p1", "p2"))


def test_fst_weir_cockerham_exclude(hm, hm_complete):
    _agree(divergence.fst_weir_cockerham(hm, "p1", "p2", missing_data="exclude"),
           divergence.fst_weir_cockerham(hm_complete, "p1", "p2"))


def test_fst_nei_exclude(hm, hm_complete):
    _agree(divergence.fst_nei(hm, "p1", "p2", missing_data="exclude"),
           divergence.fst_nei(hm_complete, "p1", "p2"))


def test_joint_sfs_exclude(hm, hm_complete):
    _agree(sfs.joint_sfs(hm, "p1", "p2", missing_data="exclude"),
           sfs.joint_sfs(hm_complete, "p1", "p2"))


def test_genetic_relatedness_exclude(hm, hm_complete):
    # Covers exclude on the grouped-sample_sets, span-normalized path;
    # test_relatedness.py checks the ungrouped drop against a host reference.
    ss = {k: list(v) for k, v in hm.sample_sets.items()}
    _agree(relatedness.genetic_relatedness(hm, sample_sets=list(ss.values()),
                                           missing_data="exclude"),
           relatedness.genetic_relatedness(hm_complete,
                                           sample_sets=list(ss.values())))


# ── selection scans ─────────────────────────────────────────────────────
def test_garud_h_exclude(hm, hm_complete):
    _agree(selection.garud_h(hm, missing_data="exclude"),
           selection.garud_h(hm_complete))


def test_nsl_exclude(hm, hm_complete):
    _agree(selection.nsl(hm, missing_data="exclude"),
           selection.nsl(hm_complete))


def test_ihs_exclude(hm, hm_complete):
    _agree(selection.ihs(hm, missing_data="exclude"),
           selection.ihs(hm_complete))


def test_ehh_decay_exclude(hm, hm_complete):
    _agree(selection.ehh_decay(hm, missing_data="exclude"),
           selection.ehh_decay(hm_complete))


def test_xpehh_exclude(hm, hm_complete):
    _agree(selection.xpehh(hm, "p1", "p2", missing_data="exclude"),
           selection.xpehh(hm_complete, "p1", "p2"))


def test_xpnsl_exclude(hm, hm_complete):
    _agree(selection.xpnsl(hm, "p1", "p2", missing_data="exclude"),
           selection.xpnsl(hm_complete, "p1", "p2"))


# ── distance statistics ─────────────────────────────────────────────────
def test_pairwise_diffs_haploid_exclude(hm, hm_complete):
    _agree(distance_stats.pairwise_diffs_haploid(hm, missing_data="exclude"),
           distance_stats.pairwise_diffs_haploid(hm_complete))


def test_pairwise_diffs_diploid_exclude(hm):
    gm = _gm_with_missing(hm)
    _agree(distance_stats.pairwise_diffs_diploid(gm, missing_data="exclude"),
           distance_stats.pairwise_diffs_diploid(_complete_gm(gm)))


# ── admixture f-statistics ──────────────────────────────────────────────
def _four_pop_hm(seed=6):
    hm = _two_pop_hm(seed=seed)
    n = hm.num_haplotypes
    q = n // 4
    hm.sample_sets = {"A": list(range(0, q)), "B": list(range(q, 2 * q)),
                      "C": list(range(2 * q, 3 * q)), "D": list(range(3 * q, n))}
    return hm


def test_patterson_f2_exclude(hm4):
    hc = _complete_hm_over(hm4, ["A", "B"])
    _agree(admixture.patterson_f2(hm4, "A", "B", missing_data="exclude"),
           admixture.patterson_f2(hc, "A", "B"))


def test_patterson_f3_exclude(hm4):
    hc = _complete_hm_over(hm4, ["C", "A", "B"])
    _agree(admixture.patterson_f3(hm4, "C", "A", "B", missing_data="exclude"),
           admixture.patterson_f3(hc, "C", "A", "B"))


def test_patterson_f4_exclude(hm4):
    hc = _complete_hm_over(hm4, ["A", "B", "C", "D"])
    _agree(admixture.patterson_f4(hm4, "A", "B", "C", "D", missing_data="exclude"),
           admixture.patterson_f4(hc, "A", "B", "C", "D"))


def test_admixture_empty_after_exclude_is_defined(hm4):
    """When exclude removes every site the empty-set guard returns a defined
    value (NaN or 0), not a crash."""
    hap = _host(hm4.haplotypes).copy()
    hap[0, :] = -1  # one gamete missing at every site -> no complete site
    m = HaplotypeMatrix(hap, _host(hm4.positions), hm4.chrom_start,
                        hm4.chrom_end)
    m.sample_sets = dict(hm4.sample_sets)
    num, den = admixture.patterson_d(m, "A", "B", "C", "D",
                                     missing_data="exclude")
    assert _host(num).size == 0 or np.all(np.isnan(_host(num))) \
        or np.all(_host(num) == 0)


# ── remaining single-population and windowed statistics ─────────────────
def test_daf_histogram_exclude(hm, hm_complete):
    # Returns (counts, bin_edges); compare both.
    exc = diversity.daf_histogram(hm, missing_data="exclude")
    inc = diversity.daf_histogram(hm_complete)
    for a, b in zip(exc, inc):
        _agree(a, b)


def test_joint_sfs_folded_exclude(hm, hm_complete):
    _agree(sfs.joint_sfs_folded(hm, "p1", "p2", missing_data="exclude"),
           sfs.joint_sfs_folded(hm_complete, "p1", "p2"))


def test_moving_garud_h_exclude(hm, hm_complete):
    _agree(selection.moving_garud_h(hm, size=50_000, missing_data="exclude"),
           selection.moving_garud_h(hm_complete, size=50_000))


def test_pbs_exclude(hm4):
    hc = _complete_hm_over(hm4, ["A", "B", "C"])
    _agree(divergence.pbs(hm4, "A", "B", "C", window_size=50_000,
                          missing_data="exclude"),
           divergence.pbs(hc, "A", "B", "C", window_size=50_000))
