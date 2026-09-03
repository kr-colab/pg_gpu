"""population= subset paths across the statistics modules.

A statistic run with ``population='p1'`` restricts to that population's rows
before computing, so it must equal the same statistic run on a matrix
hand-built from only those rows (population=None). One two-population fixture
drives the sweep; the hand-built subset is independent of the internal
``get_population_matrix``, so the equivalence actually checks its row
selection.
"""
import cupy as cp
import msprime
import numpy as np
import pytest

from pg_gpu import (HaplotypeMatrix, GenotypeMatrix, diversity, selection,
                    decomposition, divergence)


def _host(a):
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def _two_pop_hm(seed=5, n_per_pop=10, seq_length=500_000):
    """Two named populations, biallelic, fully called."""
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
    hm.sample_sets = dict(hm.sample_sets)
    return hm


def _pop_subset_hm(hm, pop):
    """A HaplotypeMatrix of only ``pop``'s rows, with no sample_sets -- the
    independent reference for ``population=pop``."""
    rows = hm.sample_sets[pop]
    hap = _host(hm.haplotypes)[rows, :]
    return HaplotypeMatrix(hap, _host(hm.positions), hm.chrom_start,
                           hm.chrom_end)


def _agree(a, b):
    np.testing.assert_allclose(_host(a), _host(b), rtol=1e-9, atol=1e-12,
                               equal_nan=True)


@pytest.fixture(scope="module")
def hm():
    return _two_pop_hm()


@pytest.fixture(scope="module")
def hm_p1(hm):
    return _pop_subset_hm(hm, "p1")


# Each entry is (name, f) where f(matrix, population) issues the call.
SINGLE_POP = [
    ("garud_h", lambda m, pop: selection.garud_h(m, population=pop)),
    ("moving_garud_h",
     lambda m, pop: selection.moving_garud_h(m, size=100_000, population=pop)),
    ("nsl", lambda m, pop: selection.nsl(m, population=pop)),
    ("ihs", lambda m, pop: selection.ihs(m, population=pop)),
    ("ehh_decay", lambda m, pop: selection.ehh_decay(m, population=pop)),
    ("mu_var", lambda m, pop: diversity.mu_var(m, population=pop)),
    ("mu_sfs", lambda m, pop: diversity.mu_sfs(m, population=pop)),
    ("pairwise_distance",
     lambda m, pop: decomposition.pairwise_distance(m, population=pop)),
]


@pytest.mark.parametrize("name,fn", SINGLE_POP, ids=[n for n, _ in SINGLE_POP])
def test_population_subset_matches_hand_subset(name, fn, hm, hm_p1):
    _agree(fn(hm, "p1"), fn(hm_p1, None))


# ── decomposition: windowed PCA (eigenvalues are rotation/sign invariant) ──
def test_local_pca_population_subset(hm, hm_p1):
    a = decomposition.local_pca(hm, window_size=50, window_type="snp", k=2,
                                population="p1")
    b = decomposition.local_pca(hm_p1, window_size=50, window_type="snp", k=2)
    _agree(a.eigvals, b.eigvals)
    # Eigenvalues are permutation-invariant; |eigenvectors| are not (each
    # sample's component moves), so this also catches a subset that selects
    # the right rows in the wrong order, immune to the per-column sign.
    _agree(np.abs(a.eigvecs), np.abs(b.eigvecs))


def test_local_pca_jackknife_population_subset(hm, hm_p1):
    a = decomposition.local_pca_jackknife(hm, window_size=50, window_type="snp",
                                          k=2, n_blocks=5, population="p1")
    b = decomposition.local_pca_jackknife(hm_p1, window_size=50,
                                          window_type="snp", k=2, n_blocks=5)
    _agree(a, b)


def test_lostruct_population_subset(hm, hm_p1):
    a = decomposition.lostruct(hm, window_size=50, window_type="snp", k=2,
                               population="p1")
    b = decomposition.lostruct(hm_p1, window_size=50, window_type="snp", k=2)
    _agree(a.local_pca.eigvals, b.local_pca.eigvals)


# ── diploid GenotypeMatrix population= (diplotype spectrum, DAF histogram) ──
@pytest.fixture(scope="module")
def gm(hm):
    g = GenotypeMatrix.from_haplotype_matrix(hm)
    n1 = len(hm.sample_sets["p1"]) // 2  # haplotype rows -> individuals
    n_ind = _host(g.genotypes).shape[0]
    g.sample_sets = {"p1": list(range(n1)), "p2": list(range(n1, n_ind))}
    return g


@pytest.fixture(scope="module")
def gm_p1(gm):
    idx = gm.sample_sets["p1"]
    return GenotypeMatrix(_host(gm.genotypes)[idx, :], _host(gm.positions))


def test_diplotype_frequency_spectrum_population_subset(gm, gm_p1):
    a = diversity.diplotype_frequency_spectrum(gm, population="p1")
    b = diversity.diplotype_frequency_spectrum(gm_p1)
    _agree(a[0], b[0])
    assert a[1] == b[1]


def test_daf_histogram_diploid_population_subset(gm, gm_p1):
    a = diversity.daf_histogram(gm, population="p1")
    b = diversity.daf_histogram(gm_p1)
    for x, y in zip(a, b):
        _agree(x, y)


@pytest.mark.parametrize("fn", [
    lambda m: diversity.diplotype_frequency_spectrum(m, population="nope"),
    lambda m: diversity.daf_histogram(m, population="nope"),
], ids=["diplotype_frequency_spectrum", "daf_histogram"])
def test_diploid_population_not_found_raises(gm, fn):
    with pytest.raises(ValueError, match="not found"):
        fn(gm)


# ── divergence._get_population_indices edge lookups ─────────────────────
def test_population_indices_unknown_name_raises(hm):
    with pytest.raises(ValueError, match="not found"):
        divergence.dxy(hm, "p1", "nope")


def test_population_indices_accepts_row_list(hm):
    # A row-list argument takes the else branch (validated rows, not a name).
    _agree(divergence.dxy(hm, hm.sample_sets["p1"], hm.sample_sets["p2"]),
           divergence.dxy(hm, "p1", "p2"))
