"""
Tests for pg_gpu.moments_ld integration layer.

Validates that pg_gpu produces the same LD and heterozygosity statistics as
moments, for both estimators: the phased haplotype estimator
(``use_genotypes=False``) and the unphased genotype estimator
(``use_genotypes=True``, which is what ``compute_ld_statistics`` defaults to).
The two are different estimators of the same quantities, so each side must be
run with the same setting as the other -- a mismatch silently compares apples to
oranges rather than failing.

Requires the 'moments' pixi environment: pixi run -e moments pytest tests/test_moments_ld.py

Missing data is deliberately not covered on the genotype path. moments builds its
0/1/2 dosages with ``to_n_alt()`` and no ``fill=`` argument, and scikit-allel
defaults to ``fill=0``, so a missing call silently becomes homozygous reference;
pg_gpu drops it from the per-pair sample count instead. The two are not
comparable once any genotype is missing, so a parity assertion there would be
pinning the wrong behavior.
"""

import os
import tempfile

import pytest
import numpy as np

# Skip entire module if moments LD is not available
try:
    import moments.LD
except (ImportError, AttributeError):
    pytest.skip("moments.LD not available (use pixi -e moments)",
                allow_module_level=True)

from pg_gpu import GenotypeMatrix
from pg_gpu.moments_ld import (
    compute_ld_statistics,
    _compute_heterozygosity,
    _interpolate_genetic_distances,
)
from pg_gpu.haplotype_matrix import HaplotypeMatrix, _ld_names, _het_names


VCF = "examples/data/im-parsing-example.vcf"
POP_FILE = "examples/data/im_pop.txt"
POPS = ["deme0", "deme1"]
BP_BINS = np.logspace(2, 6, 6)


@pytest.fixture(scope="module")
def biallelic01_vcf(tmp_path_factory):
    """The example VCF filtered to is_biallelic_01 records (present alleles
    exactly {0,1}). On this subset pg_gpu's single mode (two present alleles,
    any coding) and moments' is_biallelic_01 keep the same sites, so the two
    match -- the equivalence baseline the broad parity tests below assert.
    """
    import allel
    callset = allel.read_vcf(VCF)
    mask = allel.GenotypeArray(
        callset['calldata/GT']).count_alleles().is_biallelic_01()
    out = os.path.join(str(tmp_path_factory.mktemp("bi01")), "bi01.vcf")
    di = 0
    with open(VCF) as fin, open(out, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
            else:
                if bool(mask[di]):
                    fout.write(line)
                di += 1
    return out


@pytest.fixture(scope="module")
def moments_stats():
    """Compute moments reference stats once for the module."""
    return moments.LD.Parsing.compute_ld_statistics(
        VCF, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=False, report=False,
    )


@pytest.fixture(scope="module")
def gpu_stats(biallelic01_vcf):
    """Compute pg_gpu stats once for the module.

    Runs on the is_biallelic_01 subset so the site set matches moments (which
    filters there internally); pg_gpu's single mode would otherwise keep the
    two-present-{0,2}/{1,2} sites moments drops. The haplotype estimator
    (use_genotypes=False) matches the moments fixtures above.
    """
    return compute_ld_statistics(
        biallelic01_vcf, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=False, report=False,
    )


@pytest.fixture(scope="module")
def moments_stats_geno():
    """moments reference for the unphased genotype estimator."""
    return moments.LD.Parsing.compute_ld_statistics(
        VCF, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=True, report=False,
    )


@pytest.fixture(scope="module")
def gpu_stats_geno(biallelic01_vcf):
    """pg_gpu stats for the unphased genotype estimator (is_biallelic_01 subset,
    to match moments' site set)."""
    return compute_ld_statistics(
        biallelic01_vcf, pop_file=POP_FILE, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=True, report=False,
    )


class TestOutputFormat:
    """Verify the output dict has the correct structure."""

    def test_keys(self, gpu_stats):
        assert set(gpu_stats.keys()) == {'bins', 'sums', 'stats', 'pops'}

    def test_pops(self, gpu_stats):
        assert gpu_stats['pops'] == POPS

    def test_stats_names(self, gpu_stats):
        ld_names, het_names = gpu_stats['stats']
        assert ld_names == _ld_names(2)
        assert het_names == _het_names(2)

    def test_bins_count(self, gpu_stats):
        assert len(gpu_stats['bins']) == len(BP_BINS) - 1

    def test_sums_count(self, gpu_stats):
        # One array per LD bin + one for heterozygosity
        assert len(gpu_stats['sums']) == len(BP_BINS) - 1 + 1

    def test_ld_sums_shape(self, gpu_stats):
        for i in range(len(BP_BINS) - 1):
            assert gpu_stats['sums'][i].shape == (15,)

    def test_het_sums_shape(self, gpu_stats):
        assert gpu_stats['sums'][-1].shape == (3,)


class TestLDStatistics:
    """Verify LD statistics match moments at machine precision."""

    def test_ld_bins_match(self, moments_stats, gpu_stats):
        for m_bin, g_bin in zip(moments_stats['bins'], gpu_stats['bins']):
            assert np.isclose(m_bin[0], g_bin[0])
            assert np.isclose(m_bin[1], g_bin[1])

    def test_ld_sums_match(self, moments_stats, gpu_stats):
        for i in range(len(moments_stats['bins'])):
            m = moments_stats['sums'][i]
            g = gpu_stats['sums'][i]
            np.testing.assert_allclose(g, m, rtol=1e-6,
                err_msg=f"LD sums mismatch in bin {i}")

    def test_het_sums_match(self, moments_stats, gpu_stats):
        m = moments_stats['sums'][-1]
        g = gpu_stats['sums'][-1]
        np.testing.assert_allclose(g, m, rtol=1e-6,
            err_msg="Heterozygosity sums mismatch")


class TestHeterozygosity:
    """Verify heterozygosity computation independently."""

    def test_within_pop_positive(self, gpu_stats):
        het = gpu_stats['sums'][-1]
        assert het[0] > 0  # H_0_0
        assert het[2] > 0  # H_1_1

    def test_cross_pop_positive(self, gpu_stats):
        het = gpu_stats['sums'][-1]
        assert het[1] > 0  # H_0_1

    def test_cross_between_within(self, gpu_stats):
        """Cross-pop het should be between within-pop values for diverged pops."""
        H_0_0, H_0_1, H_1_1 = gpu_stats['sums'][-1]
        assert H_0_1 >= min(H_0_0, H_1_1)


class TestPopAssignmentAlias:
    """``pop_file`` mirrors the moments kwarg name; ``pop_assignment``
    matches the rest of pg_gpu. The wrapper accepts both and routes
    them through the same code path."""

    def test_pop_assignment_matches_pop_file(self, gpu_stats, biallelic01_vcf):
        # Match the gpu_stats fixture's estimator and input so the two are
        # comparable.
        alias = compute_ld_statistics(
            biallelic01_vcf, pop_assignment=POP_FILE, pops=POPS,
            bp_bins=BP_BINS, use_genotypes=False, report=False,
        )
        # Same VCF + same pop file -> identical structure (bins,
        # stats, pops) and per-bin sums equal to machine precision.
        # CUDA reductions reorder additions across runs so two
        # otherwise-identical invocations can differ by a few ULPs.
        assert alias["bins"] == gpu_stats["bins"]
        assert alias["pops"] == gpu_stats["pops"]
        assert alias["stats"] == gpu_stats["stats"]
        for a, b in zip(alias["sums"], gpu_stats["sums"]):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)

    def test_both_aliases_rejected(self):
        with pytest.raises(TypeError, match="pop_file or pop_assignment"):
            compute_ld_statistics(
                VCF, pop_file=POP_FILE, pop_assignment=POP_FILE,
                pops=POPS, bp_bins=BP_BINS, report=False,
            )


@pytest.mark.slow
class TestTwoPopGenotypeLD:
    """The unphased genotype estimator on the same IM dataset.

    Marked slow because moments' genotype pair loop over this VCF takes ~75 s;
    the haplotype fixtures above cost about the same, so running both by default
    would double the module's wall time.
    """

    def test_stats_names(self, gpu_stats_geno):
        ld_names, het_names = gpu_stats_geno['stats']
        assert ld_names == _ld_names(2)
        assert het_names == _het_names(2)

    def test_ld_bins_match(self, moments_stats_geno, gpu_stats_geno):
        for m_bin, g_bin in zip(moments_stats_geno['bins'], gpu_stats_geno['bins']):
            assert np.isclose(m_bin[0], g_bin[0])
            assert np.isclose(m_bin[1], g_bin[1])

    def test_ld_sums_match(self, moments_stats_geno, gpu_stats_geno):
        for i in range(len(moments_stats_geno['bins'])):
            np.testing.assert_allclose(
                gpu_stats_geno['sums'][i], moments_stats_geno['sums'][i], rtol=1e-6,
                err_msg=f"genotype LD sums mismatch in bin {i}")

    def test_het_sums_match(self, moments_stats_geno, gpu_stats_geno):
        np.testing.assert_allclose(
            gpu_stats_geno['sums'][-1], moments_stats_geno['sums'][-1], rtol=1e-6,
            err_msg="genotype heterozygosity sums mismatch")

    def test_means_match_moments(self, moments_stats_geno, gpu_stats_geno):
        means_m = moments.LD.Parsing.means_from_region_data(
            {0: moments_stats_geno}, moments_stats_geno['stats'])
        means_g = moments.LD.Parsing.means_from_region_data(
            {0: gpu_stats_geno}, gpu_stats_geno['stats'])
        for mm, mg in zip(means_m, means_g):
            np.testing.assert_allclose(mg, mm, rtol=1e-6)

    def test_differs_from_haplotype_estimator(self, gpu_stats, gpu_stats_geno):
        """The two estimators are genuinely different, so a fixture that
        silently ran the wrong one would not go unnoticed."""
        assert not np.allclose(gpu_stats_geno['sums'][0], gpu_stats['sums'][0])


class TestMomentsCompatibility:
    """Verify output can be fed into moments downstream functions."""

    def test_means_from_region_data(self, gpu_stats):
        """moments.LD.Parsing.means_from_region_data should accept our output."""
        all_data = {0: gpu_stats}
        means = moments.LD.Parsing.means_from_region_data(
            all_data, gpu_stats['stats'])
        assert len(means) == len(gpu_stats['bins']) + 1
        for m in means:
            assert isinstance(m, np.ndarray)
            assert np.all(np.isfinite(m))

    def test_means_match_moments(self, moments_stats, gpu_stats):
        """Normalized means should match between moments and pg_gpu."""
        means_m = moments.LD.Parsing.means_from_region_data(
            {0: moments_stats}, moments_stats['stats'])
        means_g = moments.LD.Parsing.means_from_region_data(
            {0: gpu_stats}, gpu_stats['stats'])
        for mm, mg in zip(means_m, means_g):
            np.testing.assert_allclose(mg, mm, rtol=1e-6)


# ---------------------------------------------------------------------------
# Multi-population integration tests (3-pop, 4-pop)
# ---------------------------------------------------------------------------

def _simulate_multipop_vcf(n_pops, n_samples=8, seq_len=30_000, seed=42):
    """Simulate a multi-population VCF and pop file using msprime."""
    import msprime

    demography = msprime.Demography()
    for i in range(n_pops):
        demography.add_population(name=f"pop{i}", initial_size=1000)
    # Chain of splits: pop{n-1} splits from pop{n-2} at time 500*(n-1-i)
    if n_pops >= 2:
        demography.add_population(name="anc01", initial_size=2000)
        demography.add_population_split(
            time=500, derived=["pop0", "pop1"], ancestral="anc01")
    if n_pops >= 3:
        demography.add_population(name="anc012", initial_size=2000)
        demography.add_population_split(
            time=1000, derived=["anc01", "pop2"], ancestral="anc012")
    if n_pops >= 4:
        demography.add_population(name="anc0123", initial_size=2000)
        demography.add_population_split(
            time=1500, derived=["anc012", "pop3"], ancestral="anc0123")

    samples = {}
    for i in range(n_pops):
        samples[f"pop{i}"] = n_samples

    ts = msprime.sim_ancestry(
        samples=samples, demography=demography,
        sequence_length=seq_len, recombination_rate=1e-8,
        random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=seed)

    vcf_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.vcf', delete=False)
    pop_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False)

    # Write VCF
    with open(vcf_file.name, 'w') as f:
        ts.write_vcf(f)

    # Write pop file
    pops_list = [f"pop{i}" for i in range(n_pops)]
    with open(pop_file.name, 'w') as f:
        f.write("sample\tpop\n")
        for ind in ts.individuals():
            pop_name = ts.population(ind.population).metadata.get(
                'name', f"pop{ind.population}")
            f.write(f"tsk_{ind.id}\t{pop_name}\n")

    return vcf_file.name, pop_file.name, pops_list


def _unlink_quietly(*paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def _multipop_data(n_pops, use_genotypes):
    """Simulate n-population data and compute both moments and pg_gpu stats
    under the same estimator."""
    vcf, pop_file, pops = _simulate_multipop_vcf(n_pops)
    bp_bins = np.array([0, 1000, 5000, 15000, 30000], dtype=np.float64)
    try:
        m_stats = moments.LD.Parsing.compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, use_genotypes=use_genotypes, report=False)
        g_stats = compute_ld_statistics(
            vcf, pop_file=pop_file, pops=pops,
            bp_bins=bp_bins, use_genotypes=use_genotypes, report=False)
        yield m_stats, g_stats, pops
    finally:
        # moments caches the parsed VCF alongside it as <stem>.h5 and reuses
        # that file on a later run, so leaving it behind would let a stale
        # cache shadow a regenerated VCF.
        _unlink_quietly(vcf, pop_file, vcf.split(".vcf")[0] + ".h5")


@pytest.fixture(scope="module", params=[False, True], ids=["hap", "geno"])
def three_pop_data(request):
    yield from _multipop_data(3, request.param)


@pytest.fixture(scope="module", params=[False, True], ids=["hap", "geno"])
def four_pop_data(request):
    yield from _multipop_data(4, request.param)


class TestThreePopLD:
    """Verify 3-population LD statistics match moments."""

    def test_output_format(self, three_pop_data):
        _, g, pops = three_pop_data
        assert g['pops'] == pops
        ld_names, het_names = g['stats']
        assert len(ld_names) == 45
        assert len(het_names) == 6
        assert ld_names == _ld_names(3)
        assert het_names == _het_names(3)

    def test_ld_sums_match(self, three_pop_data):
        m, g, _ = three_pop_data
        for i in range(len(m['bins'])):
            np.testing.assert_allclose(
                g['sums'][i], m['sums'][i], rtol=1e-6,
                err_msg=f"3-pop LD sums mismatch in bin {i}")

    def test_het_sums_match(self, three_pop_data):
        m, g, _ = three_pop_data
        np.testing.assert_allclose(
            g['sums'][-1], m['sums'][-1], rtol=1e-6,
            err_msg="3-pop heterozygosity mismatch")

    def test_moments_compatibility(self, three_pop_data):
        _, g, _ = three_pop_data
        means = moments.LD.Parsing.means_from_region_data(
            {0: g}, g['stats'])
        assert len(means) == len(g['bins']) + 1
        for m in means:
            assert np.all(np.isfinite(m))


class TestFourPopLD:
    """Verify 4-population LD statistics match moments."""

    def test_output_format(self, four_pop_data):
        _, g, pops = four_pop_data
        assert g['pops'] == pops
        ld_names, het_names = g['stats']
        assert len(ld_names) == 105
        assert len(het_names) == 10
        assert ld_names == _ld_names(4)
        assert het_names == _het_names(4)

    def test_ld_sums_match(self, four_pop_data):
        m, g, _ = four_pop_data
        for i in range(len(m['bins'])):
            np.testing.assert_allclose(
                g['sums'][i], m['sums'][i], rtol=1e-6,
                err_msg=f"4-pop LD sums mismatch in bin {i}")

    def test_het_sums_match(self, four_pop_data):
        m, g, _ = four_pop_data
        np.testing.assert_allclose(
            g['sums'][-1], m['sums'][-1], rtol=1e-6,
            err_msg="4-pop heterozygosity mismatch")

    def test_moments_compatibility(self, four_pop_data):
        _, g, _ = four_pop_data
        means = moments.LD.Parsing.means_from_region_data(
            {0: g}, g['stats'])
        assert len(means) == len(g['bins']) + 1
        for m in means:
            assert np.all(np.isfinite(m))


# ---------------------------------------------------------------------------
# Which sites the genotype path keeps
# ---------------------------------------------------------------------------

SMALL_VCF_SAMPLES = 16
SMALL_VCF_POSITIONS = list(range(100, 100 + 200 * 16, 200))
SMALL_VCF_POPS = ["popA", "popB"]
SMALL_VCF_BINS = np.array([100, 500, 1500, 3200], dtype=np.float64)
# The site whose two observed alleles are 0 and 2 rather than 0 and 1.
ALT_CODED_POSITION = SMALL_VCF_POSITIONS[7]


def _to_numpy(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def _write_small_vcf(directory, alt_coded):
    """Write a phased diploid VCF of biallelic sites, optionally including one
    site whose two observed alleles are 0 and 2 instead of 0 and 1.

    Such a site is biallelic in the sample -- allele 1 is declared in ALT but
    never called -- yet its allele codes are not {0, 1}. Mutation models that
    can revert or re-hit a site produce these at a low rate, so whether any
    given simulated dataset contains one is luck. Writing it by hand makes the
    case deterministic.
    """
    rng = np.random.default_rng(20260806)
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1,length=5000>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"s{i}" for i in range(SMALL_VCF_SAMPLES)),
    ]
    rows = []
    for pos in SMALL_VCF_POSITIONS:
        use_alt2 = alt_coded and pos == ALT_CODED_POSITION
        alt_field, alt_code = ("C,G", 2) if use_alt2 else ("C", 1)
        # Redraw until the site segregates; a monomorphic site is dropped by
        # both sides and would leave the comparison with nothing to say.
        while True:
            haps = rng.binomial(1, 0.4, size=(SMALL_VCF_SAMPLES, 2))
            if 0 < haps.sum() < 2 * SMALL_VCF_SAMPLES:
                break
        calls = "\t".join(f"{a * alt_code}|{b * alt_code}" for a, b in haps)
        rows.append(f"1\t{pos}\t.\tA\t{alt_field}\t.\tPASS\t.\tGT\t{calls}")

    suffix = "alt_coded" if alt_coded else "plain"
    vcf_path = os.path.join(directory, f"{suffix}.vcf")
    pop_path = os.path.join(directory, f"{suffix}_pops.txt")
    with open(vcf_path, "w") as f:
        f.write("\n".join(header + rows) + "\n")
    with open(pop_path, "w") as f:
        f.write("sample\tpop\n")
        for i in range(SMALL_VCF_SAMPLES):
            pop = SMALL_VCF_POPS[0] if i < SMALL_VCF_SAMPLES // 2 else SMALL_VCF_POPS[1]
            f.write(f"s{i}\t{pop}\n")
    return vcf_path, pop_path


@pytest.fixture(scope="module")
def alt_coded_vcf(tmp_path_factory):
    return _write_small_vcf(str(tmp_path_factory.mktemp("alt_coded")), alt_coded=True)


@pytest.fixture(scope="module")
def plain_vcf(tmp_path_factory):
    return _write_small_vcf(str(tmp_path_factory.mktemp("plain")), alt_coded=False)


def _moments_and_gpu_stats(vcf_fixture, use_genotypes):
    """moments and pg_gpu LD stats for a (vcf, pop_file) fixture, same estimator."""
    vcf, pop_file = vcf_fixture
    m_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf, pop_file=pop_file, pops=SMALL_VCF_POPS,
        bp_bins=SMALL_VCF_BINS, use_genotypes=use_genotypes, report=False)
    g_stats = compute_ld_statistics(
        vcf, pop_file=pop_file, pops=SMALL_VCF_POPS,
        bp_bins=SMALL_VCF_BINS, use_genotypes=use_genotypes, report=False)
    return m_stats, g_stats


def _moments_positions(vcf):
    """Positions moments keeps (its is_biallelic_01 input filter)."""
    positions, *_ = moments.LD.Parsing.get_genotypes(vcf, use_h5=False, report=False)
    return set(np.asarray(positions).tolist())


class TestAltCodedSiteParity:
    """The plain and alt-coded fixtures are built from the same RNG draws and
    differ only in one site's coding: {0,1} in plain, {0,2} in alt-coded (allele
    1 declared but never called). moments' is_biallelic_01 drops the {0,2} site;
    pg_gpu's single mode keeps it, recoded. So the two agree on the plain ({0,1})
    fixture and disagree on the alt-coded one, and the disagreement is
    attributable to that single site.

    A single extra site is easy to overlook: it shifts DD and pi2, which are
    sums of large same-sign terms, by well under a percent. Dz is signed and
    cancels heavily, so the same site moves it by whole percent.
    """

    def test_fixtures_differ_only_at_the_alt_coded_site(self, plain_vcf,
                                                        alt_coded_vcf):
        # The {0,2} site's alt-dosage equals the {0,1} labelling, so the two
        # fixtures load to an identical dosage matrix: they agree on every {0,1}
        # allele and differ only in that site's coding. moments keeps the same
        # sites from both, minus the {0,2} site it drops; pg_gpu keeps it.
        gp = GenotypeMatrix.from_vcf(plain_vcf[0])
        ga = GenotypeMatrix.from_vcf(alt_coded_vcf[0])
        np.testing.assert_array_equal(
            _to_numpy(gp.positions), _to_numpy(ga.positions))
        np.testing.assert_array_equal(
            _to_numpy(gp.genotypes), _to_numpy(ga.genotypes))
        assert ALT_CODED_POSITION in set(_to_numpy(ga.positions).tolist())
        m_plain = _moments_positions(plain_vcf[0])
        m_alt = _moments_positions(alt_coded_vcf[0])
        assert ALT_CODED_POSITION in m_plain
        assert m_alt == m_plain - {ALT_CODED_POSITION}

    @pytest.mark.parametrize("use_genotypes", [False, True], ids=["hap", "geno"])
    def test_agree_on_plain_disagree_on_alt_coded(self, plain_vcf, alt_coded_vcf,
                                                  use_genotypes):
        # (c) On the {0,1} fixture pg_gpu matches moments for every statistic...
        m, g = _moments_and_gpu_stats(plain_vcf, use_genotypes)
        for i in range(len(m['sums'])):
            np.testing.assert_allclose(
                g['sums'][i], m['sums'][i], rtol=1e-6,
                err_msg=f"plain fixture mismatch in sums[{i}]")
        # ...(b) but on the alt-coded fixture pg_gpu keeps the {0,2} site moments
        # drops, so at least one LD bin disagrees.
        m, g = _moments_and_gpu_stats(alt_coded_vcf, use_genotypes)
        assert any(
            not np.allclose(g['sums'][i], m['sums'][i], rtol=1e-6)
            for i in range(len(m['bins'])))


# ---------------------------------------------------------------------------
# The haplotype_matrix= entry point on the genotype estimator
# ---------------------------------------------------------------------------


class TestHaplotypeMatrixGenotypePath:
    """``compute_ld_statistics(haplotype_matrix=..., use_genotypes=True)`` goes
    through ``GenotypeMatrix.from_haplotype_matrix``, so the conversion has to
    recover the same individuals the VCF loader would have built.

    Haplotype rows are ordered so that sample ``i`` owns rows ``2i`` and
    ``2i + 1``, and every loader emits that order, so one VCF loaded two ways
    yields the same individuals.

    Getting it wrong builds each "individual" from two different people's
    chromosomes and scrambles the population assignment with it, which no
    downstream statistic can detect.
    """

    def test_conversion_matches_vcf_genotypes(self, plain_vcf):
        vcf, _ = plain_vcf
        direct = GenotypeMatrix.from_vcf(vcf)
        converted = GenotypeMatrix.from_haplotype_matrix(HaplotypeMatrix.from_vcf(vcf))
        np.testing.assert_array_equal(
            _to_numpy(converted.positions), _to_numpy(direct.positions))
        np.testing.assert_array_equal(
            _to_numpy(converted.genotypes), _to_numpy(direct.genotypes))

    def test_conversion_preserves_sample_sets(self, plain_vcf):
        vcf, pop_file = plain_vcf
        direct = GenotypeMatrix.from_vcf(vcf)
        direct.load_pop_file(pop_file, pops=SMALL_VCF_POPS)
        hm = HaplotypeMatrix.from_vcf(vcf)
        hm.load_pop_file(pop_file, pops=SMALL_VCF_POPS)
        converted = GenotypeMatrix.from_haplotype_matrix(hm)
        for pop in SMALL_VCF_POPS:
            assert sorted(converted.sample_sets[pop]) == sorted(direct.sample_sets[pop])

    def test_ld_sums_match_the_vcf_path(self, plain_vcf):
        vcf, pop_file = plain_vcf
        hm = HaplotypeMatrix.from_vcf(vcf)
        hm.load_pop_file(pop_file, pops=SMALL_VCF_POPS)
        from_hap = compute_ld_statistics(
            pops=SMALL_VCF_POPS, bp_bins=SMALL_VCF_BINS, report=False,
            haplotype_matrix=hm, use_genotypes=True)
        from_vcf = compute_ld_statistics(
            vcf, pop_file=pop_file, pops=SMALL_VCF_POPS,
            bp_bins=SMALL_VCF_BINS, use_genotypes=True, report=False)
        for i in range(len(from_vcf['bins'])):
            np.testing.assert_allclose(
                from_hap['sums'][i], from_vcf['sums'][i], rtol=1e-6,
                err_msg=f"haplotype_matrix= path differs from the VCF path in bin {i}")
        np.testing.assert_allclose(
            from_hap['sums'][-1], from_vcf['sums'][-1], rtol=1e-6,
            err_msg="haplotype_matrix= path differs from the VCF path on heterozygosity")
