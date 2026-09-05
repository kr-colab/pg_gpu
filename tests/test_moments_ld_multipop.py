"""Multi-population (>=3) LD moment dispatch against moments.LD.

The two-population suite in test_moments_ld.py exercises only the single- and
two-population index patterns. With four populations, _ld_names(4) enumerates
every multi-population DD/Dz/pi2 pattern the fused kernels dispatch on --
including the three-distinct-population Dz(i,j,k) and the four-distinct
pi2(i,j,k,l) -- so a four-population comparison drives all the per-pattern
branches of both fused-kernel families (haplotype_kernels / genotype_kernels)
at once and checks them against moments.LD. Four populations subsumes three:
_ld_names(4) contains the 3-distinct patterns too.

Requires the 'moments' pixi environment:
    pixi run -e moments pytest tests/test_moments_ld_multipop.py
"""
import os

import numpy as np
import pytest

# Skip the whole module if moments LD is not available.
try:
    import moments.LD
except ImportError:
    pytest.skip("moments.LD not available (use pixi -e moments)",
                allow_module_level=True)

from pg_gpu.moments_ld import compute_ld_statistics
from pg_gpu.haplotype_matrix import _ld_names

VCF = "examples/data/im-parsing-example.vcf"
POPS = ["p0", "p1", "p2", "p3"]
BP_BINS = np.logspace(2, 6, 6)


@pytest.fixture(scope="module")
def pop_file(tmp_path_factory):
    """Split the 20 example samples (tsk_0..tsk_19) into four populations
    (5 each). The assignment is arbitrary -- the test validates the
    multi-population moment formulas, not any biological structure."""
    path = os.path.join(str(tmp_path_factory.mktemp("pop4")), "pop4.txt")
    with open(path, "w") as f:
        f.write("sample\tpop\n")
        for i in range(20):
            f.write(f"tsk_{i}\tp{i // 5}\n")
    return path


@pytest.fixture(scope="module")
def biallelic01_vcf(tmp_path_factory):
    """The example VCF filtered to is_biallelic_01 records, so pg_gpu (which
    otherwise keeps recoded {0,2}/{1,2} sites) and moments keep the same site
    set. See test_moments_ld for the rationale."""
    import allel
    callset = allel.read_vcf(VCF)
    mask = allel.GenotypeArray(
        callset['calldata/GT']).count_alleles().is_biallelic_01()
    out = os.path.join(str(tmp_path_factory.mktemp("bi01_multi")), "bi01.vcf")
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


@pytest.fixture(scope="module", params=[False, True],
                ids=["haplotype", "genotype"])
def use_genotypes(request):
    # The phased (haplotype) and unphased (genotype) estimators dispatch to
    # separate fused kernels (haplotype_kernels vs genotype_kernels), each with
    # its own per-pattern multi-population branches, so both must be checked.
    return request.param


@pytest.fixture(scope="module")
def moments_stats(pop_file, use_genotypes):
    # moments filters to is_biallelic_01 internally, so it reads the raw VCF.
    return moments.LD.Parsing.compute_ld_statistics(
        VCF, pop_file=pop_file, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=use_genotypes, use_h5=False,
        report=False)


@pytest.fixture(scope="module")
def gpu_stats(biallelic01_vcf, pop_file, use_genotypes):
    return compute_ld_statistics(
        biallelic01_vcf, pop_file=pop_file, pops=POPS,
        bp_bins=BP_BINS, use_genotypes=use_genotypes, report=False)


def test_stats_names_are_multi_pop(gpu_stats):
    ld_names, _ = gpu_stats['stats']
    assert ld_names == _ld_names(4)
    # Patterns the two-population suite never reaches.
    assert "Dz_0_1_2" in ld_names        # three distinct populations
    assert "pi2_0_1_2_3" in ld_names     # four distinct (the alldiff branch)


def test_ld_bins_match(moments_stats, gpu_stats):
    for m_bin, g_bin in zip(moments_stats['bins'], gpu_stats['bins']):
        assert np.isclose(m_bin[0], g_bin[0])
        assert np.isclose(m_bin[1], g_bin[1])


def test_ld_sums_match_moments(moments_stats, gpu_stats):
    # The per-bin vector covers every multi-population index pattern the fused
    # kernels dispatch on (up to the four-distinct pi2 alldiff branch).
    for i in range(len(moments_stats['bins'])):
        np.testing.assert_allclose(
            gpu_stats['sums'][i], moments_stats['sums'][i], rtol=1e-6,
            err_msg=f"multi-pop LD sums mismatch in bin {i}")


def test_het_sums_match_moments(moments_stats, gpu_stats):
    np.testing.assert_allclose(
        gpu_stats['sums'][-1], moments_stats['sums'][-1], rtol=1e-6,
        err_msg="multi-pop heterozygosity sums mismatch")
