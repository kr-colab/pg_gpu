"""
Shared pytest fixtures and configuration for pg_gpu tests.
"""

import pytest
import numpy as np
import tempfile
import os


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return  # run everything
    skip_slow = pytest.mark.skip(reason="needs --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def simple_vcf_file():
    """Create a simple VCF file for testing."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t1|0"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
        f.write(vcf_content)
        vcf_path = f.name

    yield vcf_path

    # Cleanup
    if os.path.exists(vcf_path):
        os.unlink(vcf_path)


@pytest.fixture
def two_population_vcf():
    """Create a VCF file with samples from two populations."""
    vcf_content = """##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tind0\tind1\tind2\tind3
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t1|1\t0|1\t1|0\t0|0
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t1|0\t1|0\t1|1\t0|0
1\t500\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t1|1\t0|1\t1|0"""

    pop_content = """sample\tpop
ind0\tpop0
ind1\tpop0
ind2\tpop1
ind3\tpop1"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
        f.write(vcf_content)
        vcf_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(pop_content)
        pop_file = f.name

    yield vcf_path, pop_file

    # Cleanup
    for path in [vcf_path, pop_file]:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def sample_haplotype_counts():
    """Sample haplotype counts for testing statistics calculations."""
    # Single population counts
    single_pop = np.array([
        [10, 5, 3, 2],   # variant pair 1
        [8, 6, 4, 2],    # variant pair 2
        [12, 3, 3, 2],   # variant pair 3
    ])

    # Two population counts
    pop1_counts = np.array([
        [5, 3, 1, 1],
        [4, 3, 2, 1],
        [6, 2, 1, 1],
    ])

    pop2_counts = np.array([
        [5, 2, 2, 1],
        [4, 3, 2, 1],
        [6, 1, 2, 1],
    ])

    return {
        'single': single_pop,
        'pop1': pop1_counts,
        'pop2': pop2_counts
    }


def bgzip_index(vcf_path):
    """bgzip ``vcf_path`` in place and tabix-index it; returns the ``.gz``
    path. Skips the calling test when the htslib tools are not on PATH."""
    import shutil
    import subprocess
    if shutil.which("bgzip") is None or shutil.which("tabix") is None:
        pytest.skip("bgzip/tabix not on PATH")
    subprocess.run(["bgzip", "-f", vcf_path], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", vcf_path + ".gz"], check=True)
    return vcf_path + ".gz"


def simulate_hm(n_samples=20, seq_length=100_000, seed=42,
                mutation_model=None):
    """Build a small msprime-derived HaplotypeMatrix for tests.

    Centralizes what would otherwise be repeated per-test simulation
    boilerplate. ``mutation_model=None`` lets msprime pick its default
    (Jukes-Cantor, which can produce triallelic sites);
    ``mutation_model='binary'`` forces biallelic, which is what the
    pg_gpu-vs-allel parity tests want because pg_gpu's pi formula has
    a known multi-allelic gap (kr-colab/pg_gpu#100).
    """
    import msprime
    from pg_gpu import HaplotypeMatrix

    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=seq_length,
        recombination_rate=1e-4, random_seed=seed, ploidy=2,
    )
    kw = {}
    if mutation_model is not None:
        kw["model"] = mutation_model
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=seed, **kw)
    return HaplotypeMatrix.from_ts(ts)


@pytest.fixture(scope="session")
def multiallelic_ts():
    """JC69 tree sequence with genuine multiallelic sites (issue #100).

    Recurrent Jukes-Cantor mutation produces tri/tetra-allelic and
    reference-absent sites, which is what the tskit-oracle multiallelic tests
    need. Deterministic (fixed seeds). No missing data, so per-site n_valid
    equals the full sample size.
    """
    import msprime
    ts = msprime.sim_ancestry(
        samples=25, population_size=2e4, sequence_length=1e5,
        recombination_rate=1e-8, random_seed=42, ploidy=2,
    )
    ts = msprime.sim_mutations(
        ts, rate=1e-6, model=msprime.JC69(state_independent=True),
        random_seed=43,
    )
    return ts


@pytest.fixture
def multiallelic_hm(multiallelic_ts):
    """(ts, hm) pair for the multiallelic tree sequence; hm on GPU."""
    from pg_gpu import HaplotypeMatrix
    return multiallelic_ts, HaplotypeMatrix.from_ts(multiallelic_ts, device="GPU")


def canonical_hap_rows(call_genotype):
    """Reference (n_hap, n_var) rows for a ``(n_var, n_dip, 2)`` block.

    Spelled as explicit per-ploidy column slices rather than a reshape so it
    stays an independent statement of the canonical row order -- sample i at
    rows 2i and 2i+1 -- rather than a mirror of the loader implementation.
    """
    import numpy as np

    n_var, n_dip, _ = call_genotype.shape
    haps = np.empty((n_var, 2 * n_dip), dtype=call_genotype.dtype)
    haps[:, 0::2] = call_genotype[:, :, 0]
    haps[:, 1::2] = call_genotype[:, :, 1]
    return haps.T
