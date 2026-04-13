import pytest
import msprime
import allel
import cupy as cp
import numpy as np
import tempfile
import os

from pg_gpu.haplotype_matrix import HaplotypeMatrix

@pytest.fixture
def sample_vcf():
    """Create a temporary VCF file with simulated data for testing."""
    # Simulate some data
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=1000,
        recombination_rate=0.01,
        random_seed=42,
        ploidy=2
    )
    ts = msprime.sim_mutations(ts, rate=0.01)
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.vcf', delete=False) as tmp:
        # Write VCF to temporary file
        with open(tmp.name, 'w') as f:
            ts.write_vcf(f, allow_position_zero=True)
        yield tmp.name

    # Clean up the temporary file
    os.unlink(tmp.name)

@pytest.fixture
def sample_ts():
    """Create a sample tskit.TreeSequence for testing."""
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=1000,
        recombination_rate=0.01,
        ploidy=2,
        discrete_genome=False,
    )
    ts = msprime.sim_mutations(ts, rate=0.01, model="binary")
    return ts

def test_from_vcf(sample_vcf):
    """Test creating HaplotypeMatrix from VCF file."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    assert isinstance(hap_matrix, HaplotypeMatrix)
    # by default the arrays are on the CPU
    assert hap_matrix.device == 'CPU'
    assert isinstance(hap_matrix.get_matrix(), np.ndarray)  # Now expecting CuPy array
    assert isinstance(hap_matrix.get_positions(), np.ndarray)  # Now expecting CuPy array
    assert len(hap_matrix.get_positions()) > 0

def test_from_ts(sample_ts):
    """Test creating HaplotypeMatrix from tskit.TreeSequence."""
    hap_matrix = HaplotypeMatrix.from_ts(sample_ts)
    assert isinstance(hap_matrix, HaplotypeMatrix)
    # by default the arrays are on the CPU
    assert hap_matrix.device == 'CPU'
    assert isinstance(hap_matrix.get_matrix(), np.ndarray)  # Now expecting CuPy array
    assert isinstance(hap_matrix.get_positions(), np.ndarray)  # Now expecting CuPy array

    # make a GPU version
    hap_matrix_gpu = HaplotypeMatrix.from_ts(sample_ts, device='GPU')
    assert hap_matrix_gpu.device == 'GPU'
    assert isinstance(hap_matrix_gpu.get_matrix(), cp.ndarray)
    assert isinstance(hap_matrix_gpu.get_positions(), cp.ndarray)

    # Add test for pairwise_LD
    D = hap_matrix.pairwise_LD_v()
    assert isinstance(D, cp.ndarray)
    assert D.shape == (hap_matrix.num_variants, hap_matrix.num_variants)
    assert cp.allclose(D, D.T)  # Check symmetry
    assert cp.all(cp.abs(D) <= 0.25)  # D is bounded by ±0.25


def test_from_ts_default_population(sample_ts):
    """sim_ancestry(samples=N) with no explicit Demography -> msprime
    still names a default 'pop_0' population; hm.sample_sets exposes it."""
    hm = HaplotypeMatrix.from_ts(sample_ts)
    assert list(hm.sample_sets.keys()) == ["pop_0"]
    assert len(hm.sample_sets["pop_0"]) == hm.num_haplotypes


def test_from_ts_populates_sample_sets_from_demography():
    """Named populations in the tree sequence auto-register as sample_sets."""
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=10_000)
    demography.add_population(name="AB", initial_size=10_000)
    demography.add_population_split(time=5_000, derived=["A", "B"],
                                    ancestral="AB")
    ts = msprime.sim_ancestry(
        samples={"A": 12, "B": 8},
        sequence_length=100_000,
        recombination_rate=1e-8,
        demography=demography,
        random_seed=42, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)

    hm = HaplotypeMatrix.from_ts(ts)

    # Extant pops A and B should be present; the ancestral AB has no living
    # samples and should NOT appear.
    assert set(hm.sample_sets.keys()) == {"A", "B"}
    assert len(hm.sample_sets["A"]) == 2 * 12
    assert len(hm.sample_sets["B"]) == 2 * 8

    # Row indices must actually refer to samples drawn from that population
    # in the tree sequence, not just any rows.
    all_samples = list(ts.samples())
    a_nodes = set(int(s) for s in ts.samples(population=0))
    b_nodes = set(int(s) for s in ts.samples(population=1))
    assert set(all_samples[i] for i in hm.sample_sets["A"]) == a_nodes
    assert set(all_samples[i] for i in hm.sample_sets["B"]) == b_nodes


def test_from_ts_user_can_overwrite_sample_sets():
    """Callers who want custom groupings can still reassign the dict."""
    demography = msprime.Demography()
    demography.add_population(name="P", initial_size=10_000)
    ts = msprime.sim_ancestry(samples={"P": 20},
                              sequence_length=50_000,
                              demography=demography,
                              random_seed=7, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=7)
    hm = HaplotypeMatrix.from_ts(ts)
    assert list(hm.sample_sets.keys()) == ["P"]

    n = hm.num_haplotypes
    hm.sample_sets = {"left": list(range(n // 2)),
                      "right": list(range(n // 2, n))}
    assert set(hm.sample_sets.keys()) == {"left", "right"}


def test_shape(sample_vcf):
    """Test the shape property."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    shape = hap_matrix.shape
    print(shape)
    assert len(shape) == 2  # (n_variants, n_haplotypes)
    assert shape[0] == 20  # We simulated 10 samples * 2 haplotypes per sample

def test_repr(sample_vcf):
    """Test string representation."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    repr_str = repr(hap_matrix)
    assert "HaplotypeMatrix" in repr_str
    assert "shape=" in repr_str
    assert "first_position=" in repr_str
    assert "last_position=" in repr_str

def test_get_matrix(sample_vcf):
    """Test get_matrix method."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    matrix = hap_matrix.get_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.ndim == 2  # (n_variants, n_haplotypes)
    # move to GPU
    hap_matrix.transfer_to_gpu()
    matrix = hap_matrix.get_matrix()
    assert isinstance(matrix, cp.ndarray)

def test_get_positions(sample_vcf):
    """Test get_positions method."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    positions = hap_matrix.get_positions()
    assert isinstance(positions, np.ndarray)
    assert positions.ndim == 1
    assert np.all(np.diff(positions) >= 0)  # Positions should be sorted
    # move to GPU
    hap_matrix.transfer_to_gpu()
    positions = hap_matrix.get_positions()
    assert isinstance(positions, cp.ndarray)

def test_empty_haplotype_matrix():
    """Test handling of empty data."""
    with pytest.raises(Exception):
        # Create HaplotypeMatrix with empty arrays
        HaplotypeMatrix(cp.array([]), cp.array([]))

def test_bad_genotypes_bad_positions():
    """Test handling of bad genotypes."""
    with pytest.raises(Exception):
        HaplotypeMatrix([[0, 1, 0, 1],
                          [1, 0, 1, 0]],
                          cp.array([100, 200]))
    with pytest.raises(Exception):
        HaplotypeMatrix(
            cp.array([[0, 1, 0, 1],
                      [1, 0, 1, 0]]),
            (100, 200) #tuple input
        )

def test_mixed_device_genotypes_positions():
    """Test handling of mixed device genotypes and positions."""
    hap_matrix = HaplotypeMatrix(
        cp.array([[0, 1, 0, 1],
                  [1, 0, 1, 0]]),
        np.array([100, 200])
    )
    # this will transfer the positions to the GPU
    assert hap_matrix.device == 'GPU'
    assert isinstance(hap_matrix.haplotypes, cp.ndarray)
    assert isinstance(hap_matrix.positions, cp.ndarray)
    # try reversing the order of devices which will end up on the CPU
    hap_matrix = HaplotypeMatrix(
        np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0]]),
        cp.array([100, 200])
    )
    assert hap_matrix.device == 'CPU'
    assert isinstance(hap_matrix.positions, np.ndarray)
    # for fun transfer the positions to the GPU
    hap_matrix.transfer_to_gpu()
    assert hap_matrix.device == 'GPU'
    assert isinstance(hap_matrix.positions, cp.ndarray)

def test_get_subset_from_range(sample_vcf):
    """Test get_subset_from_range method."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    positions = hap_matrix.positions
    low = 0
    high = int(positions[10])
    count = int(cp.sum((positions >= low) & (positions < high)))
    subset = hap_matrix.get_subset_from_range(low, high)
    assert isinstance(subset, HaplotypeMatrix)
    assert subset.shape == (20, count)

def test_get_subset(sample_vcf):
    """Test get_subset method."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    subset = hap_matrix.get_subset(cp.array([0, 1, 2, 3, 4]))
    assert isinstance(subset, HaplotypeMatrix)
    assert subset.shape == (20, 5)
    subset2 = hap_matrix.get_subset(np.array([0, 1, 2, 3, 4]))
    assert isinstance(subset2, HaplotypeMatrix)
    assert subset2.shape == (20, 5)

def test_transfer_to_gpu():
    """Test transferring data from CPU to GPU."""
    # Create a small dummy haplotype matrix (using NumPy)
    genotypes = np.array([[0, 1, 0, 1],
                          [1, 0, 1, 0]], dtype=np.int32)
    positions = np.array([100, 200], dtype=np.int32)
    hm = HaplotypeMatrix(genotypes.copy(), positions.copy())

    # Verify initial device is CPU and arrays are NumPy arrays.
    assert hm.device == 'CPU'
    assert isinstance(hm.haplotypes, np.ndarray)
    assert isinstance(hm.positions, np.ndarray)

    # Transfer to GPU
    hm.transfer_to_gpu()

    # Verify that the device updates and the underlying arrays are now CuPy arrays.
    assert hm.device == 'GPU'
    assert isinstance(hm.haplotypes, cp.ndarray)
    assert isinstance(hm.positions, cp.ndarray)

    # Verify that the content remains identical (using cp.asnumpy for comparison)
    np.testing.assert_array_equal(cp.asnumpy(hm.haplotypes), genotypes)
    np.testing.assert_array_equal(cp.asnumpy(hm.positions), positions)

def test_transfer_to_cpu():
    """Test transferring data back from GPU to CPU."""
    # Create a small dummy haplotype matrix (using NumPy)
    genotypes = np.array([[0, 1, 0, 1],
                          [1, 0, 1, 0]], dtype=np.int32)
    positions = np.array([100, 200], dtype=np.int32)
    hm = HaplotypeMatrix(genotypes.copy(), positions.copy())

    # Move to GPU first.
    hm.transfer_to_gpu()
    assert hm.device == 'GPU'

    # Transfer back to CPU
    hm.transfer_to_cpu()

    # Verify that the device is back to CPU and arrays are NumPy arrays.
    assert hm.device == 'CPU'
    assert isinstance(hm.haplotypes, np.ndarray)
    assert isinstance(hm.positions, np.ndarray)

    # Ensure that the data is identical to the original.
    np.testing.assert_array_equal(hm.haplotypes, genotypes)
    np.testing.assert_array_equal(hm.positions, positions)


def test_allele_frequency_spectrum(sample_vcf):
    """Test calculation of allele frequency spectrum."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    afs = hap_matrix.allele_frequency_spectrum()
    assert isinstance(afs, np.ndarray)
    assert afs.ndim == 1
    # AFS has n+1 bins for frequencies 0 to n
    assert afs.size == hap_matrix.num_haplotypes + 1

def test_diversity(sample_vcf):
    """Test calculation of pi."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    pi = hap_matrix.diversity()
    assert isinstance(pi, float)
    pi_span = hap_matrix.diversity(span_normalize=True)
    assert isinstance(pi_span, float)
    assert pi_span <= pi

def test_diversity_tskit(sample_ts):
    """Test calculation of pi from tskit.TreeSequence."""
    hap_matrix = HaplotypeMatrix.from_ts(sample_ts)
    pi = hap_matrix.diversity(span_normalize=True)
    pi_tskit = sample_ts.diversity(mode="site")
    assert isinstance(pi, float)
    assert cp.allclose(pi, pi_tskit)

def test_pairwise_LD_v_transfers():
    """
    Test that calling pairwise_LD_v on a CPU-based HaplotypeMatrix
    automatically transfers the data to the GPU and computes the LD matrix.
    """
    # Create a dummy haplotype matrix with at least 2 variants.
    # Here rows represent variants and columns represent haplotypes.
    genotypes = np.array([[0, 1, 1, 0],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 0, 0, 1],
                          [0, 1, 1, 0],
                          [1, 0, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 1],
                          ], dtype=np.int32)
    positions = np.array([50, 150, 250, 350, 450, 550, 650, 750], dtype=np.int32)
    hm = HaplotypeMatrix(genotypes.copy(), positions.copy())

    # Initially, data should be on the CPU.
    assert hm.device == 'CPU'

    # Call the pairwise LD function. It should check and transfer data to the GPU.
    D = hm.pairwise_LD_v()

    # Verify that the device is now GPU.
    assert hm.device == 'GPU'
    # Verify that D is a CuPy array.
    assert isinstance(D, cp.ndarray)
    # Check that D is symmetric (D should equal its transpose).
    assert cp.allclose(D, D.T)
