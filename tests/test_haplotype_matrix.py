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
    assert isinstance(hap_matrix.get_matrix(), cp.ndarray)  # Now expecting CuPy array
    assert isinstance(hap_matrix.get_positions(), cp.ndarray)  # Now expecting CuPy array
    assert len(hap_matrix.get_positions()) > 0

def test_from_ts(sample_ts):
    """Test creating HaplotypeMatrix from tskit.TreeSequence."""
    hap_matrix = HaplotypeMatrix.from_ts(sample_ts)
    assert isinstance(hap_matrix, HaplotypeMatrix)
    assert isinstance(hap_matrix.get_matrix(), cp.ndarray)  # Now expecting CuPy array
    assert isinstance(hap_matrix.get_positions(), cp.ndarray)  # Now expecting CuPy array
    
    # Add test for pairwise_LD
    D = hap_matrix.pairwise_LD_v()
    assert isinstance(D, cp.ndarray)
    assert D.shape == (hap_matrix.num_variants, hap_matrix.num_variants)
    assert cp.allclose(D, D.T)  # Check symmetry
    assert cp.all(cp.abs(D) <= 0.25)  # D is bounded by ±0.25

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
    assert isinstance(matrix, cp.ndarray)
    assert matrix.ndim == 2  # (n_variants, n_haplotypes)

def test_get_positions(sample_vcf):
    """Test get_positions method."""
    hap_matrix = HaplotypeMatrix.from_vcf(sample_vcf)
    positions = hap_matrix.get_positions()
    assert isinstance(positions, cp.ndarray)
    assert positions.ndim == 1
    assert np.all(np.diff(positions) >= 0)  # Positions should be sorted

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
    assert isinstance(afs, cp.ndarray)
    assert afs.ndim == 1
    assert afs.size == hap_matrix.num_haplotypes
    
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