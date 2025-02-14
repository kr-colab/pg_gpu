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
    assert cp.allclose(pi, float(pi_tskit))
    
