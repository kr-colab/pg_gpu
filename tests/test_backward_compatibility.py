"""
Backward compatibility tests for HaplotypeMatrix methods.

Ensures that existing code using the old HaplotypeMatrix methods
continues to work and produces identical results to the new modules.
"""

import pytest
import numpy as np
import cupy as cp
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity, divergence


class TestHaplotypeMatrixCompatibility:
    """Test that old HaplotypeMatrix methods still work and match new modules."""

    @pytest.fixture
    def test_matrix(self):
        """Create a test HaplotypeMatrix for compatibility testing."""
        np.random.seed(42)
        n_haplotypes = 40
        n_variants = 150

        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.6, 0.4])
        positions = np.arange(n_variants) * 1000 + 1000

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        return matrix

    def test_diversity_method_compatibility(self, test_matrix):
        """Test that HaplotypeMatrix.diversity() matches diversity.pi()."""
        # Old method
        pi_old_normalized = test_matrix.diversity(span_normalize=True)
        pi_old_raw = test_matrix.diversity(span_normalize=False)

        # New module method
        pi_new_normalized = diversity.pi(test_matrix, span_normalize=True)
        pi_new_raw = diversity.pi(test_matrix, span_normalize=False)

        # Should be identical
        assert abs(pi_old_normalized - pi_new_normalized) < 1e-15
        assert abs(pi_old_raw - pi_new_raw) < 1e-15

        # Should also match scikit-allel
        h = allel.HaplotypeArray(test_matrix.haplotypes.T if isinstance(test_matrix.haplotypes, np.ndarray)
                                 else test_matrix.haplotypes.get().T)
        ac = h.count_alleles()
        pi_allel = allel.sequence_diversity(
            test_matrix.positions if isinstance(test_matrix.positions, np.ndarray)
            else test_matrix.positions.get(),
            ac,
            start=test_matrix.chrom_start,
            stop=test_matrix.chrom_end
        )

        assert abs(pi_old_normalized - pi_allel) < 1e-8

    def test_watersons_theta_compatibility(self, test_matrix):
        """Test that HaplotypeMatrix.watersons_theta() matches diversity.theta_w()."""
        # Old method
        theta_old_normalized = test_matrix.watersons_theta(span_normalize=True)
        theta_old_raw = test_matrix.watersons_theta(span_normalize=False)

        # New module method
        theta_new_normalized = diversity.theta_w(test_matrix, span_normalize=True)
        theta_new_raw = diversity.theta_w(test_matrix, span_normalize=False)

        # Should be identical
        assert abs(theta_old_normalized - theta_new_normalized) < 1e-15
        assert abs(theta_old_raw - theta_new_raw) < 1e-15

        # Should also match scikit-allel
        h = allel.HaplotypeArray(test_matrix.haplotypes.T if isinstance(test_matrix.haplotypes, np.ndarray)
                                 else test_matrix.haplotypes.get().T)
        ac = h.count_alleles()
        theta_allel = allel.watterson_theta(
            test_matrix.positions if isinstance(test_matrix.positions, np.ndarray)
            else test_matrix.positions.get(),
            ac,
            start=test_matrix.chrom_start,
            stop=test_matrix.chrom_end
        )

        assert abs(theta_old_normalized - theta_allel) < 1e-8

    def test_tajimas_d_compatibility(self, test_matrix):
        """Test that HaplotypeMatrix.Tajimas_D() matches diversity.tajimas_d()."""
        # Old method
        tajd_old = test_matrix.Tajimas_D()

        # New module method
        tajd_new = diversity.tajimas_d(test_matrix)

        # Should be identical
        assert abs(tajd_old - tajd_new) < 1e-15

        # Should also match scikit-allel
        h = allel.HaplotypeArray(test_matrix.haplotypes.T if isinstance(test_matrix.haplotypes, np.ndarray)
                                 else test_matrix.haplotypes.get().T)
        ac = h.count_alleles()
        tajd_allel = allel.tajima_d(
            ac,
            pos=test_matrix.positions if isinstance(test_matrix.positions, np.ndarray)
            else test_matrix.positions.get()
        )

        assert abs(tajd_old - tajd_allel) < 1e-8

    # NOTE: HaplotypeMatrix.allele_frequency_spectrum() and
    # diversity.allele_frequency_spectrum() were removed (issue #100 consolidation);
    # the SFS now lives only in the sfs module. See
    # tests/test_sfs_allel_comparison.py::TestSFSComparison::test_sfs.

    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_compatibility(self):
        """Test that old methods work on GPU."""
        np.random.seed(123)
        haplotypes = np.random.choice([0, 1], size=(30, 100), p=[0.7, 0.3])
        positions = np.arange(100) * 1000 + 1000

        # Create matrix and transfer to GPU
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        matrix.transfer_to_gpu()

        # Old methods should work on GPU
        pi_gpu_old = matrix.diversity(span_normalize=True)
        theta_gpu_old = matrix.watersons_theta(span_normalize=True)
        tajd_gpu_old = matrix.Tajimas_D()

        # New methods should give identical results
        pi_gpu_new = diversity.pi(matrix, span_normalize=True)
        theta_gpu_new = diversity.theta_w(matrix, span_normalize=True)
        tajd_gpu_new = diversity.tajimas_d(matrix)

        assert abs(pi_gpu_old - pi_gpu_new) < 1e-15
        assert abs(theta_gpu_old - theta_gpu_new) < 1e-15
        assert abs(tajd_gpu_old - tajd_gpu_new) < 1e-15

        # CPU version should match
        matrix_cpu = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        pi_cpu = diversity.pi(matrix_cpu, span_normalize=True)
        theta_cpu = diversity.theta_w(matrix_cpu, span_normalize=True)
        tajd_cpu = diversity.tajimas_d(matrix_cpu)

        assert abs(pi_gpu_old - pi_cpu) < 1e-15
        assert abs(theta_gpu_old - theta_cpu) < 1e-15
        assert abs(tajd_gpu_old - tajd_cpu) < 1e-15


class TestDeprecationWarnings:
    """Test that deprecated methods show appropriate warnings (if implemented)."""

    def test_no_unexpected_warnings(self):
        """Test that backward compatibility methods don't produce unexpected warnings."""
        np.random.seed(456)
        haplotypes = np.random.choice([0, 1], size=(20, 50), p=[0.6, 0.4])
        positions = np.arange(50) * 1000 + 1000

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])

        # These should work without warnings (for now)
        import warnings
        with warnings.catch_warnings(record=True) as warning_list:
            pi_val = matrix.diversity()
            theta_val = matrix.watersons_theta()
            tajd_val = matrix.Tajimas_D()

        # Filter out any unrelated warnings
        relevant_warnings = [w for w in warning_list if 'deprecated' in str(w.message).lower()]

        # For now, we're not expecting deprecation warnings
        # In the future, these could be uncommented to test deprecation warnings
        # assert len(relevant_warnings) == 4  # One for each deprecated method


class TestAPIConsistency:
    """Test that API behavior is consistent between old and new methods."""

    def test_parameter_handling(self):
        """Test that parameter handling is consistent."""
        np.random.seed(789)
        haplotypes = np.random.choice([0, 1], size=(25, 75), p=[0.65, 0.35])
        positions = np.arange(75) * 1000 + 1000

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])

        # Test default parameters
        pi_old_default = matrix.diversity()  # Default is span_normalize=True
        pi_new_default = diversity.pi(matrix)  # Default is span_normalize=True
        assert abs(pi_old_default - pi_new_default) < 1e-15

        # Test explicit parameters
        pi_old_explicit = matrix.diversity(span_normalize=False)
        pi_new_explicit = diversity.pi(matrix, span_normalize=False)
        assert abs(pi_old_explicit - pi_new_explicit) < 1e-15

        theta_old_default = matrix.watersons_theta()
        theta_new_default = diversity.theta_w(matrix)
        assert abs(theta_old_default - theta_new_default) < 1e-15

    def test_return_type_consistency(self):
        """Test that return types are consistent."""
        np.random.seed(999)
        haplotypes = np.random.choice([0, 1], size=(15, 30), p=[0.5, 0.5])
        positions = np.arange(30) * 1000 + 1000

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])

        # Test return types
        pi_old = matrix.diversity()
        pi_new = diversity.pi(matrix)
        assert type(pi_old) == type(pi_new) == float

        theta_old = matrix.watersons_theta()
        theta_new = diversity.theta_w(matrix)
        assert type(theta_old) == type(theta_new) == float

        tajd_old = matrix.Tajimas_D()
        tajd_new = diversity.tajimas_d(matrix)
        assert type(tajd_old) == type(tajd_new) == float


class TestDocumentationExamples:
    """Test examples that might be in documentation to ensure they still work."""

    def test_basic_usage_example(self):
        """Test a basic usage example."""
        # Example that might be in documentation
        haplotypes = np.array([
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1]
        ])
        positions = np.array([1000, 2000, 3000, 4000, 5000])

        # Old way
        matrix = HaplotypeMatrix(haplotypes, positions)
        pi_old = matrix.diversity(span_normalize=False)
        theta_old = matrix.watersons_theta(span_normalize=False)

        # New way
        pi_new = diversity.pi(matrix, span_normalize=False)
        theta_new = diversity.theta_w(matrix, span_normalize=False)

        # Should be identical
        assert abs(pi_old - pi_new) < 1e-15
        assert abs(theta_old - theta_new) < 1e-15

        # Both should produce reasonable values
        assert pi_old > 0
        assert theta_old > 0

    def test_complex_workflow_example(self):
        """Test a more complex workflow example."""
        # Simulate loading data and computing multiple statistics
        np.random.seed(100)
        haplotypes = np.random.choice([0, 1], size=(50, 200), p=[0.6, 0.4])
        positions = np.arange(200) * 1000 + 1000

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])

        # Old workflow
        old_results = {
            'pi': matrix.diversity(span_normalize=True),
            'theta': matrix.watersons_theta(span_normalize=True),
            'tajimas_d': matrix.Tajimas_D(),
        }

        # New workflow
        new_results = {
            'pi': diversity.pi(matrix, span_normalize=True),
            'theta': diversity.theta_w(matrix, span_normalize=True),
            'tajimas_d': diversity.tajimas_d(matrix),
        }

        # Should be identical
        assert abs(old_results['pi'] - new_results['pi']) < 1e-15
        assert abs(old_results['theta'] - new_results['theta']) < 1e-15
        assert abs(old_results['tajimas_d'] - new_results['tajimas_d']) < 1e-15
