"""
Tests for divergence statistics module.
"""

import pytest
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu import divergence


class TestFSTCalculations:
    """Test FST calculation methods."""

    def test_fst_identical_populations(self):
        """Test FST = 0 for identical populations."""
        # Create identical populations
        n_variants = 100
        n_samples = 40
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20))  # Same as pop1
        }

        fst_value = divergence.fst(matrix, 'pop1', 'pop2')
        # Identical populations yield FST <= 0 (slightly negative due to
        # bias correction in within-population heterozygosity)
        assert fst_value <= 0.0 + 1e-10
        assert fst_value > -0.2  # should be close to zero

    def test_fst_completely_differentiated(self):
        """Test FST = 1 for completely differentiated populations."""
        n_variants = 50
        n_samples = 40

        # Create completely differentiated populations
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:20, :] = 0  # Pop1 all 0s
        haplotypes[20:, :] = 1  # Pop2 all 1s

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))
        }

        fst_value = divergence.fst(matrix, 'pop1', 'pop2')
        assert fst_value > 0.9  # Should be close to 1

    def test_fst_hudson_vs_nei(self):
        """Test that different FST methods give reasonable results."""
        n_variants = 100
        n_samples = 60

        # Create populations with some differentiation
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        # Add some structure
        haplotypes[:20, :50] = np.random.choice([0, 1], size=(20, 50), p=[0.8, 0.2])
        haplotypes[20:40, :50] = np.random.choice([0, 1], size=(20, 50), p=[0.2, 0.8])

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40)),
            'pop3': list(range(40, 60))
        }

        # Compare methods
        fst_hudson = divergence.fst_hudson(matrix, 'pop1', 'pop2')
        fst_nei = divergence.fst_nei(matrix, 'pop1', 'pop2')
        fst_wc = divergence.fst_weir_cockerham(matrix, 'pop1', 'pop2')

        # All should be positive
        assert fst_hudson >= 0
        assert fst_nei >= 0
        assert fst_wc >= 0

        # All should be less than 1
        assert fst_hudson <= 1
        assert fst_nei <= 1
        assert fst_wc <= 1

        # They should be somewhat similar (within an order of magnitude)
        assert abs(fst_hudson - fst_nei) < 0.5
        assert abs(fst_hudson - fst_wc) < 0.5

    def test_fst_with_list_indices(self):
        """Test FST calculation with list of indices instead of population names."""
        n_variants = 50
        n_samples = 40
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)

        # Use indices directly
        pop1_indices = list(range(20))
        pop2_indices = list(range(20, 40))

        fst_value = divergence.fst(matrix, pop1_indices, pop2_indices)
        assert fst_value <= 1
        assert fst_value > -0.2  # random split, expect near zero


class TestDxyCalculations:
    """Test Dxy and related divergence metrics."""

    def test_dxy_identical_populations(self):
        """Test Dxy for identical populations."""
        n_variants = 100
        n_samples = 40
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))  # Different individuals but from same distribution
        }

        dxy_value = divergence.dxy(matrix, 'pop1', 'pop2')

        # For populations from same distribution, Dxy should be close to pi
        # But allow for some variation due to random sampling
        pi_value = divergence.pi_within_population(matrix, 'pop1')
        assert abs(dxy_value - pi_value) < 0.05  # Relaxed tolerance for sampling variance

    def test_dxy_fixed_differences(self):
        """Test Dxy for populations with fixed differences."""
        n_variants = 50
        n_samples = 40

        # Create populations with fixed differences
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:20, :25] = 0  # Pop1 fixed for 0 at first 25 sites
        haplotypes[20:, :25] = 1  # Pop2 fixed for 1 at first 25 sites
        # Rest of sites are polymorphic within populations
        haplotypes[:, 25:] = np.random.randint(0, 2, size=(n_samples, 25))

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))
        }

        dxy_value = divergence.dxy(matrix, 'pop1', 'pop2',
                                    span_normalize=False)

        # Dxy should be at least 0.5 (25 fixed differences out of 50 sites)
        assert dxy_value >= 0.5

    def test_dxy_per_site(self):
        """Test per-site Dxy calculation."""
        n_variants = 10
        n_samples = 40

        # Simple case: first 5 sites fixed different, last 5 sites identical
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:20, :5] = 0
        haplotypes[20:, :5] = 1
        haplotypes[:, 5:] = 0  # All samples identical at last 5 sites

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))
        }

        dxy_per_site = divergence.dxy(matrix, 'pop1', 'pop2', per_site=True)

        dxy_per_site = np.asarray(dxy_per_site)

        # First 5 sites should have Dxy = 1
        assert np.all(dxy_per_site[:5] == 1.0)
        # Last 5 sites should have Dxy = 0
        assert np.all(dxy_per_site[5:] == 0.0)

    def test_da_calculation(self):
        """Test net divergence (Da) calculation."""
        n_variants = 100
        n_samples = 60

        # Create structured populations
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Pop1: mostly 0s with some variation
        haplotypes[:20, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.8, 0.2])
        # Pop2: mostly 1s with some variation
        haplotypes[20:40, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.2, 0.8])
        # Pop3: intermediate
        haplotypes[40:, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.5, 0.5])

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40)),
            'pop3': list(range(40, 60))
        }

        # Calculate Da
        da_12 = divergence.da(matrix, 'pop1', 'pop2')
        da_13 = divergence.da(matrix, 'pop1', 'pop3')
        da_23 = divergence.da(matrix, 'pop2', 'pop3')

        # Da should be non-negative
        assert da_12 >= 0
        assert da_13 >= 0
        assert da_23 >= 0

        # Da between pop1 and pop2 should be largest
        assert da_12 > da_13
        assert da_12 > da_23


class TestDivergenceStats:
    """Test composite divergence statistics function."""

    def test_divergence_stats_all(self):
        """Test computing multiple divergence statistics at once."""
        n_variants = 100
        n_samples = 40
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'popA': list(range(20)),
            'popB': list(range(20, 40))
        }

        # Compute all statistics
        stats = divergence.divergence_stats(
            matrix, 'popA', 'popB',
            statistics=['fst', 'fst_hudson', 'fst_wc', 'fst_nei', 'dxy', 'da', 'pi1', 'pi2']
        )

        # Check all statistics are present
        assert 'fst' in stats
        assert 'fst_hudson' in stats
        assert 'fst_wc' in stats
        assert 'fst_nei' in stats
        assert 'dxy' in stats
        assert 'da' in stats
        assert 'pi1' in stats
        assert 'pi2' in stats

        # Diversity values should be non-negative
        for key in ['dxy', 'pi1', 'pi2']:
            assert stats[key] >= 0

        # FST and Da can be slightly negative for random splits of the same
        # population (no clipping to zero, matching scikit-allel behavior)
        for fst_key in ['fst', 'fst_hudson', 'fst_wc', 'fst_nei']:
            assert stats[fst_key] <= 1


class TestPairwiseFST:
    """Test pairwise FST matrix calculation."""

    def test_pairwise_fst_matrix(self):
        """Test pairwise FST calculation for multiple populations."""
        n_variants = 100
        n_samples = 60

        # Create three populations with different levels of differentiation
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Pop1: mostly 0s
        haplotypes[:20, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.8, 0.2])
        # Pop2: mostly 1s
        haplotypes[20:40, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.2, 0.8])
        # Pop3: intermediate
        haplotypes[40:, :] = np.random.choice([0, 1], size=(20, n_variants), p=[0.5, 0.5])

        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40)),
            'pop3': list(range(40, 60))
        }

        # Calculate pairwise FST
        fst_matrix, pop_names = divergence.pairwise_fst(matrix)

        if isinstance(fst_matrix, cp.ndarray):
            fst_matrix = fst_matrix.get()

        # Check dimensions
        assert fst_matrix.shape == (3, 3)
        assert len(pop_names) == 3

        # Diagonal should be zero
        assert fst_matrix[0, 0] == 0
        assert fst_matrix[1, 1] == 0
        assert fst_matrix[2, 2] == 0

        # Matrix should be symmetric
        assert fst_matrix[0, 1] == fst_matrix[1, 0]
        assert fst_matrix[0, 2] == fst_matrix[2, 0]
        assert fst_matrix[1, 2] == fst_matrix[2, 1]

        # FST between pop1 and pop2 should be highest
        assert fst_matrix[0, 1] > fst_matrix[0, 2]
        assert fst_matrix[0, 1] > fst_matrix[1, 2]

    def test_pairwise_fst_subset_populations(self):
        """Test pairwise FST with subset of populations."""
        n_variants = 50
        n_samples = 80
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'A': list(range(20)),
            'B': list(range(20, 40)),
            'C': list(range(40, 60)),
            'D': list(range(60, 80))
        }

        # Calculate for subset
        fst_matrix, pop_names = divergence.pairwise_fst(matrix, populations=['A', 'C', 'D'])

        if isinstance(fst_matrix, cp.ndarray):
            fst_matrix = fst_matrix.get()

        assert fst_matrix.shape == (3, 3)
        assert pop_names == ['A', 'C', 'D']


class TestGPUCalculations:
    """Test GPU-specific functionality."""

    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_calculations(self):
        """Test calculations work on GPU."""
        n_variants = 1000
        n_samples = 100
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000

        # Create matrix on GPU
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.transfer_to_gpu()
        matrix.sample_sets = {
            'pop1': list(range(50)),
            'pop2': list(range(50, 100))
        }

        # All functions should work on GPU
        fst_val = divergence.fst(matrix, 'pop1', 'pop2')
        dxy_val = divergence.dxy(matrix, 'pop1', 'pop2')
        da_val = divergence.da(matrix, 'pop1', 'pop2')

        assert isinstance(fst_val, float)
        assert isinstance(dxy_val, float)
        assert isinstance(da_val, float)

        # Per-site calculation
        dxy_per_site = divergence.dxy(matrix, 'pop1', 'pop2', per_site=True)
        assert isinstance(dxy_per_site, np.ndarray)
        assert dxy_per_site.shape == (n_variants,)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_population(self):
        """Test handling of populations with single sample."""
        n_variants = 50
        haplotypes = np.random.randint(0, 2, size=(10, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'single': [0],  # Single sample
            'multi': list(range(1, 10))
        }

        # Should handle gracefully
        pi_single = divergence.pi_within_population(matrix, 'single')
        assert pi_single == 0.0  # No diversity in single sample

        # FST should still work
        fst_val = divergence.fst(matrix, 'single', 'multi')
        assert 0 <= fst_val <= 1

    def test_missing_population(self):
        """Test error handling for missing population."""
        n_variants = 50
        haplotypes = np.random.randint(0, 2, size=(20, n_variants))
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {'pop1': list(range(20))}

        with pytest.raises(ValueError):
            divergence.fst(matrix, 'pop1', 'nonexistent')

    def test_no_variation(self):
        """Test handling of no variation."""
        n_variants = 50
        n_samples = 40

        # All samples identical
        haplotypes = np.ones((n_samples, n_variants), dtype=int)
        positions = np.arange(n_variants) * 1000

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))
        }

        # FST should be 0 (no variation to differentiate)
        fst_val = divergence.fst(matrix, 'pop1', 'pop2')
        assert fst_val == 0.0

        # Dxy should be 0
        dxy_val = divergence.dxy(matrix, 'pop1', 'pop2')
        assert dxy_val == 0.0
