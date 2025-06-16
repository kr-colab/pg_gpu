"""
Tests for diversity statistics module.
"""

import pytest
import numpy as np
import cupy as cp
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity


class TestNucleotideDiversity:
    """Test nucleotide diversity (pi) calculations."""
    
    def test_pi_no_variation(self):
        """Test pi = 0 when there's no variation."""
        # All samples identical
        n_variants = 50
        n_samples = 20
        haplotypes = np.ones((n_samples, n_variants), dtype=int)
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
        
        pi_value = diversity.pi(matrix, span_normalize=False)
        assert pi_value == 0.0
    
    def test_pi_maximum_variation(self):
        """Test pi with maximum variation (50/50 allele frequency)."""
        n_variants = 100
        n_samples = 40
        
        # Create 50/50 allele frequency at each site
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:20, :] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
        
        pi_value = diversity.pi(matrix, span_normalize=False)
        
        # Expected pi for 50/50 frequency
        expected_pi = n_variants * 2 * 0.5 * 0.5 * n_samples / (n_samples - 1)
        assert abs(pi_value - expected_pi) < 0.01
    
    def test_pi_span_normalization(self):
        """Test span normalization of pi."""
        n_variants = 50
        n_samples = 30
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        
        # Create positions with known span
        positions = np.arange(n_variants) * 100  # 100 bp spacing
        span = positions[-1] - positions[0]
        
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        pi_raw = diversity.pi(matrix, span_normalize=False)
        pi_normalized = diversity.pi(matrix, span_normalize=True)
        
        # Normalized should be raw divided by span
        assert abs(pi_normalized - pi_raw / span) < 1e-10
    
    def test_pi_with_population(self):
        """Test pi calculation for specific population."""
        n_variants = 100
        n_samples = 60
        
        # Create structured populations
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Pop1: low diversity
        haplotypes[:20, :50] = np.random.choice([0, 1], size=(20, 50), p=[0.9, 0.1])
        # Pop2: high diversity  
        haplotypes[20:40, :50] = np.random.choice([0, 1], size=(20, 50), p=[0.5, 0.5])
        # Pop3: medium diversity
        haplotypes[40:, :50] = np.random.choice([0, 1], size=(20, 50), p=[0.7, 0.3])
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40)),
            'pop3': list(range(40, 60))
        }
        
        pi1 = diversity.pi(matrix, 'pop1', span_normalize=False)
        pi2 = diversity.pi(matrix, 'pop2', span_normalize=False)
        pi3 = diversity.pi(matrix, 'pop3', span_normalize=False)
        
        # Pop2 should have highest diversity
        assert pi2 > pi1
        assert pi2 > pi3
        # Pop3 should have more than pop1
        assert pi3 > pi1


class TestWattersonTheta:
    """Test Watterson's theta calculations."""
    
    def test_theta_w_no_segregating_sites(self):
        """Test theta_w = 0 when no segregating sites."""
        n_variants = 50
        n_samples = 30
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
        
        theta_value = diversity.theta_w(matrix, span_normalize=False)
        assert theta_value == 0.0
    
    def test_theta_w_all_segregating(self):
        """Test theta_w when all sites are segregating."""
        n_variants = 50
        n_samples = 20
        
        # Make all sites segregating (at least one 0 and one 1)
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[0, :] = 1  # First sample has all 1s
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        theta_value = diversity.theta_w(matrix, span_normalize=False)
        
        # Calculate expected theta
        a1 = sum(1.0 / i for i in range(1, n_samples))
        expected_theta = n_variants / a1
        
        assert abs(theta_value - expected_theta) < 0.01
    
    def test_theta_w_with_varying_frequencies(self):
        """Test theta_w with sites at different frequencies."""
        n_samples = 100
        
        # Create data with specific allele frequencies
        # 10 sites with 1 derived allele (singletons)
        # 10 sites with 10 derived alleles  
        # 10 sites with 50 derived alleles (50% frequency)
        haplotypes = np.zeros((n_samples, 30), dtype=int)
        
        # Singletons
        for i in range(10):
            haplotypes[i, i] = 1
            
        # 10% frequency sites
        for i in range(10):
            haplotypes[:10, 10 + i] = 1
            
        # 50% frequency sites
        for i in range(10):
            haplotypes[:50, 20 + i] = 1
        
        positions = np.arange(30) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        theta_value = diversity.theta_w(matrix, span_normalize=False)
        
        # All 30 sites are segregating
        # a1 = sum(1/i for i in range(1, 100)) ≈ 5.187
        expected_theta = 30 / 5.187
        assert abs(theta_value - expected_theta) < 0.1


class TestTajimasD:
    """Test Tajima's D statistic."""
    

    def test_tajimas_d_excess_rare_variants(self):
        """Test negative Tajima's D with excess rare variants."""
        n_variants = 100
        n_samples = 50
        
        # Create many singletons (rare variants)
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Each variant appears in only one sample
        for i in range(min(n_variants, n_samples)):
            haplotypes[i, i] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        d_value = diversity.tajimas_d(matrix)
        
        # Should be negative with excess rare variants
        assert d_value < 0
    
    def test_tajimas_d_no_variation(self):
        """Test Tajima's D when there's no variation."""
        n_variants = 50
        n_samples = 30
        haplotypes = np.ones((n_samples, n_variants), dtype=int)
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        d_value = diversity.tajimas_d(matrix)
        
        # Should be NaN when no variation
        assert np.isnan(d_value)


class TestAlleleFrequencySpectrum:
    """Test AFS calculations."""
    
    def test_afs_fixed_sites(self):
        """Test AFS for fixed sites."""
        n_variants = 50
        n_samples = 20
        
        # Half sites fixed for 0, half fixed for 1
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:, 25:] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        afs = diversity.allele_frequency_spectrum(matrix)
        
        if isinstance(afs, cp.ndarray):
            afs = afs.get()
        
        # Should have 25 sites with 0 copies and 25 with n_samples copies
        assert afs[0] == 25  # Sites with 0 derived alleles
        assert afs[n_samples] == 25  # Sites with all derived alleles
        # All other frequencies should be 0
        assert np.sum(afs[1:n_samples]) == 0
    
    def test_afs_singletons(self):
        """Test AFS with singleton sites."""
        n_variants = 30
        n_samples = 30
        
        # Create singletons
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        for i in range(n_variants):
            haplotypes[i, i] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        afs = diversity.allele_frequency_spectrum(matrix)
        
        if isinstance(afs, cp.ndarray):
            afs = afs.get()
        
        # Should have n_variants singletons
        assert afs[1] == n_variants
        assert np.sum(afs[2:]) == 0


class TestSegregatingSites:
    """Test segregating sites calculation."""
    
    def test_segregating_sites_count(self):
        """Test counting segregating sites."""
        n_variants = 100
        n_samples = 40
        
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # First 30 sites are segregating
        haplotypes[:20, :30] = 1
        # Next 20 sites are fixed for 0
        # Next 20 sites are fixed for 1
        haplotypes[:, 50:70] = 1
        # Last 30 sites are segregating
        haplotypes[:10, 70:] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        seg_sites = diversity.segregating_sites(matrix)
        
        # Should have 60 segregating sites (first 30 + last 30)
        assert seg_sites == 60
    
    def test_segregating_sites_with_population(self):
        """Test segregating sites for specific population."""
        n_variants = 50
        n_samples = 40
        
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Pop1 has variation at first 20 sites
        haplotypes[:10, :20] = 1
        # Pop2 has variation at last 20 sites
        haplotypes[30:, 30:] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.sample_sets = {
            'pop1': list(range(20)),
            'pop2': list(range(20, 40))
        }
        
        seg1 = diversity.segregating_sites(matrix, 'pop1')
        seg2 = diversity.segregating_sites(matrix, 'pop2')
        
        assert seg1 == 20  # Pop1 segregating at first 20 sites
        assert seg2 == 20  # Pop2 segregating at last 20 sites


class TestSingletons:
    """Test singleton counting."""
    
    def test_singleton_count(self):
        """Test counting singletons."""
        n_variants = 50
        n_samples = 30
        
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Create 20 singletons
        for i in range(20):
            haplotypes[i, i] = 1
        # Create 10 doubletons
        for i in range(20, 30):
            haplotypes[0:2, i] = 1
        # Rest are higher frequency
        haplotypes[:10, 30:] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        singletons = diversity.singleton_count(matrix)
        assert singletons == 20


class TestFayWusH:
    """Test Fay and Wu's H statistic."""
    
    def test_fay_wus_h_neutral(self):
        """Test H ≈ 0 under neutrality."""
        n_variants = 500
        n_samples = 50
        
        # Random data
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        h_value = diversity.fay_wus_h(matrix)
        
        # Under neutrality, H should be close to 0
        # This is a statistical test, so we allow some variation
        assert -20 < h_value < 20
    
    def test_fay_wus_h_high_frequency_derived(self):
        """Test negative H with excess high-frequency derived alleles."""
        n_variants = 100
        n_samples = 40
        
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Create many high-frequency derived alleles
        # (almost fixed for derived allele)
        haplotypes[:35, :50] = 1
        
        positions = np.arange(n_variants) * 1000
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        h_value = diversity.fay_wus_h(matrix)
        
        # Should be negative with excess high-frequency derived
        assert h_value < 0


class TestDiversityStats:
    """Test composite diversity statistics function."""
    
    def test_diversity_stats_all(self):
        """Test computing multiple statistics at once."""
        n_variants = 200
        n_samples = 50
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        stats = diversity.diversity_stats(
            matrix,
            statistics=['pi', 'theta_w', 'tajimas_d', 'segregating_sites', 'singletons']
        )
        
        # Check all statistics are present
        assert 'pi' in stats
        assert 'theta_w' in stats
        assert 'tajimas_d' in stats
        assert 'segregating_sites' in stats
        assert 'singletons' in stats
        
        # All should be reasonable values
        assert stats['pi'] >= 0
        assert stats['theta_w'] >= 0
        assert -10 < stats['tajimas_d'] < 10
        assert 0 <= stats['segregating_sites'] <= n_variants
        assert 0 <= stats['singletons'] <= stats['segregating_sites']


class TestNeutralityTests:
    """Test neutrality test suite."""
    
    def test_neutrality_tests(self):
        """Test neutrality tests function."""
        n_variants = 300
        n_samples = 60
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions)
        
        results = diversity.neutrality_tests(matrix)
        
        # Check all expected statistics
        assert 'tajimas_d' in results
        assert 'fay_wus_h' in results
        assert 'pi' in results
        assert 'theta_w' in results
        assert 'segregating_sites' in results
        
        # All should be valid
        assert not np.isnan(results['tajimas_d'])
        assert results['pi'] >= 0
        assert results['theta_w'] >= 0
        assert results['segregating_sites'] >= 0


class TestGPUCalculations:
    """Test GPU-specific functionality."""
    
    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_calculations(self):
        """Test diversity calculations on GPU."""
        n_variants = 500
        n_samples = 100
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000
        
        # Create matrix on GPU
        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.transfer_to_gpu()
        
        # All functions should work on GPU
        pi_val = diversity.pi(matrix)
        theta_val = diversity.theta_w(matrix)
        d_val = diversity.tajimas_d(matrix)
        seg_sites = diversity.segregating_sites(matrix)
        
        assert isinstance(pi_val, float)
        assert isinstance(theta_val, float)
        assert isinstance(d_val, float)
        assert isinstance(seg_sites, int)
        
        # AFS should return GPU array
        afs = diversity.allele_frequency_spectrum(matrix)
        assert isinstance(afs, cp.ndarray)


class TestBackwardCompatibility:
    """Test that HaplotypeMatrix methods still work."""
    
    def test_haplotype_matrix_methods(self):
        """Test deprecated methods still work."""
        n_variants = 100
        n_samples = 50
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))
        positions = np.arange(n_variants) * 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
        
        # Old methods should still work
        pi_old = matrix.diversity(span_normalize=False)
        theta_old = matrix.watersons_theta(span_normalize=False)
        d_old = matrix.Tajimas_D()
        
        # Should give same results as new module
        pi_new = diversity.pi(matrix, span_normalize=False)
        theta_new = diversity.theta_w(matrix, span_normalize=False)
        d_new = diversity.tajimas_d(matrix)
        
        assert abs(pi_old - pi_new) < 1e-10
        assert abs(theta_old - theta_new) < 1e-10
        assert abs(d_old - d_new) < 1e-10


class TestHaplotypeDiversity:
    """Test haplotype diversity calculations against scikit-allel."""
    
    def test_haplotype_diversity_simple(self):
        """Test haplotype diversity with simple known data."""
        # Create simple test case with known haplotype patterns
        haplotypes = np.array([
            [0, 0, 0, 0],  # haplotype 1: 0000
            [0, 0, 0, 0],  # haplotype 2: 0000 (same as 1)
            [1, 1, 1, 1],  # haplotype 3: 1111
            [1, 1, 1, 1],  # haplotype 4: 1111 (same as 3)
            [0, 1, 0, 1],  # haplotype 5: 0101 (unique)
            [1, 0, 1, 0],  # haplotype 6: 1010 (unique)
        ], dtype=np.int8)
        
        positions = np.array([1000, 2000, 3000, 4000])
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        # Calculate using our implementation
        h_div_pg = diversity.haplotype_diversity(matrix)
        
        # Calculate using scikit-allel
        import allel
        h_allel = allel.HaplotypeArray(haplotypes.T)
        h_div_allel = allel.haplotype_diversity(h_allel)
        
        # Should match within numerical precision
        assert abs(h_div_pg - h_div_allel) < 1e-10
        
        # Manual calculation: 4 unique haplotypes out of 6 total
        # Expected diversity = (1 - sum(p_i^2)) * n/(n-1) where p_i are frequencies (Nei's correction)
        # Frequencies: [2/6, 2/6, 1/6, 1/6] = [1/3, 1/3, 1/6, 1/6]
        # Uncorrected = 1 - ((1/3)^2 + (1/3)^2 + (1/6)^2 + (1/6)^2) = 1 - (1/9 + 1/9 + 1/36 + 1/36) = 1 - 10/36 = 26/36 = 13/18
        # Corrected = (13/18) * 6/5 = 13/15
        expected = 13/15
        assert abs(h_div_pg - expected) < 1e-10
    
    def test_haplotype_diversity_random_data(self):
        """Test haplotype diversity with random data against scikit-allel."""
        np.random.seed(42)
        n_haplotypes = 50
        n_variants = 20
        
        # Generate random haplotypes
        haplotypes = np.random.randint(0, 2, size=(n_haplotypes, n_variants), dtype=np.int8)
        positions = np.arange(n_variants) * 1000 + 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        # Calculate using our implementation
        h_div_pg = diversity.haplotype_diversity(matrix)
        
        # Calculate using scikit-allel
        import allel
        h_allel = allel.HaplotypeArray(haplotypes.T)
        h_div_allel = allel.haplotype_diversity(h_allel)
        
        # Should match within numerical precision
        assert abs(h_div_pg - h_div_allel) < 1e-10
    
    def test_haplotype_diversity_edge_cases(self):
        """Test edge cases for haplotype diversity."""
        # Case 1: All haplotypes identical
        haplotypes = np.ones((10, 5), dtype=np.int8)
        positions = np.arange(5) * 1000 + 1000
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        h_div = diversity.haplotype_diversity(matrix)
        assert h_div == 0.0  # No diversity when all identical
        
        # Case 2: All haplotypes unique
        n_variants = 4
        haplotypes = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
        ], dtype=np.int8)
        positions = np.arange(n_variants) * 1000 + 1000
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        h_div = diversity.haplotype_diversity(matrix)
        # All unique: diversity = (1 - sum(1/n)^2) * n/(n-1) = (1 - n*(1/n)^2) * n/(n-1) = (1 - 1/n) * n/(n-1) = ((n-1)/n) * n/(n-1) = 1
        expected = 1.0
        assert abs(h_div - expected) < 1e-10
    
    def test_haplotype_diversity_with_population(self):
        """Test haplotype diversity with population subset."""
        np.random.seed(123)
        n_haplotypes = 30
        n_variants = 15
        
        haplotypes = np.random.randint(0, 2, size=(n_haplotypes, n_variants), dtype=np.int8)
        positions = np.arange(n_variants) * 1000 + 1000
        
        # Set up populations
        pop1_indices = list(range(10))
        pop2_indices = list(range(10, 20))
        
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        matrix.sample_sets = {'pop1': pop1_indices, 'pop2': pop2_indices}
        
        # Test population-specific haplotype diversity
        h_div_pop1 = diversity.haplotype_diversity(matrix, 'pop1')
        h_div_pop2 = diversity.haplotype_diversity(matrix, 'pop2')
        h_div_all = diversity.haplotype_diversity(matrix)
        
        # Compare with scikit-allel
        import allel
        h_allel = allel.HaplotypeArray(haplotypes.T)
        h_div_allel_pop1 = allel.haplotype_diversity(h_allel.subset(sel1=pop1_indices))
        h_div_allel_pop2 = allel.haplotype_diversity(h_allel.subset(sel1=pop2_indices))
        h_div_allel_all = allel.haplotype_diversity(h_allel)
        
        assert abs(h_div_pop1 - h_div_allel_pop1) < 1e-10
        assert abs(h_div_pop2 - h_div_allel_pop2) < 1e-10
        assert abs(h_div_all - h_div_allel_all) < 1e-10