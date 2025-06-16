"""
Windowed analysis comparison tests with scikit-allel.

Tests that pg_gpu windowed analysis produces equivalent results to
scikit-allel windowed functions for overlapping functionality.
"""

import pytest
import numpy as np
import allel
import pandas as pd
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import WindowedAnalyzer, windowed_analysis


class TestWindowedDiversityComparison:
    """Compare windowed diversity calculations with scikit-allel."""
    
    @pytest.fixture
    def large_test_data(self):
        """Create larger test dataset suitable for windowing."""
        np.random.seed(42)
        n_haplotypes = 50
        n_variants = 1000
        
        # Create haplotypes with realistic structure
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
        
        # Add variation with different patterns across the genome
        for i in range(0, n_variants, 100):
            end_i = min(i + 100, n_variants)
            # Vary allele frequency across regions
            p = 0.1 + 0.4 * (i / n_variants)  # Frequency gradient
            haplotypes[:, i:end_i] = np.random.choice([0, 1], size=(n_haplotypes, end_i - i), p=[1-p, p])
        
        # Add some singletons
        for i in range(0, min(50, n_variants)):
            haplotypes[i % n_haplotypes, i] = 1 - haplotypes[i % n_haplotypes, i]
        
        # Positions with 1kb spacing
        positions = np.arange(n_variants) * 1000 + 10000
        
        return {
            'haplotypes': haplotypes,
            'positions': positions,
            'start': positions[0],
            'end': positions[-1]
        }
    
    def test_windowed_diversity_comparison(self, large_test_data):
        """Compare windowed diversity with scikit-allel."""
        haplotypes = large_test_data['haplotypes']
        positions = large_test_data['positions']
        start, end = large_test_data['start'], large_test_data['end']
        
        window_size = 50000  # 50kb windows
        
        # pg_gpu windowed analysis
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pg_results = windowed_analysis(
            matrix,
            window_size=window_size,
            statistics=['pi'],
            progress_bar=False
        )
        
        # scikit-allel windowed diversity
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        allel_pi, allel_windows, allel_n_bases, allel_counts = allel.windowed_diversity(
            positions, ac, size=window_size, start=start, stop=end
        )
        
        # Compare results
        # Note: there might be small differences due to window boundaries
        assert len(pg_results) == len(allel_pi)
        
        # Check that values are in similar range (allowing for boundary differences)
        pi_correlation = np.corrcoef(pg_results['pi'].values, allel_pi)[0, 1]
        assert pi_correlation > 0.95, f"Low correlation: {pi_correlation}"
        
        # Check individual windows where we have data
        for i, (pg_pi, allel_pi_val) in enumerate(zip(pg_results['pi'], allel_pi)):
            if not np.isnan(allel_pi_val) and not np.isnan(pg_pi):
                # Allow for some difference due to implementation details
                rel_diff = abs(pg_pi - allel_pi_val) / max(pg_pi, allel_pi_val)
                assert rel_diff < 0.1, f"Window {i}: pg_gpu={pg_pi}, allel={allel_pi_val}, rel_diff={rel_diff}"
    
    def test_windowed_theta_comparison(self, large_test_data):
        """Compare windowed Watterson's theta with scikit-allel."""
        haplotypes = large_test_data['haplotypes']
        positions = large_test_data['positions']
        start, end = large_test_data['start'], large_test_data['end']
        
        window_size = 40000
        
        # pg_gpu windowed analysis
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pg_results = windowed_analysis(
            matrix,
            window_size=window_size,
            statistics=['theta_w'],
            progress_bar=False
        )
        
        # scikit-allel windowed theta
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        allel_theta, allel_windows, allel_n_bases, allel_counts = allel.windowed_watterson_theta(
            positions, ac, size=window_size, start=start, stop=end
        )
        
        # Compare results
        assert len(pg_results) == len(allel_theta)
        
        # Check correlation (if there's variance in the data)
        valid_mask = ~(np.isnan(pg_results['theta_w']) | np.isnan(allel_theta))
        if np.sum(valid_mask) > 1:
            # Check if there's sufficient variance for correlation
            pg_var = np.var(pg_results['theta_w'][valid_mask])
            allel_var = np.var(allel_theta[valid_mask])
            
            if pg_var > 1e-10 and allel_var > 1e-10:  # Both have meaningful variance
                theta_correlation = np.corrcoef(
                    pg_results['theta_w'][valid_mask],
                    allel_theta[valid_mask]
                )[0, 1]
                assert theta_correlation > 0.95, f"Low theta correlation: {theta_correlation}"
            else:
                # If no variance, check that values are similar
                mean_diff = np.mean(np.abs(pg_results['theta_w'][valid_mask] - allel_theta[valid_mask]))
                max_val = np.max(allel_theta[valid_mask])
                rel_diff = mean_diff / max_val if max_val > 0 else 0
                assert rel_diff < 0.01, f"Values differ too much when no variance: rel_diff={rel_diff}"
    
    def test_windowed_tajimas_d_comparison(self, large_test_data):
        """Compare windowed Tajima's D with scikit-allel."""
        haplotypes = large_test_data['haplotypes']
        positions = large_test_data['positions']
        start, end = large_test_data['start'], large_test_data['end']
        
        window_size = 60000
        
        # pg_gpu windowed analysis
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pg_results = windowed_analysis(
            matrix,
            window_size=window_size,
            statistics=['tajimas_d'],
            progress_bar=False
        )
        
        # scikit-allel windowed Tajima's D
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        allel_tajd, allel_windows, allel_counts = allel.windowed_tajima_d(
            positions, ac, size=window_size, start=start, stop=end
        )
        
        # Compare results
        assert len(pg_results) == len(allel_tajd)
        
        # Check correlation for valid values
        valid_mask = ~(np.isnan(pg_results['tajimas_d']) | np.isnan(allel_tajd))
        if np.sum(valid_mask) > 1:
            tajd_correlation = np.corrcoef(
                pg_results['tajimas_d'][valid_mask], 
                allel_tajd[valid_mask]
            )[0, 1]
            assert tajd_correlation > 0.95, f"Low Tajima's D correlation: {tajd_correlation}"


class TestWindowedDivergenceComparison:
    """Compare windowed divergence calculations with scikit-allel."""
    
    @pytest.fixture
    def population_windowed_data(self):
        """Create test data with population structure for windowed analysis."""
        np.random.seed(123)
        n_haplotypes = 60
        n_variants = 800
        
        # Create structured populations
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
        
        # Population differentiation gradient
        for i in range(0, n_variants, 80):
            end_i = min(i + 80, n_variants)
            diff_level = 0.1 + 0.3 * (i / n_variants)  # Increasing differentiation
            
            # Pop1: first 30 haplotypes
            p1 = min(0.8, 0.3 + diff_level)  # Cap at 0.8
            haplotypes[:30, i:end_i] = np.random.choice([0, 1], size=(30, end_i - i), p=[1-p1, p1])
            
            # Pop2: last 30 haplotypes  
            p2 = max(0.1, 0.3 - diff_level)  # Floor at 0.1
            haplotypes[30:, i:end_i] = np.random.choice([0, 1], size=(30, end_i - i), p=[1-p2, p2])
        
        positions = np.arange(n_variants) * 1000 + 5000
        
        return {
            'haplotypes': haplotypes,
            'positions': positions,
            'start': positions[0],
            'end': positions[-1],
            'pop1_indices': list(range(30)),
            'pop2_indices': list(range(30, 60))
        }
    
    def test_windowed_divergence_comparison(self, population_windowed_data):
        """Compare windowed divergence with scikit-allel."""
        haplotypes = population_windowed_data['haplotypes']
        positions = population_windowed_data['positions']
        start, end = population_windowed_data['start'], population_windowed_data['end']
        pop1_indices = population_windowed_data['pop1_indices']
        pop2_indices = population_windowed_data['pop2_indices']
        
        window_size = 50000
        
        # pg_gpu windowed analysis
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        matrix.sample_sets = {'pop1': pop1_indices, 'pop2': pop2_indices}
        
        analyzer = WindowedAnalyzer(
            window_size=window_size,
            statistics=['dxy'],
            populations=['pop1', 'pop2'],
            progress_bar=False
        )
        pg_results = analyzer.compute(matrix)
        
        # scikit-allel windowed divergence
        h = allel.HaplotypeArray(haplotypes.T)
        ac1 = h.count_alleles(subpop=pop1_indices)
        ac2 = h.count_alleles(subpop=pop2_indices)
        
        allel_dxy, allel_windows, allel_n_bases, allel_counts = allel.windowed_divergence(
            positions, ac1, ac2, size=window_size, start=start, stop=end
        )
        
        # Compare results
        assert len(pg_results) == len(allel_dxy)
        
        # Check correlation for valid values
        pg_dxy = pg_results['dxy_pop1_pop2'].values
        valid_mask = ~(np.isnan(pg_dxy) | np.isnan(allel_dxy))
        
        if np.sum(valid_mask) > 1:
            dxy_correlation = np.corrcoef(pg_dxy[valid_mask], allel_dxy[valid_mask])[0, 1]
            assert dxy_correlation > 0.95, f"Low Dxy correlation: {dxy_correlation}"


class TestNonOverlappingWindows:
    """Test non-overlapping windows against scikit-allel."""
    
    def test_non_overlapping_diversity(self):
        """Test non-overlapping windows produce identical results."""
        np.random.seed(789)
        n_haplotypes = 40
        n_variants = 400
        
        # Simple regular pattern
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.6, 0.4])
        positions = np.arange(n_variants) * 1000 + 1000
        start, end = positions[0], positions[-1]
        
        window_size = 50000
        
        # pg_gpu non-overlapping windows
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pg_results = windowed_analysis(
            matrix,
            window_size=window_size,
            step_size=window_size,  # Non-overlapping
            statistics=['pi', 'theta_w'],
            progress_bar=False
        )
        
        # scikit-allel non-overlapping windows
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        
        allel_pi, allel_windows_pi, _, _ = allel.windowed_diversity(
            positions, ac, size=window_size, step=window_size, start=start, stop=end
        )
        
        allel_theta, allel_windows_theta, _, _ = allel.windowed_watterson_theta(
            positions, ac, size=window_size, step=window_size, start=start, stop=end
        )
        
        # Results should be very close for non-overlapping windows
        assert len(pg_results) == len(allel_pi) == len(allel_theta)
        
        for i in range(len(pg_results)):
            if not np.isnan(allel_pi[i]) and not np.isnan(pg_results.iloc[i]['pi']):
                rel_diff_pi = abs(pg_results.iloc[i]['pi'] - allel_pi[i]) / max(pg_results.iloc[i]['pi'], allel_pi[i])
                assert rel_diff_pi < 0.05, f"Pi difference too large in window {i}: {rel_diff_pi}"
            
            if not np.isnan(allel_theta[i]) and not np.isnan(pg_results.iloc[i]['theta_w']):
                rel_diff_theta = abs(pg_results.iloc[i]['theta_w'] - allel_theta[i]) / max(pg_results.iloc[i]['theta_w'], allel_theta[i])
                assert rel_diff_theta < 0.05, f"Theta difference too large in window {i}: {rel_diff_theta}"


class TestWindowBoundaryHandling:
    """Test that window boundaries are handled consistently."""
    
    def test_boundary_consistency(self):
        """Test that window boundaries produce consistent results."""
        # Create data where we know the exact window contents
        n_haplotypes = 20
        n_variants = 100
        
        # Create data with clear window structure
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
        
        # First 50 variants: high diversity
        haplotypes[:10, :50] = 1
        
        # Last 50 variants: low diversity  
        haplotypes[:2, 50:] = 1
        
        # Positions every 1kb
        positions = np.arange(n_variants) * 1000 + 1000
        start, end = positions[0], positions[-1]
        
        window_size = 50000  # Should split exactly at variant 50
        
        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pg_results = windowed_analysis(
            matrix,
            window_size=window_size,
            step_size=window_size,
            statistics=['pi'],
            progress_bar=False
        )
        
        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        allel_pi, allel_windows, _, _ = allel.windowed_diversity(
            positions, ac, size=window_size, step=window_size, start=start, stop=end
        )
        
        # Should have 2 windows
        assert len(pg_results) == len(allel_pi) == 2
        
        # First window should have higher diversity than second
        assert pg_results.iloc[0]['pi'] > pg_results.iloc[1]['pi']
        assert allel_pi[0] > allel_pi[1]
        
        # Values should be close
        for i in range(2):
            if not np.isnan(allel_pi[i]):
                rel_diff = abs(pg_results.iloc[i]['pi'] - allel_pi[i]) / max(pg_results.iloc[i]['pi'], allel_pi[i])
                assert rel_diff < 0.1, f"Window {i} pi difference: {rel_diff}"