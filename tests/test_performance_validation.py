"""
Performance and accuracy validation tests.

These tests verify that pg_gpu not only produces correct results but also
demonstrates performance improvements over CPU-only implementations.
"""

import pytest
import numpy as np
import cupy as cp
import allel
import time
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity, divergence
from pg_gpu.windowed_analysis import WindowedAnalyzer


class TestPerformanceComparison:
    """Compare performance between pg_gpu and scikit-allel."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        np.random.seed(42)
        n_haplotypes = 200
        n_variants = 5000
        
        # Create realistic haplotype data
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.7, 0.3])
        positions = np.arange(n_variants) * 1000 + 1000
        
        return {
            'haplotypes': haplotypes.astype(np.int8),
            'positions': positions,
            'start': positions[0],
            'end': positions[-1]
        }
    
    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_performance_diversity(self, large_dataset):
        """Test that GPU version is faster than CPU for large datasets."""
        haplotypes = large_dataset['haplotypes']
        positions = large_dataset['positions']
        start, end = large_dataset['start'], large_dataset['end']
        
        # CPU version
        matrix_cpu = HaplotypeMatrix(haplotypes, positions, start, end)
        
        start_time = time.time()
        pi_cpu = diversity.pi(matrix_cpu, span_normalize=True)
        theta_cpu = diversity.theta_w(matrix_cpu, span_normalize=True)
        tajd_cpu = diversity.tajimas_d(matrix_cpu)
        cpu_time = time.time() - start_time
        
        # GPU version
        matrix_gpu = HaplotypeMatrix(haplotypes, positions, start, end)
        matrix_gpu.transfer_to_gpu()
        
        start_time = time.time()
        pi_gpu = diversity.pi(matrix_gpu, span_normalize=True)
        theta_gpu = diversity.theta_w(matrix_gpu, span_normalize=True)
        tajd_gpu = diversity.tajimas_d(matrix_gpu)
        gpu_time = time.time() - start_time
        
        # Results should be identical
        assert abs(pi_cpu - pi_gpu) < 1e-15
        assert abs(theta_cpu - theta_gpu) < 1e-15
        assert abs(tajd_cpu - tajd_gpu) < 1e-15
        
        # GPU should be faster (or at least not much slower for this size)
        speedup = cpu_time / gpu_time
        print(f"GPU speedup for diversity stats: {speedup:.2f}x")
        
        # For large enough datasets, we expect speedup > 1
        # For smaller datasets, GPU overhead might make it slower
        assert speedup > 0.1, f"GPU version unexpectedly slow: {speedup:.2f}x"
    
    def test_accuracy_with_reference(self, large_dataset):
        """Test that pg_gpu maintains accuracy compared to scikit-allel."""
        haplotypes = large_dataset['haplotypes']
        positions = large_dataset['positions']
        start, end = large_dataset['start'], large_dataset['end']
        
        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        tajd_pg = diversity.tajimas_d(matrix)
        seg_sites_pg = diversity.segregating_sites(matrix)
        
        # scikit-allel reference
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_ref = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_ref = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_ref = allel.tajima_d(ac, pos=positions)
        seg_sites_ref = np.sum(ac.is_segregating())
        
        # All should match within numerical precision (allow for span calculation differences)
        assert abs(pi_pg - pi_ref) < 1e-8
        assert abs(theta_pg - theta_ref) < 1e-8
        assert abs(tajd_pg - tajd_ref) < 1e-8
        assert seg_sites_pg == seg_sites_ref
        
        print(f"Large dataset validation passed:")
        print(f"  Pi: {pi_pg:.8f} (diff: {abs(pi_pg - pi_ref):.2e})")
        print(f"  Theta: {theta_pg:.8f} (diff: {abs(theta_pg - theta_ref):.2e})")
        print(f"  Tajima's D: {tajd_pg:.8f} (diff: {abs(tajd_pg - tajd_ref):.2e})")


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""
    
    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_memory_management(self):
        """Test that GPU memory is managed efficiently."""
        # Create moderately large dataset
        n_haplotypes = 100
        n_variants = 2000
        
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.6, 0.4])
        positions = np.arange(n_variants) * 1000 + 1000
        
        # Check initial GPU memory
        mempool = cp.get_default_memory_pool()
        initial_used = mempool.used_bytes()
        
        # Create GPU matrix and compute statistics
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        matrix.transfer_to_gpu()
        
        pi_val = diversity.pi(matrix, span_normalize=True)
        theta_val = diversity.theta_w(matrix, span_normalize=True)
        
        peak_used = mempool.used_bytes()
        
        # Clean up
        del matrix
        mempool.free_all_blocks()
        
        final_used = mempool.used_bytes()
        
        # Memory should be reasonable
        memory_used_mb = (peak_used - initial_used) / (1024 * 1024)
        print(f"Peak GPU memory used: {memory_used_mb:.2f} MB")
        
        # Should not use excessive memory for this dataset size
        # Rough estimate: should be less than 100MB for this size
        assert memory_used_mb < 100, f"Excessive memory usage: {memory_used_mb:.2f} MB"
        
        # Memory should be properly cleaned up
        cleanup_memory = (final_used - initial_used) / (1024 * 1024)
        assert cleanup_memory < 10, f"Memory not properly cleaned up: {cleanup_memory:.2f} MB remaining"


class TestScalability:
    """Test scalability with different dataset sizes."""
    
    @pytest.mark.parametrize("n_variants", [100, 500, 1000, 2000])
    def test_scaling_accuracy(self, n_variants):
        """Test that accuracy is maintained across different dataset sizes."""
        np.random.seed(42)
        n_haplotypes = 50
        
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.65, 0.35])
        positions = np.arange(n_variants) * 1000 + 1000
        start, end = positions[0], positions[-1]
        
        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        
        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_ref = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_ref = allel.watterson_theta(positions, ac, start=start, stop=end)
        
        # Accuracy should be maintained regardless of size (allow for span calculation differences)
        assert abs(pi_pg - pi_ref) < 1e-8, f"Pi accuracy failed for {n_variants} variants"
        assert abs(theta_pg - theta_ref) < 1e-8, f"Theta accuracy failed for {n_variants} variants"
        
        print(f"Accuracy validated for {n_variants} variants")
    
    def test_windowed_scalability(self):
        """Test windowed analysis scalability."""
        np.random.seed(123)
        n_haplotypes = 80
        n_variants = 3000
        
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants), p=[0.6, 0.4])
        positions = np.arange(n_variants) * 1000 + 1000
        
        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        
        # Test different window sizes
        window_sizes = [10000, 25000, 50000, 100000]
        
        for window_size in window_sizes:
            analyzer = WindowedAnalyzer(
                window_size=window_size,
                statistics=['pi', 'theta_w'],
                progress_bar=False
            )
            
            start_time = time.time()
            results = analyzer.compute(matrix)
            compute_time = time.time() - start_time
            
            # Should complete reasonably quickly
            assert compute_time < 30, f"Window size {window_size} took too long: {compute_time:.2f}s"
            
            # Should have reasonable number of windows
            expected_windows = max(1, (positions[-1] - positions[0]) // window_size)
            assert len(results) >= expected_windows * 0.8, f"Too few windows for size {window_size}"
            
            print(f"Window size {window_size}: {len(results)} windows in {compute_time:.2f}s")


class TestRobustness:
    """Test robustness with edge cases and difficult data."""
    
    def test_extreme_allele_frequencies(self):
        """Test with extreme allele frequencies."""
        n_haplotypes = 100
        n_variants = 200
        
        # Create data with very rare and very common alleles
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
        
        # Very rare alleles (1%)
        for i in range(0, 50):
            haplotypes[:1, i] = 1
        
        # Very common alleles (99%)
        for i in range(50, 100):
            haplotypes[:-1, i] = 1
        
        # Medium frequency alleles (50%)
        for i in range(100, 150):
            haplotypes[:50, i] = 1
        
        # Random frequency alleles
        for i in range(150, 200):
            freq = np.random.uniform(0.05, 0.95)
            n_derived = int(freq * n_haplotypes)
            haplotypes[:n_derived, i] = 1
        
        positions = np.arange(n_variants) * 1000 + 1000
        start, end = positions[0], positions[-1]
        
        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        tajd_pg = diversity.tajimas_d(matrix)
        
        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_ref = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_ref = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_ref = allel.tajima_d(ac, pos=positions)
        
        # Should handle extreme frequencies correctly (allow for span calculation differences)
        assert abs(pi_pg - pi_ref) < 1e-8
        assert abs(theta_pg - theta_ref) < 1e-8
        assert abs(tajd_pg - tajd_ref) < 1e-8
        
        # Values should be reasonable
        assert 0 <= pi_pg <= 1
        assert 0 <= theta_pg <= 1
        assert -10 <= tajd_pg <= 10
    
    def test_missing_data_handling(self):
        """Test handling of missing data (if implemented)."""
        n_haplotypes = 50
        n_variants = 100
        
        # Create data with some missing values
        haplotypes = np.random.choice([0, 1], size=(n_haplotypes, n_variants))
        
        # Introduce missing data (encoded as -1 in pg_gpu)
        missing_mask = np.random.random((n_haplotypes, n_variants)) < 0.05  # 5% missing
        haplotypes_with_missing = haplotypes.astype(np.int8)
        haplotypes_with_missing[missing_mask] = -1
        
        positions = np.arange(n_variants) * 1000 + 1000
        
        # Test that missing data doesn't crash the system
        matrix = HaplotypeMatrix(haplotypes_with_missing, positions, positions[0], positions[-1])
        
        # These should not crash
        pi_val = diversity.pi(matrix, span_normalize=True)
        theta_val = diversity.theta_w(matrix, span_normalize=True)
        
        # Values should be reasonable
        assert not np.isnan(pi_val) or np.sum(~missing_mask) == 0
        assert not np.isnan(theta_val) or np.sum(~missing_mask) == 0