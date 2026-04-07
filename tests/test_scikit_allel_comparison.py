"""
Direct comparison tests between pg_gpu and scikit-allel implementations.

These tests validate that pg_gpu produces identical results to scikit-allel
for all shared statistics, ensuring correctness of our implementations.
"""

import pytest
import numpy as np
import cupy as cp
import allel
from pg_gpu import HaplotypeMatrix
from pg_gpu import diversity, divergence


class TestDiversityComparison:
    """Compare diversity statistics between pg_gpu and scikit-allel."""

    @pytest.fixture
    def test_data(self):
        """Create test haplotype data for comparison."""
        # Create deterministic test data
        np.random.seed(42)
        n_haplotypes = 50
        n_variants = 100

        # Generate haplotypes with various allele frequencies
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)

        # Different frequency classes
        # 20 sites with low frequency (10%)
        haplotypes[:5, :20] = 1

        # 20 sites with medium frequency (30%)
        haplotypes[:15, 20:40] = 1

        # 20 sites with high frequency (50%)
        haplotypes[:25, 40:60] = 1

        # 20 sites with very high frequency (80%)
        haplotypes[:40, 60:80] = 1

        # 20 singletons
        for i in range(20):
            haplotypes[i, 80 + i] = 1

        positions = np.arange(n_variants) * 1000 + 1000

        return {
            'haplotypes': haplotypes,
            'positions': positions,
            'start': positions[0],
            'end': positions[-1],
            'span': positions[-1] - positions[0] + 1
        }

    def test_segregating_sites(self, test_data):
        """Test segregating sites count matches scikit-allel."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, test_data['start'], test_data['end'])
        seg_sites_pg = diversity.segregating_sites(matrix)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        seg_sites_allel = np.sum(ac.is_segregating())

        assert seg_sites_pg == seg_sites_allel

    def test_nucleotide_diversity(self, test_data):
        """Test nucleotide diversity matches scikit-allel."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']
        start, end = test_data['start'], test_data['end']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac, start=start, stop=end)

        # Should match within numerical precision (allow for span calculation differences)
        assert abs(pi_pg - pi_allel) < 1e-8

    def test_watterson_theta(self, test_data):
        """Test Watterson's theta matches scikit-allel."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']
        start, end = test_data['start'], test_data['end']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        theta_allel = allel.watterson_theta(positions, ac, start=start, stop=end)

        # Should match within numerical precision
        assert abs(theta_pg - theta_allel) < 1e-8

    def test_tajimas_d(self, test_data):
        """Test Tajima's D matches scikit-allel."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, test_data['start'], test_data['end'])
        tajd_pg = diversity.tajimas_d(matrix)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        tajd_allel = allel.tajima_d(ac, pos=positions)

        # Should match within numerical precision
        assert abs(tajd_pg - tajd_allel) < 1e-8

    def test_allele_frequency_spectrum(self, test_data):
        """Test AFS matches scikit-allel SFS."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, test_data['start'], test_data['end'])
        afs_pg = diversity.allele_frequency_spectrum(matrix)
        if hasattr(afs_pg, 'get'):
            afs_pg = np.asarray(afs_pg)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        sfs_allel = allel.sfs(ac[:, 1])  # SFS for derived allele

        # pg_gpu AFS has n+1 bins, scikit-allel SFS length varies based on max frequency
        # Compare the overlapping portions (skip index 0, and match lengths)
        sfs_length = len(sfs_allel) - 1  # Exclude sfs_allel[0] since we start from index 1
        np.testing.assert_array_equal(afs_pg[1:1+sfs_length], sfs_allel[1:])

    def test_singleton_count(self, test_data):
        """Test singleton count matches scikit-allel."""
        haplotypes = test_data['haplotypes']
        positions = test_data['positions']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, test_data['start'], test_data['end'])
        singletons_pg = diversity.singleton_count(matrix)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        is_singleton = ac.is_singleton(1)  # singletons for alt allele
        singletons_allel = np.sum(is_singleton)

        assert singletons_pg == singletons_allel


class TestDivergenceComparison:
    """Compare divergence statistics between pg_gpu and scikit-allel."""

    @pytest.fixture
    def population_data(self):
        """Create test data with population structure."""
        np.random.seed(123)
        n_haplotypes = 40
        n_variants = 80

        # Create differentiated populations
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)

        # Population 1: first 20 haplotypes
        # High frequency for first 40 variants
        haplotypes[:20, :40] = np.random.choice([0, 1], size=(20, 40), p=[0.3, 0.7])
        # Low frequency for last 40 variants
        haplotypes[:20, 40:] = np.random.choice([0, 1], size=(20, 40), p=[0.8, 0.2])

        # Population 2: last 20 haplotypes
        # Low frequency for first 40 variants
        haplotypes[20:, :40] = np.random.choice([0, 1], size=(20, 40), p=[0.7, 0.3])
        # High frequency for last 40 variants
        haplotypes[20:, 40:] = np.random.choice([0, 1], size=(20, 40), p=[0.2, 0.8])

        positions = np.arange(n_variants) * 1000 + 1000

        return {
            'haplotypes': haplotypes,
            'positions': positions,
            'start': positions[0],
            'end': positions[-1],
            'pop1_indices': list(range(20)),
            'pop2_indices': list(range(20, 40))
        }

    def test_sequence_divergence_dxy(self, population_data):
        """Test Dxy matches scikit-allel sequence divergence."""
        haplotypes = population_data['haplotypes']
        positions = population_data['positions']
        start, end = population_data['start'], population_data['end']
        pop1_indices = population_data['pop1_indices']
        pop2_indices = population_data['pop2_indices']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        matrix.sample_sets = {'pop1': pop1_indices, 'pop2': pop2_indices}
        # pg_gpu: per-base by default (auto span normalization)
        dxy_pg = divergence.dxy(matrix, 'pop1', 'pop2', per_site=False)

        # scikit-allel: also per-base
        h = allel.HaplotypeArray(haplotypes.T)
        ac1 = h.count_alleles(subpop=pop1_indices)
        ac2 = h.count_alleles(subpop=pop2_indices)
        dxy_allel = allel.sequence_divergence(positions, ac1, ac2, start=start, stop=end)

        # Should match within numerical precision
        assert abs(dxy_pg - dxy_allel) < 1e-8

    def test_within_population_diversity(self, population_data):
        """Test within-population diversity matches scikit-allel."""
        haplotypes = population_data['haplotypes']
        positions = population_data['positions']
        start, end = population_data['start'], population_data['end']
        pop1_indices = population_data['pop1_indices']

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        matrix.sample_sets = {'pop1': pop1_indices}
        pi_pg = diversity.pi(matrix, 'pop1', span_normalize=True)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac1 = h.count_alleles(subpop=pop1_indices)
        pi_allel = allel.sequence_diversity(positions, ac1, start=start, stop=end)

        # Should match within numerical precision (allow for span calculation differences)
        assert abs(pi_pg - pi_allel) < 1e-8


class TestEdgeCasesComparison:
    """Test edge cases against scikit-allel."""

    def test_no_segregating_sites(self):
        """Test behavior when no segregating sites."""
        # All sites fixed for reference allele
        haplotypes = np.zeros((20, 50), dtype=np.int8)
        positions = np.arange(50) * 1000 + 1000
        start, end = positions[0], positions[-1]

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        tajd_pg = diversity.tajimas_d(matrix)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_allel = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_allel = allel.tajima_d(ac, pos=positions)

        # All should be 0 or NaN
        assert pi_pg == pi_allel == 0.0
        assert theta_pg == theta_allel == 0.0
        assert np.isnan(tajd_pg) and np.isnan(tajd_allel)

    def test_all_singletons(self):
        """Test behavior with all singleton variants."""
        n_haplotypes = 20
        n_variants = 20

        # Create all singletons
        haplotypes = np.zeros((n_haplotypes, n_variants), dtype=np.int8)
        for i in range(n_variants):
            haplotypes[i, i] = 1

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
        pi_allel = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_allel = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_allel = allel.tajima_d(ac, pos=positions)

        # Should match within reasonable precision (edge cases may have larger span differences)
        assert abs(pi_pg - pi_allel) < 1e-6
        assert abs(theta_pg - theta_allel) < 1e-6
        assert abs(tajd_pg - tajd_allel) < 1e-6

    def test_single_segregating_site(self):
        """Test behavior with only one segregating site."""
        n_haplotypes = 10

        # Only one segregating site
        haplotypes = np.zeros((n_haplotypes, 3), dtype=np.int8)
        haplotypes[:5, 1] = 1  # Middle site is segregating

        positions = np.array([1000, 2000, 3000])
        start, end = positions[0], positions[-1]

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_allel = allel.watterson_theta(positions, ac, start=start, stop=end)

        # Should match within reasonable precision (edge cases may have larger span differences)
        assert abs(pi_pg - pi_allel) < 1e-6
        assert abs(theta_pg - theta_allel) < 1e-6


class TestGPUConsistency:
    """Test that GPU and CPU versions give identical results to scikit-allel."""

    def test_gpu_cpu_consistency(self):
        """Test GPU and CPU versions both match scikit-allel."""
        # Create test data
        np.random.seed(456)
        haplotypes = np.random.randint(0, 2, size=(30, 100), dtype=np.int8)
        positions = np.arange(100) * 1000 + 1000
        start, end = positions[0], positions[-1]

        # scikit-allel reference
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_ref = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_ref = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_ref = allel.tajima_d(ac, pos=positions)

        # pg_gpu CPU version
        matrix_cpu = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_cpu = diversity.pi(matrix_cpu, span_normalize=True)
        theta_cpu = diversity.theta_w(matrix_cpu, span_normalize=True)
        tajd_cpu = diversity.tajimas_d(matrix_cpu)

        # pg_gpu GPU version (if available)
        if cp.cuda.is_available():
            matrix_gpu = HaplotypeMatrix(haplotypes, positions, start, end)
            matrix_gpu.transfer_to_gpu()

            pi_gpu = diversity.pi(matrix_gpu, span_normalize=True)
            theta_gpu = diversity.theta_w(matrix_gpu, span_normalize=True)
            tajd_gpu = diversity.tajimas_d(matrix_gpu)

            # All versions should match
            assert abs(pi_cpu - pi_ref) < 1e-8
            assert abs(pi_gpu - pi_ref) < 1e-8
            assert abs(theta_cpu - theta_ref) < 1e-8
            assert abs(theta_gpu - theta_ref) < 1e-8
            assert abs(tajd_cpu - tajd_ref) < 1e-8
            assert abs(tajd_gpu - tajd_ref) < 1e-8
        else:
            # Just test CPU version
            assert abs(pi_cpu - pi_ref) < 1e-8
            assert abs(theta_cpu - theta_ref) < 1e-8
            assert abs(tajd_cpu - tajd_ref) < 1e-8


class TestRandomDataComparison:
    """Test with multiple random datasets to catch edge cases."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
    def test_random_datasets(self, seed):
        """Test multiple random datasets against scikit-allel."""
        np.random.seed(seed)

        # Random dataset parameters
        n_haplotypes = np.random.randint(20, 100)
        n_variants = np.random.randint(50, 200)

        # Generate random haplotypes
        haplotypes = np.random.randint(0, 2, size=(n_haplotypes, n_variants), dtype=np.int8)
        positions = np.sort(np.random.randint(1000, 100000, size=n_variants))
        start, end = positions[0], positions[-1]

        # Skip if no segregating sites
        h_test = allel.HaplotypeArray(haplotypes.T)
        ac_test = h_test.count_alleles()
        if np.sum(ac_test.is_segregating()) == 0:
            pytest.skip("No segregating sites in random data")

        # pg_gpu
        matrix = HaplotypeMatrix(haplotypes, positions, start, end)
        pi_pg = diversity.pi(matrix, span_normalize=True)
        theta_pg = diversity.theta_w(matrix, span_normalize=True)
        tajd_pg = diversity.tajimas_d(matrix)

        # scikit-allel
        h = allel.HaplotypeArray(haplotypes.T)
        ac = h.count_alleles()
        pi_allel = allel.sequence_diversity(positions, ac, start=start, stop=end)
        theta_allel = allel.watterson_theta(positions, ac, start=start, stop=end)
        tajd_allel = allel.tajima_d(ac, pos=positions)

        # Should match within numerical precision (allow for span calculation differences)
        assert abs(pi_pg - pi_allel) < 1e-8, f"Pi mismatch for seed {seed}"
        assert abs(theta_pg - theta_allel) < 1e-8, f"Theta mismatch for seed {seed}"
        assert abs(tajd_pg - tajd_allel) < 1e-8, f"Tajima's D mismatch for seed {seed}"
