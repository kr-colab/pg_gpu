"""
Tests for windowed analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import (
    WindowedAnalyzer, windowed_analysis, WindowParams,
    WindowIterator, WindowData
)


class TestWindowIterator:
    """Test window iteration functionality."""

    def test_bp_windows_non_overlapping(self):
        """Test base pair windows without overlap."""
        # Create test data
        n_variants = 100
        positions = np.linspace(0, 100000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(10, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)
        params = WindowParams(window_type='bp', window_size=10000, step_size=10000)

        iterator = WindowIterator(matrix, params)
        windows = list(iterator)

        # Should have 10 windows
        assert len(windows) == 10

        # Check first and last windows
        assert windows[0].start == 0
        assert windows[0].end == 10000
        assert windows[-1].start == 90000
        assert windows[-1].end == 100000

        # Check no overlap
        for i in range(len(windows) - 1):
            assert windows[i].end == windows[i+1].start

    def test_bp_windows_overlapping(self):
        """Test base pair windows with overlap."""
        n_variants = 100
        positions = np.linspace(0, 100000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(10, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)
        params = WindowParams(window_type='bp', window_size=10000, step_size=5000)

        iterator = WindowIterator(matrix, params)
        windows = list(iterator)

        # Should have 19 windows with 50% overlap
        assert len(windows) >= 18

        # Check overlap
        for i in range(len(windows) - 1):
            assert windows[i].start + 5000 == windows[i+1].start

    def test_snp_windows(self):
        """Test SNP-based windows."""
        n_variants = 100
        positions = np.arange(n_variants) * 1000
        haplotypes = np.random.randint(0, 2, size=(10, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)
        params = WindowParams(window_type='snp', window_size=10, step_size=5)

        iterator = WindowIterator(matrix, params)
        windows = list(iterator)

        # Check window sizes
        for window in windows[:-1]:  # Except possibly the last
            assert window.n_variants == 10

        # Check step size
        assert len(windows) == 19  # (100 - 10) / 5 + 1

    def test_region_windows(self):
        """Test custom region windows."""
        n_variants = 100
        positions = np.linspace(0, 100000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(10, n_variants))

        regions = pd.DataFrame({
            'chrom': [1, 1, 1],
            'start': [10000, 30000, 70000],
            'end': [20000, 50000, 90000]
        })

        matrix = HaplotypeMatrix(haplotypes, positions)
        params = WindowParams(window_type='regions', window_size=0,
                            step_size=0, regions=regions)

        iterator = WindowIterator(matrix, params)
        windows = list(iterator)

        assert len(windows) == 3
        assert windows[0].start == 10000
        assert windows[0].end == 20000
        assert windows[1].start == 30000
        assert windows[1].end == 50000


class TestStatisticsComputer:
    """Test statistics computation."""

    def test_single_pop_stats(self):
        """Test single population statistics."""
        n_variants = 50
        n_samples = 20
        positions = np.arange(n_variants) * 1000

        # Create haplotypes with known properties
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        # Add some variation
        haplotypes[:10, :25] = 1  # First half samples, first half variants

        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])

        from pg_gpu.windowed_analysis import StatisticsComputer
        computer = StatisticsComputer(['pi', 'n_variants'])

        window = WindowData(
            chrom=1, start=0, end=50000, center=25000,
            matrix=matrix, n_variants=n_variants, window_id=0
        )

        results = computer.compute(window)

        assert 'pi' in results
        assert 'n_variants' in results
        assert results['n_variants'] == n_variants
        assert results['pi'] > 0  # Should have some diversity

    def test_two_pop_stats(self):
        """Test two population statistics."""
        n_variants = 50
        n_samples = 20
        positions = np.arange(n_variants) * 1000

        # Create differentiated populations
        haplotypes = np.zeros((n_samples, n_variants), dtype=int)
        haplotypes[:10, :25] = 1  # Pop1 fixed for first 25 variants
        haplotypes[10:, 25:] = 1  # Pop2 fixed for last 25 variants

        matrix = HaplotypeMatrix(haplotypes, positions, 0, positions[-1])
        matrix.sample_sets = {
            'pop1': list(range(10)),
            'pop2': list(range(10, 20))
        }

        from pg_gpu.windowed_analysis import StatisticsComputer
        computer = StatisticsComputer(['fst'], populations=['pop1', 'pop2'])

        window = WindowData(
            chrom=1, start=0, end=50000, center=25000,
            matrix=matrix, n_variants=n_variants, window_id=0
        )

        results = computer.compute(window)

        assert 'fst_pop1_pop2' in results
        # Should show differentiation
        assert results['fst_pop1_pop2'] > 0

    def test_empty_window(self):
        """Test handling of windows with no variants."""
        positions = np.array([1000, 2000, 3000])
        haplotypes = np.random.randint(0, 2, size=(10, 3))

        matrix = HaplotypeMatrix(haplotypes, positions)

        from pg_gpu.windowed_analysis import StatisticsComputer
        computer = StatisticsComputer(['pi', 'tajimas_d'])

        # Create empty window
        empty_matrix = matrix.get_subset(np.array([]))
        window = WindowData(
            chrom=1, start=5000, end=6000, center=5500,
            matrix=empty_matrix, n_variants=0, window_id=0
        )

        results = computer.compute(window)

        assert np.isnan(results['pi'])
        assert np.isnan(results['tajimas_d'])


class TestWindowedAnalyzer:
    """Test main WindowedAnalyzer class."""

    def test_basic_analysis(self):
        """Test basic windowed analysis."""
        n_variants = 1000
        n_samples = 50
        positions = np.sort(np.random.randint(0, 1000000, size=n_variants))
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])

        analyzer = WindowedAnalyzer(
            window_size=100000,
            step_size=50000,
            statistics=['pi', 'n_variants'],
            progress_bar=False
        )

        results = analyzer.compute(matrix)

        assert isinstance(results, pd.DataFrame)
        assert 'pi' in results.columns
        assert 'n_variants' in results.columns
        assert 'start' in results.columns
        assert 'end' in results.columns
        assert len(results) > 0

    def test_population_analysis(self):
        """Test analysis with multiple populations."""
        n_variants = 500
        n_samples = 40
        positions = np.sort(np.random.randint(0, 500000, size=n_variants))
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions, positions[0], positions[-1])
        matrix.sample_sets = {
            'popA': list(range(20)),
            'popB': list(range(20, 40))
        }

        analyzer = WindowedAnalyzer(
            window_size=100000,
            statistics=['pi', 'fst'],
            populations=['popA', 'popB'],
            progress_bar=False
        )

        results = analyzer.compute(matrix)

        assert 'pi_popA' in results.columns
        assert 'pi_popB' in results.columns
        assert 'fst_popA_popB' in results.columns

    def test_convenience_function(self):
        """Test windowed_analysis convenience function."""
        n_variants = 200
        n_samples = 20
        positions = np.linspace(0, 100000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)

        results = windowed_analysis(
            matrix,
            window_size=20000,
            statistics=['pi'],
            progress_bar=False
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5  # 100k / 20k

    def test_streaming_computation(self):
        """Test streaming mode."""
        n_variants = 500
        n_samples = 20
        positions = np.linspace(0, 500000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)

        analyzer = WindowedAnalyzer(
            window_size=50000,
            statistics=['pi'],
            progress_bar=False
        )

        batches = list(analyzer.compute_streaming(matrix, batch_size=3))

        assert len(batches) > 1
        assert all(isinstance(batch, pd.DataFrame) for batch in batches)

        # Combine all batches
        all_results = pd.concat(batches, ignore_index=True)
        assert len(all_results) == 10  # 500k / 50k

    @pytest.mark.skipif(not hasattr(np, 'GPU'), reason="GPU not available")
    def test_gpu_computation(self):
        """Test computation on GPU."""
        n_variants = 500
        n_samples = 50
        positions = np.sort(np.random.randint(0, 500000, size=n_variants))
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)
        matrix.transfer_to_gpu()

        analyzer = WindowedAnalyzer(
            window_size=100000,
            statistics=['pi', 'tajimas_d'],
            progress_bar=False
        )

        results = analyzer.compute(matrix)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        assert matrix.device == 'GPU'  # Should stay on GPU


class TestCustomStatistics:
    """Test custom statistics functionality."""

    def test_custom_function(self):
        """Test using custom statistic function."""
        def allele_count_variance(window):
            """Variance in allele counts across variants."""
            matrix = window.matrix
            if matrix.device == 'GPU':
                import cupy as cp
                counts = cp.sum(matrix.haplotypes, axis=0)
                return float(cp.var(counts).get())
            else:
                counts = np.sum(matrix.haplotypes, axis=0)
                return float(np.var(counts))

        n_variants = 200
        n_samples = 30
        positions = np.linspace(0, 100000, n_variants).astype(int)
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)

        analyzer = WindowedAnalyzer(
            window_size=25000,
            statistics=['pi', allele_count_variance],
            progress_bar=False
        )

        results = analyzer.compute(matrix)

        assert 'allele_count_variance' in results.columns
        assert results['allele_count_variance'].notna().any()

    def test_custom_with_kwargs(self):
        """Test custom statistic with keyword arguments."""
        def maf_above_threshold(window, threshold=0.05):
            """Count variants with MAF above threshold."""
            matrix = window.matrix
            n_haps = matrix.num_haplotypes
            if matrix.device == 'GPU':
                import cupy as cp
                af = cp.sum(matrix.haplotypes, axis=0) / n_haps
                maf = cp.minimum(af, 1 - af)
                return int(cp.sum(maf > threshold).get())
            else:
                af = np.sum(matrix.haplotypes, axis=0) / n_haps
                maf = np.minimum(af, 1 - af)
                return int(np.sum(maf > threshold))

        n_variants = 100
        n_samples = 40
        positions = np.arange(n_variants) * 1000
        haplotypes = np.random.randint(0, 2, size=(n_samples, n_variants))

        matrix = HaplotypeMatrix(haplotypes, positions)

        analyzer = WindowedAnalyzer(
            window_size=25000,
            statistics=[maf_above_threshold],
            custom_stat_kwargs={'maf_above_threshold': {'threshold': 0.1}},
            progress_bar=False
        )

        results = analyzer.compute(matrix)

        assert 'maf_above_threshold' in results.columns


class TestChunkedFused:
    """Test that chunked fused path matches single-shot fused."""

    @pytest.fixture
    def matrix_with_pops(self):
        import msprime
        ts = msprime.sim_ancestry(
            samples=50, sequence_length=500_000,
            recombination_rate=1e-8, population_size=10_000,
            random_seed=42, ploidy=2)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
        hm = HaplotypeMatrix.from_ts(ts)
        n = hm.num_haplotypes
        hm.sample_sets = {
            "pop1": list(range(n // 2)),
            "pop2": list(range(n // 2, n)),
        }
        return hm

    def test_single_pop_chunked_matches_fused(self, matrix_with_pops):
        from pg_gpu.windowed_analysis import (
            windowed_statistics_fused,
            windowed_statistics_fused_chunked,
        )
        from pg_gpu import _memutil

        hm = matrix_with_pops
        hm.transfer_to_gpu()

        bp_bins = np.arange(0, 500_001, 50_000, dtype=np.float64)

        r1 = windowed_statistics_fused(
            hm, bp_bins=bp_bins,
            statistics=('pi', 'theta_w', 'tajimas_d', 'segregating_sites'))

        # Force small chunks
        orig = _memutil.estimate_fused_chunk_size
        _memutil.estimate_fused_chunk_size = lambda n, memory_fraction=0.35: 500
        try:
            r2 = windowed_statistics_fused_chunked(
                hm, bp_bins=bp_bins,
                statistics=('pi', 'theta_w', 'tajimas_d', 'segregating_sites'))
        finally:
            _memutil.estimate_fused_chunk_size = orig

        for k in ('pi', 'theta_w', 'tajimas_d', 'segregating_sites'):
            np.testing.assert_allclose(r1[k], r2[k], rtol=1e-12, equal_nan=True,
                                       err_msg=f"Mismatch in {k}")

    def test_two_pop_chunked_matches_fused(self, matrix_with_pops):
        from pg_gpu.windowed_analysis import (
            windowed_statistics_fused,
            windowed_statistics_fused_chunked,
        )
        from pg_gpu import _memutil

        hm = matrix_with_pops
        hm.transfer_to_gpu()

        bp_bins = np.arange(0, 500_001, 50_000, dtype=np.float64)

        r1 = windowed_statistics_fused(
            hm, bp_bins=bp_bins,
            statistics=('fst', 'fst_wc', 'dxy', 'da'),
            pop1='pop1', pop2='pop2')

        orig = _memutil.estimate_fused_chunk_size
        _memutil.estimate_fused_chunk_size = lambda n, memory_fraction=0.35: 500
        try:
            r2 = windowed_statistics_fused_chunked(
                hm, bp_bins=bp_bins,
                statistics=('fst', 'fst_wc', 'dxy', 'da'),
                pop1='pop1', pop2='pop2')
        finally:
            _memutil.estimate_fused_chunk_size = orig

        for k in ('fst', 'fst_wc', 'dxy', 'da'):
            np.testing.assert_allclose(r1[k], r2[k], rtol=1e-12, equal_nan=True,
                                       err_msg=f"Mismatch in {k}")

    def test_mixed_single_twopop_chunked(self, matrix_with_pops):
        """Mixed single+two-pop stats should not crash (KeyError regression)."""
        from pg_gpu.windowed_analysis import windowed_statistics_fused_chunked
        from pg_gpu import _memutil

        hm = matrix_with_pops
        hm.transfer_to_gpu()

        bp_bins = np.arange(0, 500_001, 50_000, dtype=np.float64)

        orig = _memutil.estimate_fused_chunk_size
        _memutil.estimate_fused_chunk_size = lambda n, memory_fraction=0.35: 500
        try:
            r = windowed_statistics_fused_chunked(
                hm, bp_bins=bp_bins,
                statistics=('pi', 'theta_w', 'fst', 'fst_wc', 'dxy'),
                population='pop1', pop1='pop1', pop2='pop2')
        finally:
            _memutil.estimate_fused_chunk_size = orig

        for k in ('pi', 'theta_w', 'fst', 'fst_wc', 'dxy'):
            assert k in r, f"Missing {k}"
            assert len(r[k]) > 0


class TestFusedMissingData:
    """Fused two-pop kernel must use per-site valid counts under missingness."""

    @pytest.fixture
    def matrix_with_missing(self):
        """Two-pop matrix where ~50% of sites have missing data in one pop."""
        rng = np.random.RandomState(123)
        n_hap, n_var = 40, 2000
        hap = rng.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        # Inject missing data: ~50% of pop1 entries at odd-indexed variants
        for v in range(0, n_var, 2):
            for h in range(n_hap // 2):
                if rng.random() < 0.5:
                    hap[h, v] = -1
        pos = np.arange(1, n_var + 1) * 100
        hm = HaplotypeMatrix(hap, pos, 0, (n_var + 1) * 100)
        hm.sample_sets = {
            "pop1": list(range(n_hap // 2)),
            "pop2": list(range(n_hap // 2, n_hap)),
        }
        return hm

    def _compare_fused_vs_scatter(self, hm):
        """Compare fused kernel FST with scatter-add FST using aligned windows.

        Uses windowed_analysis() which routes to scatter-add for pure two-pop,
        vs calling windowed_statistics_fused() directly.
        Both paths use the same window boundaries via the convenience function.
        """
        from pg_gpu.windowed_analysis import windowed_statistics_fused

        hm.transfer_to_gpu()

        # Use windowed_analysis to get scatter-add results (pure two-pop request)
        r_scatter = windowed_analysis(
            hm, window_size=50_000,
            statistics=["fst", "dxy"],
            populations=["pop1", "pop2"],
            span_normalize=False)

        # Build the same windows the scatter path used
        ws = r_scatter['start'].values.astype(np.float64)
        we = r_scatter['stop'].values.astype(np.float64)
        bp_bins = np.concatenate([ws, [we[-1]]])

        # Fused kernel path with identical windows
        r_fused = windowed_statistics_fused(
            hm, bp_bins=bp_bins,
            statistics=('fst', 'dxy'),
            pop1='pop1', pop2='pop2',
            _win_starts=ws, _win_stops=we,
            per_base=False)

        both_valid = np.isfinite(r_fused['fst']) & np.isfinite(r_scatter['fst'].values)
        assert np.sum(both_valid) > 0, "No valid FST values to compare"
        np.testing.assert_allclose(
            r_fused['fst'][both_valid],
            r_scatter['fst'].values[both_valid],
            rtol=1e-10,
            err_msg="Fused kernel FST disagrees with scatter-add")

    def test_fused_twopop_missing_matches_scatter(self, matrix_with_missing):
        """Fused kernel FST must match scatter-add FST under missingness."""
        self._compare_fused_vs_scatter(matrix_with_missing)

    def test_fused_twopop_no_missing_matches_scatter(self, matrix_with_missing):
        """Sanity: without missing data, fused and scatter should agree."""
        rng = np.random.RandomState(456)
        n_hap, n_var = 40, 2000
        hap = rng.randint(0, 2, (n_hap, n_var), dtype=np.int8)
        pos = np.arange(1, n_var + 1) * 100
        hm = HaplotypeMatrix(hap, pos, 0, (n_var + 1) * 100)
        hm.sample_sets = {
            "pop1": list(range(n_hap // 2)),
            "pop2": list(range(n_hap // 2, n_hap)),
        }
        self._compare_fused_vs_scatter(hm)
