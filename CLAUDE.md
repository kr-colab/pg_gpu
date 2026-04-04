# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pg_gpu is a GPU-accelerated population genetics statistics library. It uses CuPy for GPU computation.

## Rules
- Never include Claude as a co-author or contributor in any files, commits, documentation, or PRs.
- always follow best practices for Python coding, testing, and documentation.
- ensure all code changes are compatible with both CPU and GPU execution where applicable.
- maintain high test coverage and ensure all tests pass before merging any changes.
- keep commit messages terse
- keep PR descriptions concise and focused on the changes made.
- Never use emoticons in commit messages or PR descriptions or code or comments.
- always draft terse commit messages
- use `pixi run python` or the pixi shell for all Python execution
- put all debug scripts in `debug/` -- use this location for scripts that will not eventually be part of the test suite in `tests/`

## Commands

### Setup and Environment
```bash
# Install pixi if not already available
# https://pixi.sh

# Install and activate the environment
pixi install
pixi shell
```

### Testing
```bash
# Run all tests
pixi run pytest tests/

# Run tests with verbose output
pixi run pytest tests/ -v

# Run tests matching a pattern
pixi run pytest tests/ -k "test_ld"

# Run tests in parallel (faster)
pixi run pytest tests/ -n 10

# Full cross-validation against scikit-allel (29 statistics)
pixi run python tests/validate_against_allel.py

# Lint
pixi run -e lint ruff check pg_gpu/
```

### Development
```bash
# Dev dependencies are included in the default environment
# Profile GPU performance
pixi run python examples/performance_comparison.py
```

## Architecture

### Core Components

1. **HaplotypeMatrix** (`pg_gpu/haplotype_matrix.py`): Central data structure that manages haplotype data on CPU or GPU. Supports:
   - Conversion between CPU/GPU
   - Missing data handling (-1 encoded)
   - Accessible site masks from BED files (non-destructive property-based filtering)
2. **GPU Statistics Modules**:
   - `pg_gpu/diversity.py`: Within-population diversity (pi, theta_w, theta_h, theta_l, tajimas_d, fay_wus_h, etc.)
   - `pg_gpu/divergence.py`: Between-population divergence (fst_hudson, fst_weir_cockerham, fst_nei, dxy, da, pbs)
   - `pg_gpu/selection.py`: Selection scans (ihs, nsl, xpehh, xpnsl, garud_h, ehh_decay)
   - `pg_gpu/sfs.py`: Site frequency spectra (sfs, joint_sfs, folded variants, scaling)
   - `pg_gpu/ld_statistics.py`: Two-locus LD statistics (dd, dz, pi2)
   - `pg_gpu/admixture.py`: Patterson's f-statistics (f2, f3, D)
   - `pg_gpu/decomposition.py`: PCA, randomized PCA, PCoA, pairwise distance
   - `pg_gpu/relatedness.py`: GRM, IBS
   - `pg_gpu/windowed_analysis.py`: Fused CUDA kernels for windowed statistics
   - `pg_gpu/moments_ld.py`: Drop-in replacement for moments.LD.Parsing (requires moments env)
3. **Accessible Site Masks** (`pg_gpu/accessible.py`): BED file parsing and `AccessibleMask` class for genome accessibility. Dense boolean arrays with lazy prefix-sum for O(1) windowed range queries. Integrated into HaplotypeMatrix, GenotypeMatrix, and windowed analysis.
4. **Memory Management** (`pg_gpu/_memutil.py`): Adaptive chunked GPU processing. All modules use `chunked_dac_and_n` for memory-safe allele counting. Large operations auto-detect available GPU memory and chunk accordingly.

### Key Design Patterns

- **Missing Data Handling**: All functions default to 'include' mode (per-site valid data). 'pairwise' mode uses pixy-style comparison counting. 'exclude' drops sites with any missing. Missing values (-1) are never treated as reference alleles.
- **Accessible Site Masks**: BED-based accessibility masks on HaplotypeMatrix/GenotypeMatrix. `set_accessible_mask()` is non-destructive: original data is preserved in `_haplotypes`/`_positions`, and the `haplotypes`/`positions` properties transparently return filtered views. Masks can be replaced or removed at any time. Used for per-base normalization (`span_denominator='accessible'`), gap scaling in selection scans (iHS, XPEHH), and per-window `n_total_sites` in pairwise mode. Mask stays on CPU; uses lazy prefix-sum for O(1) range queries.
- **GPU Memory Management**: Adaptive chunking via `_memutil.py` prevents OOM on large datasets. Fast path when data fits, chunked fallback when it doesn't.
- **All public functions return NumPy arrays** (not CuPy). GPU stays internal.
- **Vectorized Algorithms**: All diversity and divergence functions use fully vectorized GPU operations (no Python loops over variants)

### Data Flow

1. Load haplotype data → HaplotypeMatrix (from VCF, tree sequence, or Zarr)
2. Optionally attach accessible mask (`hm.set_accessible_mask("mask.bed", chrom="chr1")`)
3. Transfer to GPU (`hm.transfer_to_gpu()`)
4. Compute statistics using GPU kernels (properties return filtered data transparently)
5. Return results as NumPy arrays

## Important Implementation Details

- Missing data is encoded as -1 in haplotype matrices
- All functions default to `missing_data='include'` (per-site valid data)
- Three modes: 'include' (skip missing per-site), 'exclude' (drop sites), 'pairwise' (comparison counting)
- The 'pairwise' mode implements pixy-style sum(diffs)/sum(comps) normalization
- `n_total_sites` on HaplotypeMatrix/GenotypeMatrix enables invariant site correction in pairwise mode
- `accessible_mask` is a property on HaplotypeMatrix/GenotypeMatrix; assigning it (directly or via `set_accessible_mask()`) automatically computes filtered variant indices
- `haplotypes` and `positions` are properties that return filtered views when a mask is set, or raw data when no mask is present
- `set_accessible_mask()` is non-destructive and returns self for chaining; `remove_accessible_mask()` restores all original variants
- `set_accessible_mask()` always updates `n_total_sites` to the accessible count; `remove_accessible_mask()` clears it
- `get_span('accessible')` returns accessible base count; falls back to 'total' when no mask is set
- `get_subset()` and `get_population_matrix()` read from the filtered properties, so children automatically contain only accessible variants
- For haplotype identity (Garud's H, haplotype_diversity), missing is treated as wildcard
- GPU functions use shared memory optimization for performance
- Statistics are computed pairwise across all SNPs by default
- Two-population statistics require population masks to identify samples

## Testing Structure

### Test Organization
- **Unit Tests**: `test_haplotype_counting.py`, `test_ld_statistics_comparison.py` - test individual algorithms
- **Integration Tests**: `test_ld_statistics_gpu.py`, `test_haplotype_matrix.py` - test full pipelines
- **Validation Tests**: `test_ld_validation_synthetic.py` (quick), `test_ld_validation_full.py` (comprehensive)
- **Missing Data Tests**: `test_ld_missing_data.py`, `test_ld_missing_data_detailed.py`

### Key Testing Patterns
- Tests validated against scikit-allel reference implementations
- Synthetic data generated with msprime for realistic genetic patterns
- Relative error tolerance of 1% (1e-2) for all validation tests
- Cached reference calculations in `tests/cache/` for performance
- Fixtures in `conftest.py` for common test data generation
- Supports parallel test execution with pytest-xdist

### Generating Test Data
To run full validation tests, generate the IM model test data:
```bash
python data/simulate_im_vcf.py
```

## Development Status

29 statistics validated against scikit-allel at machine precision using real Ag1000G data.
Run `pixi run python tests/validate_against_allel.py` for the full cross-validation.

## Performance Summary (vs scikit-allel)

Scalar statistics (1M variants, 200 haplotypes):
- **Weir-Cockerham FST**: 475x faster
- **Patterson F2/F3/D**: 7-18x faster
- **nSL**: 15x faster
- **iHS**: 6.5x faster (fused kernel, O(n) memory)
- **Hudson FST, Dxy**: 4-7x faster

Windowed statistics (fused CUDA kernels, single kernel launch):
- **pi + theta_w + tajimas_d**: 40x faster (50kb windows)
- **All 7 stats in one call**: 33ms total
- **Weir-Cockerham windowed FST**: 300x faster

All diploSHIC feature vector statistics supported via `windowed_analysis()`:
- 43 feature columns including ZnS, Omega, Garud's H, DAF histogram
- ZnS + Omega: 38x faster than diploSHIC's numba+BLAS path