# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pg_gpu is a GPU-accelerated population genetics statistics library that integrates with the moments population genetics package. It uses CuPy for GPU computation.

## Rules
- Never include Claude as a co-author or contributor in any files, commits, documentation, or PRs.
- always follow best practices for Python coding, testing, and documentation.
- ensure all code changes are compatible with both CPU and GPU execution where applicable.
- maintain high test coverage and ensure all tests pass before merging any changes.
- keep commit messages terse
- keep PR descriptions concise and focused on the changes made.
- Never use emoticons in commit messages or PR descriptions or code or comments.
- always draft terse commit messages, save them to disk for me to review
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
pytest tests/

# Run tests with verbose output
pytest tests/ -v

# Run specific test categories
pytest tests/test_haplotype_counting.py      # Unit tests for core algorithms
pytest tests/test_ld_statistics_gpu.py       # GPU integration tests
pytest tests/test_ld_validation_synthetic.py  # Quick validation with synthetic data
pytest tests/test_ld_validation_full.py       # Full validation (slower, uses cached data)

# Run tests matching a pattern
pytest tests/ -k "test_ld"
pytest tests/ -k "missing_data"

# Run specific test class
pytest tests/test_ld_statistics_gpu.py::TestLDStatisticsGPU

# Run tests in parallel (faster)
pytest tests/ -n 10
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
   - Integration with moments SFS spectrum objects

2. **GPU Statistics Modules**:
   - `pg_gpu/ld_statistics.py`: Unified implementation for all LD statistics
     - Clean, consistent API with functions: `dd()`, `dz()`, `pi2()`
     - Handles both single and two-population statistics seamlessly
     - Automatically detects and handles missing data transparently
     - Supports flexible population configurations via `populations` parameter
   - `pg_gpu/diversity.py`: Comprehensive within-population diversity statistics
     - All functions fully vectorized for GPU acceleration
     - Three missing data strategies: 'ignore', 'include', 'exclude'
     - Functions: `pi()`, `theta_w()`, `tajimas_d()`, `segregating_sites()`, `singleton_count()`, `fay_wus_h()`
   - `pg_gpu/divergence.py`: Between-population divergence statistics
     - Efficient vectorized FST, Dxy, Da calculations
     - Multiple FST estimators: Hudson, Weir-Cockerham, Nei
     - Comprehensive missing data support across all functions

3. **Integration Layer** (`pg_gpu/integration.py`): Bridges pg_gpu with moments package, enabling GPU acceleration for existing moments workflows.

### Key Design Patterns

- **Automatic Missing Data Detection**: Functions check for -1 values and switch to missing data algorithms automatically
- **GPU Memory Management**: Uses CuPy for efficient GPU memory handling
- **Batch Processing**: Processes multiple statistics simultaneously for efficiency
- **Vectorized Algorithms**: All diversity and divergence functions use fully vectorized GPU operations (no Python loops over variants)

### Data Flow

1. Load haplotype data → HaplotypeMatrix
2. Convert to GPU if needed
3. Compute statistics using GPU kernels
4. Return results as CuPy/NumPy arrays

## Important Implementation Details

- Missing data is encoded as -1 in haplotype matrices
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
- Tests use moments package as reference implementation
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

The validation tests confirm excellent agreement between pg_gpu and moments:

1. **Two-Population LD Statistics**: ✅ All 15 statistics (DD, Dz, pi2) pass validation with <1% error
2. **Numerical Precision**: ✅ All statistics including Dz meet 1% tolerance threshold
3. **Missing Data Handling**: ✅ Automatic detection and proper handling implemented
4. **GPU Acceleration**: ✅ Significant performance improvements over CPU implementation
5. **Unified Statistics Design**: ✅ Single implementation handles missing data transparently (not a separate mode)
6. **Diversity Statistics**: ✅ All functions (pi, theta_w, tajimas_d, segregating_sites, singleton_count, fay_wus_h) fully vectorized
7. **Divergence Statistics**: ✅ All functions (FST variants, Dxy, Da, pi_within) efficiently vectorized
8. **Performance Optimization**: ✅ All Python loops eliminated from genetic statistics calculations

## Next Development Goals

1. **LD Statistics with Missing Data**: Optimize O(n_variants²) loops in haplotype_matrix.py (lines 746, 822) for LD calculations
2. **Performance Scaling**: Optimize for larger datasets (100k+ variants) 
3. **Memory Efficiency**: Implement chunked processing for datasets exceeding GPU memory
4. **Additional Statistics**: Extend to other population genetics statistics beyond diversity/divergence
5. **Integration**: Deeper integration with moments workflows

## Performance Summary

Current performance on GPU (after vectorization):
- **Diversity functions** (6 total): ~0.01s for 10k variants, 200 haplotypes
- **Divergence functions** (7 total): ~0.026s for 5k variants, 100 haplotypes  
- **Missing data strategies**: All three modes ('ignore', 'include', 'exclude') are fast
- **Remaining bottlenecks**: Only LD calculations with missing data (O(n_variants²) scaling)