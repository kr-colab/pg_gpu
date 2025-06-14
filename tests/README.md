# pg_gpu Test Suite

This directory contains the test suite for the pg_gpu package. All tests are designed to be run with pytest.

## Test Organization

### Core Functionality Tests

- **`test_haplotype_matrix.py`**: Tests for the HaplotypeMatrix class
  - Loading from VCF files
  - Loading from tree sequences
  - Device transfers (CPU/GPU)
  - Population genetics statistics (diversity, Tajima's D, etc.)
  - Subsetting and manipulation

- **`test_haplotype_counting.py`**: Tests for haplotype counting algorithms
  - Within-population counting
  - Between-population counting
  - Multiple variant pair handling

### LD Statistics Tests

- **`test_ld_statistics_comparison.py`**: Unit tests comparing individual statistic formulas
  - DD (D-squared) statistics
  - Dz statistics
  - π₂ (pi2) statistics
  - Tests both single and multi-population cases
  - Directly compares GPU implementations against moments formulas

- **`test_ld_statistics_gpu.py`**: Integration tests for GPU LD computation
  - Full pipeline tests using HaplotypeMatrix
  - Tests binning by physical distance
  - Compares aggregated statistics against moments
  - Tests both raw sums and averaged statistics

### Validation Tests

- **`test_ld_validation_synthetic.py`**: Fast validation using synthetic data
  - Uses msprime to generate test data
  - Tests all 15 two-population LD statistics
  - Provides quick validation without external data files
  - Good for continuous integration

- **`test_ld_validation_full.py`**: Full validation against IM model dataset
  - Requires `data/im-parsing-example.vcf` and `data/im_pop.txt`
  - Tests against the same dataset used in `validate_two_pop_ld.py`
  - Parametrized tests for each statistic
  - Tests overall correlation and mean relative error
  - Caches moments results for faster repeated runs

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_haplotype_matrix.py
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run specific test class or method:
```bash
pytest tests/test_ld_statistics_gpu.py::TestLDStatisticsGPU::test_gpu_ld_statistics_between_populations
```

Run validation tests only:
```bash
# Fast synthetic validation
pytest tests/test_ld_validation_synthetic.py -v

# Full validation (if IM data available)
pytest tests/test_ld_validation_full.py -v
```

## Test Data

Tests use temporary files created during test execution. The `conftest.py` file provides shared fixtures for:
- Simple VCF files
- Two-population VCF files with population assignments
- Sample haplotype count data

## Requirements

- pytest
- numpy
- cupy (with CUDA support)
- scikit-allel
- moments (included in repo)

## Notes

All tests are designed to work on both CPU and GPU. GPU tests will automatically transfer data as needed.