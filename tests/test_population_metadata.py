"""Population subsets keep sample names and QC fields.

A population subset takes rows, so ``samples`` and per-genotype fields keep
the population's individuals, per-variant fields keep every variant, and
both follow the accessible mask that the subset's variants already follow.
"""
import warnings

import numpy as np
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix
from pg_gpu._utils import get_population_matrix, population_rows
from pg_gpu._warnings import UnpairedRowsWarning, paired_rows_problem

N_VAR = 6
SAMPLES = ['a', 'b', 'c', 'd']


def _fields(n_ind=4):
    return {
        'MQ': np.arange(N_VAR, dtype=np.float32),
        'DP': np.arange(N_VAR * n_ind, dtype=np.int16).reshape(N_VAR, n_ind),
    }


def _hm(**kw):
    hap = np.zeros((8, N_VAR), dtype=np.int8)   # 4 individuals, 8 haplotypes
    return HaplotypeMatrix(hap, np.arange(N_VAR) * 10 + 1, 1, N_VAR * 10,
                           samples=SAMPLES, fields=_fields(), **kw)


def _gm(**kw):
    geno = np.zeros((4, N_VAR), dtype=np.int8)
    return GenotypeMatrix(geno, np.arange(N_VAR) * 10 + 1, 1, N_VAR * 10,
                          samples=SAMPLES, fields=_fields(), **kw)


def test_haplotype_subset_keeps_samples_and_fields():
    # The example from the issue: rows 0-3 are individuals a and b.
    sub = get_population_matrix(_hm(sample_sets={'pop1': [0, 1, 2, 3]}), 'pop1')
    assert sub.samples == ['a', 'b']
    np.testing.assert_array_equal(sub.fields['MQ'], _fields()['MQ'])
    np.testing.assert_array_equal(sub.fields['DP'], _fields()['DP'][:, [0, 1]])


def test_haplotype_subset_follows_row_pair_order():
    # Pairs may come in any order; each pair names one individual.
    sub = get_population_matrix(_hm(), [7, 6, 2, 3])
    assert sub.samples == ['d', 'b']
    np.testing.assert_array_equal(sub.fields['DP'], _fields()['DP'][:, [3, 1]])


def test_genotype_subset_keeps_samples_and_fields():
    sub = get_population_matrix(_gm(), [3, 1])
    assert sub.samples == ['d', 'b']
    np.testing.assert_array_equal(sub.fields['MQ'], _fields()['MQ'])
    np.testing.assert_array_equal(sub.fields['DP'], _fields()['DP'][:, [3, 1]])


@pytest.mark.parametrize("build", [_hm, _gm], ids=["haplotype", "genotype"])
def test_fields_follow_the_accessible_mask(build):
    # Only positions 1, 21 and 41 are accessible (array index 0 is position
    # 1); the subset holds those three variants, and each field must line
    # up with them.
    acc = np.zeros(N_VAR * 10, dtype=bool)
    acc[[0, 20, 40]] = True
    m = build()
    m.set_accessible_mask(acc)
    rows = [0, 1] if build is _hm else [0]
    sub = get_population_matrix(m, rows)
    assert sub.num_variants == 3
    np.testing.assert_array_equal(sub.fields['MQ'], _fields()['MQ'][[0, 2, 4]])
    np.testing.assert_array_equal(sub.fields['DP'],
                                  _fields()['DP'][[0, 2, 4]][:, [0]])


@pytest.mark.parametrize("rows", [[0, 2], [0, 1, 2]], ids=["split-pair", "odd"])
def test_unpaired_rows_drop_individual_metadata_with_warning(rows):
    with pytest.warns(UnpairedRowsWarning, match="per-genotype fields"):
        sub = get_population_matrix(_hm(), rows)
    assert sub.samples is None
    assert 'DP' not in sub.fields
    np.testing.assert_array_equal(sub.fields['MQ'], _fields()['MQ'])


def test_unpaired_rows_without_metadata_are_silent():
    hap = np.zeros((8, N_VAR), dtype=np.int8)
    m = HaplotypeMatrix(hap, np.arange(N_VAR) * 10 + 1, 1, N_VAR * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_population_matrix(m, [0, 2])


def test_statistics_form_carries_no_metadata():
    sub = population_rows(_hm(), [0, 2])      # unpaired, but no warning
    assert sub.samples is None and sub.fields == {}


def test_subset_round_trips_through_zarr(tmp_path):
    sub = get_population_matrix(_hm(sample_sets={'pop1': [4, 5, 6, 7]}), 'pop1')
    path = str(tmp_path / "sub.vcz")
    sub.to_zarr(path, format="vcz", contig_name="1")
    back = HaplotypeMatrix.from_zarr(path, streaming="never")
    assert list(back.samples) == ['c', 'd']


def test_names_not_one_per_individual_are_dropped_with_warning():
    # One name per haplotype, not per individual: picking by individual
    # would label rows 4 and 5 with 'h2', so the subset drops the names.
    hap = np.zeros((8, N_VAR), dtype=np.int8)
    m = HaplotypeMatrix(hap, np.arange(N_VAR) * 10 + 1, 1, N_VAR * 10,
                        samples=[f"h{i}" for i in range(8)],
                        fields={'DP': np.zeros((N_VAR, 8), dtype=np.int16),
                                'MQ': _fields()['MQ']})
    with pytest.warns(UserWarning, match="samples"):
        sub = get_population_matrix(m, [4, 5])
    assert sub.samples is None
    assert 'DP' not in sub.fields and 'MQ' in sub.fields


@pytest.mark.parametrize("rows, expected", [
    ([0, 1, 3, 2], None),
    ([1, 2], "individuals 0 and 1"),
    ([0, 1, 2], "3 rows"),
    ([[0], [1, 2]], None),       # malformed: left to the validating checks
    (["a", "b"], None),
])
def test_paired_rows_problem(rows, expected):
    problem = paired_rows_problem(rows)
    if expected is None:
        assert problem is None
    else:
        assert expected in problem
