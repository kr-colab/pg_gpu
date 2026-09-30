"""
Shared utilities for pg_gpu modules.
"""

import warnings
from typing import Union

import cupy as cp
import numpy as np

from .accessible import slice_fields
from .haplotype_matrix import HaplotypeMatrix
from ._warnings import (UnpairedRowsWarning, _rows_as_numpy,
                        check_sample_set_rows, paired_rows_problem)


def get_population_matrix(matrix, population: Union[str, list],
                          metadata: bool = True):
    """Extract a population-specific subset matrix.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
        The full data. Rows are haplotypes (HaplotypeMatrix) or individuals
        (GenotypeMatrix); the subset is taken along that row axis.
    population : str or list
        Population name (looked up in sample_sets) or list of row indices.
    metadata : bool
        Carry ``samples`` and QC ``fields`` over to the subset (default);
        see ``_population_metadata``. ``False`` skips the copies.

    Returns
    -------
    HaplotypeMatrix or GenotypeMatrix
        Subset matrix of the same type for the specified population.
    """
    from .genotype_matrix import GenotypeMatrix

    if isinstance(population, str):
        if matrix.sample_sets is None:
            raise ValueError("No sample_sets defined in matrix")
        if population not in matrix.sample_sets:
            raise ValueError(
                f"Population {population} not found in sample_sets")
        pop_indices = matrix.sample_sets[population]
    else:
        pop_indices = list(population)
        # Direct row-list arguments never pass the sample_sets setter, so
        # the same range and duplicate rules apply here before the rows
        # reach CuPy's unchecked fancy indexing.
        check_sample_set_rows("population row list", pop_indices,
                              matrix.shape[0])

    is_genotype = isinstance(matrix, GenotypeMatrix)
    extra = {}
    if metadata:
        rows = _rows_as_numpy(pop_indices)
        if is_genotype:
            individuals = rows
        else:
            # Sample i owns rows 2i and 2i + 1; only a list that pairs into
            # whole individuals names a set of samples.
            problem = paired_rows_problem(rows)
            individuals = None if problem else rows[0::2] // 2
            if problem and (matrix.samples is not None
                            or any(a.ndim == 2 for a in matrix.fields.values())):
                warnings.warn(
                    f"get_population_matrix: the population row list "
                    f"{problem}, so the subset has no sample names or "
                    f"per-genotype fields.", UnpairedRowsWarning, stacklevel=2)
        extra = _population_metadata(matrix, individuals)
    cls = GenotypeMatrix if is_genotype else HaplotypeMatrix
    data = matrix.genotypes if is_genotype else matrix.haplotypes
    return cls(
        data[pop_indices, :],
        matrix.positions,
        matrix.chrom_start,
        matrix.chrom_end,
        sample_sets={'all': list(range(len(pop_indices)))},
        n_total_sites=matrix.n_total_sites,
        **extra,
    )


def population_rows(matrix, population: Union[str, list]):
    """``get_population_matrix`` without sample names or QC fields.

    The statistics read only the genotype data, so they take this form and
    skip copying per-genotype fields, which can be as large as the matrix.
    """
    return get_population_matrix(matrix, population, metadata=False)


def _population_metadata(matrix, individuals):
    """``samples`` and ``fields`` of a population subset.

    Per-variant fields keep every variant of the subset; per-genotype fields
    (2-D, ``(n_var, n_samples)``) and ``samples`` keep ``individuals``, or
    are dropped when it is None. Fields sit on the unfiltered variant axis
    while the subset holds only the accessible variants, so they are cut to
    those too.
    """
    acc = matrix._accessible_idx
    acc = None if acc is None else cp.asnumpy(acc)
    per_variant = {t: a for t, a in matrix.fields.items() if a.ndim != 2}
    fields = per_variant if acc is None else slice_fields(per_variant, acc)
    samples = None
    if individuals is not None:
        for tag, arr in matrix.fields.items():
            if arr.ndim == 2:
                # One fancy index on both axes, so a masked matrix makes no
                # full-width intermediate copy.
                var_idx = np.arange(arr.shape[0]) if acc is None else acc
                fields[tag] = arr[np.ix_(var_idx, individuals)]
        if matrix.samples is not None:
            samples = [matrix.samples[i] for i in individuals]
    return {'samples': samples, 'fields': fields}
