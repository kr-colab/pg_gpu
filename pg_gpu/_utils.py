"""
Shared utilities for pg_gpu modules.
"""

from typing import Union
from .haplotype_matrix import HaplotypeMatrix


def get_population_matrix(matrix, population: Union[str, list]):
    """Extract a population-specific subset matrix.

    Parameters
    ----------
    matrix : HaplotypeMatrix or GenotypeMatrix
        The full data. Rows are haplotypes (HaplotypeMatrix) or individuals
        (GenotypeMatrix); the subset is taken along that row axis.
    population : str or list
        Population name (looked up in sample_sets) or list of row indices.

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
        from ._warnings import check_sample_set_rows
        check_sample_set_rows("population row list", pop_indices,
                              matrix.shape[0])

    subset_sets = {'all': list(range(len(pop_indices)))}
    if isinstance(matrix, GenotypeMatrix):
        return GenotypeMatrix(
            matrix.genotypes[pop_indices, :],
            matrix.positions,
            matrix.chrom_start,
            matrix.chrom_end,
            sample_sets=subset_sets,
            n_total_sites=matrix.n_total_sites,
        )
    return HaplotypeMatrix(
        matrix.haplotypes[pop_indices, :],
        matrix.positions,
        matrix.chrom_start,
        matrix.chrom_end,
        sample_sets=subset_sets,
        n_total_sites=matrix.n_total_sites,
    )
