"""Windowed PCA: per-window PCA coordinates across a chromosome.

A winpca-style API (https://github.com/MoritzBlumer/winpca) layered on
top of :func:`pg_gpu.local_pca` (lostruct, Li & Ralph 2019). Both functions
compute per-window PCA; they differ in defaults and in how the output is
typically plotted:

* :func:`local_pca` is tuned for the lostruct workflow — find genomic regions
  where population structure *changes*, usually via MDS on inter-window
  PC distances (``pc_dist`` / ``lostruct``).
* :func:`windowed_pca` (this module) is tuned for the winpca workflow —
  plot per-window PC1/PC2 *trajectories* per sample across the chromosome,
  with biallelic + MAF filtering and the Patterson scaler that
  ``scikit-allel`` uses by default.

The two share an SVD-per-window kernel. The shapes of the outputs are
related by a transpose:

* :class:`local_pca`'s :class:`LocalPCAResult.eigvecs` has shape
  ``(n_windows, k, n_samples)``.
* :func:`windowed_pca` returns :class:`WindowedPCAResult.coords` with shape
  ``(n_windows, n_samples, n_components)`` — the conventional "rows are
  samples, columns are PCs" orientation winpca plots from.
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from pg_gpu.haplotype_matrix import HaplotypeMatrix
from pg_gpu.decomposition import local_pca


@dataclasses.dataclass
class WindowedPCAResult:
    """Output of :func:`windowed_pca`.

    Attributes
    ----------
    windows : pandas.DataFrame
        One row per window. Columns:

        * ``chrom``, ``start``, ``end``, ``center`` -- window genomic coords
        * ``n_variants`` -- variants used in the window
        * ``ev_1`` .. ``ev_{n_components}`` -- per-PC eigenvalues
    coords : numpy.ndarray
        Per-sample PCA coordinates, shape
        ``(n_windows, n_samples, n_components)``. NaN for windows that had
        fewer variants than ``n_components``.
    sample_ids : list of str
        Length ``n_samples``. Labels for ``coords`` axis 1.
    component_labels : list of str
        Length ``n_components``. Labels for ``coords`` axis 2.
    """

    windows: pd.DataFrame
    coords: np.ndarray
    sample_ids: list
    component_labels: list


def _maf_keep_indices(haps, maf_threshold: float) -> np.ndarray:
    """Numpy indices of sites where minor allele frequency exceeds threshold.

    Handles both numpy and cupy arrays. ``haps`` shape is
    ``(n_haplotypes, n_variants)`` with ``-1`` marking missing.
    """
    if type(haps).__module__.startswith("cupy"):
        import cupy as xp
    else:
        xp = np
    valid    = haps >= 0
    n_valid  = valid.sum(axis=0)
    alt      = (haps == 1).sum(axis=0)
    af       = xp.where(n_valid > 0, alt / xp.maximum(n_valid, 1), 0.0)
    maf      = xp.minimum(af, 1.0 - af)
    keep     = xp.where(maf > maf_threshold)[0]
    return keep.get() if hasattr(keep, "get") else np.asarray(keep)


def windowed_pca(
    haplotype_matrix: HaplotypeMatrix,
    window_size: int,
    step_size: Optional[int] = None,
    *,
    n_components: int = 10,
    window_type: str = "bp",
    maf_threshold: float = 0.05,
    ld_prune: bool = True,
    ld_size: int = 100,
    ld_step: int = 20,
    ld_threshold: float = 0.1,
    biallelic_only: bool = True,
    scaler: str = "patterson",
    population: Optional[str] = None,
    random_state: Optional[int] = None,
) -> WindowedPCAResult:
    """Per-window PCA across the genome with winpca-style defaults.

    Thin wrapper over :func:`pg_gpu.local_pca` that:

    1. Optionally pre-filters the matrix to biallelic / MAF > threshold /
       LD-pruned sites (applied **once globally** — see Notes below).
    2. Runs ``local_pca`` with ``scaler='patterson'`` and ``k=n_components``.
    3. Repackages the result with shape ``(n_windows, n_samples,
       n_components)`` (transpose of ``LocalPCAResult.eigvecs``) so it's
       ready for the winpca-style plotting idiom — one PC1 line per
       sample, x-axis = window center.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Eager (non-streaming) matrix. ``local_pca`` requires the matrix in
        memory.
    window_size : int
        Window size in bp (when ``window_type='bp'``) or number of SNPs
        (when ``window_type='snp'``).
    step_size : int, optional
        Sliding step. Defaults to ``window_size`` (non-overlapping windows).
    n_components : int, default 10
        Number of PCs to retain per window.
    window_type : {'bp', 'snp'}, default 'bp'
        Forwarded to ``local_pca``.
    maf_threshold : float, default 0.05
        Drop sites with MAF below this. Set to 0 to skip.
    ld_prune : bool, default True
        Apply ``locate_unlinked`` LD pruning before the per-window PCA.
    ld_size, ld_step, ld_threshold : int, int, float
        Args to ``locate_unlinked``.
    biallelic_only : bool, default True
        Apply ``apply_biallelic_filter()`` first.
    scaler : str, default 'patterson'
        Forwarded to ``local_pca`` (matches scikit-allel / winpca default).
    population : str, optional
        Restrict to a single population from ``hm.sample_sets``.
    random_state : int, optional
        Reproducible randomized-SVD seed.

    Returns
    -------
    WindowedPCAResult

    Notes
    -----
    Per-window MAF and LD-prune adaptivity (winpca's stricter mode) is NOT
    supported here — the filters apply once to the whole matrix before
    windowing. The alternative — iterating windows and re-filtering each —
    duplicates ``local_pca``'s iteration logic; if you need it, file an
    issue or compose the loop yourself with
    :func:`pg_gpu.decomposition.randomized_pca`.

    Examples
    --------
    >>> from pg_gpu import HaplotypeMatrix, windowed_pca
    >>> hm = HaplotypeMatrix.from_zarr(VCZ, region='X:1-10000000',
    ...                                streaming='never')
    >>> result = windowed_pca(hm, window_size=200_000)
    >>> result.coords.shape          # (n_windows, n_samples, n_components)
    >>> result.windows.head()        # window metadata
    """
    if type(haplotype_matrix).__name__ == "StreamingHaplotypeMatrix":
        raise ValueError(
            "windowed_pca requires an eager HaplotypeMatrix; "
            "streaming matrices are not supported. Reload the region with "
            "streaming='never' (and a smaller region if it won't fit on GPU)."
        )

    if step_size is None:
        step_size = window_size

    hm = haplotype_matrix
    if biallelic_only:
        hm = hm.apply_biallelic_filter()
    if maf_threshold > 0 and hm.num_variants > 0:
        keep = _maf_keep_indices(hm.haplotypes, maf_threshold)
        if len(keep) == 0:
            warnings.warn(
                "No sites passed the MAF > %g filter; returning empty result."
                % maf_threshold
            )
            return _empty_result(haplotype_matrix, n_components)
        hm = hm.get_subset(keep)
    if ld_prune and hm.num_variants > 0:
        unlinked = hm.locate_unlinked(
            size=ld_size, step=ld_step, threshold=ld_threshold,
        )
        if hasattr(unlinked, "get"):
            unlinked = unlinked.get()
        keep = np.where(unlinked)[0]
        if len(keep) == 0:
            warnings.warn(
                "No sites surviving LD prune; returning empty result."
            )
            return _empty_result(haplotype_matrix, n_components)
        hm = hm.get_subset(keep)

    lpca = local_pca(
        hm,
        window_size=window_size,
        step_size=step_size,
        window_type=window_type,
        k=n_components,
        scaler=scaler,
        population=population,
        random_state=random_state,
    )

    eigvecs = lpca.eigvecs
    n_rows = eigvecs.shape[-1]
    n_samples = haplotype_matrix.num_haplotypes // 2
    # Fold haplotype-pair rows to per-sample under pg_gpu's
    # [hap0_of_all, hap1_of_all] layout (sample i at rows i, i+n_samples).
    if isinstance(haplotype_matrix, HaplotypeMatrix) and n_rows == 2 * n_samples:
        eigvecs = (eigvecs[..., :n_samples] + eigvecs[..., n_samples:]) / 2.0

    coords = np.transpose(eigvecs, (0, 2, 1))

    windows = lpca.windows.copy()
    eigvals = lpca.eigvals
    for i in range(n_components):
        windows[f"ev_{i + 1}"] = eigvals[:, i] if i < eigvals.shape[1] else np.nan

    sample_ids = list(getattr(haplotype_matrix, "samples", []))
    if not sample_ids:
        n_samples = coords.shape[1] if coords.size else 0
        sample_ids = [f"sample_{i}" for i in range(n_samples)]

    return WindowedPCAResult(
        windows=windows,
        coords=coords,
        sample_ids=sample_ids,
        component_labels=[f"PC{i + 1}" for i in range(n_components)],
    )


def _empty_result(hm, n_components):
    n_samples = hm.num_haplotypes // 2
    return WindowedPCAResult(
        windows=pd.DataFrame(),
        coords=np.empty((0, n_samples, n_components), dtype=np.float64),
        sample_ids=list(getattr(hm, "samples", [])) or [f"sample_{i}" for i in range(n_samples)],
        component_labels=[f"PC{i + 1}" for i in range(n_components)],
    )
