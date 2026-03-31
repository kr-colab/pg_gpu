"""
GPU-accelerated dimensionality reduction and distance computation.

Provides PCA, randomized PCA, PCoA, and pairwise genetic distance
functions using CuPy for GPU acceleration.
"""

import numpy as np
import cupy as cp
from typing import Union, Optional, Tuple
from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix


def _prepare_matrix(haplotype_matrix, scaler, population, missing_data='include'):
    """Center and scale haplotype matrix for PCA.

    Returns the prepared matrix X and its dimensions.
    """
    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes
    has_missing = cp.any(hap < 0)

    if missing_data == 'exclude' and has_missing:
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap = hap[:, complete]
        has_missing = False

    if missing_data in ('include', 'pairwise') and has_missing:
        # Impute missing values to per-site mean frequency
        valid_mask = hap >= 0
        hap_clean = cp.where(valid_mask, hap, 0).astype(cp.float64)
        n_valid = cp.sum(valid_mask, axis=0).astype(cp.float64)
        site_mean = cp.where(n_valid > 0, cp.sum(hap_clean, axis=0) / n_valid, 0.0)
        X = cp.where(valid_mask, hap_clean, site_mean[None, :])
    else:
        X = cp.where(hap < 0, 0, hap).astype(cp.float64) if has_missing else hap.astype(cp.float64)

    X = X.copy()

    # center
    mean = cp.mean(X, axis=0)
    X -= mean

    # scale
    if scaler == 'patterson':
        p = cp.mean(X + mean, axis=0)  # original mean freq
        scale = cp.sqrt(p * (1 - p))
        valid = scale > 0
        X[:, valid] /= scale[valid]
        X[:, ~valid] = 0
    elif scaler == 'standard':
        std = cp.std(X, axis=0)
        valid = std > 0
        X[:, valid] /= std[valid]
        X[:, ~valid] = 0

    return X


def pca(haplotype_matrix: HaplotypeMatrix,
        n_components: int = 10,
        scaler: Optional[str] = 'patterson',
        population: Optional[Union[str, list]] = None,
        missing_data: str = 'include'):
    """Principal Component Analysis on haplotype data.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data. Rows are haplotypes, columns are variants.
    n_components : int
        Number of principal components to compute.
    scaler : str or None
        Scaling method before PCA:
        'patterson' - scale by sqrt(p*(1-p)), Patterson et al. (2006)
        'standard' - standardize to unit variance per variant
        None - center only (subtract mean)
    population : str or list, optional
        Population subset to use.
    missing_data : str
        'include' or 'pairwise' - impute missing to per-site mean
        'exclude' - filter sites with any missing

    Returns
    -------
    coords : ndarray, float64, shape (n_samples, n_components)
        Sample coordinates in PC space.
    explained_variance_ratio : ndarray, float64, shape (n_components,)
        Proportion of variance explained by each component.
    """
    X = _prepare_matrix(haplotype_matrix, scaler, population, missing_data)
    n_samples, n_variants = X.shape
    n_components = min(n_components, min(n_samples, n_variants))

    U, S, Vt = cp.linalg.svd(X, full_matrices=False)

    coords = U[:, :n_components] * S[:n_components]

    var = (S ** 2) / n_samples
    explained_variance_ratio = var[:n_components] / cp.sum(var)

    return coords.get(), explained_variance_ratio.get()


def randomized_pca(haplotype_matrix: HaplotypeMatrix,
                   n_components: int = 10,
                   scaler: Optional[str] = 'patterson',
                   population: Optional[Union[str, list]] = None,
                   n_iter: int = 3,
                   random_state: Optional[int] = None,
                   missing_data: str = 'include'):
    """Randomized PCA using truncated SVD approximation.

    Faster than full PCA for large datasets where only a few
    components are needed.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    n_components : int
        Number of components.
    scaler : str or None
        'patterson', 'standard', or None.
    population : str or list, optional
        Population subset.
    n_iter : int
        Number of power iterations for accuracy.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    coords : ndarray, float64, shape (n_samples, n_components)
    explained_variance_ratio : ndarray, float64, shape (n_components,)
    """
    X = _prepare_matrix(haplotype_matrix, scaler, population, missing_data)
    n_samples, n_variants = X.shape
    n_components = min(n_components, min(n_samples, n_variants))

    # randomized SVD on GPU
    if random_state is not None:
        cp.random.seed(random_state)

    # random projection
    k = n_components + 10  # oversampling
    k = min(k, min(n_samples, n_variants))
    Omega = cp.random.randn(n_variants, k, dtype=cp.float64)
    Y = X @ Omega

    # power iteration for accuracy
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)

    # QR decomposition
    Q, _ = cp.linalg.qr(Y)

    # project and SVD in reduced space
    B = Q.T @ X
    Uhat, S, Vt = cp.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    coords = U[:, :n_components] * S[:n_components]

    # explained variance
    total_var = cp.sum(cp.var(X, axis=0))
    var = (S[:n_components] ** 2) / n_samples
    explained_variance_ratio = var / total_var

    return coords.get(), explained_variance_ratio.get()


def pairwise_distance(haplotype_matrix: HaplotypeMatrix,
                      metric: str = 'euclidean',
                      population: Optional[Union[str, list]] = None,
                      missing_data: str = 'include'):
    """Compute pairwise distances between samples.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
        Haplotype data.
    metric : str
        Distance metric. Supported on GPU: 'euclidean', 'cityblock',
        'sqeuclidean'. Falls back to scipy for other metrics.
    population : str or list, optional
        Population subset.
    missing_data : str
        'include' or 'pairwise' - mask missing, normalize per pair
        'exclude' - filter sites with any missing

    Returns
    -------
    dist : ndarray, float64, shape (n_samples * (n_samples - 1) // 2,)
        Condensed distance matrix.
    """
    if missing_data == 'pairwise':
        missing_data = 'include'

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes

    if missing_data == 'exclude':
        missing_per_var = cp.sum(hap < 0, axis=0)
        complete = missing_per_var == 0
        hap = hap[:, complete]

    X = cp.where(hap < 0, 0, hap).astype(cp.float64)
    n = X.shape[0]

    if metric in ('euclidean', 'sqeuclidean', 'cityblock'):
        idx_i, idx_j = cp.triu_indices(n, k=1)
        n_pairs = len(idx_i)

        # batch to avoid materializing huge (n_pairs, n_variants) arrays
        batch_size = max(1, min(n_pairs, 50000))
        dist_parts = []

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            bi = idx_i[start:end]
            bj = idx_j[start:end]

            if metric == 'cityblock':
                d = cp.sum(cp.abs(X[bi] - X[bj]), axis=1)
            else:
                d = cp.sum((X[bi] - X[bj]) ** 2, axis=1)
                if metric == 'euclidean':
                    d = cp.sqrt(d)
            dist_parts.append(d)

        return cp.concatenate(dist_parts).get()
    else:
        # fall back to scipy for exotic metrics
        from scipy.spatial.distance import pdist
        X_cpu = X.get()
        return pdist(X_cpu, metric=metric)


def pcoa(dist, n_components: Optional[int] = None):
    """Principal Coordinate Analysis (classical MDS).

    Parameters
    ----------
    dist : array_like
        Pairwise distances in condensed form (from pairwise_distance)
        or as a square distance matrix.
    n_components : int, optional
        Number of dimensions to return. If None, returns all
        dimensions with positive eigenvalues.

    Returns
    -------
    coords : ndarray, float64, shape (n_samples, n_components)
        Sample coordinates.
    explained_variance_ratio : ndarray, float64, shape (n_components,)
        Proportion of variance explained by each axis.
    """
    from scipy.spatial.distance import squareform

    dist = np.asarray(dist, dtype='f8')

    # convert condensed to square if needed
    if dist.ndim == 1:
        D = squareform(dist)
    else:
        D = dist

    n = D.shape[0]

    # double-centering
    D_sq = D ** 2
    row_mean = D_sq.mean(axis=1, keepdims=True)
    col_mean = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    F = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    # eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(F)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # keep positive eigenvalues
    pos = eigvals > 0
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    # project
    coords = eigvecs * np.sqrt(eigvals)

    # explained variance
    explained_variance_ratio = eigvals / np.sum(eigvals)

    return coords, explained_variance_ratio
