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

    Uses chunked processing for large matrices to avoid OOM.
    Returns the prepared matrix X on GPU.
    """
    from ._memutil import chunked_dac_and_n, estimate_variant_chunk_size

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes
    n_samples, n_var = hap.shape

    if missing_data == 'exclude':
        dac, nv = chunked_dac_and_n(hap)
        complete = nv == n_samples
        if not cp.all(complete):
            hap = hap[:, complete]
            n_var = hap.shape[1]

    # Compute per-site mean and scale from allele counts (memory-safe)
    dac, nv = chunked_dac_and_n(hap)
    site_mean = cp.where(nv > 0, dac.astype(cp.float64) / nv.astype(cp.float64), 0.0)

    if scaler == 'patterson':
        p = site_mean
        scale = cp.sqrt(p * (1 - p))
        scale = cp.where(scale > 0, scale, 1.0)
    elif scaler == 'standard':
        # Need variance -- compute via second pass or approximate
        scale = None  # handled below
    else:
        scale = None

    # Check if full float64 matrix fits in memory
    free = cp.cuda.Device().mem_info[0]
    needed = n_samples * n_var * 8  # float64
    if needed < free * 0.4:
        # Fast path: materialize full matrix
        has_missing = bool(cp.any(hap < 0).get())
        if has_missing:
            valid_mask = hap >= 0
            X = cp.where(valid_mask, hap, 0).astype(cp.float64)
            X = cp.where(valid_mask, X, site_mean[None, :])
        else:
            X = hap.astype(cp.float64)
        X -= site_mean
        if scaler == 'patterson':
            X /= scale
        elif scaler == 'standard':
            std = cp.std(X, axis=0)
            valid = std > 0
            X[:, valid] /= std[valid]
            X[:, ~valid] = 0
        return X

    # Memory-constrained path: return hap + metadata for chunked PCA
    # Store scaling info so pca() can chunk the matmul
    return _DeferredPCA(hap, site_mean, scale, scaler)


class _DeferredPCA:
    """Wrapper for memory-constrained PCA: stores hap + scaling metadata
    so the caller can compute X @ X.T via chunked matmul without
    materializing the full float64 matrix."""

    def __init__(self, hap, site_mean, scale, scaler):
        self.hap = hap
        self.site_mean = site_mean
        self.scale = scale
        self.scaler = scaler
        self.shape = hap.shape

    def chunked_gram(self):
        """Compute X @ X.T via chunked processing."""
        from ._memutil import estimate_variant_chunk_size
        n_samples, n_var = self.hap.shape
        chunk_size = estimate_variant_chunk_size(n_samples, bytes_per_element=8,
                                                  n_intermediates=2)
        C = cp.zeros((n_samples, n_samples), dtype=cp.float64)
        has_missing = bool(cp.any(self.hap < 0).get())

        for start in range(0, n_var, chunk_size):
            end = min(start + chunk_size, n_var)
            chunk = self.hap[:, start:end]
            if has_missing:
                valid = chunk >= 0
                x = cp.where(valid, chunk, 0).astype(cp.float64)
                x = cp.where(valid, x, self.site_mean[start:end])
            else:
                x = chunk.astype(cp.float64)
            x -= self.site_mean[start:end]
            if self.scaler == 'patterson' and self.scale is not None:
                x /= self.scale[start:end]
            C += x @ x.T
            del x

        return C


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
    prepared = _prepare_matrix(haplotype_matrix, scaler, population, missing_data)

    if isinstance(prepared, _DeferredPCA):
        # Memory-constrained: use covariance trick (eigendecompose X @ X.T)
        n_samples, n_variants = prepared.shape
        n_components = min(n_components, n_samples)
        C = prepared.chunked_gram()  # (n_samples, n_samples)
        eigvals, eigvecs = cp.linalg.eigh(C)
        # eigh returns ascending order; reverse for descending
        idx = cp.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Eigenvalues of C = singular values^2 of X
        coords = eigvecs[:, :n_components] * cp.sqrt(cp.maximum(eigvals[:n_components], 0))
        var = eigvals / n_samples
        explained_variance_ratio = var[:n_components] / cp.sum(cp.maximum(var, 0))
        return coords.get(), explained_variance_ratio.get()

    # Fast path: full SVD
    X = prepared
    n_samples, n_variants = X.shape
    n_components = min(n_components, min(n_samples, n_variants))

    try:
        U, S, Vt = cp.linalg.svd(X, full_matrices=False)
    except Exception as e:
        if 'CUSOLVER' in str(type(e).__name__) or 'CUSOLVER' in str(e):
            raise RuntimeError(
                f"Full SVD failed on matrix of shape ({n_samples}, {n_variants}). "
                f"This dataset is too large for exact PCA. "
                f"Use randomized_pca() instead, which handles large datasets efficiently."
            ) from e
        raise
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

    X = cp.where(hap >= 0, hap, 0).astype(cp.float64)
    valid_mask = (hap >= 0).astype(cp.float64)
    has_missing = cp.any(hap < 0)
    n = X.shape[0]

    if metric in ('euclidean', 'sqeuclidean', 'cityblock'):
        idx_i, idx_j = cp.triu_indices(n, k=1)
        n_pairs = len(idx_i)

        batch_size = max(1, min(n_pairs, 50000))
        dist_parts = []

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            bi = idx_i[start:end]
            bj = idx_j[start:end]

            if has_missing:
                # only compare at jointly-valid sites
                joint = valid_mask[bi] * valid_mask[bj]
                n_joint = cp.sum(joint, axis=1)
            else:
                n_joint = cp.float64(X.shape[1])

            if metric == 'cityblock':
                raw = cp.sum(cp.abs(X[bi] - X[bj]) * (joint if has_missing else 1.0), axis=1)
            else:
                raw = cp.sum(((X[bi] - X[bj]) ** 2) * (joint if has_missing else 1.0), axis=1)

            # normalize by jointly-valid sites
            if has_missing:
                d = cp.where(n_joint > 0, raw * X.shape[1] / n_joint, 0.0)
            else:
                d = raw

            if metric == 'euclidean':
                d = cp.sqrt(d)
            dist_parts.append(d)

        return cp.concatenate(dist_parts).get()
    else:
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
