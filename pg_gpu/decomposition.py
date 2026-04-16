"""
GPU-accelerated dimensionality reduction and distance computation.

Provides PCA, randomized PCA, PCoA, pairwise genetic distance, and
Li & Ralph (2019) local PCA (lostruct) functions using CuPy.
"""

from dataclasses import dataclass
from typing import Union, Optional, Tuple

import numpy as np
import cupy as cp
import pandas as pd

from .haplotype_matrix import HaplotypeMatrix
from ._utils import get_population_matrix as _get_population_matrix

_GPU_MEM_BUDGET = 0.3


def _prepare_matrix(haplotype_matrix, scaler, population, missing_data='include'):
    """Center and scale haplotype matrix for PCA.

    Uses chunked processing for large matrices to avoid OOM.
    Returns the prepared matrix X on GPU.
    """
    from ._memutil import dac_and_n, estimate_variant_chunk_size

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix

    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    hap = matrix.haplotypes
    n_samples, n_var = hap.shape

    if missing_data == 'exclude':
        dac, nv = dac_and_n(hap)
        complete = nv == n_samples
        if not cp.all(complete):
            hap = hap[:, complete]
            n_var = hap.shape[1]

    # Compute per-site mean and scale from allele counts (memory-safe)
    dac, nv = dac_and_n(hap)
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

    def _scale_chunk(self, start, end):
        """Prepare a scaled float64 chunk of the matrix."""
        chunk = self.hap[:, start:end]
        # Check this chunk for missing (avoids full-matrix boolean allocation)
        has_missing_chunk = bool(cp.any(chunk < 0).get())
        if has_missing_chunk:
            valid = chunk >= 0
            x = cp.where(valid, chunk, 0).astype(cp.float64)
            x = cp.where(valid, x, self.site_mean[start:end])
        else:
            x = chunk.astype(cp.float64)
        x -= self.site_mean[start:end]
        if self.scaler == 'patterson' and self.scale is not None:
            x /= self.scale[start:end]
        return x

    @property
    def _chunk_size(self):
        from ._memutil import estimate_variant_chunk_size
        return estimate_variant_chunk_size(self.shape[0], bytes_per_element=8,
                                           n_intermediates=2)

    def chunked_gram(self):
        """Compute X @ X.T via chunked processing."""
        n_samples, n_var = self.shape
        C = cp.zeros((n_samples, n_samples), dtype=cp.float64)
        for start in range(0, n_var, self._chunk_size):
            end = min(start + self._chunk_size, n_var)
            x = self._scale_chunk(start, end)
            C += x @ x.T
            del x
        return C

    def __matmul__(self, other):
        """Compute X @ other via chunked processing."""
        n_samples, n_var = self.shape
        result = cp.zeros((n_samples, other.shape[1]), dtype=cp.float64)
        for start in range(0, n_var, self._chunk_size):
            end = min(start + self._chunk_size, n_var)
            x = self._scale_chunk(start, end)
            result += x @ other[start:end, :]
            del x
        return result

    @property
    def T(self):
        """Return a transposed view for X.T @ Y operations."""
        return _DeferredPCATranspose(self)


class _DeferredPCATranspose:
    """Transpose view of _DeferredPCA for X.T @ Y operations."""

    def __init__(self, parent):
        self.parent = parent
        self.shape = (parent.shape[1], parent.shape[0])

    def __matmul__(self, other):
        """Compute X.T @ other via chunked processing."""
        n_samples, n_var = self.parent.shape
        result = cp.zeros((n_var, other.shape[1]), dtype=cp.float64)
        for start in range(0, n_var, self.parent._chunk_size):
            end = min(start + self.parent._chunk_size, n_var)
            x = self.parent._scale_chunk(start, end)
            result[start:end, :] = x.T @ other
            del x
        return result


def _pca_from_gram(C, n_samples, n_components):
    """Extract PCA coordinates from a Gram matrix (X @ X.T).

    Eigendecomposes the n x n Gram matrix instead of running SVD on the
    full n x m data matrix. Equivalent when n_samples <= n_variants.
    """
    eigvals, eigvecs = cp.linalg.eigh(C)
    idx = cp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    coords = eigvecs[:, :n_components] * cp.sqrt(cp.maximum(eigvals[:n_components], 0))
    var = eigvals / n_samples
    explained_variance_ratio = var[:n_components] / cp.sum(cp.maximum(var, 0))
    return coords.get(), explained_variance_ratio.get()


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
        'include' - impute missing to per-site mean
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
        n_samples = prepared.shape[0]
        n_components = min(n_components, n_samples)
        C = prepared.chunked_gram()
        return _pca_from_gram(C, n_samples, n_components)

    X = prepared
    n_samples, n_variants = X.shape
    n_components = min(n_components, min(n_samples, n_variants))

    # When n_samples <= n_variants (typical for popgen), eigendecompose
    # the n x n Gram matrix X @ X.T instead of full SVD on n x m.
    if n_samples <= n_variants:
        C = X @ X.T
        return _pca_from_gram(C, n_samples, n_components)

    # Fallback: full SVD when n_samples > n_variants
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
    if isinstance(X, _DeferredPCA):
        # B = Q.T @ X has shape (k, n_var) — too wide for cuSOLVER SVD.
        # Instead compute B @ B.T = Q.T @ (X @ X.T) @ Q = Q.T @ gram @ Q
        # and eigendecompose the small (k, k) matrix.
        gram = X.chunked_gram()  # (n_samples, n_samples)
        M = Q.T @ gram @ Q  # (k, k)
        eigvals, eigvecs = cp.linalg.eigh(M)
        idx = cp.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        S = cp.sqrt(cp.maximum(eigvals, 0))
        Uhat = eigvecs
    else:
        B = Q.T @ X
        Uhat, S, Vt = cp.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    coords = U[:, :n_components] * S[:n_components]

    # explained variance
    if isinstance(X, _DeferredPCA):
        # Reuse gram from above (already computed for eigendecomposition)
        total_var = cp.trace(gram) / n_samples
    else:
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
        'include' - mask missing, normalize per pair
        'exclude' - filter sites with any missing

    Returns
    -------
    dist : ndarray, float64, shape (n_samples * (n_samples - 1) // 2,)
        Condensed distance matrix.
    """

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

        # Estimate batch size from available GPU memory
        n_variants = X.shape[1]
        free_mem = cp.cuda.Device().mem_info[0]
        # Each pair needs ~3 float64 arrays of n_variants (diff, joint, result)
        bytes_per_pair = n_variants * 8 * 3
        batch_size = max(1, min(n_pairs, int(free_mem * 0.3 / bytes_per_pair)))
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


# ---------------------------------------------------------------------------
# Local PCA (lostruct; Li & Ralph 2019)
# ---------------------------------------------------------------------------


@dataclass
class LocalPCAResult:
    """Per-window local PCA output.

    Attributes
    ----------
    windows : pandas.DataFrame
        One row per window (chrom, start, end, center, n_variants, window_id).
    eigvals : numpy.ndarray, float64, shape (n_windows, k)
        Top-k eigenvalues of each window's sample-sample covariance matrix,
        descending. NaN for windows with fewer than k variants.
    eigvecs : numpy.ndarray, float64, shape (n_windows, k, n_samples)
        Top-k eigenvectors. NaN for invalid windows.
    sumsq : numpy.ndarray, float64, shape (n_windows,)
        Sum of squared entries of each window's covariance matrix; used by
        `pc_dist` for the proportion-of-variance denominator.
    k : int
        Number of PCs retained.
    scaler : str or None
    missing_data : str
    jackknife_se : numpy.ndarray or None
        Delete-1 block jackknife SE of the top-k eigenvectors. Shape
        ``(n_windows, k)`` with ``aggregate='mean'``, or
        ``(n_windows, k, n_samples)`` with ``aggregate=None``.
        ``None`` when jackknife was not computed.
    """

    windows: "pd.DataFrame"
    eigvals: np.ndarray
    eigvecs: np.ndarray
    sumsq: np.ndarray
    k: int
    scaler: Optional[str]
    missing_data: str
    jackknife_se: Optional[np.ndarray] = None

    @property
    def n_windows(self) -> int:
        return self.eigvals.shape[0]

    @property
    def n_samples(self) -> int:
        return self.eigvecs.shape[2]

    def to_lostruct_matrix(self) -> np.ndarray:
        """Flat `(n_windows, 1 + k + k*n_samples)` matrix matching
        `lostruct::eigen_windows()` output layout.

        Eigenvector block is column-major per R: PC_1 samples 1..N, then PC_2
        samples 1..N, etc. That coincides with the C-order flatten of our
        `(k, n_samples)` per-window array.
        """
        nw = self.n_windows
        out = np.empty((nw, 1 + self.k + self.k * self.n_samples), dtype=np.float64)
        out[:, 0] = self.sumsq
        out[:, 1:1 + self.k] = self.eigvals
        out[:, 1 + self.k:] = self.eigvecs.reshape(nw, -1)
        return out


def _window_gram(X: "cp.ndarray", n_var: int) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """Return lostruct-equivalent sample-sample covariance and its sum-of-squares.

    Replicates R's `cov(sweep(x, 1, rowMeans(x), "-"))`: already-variant-centered
    `X` (rows=samples, cols=variants) gets an additional sample-centering, then
    `C = (X @ X.T) / (n_var - 1)` and `sumsq = sum(C**2)`. Sum-of-squares is
    returned as a 0-d GPU scalar so callers can defer the host sync.
    """
    X = X - X.mean(axis=1, keepdims=True)
    denom = max(n_var - 1, 1)
    C = (X @ X.T) / denom
    return C, (C ** 2).sum()


def _batched_top_k_eigh(gram_stack: "cp.ndarray", k: int,
                        batch_size: Optional[int] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Batched eigendecomposition on a stack of symmetric matrices.

    Returns top-k eigvals (descending) and eigvecs as host arrays.

    Parameters
    ----------
    gram_stack : cupy.ndarray, float64, shape (n_windows, n, n)
    k : int
    batch_size : int, optional
        Chunk size across the window axis. If None, uses all windows at once
        unless memory budget is exceeded.

    Returns
    -------
    eigvals : numpy.ndarray, shape (n_windows, k)
    eigvecs : numpy.ndarray, shape (n_windows, k, n)
    """
    n_windows, n, _ = gram_stack.shape
    if batch_size is None:
        free = cp.cuda.Device().mem_info[0]
        per_window = n * n * 8 * 4  # gram + workspace + eigvecs + scratch
        batch_size = max(1, min(n_windows,
                                int(free * _GPU_MEM_BUDGET) // max(per_window, 1)))

    eigvals_out = np.empty((n_windows, k), dtype=np.float64)
    eigvecs_out = np.empty((n_windows, k, n), dtype=np.float64)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        evals, evecs = cp.linalg.eigh(gram_stack[start:end])
        # eigh returns ascending; top-k is the last k reversed.
        top = cp.arange(n - 1, n - 1 - k, -1)
        top_vals = evals[:, top]
        top_vecs = cp.transpose(evecs[:, :, top], (0, 2, 1))
        eigvals_out[start:end] = top_vals.get()
        eigvecs_out[start:end] = top_vecs.get()

    return eigvals_out, eigvecs_out


def _resolve_window_params(window_params, window_size, step_size, window_type,
                           regions, caller: str):
    """Return a WindowParams, either passed through or built from short-hand."""
    from .windowed_analysis import WindowParams

    if window_params is not None:
        return window_params
    if window_size is None:
        raise ValueError(f"{caller} requires window_params or window_size.")
    return WindowParams(
        window_type=window_type,
        window_size=window_size,
        step_size=step_size if step_size is not None else window_size,
        regions=regions,
    )


def _materialize_prepared(X):
    """If `_prepare_matrix` returned a deferred chunked view, concatenate it.

    Per-window matrices are small enough that a single contiguous copy is fine
    and it lets us apply additional in-place centering downstream.
    """
    if isinstance(X, _DeferredPCA):
        n_var = X.shape[1]
        return cp.concatenate(
            [X._scale_chunk(s, min(s + X._chunk_size, n_var))
             for s in range(0, n_var, X._chunk_size)], axis=1)
    return X


def local_pca(haplotype_matrix: "HaplotypeMatrix",
              window_params=None,
              k: int = 2,
              scaler: Optional[str] = None,
              missing_data: str = 'include',
              population: Optional[Union[str, list]] = None,
              batch_size: Optional[int] = None,
              window_size: Optional[int] = None,
              step_size: Optional[int] = None,
              window_type: str = 'snp',
              regions=None) -> LocalPCAResult:
    """Local PCA along the genome (lostruct; Li & Ralph 2019).

    For each genomic window, computes the top-k eigenvalues/eigenvectors of the
    sample-sample covariance matrix. The covariance matches R's
    `cov(sweep(x, 1, rowMeans(x), "-"))` so outputs are directly comparable to
    `lostruct::eigen_windows()`.

    Parameters
    ----------
    haplotype_matrix : HaplotypeMatrix
    window_params : WindowParams, optional
        Pre-built window spec. Required when ``window_size`` is not given.
    k : int
        Number of PCs to retain per window.
    scaler : str or None
        None (lostruct default), 'patterson', or 'standard'. See `pca()`.
    missing_data : str
        'include' (impute to per-site mean) or 'exclude' (drop missing sites).
        lostruct uses pairwise-complete; we approximate with mean imputation.
    population : str or list, optional
    batch_size : int, optional
        Windows per batched eigh call. Auto-sized if None.
    window_size, step_size, window_type, regions
        Short-hand for constructing a `WindowParams` without importing it.

    Returns
    -------
    LocalPCAResult
    """
    from .windowed_analysis import WindowIterator

    window_params = _resolve_window_params(
        window_params, window_size, step_size, window_type, regions,
        caller='local_pca')

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    n_samples = matrix.num_haplotypes
    iterator = WindowIterator(matrix, window_params)
    identity_placeholder = cp.eye(n_samples, dtype=cp.float64)
    nan_scalar = cp.asarray(np.nan, dtype=cp.float64)

    window_meta = []
    gram_list = []
    sumsq_list = []
    valid_flags = []

    for window in iterator:
        window_meta.append(window)
        if window.n_variants < max(k, 2):
            gram_list.append(identity_placeholder)
            sumsq_list.append(nan_scalar)
            valid_flags.append(False)
            continue

        X = _prepare_matrix(window.matrix, scaler, population=None,
                            missing_data=missing_data)
        X = _materialize_prepared(X)
        C, sumsq = _window_gram(X, X.shape[1])
        gram_list.append(C)
        sumsq_list.append(sumsq)
        valid_flags.append(True)

    if len(window_meta) == 0:
        raise ValueError("WindowIterator produced no windows.")

    gram_stack = cp.stack(gram_list, axis=0)
    eigvals_host, eigvecs_host = _batched_top_k_eigh(gram_stack, k, batch_size)
    sumsq_host = cp.stack(sumsq_list).get()

    valid_mask = np.array(valid_flags, dtype=bool)
    eigvals_host[~valid_mask] = np.nan
    eigvecs_host[~valid_mask] = np.nan
    sumsq_host[~valid_mask] = np.nan

    windows_df = pd.DataFrame({
        'chrom': [w.chrom for w in window_meta],
        'start': [w.start for w in window_meta],
        'end': [w.end for w in window_meta],
        'center': [w.center for w in window_meta],
        'n_variants': [w.n_variants for w in window_meta],
        'window_id': [w.window_id for w in window_meta],
    })

    return LocalPCAResult(
        windows=windows_df,
        eigvals=eigvals_host,
        eigvecs=eigvecs_host,
        sumsq=sumsq_host,
        k=k,
        scaler=scaler,
        missing_data=missing_data,
    )


def _local_pca_with_jackknife(
    haplotype_matrix: "HaplotypeMatrix",
    window_params=None,
    k: int = 2,
    n_blocks: int = 10,
    scaler: Optional[str] = None,
    missing_data: str = 'include',
    population: Optional[Union[str, list]] = None,
    aggregate: Optional[str] = 'mean',
    batch_size: Optional[int] = None,
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    window_type: str = 'snp',
    regions=None,
) -> LocalPCAResult:
    """Local PCA with fused jackknife SE, sharing per-window matrix preparation.

    Iterates windows once; for each valid window calls ``_prepare_matrix`` once
    and from the same materialized ``X`` computes both the full Gram matrix (for
    eigvals/eigvecs) and the leave-one-block-out Gram matrices (for jackknife).

    Two-tier validity: windows with enough variants for PCA (``>= max(k, 2)``)
    but too few for jackknife (``< max(k, 2) * n_blocks``) get valid
    eigvals/eigvecs but NaN ``jackknife_se``.
    """
    from .windowed_analysis import WindowIterator

    window_params = _resolve_window_params(
        window_params, window_size, step_size, window_type, regions,
        caller='_local_pca_with_jackknife')

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    n_samples = matrix.num_haplotypes
    iterator = WindowIterator(matrix, window_params)
    identity_placeholder = cp.eye(n_samples, dtype=cp.float64)
    nan_scalar = cp.asarray(np.nan, dtype=cp.float64)
    identity_blocks = cp.broadcast_to(
        identity_placeholder, (n_blocks, n_samples, n_samples))

    min_pca = max(k, 2)
    min_jk = min_pca * n_blocks

    window_meta = []
    gram_list = []
    sumsq_list = []
    jk_gram_list = []
    valid_pca = []
    valid_jk = []

    for window in iterator:
        window_meta.append(window)

        if window.n_variants < min_pca:
            # Too few variants for PCA or jackknife.
            gram_list.append(identity_placeholder)
            sumsq_list.append(nan_scalar)
            jk_gram_list.append(identity_blocks)
            valid_pca.append(False)
            valid_jk.append(False)
            continue

        X = _prepare_matrix(window.matrix, scaler, population=None,
                            missing_data=missing_data)
        X = _materialize_prepared(X)
        n_var_win = X.shape[1]

        # Full Gram for local_pca.
        C, sumsq = _window_gram(X, n_var_win)
        gram_list.append(C)
        sumsq_list.append(sumsq)
        valid_pca.append(True)

        # Leave-one-block-out Grams for jackknife.
        step = n_var_win // n_blocks
        if n_var_win < min_jk or step < 1:
            jk_gram_list.append(identity_blocks)
            valid_jk.append(False)
            continue

        window_grams = cp.empty((n_blocks, n_samples, n_samples),
                                dtype=cp.float64)
        for b in range(n_blocks):
            lo = b * step
            hi = (b + 1) * step
            keep = cp.concatenate([cp.arange(0, lo), cp.arange(hi, n_var_win)])
            Xk = X[:, keep]
            Xk = Xk - Xk.mean(axis=1, keepdims=True)
            denom = max(Xk.shape[1] - 1, 1)
            window_grams[b] = (Xk @ Xk.T) / denom
        jk_gram_list.append(window_grams)
        valid_jk.append(True)

    n_windows = len(window_meta)
    if n_windows == 0:
        raise ValueError("WindowIterator produced no windows.")

    # --- Main eigendecomposition (local_pca) ---
    gram_stack = cp.stack(gram_list, axis=0)
    eigvals_host, eigvecs_host = _batched_top_k_eigh(gram_stack, k, batch_size)
    sumsq_host = cp.stack(sumsq_list).get()

    pca_mask = np.array(valid_pca, dtype=bool)
    eigvals_host[~pca_mask] = np.nan
    eigvecs_host[~pca_mask] = np.nan
    sumsq_host[~pca_mask] = np.nan

    # --- Jackknife eigendecomposition ---
    jk_stack = cp.concatenate(jk_gram_list, axis=0)
    _, jk_vecs = _batched_top_k_eigh(jk_stack, k, batch_size)
    jk_vecs_4d = cp.asarray(jk_vecs).reshape(n_windows, n_blocks, k, n_samples)
    aligned = _sign_align_replicates(jk_vecs_4d)
    mean = aligned.mean(axis=1, keepdims=True)
    jackknife_scale = (n_blocks - 1) / n_blocks
    var = jackknife_scale * cp.sum((aligned - mean) ** 2, axis=1)
    se_out = cp.sqrt(var).get()  # (n_windows, k, n_samples)

    jk_mask = np.array(valid_jk, dtype=bool)
    se_out[~jk_mask] = np.nan

    if aggregate == 'mean':
        jackknife_se = np.nanmean(se_out, axis=2)  # (n_windows, k)
    elif aggregate is None:
        jackknife_se = se_out
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    windows_df = pd.DataFrame({
        'chrom': [w.chrom for w in window_meta],
        'start': [w.start for w in window_meta],
        'end': [w.end for w in window_meta],
        'center': [w.center for w in window_meta],
        'n_variants': [w.n_variants for w in window_meta],
        'window_id': [w.window_id for w in window_meta],
    })

    return LocalPCAResult(
        windows=windows_df,
        eigvals=eigvals_host,
        eigvecs=eigvecs_host,
        sumsq=sumsq_host,
        k=k,
        scaler=scaler,
        missing_data=missing_data,
        jackknife_se=jackknife_se,
    )


def pc_dist(source, npc: Optional[int] = None,
            normalize: Optional[str] = 'L1',
            w=1) -> np.ndarray:
    """Pairwise Frobenius distance between windows' low-rank covariance reps.

    Implements the trace identity from ``lostruct::pc_dist``::

        ||A - B||^2 = sum(a^2) + sum(b^2) - 2 * tr(diag(a) X diag(b) X^T)

    where ``X = U^T V`` for eigenvector blocks U, V.

    Parameters
    ----------
    source : LocalPCAResult or numpy.ndarray
        Either a LocalPCAResult or a flat lostruct-style matrix of shape
        (n_windows, 1 + k + k*n_samples).
    npc : int, optional
        Number of PCs to use. Defaults to ``source.k`` when ``source`` is a
        LocalPCAResult; required when ``source`` is a flat matrix.
    normalize : {'L1', 'L2', None}
        Per-window eigenvalue normalization before distance computation.
    w : array_like or scalar
        Sample weights (default 1).

    Returns
    -------
    numpy.ndarray, shape (n_windows, n_windows)
        Symmetric, non-negative distance matrix. NaN for invalid windows.
    """
    if isinstance(source, LocalPCAResult):
        if npc is None:
            npc = source.k
        eigvals = source.eigvals[:, :npc].copy()
        eigvecs = source.eigvecs[:, :npc, :].copy()
    else:
        arr = np.asarray(source, dtype=np.float64)
        if npc is None:
            raise ValueError("npc must be provided when source is a flat matrix.")
        n_windows = arr.shape[0]
        n_samples = (arr.shape[1] - 1 - npc) // npc
        eigvals = arr[:, 1:1 + npc].copy()
        eigvecs = arr[:, 1 + npc:].reshape(n_windows, npc, n_samples).copy()

    n_windows, npc_actual, n_samples = eigvecs.shape
    if npc_actual != npc:
        raise ValueError(f"npc mismatch: requested {npc}, have {npc_actual}")

    w_arr = np.broadcast_to(np.sqrt(np.asarray(w, dtype=np.float64)),
                            (n_samples,))
    eigvecs = eigvecs * w_arr[None, None, :]

    if normalize == 'L1':
        denom = np.sum(np.abs(eigvals), axis=1, keepdims=True)
        eigvals = np.divide(eigvals, denom,
                            out=np.full_like(eigvals, np.nan),
                            where=denom != 0)
    elif normalize == 'L2':
        denom = np.sqrt(np.sum(eigvals ** 2, axis=1, keepdims=True))
        eigvals = np.divide(eigvals, denom,
                            out=np.full_like(eigvals, np.nan),
                            where=denom != 0)
    elif normalize is not None:
        raise ValueError(f"Unknown normalize: {normalize}")

    vals = cp.asarray(eigvals)             # (nw, k)
    vecs = cp.asarray(eigvecs)             # (nw, k, n_samples)
    self_norm = cp.sum(vals ** 2, axis=1)  # (nw,)

    # Trace-identity cross term (pc_dist.R:57-61):
    #   X[i,j] = U_i^T V_j                            (k, k)
    #   tr(diag(a_i) X diag(b_j) X^T)
    #     = sum_{m,n} a_i[m] * b_j[n] * X[m,n]^2
    # Full-matrix path materializes the (nw, nw, k, k) tensor; chunked path
    # splits the outer window axis when that would exceed the memory budget.
    nw = vals.shape[0]
    free = cp.cuda.Device().mem_info[0]
    k_pc = vals.shape[1]
    full_bytes = nw * nw * k_pc * k_pc * 8
    if full_bytes < free * _GPU_MEM_BUDGET:
        X_all = cp.einsum('ims,jns->ijmn', vecs, vecs)
        cross = cp.einsum('im,jn,ijmn->ij', vals, vals, X_all ** 2)
    else:
        cross = cp.zeros((nw, nw), dtype=cp.float64)
        per_row_bytes = nw * k_pc * k_pc * 8 * 4
        chunk = max(1, int(free * _GPU_MEM_BUDGET) // max(per_row_bytes, 1))
        for start in range(0, nw, chunk):
            end = min(start + chunk, nw)
            X_chunk = cp.einsum('ims,jns->ijmn', vecs[start:end], vecs)
            cross[start:end] = cp.einsum('im,jn,ijmn->ij',
                                         vals[start:end], vals, X_chunk ** 2)

    sq_dist = self_norm[:, None] + self_norm[None, :] - 2.0 * cross
    # Symmetrize and clamp negative numerical error to 0
    sq_dist = 0.5 * (sq_dist + sq_dist.T)
    sq_dist = cp.maximum(sq_dist, 0.0)
    dist = cp.sqrt(sq_dist).get()

    # NaN-propagate: invalid windows (any NaN in eigvals) produce NaN rows/cols
    if np.any(np.isnan(eigvals)):
        bad = np.any(np.isnan(eigvals), axis=1)
        dist[bad, :] = np.nan
        dist[:, bad] = np.nan

    return dist


# ---------------------------------------------------------------------------
# Minimum enclosing circle (Welzl) + corners post-processing
# ---------------------------------------------------------------------------


def _circle_from_2(p1, p2):
    c = 0.5 * (p1 + p2)
    r = float(np.linalg.norm(p1 - c))
    return c, r


def _circle_from_3(p1, p2, p3):
    # Numerically-stable circumcircle via perpendicular bisectors.
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # Collinear: fall back to diameter of the farthest pair
        pairs = [(p1, p2), (p1, p3), (p2, p3)]
        pairs.sort(key=lambda pr: np.linalg.norm(pr[0] - pr[1]), reverse=True)
        return _circle_from_2(*pairs[0])
    ux = ((ax * ax + ay * ay) * (by - cy)
          + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx)
          + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    c = np.array([ux, uy])
    r = float(max(np.linalg.norm(p - c) for p in (p1, p2, p3)))
    return c, r


def _in_circle(p, c, r, tol=1e-10):
    return float(np.linalg.norm(p - c)) <= r + tol


def _welzl_mec(points: np.ndarray, rng: np.random.Generator):
    """Randomized Welzl minimum enclosing circle, iterative form."""
    pts = points.copy()
    rng.shuffle(pts)
    n = len(pts)

    c = np.array([0.0, 0.0])
    r = 0.0
    boundary = []
    for i in range(n):
        if r > 0 and _in_circle(pts[i], c, r):
            continue
        # Rebuild with pts[i] on boundary
        c, r = pts[i], 0.0
        for j in range(i):
            if _in_circle(pts[j], c, r):
                continue
            c, r = _circle_from_2(pts[i], pts[j])
            for k in range(j):
                if _in_circle(pts[k], c, r):
                    continue
                c, r = _circle_from_3(pts[i], pts[j], pts[k])
    return c, r


def _mec_defining_points(xy: np.ndarray, rng: np.random.Generator,
                         tol: float = 1e-6):
    """Find up to 3 input points that lie on the MEC boundary.

    Returns (center, radius, indices_on_boundary) using original xy indices
    (NaN rows already filtered by caller).
    """
    c, r = _welzl_mec(xy, rng)
    if r <= 0:
        return c, r, np.array([0], dtype=np.int64)
    dists = np.linalg.norm(xy - c, axis=1)
    on_circle = np.where(np.abs(dists - r) <= tol * max(r, 1.0))[0]
    return c, r, on_circle


def corners(xy: np.ndarray, prop: float, k: int = 3,
            random_state: Optional[int] = None) -> np.ndarray:
    """Find `k` "corner" clusters in a 2D embedding (lostruct::corners).

    Computes the minimum enclosing circle of `xy`, then for each of the `k`
    defining points returns the `int(prop * n)` nearest `xy` points. If fewer
    than `k` defining points exist on the MEC boundary, extra corners are
    added greedily from points farthest from the MEC center and not already
    in an existing corner's neighborhood (mirrors `corners.R:42-53`).

    Parameters
    ----------
    xy : numpy.ndarray, shape (n, 2)
        Coordinates (e.g. MDS output). NaN rows are dropped before the MEC.
    prop : float
        Fraction of points per corner.
    k : int
        Number of corners (effective minimum 3, matching R).
    random_state : int, optional

    Returns
    -------
    numpy.ndarray, int64, shape (n_per_corner, k)
        Column i = indices (into the original, NaN-inclusive xy) of points
        closest to the i-th corner.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (n, 2)")

    k_eff = max(k, 3)
    valid_mask = ~np.any(np.isnan(xy), axis=1)
    valid_idx = np.where(valid_mask)[0]
    pts = xy[valid_idx]
    if len(pts) < k_eff:
        raise ValueError(
            f"Need at least {k_eff} non-NaN points; got {len(pts)}")

    n_per = max(1, int(prop * len(pts)))
    rng = np.random.default_rng(random_state)
    center, radius, boundary_local = _mec_defining_points(pts, rng)

    # Build corner index list (indices into `pts`)
    corner_point_idx = list(boundary_local[:k_eff])

    # Fallback: extend with farthest-from-center points not already covered
    if len(corner_point_idx) < k_eff:
        used = set()
        for ci in corner_point_idx:
            cdist = np.linalg.norm(pts - pts[ci], axis=1)
            nearest = np.argsort(cdist)[:n_per]
            used.update(nearest.tolist())
        center_dist = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(center_dist)[::-1]
        for cand in order:
            if len(corner_point_idx) >= k_eff:
                break
            if int(cand) in used:
                continue
            corner_point_idx.append(int(cand))
            cdist = np.linalg.norm(pts - pts[cand], axis=1)
            used.update(np.argsort(cdist)[:n_per].tolist())

    # For each corner, take n_per nearest (in the valid point set)
    out = np.empty((n_per, k_eff), dtype=np.int64)
    for i, ci in enumerate(corner_point_idx[:k_eff]):
        d = np.linalg.norm(pts - pts[ci], axis=1)
        nearest = np.argsort(d)[:n_per]
        # Map back to original (NaN-inclusive) indices
        out[:, i] = valid_idx[nearest]

    return out


# ---------------------------------------------------------------------------
# Jackknife SE for local PCs
# ---------------------------------------------------------------------------


def _sign_align_replicates(vecs: "cp.ndarray") -> "cp.ndarray":
    """Sign-align replicate eigenvectors to replicate 0.

    Accepts either ``(n_reps, k, n_samples)`` (single window) or
    ``(n_windows, n_reps, k, n_samples)``. For each (rep, pc) within its
    window, flip sign if ``||v - v_0||^2 > ||v + v_0||^2``, equivalent to
    ``sign(<v, v_0>) < 0``.
    """
    if vecs.ndim == 3:
        ref = vecs[0]                                 # (k, n_samples)
        dot = cp.einsum('rks,ks->rk', vecs, ref)      # (n_reps, k)
        sign = cp.where(dot >= 0, 1.0, -1.0)[:, :, None]
    elif vecs.ndim == 4:
        ref = vecs[:, 0]                              # (n_windows, k, n_samples)
        dot = cp.einsum('wrks,wks->wrk', vecs, ref)   # (n_windows, n_reps, k)
        sign = cp.where(dot >= 0, 1.0, -1.0)[:, :, :, None]
    else:
        raise ValueError(
            f"vecs must have 3 or 4 dims, got {vecs.ndim}")
    return vecs * sign


def local_pca_jackknife(haplotype_matrix: "HaplotypeMatrix",
                        window_params=None,
                        k: int = 2,
                        n_blocks: int = 10,
                        scaler: Optional[str] = None,
                        missing_data: str = 'include',
                        population: Optional[Union[str, list]] = None,
                        aggregate: Optional[str] = 'mean',
                        batch_size: Optional[int] = None,
                        window_size: Optional[int] = None,
                        step_size: Optional[int] = None,
                        window_type: str = 'snp',
                        regions=None) -> np.ndarray:
    """Delete-1 block jackknife SE of local PCs (DPGP_jackknife_var.R port).

    For each window, partitions variants into `n_blocks` contiguous blocks;
    for each block, recomputes the sample-sample covariance with that block
    removed, eigendecomposes it, takes the top-k eigenvectors, sign-aligns
    across replicates, and computes the jackknife SE per sample.

    Parameters
    ----------
    n_blocks : int
        Number of jackknife blocks per window.
    aggregate : {'mean', None}
        'mean' returns shape (n_windows, k) — per-PC SE averaged across
        samples (matches R's `mean(b)`). None returns the full
        (n_windows, k, n_samples) tensor.

    Returns
    -------
    numpy.ndarray
        NaN rows for windows with fewer than `max(k, n_blocks+1)` variants.
    """
    from .windowed_analysis import WindowIterator

    window_params = _resolve_window_params(
        window_params, window_size, step_size, window_type, regions,
        caller='local_pca_jackknife')

    if population is not None:
        matrix = _get_population_matrix(haplotype_matrix, population)
    else:
        matrix = haplotype_matrix
    if matrix.device == 'CPU':
        matrix.transfer_to_gpu()

    n_samples = matrix.num_haplotypes
    iterator = WindowIterator(matrix, window_params)
    identity_blocks = cp.broadcast_to(
        cp.eye(n_samples, dtype=cp.float64),
        (n_blocks, n_samples, n_samples))

    min_variants = max(k, 2) * n_blocks
    all_grams = []
    valid_flags = []

    for window in iterator:
        if window.n_variants < min_variants:
            all_grams.append(identity_blocks)
            valid_flags.append(False)
            continue

        X = _prepare_matrix(window.matrix, scaler, population=None,
                            missing_data=missing_data)
        X = _materialize_prepared(X)
        n_var_win = X.shape[1]
        # Block width truncates (matches R's `round(nrow/10)` in DPGP_jackknife_var.R).
        step = n_var_win // n_blocks
        if step < 1:
            all_grams.append(identity_blocks)
            valid_flags.append(False)
            continue

        window_grams = cp.empty((n_blocks, n_samples, n_samples),
                                dtype=cp.float64)
        for b in range(n_blocks):
            lo = b * step
            hi = (b + 1) * step
            keep = cp.concatenate([cp.arange(0, lo), cp.arange(hi, n_var_win)])
            Xk = X[:, keep]
            Xk = Xk - Xk.mean(axis=1, keepdims=True)
            denom = max(Xk.shape[1] - 1, 1)
            window_grams[b] = (Xk @ Xk.T) / denom
        all_grams.append(window_grams)
        valid_flags.append(True)

    n_windows = len(all_grams)
    if n_windows == 0:
        raise ValueError("WindowIterator produced no windows.")

    gram_stack = cp.concatenate(all_grams, axis=0)
    eigvals_host, eigvecs_host = _batched_top_k_eigh(gram_stack, k, batch_size)
    eigvecs_4d_gpu = cp.asarray(eigvecs_host).reshape(
        n_windows, n_blocks, k, n_samples)
    aligned = _sign_align_replicates(eigvecs_4d_gpu)
    mean = aligned.mean(axis=1, keepdims=True)
    jackknife_scale = (n_blocks - 1) / n_blocks
    var = jackknife_scale * cp.sum((aligned - mean) ** 2, axis=1)
    se_out = cp.sqrt(var).get()                # (n_windows, k, n_samples)

    valid_mask = np.array(valid_flags, dtype=bool)
    se_out[~valid_mask] = np.nan

    if aggregate == 'mean':
        return np.nanmean(se_out, axis=2)      # (n_windows, k)
    elif aggregate is None:
        return se_out
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
