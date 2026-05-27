"""Tests for pg_gpu.windowed_pca (winpca-style API over local_pca).

Uses the shipped Ag1000G X-chromosome 8-12 Mb / 100-diploid VCZ fixture
under examples/data/.
"""
from pathlib import Path

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, windowed_pca, WindowedPCAResult, local_pca


REPO_ROOT = Path(__file__).parent.parent
GAMB_ZARR = REPO_ROOT / "examples/data/gamb.X.8-12Mb.n100.derived.zarr"


@pytest.fixture(scope="module")
def gamb_region_hm():
    if not GAMB_ZARR.exists():
        pytest.skip(f"fixture not present: {GAMB_ZARR}")
    return HaplotypeMatrix.from_zarr(
        str(GAMB_ZARR), region="X:8000000-10000000", streaming="never",
    )


def test_returns_windowed_pca_result(gamb_region_hm):
    result = windowed_pca(
        gamb_region_hm, window_size=500_000, step_size=500_000,
        n_components=3,
    )
    assert isinstance(result, WindowedPCAResult)
    assert result.coords.ndim == 3
    n_windows, n_samples, n_components = result.coords.shape
    assert n_components == 3
    assert n_samples == gamb_region_hm.num_haplotypes // 2
    assert n_windows == len(result.windows)
    assert n_windows > 0


def test_window_metadata_columns(gamb_region_hm):
    result = windowed_pca(
        gamb_region_hm, window_size=500_000, n_components=2,
    )
    cols = set(result.windows.columns)
    for c in ("chrom", "start", "end", "center", "n_variants",
              "ev_1", "ev_2"):
        assert c in cols, f"missing column: {c}"


def test_eigenvalues_non_increasing(gamb_region_hm):
    """Per-window eigenvalues must be sorted descending (top-k PCs)."""
    result = windowed_pca(
        gamb_region_hm, window_size=500_000, n_components=5,
    )
    ev_cols = [f"ev_{i+1}" for i in range(5)]
    ev = result.windows[ev_cols].to_numpy()
    finite = ev[~np.isnan(ev).all(axis=1)]
    for row in finite:
        valid = row[~np.isnan(row)]
        assert np.all(np.diff(valid) <= 1e-9), \
            f"eigenvalues not descending: {row}"


def test_sample_ids_count(gamb_region_hm):
    result = windowed_pca(
        gamb_region_hm, window_size=500_000, n_components=2,
    )
    assert len(result.sample_ids) == gamb_region_hm.num_haplotypes // 2


def test_component_labels(gamb_region_hm):
    result = windowed_pca(
        gamb_region_hm, window_size=500_000, n_components=4,
    )
    assert result.component_labels == ["PC1", "PC2", "PC3", "PC4"]


def test_overlapping_windows_more_than_non_overlapping(gamb_region_hm):
    r_step  = windowed_pca(gamb_region_hm, window_size=500_000, step_size=500_000, n_components=2)
    r_slide = windowed_pca(gamb_region_hm, window_size=500_000, step_size=250_000, n_components=2)
    assert len(r_slide.windows) > len(r_step.windows)


def test_coords_match_local_pca_eigvecs_transpose(gamb_region_hm):
    """With pre-filters disabled, windowed_pca output is local_pca output
    folded across the haplotype-pair axis and transposed (w, k, n) -> (w, n, k).
    """
    kwargs = dict(window_size=500_000, step_size=500_000, window_type="bp")
    wp = windowed_pca(gamb_region_hm,
                      n_components=3, scaler="patterson",
                      maf_threshold=0.0, ld_prune=False,
                      biallelic_only=False,
                      **kwargs)
    lp = local_pca(gamb_region_hm,
                   k=3, scaler="patterson",
                   **kwargs)
    assert wp.coords.shape[0] == lp.eigvecs.shape[0]

    n_samples = gamb_region_hm.num_haplotypes // 2
    eigvecs_folded = (
        lp.eigvecs[..., :n_samples] + lp.eigvecs[..., n_samples:]
    ) / 2.0
    np.testing.assert_allclose(
        wp.coords,
        np.transpose(eigvecs_folded, (0, 2, 1)),
        equal_nan=True,
    )


def test_maf_filter_reduces_variant_count(gamb_region_hm):
    no_maf  = windowed_pca(gamb_region_hm, window_size=500_000,
                           maf_threshold=0.0, ld_prune=False, n_components=2)
    with_maf = windowed_pca(gamb_region_hm, window_size=500_000,
                            maf_threshold=0.05, ld_prune=False, n_components=2)
    merged = no_maf.windows.merge(
        with_maf.windows, on="start", suffixes=("_raw", "_maf"),
    )
    assert (merged["n_variants_maf"] <= merged["n_variants_raw"]).all()


def test_streaming_matrix_rejected():
    class FakeStreaming:
        __class__ = type("StreamingHaplotypeMatrix", (), {})
    fake = FakeStreaming()
    # __class__.__name__ is what the guard checks
    type(fake).__name__ = "StreamingHaplotypeMatrix"
    with pytest.raises(ValueError, match="streaming"):
        windowed_pca(fake, window_size=10_000)
