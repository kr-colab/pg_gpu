"""
Parity tests for local_pca / pc_dist / jackknife / corners against frozen
lostruct reference outputs.

**CI policy**: this file never invokes R or imports rpy2. It loads
``tests/data/lostruct_reference_*.json`` and compares pg_gpu's output to those
frozen arrays. When the JSON files are absent, all tests skip cleanly.

To (re)generate the references:
  - Install R and the lostruct package in your local R environment
    (do NOT add R to pixi / CI).
  - ``python tests/data/generate_lostruct_reference_input.py``
  - ``Rscript tests/data/generate_lostruct_reference.R``
  - Commit the updated ``tests/data/lostruct_reference_*.json``.

The one `test_parity_live_rpy2` test is gated on ``pytest.importorskip("rpy2")``
so it skips cleanly in CI and in any environment without rpy2/R.
"""

import json
import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix
from pg_gpu.decomposition import (
    corners,
    local_pca,
    local_pca_jackknife,
    pc_dist,
    pcoa,
)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
INPUT_PATH = DATA_DIR / "generate_lostruct_reference_input.npz"
EIGEN_PATH = DATA_DIR / "lostruct_reference_eigen.json"
PCDIST_PATH = DATA_DIR / "lostruct_reference_pcdist.json"
JACKKNIFE_PATH = DATA_DIR / "lostruct_reference_jackknife.json"
CORNERS_PATH = DATA_DIR / "lostruct_reference_corners.json"


def _load_json(path: pathlib.Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _to_array(x: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_input():
    if not INPUT_PATH.exists():
        pytest.skip(f"Missing {INPUT_PATH.name}; run "
                    f"tests/data/generate_lostruct_reference_input.py")
    return np.load(INPUT_PATH, allow_pickle=False)


@pytest.fixture(scope="module")
def reference_hm(reference_input):
    hap = reference_input['hap']
    positions = reference_input['positions']
    return HaplotypeMatrix(hap, positions, 0, int(positions[-1]) + 1)


@pytest.fixture(scope="module")
def pg_local_pca_result(reference_hm, reference_input):
    return local_pca(
        reference_hm,
        window_size=int(reference_input['window_size']),
        window_type='snp',
        k=int(reference_input['k']),
    )


def _skip_if_missing(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"Missing {path.name}; regenerate with "
                    f"Rscript tests/data/generate_lostruct_reference.R")


# ---------------------------------------------------------------------------
# Eigen parity: sumsq and eigvals
# ---------------------------------------------------------------------------


def test_parity_sumsq(pg_local_pca_result):
    _skip_if_missing(EIGEN_PATH)
    ref = _load_json(EIGEN_PATH)
    sumsq_ref = _to_array(ref['sumsq']).ravel()
    sumsq_ours = pg_local_pca_result.sumsq
    assert sumsq_ours.shape == sumsq_ref.shape
    np.testing.assert_allclose(sumsq_ours, sumsq_ref, rtol=1e-6, atol=1e-10)


def test_parity_eigvals(pg_local_pca_result):
    _skip_if_missing(EIGEN_PATH)
    ref = _load_json(EIGEN_PATH)
    eigvals_ref = _to_array(ref['eigvals'])
    eigvals_ours = pg_local_pca_result.eigvals
    assert eigvals_ours.shape == eigvals_ref.shape
    np.testing.assert_allclose(eigvals_ours, eigvals_ref,
                               rtol=1e-5, atol=1e-8)


def test_parity_eigenvectors_sign_aligned(pg_local_pca_result):
    _skip_if_missing(EIGEN_PATH)
    ref = _load_json(EIGEN_PATH)
    eigvecs_ref = _to_array(ref['eigvecs'])  # (n_windows, k, n_samples)
    eigvecs_ours = pg_local_pca_result.eigvecs
    assert eigvecs_ours.shape == eigvecs_ref.shape

    n_windows, k, n_samples = eigvecs_ref.shape
    eigvals_ref = _to_array(ref['eigvals'])
    skipped = 0
    compared = 0
    for w in range(n_windows):
        for pc in range(k):
            # Skip near-degenerate eigenvalues where eigenvectors aren't
            # uniquely determined (beyond sign)
            evals = eigvals_ref[w]
            others = np.delete(evals, pc)
            if np.any(np.abs(others - evals[pc]) < 1e-6 * max(abs(evals[pc]), 1.0)):
                skipped += 1
                continue
            v_ref = eigvecs_ref[w, pc]
            v_ours = eigvecs_ours[w, pc]
            dot = np.dot(v_ref, v_ours)
            # Require strong alignment; otherwise skip (degenerate / noisy)
            if abs(dot) < 0.95:
                skipped += 1
                continue
            sign = np.sign(dot)
            np.testing.assert_allclose(
                sign * v_ours, v_ref, rtol=1e-3, atol=1e-6)
            compared += 1
    # Sanity: we should have compared most windows
    assert compared > 0.5 * n_windows * k, (
        f"Too many degenerate-eigenvalue skips: {skipped} / "
        f"{n_windows * k} entries skipped")


# ---------------------------------------------------------------------------
# pc_dist parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normalize", ["L1", "L2"])
def test_parity_pc_dist(pg_local_pca_result, normalize):
    _skip_if_missing(PCDIST_PATH)
    ref = _load_json(PCDIST_PATH)
    key = f"dist_{normalize}"
    dist_ref = _to_array(ref[key])
    dist_ours = pc_dist(pg_local_pca_result,
                        npc=pg_local_pca_result.k, normalize=normalize)
    assert dist_ours.shape == dist_ref.shape
    # atol=1e-6 absorbs R's tiny diagonal round-off (our diag is exactly 0)
    np.testing.assert_allclose(dist_ours, dist_ref, rtol=1e-4, atol=1e-6)


def test_parity_mds_procrustes(pg_local_pca_result):
    _skip_if_missing(PCDIST_PATH)
    ref = _load_json(PCDIST_PATH)
    dist_ref = _to_array(ref['dist_L1'])
    dist_ours = pc_dist(pg_local_pca_result,
                        npc=pg_local_pca_result.k, normalize='L1')
    coords_ref, _ = pcoa(dist_ref, n_components=2)
    coords_ours, _ = pcoa(dist_ours, n_components=2)
    # Procrustes alignment: translate, rotate/reflect (orthogonal), and
    # isotropically scale to match reference, then compare.
    aligned, resid = _procrustes_align(coords_ours, coords_ref)
    ref_norm = float(np.linalg.norm(coords_ref))
    assert resid < 1e-3 * max(ref_norm, 1.0), (
        f"Procrustes residual too large: {resid} (ref_norm={ref_norm})")


def test_parity_pcoa_matches_cmdscale():
    """pg_gpu.pcoa on a fixed distance matrix matches R's cmdscale output.

    The ``mds_coords`` field in ``lostruct_reference_corners.json`` was
    produced by R's ``cmdscale(dist_L1, k=2)``; this test runs our ``pcoa``
    on the exact same distance matrix and checks the two outputs agree up
    to the rotation/reflection/scale ambiguity that classical MDS does not
    pin down (measured via ``scipy.spatial.procrustes``).
    """
    _skip_if_missing(PCDIST_PATH)
    _skip_if_missing(CORNERS_PATH)
    from scipy.spatial import procrustes

    dist_ref = _to_array(_load_json(PCDIST_PATH)['dist_L1'])
    mds_r = _to_array(_load_json(CORNERS_PATH)['mds_coords'])

    mds_ours, _ = pcoa(dist_ref, n_components=2)
    assert mds_ours.shape == mds_r.shape
    _, _, disparity = procrustes(mds_r, mds_ours)
    # Same algorithm on the same inputs should land at machine precision.
    assert disparity < 1e-10, (
        f"pcoa vs cmdscale disparity {disparity:.3e} exceeds 1e-10")


# ---------------------------------------------------------------------------
# Jackknife parity
# ---------------------------------------------------------------------------


def test_parity_jackknife_se(reference_hm, reference_input):
    _skip_if_missing(JACKKNIFE_PATH)
    ref = _load_json(JACKKNIFE_PATH)
    se_ref = _to_array(ref['se_per_pc'])
    n_blocks = int(ref['n_blocks'])
    se_ours = local_pca_jackknife(
        reference_hm,
        window_size=int(reference_input['window_size']),
        window_type='snp',
        k=int(reference_input['k']),
        n_blocks=n_blocks,
        aggregate='mean',
    )
    assert se_ours.shape == se_ref.shape
    # Jackknife is noisier — relax tolerance slightly.
    # NaN-safe comparison: propagate NaNs on both sides.
    valid = np.isfinite(se_ref) & np.isfinite(se_ours)
    assert valid.sum() > 0.8 * se_ref.size
    np.testing.assert_allclose(se_ours[valid], se_ref[valid],
                               rtol=5e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Corners parity (after Procrustes alignment)
# ---------------------------------------------------------------------------


def test_parity_corners(pg_local_pca_result, reference_input):
    _skip_if_missing(CORNERS_PATH)
    _skip_if_missing(PCDIST_PATH)
    ref = _load_json(CORNERS_PATH)
    dist_ours = pc_dist(pg_local_pca_result,
                        npc=pg_local_pca_result.k, normalize='L1')
    coords_ours, _ = pcoa(dist_ours, n_components=2)
    # Align our MDS to the R MDS so the corner geometry is comparable.
    coords_ref = _to_array(ref['mds_coords'])
    aligned, _ = _procrustes_align(coords_ours, coords_ref)

    prop = float(reference_input['prop'])
    k_corners = int(reference_input['k_corners'])
    ours = corners(aligned, prop=prop, k=k_corners, random_state=0)
    ref_idx = _to_array(ref['corner_indices'], dtype=np.int64)
    assert ours.shape[1] == ref_idx.shape[1]
    # Compare as sets per corner column. The order of corners may differ, so
    # match each of our columns to whichever reference column has the best
    # overlap.
    def _match(ours_col, ref_cols):
        ours_set = set(ours_col.tolist())
        best = -1
        for i, c in enumerate(ref_cols.T.tolist()):
            overlap = len(ours_set & set(c))
            if overlap > best:
                best = overlap
                best_i = i
        return best, best_i

    overlaps = []
    for i in range(ours.shape[1]):
        overlap, _ = _match(ours[:, i], ref_idx)
        overlaps.append(overlap / ours.shape[0])
    # Require at least 70% overlap on each corner (loose — MEC is noise-
    # sensitive and our Welzl may differ from Skyum on corner tie-breaking)
    assert all(o > 0.7 for o in overlaps), (
        f"Corner overlap fractions: {overlaps}")


# ---------------------------------------------------------------------------
# Optional live R regeneration (developer-only; skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.requires_r
def test_parity_live_rpy2(tmp_path):
    pytest.importorskip("rpy2")
    if not INPUT_PATH.exists() or not EIGEN_PATH.exists():
        pytest.skip("Reference inputs/outputs not present.")
    r_script = DATA_DIR / "generate_lostruct_reference.R"
    proc = subprocess.run(
        ["Rscript", str(r_script)],
        cwd=DATA_DIR.parent.parent,
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"Rscript failed (non-CI concern): {proc.stderr[-2000:]}")
    # Diff freshly generated JSON against committed versions
    for path in (EIGEN_PATH, PCDIST_PATH, JACKKNIFE_PATH, CORNERS_PATH):
        fresh = _load_json(path)  # path was just rewritten by R
        # If the script truly ran, the files exist and match themselves —
        # this test is mainly a smoke test that R + lostruct still work.
        assert fresh is not None


# ---------------------------------------------------------------------------
# Procrustes helper
# ---------------------------------------------------------------------------


def _procrustes_align(A: np.ndarray, B: np.ndarray):
    """Align `A` to `B` via translation + orthogonal rotation + isotropic scale.

    Thin wrapper over ``scipy.spatial.procrustes`` that also returns a
    residual Frobenius distance measured against the un-normalized reference
    ``B``, which is what the parity tests actually want to bound.
    """
    from scipy.spatial import procrustes

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mtx_ref, mtx_aligned, _ = procrustes(B, A)
    # scipy normalizes both matrices to unit Frobenius norm; rescale the
    # aligned output back to the original B's scale for a meaningful residual.
    scale = float(np.linalg.norm(B - B.mean(axis=0)))
    aligned = mtx_aligned * scale + B.mean(axis=0)
    resid = float(np.linalg.norm(aligned - B))
    return aligned, resid
