# Streaming + randomized local-PCA: design and TDD plan

## Status quo

`pg_gpu.decomposition.local_pca` (and the `lostruct` wrapper around it) materializes one (n_hap × n_hap) sample-sample Gram matrix per genomic window, accumulates the full list on GPU, then calls `cp.linalg.eigh` on the stacked tensor. Two costs explode together:

- **Memory.** Peak GPU memory is `2 × n_windows × n_hap² × 8` bytes (the Gram list and its `cp.stack` copy coexist), so 100 kb / 50 kb windows on 53 Mb of Ag1000G 3R at 2940 haplotypes need ~140 GB before stacking finishes -- well over the 80 GB on a single A100.
- **Compute.** Full `cp.linalg.eigh` on each (n_hap × n_hap) Gram is `O(n_hap³)`, even though we only ever read the top `k=2-4` eigenpairs out the other side.

We need an algorithm that (a) does not require all Grams resident at once, and (b) only computes the leading `k` eigenpairs. Both axes can be improved together.

## Algorithm

### Two complementary changes

**1. Replace per-window full eigh with randomized SVD on the centred haplotype block.**

For window `w` with centred haplotype block `X_w ∈ ℝ^{n_hap × n_var_w}`, the eigenpairs of `C_w = X_w X_wᵀ / (n_var_w − 1)` are exactly the squared singular values and left singular vectors of `X_w`, scaled by `1/(n_var_w − 1)`. Randomized SVD with oversampled rank `r = k + p` (Halko, Martinsson & Tropp 2011) computes the leading `k` left singular vectors in:

| Step | Cost | Notes |
|---|---|---|
| `Ω ∈ ℝ^{n_var_w × r}` Gaussian | `n_var_w × r` flops | shared across windows |
| `Y = X_w Ω` | `n_hap × n_var_w × r` flops | thin GEMM, output `(n_hap × r)` |
| `Q = qr(Y)` (thin QR) | `n_hap × r²` flops | cuSOLVER `geqrf+orgqr` on a thin matrix |
| `B = Qᵀ X_w` | `r × n_hap × n_var_w` flops | thin GEMM, output `(r × n_var_w)` |
| `(σ, V_k) = svd(B)` | `r × n_var_w × r` flops | small SVD on `r × n_var_w`, take top-k |
| `U_k = Q V_k` | `n_hap × r × k` flops | tiny GEMM |
| eigvals = `σ² / (n_var_w − 1)` | trivial | |

Total per window: roughly `2 × n_hap × n_var_w × r` flops. For n_hap=2940, n_var_w≈50000, r=10 (k=2 + p=8) this is ~3 G-flops, vs. `n_hap × n_var_w² + n_hap³ ≈ 432 G-flops` for the current Gram + full eigh path. **Asymptotic per-window speedup ≈ 100×**, larger when `n_hap` grows.

Randomized SVD with `p ≥ 5` oversampling and one or two power iterations is essentially exact to working precision for a low-rank target on a positive-semidefinite matrix; lostruct only ever uses `k = 2-4`, so this regime is precisely where randomization is cheapest and tightest. We will follow the standard Halko-Martinsson-Tropp recipe (random projection + 1-2 subspace iterations + thin QR + small SVD).

**2. Stream windows in tiles, never holding all Gram-equivalent results at once.**

Even with randomized SVD eliminating the Gram, we still want to keep the per-tile working set bounded. The tile loop becomes:

```
for tile in chunks(windows, tile_size):
    # Build small per-window outputs only (no full Gram)
    eigvals_tile, eigvecs_tile, sumsq_tile = batch_window_rsvd(matrix, tile, k, oversample, rng)
    # Eigvals, eigvecs, sumsq are tiny -- copy to host
    eigvals_host[tile.start:tile.end] = eigvals_tile.get()
    eigvecs_host[tile.start:tile.end] = eigvecs_tile.get()
    sumsq_host[tile.start:tile.end] = sumsq_tile.get()
    # Free per-tile buffers before next tile
```

Persistent GPU memory: just the haplotype matrix (~32 GB for 3R). Per-tile transient memory scales with `tile_size × max(n_hap, max_n_var_per_window) × r × 8` bytes. For the Ag1000G full-arm case at tile_size=128 this is < 1 GB, giving ~78 GB of headroom. The same code path scales effortlessly to k=10, dense panels, or full-genome runs.

**Optional batched optimisation.** Within a tile we can stack the per-window centred blocks into a 3-D `(tile_size, n_hap, n_var_max)` tensor (with zero-padding for short windows) and use `cublas` batched GEMMs for the projection, QR, and reconstruction steps. This trades a small memory increase for ~5-10× higher GEMM throughput on small problems where launch overhead dominates. Implement only if profiling shows kernel-launch overhead is significant.

### `sumsq` and the lostruct distance metric

Lostruct's `LocalPCAResult` exposes `sumsq[w] = ‖C_w‖_F²` per window. By the singular-value identity, `‖C_w‖_F² = sum_i σ_i⁴ / (n_var_w − 1)²` where `σ_i` are all singular values of the centred block `X_w` -- so `sumsq` cannot be recovered from `‖X_w‖_F² = sum σ_i²` alone. Only `pc_dist` consumes the rest of `LocalPCAResult` directly; the parity tests against R lostruct compare `sumsq` against the R reference at machine precision and `pc_dist` ignores the value entirely (it parses sumsq from the flat lostruct matrix layout but never uses it in the distance math). So `sumsq` is part of the public contract but not used internally.

This forces a two-tier design:

- **Tier A: streaming + per-window dense eigh.** Form `C_w` transiently per window (~70 MB float64 at 2940 hap), compute `sumsq = (C_w**2).sum()` from it (one reduction), do `eigh(C_w)`, take top-k, discard `C_w` before the next window. Memory bound: `tile_size × n_hap² × 8` bytes peak. Per-window compute identical to today (`O(n_hap² × n_var_w)` Gram + `O(n_hap³)` eigh), and `sumsq` is bit-identical to the legacy path. This solves the OOM and is the default for parity correctness.
- **Tier B: streaming + per-window randomized SVD on `X_w`.** Skip the Gram entirely. Compute top-k via the projection-and-thin-SVD recipe above (`O(n_hap × n_var_w × (k+p))` per window, ≈100× faster than tier A's per-window cost in the Ag1000G regime). Compute `sumsq ≈ sum_{i=1}^{k+p} σ_i⁴ / d²` using the singular values returned by the randomized step. With `oversample = max(20, 2k)` the sum already covers the dominant contribution; for ranks beyond the oversample budget we add an upper-bound term `(σ_{k+p+1})² × (‖X_w‖_F² − sum_{i≤k+p} σ_i²) / d²` to bracket the tail. This tier is opt-in via `engine='streaming-rsvd'` and tolerated by the parity tests under a `rtol≈5e-3` budget; it is recommended for whole-genome scans where exact `sumsq` is not load-bearing.

The two tiers share the same outer streaming loop and per-tile dispatcher; only the per-window kernel differs.

### Numerical considerations

- One subspace iteration (`Y ← X_w X_wᵀ Y`, then orthonormalise) is recommended for the Ag1000G regime where the spectrum decays slowly; we'll make this configurable via `n_subspace_iter` defaulting to `1`. With one iteration and `oversample = 8`, expected eigenvalue relative error on simulated coalescent data is below 1e-4 -- well inside the noise floor of the downstream MDS embedding.
- Randomized SVD eigenvectors are determined up to sign, like `eigh`. Procrustes alignment in the parity tests already absorbs sign and orthogonal rotation ambiguities, so no test changes are needed for that axis.
- Edge case: when `n_var_w < k + p`, fall back to deterministic eigh on the (small) Gram for that single window. This is identical in cost to the current path for tiny windows.

## Implementation roadmap

We work in seven small steps, each with its own test artefact landing first (TDD).

### Step 0 -- branch and scaffold
- [x] Branch `feat/local-pca-streaming-randomized` off `main`.
- [ ] Add this plan as `docs/local_pca_streaming_plan.md` (this file).

### Step 1 -- property tests for the new randomized kernel
*Goal: lock down the contract `randomized_pca_window` must satisfy before writing it.*

Add `tests/test_local_pca_streaming.py` with:

- `test_rsvd_window_eigvals_match_exact_eigh` -- on a small (50 hap × 200 var) random PSD block, randomized eigvals match exact `cp.linalg.eigh` top-k within `rtol=1e-3` for `k=2, oversample=8, n_iter=1`.
- `test_rsvd_window_subspace_alignment` -- subspace projection `‖U_kᵀ U_exact_k U_exact_kᵀ U_k − I‖_F` < 1e-3 (sign- and rotation-invariant).
- `test_rsvd_window_recovers_planted_signal` -- generate a low-rank-plus-noise block with planted top-2 directions; randomized recovers them.
- `test_rsvd_window_handles_thin_window` -- when `n_var_w < k + oversample`, falls back to exact eigh and matches.
- `test_rsvd_window_sumsq_matches_frobenius` -- returned `sumsq` equals `(X_w @ X_w.T).trace()²`-style Frobenius squared, computed two ways.

These tests should *fail* against an unwritten `_randomized_pca_window` helper.

### Step 2 -- implement `_randomized_pca_window`
- New private helper in `pg_gpu/decomposition.py` taking a centred GPU array `X_w`, returning `(eigvals (k,), eigvecs (k, n_hap), sumsq scalar)`.
- Internal contract: never form `X_w X_wᵀ`; pure GEMM/QR/SVD path.
- Reuses a caller-supplied `cp.random.RandomState` so all windows share an RNG.
- Make Step 1 tests pass.

### Step 3 -- streaming dispatcher
*Goal: cleanly tile windows and route through the new kernel without mutating the public `LocalPCAResult` / `LostructResult` contract.*

Add tests in `test_local_pca_streaming.py`:

- `test_streaming_matches_existing_local_pca_eigvals` -- on `structured_hm` (see existing fixture): `local_pca(..., engine='streaming-rsvd')` produces eigvals equal to the existing path within `rtol=1e-3`.
- `test_streaming_matches_existing_local_pca_subspace` -- per-window subspace alignment within tolerance (Procrustes helper already exists in `test_local_pca_parity.py`).
- `test_streaming_tile_size_invariance` -- `tile_size=8` and `tile_size=64` produce identical outputs.
- `test_streaming_pc_dist_matches` -- downstream `pc_dist(streaming_result)` matches `pc_dist(existing_result)` to within `1e-2` Frobenius (rotation-invariant; lostruct's distance is sign-symmetric).
- `test_streaming_lostruct_corners_recover_sweep` -- on the `examples/local_pca.py` simulation (sweep at midpoint), corners identified via streaming engine include the sweep-region windows.

Implement:
- `_streaming_local_pca(matrix, window_iter, k, tile_size, oversample, n_iter, rng)`.
- Memory-aware default `tile_size` from `cp.cuda.Device().mem_info` plus `n_hap`.
- Wire `engine='streaming-rsvd'` (default) and `engine='dense-eigh'` (legacy) parameters into `local_pca` and `lostruct`. Default switches to streaming-rsvd; existing tests must pass with the default.

### Step 4 -- promote streaming as the default and verify parity tests still pass
- Run the full `tests/test_local_pca.py` suite; existing tests should pass with the new default.
- Run `tests/test_local_pca_parity.py` against the R reference data: eigvals within `rtol=5e-3`, subspaces aligned under Procrustes, pc_dist Frobenius < 1e-2, MDS embedding aligned.
- Investigate any tolerance breaches before tightening tolerances permanently.

### Step 5 -- OOM smoke at Ag1000G scale (integration test)
*Goal: the original failure mode never returns.*

Add `tests/integration/test_local_pca_full_arm_smoke.py` (gated behind `pytest.mark.slow` and a `PG_GPU_SMOKE_DATA` env var pointing at a local Ag1000G zarr):

- `test_lostruct_full_3r_no_oom` -- run `lostruct(hm, window_size=100_000, step_size=50_000, k=2)` on a full chromosome arm; assert no OOM, `n_windows > 1000`, MDS shape correct.
- Capture `cp.cuda.Device().mem_info` peak via the cupy memory-pool snapshot API and assert peak < 50 GB.

### Step 6 -- benchmark script and microbench
- Add `debug/bench_local_pca_streaming.py` running both engines on a fixed simulated dataset and reporting per-window time + peak memory for `k ∈ {2, 4, 8}`, `n_hap ∈ {500, 1500, 2940}`, `n_windows ∈ {100, 500, 2000}`.
- Target: streaming-rsvd is at least 10× faster than dense-eigh at `n_hap ≥ 1500` and uses ≤ 5% of dense-eigh peak memory at the same configuration.

### Step 7 -- documentation and release notes
- Update `decomposition.py` docstrings to describe the engine choice.
- Add a one-paragraph note to `docs/source/` (RTD docs) describing the new engine, default, and when to fall back to `dense-eigh`.
- Note the change in CHANGELOG (or equivalent).

## Test taxonomy summary

| Layer | Test file | Asserts |
|---|---|---|
| Unit (kernel) | `test_local_pca_streaming.py::test_rsvd_window_*` | randomized SVD on a single block matches exact eigh, with sign and rotation tolerated |
| Unit (dispatcher) | `test_local_pca_streaming.py::test_streaming_*` | tile-size invariance, eigval/subspace parity vs existing path |
| Pipeline | `test_local_pca_streaming.py::test_streaming_lostruct_*` | downstream `pc_dist`, `pcoa`, `corners` unchanged in lostruct semantics |
| Regression vs R | existing `test_local_pca_parity.py` | streaming engine produces R-compatible outputs under Procrustes |
| Integration | `test_local_pca_full_arm_smoke.py` | full-arm Ag1000G runs without OOM, peak memory bounded |
| Benchmark | `debug/bench_local_pca_streaming.py` | streaming ≥ 10× faster, ≤ 5% peak memory at scale |

Each step lands a test artefact *first*; only after the test is committed and (intentionally) failing do we write the implementation against it.

## Risks and mitigations

- **Randomization tolerance.** Mitigated by oversample=8 default + 1 subspace iteration + tolerance budget calibrated against the Procrustes-aligned R parity tests; if any single test breaks at the chosen tolerance we investigate before relaxing.
- **`sumsq` semantic drift.** Mitigated by computing it from the centred block directly (exact, deterministic), not from the truncated factorisation.
- **API churn.** The `engine` keyword is additive and defaults to streaming-rsvd; the legacy path remains reachable via `engine='dense-eigh'` for any caller that needs bit-identical outputs to the old path.
- **Numerical drift in `pc_dist` / `corners`.** Lostruct's downstream geometry is invariant to per-window rotations and signs by construction; tested explicitly via the streaming parity tests.

## Definition of done

1. All seven steps complete; new test file is green; parity suite still green.
2. `examples/local_pca.py` simulation runs with the streaming engine and produces a figure visually indistinguishable from the legacy run (corners on the sweep region, MDS partition into neutral / linked / sweep).
3. Full Ag1000G 3R lostruct at 100 kb / 50 kb runs in < 5 minutes on a single A100 with peak memory < 50 GB.
4. Benchmark script reports ≥ 10× per-window speedup over dense-eigh at n_hap=2940, k=2.
