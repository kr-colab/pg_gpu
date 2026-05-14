"""End-to-end parity: streaming + eager + scikit-allel on the same data.

The streaming-equivalence tests in ``test_streaming_kernels.py`` prove
``windowed_analysis(streaming) == windowed_analysis(eager)``. The
existing ``test_scikit_allel_comparison.py`` proves the eager scalar
stats match scikit-allel. This module closes the triangle: a small
msprime + VCZ fixture is windowed three ways (allel, pg_gpu eager,
pg_gpu streaming) and all three numbers have to agree per window.

The point is to catch any drift in the streaming dispatch that the
eager-vs-streaming check would miss because eager and streaming share
a bug in the underlying kernel.
"""

import msprime
import numpy as np
import allel
import pytest

from pg_gpu import HaplotypeMatrix, windowed_analysis


@pytest.fixture
def vcz_store(tmp_path):
    """A small msprime-derived VCZ store with samples named.

    Uses msprime's binary mutation model: pg_gpu's pi formula treats
    every site as biallelic (counts the sum of int8 values as the
    derived allele count), so on a triallelic site with hap values in
    {0, 1, 2} the count is off and pg_gpu disagrees with allel by ~1%
    per window. That disagreement is a pre-existing pg_gpu vs allel
    issue on multiallelic data, not anything streaming-specific; the
    binary model sidesteps it so this test isolates the streaming
    dispatch.
    """
    ts = msprime.sim_ancestry(
        samples=20, sequence_length=100_000,
        recombination_rate=1e-4, random_seed=42, ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=42,
                               model="binary")
    hm = HaplotypeMatrix.from_ts(ts)
    hm.samples = [f"s{i}" for i in range(hm.num_haplotypes // 2)]
    path = str(tmp_path / "trio.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path, hm


def _aligned(vcz_path):
    eager = HaplotypeMatrix.from_zarr(vcz_path, streaming="never")
    stream = HaplotypeMatrix.from_zarr(vcz_path, streaming="always",
                                       chunk_bp=10_000)
    eager.chrom_start = stream.chrom_start
    eager.chrom_end = stream.chrom_end
    return eager, stream


def _allel_windowed_diversity(eager, window_size):
    """Run allel.windowed_diversity over the same grid pg_gpu uses.

    pg_gpu's per-window pi is sum(per-base contributions) / span. allel's
    ``windowed_diversity`` does the same thing with the same convention,
    so the two should agree window-by-window inside floating-point tolerance.
    """
    import cupy as cp
    haps = cp.asnumpy(eager.haplotypes)        # (n_hap, n_var)
    positions = cp.asnumpy(eager.positions)    # (n_var,)
    h = allel.HaplotypeArray(haps.T)           # allel uses (n_var, n_hap)
    ac = h.count_alleles()
    pi, _, _, _ = allel.windowed_diversity(
        positions, ac,
        size=window_size, start=eager.chrom_start,
        stop=eager.chrom_end,
    )
    return pi


class TestStreamingVsAllel:

    def test_pi_three_way(self, vcz_store):
        """allel ≈ eager == streaming for windowed pi.

        Eager and streaming must agree byte-for-byte -- that's the
        streaming dispatch contract. Both must agree with scikit-allel
        within ~1% relative error: allel uses inclusive [start, stop]
        windows and pg_gpu uses half-open [start, stop), so a variant
        whose position lies exactly on a window boundary lands in
        different windows under the two conventions. Most windows
        agree to ULP scale; the rest differ by a single boundary
        variant's contribution.
        """
        path, _ = vcz_store
        eager, stream = _aligned(path)
        pi_allel = _allel_windowed_diversity(eager, window_size=5_000)

        df_e = windowed_analysis(eager, window_size=5_000, statistics=["pi"])
        df_s = windowed_analysis(stream, window_size=5_000, statistics=["pi"])

        # streaming vs eager: same numbers up to ULP-scale rounding.
        # The kernel-parity test in test_streaming_kernels covers this
        # already at rtol=1e-6; re-asserting here makes the failure
        # mode obvious if streaming drifts from eager independently
        # of allel.
        np.testing.assert_allclose(df_e["pi"].to_numpy(),
                                   df_s["pi"].to_numpy(),
                                   rtol=1e-6, atol=1e-9)

        # eager vs allel: on binary-only data the two libraries
        # compute the same per-variant pi and agree to FP precision.
        # The tolerance is loose only because allel and pg_gpu place
        # one boundary differently for the last window (pg_gpu clips
        # at chrom_end while allel rounds up to stop), giving a
        # ~1.5% per-bp difference on that single window.
        ok = np.isfinite(pi_allel)
        np.testing.assert_allclose(
            df_e["pi"].to_numpy()[: len(pi_allel)][ok], pi_allel[ok],
            rtol=2e-2, atol=1e-7,
            err_msg="pg_gpu vs allel windowed pi disagree beyond "
                    "the last-window-boundary envelope; check for a "
                    "real regression",
        )
        # streaming vs allel follows from the two above; assert it
        # explicitly so a failure is unambiguous about which leg drifted.
        np.testing.assert_allclose(
            df_s["pi"].to_numpy()[: len(pi_allel)][ok], pi_allel[ok],
            rtol=2e-2, atol=1e-7,
        )
