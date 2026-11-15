"""The SFS functions on a stream, against the eager matrix of the same store.

The store has no variant between 20 kb and 60 kb, so a stream over the whole
contig meets chunks with no variants, and a stream over that gap has no
variant at all. In both cases every function must return an array of its
usual shape: the eager value for the whole contig, zeros for the gap.
"""
import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix, sfs

POPS = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
GAP = "1:25000-55000"

CALLS = {
    "sfs": lambda m, md: sfs.sfs(m, missing_data=md),
    "sfs_population": lambda m, md: sfs.sfs(m, "B", missing_data=md),
    "sfs_folded": lambda m, md: sfs.sfs_folded(m, missing_data=md),
    "sfs_folded_population": lambda m, md: sfs.sfs_folded(m, "B", missing_data=md),
    "sfs_scaled": lambda m, md: sfs.sfs_scaled(m, missing_data=md),
    "sfs_folded_scaled": lambda m, md: sfs.sfs_folded_scaled(m, missing_data=md),
    "sfs_folded_scaled_population": lambda m, md: sfs.sfs_folded_scaled(m, "B", missing_data=md),
    # Three rows: an odd n, which the folded length alone cannot give.
    "sfs_folded_scaled_row_list": lambda m, md: sfs.sfs_folded_scaled(m, [4, 5, 6], missing_data=md),
    "joint_sfs": lambda m, md: sfs.joint_sfs(m, "A", "B", missing_data=md),
    "joint_sfs_folded": lambda m, md: sfs.joint_sfs_folded(m, "A", "B", missing_data=md),
    "joint_sfs_scaled": lambda m, md: sfs.joint_sfs_scaled(m, "A", "B", missing_data=md),
    "joint_sfs_folded_scaled": lambda m, md: sfs.joint_sfs_folded_scaled(m, "A", "B", missing_data=md),
    "project_joint_sfs": lambda m, md: sfs.project_joint_sfs(
        m, "A", "B", target_n1=2, target_n2=2, missing_data=md),
}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    rng = np.random.default_rng(0)
    pos = np.concatenate([np.arange(1, 20_000, 500), np.arange(60_001, 80_000, 500)])
    hap = rng.integers(0, 2, (8, pos.size)).astype(np.int8)
    hap[rng.random(hap.shape) < 0.05] = -1
    hm = HaplotypeMatrix(hap, pos, 1, 80_000)
    hm.samples = [f"s{i}" for i in range(4)]
    path = str(tmp_path_factory.mktemp("sfs") / "gapped.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return path


def _load(store, streaming, region=None):
    m = HaplotypeMatrix.from_zarr(store, region=region, streaming=streaming, chunk_bp=10_000)
    m.sample_sets = POPS
    return m


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("name", CALLS)
def test_stream_matches_eager(store, name, missing_data):
    expected = CALLS[name](_load(store, "never"), missing_data)
    got = CALLS[name](_load(store, "always"), missing_data)
    np.testing.assert_allclose(got, expected)
    assert expected.sum() > 0


@pytest.mark.parametrize("missing_data", ["include", "exclude"])
@pytest.mark.parametrize("name", CALLS)
def test_stream_with_no_variants_gives_zeros(store, name, missing_data):
    full = CALLS[name](_load(store, "always"), missing_data)
    got = CALLS[name](_load(store, "always", region=GAP), missing_data)
    assert isinstance(got, np.ndarray)
    assert got.shape == full.shape and got.dtype == full.dtype
    assert not got.any()
