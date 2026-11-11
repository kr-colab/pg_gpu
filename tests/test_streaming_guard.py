"""Every function that cannot stream rejects a streaming matrix up front.

docs/dev/pipeline_map.toml records, for each public function, whether it runs
on a streaming matrix chunk by chunk (``native``) or rejects one
(``rejects``). A rejecting function must raise NotImplementedError that points
to ``.materialize()``, not fail deep inside on a missing attribute.
"""

import importlib

import numpy as np
import pytest

from pg_gpu import HaplotypeMatrix

from .test_pipeline_map import MAP

REJECTS = sorted(name for name, row in MAP["functions"].items()
                 if row.get("streaming") == "rejects")


@pytest.fixture(scope="module")
def stream(tmp_path_factory):
    # The guard rejects any streaming matrix before it looks at the input
    # type, so one haplotype stream serves every function.
    rng = np.random.default_rng(0)
    hm = HaplotypeMatrix(rng.integers(0, 2, (20, 200)).astype(np.int8),
                         np.arange(200) * 50 + 1, 1, 10_000)
    hm.samples = [f"s{i}" for i in range(10)]
    path = str(tmp_path_factory.mktemp("guard") / "s.vcz")
    hm.to_zarr(path, format="vcz", contig_name="1")
    return HaplotypeMatrix.from_zarr(path, streaming="always", chunk_bp=2000)


@pytest.mark.parametrize("name", sorted(MAP["functions"]))
def test_streaming_value_matches_inputs(name):
    # A function that takes a matrix must either stream (and list a
    # streaming input) or reject a stream; only a function without a
    # matrix input is n/a. So a new function cannot skip the guard by
    # being marked n/a or native by mistake.
    row = MAP["functions"][name]
    inputs = row.get("inputs", [])
    takes_matrix = any(i.endswith("Matrix") for i in inputs)
    takes_stream = any(i.startswith("Streaming") for i in inputs)
    expected = ("native" if takes_stream else "rejects") if takes_matrix else "n/a"
    assert row.get("streaming") == expected


@pytest.mark.parametrize("name", REJECTS)
def test_rejecting_function_points_to_materialize(name, stream):
    mod, attr = name.split(".", 1)
    func = getattr(importlib.import_module(f"pg_gpu.{mod}"), attr)
    with pytest.raises(NotImplementedError, match="windowed_analysis.*materialize"):
        func(stream)
