"""The fused engines apply missing_data='exclude' without copying the matrix.

The chunked engine runs when the matrix does not fit on the GPU twice, so a
filtered copy of the complete sites could fail before the first chunk. The
engines keep an index of the kept sites instead and gather through it.
"""

import numpy as np
import cupy as cp
import pytest
from cupy.cuda import memory_hook

from pg_gpu import HaplotypeMatrix
from pg_gpu.windowed_analysis import (
    windowed_statistics_fused, windowed_statistics_fused_chunked)

POPS = {'a': list(range(0, 40)), 'b': list(range(40, 80))}


class _PeakBytes(memory_hook.MemoryHook):
    """Peak of the bytes the CuPy pool hands out while the hook is active."""
    name = 'peak-bytes'

    def __init__(self):
        self.current = self.peak = 0
        self._sizes = {}

    def malloc_postprocess(self, **kw):
        self._sizes[kw['mem_ptr']] = kw['mem_size']
        self.current += kw['mem_size']
        self.peak = max(self.peak, self.current)

    def free_postprocess(self, **kw):
        self.current -= self._sizes.pop(kw['mem_ptr'], 0)


def _data(n_var, seed=0, missing_sites=0.1):
    rng = np.random.default_rng(seed)
    hap = rng.integers(0, 2, (80, n_var)).astype(np.int8)
    cols = rng.choice(n_var, int(n_var * missing_sites), replace=False)
    hap[rng.integers(0, 80, cols.size), cols] = -1
    return hap, np.arange(n_var, dtype=np.int64) * 10 + 1


def _hm(hap, pos):
    return HaplotypeMatrix(cp.asarray(hap), cp.asarray(pos), 1,
                           int(pos[-1]) + 9, sample_sets=POPS)


@pytest.fixture(params=[5_000, 100_000], ids=["small-chunks", "large-chunks"])
def small_blocks(request, monkeypatch):
    """Chunks and row blocks smaller than the test matrix, as they are for a
    matrix that needs the chunked engine. The large chunks are a third of
    the matrix, so a copy of a few chunks shows in the peak too."""
    monkeypatch.setattr("pg_gpu._memutil.estimate_fused_chunk_size",
                        lambda n, memory_fraction=0.35: request.param)
    monkeypatch.setattr("pg_gpu.haplotype_matrix._ROW_BLOCK_BYTES", 2 ** 20)


@pytest.mark.parametrize("stats, kw", [
    (('pi', 'theta_w', 'tajimas_d'), {}),
    (('pi', 'mu_sfs', 'daf_hist', 'het_expected'), {}),
    (('pi',), {'population': 'a'}),
    (('dxy', 'fst', 'da'), {'pop1': 'a', 'pop2': 'b'}),
], ids=["single", "per-site", "one-pop", "two-pop"])
def test_chunked_exclude_peak_is_near_include(small_blocks, stats, kw):
    hap, pos = _data(300_000)
    bins = np.arange(1, 3_000_002, 100_000, dtype=float)

    def peak(missing_data):
        hm = _hm(hap, pos)
        cp.get_default_memory_pool().free_all_blocks()
        with _PeakBytes() as hook:
            windowed_statistics_fused_chunked(
                hm, bins, statistics=stats, missing_data=missing_data, **kw)
        return hook.peak

    extra = peak('exclude') - peak('include')
    # A filtered copy would add about 0.9 of the matrix; the keep index and
    # the kept positions add a few bytes per site.
    assert extra < 0.25 * hap.nbytes


@pytest.mark.parametrize("engine", [windowed_statistics_fused,
                                    windowed_statistics_fused_chunked])
@pytest.mark.parametrize("stats, kw", [
    (('pi', 'theta_w', 'tajimas_d', 'max_daf', 'mu_sfs', 'daf_hist',
      'het_expected', 'garud_h1', 'mean_nsl', 'zns', 'snp_dist_mean'), {}),
    (('pi', 'theta_h', 'mu_sfs', 'het_expected'), {'population': 'b'}),
    (('pi', 'dxy', 'fst', 'fst_wc', 'da', 'mu_sfs'),
     {'population': 'a', 'pop1': 'a', 'pop2': 'b'}),
], ids=["all-rows", "one-pop", "two-pop"])
def test_exclude_with_allele_cap_matches_prefiltered_include(
        small_blocks, engine, stats, kw):
    # 'exclude' plus the allele cap, through the keep index, equals
    # 'include' on a matrix that holds only the kept sites.
    hap, pos = _data(4_000, seed=3)
    hap[5, 100] = 9             # over the kernels' allele capacity
    hap[70, 2_500] = 9
    bins = np.arange(1, 40_002, 4_000, dtype=float)
    with pytest.warns(Warning):
        got = engine(_hm(hap, pos), bins, statistics=stats,
                     missing_data='exclude', **kw)

    rows = slice(None)
    if 'pop1' in kw:
        rows = POPS['a'] + POPS['b']
    elif 'population' in kw:
        rows = POPS[kw['population']]
    cap_rows = slice(None) if 'pop1' in kw else rows
    keep = (~(hap[rows] < 0).any(axis=0)) & (hap[cap_rows].max(axis=0) < 8)
    assert 0 < keep.sum() < hap.shape[1]
    ref = engine(_hm(hap[:, keep], pos[keep]), bins, statistics=stats,
                 missing_data='include', **kw)
    for name, values in ref.items():
        if name == 'chrom':
            continue
        np.testing.assert_allclose(np.asarray(got[name], dtype=float),
                                   np.asarray(values, dtype=float),
                                   rtol=1e-12, equal_nan=True, err_msg=name)


def test_site_has_missing_reads_population_rows_in_blocks(monkeypatch):
    monkeypatch.setattr("pg_gpu.haplotype_matrix._ROW_BLOCK_BYTES", 4096)
    hap, pos = _data(5_000, seed=5)
    hm = _hm(hap, pos)
    for pops, rows in ((['a'], POPS['a']), (['a', 'b'], POPS['a'] + POPS['b']),
                       ([[3, 50, 7]], [3, 50, 7])):
        np.testing.assert_array_equal(
            hm._site_has_missing(pops).get(), (hap[rows] < 0).any(axis=0))
