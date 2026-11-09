"""Per-window denominators when the matrix has n_total_sites.

With n_total_sites and no accessible mask, every windowed engine divides a
window by its share of n_total_sites, in proportion to its length. A VCF or
store loaded with include_invariant=True gets a mask of its record
positions instead, so each window divides by its own called bases.
"""

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from pg_gpu import GenotypeMatrix, HaplotypeMatrix, diversity
from pg_gpu.windowed_analysis import (WindowedAnalyzer, windowed_analysis,
                                     windowed_statistics_fused_chunked)

N_TOTAL = 800
POPS = {"a": list(range(10)), "b": list(range(10, 20))}


def _hm():
    rng = np.random.default_rng(1)
    pos = np.sort(rng.choice(np.arange(1, 2001), 60, replace=False))
    hap = rng.integers(0, 2, (20, 60)).astype(np.int8)
    return HaplotypeMatrix(hap, pos, 1, 2000, n_total_sites=N_TOTAL,
                           sample_sets=POPS)


def _denominators(stats, stat="pi", **kw):
    raw = windowed_analysis(_hm(), statistics=[stat], span_normalize=False, **kw)[stat]
    return (raw / windowed_analysis(_hm(), statistics=stats, **kw)[stat]).to_numpy()


@pytest.mark.parametrize("stats", [["pi"], ["pi", "mu_var"], ["pi", "n_singletons"]],
                         ids=["scatter", "fused", "fallback"])
def test_bp_windows_share_n_total_sites(stats):
    # Four 500-bp windows tile the 2000-bp region: each gets a quarter.
    np.testing.assert_allclose(_denominators(stats, window_size=500), N_TOTAL / 4)


def test_chunked_engine_shares_n_total_sites(monkeypatch):
    monkeypatch.setattr("pg_gpu._memutil.estimate_fused_chunk_size",
                        lambda n, memory_fraction=0.35: 7)
    hm = _hm()
    bins = np.array([1, 501, 1001, 1501, 2001], dtype=float)
    raw = windowed_statistics_fused_chunked(hm, bins, statistics=("pi",), per_base=False)
    norm = windowed_statistics_fused_chunked(hm, bins, statistics=("pi",), per_base=True)
    np.testing.assert_allclose(np.asarray(raw["pi"]) / np.asarray(norm["pi"]), N_TOTAL / 4)


@pytest.mark.parametrize("stats", [["pi"], ["pi", "n_singletons"]], ids=["fused", "fallback"])
def test_region_windows_share_n_total_sites(stats):
    regions = pd.DataFrame({"start": [1, 1001], "end": [501, 2001]})
    den = _denominators(stats, window_type="regions", regions=regions)
    np.testing.assert_allclose(den, [N_TOTAL * 500 / 2000, N_TOTAL * 1000 / 2000])


@pytest.mark.parametrize("stats", [["dxy"], ["dxy", "mu_var"], ["dxy", "n_singletons"]],
                         ids=["scatter", "fused", "fallback"])
def test_two_pop_windows_share_n_total_sites(stats):
    den = _denominators(stats, stat="dxy", window_size=500, populations=["a", "b"])
    np.testing.assert_allclose(den, N_TOTAL / 4)


def test_snp_windows_share_n_total_sites():
    # A snp window spans its first to its last variant, both inclusive.
    hm = _hm()
    pos = cp.asnumpy(hm.positions)
    df = WindowedAnalyzer(window_type="snp", window_size=20, step_size=20,
                          statistics=["pi"], progress_bar=False).compute(hm)
    for i, row in enumerate(df.itertuples()):
        sub = hm.get_subset(np.arange(20 * i, 20 * i + 20))
        share = N_TOTAL * (pos[20 * i + 19] - pos[20 * i] + 1) / 2000
        assert row.pi == pytest.approx(diversity.pi(sub, span_normalize=False) / share)


# ── include_invariant: the record positions become the mask ────────────

# Records at 1..30 except 11..15 (filtered out of the VCF); 4 and 20 are
# variable, the rest invariant. 10 is a second record at a split
# multiallelic site, so 25 records hold 24 distinct positions.
_SITES = [p for p in range(1, 31) if not 11 <= p <= 15]


def _write_vcf(path):
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=1,length=100>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2"]
    for p in _SITES:
        alt, gts = (("T", ["0|1", "1|1", "0|0"]) if p in (4, 20)
                    else (".", ["0|0", "0|0", "0|0"]))
        lines.append(f"1\t{p}\t.\tA\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(gts))
        if p == 10:
            lines.append("1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t1|0\t0|0\t0|0")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("cls", [HaplotypeMatrix, GenotypeMatrix])
def test_include_invariant_masks_the_record_positions(tmp_path, cls):
    m = cls.from_vcf(_write_vcf(tmp_path / "inv.vcf"), include_invariant=True)
    assert m.has_accessible_mask
    assert m.n_total_sites == len(_SITES)          # distinct positions
    assert m.accessible_mask.is_accessible_at([10, 11, 15, 16]).tolist() == \
        [True, False, False, True]


def test_include_invariant_windows_divide_by_called_bases(tmp_path):
    hm = HaplotypeMatrix.from_vcf(_write_vcf(tmp_path / "inv.vcf"), include_invariant=True)
    df = windowed_analysis(hm, window_size=10, statistics=["pi"])
    raw = windowed_analysis(hm, window_size=10, statistics=["pi"], span_normalize=False)
    # Windows [1, 11), [11, 21), [21, 31): 10, 5 and 10 called bases.
    np.testing.assert_allclose(df["pi"], raw["pi"] / [10, 5, 10])
    assert diversity.pi(hm) == pytest.approx(raw["pi"].sum() / len(_SITES))


@pytest.mark.parametrize("cls", [HaplotypeMatrix, GenotypeMatrix])
def test_include_invariant_with_bed_counts_records_in_the_bed(tmp_path, cls):
    # BED 0-based [5, 20) is 1-based 6..20; the called bases there are
    # 6..10 and 16..20.
    bed = tmp_path / "acc.bed"
    bed.write_text("1\t5\t20\n")
    m = cls.from_vcf(_write_vcf(tmp_path / "inv.vcf"), include_invariant=True,
                     accessible_bed=str(bed))
    assert m.n_total_sites == 10


def test_include_invariant_genotype_from_zarr(tmp_path):
    hm = HaplotypeMatrix.from_vcf(_write_vcf(tmp_path / "inv.vcf"))
    store = str(tmp_path / "inv.vcz")
    hm.to_zarr(store, format="vcz", contig_name="1")
    gm = GenotypeMatrix.from_zarr(store, include_invariant=True)
    assert gm.has_accessible_mask
    assert gm.n_total_sites == len(_SITES)


def test_range_subset_before_the_first_record_keeps_the_mask_aligned(tmp_path):
    # The mask starts at the first record (1); a range from 0 begins before
    # it, and the subset must still see the same called bases.
    hm = HaplotypeMatrix.from_vcf(_write_vcf(tmp_path / "inv.vcf"), include_invariant=True)
    sub = hm.get_subset_from_range(0, 21)
    assert sub.accessible_mask.is_accessible_at([10, 11, 16, 20]).tolist() == \
        [True, False, True, True]
    assert sub.n_total_sites == 15                  # 1..10 and 16..20
