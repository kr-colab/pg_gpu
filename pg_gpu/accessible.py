"""Accessible site mask for genome-wide statistics normalization.

Provides BED file parsing and an AccessibleMask class that wraps a dense
boolean array with offset-aware, O(1) range queries via prefix sums.
"""

import warnings

import cupy as cp
import numpy as np

from ._warnings import _PKG_PREFIXES, ContigNameWarning, _format_sample_list
from .zarr_io import parse_region


class AccessibleMask:
    """Dense boolean mask over genomic coordinates with fast range queries.

    Parameters
    ----------
    mask : array_like
        Boolean array where True indicates an accessible/callable site.
    offset : int
        1-based genomic coordinate corresponding to mask[0]. For example,
        if offset=1000, mask[0] represents 1-based position 1000.
        Positions passed to is_accessible_at/count_accessible are interpreted
        as 1-based (matching VCF conventions). BED intervals consumed by
        bed_to_mask remain 0-based half-open per BED standard.
    """

    def __init__(self, mask, offset=0):
        self.mask = np.asarray(mask, dtype=bool)
        self.offset = int(offset)
        self._cumsum = None

    @property
    def cumsum(self):
        """Lazy prefix-sum array for O(1) range queries."""
        if self._cumsum is None:
            self._cumsum = np.empty(len(self.mask) + 1, dtype=np.int64)
            self._cumsum[0] = 0
            np.cumsum(self.mask, out=self._cumsum[1:])
        return self._cumsum

    def count_accessible(self, start, end):
        """Count accessible bases in [start, end).

        Parameters
        ----------
        start, end : int
            Genomic coordinates (half-open interval).

        Returns
        -------
        int
            Number of accessible (True) positions in the range.
        """
        i = max(0, start - self.offset)
        j = min(len(self.mask), end - self.offset)
        if j <= i:
            return 0
        return int(self.cumsum[j] - self.cumsum[i])

    @property
    def total_accessible(self):
        """Total number of accessible sites in the mask."""
        return int(self.cumsum[-1])

    def slice(self, start, end):
        """Return an AccessibleMask for the genomic sub-range [start, end).

        Parameters
        ----------
        start, end : int
            Genomic coordinates (half-open interval).

        Returns
        -------
        AccessibleMask
            New mask covering the requested range.
        """
        i = max(0, start - self.offset)
        j = min(len(self.mask), end - self.offset)
        if j <= i:
            return AccessibleMask(np.array([], dtype=bool), offset=start)
        # The kept bases start at offset + i, which is past start when the
        # range begins before this mask does.
        return AccessibleMask(self.mask[i:j], offset=self.offset + i)

    def count_accessible_windows(self, starts, ends):
        """Count accessible bases for multiple windows at once.

        Parameters
        ----------
        starts, ends : array_like, int
            Window boundaries (half-open intervals).

        Returns
        -------
        numpy.ndarray, float64
            Accessible base count per window.
        """
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        i = np.clip(starts - self.offset, 0, len(self.mask))
        j = np.clip(ends - self.offset, 0, len(self.mask))
        j = np.maximum(j, i)
        cs = self.cumsum
        return (cs[j] - cs[i]).astype(np.float64)

    def is_accessible_at(self, positions):
        """Check which variant positions fall in accessible regions.

        Parameters
        ----------
        positions : array_like, int
            Genomic positions to check.

        Returns
        -------
        numpy.ndarray, bool
            True for positions that are accessible.
        """
        positions = np.asarray(positions, dtype=np.int64)
        idx = positions - self.offset
        in_range = (idx >= 0) & (idx < len(self.mask))
        result = np.zeros(len(positions), dtype=bool)
        result[in_range] = self.mask[idx[in_range]]
        return result

    def __len__(self):
        return len(self.mask)

    def __repr__(self):
        return (f"AccessibleMask(length={len(self.mask)}, "
                f"offset={self.offset}, "
                f"accessible={self.total_accessible})")


def parse_bed(path, chrom=None):
    """Parse a BED file into a list of (chrom, start, end) tuples.

    Parameters
    ----------
    path : str or path-like
        Path to a BED file (tab-delimited: chrom, start, end, ...).
        Lines starting with '#', 'track', or 'browser' are skipped.
    chrom : str, optional
        If provided, only return intervals on this chromosome. A name the
        BED lacks matches a BED contig by name without a leading 'chr' and
        case ('1' and 'chr1'), with a ContigNameWarning.

    Returns
    -------
    list of (str, int, int)
        Parsed intervals as (chromosome, start, end) with 0-based half-open
        coordinates.
    """
    return _read_bed(path, chrom)[0]


def _contig_key(name):
    """A contig name without a leading 'chr' and without case, so the UCSC
    name ('chr1') and the Ensembl or NCBI name ('1') of a contig match."""
    key = str(name).lower()
    return key[3:] if key.startswith('chr') else key


def _read_bed(path, chrom=None):
    """``(intervals, contigs)`` from one pass over a BED file: the intervals
    on ``chrom`` (every interval when it is None), and every contig the BED
    names, in file order. ``chrom`` matches by name as ``parse_bed`` says."""
    key = None if chrom is None else _contig_key(chrom)
    contigs = {}
    kept = []
    last, keep = None, False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') \
                    or line.startswith('browser'):
                continue
            fields = line.split('\t')
            if len(fields) < 3:
                fields = line.split()
            if len(fields) < 3:
                continue
            c = fields[0]
            # BED lines come grouped by contig, so the name test runs once
            # per contig, not once per line.
            if c != last:
                last = c
                contigs[c] = None
                keep = key is None or _contig_key(c) == key
            if keep:
                kept.append((c, int(fields[1]), int(fields[2])))
    if chrom is not None:
        chrom = str(chrom)
        names = list(dict.fromkeys(c for c, _, _ in kept))
        if chrom not in names and len(names) == 1:
            warnings.warn(
                f"BED contig {names[0]!r} does not match data contig "
                f"{chrom!r} exactly; inferring that they are the same contig.",
                ContigNameWarning, skip_file_prefixes=_PKG_PREFIXES)
            chrom = names[0]
        kept = [iv for iv in kept if iv[0] == chrom]
    return kept, list(contigs)


def _show_contigs(contigs):
    return _format_sample_list(contigs, range(len(contigs)), limit=5)


def _data_contig(contigs, source, remedy):
    """The contig of data whose variants lie on ``contigs``, or None when the
    data name none. An accessible mask covers one contig, so data on several
    raise ValueError with ``remedy``: one contig's intervals applied to all
    of them would be wrong."""
    if len(contigs) > 1:
        raise ValueError(
            f"{source} has variants on {len(contigs)} contigs "
            f"({_show_contigs(contigs)}), but an accessible mask covers one "
            f"contig; {remedy}.")
    return contigs[0] if contigs else None


def _contig_intervals(path, chrom):
    """``_read_bed(path, chrom)``, but a BED with more than one contig raises
    ValueError when ``chrom`` is None: merging them would count other
    contigs' intervals as accessible bases of the data."""
    intervals, contigs = _read_bed(path, chrom)
    if chrom is None and len(contigs) > 1:
        raise ValueError(
            f"BED file {path} has intervals on {len(contigs)} contigs "
            f"({_show_contigs(contigs)}); "
            "name the contig of the data with chrom= (set_accessible_mask, "
            "windowed_analysis, from_ts) or region= (from_zarr, from_vcf).")
    return intervals, contigs


def bed_to_mask(path, chrom, length, offset=0):
    """Convert BED intervals into a dense boolean AccessibleMask.

    Parameters
    ----------
    path : str or path-like
        Path to a BED file defining accessible regions.
    chrom : str or None
        Chromosome name to extract from the BED file. None is allowed only
        for a BED with one contig.
    length : int
        Total length of the mask array (number of genomic positions to cover).
    offset : int
        1-based genomic coordinate of mask[0].

    Returns
    -------
    AccessibleMask
        Boolean mask where True = accessible. Positions outside any BED
        interval are False.

    Notes
    -----
    BED intervals are 0-based half-open: an interval (s, e) covers 1-based
    positions s+1 through e (inclusive). Mask index k represents 1-based
    position offset + k.
    """
    return _intervals_to_mask(_contig_intervals(path, chrom)[0], length, offset)


def _intervals_to_mask(intervals, length, offset):
    """AccessibleMask of ``length`` bases from ``offset`` with ``intervals``
    (0-based half-open BED records) set True."""
    mask = np.zeros(length, dtype=bool)
    for _, s, e in intervals:
        # BED (s, e) [0-based half-open] = 1-based positions s+1..e
        # mask[k] = True for k where offset + k in [s+1, e]
        i = max(0, s + 1 - offset)
        j = min(length, e + 1 - offset)
        if j > i:
            mask[i:j] = True
    return AccessibleMask(mask, offset=offset)


def _as_accessible_mask(is_accessible):
    """The ``is_accessible`` argument of a statistic, as an AccessibleMask.

    A bool array follows scikit-allel: element 0 is position 1. An
    AccessibleMask keeps its own offset.
    """
    if isinstance(is_accessible, AccessibleMask):
        return is_accessible
    return AccessibleMask(is_accessible, offset=1)


def _called_sites_mask(positions, chrom_start, chrom_end, accessible_bed=None,
                       chrom=None):
    """AccessibleMask whose accessible bases are the given record positions.

    A VCF or store loaded with its invariant sites has one record for each
    called base, so the record positions are the callable bases. A mask of
    them gives every window its exact count of called bases, which a single
    total for the whole region cannot.

    Parameters
    ----------
    positions : array_like, int
        1-based positions of every loaded record, invariant sites included.
    chrom_start, chrom_end : int
        Inclusive bounds of the mask. Positions outside them are ignored.
    accessible_bed : str, optional
        BED file of the same data. Only positions inside it count.
    chrom : str, optional
        Contig to read from ``accessible_bed``; see
        ``resolve_accessible_mask``.

    Returns
    -------
    AccessibleMask
    """
    start, end = int(chrom_start), int(chrom_end)
    positions = np.asarray(positions, dtype=np.int64)
    positions = positions[(positions >= start) & (positions <= end)]
    if accessible_bed is not None:
        bed_mask = resolve_accessible_mask(accessible_bed, start, end, chrom)
        positions = positions[bed_mask.is_accessible_at(positions)]
    mask = np.zeros(end - start + 1, dtype=bool)
    mask[positions - start] = True
    return AccessibleMask(mask, offset=start)


def resolve_accessible_mask(mask_or_path, chrom_start, chrom_end, chrom=None):
    """Resolve a mask input to an AccessibleMask instance.

    Parameters
    ----------
    mask_or_path : str, path-like, numpy.ndarray, or AccessibleMask
        BED file path, boolean array, or AccessibleMask instance.
    chrom_start, chrom_end : int or None
        Matrix chromosome bounds (1-based inclusive). Used to widen the
        mask range so it covers both the matrix and the BED. This avoids
        silently dropping BED accessible bases that fall outside the
        matrix's variant range (relevant for variants-only VCFs).
    chrom : str, optional
        Chromosome name. Required for a BED file with more than one contig;
        a one-contig BED needs none.

    Returns
    -------
    AccessibleMask
    """
    if isinstance(mask_or_path, AccessibleMask):
        return mask_or_path
    if isinstance(mask_or_path, np.ndarray):
        offset = chrom_start if chrom_start is not None else 0
        return AccessibleMask(mask_or_path, offset=offset)
    # Treat as file path. Build the mask over the union of [chrom_start,
    # chrom_end] and the BED's own extent so no accessible bases are lost
    # when the matrix's variant range is narrower than the BED.
    intervals, contigs = _contig_intervals(mask_or_path, chrom)
    if not intervals:
        where = "" if chrom is None else f" for contig {str(chrom)!r}"
        raise ValueError(
            f"No BED intervals found{where} in {mask_or_path}; its contigs "
            f"are: {_show_contigs(contigs) or 'none'}.")
    # 1-based inclusive bounds of the BED (BED is 0-based half-open: a record
    # (s, e) covers 1-based positions s+1..e).
    bed_min = min(s + 1 for _, s, _ in intervals)
    bed_max = max(e for _, _, e in intervals)
    matrix_min = chrom_start if chrom_start is not None else bed_min
    matrix_max = chrom_end if chrom_end is not None else bed_max
    offset = min(bed_min, matrix_min)
    end = max(bed_max, matrix_max)
    length = end - offset + 1
    if length <= 0:
        raise ValueError(
            "Could not determine mask range from BED and chromosome bounds")
    return _intervals_to_mask(intervals, length, offset)


def slice_fields(fields, keep_idx, accessible_idx=None):
    """Slice every field on the variant axis by keep_idx.

    Fields are host numpy arrays regardless of matrix device, so the index
    is moved to host first. keep_idx is relative to the masked view; pass
    accessible_idx to translate it to the underlying array. Pass None when
    keep_idx already indexes the underlying array, as filter() does.
    """
    if not fields:
        return {}
    keep_idx = cp.asnumpy(keep_idx)
    if accessible_idx is not None:
        keep_idx = cp.asnumpy(accessible_idx)[keep_idx]
    return {tag: arr[keep_idx] for tag, arr in fields.items()}


def resolve_streaming_accessible_mask(accessible_bed, source, region=None,
                                      include_invariant=False):
    """Resolve an accessible BED over a streaming source's grid bounds.

    Both streaming loaders (haplotype and genotype) resolve the same mask
    once over the source's position span and hand it to the streaming
    matrix, so per-chunk variant filtering -- and, for span-normalized
    reductions, the accessible-base denominator -- matches the eager path.
    The mask is resolved over the source's inclusive chunk-grid bounds,
    ``grid_start``/``grid_end``: the requested region when one was given
    (so the full region is covered even past the first/last variant), else
    the variant span ``mappable_lo``/``mappable_hi``. Both bounds are
    inclusive, as ``resolve_accessible_mask`` expects.

    Parameters
    ----------
    accessible_bed : str, path-like, numpy.ndarray, AccessibleMask, or None
        Passthrough to ``resolve_accessible_mask``. ``None`` returns ``None``.
    source : ZarrGenotypeSource
        Streaming source; supplies ``grid_start``/``grid_end``/``chrom``.
    region : str, optional
        ``"chrom:start-end"`` region string (both ends inclusive, as in
        samtools); its contig overrides
        ``source.chrom`` for matching a BED file's contig name.
    include_invariant : bool
        The store has a record for each called base: the mask is the
        record positions (only those in ``accessible_bed``, if given), as
        on the eager path.

    Returns
    -------
    AccessibleMask or None
    """
    if accessible_bed is None and not include_invariant:
        return None
    chrom = parse_region(region)[0] or source.chrom
    if include_invariant:
        return _called_sites_mask(source.site_pos, source.grid_start,
                                  source.grid_end, accessible_bed, chrom)
    return resolve_accessible_mask(
        accessible_bed, source.grid_start, source.grid_end, chrom)
