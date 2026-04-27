"""Accessible site mask for genome-wide statistics normalization.

Provides BED file parsing and an AccessibleMask class that wraps a dense
boolean array with offset-aware, O(1) range queries via prefix sums.
"""

import numpy as np


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
        return AccessibleMask(self.mask[i:j], offset=start)

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
        If provided, only return intervals on this chromosome.

    Returns
    -------
    list of (str, int, int)
        Parsed intervals as (chromosome, start, end) with 0-based half-open
        coordinates.
    """
    intervals = []
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
            c, s, e = fields[0], int(fields[1]), int(fields[2])
            if chrom is None or c == chrom:
                intervals.append((c, s, e))
    return intervals


def bed_to_mask(path, chrom, length, offset=0):
    """Convert BED intervals into a dense boolean AccessibleMask.

    Parameters
    ----------
    path : str or path-like
        Path to a BED file defining accessible regions.
    chrom : str
        Chromosome name to extract from the BED file.
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
    intervals = parse_bed(path, chrom=chrom)
    mask = np.zeros(length, dtype=bool)
    for _, s, e in intervals:
        # BED (s, e) [0-based half-open] = 1-based positions s+1..e
        # mask[k] = True for k where offset + k in [s+1, e]
        i = max(0, s + 1 - offset)
        j = min(length, e + 1 - offset)
        if j > i:
            mask[i:j] = True
    return AccessibleMask(mask, offset=offset)


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
        Chromosome name (required for BED file input).

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
    intervals = parse_bed(mask_or_path, chrom=chrom)
    if not intervals:
        raise ValueError(
            f"No BED intervals found for chrom={chrom!r} in {mask_or_path}")
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
    return bed_to_mask(mask_or_path, chrom=chrom, length=length, offset=offset)
