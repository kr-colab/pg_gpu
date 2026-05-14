"""``BiobankScaleWarning`` and the size check ``from_vcf`` uses to fire it.

VCF text parsing is single-threaded in htslib and dominates loading
wall time at biobank scale. ``HaplotypeMatrix.from_vcf`` and
``GenotypeMatrix.from_vcf`` call ``_maybe_biobank_warn`` before
parsing so users with very large VCFs get pointed at the
one-time-conversion path -- ``HaplotypeMatrix.vcf_to_zarr`` builds a
VCZ store, and subsequent reads via ``from_zarr`` are seconds to
minutes instead of hours.
"""

import collections
import gzip
import os
import warnings


class BiobankScaleWarning(UserWarning):
    """A VCF load that will be slow because of its size or sample count.

    Emitted before parsing by ``HaplotypeMatrix.from_vcf`` and
    ``GenotypeMatrix.from_vcf``. The text includes a copy-pastable
    recipe for converting the VCF to VCZ via
    ``HaplotypeMatrix.vcf_to_zarr``. Silence with::

        import warnings
        from pg_gpu import BiobankScaleWarning
        warnings.filterwarnings("ignore", category=BiobankScaleWarning)
    """


#: Trigger thresholds. Above these sizes / sample counts a VCF load is
#: slow enough that converting to VCZ once pays for itself within a
#: single re-read. The constants are kwargs on ``_maybe_biobank_warn``
#: so tests can lower them without a global monkeypatch.
BIOBANK_VCF_WARN_BYTES = 10 * 1024 ** 3      # 10 GiB on disk
BIOBANK_VCF_WARN_SAMPLES = 5_000
BIOBANK_VCF_WARN_REGION_BP = 5_000_000       # 5 Mb


#: Maximum number of distinct VCF paths the in-process warn cache will
#: track before it starts evicting the oldest entries. Caps memory in
#: long-running processes (Jupyter, servers) that hit many unique VCFs.
_WARN_CACHE_MAX = 1000

#: Cache of already-warned paths -- keys are canonicalized via
#: ``os.path.realpath`` so ``./foo.vcf`` and ``/abs/foo.vcf`` collapse.
#: Insertion-ordered: when full, the oldest entry is popped first.
_warned_paths = collections.OrderedDict()


def _vcf_header_sample_count(path):
    """Return the number of samples in a VCF's ``#CHROM`` header line,
    or ``None`` if the file cannot be opened or has no header.

    Reads the header only -- stops at the first non-``##`` line. On a
    biobank-scale VCF that finishes in milliseconds even when the file
    is hundreds of GB."""
    opener = (gzip.open if path.endswith(".gz") or path.endswith(".bgz")
              else open)
    try:
        with opener(path, "rt") as f:
            for line in f:
                if line.startswith("#CHROM"):
                    parts = line.rstrip("\n").rstrip("\r").split("\t")
                    # columns 0-8 are CHROM, POS, ..., FORMAT; samples
                    # start at column 9.
                    return max(0, len(parts) - 9)
                if not line.startswith("##"):
                    return None
    except OSError:
        return None
    return None


def _region_span_bp(region):
    """Width of a ``'chrom:start-end'`` string in bp, or 0 on a parse
    failure. Used to decide whether a region argument is small enough
    that VCF loading is the right tool even on a large file."""
    if region is None:
        return 0
    from .zarr_io import _parse_region
    try:
        _, start, end = _parse_region(region)
    except (ValueError, AttributeError, TypeError):
        return 0
    return max(0, end - start)


def _format_warning_text(path, size_bytes, n_samples):
    return (
        f"{path} is {size_bytes / 1e9:.1f} GB"
        f"{f' with {n_samples:,} samples' if n_samples is not None else ''}. "
        "Loading this VCF will be slow because VCF text parsing is "
        "single-threaded in htslib. For biobank-scale repeated "
        "analysis, convert once to VCZ and use "
        "HaplotypeMatrix.from_zarr() instead -- subsequent reads finish "
        "in seconds rather than hours, and the kvikio backend can "
        "speed up sample-subset reads by 100x+.\n\n"
        "One-time conversion:\n"
        "    from pg_gpu import HaplotypeMatrix\n"
        f"    HaplotypeMatrix.vcf_to_zarr({path!r},\n"
        f"                                {(path.rsplit('.', 1)[0] + '.vcz')!r})\n\n"
        "Then in pg_gpu:\n"
        f"    hm = HaplotypeMatrix.from_zarr({(path.rsplit('.', 1)[0] + '.vcz')!r})\n\n"
        "To silence:\n"
        "    import warnings\n"
        "    from pg_gpu import BiobankScaleWarning\n"
        "    warnings.filterwarnings('ignore', category=BiobankScaleWarning)\n"
    )


def _maybe_biobank_warn(path, *, region=None,
                       warn_bytes=BIOBANK_VCF_WARN_BYTES,
                       warn_samples=BIOBANK_VCF_WARN_SAMPLES,
                       warn_region_bp=BIOBANK_VCF_WARN_REGION_BP,
                       stacklevel=3):
    """Emit ``BiobankScaleWarning`` if loading ``path`` will be slow.

    Triggers when either:

    * The on-disk file size exceeds ``warn_bytes`` AND either no
      ``region`` was passed or the region's span exceeds
      ``warn_region_bp``. A tabix-region read of a few Mb from a 50 GB
      VCF is fine and does not warn.
    * The ``#CHROM`` header lists more than ``warn_samples`` samples.

    Cached per path within a process so repeat loads of the same file
    only warn once. Used by both ``HaplotypeMatrix.from_vcf`` and
    ``GenotypeMatrix.from_vcf``.
    """
    # Symlinks and relative paths to the same inode should only warn
    # once; canonicalize before consulting the cache. Stat is cheap and
    # only runs on a path we have not warned on yet.
    try:
        canonical = os.path.realpath(path)
    except OSError:
        canonical = path
    if canonical in _warned_paths:
        return
    try:
        size = os.path.getsize(canonical)
    except OSError:
        return

    big_file = size > warn_bytes
    region_big = (region is None
                  or _region_span_bp(region) > warn_region_bp)

    # The sample-count check requires opening the VCF header, which on a
    # gzipped biobank-scale file decompresses several KB of metadata
    # lines. Skip it when the size check already cannot trigger -- a
    # below-threshold file with a too-small region cannot warn on
    # sample count alone either, so the read is pure waste.
    if not big_file and (region is not None and not region_big):
        return

    n_samples = _vcf_header_sample_count(canonical)
    too_many_samples = (n_samples is not None and n_samples > warn_samples)

    if (big_file and region_big) or too_many_samples:
        warnings.warn(
            _format_warning_text(canonical, size, n_samples),
            BiobankScaleWarning,
            stacklevel=stacklevel,
        )
        _warned_paths[canonical] = None
        while len(_warned_paths) > _WARN_CACHE_MAX:
            _warned_paths.popitem(last=False)
