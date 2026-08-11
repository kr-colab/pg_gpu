"""pg_gpu's warning categories and the load-time checks that emit them.

All of pg_gpu's user-facing ``UserWarning`` subclasses live here so they can be
imported and silenced from one place, e.g.::

    import warnings
    from pg_gpu import MemoryLimitedWarning
    warnings.filterwarnings("ignore", category=MemoryLimitedWarning)

The module also holds the load-time helpers that raise these warnings (the VCF
size heuristic and the diploid-encoding check), since they exist only to emit
them. ``BadlyChunkedWarning`` is raised from ``streaming_matrix`` where the
chunk geometry is known, but its class is defined here with the others.
"""

import collections
import gzip
import os
import warnings

import numpy as np

ISSUE_URL = "https://github.com/kr-colab/pg_gpu/issues/121"


# ── Warning categories ──────────────────────────────────────────────────────


class MemoryLimitedWarning(UserWarning):
    """A load that is at risk of exhausting memory or taking far longer
    than the VCZ path would.

    Emitted before parsing by ``HaplotypeMatrix.from_vcf`` and
    ``GenotypeMatrix.from_vcf`` when the VCF on disk is large enough,
    or has enough samples, that VCF text parsing will be slow and the
    full-matrix host load may not even fit. The warning text includes
    a copy-pastable recipe for converting the VCF to VCZ via
    ``HaplotypeMatrix.vcf_to_zarr``. Silence with::

        import warnings
        from pg_gpu import MemoryLimitedWarning
        warnings.filterwarnings("ignore", category=MemoryLimitedWarning)
    """


class HaploidDataWarning(UserWarning):
    """A loaded dataset is polymorphic yet contains no heterozygous calls.

    Emitted by the ``from_vcf`` / ``from_zarr`` loaders when not a single
    genotype is heterozygous even though alternate alleles are present. This is
    what a hemizygous sample set (e.g. males on the X chromosome) looks like
    when its calls are encoded as pseudo-diploid homozygotes -- but it is also
    what a fully inbred panel looks like, so it cannot be diagnosed from the
    data alone. If the samples really are hemizygous, treating them as diploid
    doubles their haplotype count and biases every statistic. Silence with::

        import warnings
        from pg_gpu import HaploidDataWarning
        warnings.filterwarnings("ignore", category=HaploidDataWarning)
    """


class BadlyChunkedWarning(UserWarning):
    """Emitted when ``backend='auto'`` picks ``host`` on a store whose
    call_genotype chunking would have defeated the kvikio fetcher's
    win. The store is functional but a bio2zarr-style re-encode would
    unlock the GPU-decode fast path."""


class MultiallelicCapWarning(UserWarning):
    """Emitted when a fused windowed kernel drops sites with more alleles
    than the kernel's fixed per-allele capacity.

    The fused windowed-statistics CUDA kernels count alleles into a fixed-size
    per-variant array (``MAX_ALLELES``), so any site with more alleles than that
    cap is excluded from the windowed scalar statistics and a count is reported.
    The cap is generous (nucleotide data has at most 4 alleles); this only fires
    on unusually multiallelic input. The (non-windowed) scalar functions in
    ``diversity`` / ``divergence`` handle arbitrary allele counts. Silence with::

        import warnings
        from pg_gpu import MultiallelicCapWarning
        warnings.filterwarnings("ignore", category=MultiallelicCapWarning)
    """


class BiallelicOnlyWarning(UserWarning):
    """Emitted when a biallelic-only statistic or loader drops multiallelic sites.

    Some pg_gpu statistics and loaders are computed on biallelic sites only and
    otherwise exclude multiallelic ones silently. This warning reports how many
    sites were dropped, so a partial result is not mistaken for a whole-data one.
    It marks an implementation-scope restriction, distinct from ``MultiallelicCapWarning``,
    which caps a multiallelic-capable windowed kernel at its fixed per-allele
    capacity. Silence with::

        import warnings
        from pg_gpu import BiallelicOnlyWarning
        warnings.filterwarnings("ignore", category=BiallelicOnlyWarning)
    """


def _warn_biallelic_only(n_dropped, *, context, stacklevel=3):
    """Emit ``BiallelicOnlyWarning`` that ``context`` dropped ``n_dropped`` sites.

    A no-op when ``n_dropped`` is 0, so callers can pass a raw count
    unconditionally. ``context`` names the operation that restricted to biallelic
    (e.g. ``"GenotypeMatrix.from_vcf"``, ``"patterson_d"``) and leads the message.
    """
    n_dropped = int(n_dropped)
    if n_dropped <= 0:
        return
    warnings.warn(
        f"{context} is defined on biallelic sites; dropped {n_dropped} "
        f"multiallelic site(s). To silence:\n"
        "    import warnings\n"
        "    from pg_gpu import BiallelicOnlyWarning\n"
        "    warnings.filterwarnings('ignore', category=BiallelicOnlyWarning)",
        BiallelicOnlyWarning, stacklevel=stacklevel,
    )


# ── VCF size heuristic ──────────────────────────────────────────────────────
#
# VCF text parsing is single-threaded in htslib and the whole genotype matrix
# gets pulled into host memory at once, so a very large VCF either takes hours
# to load or exhausts RAM. ``from_vcf`` calls ``_maybe_memory_warn`` before
# parsing to point users at the one-time VCZ conversion, whose re-reads are
# seconds to minutes and can stream a chromosome that doesn't fit in memory.


#: Trigger thresholds. Above these sizes / sample counts a VCF load is
#: slow enough that converting to VCZ once pays for itself within a
#: single re-read. The constants are kwargs on ``_maybe_memory_warn``
#: so tests can lower them without a global monkeypatch.
VCF_WARN_BYTES = 10 * 1024 ** 3      # 10 GiB on disk
VCF_WARN_SAMPLES = 5_000
VCF_WARN_REGION_BP = 5_000_000       # 5 Mb


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
    very large VCF this finishes in milliseconds even when the file
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
        "single-threaded in htslib, and the full genotype matrix "
        "must fit in host memory. Convert once to VCZ and use "
        "HaplotypeMatrix.from_zarr() instead -- subsequent reads finish "
        "in seconds rather than hours, and the streaming path can "
        "walk a chromosome larger than memory chunk by chunk.\n\n"
        "One-time conversion:\n"
        "    from pg_gpu import HaplotypeMatrix\n"
        f"    HaplotypeMatrix.vcf_to_zarr({path!r},\n"
        f"                                {(path.rsplit('.', 1)[0] + '.vcz')!r})\n\n"
        "Then in pg_gpu:\n"
        f"    hm = HaplotypeMatrix.from_zarr({(path.rsplit('.', 1)[0] + '.vcz')!r})\n\n"
        "To silence:\n"
        "    import warnings\n"
        "    from pg_gpu import MemoryLimitedWarning\n"
        "    warnings.filterwarnings('ignore', category=MemoryLimitedWarning)\n"
    )


def _maybe_memory_warn(path, *, region=None,
                       warn_bytes=VCF_WARN_BYTES,
                       warn_samples=VCF_WARN_SAMPLES,
                       warn_region_bp=VCF_WARN_REGION_BP,
                       stacklevel=3):
    """Emit ``MemoryLimitedWarning`` if loading ``path`` will be slow.

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
    # gzipped multi-GB file decompresses several KB of metadata
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
            MemoryLimitedWarning,
            stacklevel=stacklevel,
        )
        _warned_paths[canonical] = None
        while len(_warned_paths) > _WARN_CACHE_MAX:
            _warned_paths.popitem(last=False)


# ── Diploid-encoding check ──────────────────────────────────────────────────
#
# pg_gpu's loaders assume diploid genotypes: every genotype call is split into
# two haplotype rows (or summed into a 0/1/2 dosage). Haploid data violates
# that in two ways that otherwise pass silently:
#
# * True haploid calls (GT=1) are padded by scikit-allel to a second -1 allele,
#   so the second haplotype row of every sample is spurious missing data and
#   the haplotype count is doubled.
# * Pseudo-diploid homozygous calls (GT=1/1, the convention many callers emit
#   for haploid regions such as the male X) load as two identical haplotypes
#   per sample, inflating homozygosity and biasing the SFS, pi, and Tajima's D.
#
# Neither is distinguishable from valid diploid data by array shape alone --
# read_vcf returns a ploidy-2 layout in both cases -- so the check inspects the
# allele values: a genuine haploid call raises, an all-homozygous polymorphic
# dataset (equally what an inbred panel looks like) warns.


def _format_sample_list(names, idx, limit=6):
    """Render up to ``limit`` flagged samples as a readable string."""
    flagged = [str(names[i]) for i in idx]
    head = ", ".join(flagged[:limit])
    if len(flagged) > limit:
        head += f", ... (+{len(flagged) - limit} more)"
    return head


def _require_diploid_ploidy(ploidy, source):
    """Raise unless the ploidy axis is 2.

    Shared by the streaming reader (ploidy from store metadata) and the
    value-based detector so the message lives in one place.
    """
    if ploidy != 2:
        raise ValueError(
            f"{source} has ploidy {ploidy}; pg_gpu's loaders support diploid "
            f"(ploidy 2) data only. Haploid and polyploid data are not yet "
            f"supported (see {ISSUE_URL}).")


def check_diploid_encoding(gt, sample_names=None, source="input"):
    """Validate that ``gt`` is genuinely diploid before it is loaded.

    Parameters
    ----------
    gt : array-like
        Genotype calls, shape ``(n_variants, n_samples, ploidy)``, alleles
        encoded as non-negative ints with ``-1`` for missing or for the
        padding scikit-allel adds to haploid calls.
    sample_names : sequence of str, optional
        Sample ids, used to name flagged samples in messages. Falls back to
        positional indices when not given or the wrong length.
    source : str
        Loader label woven into error/warning text (e.g. ``"VCF 'x.vcf'"``).

    Raises
    ------
    ValueError
        If the array has a ploidy other than 2, or contains a sample whose
        calls are all haploid (a called first allele with a padded second
        allele, and never any diploid call). These are encodings the diploid
        loaders handle incorrectly with no valid interpretation.

    Warns
    -----
    HaploidDataWarning
        If the dataset is polymorphic but contains no heterozygous genotype,
        which is what a pseudo-diploid hemizygous sample set looks like.
    """
    gt = np.asarray(gt)
    if gt.ndim != 3:
        raise ValueError(
            f"Expected a 3-D genotype array (n_variants, n_samples, ploidy) "
            f"from {source}; got shape {gt.shape}.")

    _, n_samples, ploidy = gt.shape
    if sample_names is None or len(sample_names) != n_samples:
        sample_names = [f"sample_{i}" for i in range(n_samples)]

    _require_diploid_ploidy(ploidy, source)

    a0 = gt[:, :, 0]
    a1 = gt[:, :, 1]
    called0 = a0 >= 0
    called1 = a1 >= 0

    # scikit-allel pads a haploid call (GT=1) as [allele, -1], but a half-missing
    # diploid (GT=1/.) looks identical. So a sample is haploid only if its second
    # allele is missing at every site; one full or [-1, allele] call rules it out.
    haploid_sample = called0.any(axis=0) & ~called1.any(axis=0)
    if haploid_sample.any():
        idx = np.nonzero(haploid_sample)[0]
        listed = _format_sample_list(sample_names, idx)
        raise ValueError(
            f"{source} contains haploid genotype calls for {len(idx)} "
            f"sample(s): {listed}. pg_gpu loads these as diploid, so each "
            f"haploid call becomes a half-missing diploid -- doubling the "
            f"haplotype count and biasing every statistic. Haploid / "
            f"hemizygous input is not yet supported (see {ISSUE_URL}).")

    # Polymorphic but not one heterozygote anywhere: looks like hemizygous
    # samples encoded as homozygous diploids (or a fully inbred panel), so warn
    # rather than raise. Checked dataset-wide -- a single real het rules it out,
    # so ordinary diploid data with a few all-homozygous samples stays quiet.
    carries_alt = (a0 > 0) | (a1 > 0)
    has_heterozygote = bool((called0 & called1 & (a0 != a1)).any())
    if carries_alt.any() and not has_heterozygote:
        carriers = np.nonzero(carries_alt.any(axis=0))[0]
        listed = _format_sample_list(sample_names, carriers)
        warnings.warn(
            f"{source} is polymorphic but contains no heterozygous genotype "
            f"across {n_samples} sample(s); samples carrying alternate alleles "
            f"include {listed}. If these are hemizygous (e.g. males on a sex "
            f"chromosome) encoded as homozygous diploids, treating them as "
            f"diploid doubles their haplotype count and biases pi, the SFS, "
            f"and Tajima's D; if they are an inbred panel this is expected "
            f"(see {ISSUE_URL}).",
            HaploidDataWarning, stacklevel=3,
        )
