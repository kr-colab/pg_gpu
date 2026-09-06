"""Zarr I/O utilities for VCZ (bio2zarr) and scikit-allel formats."""

import os
import shutil
import re
import sys

import numpy as np


# A colon-less string is a bare span ('1000-2000') only if it looks like
# one; anything else is a chromosome name.
_BARE_SPAN = re.compile(r'^[\d,_]+-[\d,_]*$')


def parse_region(region):
    """Parse a samtools-style region string into ``(chrom, start, stop)``.

    Accepts the forms samtools, tabix, and bcftools do: ``'chrom'``,
    ``'chrom:start'``, ``'chrom:start-'``, and ``'chrom:start-end'``, with
    optional thousands separators, comma or underscore
    (``'chr1:1,000-2,000'``, ``'chr1:1_000-2_000'``). Both ends are
    inclusive, so ``'X:500-500'`` names one position. A bare
    ``'start-end'`` span with no chromosome is also accepted and returns
    ``chrom=None``.

    The returned ``stop`` is ``end + 1``, so ``(start, stop)`` is the
    half-open interval the rest of pg_gpu works in (``slice_region``,
    ``materialize``, windows) and ``positions < stop`` is the right test.
    A bound the string leaves open is ``None``, and ``None`` itself
    parses to ``(None, None, None)``.
    """
    if region is None:
        return None, None, None
    chrom, sep, coords = region.rpartition(':')
    if not sep:
        chrom, coords = ((None, region) if _BARE_SPAN.match(region)
                         else (region, ''))
    start_s, _, end_s = coords.partition('-')
    try:
        start = int(start_s.replace(',', '')) if start_s else None
        end = int(end_s.replace(',', '')) if end_s else None
    except ValueError:
        raise ValueError(
            f"bad region string {region!r}: expected 'chrom', 'chrom:start', "
            f"or 'chrom:start-end'") from None
    return chrom or None, start, None if end is None else end + 1


def _region_mask(positions, start, stop):
    """Boolean mask of ``positions`` inside the half-open ``[start, stop)``;
    a ``None`` bound is open on that side."""
    if start is None and stop is None:
        return np.ones(len(positions), dtype=bool)
    if start is None:
        return positions < stop
    if stop is None:
        return positions >= start
    return (positions >= start) & (positions < stop)


def _vcz_contig_index(store, contig_arr, chrom):
    """``(index, name)`` of ``chrom`` in a VCZ store's contig table, or of
    the only contig with variants when ``chrom`` is ``None``.

    Raises ``ValueError`` for an unknown name, or for ``None`` on a store
    whose variants span more than one contig.
    """
    contig_ids = list(np.array(store['contig_id']))
    if chrom is not None:
        if chrom not in contig_ids:
            raise ValueError(f"Contig {chrom!r} not found. Available: {contig_ids}")
        idx = contig_ids.index(chrom)
    elif contig_arr.size == 0:
        idx = 0
    else:
        # min and max settle the single-contig case without sorting a
        # copy of the whole variant axis.
        idx = int(contig_arr.min())
        if int(contig_arr.max()) != idx:
            names = [contig_ids[i] for i in np.unique(contig_arr)]
            raise ValueError(
                f"Store contains {len(names)} contigs: {names}. "
                "Pass region='chrom' or region='chrom:start-end' to select one."
            )
    return idx, (contig_ids[idx] if idx < len(contig_ids) else None)


def _allel_group(store, chrom):
    """The chromosome group ``chrom`` of a grouped scikit-allel store."""
    if chrom is None or chrom not in store:
        available = [k for k in store if hasattr(store[k], 'keys')]
        raise ValueError(f"Chromosome {chrom!r} not found. Available: {available}")
    return store[chrom]


def normalize_pop_input(pop_assignment, *, zarr_path, sample_names,
                         zarr_store=None, announce_prefix=""):
    """Normalize the flexible ``pop_assignment`` kwarg into a
    ``{sample_id -> pop_label}`` dict (or ``None`` to mean "no
    assignments").

    Accepted forms:

    * ``False`` -- disable both the companion ``.pops.tsv`` auto-load
      and any other source. Returns ``None``.
    * ``None`` -- auto-load ``<zarr_path>.pops.tsv`` if it exists.
      Returns the parsed mapping (or ``None`` if no companion is
      present). Announces the auto-load to stderr.
    * ``dict`` -- returned as-is.
    * ``numpy.ndarray`` / ``list`` of length ``len(sample_names)`` --
      one population label per sample, in the same order as
      ``sample_names``. Empty / ``""`` / ``None`` entries are skipped.
    * ``str`` -- first checked against ``zarr_store`` if provided: a
      key that resolves to a 1-D string array of the right length is
      read out of the store. Otherwise interpreted as a filesystem
      path to a tab-delimited ``sample\\tpop`` file with a header
      row.

    ``sample_names`` must be the store's sample axis (used both for
    array-shaped inputs and for path-shaped inputs to filter sample
    membership). ``zarr_store`` is optional; pass it when the caller
    has the open store handy so a zarr-key string is preferred over
    a same-named file on disk.
    """
    import numpy as np

    if pop_assignment is False:
        return None

    if pop_assignment is None:
        companion = str(zarr_path).rstrip("/") + ".pops.tsv"
        if not os.path.exists(companion):
            return None
        print(f"{announce_prefix}: auto-loaded pop file {companion}",
              file=sys.stderr, flush=True)
        pop_assignment = companion

    if isinstance(pop_assignment, dict):
        return {str(k): str(v) for k, v in pop_assignment.items() if v}

    if isinstance(pop_assignment, (list, tuple, np.ndarray)):
        labels = np.asarray(pop_assignment)
        if labels.ndim != 1:
            raise ValueError(
                f"pop_assignment array must be 1-D; got shape {labels.shape}")
        if len(labels) != len(sample_names):
            raise ValueError(
                f"pop_assignment array length {len(labels)} does not "
                f"match sample axis length {len(sample_names)}")
        return _pop_array_to_map(labels, sample_names)

    if isinstance(pop_assignment, str):
        if zarr_store is not None and pop_assignment in zarr_store:
            labels = np.asarray(zarr_store[pop_assignment][:])
            if labels.ndim != 1 or len(labels) != len(sample_names):
                raise ValueError(
                    f"zarr key {pop_assignment!r} has shape {labels.shape}; "
                    f"expected 1-D of length {len(sample_names)} to "
                    f"line up with the sample axis")
            return _pop_array_to_map(labels, sample_names)
        return _read_pop_tsv(pop_assignment)

    raise TypeError(
        f"pop_assignment must be a path, dict, array, zarr key, False, "
        f"or None; got {type(pop_assignment).__name__}")


def _pop_array_to_map(labels, sample_names):
    out = {}
    for sample, label in zip(sample_names, labels):
        if label is None:
            continue
        label = str(label)
        if not label or label.lower() == "nan":
            continue
        out[str(sample)] = label
    return out


def _read_pop_tsv(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] != "sample":
                out[parts[0]] = parts[1]
    return out


def detect_zarr_layout(store):
    """Detect whether a zarr store uses VCZ or scikit-allel layout.

    Parameters
    ----------
    store : zarr.Group
        Opened zarr store.

    Returns
    -------
    str
        One of 'vcz', 'scikit-allel', or 'scikit-allel-grouped'.
    """
    if 'call_genotype' in store:
        return 'vcz'
    if 'calldata' in store:
        return 'scikit-allel'
    # Check for chromosome-group layout (e.g., Ag1000G: store['3L']['calldata/GT'])
    for key in store:
        item = store[key]
        if hasattr(item, 'keys') and 'calldata' in item:
            return 'scikit-allel-grouped'
    raise ValueError(
        "Unrecognized zarr layout. Expected VCZ (call_genotype) or "
        "scikit-allel (calldata/GT) fields."
    )


def read_genotypes_vcz(store, region=None):
    """Read genotype data from a VCZ-format zarr store.

    Parameters
    ----------
    store : zarr.Group
        Opened VCZ zarr store.
    region : str, optional
        Genomic region string; see ``parse_region`` for the accepted
        forms.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples',
        and 'variant_indices' -- the index array used to slice the
        store when ``region`` is given (``None`` for a whole-store
        read). Auxiliary callers (e.g. QC field readers) can reuse the
        same indices to keep their slices aligned with the genotype
        matrix.
    """
    contig_arr = np.array(store['variant_contig'])
    chrom, start, stop = parse_region(region)
    # On a whole-store read this call is the multi-contig guard: a store
    # spanning several contigs needs a region to name one.
    contig_idx, _ = _vcz_contig_index(store, contig_arr, chrom)
    if region is not None:
        pos_arr = np.array(store['variant_position'])
        mask = (contig_arr == contig_idx) & _region_mask(pos_arr, start, stop)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No variants in region {region}")
        gt = np.array(store['call_genotype'][indices])
        positions = pos_arr[indices]
        variant_indices = indices
    else:
        gt = np.array(store['call_genotype'])
        positions = np.array(store['variant_position'])
        variant_indices = None

    samples = list(np.array(store['sample_id'])) if 'sample_id' in store else None
    return {'gt': gt, 'positions': positions, 'samples': samples,
            'variant_indices': variant_indices}


def read_qc_fields(zarr_path, fields, variant_indices=None, region=None):
    """Pull requested VCF FORMAT/INFO arrays from any supported zarr layout.

    Dispatches on the store layout (VCZ vs scikit-allel flat vs scikit-allel
    grouped). For each bare tag, look up the per-variant key first and fall
    back to the per-genotype key; the lookup names depend on the layout:

    * **VCZ** (``bio2zarr`` output): ``variant_<tag>`` (INFO) then
      ``call_<tag>`` (FORMAT).
    * **scikit-allel** (flat or grouped): ``variants/<tag>`` (INFO) then
      ``calldata/<tag>`` (FORMAT). For a grouped layout (chromosome-keyed
      subgroups), the chrom in ``region`` selects which subgroup to read
      from.

    Tags matching neither are warned and dropped silently. ``variant_indices``,
    when provided, slices the variant axis to keep the returned arrays
    aligned with a windowed genotype matrix.
    """
    import zarr
    store = zarr.open_group(zarr_path, mode='r')
    layout = detect_zarr_layout(store)
    if layout == 'vcz':
        return _read_qc_fields_vcz(store, fields, variant_indices)
    if layout == 'scikit-allel-grouped':
        if region is None:
            # The genotype reader for this layout requires region= and would
            # have already raised; return empty rather than introducing a
            # second error message.
            return {}
        return _read_qc_fields_allel(_allel_group(store, parse_region(region)[0]),
                                     fields, variant_indices)
    return _read_qc_fields_allel(store, fields, variant_indices)


def _read_qc_fields_vcz(store, fields, variant_indices):
    """VCZ-style lookup: ``variant_<tag>`` / ``call_<tag>``."""
    return _pull_qc_fields(
        store, fields, variant_indices,
        var_key=lambda tag: f'variant_{tag}',
        call_key=lambda tag: f'call_{tag}',
        layout_label='VCZ',
    )


def _read_qc_fields_allel(group, fields, variant_indices):
    """scikit-allel-style lookup: ``variants/<tag>`` / ``calldata/<tag>``."""
    return _pull_qc_fields(
        group, fields, variant_indices,
        var_key=lambda tag: f'variants/{tag}',
        call_key=lambda tag: f'calldata/{tag}',
        layout_label='scikit-allel',
    )


def _pull_qc_fields(group, fields, variant_indices, *, var_key, call_key,
                     layout_label):
    """Layout-agnostic core: probe ``var_key(tag)`` then ``call_key(tag)``."""
    import warnings
    out = {}
    missing = []
    for tag in fields:
        vk = var_key(tag)
        ck = call_key(tag)
        if vk in group:
            arr = group[vk]
        elif ck in group:
            arr = group[ck]
        else:
            missing.append(tag)
            continue
        out[tag] = (np.array(arr[variant_indices])
                    if variant_indices is not None else np.array(arr))
    if missing:
        warnings.warn(
            f"{layout_label} quality fields not found and dropped: {missing}",
            stacklevel=4,
        )
    return out


def read_genotypes_allel(store, region=None):
    """Read genotype data from a scikit-allel format zarr store.

    Parameters
    ----------
    store : zarr.Group
        Opened scikit-allel zarr store.
    region : str, optional
        Genomic region string; see ``parse_region`` for the accepted
        forms.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples',
        and 'variant_indices' -- the int index array used to slice the
        store when ``region`` is given (``None`` for a whole-store
        read). Auxiliary callers (e.g. QC field readers) reuse the same
        indices to stay aligned with the genotype matrix.
    """
    positions = np.array(store['variants/POS'])
    gt = np.array(store['calldata/GT'])
    samples = list(np.array(store['samples'])) if 'samples' in store else None
    variant_indices = None

    if region is not None:
        _, start, stop = parse_region(region)
        mask = _region_mask(positions, start, stop)
        variant_indices = np.where(mask)[0]
        positions = positions[mask]
        gt = gt[mask]
        if len(positions) == 0:
            raise ValueError(f"No variants in region {region}")

    return {'gt': gt, 'positions': positions, 'samples': samples,
            'variant_indices': variant_indices}


def read_genotypes_allel_grouped(store, region):
    """Read genotype data from a chromosome-grouped scikit-allel store.

    Parameters
    ----------
    store : zarr.Group
        Opened zarr store with chromosome-level groups.
    region : str
        Genomic region string; see ``parse_region`` for the accepted
        forms.
        Required for grouped stores.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples'.
    """
    chrom, start, stop = parse_region(region)
    if chrom is None:
        available = [k for k in store if hasattr(store[k], 'keys')]
        raise ValueError(
            f"Grouped zarr store requires region='chrom' or "
            f"'chrom:start-end'. Available groups: {available}"
        )
    grp = _allel_group(store, chrom)
    positions = np.array(grp['variants/POS'])
    gt = np.array(grp['calldata/GT'])
    samples = list(np.array(grp['samples'])) if 'samples' in grp else None

    mask = _region_mask(positions, start, stop)
    variant_indices = np.where(mask)[0]
    positions = positions[mask]
    gt = gt[mask]
    if len(positions) == 0:
        raise ValueError(f"No variants in region {region}")

    return {'gt': gt, 'positions': positions, 'samples': samples,
            'variant_indices': variant_indices}


def write_vcz(zarr_path, gt, positions, samples=None, contig_name=None,
              chunks=None, fields=None):
    """Write genotype data in VCZ format.

    Parameters
    ----------
    zarr_path : str
        Output zarr store path.
    gt : ndarray, shape (n_variants, n_samples, ploidy)
        Genotype array.
    positions : ndarray, shape (n_variants,)
        Variant positions.
    samples : list of str, optional
        Sample names.
    contig_name : str, optional
        Chromosome/contig name.
    chunks : tuple of int, optional
        Chunk shape for ``call_genotype`` and ``call_genotype_mask``,
        e.g. ``(10000, 1000, 2)`` to mirror bio2zarr's defaults. When
        ``None`` (default) zarr picks the chunking, which on a small
        array is the whole array as a single chunk.
    fields : dict, optional
        Optional VCF FORMAT/INFO arrays to round-trip alongside the
        genotype matrix. Keys are bare VCF tags (``'GQ'``, ``'DP'``,
        ``'MQ'``, ...); values are arrays whose first axis is the
        variant axis. Shape disambiguates INFO vs FORMAT:

        * ``(n_variants,)`` writes to ``variant_<tag>`` (INFO).
        * ``(n_variants, n_samples)`` writes to ``call_<tag>`` (FORMAT).

        Anything else raises ``ValueError``. The dtype written matches
        the input array so the round-trip is byte-exact.
    """
    import zarr
    n_var = len(positions)
    n_samples = gt.shape[1]
    store = zarr.open(zarr_path, mode='w')
    if chunks is None:
        store.create_array('call_genotype', data=gt.astype(np.int8))
        store.create_array('call_genotype_mask', data=(gt < 0))
    else:
        # create_array with explicit chunks needs shape + dtype rather
        # than `data=`, so the chunks=... kwarg is honored before any
        # write resizes the array.
        cg = store.create_array('call_genotype',
                                shape=gt.shape, chunks=chunks,
                                dtype='int8')
        cg[:] = gt.astype(np.int8)
        cm = store.create_array('call_genotype_mask',
                                shape=gt.shape, chunks=chunks,
                                dtype='bool')
        cm[:] = (gt < 0)
    store.create_array('variant_position', data=positions.astype(np.int32))
    if samples is not None:
        store.create_array('sample_id', data=np.array(samples, dtype='U'))
    if contig_name is not None:
        store.create_array('contig_id',
                           data=np.array([contig_name], dtype='U'))
        store.create_array('variant_contig',
                           data=np.zeros(len(positions), dtype=np.int8))
    if fields:
        for tag, arr in fields.items():
            arr = np.asarray(arr)
            if arr.ndim == 0 or arr.shape[0] != n_var:
                raise ValueError(
                    f"field {tag!r} must have a leading axis of length "
                    f"{n_var} (the variant axis); got shape "
                    f"{tuple(arr.shape)}")
            # FORMAT is exactly (n_var, n_samples); anything else with
            # the right leading axis is treated as INFO. Multi-dim INFO
            # (e.g. ``variant_AF`` arrives as ``(n_var, n_alt)`` from
            # bio2zarr) round-trips into ``variant_<tag>`` with its
            # full shape preserved.
            if arr.ndim == 2 and arr.shape == (n_var, n_samples):
                store.create_array(f'call_{tag}', data=arr)
            else:
                store.create_array(f'variant_{tag}', data=arr)


def allel_zarr_to_vcz(allel_path, vcz_path, *, contig=None, region=None,
                      variant_chunk=10_000, sample_chunk=1_000,
                      progress=False):
    """Convert a scikit-allel zarr store to vcz (bio2zarr) layout.

    Streams ``calldata/GT`` in variant blocks so chromosome-scale allel
    stores -- which the eager ``read_genotypes_allel`` would OOM on --
    can be converted without materializing the full genotype matrix.
    Writes ``call_genotype`` with bio2zarr-style sample-axis chunking so
    the pg_gpu streaming reader's kvikio backend can decode chunks on
    the GPU.

    Parameters
    ----------
    allel_path : str
        Source allel store. Either a flat layout with ``calldata/GT`` at
        the root, or a chromosome-grouped layout where each
        ``<contig>/calldata/GT`` is one chromosome. Both are accepted.
    vcz_path : str
        Destination vcz store path. Overwritten if it exists.
    contig : str, optional
        For grouped stores, the chromosome key to read. Required when
        the source is grouped and ``region`` does not name a contig.
        For flat stores, used as the ``contig_id`` label in the output.
    region : str, optional
        Region string restricting the conversion to a position range;
        see ``parse_region`` for the accepted forms, including a bare
        ``"start-end"`` on a flat store.
    variant_chunk : int
        Source variants read per streaming pass. Larger amortizes zarr
        read overhead; smaller bounds host RAM at biobank sample counts.
    sample_chunk : int
        Sample-axis chunk size in the output ``call_genotype``. Smaller
        chunks let the streaming reader's kvikio + nvCOMP decode overlap
        with compute on biobank-scale stores.
    progress : bool
        Print a one-line status per variant chunk to stderr.

    Notes
    -----
    Only the fields pg_gpu's streaming reader requires are written:
    ``call_genotype``, ``call_genotype_mask``, ``variant_position``,
    ``sample_id``, ``contig_id``, ``variant_contig``. Other allel
    columns (REF, ALT, FILTER, etc.) are not preserved.
    """
    import sys
    import zarr

    src = zarr.open_group(allel_path, mode='r')
    layout = detect_zarr_layout(src)
    if layout == 'vcz':
        raise ValueError(
            f"{allel_path} is already vcz; nothing to convert"
        )
    if layout not in ('scikit-allel', 'scikit-allel-grouped'):
        raise ValueError(
            f"Expected scikit-allel layout at {allel_path}, got {layout}"
        )

    chrom_from_region, lo, stop = parse_region(region)

    # Resolve which chromosome group to read (grouped) or use the root
    # (flat); derive the contig label that goes into the output.
    if layout == 'scikit-allel-grouped':
        contig = contig or chrom_from_region
        if contig is None:
            available = [k for k in src if hasattr(src[k], 'keys')]
            raise ValueError(
                f"Grouped allel store requires contig=... "
                f"Available: {available}"
            )
        src_grp = _allel_group(src, contig)
    else:
        src_grp = src
        if contig is None:
            contig = 'unknown'

    gt_arr = src_grp['calldata/GT']
    pos_arr = src_grp['variants/POS']
    n_var, n_samp, ploidy = gt_arr.shape

    if region:
        # Loading positions in full is fine -- they're int32, so even a
        # human-genome-scale chromosome (~10 M sites) is ~40 MB.
        pos_host = np.asarray(pos_arr[:])
        lo_idx = (0 if lo is None
                  else int(np.searchsorted(pos_host, lo, side='left')))
        hi_idx = (n_var if stop is None
                  else int(np.searchsorted(pos_host, stop, side='left')))
        if lo_idx == hi_idx:
            raise ValueError(f"No variants in region {region}")
        pos_out = pos_host[lo_idx:hi_idx]
    else:
        lo_idx, hi_idx = 0, n_var
        pos_out = np.asarray(pos_arr[:])

    n_out = hi_idx - lo_idx
    samples = (np.asarray(src_grp['samples'][:])
               if 'samples' in src_grp else None)

    dst = zarr.open_group(vcz_path, mode='w')
    cg = dst.create_array(
        'call_genotype',
        shape=(n_out, n_samp, ploidy),
        chunks=(variant_chunk, sample_chunk, ploidy),
        dtype='int8',
    )
    cm = dst.create_array(
        'call_genotype_mask',
        shape=(n_out, n_samp, ploidy),
        chunks=(variant_chunk, sample_chunk, ploidy),
        dtype='bool',
    )
    for s in range(lo_idx, hi_idx, variant_chunk):
        e = min(s + variant_chunk, hi_idx)
        block = np.asarray(gt_arr[s:e])
        dst_s, dst_e = s - lo_idx, e - lo_idx
        cg[dst_s:dst_e] = block.astype(np.int8)
        cm[dst_s:dst_e] = (block < 0)
        if progress:
            done = dst_e
            print(f"[allel_zarr_to_vcz] {done}/{n_out} variants",
                  file=sys.stderr, flush=True)

    dst.create_array('variant_position', data=pos_out.astype(np.int32))
    if samples is not None:
        # Route through plain Python strings: zarr-backed sample arrays
        # arrive as numpy.StringDType (variable-length) which can't be
        # cast to fixed-width 'U<n>' without specifying the size; the
        # list detour lets numpy pick the width itself.
        dst.create_array(
            'sample_id',
            data=np.array([str(s) for s in samples], dtype='U'),
        )
    dst.create_array('contig_id', data=np.array([contig], dtype='U'))
    dst.create_array('variant_contig', data=np.zeros(n_out, dtype=np.int8))


def write_allel(zarr_path, gt, positions, samples=None):
    """Write genotype data in scikit-allel format.

    Parameters
    ----------
    zarr_path : str
        Output zarr store path.
    gt : ndarray, shape (n_variants, n_samples, ploidy)
        Genotype array.
    positions : ndarray, shape (n_variants,)
        Variant positions.
    samples : list of str, optional
        Sample names.
    """
    import zarr
    store = zarr.open(zarr_path, mode='w')
    store.create_array('calldata/GT', data=np.asarray(gt))
    store.create_array('variants/POS', data=np.asarray(positions))
    if samples is not None:
        store.create_array('samples', data=np.array(samples, dtype='U'))


def read_genotypes(path, region=None):
    """Auto-detect zarr layout and read genotype data.

    Parameters
    ----------
    path : str
        Path to zarr store directory.
    region : str, optional
        Genomic region string; see ``parse_region`` for the accepted
        forms.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples'.
    """
    import zarr
    store = zarr.open(path, mode='r')
    layout = detect_zarr_layout(store)
    if layout == 'vcz':
        return read_genotypes_vcz(store, region)
    elif layout == 'scikit-allel-grouped':
        return read_genotypes_allel_grouped(store, region)
    else:
        return read_genotypes_allel(store, region)


def vcf_to_zarr(vcf_paths, zarr_path, worker_processes=None,
                icf_path=None, max_memory='4GB', show_progress=True,
                zarr_format=None):
    """Convert VCF file(s) to VCZ-format zarr store using bio2zarr.

    Uses the two-step explode + encode pipeline for control over
    intermediate file placement.

    Parameters
    ----------
    vcf_paths : str or list of str
        Path(s) to VCF/BCF files (bgzipped + indexed).
    zarr_path : str
        Output zarr store path.
    worker_processes : int, optional
        Number of worker processes. Defaults to os.cpu_count().
    icf_path : str, optional
        Directory for intermediate columnar format files.
        Defaults to ``<zarr_path>.icf_tmp`` (same filesystem as output).
        ICF files can be 1-3x the VCF size.
    max_memory : str
        Maximum memory for encoding step. Default '4GB'.
    show_progress : bool
        Show progress bars. Default True.
    zarr_format : int, optional
        Zarr metadata version to write (2 or 3). ``None`` (default)
        uses bio2zarr's own default (v2). Pass ``3`` to write a v3
        store, which the kvikio GPU-decode backend can read; a v2
        store decodes on the CPU.
    """
    from bio2zarr.vcf import explode, encode

    if isinstance(vcf_paths, str):
        vcf_paths = [vcf_paths]
    if worker_processes is None:
        worker_processes = os.cpu_count() or 1

    if icf_path is None:
        icf_path = zarr_path + '.icf_tmp'

    try:
        explode(icf_path, vcf_paths,
                worker_processes=worker_processes,
                show_progress=show_progress)
        encode(icf_path, zarr_path,
               worker_processes=worker_processes,
               max_memory=max_memory,
               show_progress=show_progress,
               zarr_format=zarr_format)
    finally:
        try:
            if os.path.exists(icf_path):
                shutil.rmtree(icf_path)
        except OSError:
            pass
