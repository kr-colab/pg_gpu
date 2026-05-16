"""Zarr I/O utilities for VCZ (bio2zarr) and scikit-allel formats."""

import os
import shutil
import sys

import numpy as np


def _parse_region(region):
    """Parse 'chrom:start-end' into (chrom, start, end)."""
    chrom, coords = region.split(':')
    start, end = [int(x) for x in coords.split('-')]
    return chrom, start, end


def resolve_pop_file_path(zarr_path, pop_file, *, announce_prefix):
    """Map a ``pop_file`` kwarg that is *only ever a path or auto-load*
    to a filesystem path or ``None``.

    ``pop_file=False`` disables the auto-load and returns ``None``.
    ``pop_file=<str>`` returns the path as-is. ``pop_file=None`` looks
    for ``<zarr_path>.pops.tsv`` next to the store; if it exists,
    announces the auto-load to stderr via ``announce_prefix`` and
    returns the companion path.

    For the richer ``pop_file`` accepted by ``HaplotypeMatrix.from_zarr``
    (numpy arrays, dicts, zarr key names) see ``normalize_pop_input``.
    """
    if pop_file is False:
        return None
    if pop_file is not None:
        return pop_file
    companion = str(zarr_path).rstrip("/") + ".pops.tsv"
    if not os.path.exists(companion):
        return None
    print(f"{announce_prefix}: auto-loaded pop file {companion}",
          file=sys.stderr, flush=True)
    return companion


def normalize_pop_input(pop_file, *, zarr_path, sample_names,
                         zarr_store=None, announce_prefix=""):
    """Normalize the flexible ``pop_file`` kwarg into a ``{sample_id ->
    pop_label}`` dict (or ``None`` to mean "no assignments").

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

    if pop_file is False:
        return None

    if pop_file is None:
        companion = str(zarr_path).rstrip("/") + ".pops.tsv"
        if not os.path.exists(companion):
            return None
        print(f"{announce_prefix}: auto-loaded pop file {companion}",
              file=sys.stderr, flush=True)
        pop_file = companion

    if isinstance(pop_file, dict):
        return {str(k): str(v) for k, v in pop_file.items() if v}

    if isinstance(pop_file, (list, tuple, np.ndarray)):
        labels = np.asarray(pop_file)
        if labels.ndim != 1:
            raise ValueError(
                f"pop_file array must be 1-D; got shape {labels.shape}")
        if len(labels) != len(sample_names):
            raise ValueError(
                f"pop_file array length {len(labels)} does not match "
                f"sample axis length {len(sample_names)}")
        return _pop_array_to_map(labels, sample_names)

    if isinstance(pop_file, str):
        if zarr_store is not None and pop_file in zarr_store:
            labels = np.asarray(zarr_store[pop_file][:])
            if labels.ndim != 1 or len(labels) != len(sample_names):
                raise ValueError(
                    f"zarr key {pop_file!r} has shape {labels.shape}; "
                    f"expected 1-D of length {len(sample_names)} to "
                    f"line up with the sample axis")
            return _pop_array_to_map(labels, sample_names)
        return _read_pop_tsv(pop_file)

    raise TypeError(
        f"pop_file must be a path, dict, array, zarr key, False, or "
        f"None; got {type(pop_file).__name__}")


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
        Genomic region 'chrom:start-end'.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples'.
    """
    if region is not None:
        chrom, start, end = _parse_region(region)
        contig_ids = list(np.array(store['contig_id']))
        if chrom not in contig_ids:
            raise ValueError(
                f"Contig '{chrom}' not found. Available: {contig_ids}"
            )
        contig_idx = contig_ids.index(chrom)
        contig_arr = np.array(store['variant_contig'])
        pos_arr = np.array(store['variant_position'])
        mask = (contig_arr == contig_idx) & (pos_arr >= start) & (pos_arr < end)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No variants in region {region}")
        gt = np.array(store['call_genotype'][indices])
        positions = pos_arr[indices]
    else:
        contig_arr = np.array(store['variant_contig'])
        unique_contigs = np.unique(contig_arr)
        if len(unique_contigs) > 1:
            contig_ids = list(np.array(store['contig_id']))
            names = [contig_ids[i] for i in unique_contigs]
            raise ValueError(
                f"Store contains {len(unique_contigs)} contigs: {names}. "
                "Specify region='chrom:start-end' to select one."
            )
        gt = np.array(store['call_genotype'])
        positions = np.array(store['variant_position'])

    samples = list(np.array(store['sample_id'])) if 'sample_id' in store else None
    return {'gt': gt, 'positions': positions, 'samples': samples}


def read_genotypes_allel(store, region=None):
    """Read genotype data from a scikit-allel format zarr store.

    Parameters
    ----------
    store : zarr.Group
        Opened scikit-allel zarr store.
    region : str, optional
        Genomic region 'chrom:start-end'.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples'.
    """
    positions = np.array(store['variants/POS'])
    gt = np.array(store['calldata/GT'])
    samples = list(np.array(store['samples'])) if 'samples' in store else None

    if region is not None:
        _, start, end = _parse_region(region)
        mask = (positions >= start) & (positions < end)
        positions = positions[mask]
        gt = gt[mask]
        if len(positions) == 0:
            raise ValueError(f"No variants in region {region}")

    return {'gt': gt, 'positions': positions, 'samples': samples}


def read_genotypes_allel_grouped(store, region):
    """Read genotype data from a chromosome-grouped scikit-allel store.

    Parameters
    ----------
    store : zarr.Group
        Opened zarr store with chromosome-level groups.
    region : str
        Genomic region 'chrom:start-end'. Required for grouped stores.

    Returns
    -------
    dict
        Keys: 'gt' (n_var, n_samples, ploidy), 'positions', 'samples'.
    """
    if region is None:
        available = [k for k in store if hasattr(store[k], 'keys')]
        raise ValueError(
            f"Grouped zarr store requires region='chrom:start-end'. "
            f"Available groups: {available}"
        )
    chrom, start, end = _parse_region(region)
    if chrom not in store:
        available = [k for k in store if hasattr(store[k], 'keys')]
        raise ValueError(
            f"Chromosome '{chrom}' not found. Available: {available}"
        )
    grp = store[chrom]
    positions = np.array(grp['variants/POS'])
    gt = np.array(grp['calldata/GT'])
    samples = list(np.array(grp['samples'])) if 'samples' in grp else None

    mask = (positions >= start) & (positions < end)
    positions = positions[mask]
    gt = gt[mask]
    if len(positions) == 0:
        raise ValueError(f"No variants in region {region}")

    return {'gt': gt, 'positions': positions, 'samples': samples}


def write_vcz(zarr_path, gt, positions, samples=None, contig_name=None,
              chunks=None):
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
    """
    import zarr
    store = zarr.open(zarr_path, mode='w')
    cg_kwargs = {} if chunks is None else {"chunks": chunks}
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
        ``"chrom:start-end"`` (or ``"start-end"`` on a flat store)
        restricting the conversion to a position range.
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

    # Resolve which chromosome group to read (grouped) or use the root
    # (flat); derive the contig label that goes into the output.
    if layout == 'scikit-allel-grouped':
        chrom_from_region = (region.split(':', 1)[0]
                              if region and ':' in region else None)
        contig = contig or chrom_from_region
        if contig is None:
            available = [k for k in src if hasattr(src[k], 'keys')]
            raise ValueError(
                f"Grouped allel store requires contig=... "
                f"Available: {available}"
            )
        src_grp = src[contig]
    else:
        src_grp = src
        if contig is None:
            contig = 'unknown'

    gt_arr = src_grp['calldata/GT']
    pos_arr = src_grp['variants/POS']
    n_var, n_samp, ploidy = gt_arr.shape

    if region:
        span = region.split(':', 1)[-1] if ':' in region else region
        lo, hi = (int(x) for x in span.split('-'))
        # Loading positions in full is fine -- they're int32, so even a
        # human-genome-scale chromosome (~10 M sites) is ~40 MB.
        pos_host = np.asarray(pos_arr[:])
        lo_idx = int(np.searchsorted(pos_host, lo, side='left'))
        hi_idx = int(np.searchsorted(pos_host, hi, side='left'))
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
        Genomic region 'chrom:start-end'.

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
                icf_path=None, max_memory='4GB', show_progress=True):
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
               show_progress=show_progress)
    finally:
        try:
            if os.path.exists(icf_path):
                shutil.rmtree(icf_path)
        except OSError:
            pass
