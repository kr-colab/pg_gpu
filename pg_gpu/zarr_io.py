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
    """Map a ``pop_file`` kwarg to a filesystem path or None.

    ``pop_file=False`` disables the auto-load and returns None.
    ``pop_file=<str>`` returns the path as-is. ``pop_file=None`` looks
    for ``<zarr_path>.pops.tsv`` next to the store; if it exists,
    announces the auto-load to stderr via ``announce_prefix`` and
    returns the companion path.
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
