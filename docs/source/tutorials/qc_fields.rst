Quality-Aware Filtering: GQ, DP, MQ from VCF/VCZ
================================================

Packaged script: ``examples/vcf_qc_filter.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/vcf_qc_filter.py
   pixi run python examples/vcf_qc_filter.py --min-ac 8 --min-aa-prob 0.95
   pixi run python examples/vcf_qc_filter.py --region X:8000000-12000000

Background
----------

Every routine call set comes with per-variant and per-genotype quality
annotations -- ``INFO/MQ``, ``INFO/QD``, ``FMT/GQ``, ``FMT/DP``, ``FMT/AD``,
and so on. 

bio2zarr preserves all those FORMAT and INFO fields when it
writes a VCZ; the arrays sit next to ``call_genotype`` on disk. The
``fields=`` kwarg on ``HaplotypeMatrix.from_vcf`` / ``from_zarr`` (and
the same on ``GenotypeMatrix``) surfaces them to Python; ``filter()``
applies the masks; ``to_zarr`` round-trips the survivors so the
resulting "clean" VCZ is self-describing and re-loadable with the same
``fields=`` set.

This tutorial walks through that pipeline end to end on the real
Anopheles X-chromosome VCZ that ships under ``examples/data/`` -- a
5.3 M-variant, 100-diploid store that bio2zarr preserved with all its
INFO and FORMAT arrays. The packaged script reads a 4 Mb window so a
single GPU run takes a couple of seconds.

What the script does
--------------------

1. Opens a 4 Mb window of ``examples/data/gamb.X.phased.n100.vcz`` with
   ``fields=['AC', 'AAProb', 'PQ']``. ``AC`` lands as per-variant
   ``(n_var,)``; ``AAProb`` is per-variant ``(n_var, 1)`` (bio2zarr's
   shape for INFO with ``Number=A``); ``PQ`` is per-genotype
   ``(n_var, n_samples)``. The reader auto-resolves which kind each is.
2. Summarizes each field with one ``np.percentile`` call. The arrays
   are plain numpy, so any matplotlib / seaborn snippet works for
   plotting -- e.g. ``plt.hist(hm.fields['AC'], bins=50)`` for the
   allele-count distribution.
3. Builds a per-variant mask from ``AC`` and ``AAProb`` -- drop
   singletons / doubletons (``AC >= 4``) and keep only confidently
   polarized sites (``AAProb >= 0.9``).
4. Constructs a per-genotype mask placeholder (``PQ >= -1``). The gamb
   fixture carries ``-1`` everywhere in ``call_PQ`` because the source
   VCF didn't populate phasing quality, so this mask is True for every
   call. On a VCZ derived from a VCF that DID carry ``FMT/GQ`` or
   ``FMT/DP`` the same line would read
   ``(hm.fields['GQ'] >= 20) & (hm.fields['DP'] >= 10)``.
5. Runs ``hm.filter`` and writes the survivor to a fresh VCZ. The
   surviving ``variant_*`` / ``call_*`` arrays are preserved in the
   clean store; a reload with the same ``fields=`` set returns them
   byte-exact.

A typical run on the default settings looks like:

.. code-block:: text

   Opening examples/data/gamb.X.phased.n100.vcz
     region: X:1000000-5000000
     fields: ['AC', 'AAProb', 'PQ']
     loaded 1,041,651 variants x 100 diploids

   Field summaries (5 / 25 / 50 / 75 / 95 percentiles):
           AC: shape=(1041651,),     5/25/50/75/95% = 0.000 / 0.000 / 0.000 / 1.000 / 4.000
       AAProb: shape=(1041651, 1),   5/25/50/75/95% = 0.991 / 1.000 / 1.000 / 1.000 / 1.000
           PQ: shape=(1041651, 100), 5/25/50/75/95% = -1.000 / -1.000 / -1.000 / -1.000 / -1.000

   Filtering: variants kept where AC >= 4 and AAProb >= 0.9
     variants kept: 52,826 / 1,041,651 (5.1 %)
     after filter: 52,826 variants
   Round-trip OK -- reloaded fields match the filtered matrix.

Recipe
------

The whole pipeline reduces to four method calls:

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   # 1. Load with quality fields. Bare VCF tags; pg_gpu auto-resolves
   #    INFO vs FORMAT from the VCF header (or zarr layout).
   hm = HaplotypeMatrix.from_zarr(
       "cohort.vcz", fields=["MQ", "QD", "GQ", "DP"], streaming="never",
   )

   # 2. Inspect. Per-variant fields are (n_var,); per-genotype are
   #    (n_var, n_samples). Shape tells you which is which.
   hm.fields["MQ"]   # shape (n_var,)
   hm.fields["GQ"]   # shape (n_var, n_samples)

   # 3. Filter. ``variants`` drops rows; ``genotypes`` sets cells to -1
   #    on the haplotype matrix (both allele rows for each sample).
   hm_clean = hm.filter(
       variants=(hm.fields["MQ"] >= 40.0) & (hm.fields["QD"] >= 2.0),
       genotypes=(hm.fields["GQ"] >= 20) & (hm.fields["DP"] >= 10),
       drop_all_missing=True,
   )

   # 4. Persist. The surviving QC arrays round-trip into the new VCZ
   #    under the same field names.
   hm_clean.to_zarr("cohort.clean.vcz", format="vcz", contig_name="chr1")

The same four lines work against a VCF source -- swap the loader:

.. code-block:: python

   hm = HaplotypeMatrix.from_vcf(
       "cohort.vcf.gz", fields=["MQ", "QD", "GQ", "DP"],
   )
   # ... filter + to_zarr same as above

For very large VCFs the recommended path is "encode once via
``pg_gpu.zarr_io.vcf_to_zarr`` (bio2zarr under the hood, which
preserves every FORMAT and INFO field by default), then iterate on
thresholds against the VCZ." Re-encoding a multi-GB VCF dwarfs every
other step in the workflow; doing it once and tuning filters
read-side is a real time saver.

What gets read
--------------

For each bare tag, pg_gpu probes the source for the per-variant entry
first and falls back to per-genotype:

* **VCZ** (``bio2zarr`` output): ``variant_<tag>`` then ``call_<tag>``.
* **scikit-allel zarr**: ``variants/<tag>`` then ``calldata/<tag>``
  (works for both flat and chromosome-grouped layouts).
* **VCF** (``allel.read_vcf``): the INFO section first, then FORMAT,
  resolved via ``allel.read_vcf_headers``.

Tags missing from the source emit a ``UserWarning`` and are silently
dropped from ``hm.fields``. The shape of each returned array
disambiguates per-variant (``ndim=1``) from per-genotype (``ndim=2``),
so a single dict can hold both kinds without parallel namespaces.

When you load with ``region="chr1:1_000_000-2_000_000"``, the QC arrays
are sliced down to the same variant range as ``hm.haplotypes`` --
no manual realignment.

Filter semantics
----------------

``hm.filter(variants=..., genotypes=..., drop_all_missing=True)``:

* ``variants`` -- ``(n_var,)`` bool. False rows are dropped from every
  array on the returned matrix (haplotypes, positions, every entry in
  ``fields``).
* ``genotypes`` -- ``(n_var, n_samples)`` bool. False cells set the
  haplotype matrix to ``-1`` at that position (both allele rows for
  the same sample). ``fields`` arrays are *not* zeroed out -- the QC
  values stay accessible for downstream inspection.
* ``drop_all_missing`` -- after applying both masks, drop variants
  whose every call is ``-1``. Default ``True``; turn off when you want
  the genotype mask to keep variants with at least one valid call.

The returned matrix is a fresh allocation, not a view. ``samples``,
``sample_sets``, and ``chrom_start`` / ``chrom_end`` are propagated;
``n_total_sites`` and any attached accessibility mask are not (the
variant axis changed; if you need span normalization on the result,
re-attach an accessibility mask explicitly).

Current Limitations
---------------------

* The streaming opener (``streaming='always'``) does not yet accept
  ``fields=``; both raise ``NotImplementedError`` when combined. 
* ``to_zarr(format='scikit-allel')`` does not round-trip ``hm.fields``
  yet; combining the two raises so values are not silently dropped.
  Stick to ``format='vcz'`` (the default) for the clean output.

