Quality-Aware Filtering: GQ, DP, MQ from VCF/VCZ
================================================

Packaged script: ``examples/vcf_qc_filter.py``

Run it from the repo root:

.. code-block:: bash

   pixi run python examples/vcf_qc_filter.py
   pixi run python examples/vcf_qc_filter.py --min-mq 40 --min-gq 25 --min-dp 12
   pixi run python examples/vcf_qc_filter.py --out /tmp/clean.vcz

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

This tutorial walks through that pipeline end to end on simulated data
(so the script needs no external VCF or VCZ to run). At the end of the
recipe section there's a code block showing the equivalent against a
real VCF -- the only line that changes is the loader.

What the script does
--------------------

1. Simulates a 1 Mb chromosome with msprime (30 diploid individuals).
2. Writes the result as a VCZ store and stamps in synthetic
   ``variant_MQ``, ``call_GQ``, and ``call_DP`` arrays. In a real
   workflow these come straight from bio2zarr converting a VCF that
   already carries those FORMAT / INFO fields; the injection here keeps
   the example self-contained.
3. Reopens the store with ``fields=['MQ', 'GQ', 'DP']`` so the arrays
   land on ``hm.fields``.
4. Summarizes each field with a single ``np.percentile`` call. The
   arrays are plain numpy, so any matplotlib / seaborn snippet works
   for plotting -- ``plt.hist(hm.fields['GQ'].ravel(), bins=50)`` and
   you have a quality-score distribution.
5. Builds a per-variant boolean mask from ``MQ`` and a per-genotype
   boolean mask from ``GQ`` and ``DP``, then runs ``hm.filter`` to
   produce a clean matrix. ``drop_all_missing=True`` (the default)
   also drops any variant whose every call was set to ``-1`` by the
   per-genotype mask.
6. Writes the filtered matrix back out via ``to_zarr``. The surviving
   QC arrays land in the new store under the same field names, so a
   reload with ``fields=`` returns them unchanged.

A typical run on the default settings looks like:

.. code-block:: text

   31 segregating sites
   hm.fields keys: ['DP', 'GQ', 'MQ']
   Field summaries (5 / 25 / 50 / 75 / 95 percentiles):
       MQ: shape=(31,),    5/25/50/75/95% = 18.14 / 28.67 / 41.00 / 49.88 / 60.11
       GQ: shape=(31, 30), 5/25/50/75/95% = 4.00 / 25.00 / 49.00 / 75.00 / 94.00
       DP: shape=(31, 30), 5/25/50/75/95% = 2.00 / 10.00 / 19.00 / 29.00 / 38.00
   variants kept  (MQ >= 35.0):                       21 / 31   (67.7 %)
   genotypes masked (GQ >= 20 & DP >= 10):           359 / 930  (38.6 %)
   21 variants survive both filters
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

