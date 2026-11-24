Polarization
============

pg_gpu reads allele code 0 as the ancestral allele and every other code as
derived. A VCF gives code 0 to REF, and REF is not the ancestral allele in
general. A statistic of a derived state on such data is wrong with no sign
of it, so a ``HaplotypeMatrix`` carries a flag, ``polarized``, that records
whether code 0 is the ancestral allele at every site.

.. list-table::
   :header-rows: 1
   :widths: 45 25

   * - How the matrix was made
     - ``polarized``
   * - ``HaplotypeMatrix.from_vcf``
     - ``False``
   * - ``HaplotypeMatrix.from_zarr`` (eager and streaming)
     - the flag that ``to_zarr`` saved; ``False`` for any other store
   * - ``HaplotypeMatrix(haplotypes, positions, ...)``
     - ``False``
   * - ``from_vcf`` or ``from_zarr`` with ``ancestral=``
     - ``True``
   * - ``HaplotypeMatrix.polarize``
     - ``True``
   * - ``HaplotypeMatrix.from_ts``
     - ``True``
   * - ``GenotypeMatrix.to_haplotype_matrix``
     - ``False``

Subsets, population slices and filtered copies inherit the flag.

Making a matrix polarized
-------------------------

.. code-block:: python

   from pg_gpu import HaplotypeMatrix

   # Recode by an ancestral-allele INFO tag at load time
   hm = HaplotypeMatrix.from_vcf("data.vcf.gz", ancestral="AA")
   hm = HaplotypeMatrix.from_zarr("data.vcz", ancestral="AA")

   # Recode a loaded matrix by an array of ancestral alleles or codes
   hm = hm.polarize(ancestral_codes)

   # REF already is the ancestral allele (the VCF was polarized before):
   # declare it; no allele code changes
   hm = HaplotypeMatrix.from_vcf("polarized.vcf.gz", polarized=True)
   hm.set_polarized()

A recode drops the sites with no known ancestral allele and emits an
``UnknownAncestralWarning`` with their count. A recode of a matrix that is
marked polarized raises if it would move code 0, so a matrix is not
polarized twice by mistake.

Which statistics read the derived state
---------------------------------------

These emit ``UnpolarizedDataWarning`` on a matrix that is not marked
polarized. Their value changes when the ancestral and a derived allele
change places.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Functions
   * - ``diversity``
     - ``theta_h``, ``theta_l``, ``fay_wus_h``, ``normalized_fay_wus_h``,
       ``zeng_e``, ``zeng_dh``, ``max_daf``, ``daf_histogram``,
       ``singleton_count``, ``neutrality_tests``
   * - ``diversity.diversity_stats``
     - when asked for ``theta_h``, ``theta_l``, ``fay_wus_h``,
       ``normalized_fay_wus_h``, ``zeng_e``, ``zeng_dh``, ``singletons``,
       ``eta1`` or ``minus_eta1``
   * - ``diversity.FrequencySpectrum``
     - its spectrum (``sfs``), and ``theta`` with a weight that is not
       symmetric (``theta_h``, ``theta_l``, ``eta1``, ``minus_eta1``) or
       with a function of the caller, which is taken to be one; once for
       each object
   * - ``sfs``
     - ``sfs``, ``sfs_scaled``, ``joint_sfs``, ``joint_sfs_scaled``,
       ``project_joint_sfs``
   * - ``selection``
     - ``ihs``, ``nsl``
   * - ``relatedness``
     - ``genetic_relatedness`` with ``polarised=True``
   * - ``windowed_analysis``, ``WindowedAnalyzer`` and the fused engines
     - when asked for ``theta_h``, ``theta_l``, ``fay_wu_h``,
       ``normalized_fay_wu_h``, ``zeng_e``, ``zeng_dh``, ``max_daf``,
       ``daf_hist``, ``singletons``, ``n_singletons`` or ``mean_nsl``

One call warns once, whatever the number of windows or chunks. A
statistic of your own that you pass to ``windowed_analysis`` warns as it
does when you call it yourself.

Everything else gives the same value either way: ``pi``, ``theta_w``,
Tajima's D, the folded spectra, FST, dxy, the LD statistics, Garud's H,
``xpehh``, PCA and the F-statistics. The sign of the raw ``D`` from
``pairwise_LD_v`` follows the allele codes, as it does for any labelling;
it does not warn.

Multiallelic sites
------------------

A few statistics count each derived allele of a site apart. At a biallelic
site they give the same value whichever allele has code 0, so they do not
warn; at a site with three or more alleles their value does depend on it:
``mu_sfs`` (scalar and windowed) and the ``eta1_star`` and
``minus_eta1_star`` estimators. ``genetic_relatedness`` with
``polarised=True`` is the same with centring, and it does warn, since
that argument asks for the derived state.

Genotype matrices
-----------------

A ``GenotypeMatrix`` has no such flag. ``daf_histogram`` on one counts the
allele that its dosage counts, with no warning.

To silence the warning:

.. code-block:: python

   import warnings
   from pg_gpu import UnpolarizedDataWarning
   warnings.filterwarnings("ignore", category=UnpolarizedDataWarning)
