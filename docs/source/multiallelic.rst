Multiallelic Sites
==================

Most sites in a real dataset have two alleles: the reference base and
one alternate. A few have three or four. Those are called
**multiallelic** sites, and they need a little care.

pg_gpu handles them with one rule: **every allele at a site is counted
on its own**, as if each alternate were a separate site. This is how
``tskit`` defines its statistics, so each function on this page gives
the answer ``tskit`` would. The few statistics that are only defined
for two alleles do not bend the rule. They drop the extra sites and
tell you how many with a ``BiallelicOnlyWarning``.

pg_gpu stores each haplotype's allele as an integer. ``0``
means the reference base, ``1`` the first alternate, ``2`` the second,
and so on. Missing data is ``-1``.

Each allele is counted on its own
---------------------------------

Suppose a site has reference ``A`` and two alternates, ``T`` and ``C``.
Across 10 haplotypes you observe 6 A's, 3 T's, and 1 C.

One way to handle this is to lump the alternates together and say "4 of
10 haplotypes are non-reference". **pg_gpu does not do that.** It keeps
T and C apart and counts each separately: T is at 3/10, C is at 1/10.

Lumping them would be misleading, because T and C are two different
mutations with two different histories. Treating them as one allele at
frequency 4/10 describes a variant that does not exist. Counting them
separately is also what ``tskit`` does, so :math:`\pi`, the theta
estimators, Tajima's D, Fay-Wu's H, Zeng's E, and every SFS function
give the same answer here that ``tskit`` would.

Two things follow from this.

**A three-allele site counts as two mutations.** It takes two mutations
to produce three different alleles, so that site adds 2 to
``segregating_sites`` rather than 1. In general a site with :math:`k`
alleles adds :math:`k - 1`. ``theta_w`` and Tajima's D use the same
count. On ordinary two-allele data this is just the number of variable
sites, as you would expect. On multiallelic data it is larger -- and
larger than what scikit-allel reports, which counts variable sites
rather than mutations.

**Frequencies are per allele.** For the site above, the frequency
spectrum gets one entry at 3 and another at 1 -- not a single entry at
4. Likewise ``max_daf`` reports the most common alternate (3/10 here),
and ``daf_histogram`` records one frequency for T and one for C.

If your data is entirely two-allele, none of this changes anything. A
site with one alternate has nothing to keep apart, so these rules give
the familiar answers.

.. _sfs-conventions:

Three SFS rules that also affect two-allele data
------------------------------------------------

The frequency-spectrum functions follow ``tskit`` on three points.
Unlike everything above, these change the output even when every site
has just two alleles, so they are worth reading if you are comparing
against older results or against another tool.

* **Sites where nothing varies are left out.** If every haplotype
  carries the same allele, the site contributes nothing -- not to the
  first bin, not to the last. This changes ``sfs`` and
  ``daf_histogram`` whenever invariant sites are in your matrix. No
  theta estimator changes, because none of them uses those two bins.

* **Folded spectra are returned as floating point numbers** (``float64``) rather than
  integers. On two-allele data the values still come out whole, so
  only the array's type changes. On a multiallelic site each allele is
  folded on its own and contributes one half, so you will see genuine
  half-counts there. For example, a site with three alleles at counts
  3, 2, and 1 out of 6 folds to ``0.5`` in each of three bins.

* **``joint_sfs_folded`` folds the site once, not each population
  separately.** The site is folded using its overall minor allele, so
  every cell in the result refers to one specific allele.
  scikit-allel folds each population's axis on its own, which can put
  two different alleles into the same cell. The result is a full
  ``(n1 + 1, n2 + 1)`` grid rather than the smaller
  ``(n1 // 2 + 1, n2 // 2 + 1)``.

Which value to trust on multiallelic data
-----------------------------------------

The plain functions -- ``diversity.pi``, ``diversity.tajimas_d``, and
friends -- are the ones to trust. They match ``tskit`` exactly.

``FrequencySpectrum`` also offers :math:`\pi`, Tajima's D, and Fay-Wu's
H, but on multiallelic data these are close approximations rather than
exact. The reason is that a frequency spectrum records *how many* sites
sit at each frequency, but not *which alleles came from the same site*,
and you need that grouping to get :math:`\pi` exactly right. On
two-allele data the two routes agree exactly.

PCA and local PCA keep one column per allele
--------------------------------------------

``pca``, ``local_pca``, ``lostruct``, and ``local_pca_jackknife`` take a
``HaplotypeMatrix`` and give every allele its own column, so they work
on multiallelic sites with no filtering. Allele numbers are arbitrary
labels, and this is what keeps the result independent of how the
alleles were numbered.

One consequence: ``local_pca`` and ``lostruct`` no longer match the R
``lostruct`` package number for number. R works on genotype dosages,
one column per site, and the two center the data differently, which
shifts the scale of the eigenvalues. The directions the components
point in still agree closely (correlations above 0.999 in the test
suite), so window-to-window comparisons, the MDS plot, and outlier
detection behave the same. ``pca_dosage`` is the classical diploid
version and takes a ``GenotypeMatrix``, which only holds two-allele
sites. See :doc:`features` for more on the two PCAs.

Statistics that only work on two-allele sites
---------------------------------------------

A few statistics are only defined for sites with two alleles. When one
of them throws multiallelic sites away, it tells you how many with a
``BiallelicOnlyWarning``, so you do not mistake a partial answer for a
complete one.

That means ``admixture.patterson_d`` and the
``GenotypeMatrix`` loaders described below. If you want a statistic like
Patterson's D that does handle multiallelic sites, use
``admixture.patterson_f4``, which measures the same thing without the
restriction, at the expense of normalization (it is not bounded in [-1, 1]).

To turn the warning off:

.. code-block:: python

   import warnings
   from pg_gpu import BiallelicOnlyWarning

   warnings.filterwarnings("ignore", category=BiallelicOnlyWarning)

There is a second, unrelated warning called ``MultiallelicCapWarning``.
It does *not* mean a statistic is two-allele-only. It means a windowed
calculation that normally handles multiallelic sites just fine hit its
internal limit of 8 alleles at one site and discarded the site. DNA has 4 bases, so you will
almost certainly never see it.

Some other statistics are two-allele-only but never have to drop
anything, because the restriction was already applied when the data was
loaded. ``relatedness.grm``, ``relatedness.ibs``, the dosage PCA, and
``distance_stats.pairwise_diffs_diploid`` all take a ``GenotypeMatrix``,
which only ever holds two-allele sites. The LD statistics apply their
own filter.

What the ``GenotypeMatrix`` loaders keep
----------------------------------------

``GenotypeMatrix.from_vcf``, ``GenotypeMatrix.from_zarr``, and
``GenotypeMatrix.from_haplotype_matrix`` all agree on what counts as a
two-allele site, so the same data gives you the same matrix no matter
which one you use.

* **A site is kept if at most two different alleles actually show up in
  the data.** What matters is how many distinct alleles are present,
  not which numbers they happen to be. A site where you only see
  alleles ``0`` and ``2`` is kept. So is a site showing only ``1`` and
  ``2``, which can happen after you subset to a few samples and the
  reference allele disappears.

* **The genotype value is a count of the alternate allele**, giving the
  usual 0, 1, or 2 per individual. Allele ``0`` is the reference and the
  alternate is the highest-numbered allele present.

* **Sites with three or more alleles cannot be written as a 0/1/2
  count**, so they are removed. ``from_vcf`` drops the site outright;
  the zarr loaders instead mark the whole site as missing (``-1``),
  which keeps the rows lined up for the streaming reader. Either way
  you get a ``BiallelicOnlyWarning`` with the count.

* **Sites with no variation (a single present allele) are kept.** Loaders load your data as it
  is; throwing away uninformative sites is
  ``restrict_to_segregating``'s job, not theirs.

Ordinary data using only alleles ``0`` and ``1`` is unaffected by all of
this and loads without any warning.

Restricting to biallelic or segregating sites yourself
---------------------------------------------------------

Two orthogonal filters are available. ``restrict_to_biallelic()`` (on a
``HaplotypeMatrix``) keeps sites with at most two distinct present alleles,
leaving the allele codes unchanged -- so ``{0,1}``, ``{0,2}`` and
reference-absent ``{1,2}`` sites are all retained -- and drops sites with three
or more alleles without warning. The LD statistics are defined on biallelic
sites and apply it themselves, warning when they do, so you rarely need to call
it directly:

.. code-block:: python

   h = h.restrict_to_biallelic()      # drop >=3-allele sites, codes unchanged

``restrict_to_segregating()`` (on either matrix) is unrelated to allele count: it
drops sites where nothing varies, keeping those with at least two distinct
alleles present:

.. code-block:: python

   gm = gm.restrict_to_segregating()  # drop sites that aren't variable

