API Reference
=============

HaplotypeMatrix
---------------

.. autoclass:: pg_gpu.HaplotypeMatrix
   :members:
   :undoc-members:
   :show-inheritance:

Missing Data Methods
~~~~~~~~~~~~~~~~~~~~

.. automethod:: pg_gpu.HaplotypeMatrix.is_missing
.. automethod:: pg_gpu.HaplotypeMatrix.is_called
.. automethod:: pg_gpu.HaplotypeMatrix.count_missing
.. automethod:: pg_gpu.HaplotypeMatrix.count_called
.. automethod:: pg_gpu.HaplotypeMatrix.get_span
.. automethod:: pg_gpu.HaplotypeMatrix.filter_variants_by_missing
.. automethod:: pg_gpu.HaplotypeMatrix.summarize_missing_data

LD Methods
~~~~~~~~~~

.. automethod:: pg_gpu.HaplotypeMatrix.pairwise_r2
.. automethod:: pg_gpu.HaplotypeMatrix.locate_unlinked
.. automethod:: pg_gpu.HaplotypeMatrix.windowed_r_squared

LD Statistics
-------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd

.. autofunction:: pg_gpu.ld_statistics.dz

.. autofunction:: pg_gpu.ld_statistics.pi2

.. autofunction:: pg_gpu.ld_statistics.r

.. autofunction:: pg_gpu.ld_statistics.r_squared

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.ld_statistics.dd_within

.. autofunction:: pg_gpu.ld_statistics.dd_between

.. autofunction:: pg_gpu.ld_statistics.compute_ld_statistics

.. autofunction:: pg_gpu.ld_statistics.zns

.. autofunction:: pg_gpu.ld_statistics.omega

.. autofunction:: pg_gpu.ld_statistics.mu_ld

Population Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

* Single population: ``populations=None`` or omit
* Two populations: ``populations=(0, 1)`` for DD
* Three indices: ``populations=(0, 0, 1)`` for Dz
* Four indices: ``populations=(0, 0, 1, 1)`` for pi2

Diversity Statistics
--------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.pi
.. autofunction:: pg_gpu.diversity.theta_w
.. autofunction:: pg_gpu.diversity.tajimas_d
.. autofunction:: pg_gpu.diversity.haplotype_diversity
.. autofunction:: pg_gpu.diversity.allele_frequency_spectrum
.. autofunction:: pg_gpu.diversity.segregating_sites
.. autofunction:: pg_gpu.diversity.singleton_count
.. autofunction:: pg_gpu.diversity.fay_wus_h
.. autofunction:: pg_gpu.diversity.heterozygosity_expected
.. autofunction:: pg_gpu.diversity.heterozygosity_observed
.. autofunction:: pg_gpu.diversity.inbreeding_coefficient

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.diversity.diversity_stats
.. autofunction:: pg_gpu.diversity.neutrality_tests
.. autofunction:: pg_gpu.diversity.theta_h
.. autofunction:: pg_gpu.diversity.max_daf
.. autofunction:: pg_gpu.diversity.haplotype_count
.. autofunction:: pg_gpu.diversity.daf_histogram
.. autofunction:: pg_gpu.diversity.diplotype_frequency_spectrum
.. autofunction:: pg_gpu.diversity.mu_var
.. autofunction:: pg_gpu.diversity.mu_sfs

Divergence Statistics
---------------------

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.fst
.. autofunction:: pg_gpu.divergence.fst_hudson
.. autofunction:: pg_gpu.divergence.fst_weir_cockerham
.. autofunction:: pg_gpu.divergence.fst_nei
.. autofunction:: pg_gpu.divergence.dxy
.. autofunction:: pg_gpu.divergence.da

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.divergence.divergence_stats
.. autofunction:: pg_gpu.divergence.pairwise_fst
.. autofunction:: pg_gpu.divergence.pbs

Selection Scan Statistics
-------------------------

Haplotype-Based Selection Scans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.ihs
.. autofunction:: pg_gpu.selection.xpehh
.. autofunction:: pg_gpu.selection.nsl
.. autofunction:: pg_gpu.selection.xpnsl

EHH and Haplotype Homozygosity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.ehh_decay
.. autofunction:: pg_gpu.selection.garud_h
.. autofunction:: pg_gpu.selection.moving_garud_h

Standardization
~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.selection.standardize
.. autofunction:: pg_gpu.selection.standardize_by_allele_count

Site Frequency Spectrum
-----------------------

Single Population
~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.sfs
.. autofunction:: pg_gpu.sfs.sfs_folded
.. autofunction:: pg_gpu.sfs.sfs_scaled
.. autofunction:: pg_gpu.sfs.sfs_folded_scaled

Joint (Two-Population)
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.joint_sfs
.. autofunction:: pg_gpu.sfs.joint_sfs_folded
.. autofunction:: pg_gpu.sfs.joint_sfs_scaled
.. autofunction:: pg_gpu.sfs.joint_sfs_folded_scaled

Scaling and Folding Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.sfs.scale_sfs
.. autofunction:: pg_gpu.sfs.scale_sfs_folded
.. autofunction:: pg_gpu.sfs.scale_joint_sfs
.. autofunction:: pg_gpu.sfs.scale_joint_sfs_folded
.. autofunction:: pg_gpu.sfs.fold_sfs
.. autofunction:: pg_gpu.sfs.fold_joint_sfs

Admixture and F-Statistics
--------------------------

Per-Variant Statistics
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.patterson_f2
.. autofunction:: pg_gpu.admixture.patterson_f3
.. autofunction:: pg_gpu.admixture.patterson_d

Windowed Statistics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.moving_patterson_f3
.. autofunction:: pg_gpu.admixture.moving_patterson_d

Block-Jackknife Averaged
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.admixture.average_patterson_f3
.. autofunction:: pg_gpu.admixture.average_patterson_d

Dimensionality Reduction and Distance
--------------------------------------

PCA
~~~

.. autofunction:: pg_gpu.decomposition.pca
.. autofunction:: pg_gpu.decomposition.randomized_pca

Distance
~~~~~~~~

.. autofunction:: pg_gpu.decomposition.pairwise_distance
.. autofunction:: pg_gpu.decomposition.pcoa

Windowed Statistics (GPU-Native)
--------------------------------

These functions compute statistics across all genomic windows in a single
GPU pass, avoiding Python loops over windows.

.. autofunction:: pg_gpu.windowed_analysis.windowed_statistics

.. autofunction:: pg_gpu.windowed_analysis.windowed_statistics_fused

GenotypeMatrix
--------------

.. autoclass:: pg_gpu.GenotypeMatrix
   :members:
   :undoc-members:

Distance Distribution Statistics
---------------------------------

.. autofunction:: pg_gpu.distance_stats.pairwise_diffs
.. autofunction:: pg_gpu.distance_stats.dist_moments
.. autofunction:: pg_gpu.distance_stats.dist_var
.. autofunction:: pg_gpu.distance_stats.dist_skew
.. autofunction:: pg_gpu.distance_stats.dist_kurt

Visualization
-------------

Site Frequency Spectrum
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_sfs
.. autofunction:: pg_gpu.plotting.plot_joint_sfs

Linkage Disequilibrium
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_pairwise_ld
.. autofunction:: pg_gpu.plotting.plot_ld_decay

PCA and Population Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_pca
.. autofunction:: pg_gpu.plotting.plot_pairwise_distance

Windowed Statistics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_windowed
.. autofunction:: pg_gpu.plotting.plot_windowed_panel

Haplotype Structure
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pg_gpu.plotting.plot_haplotype_frequencies
.. autofunction:: pg_gpu.plotting.plot_variant_locator
