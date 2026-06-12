"""Detect haploid / hemizygous genotype encodings at load time.

pg_gpu's loaders assume diploid genotypes: every genotype call is split into
two haplotype rows (or summed into a 0/1/2 dosage). Haploid data violates that
assumption in two ways that otherwise pass silently:

* True haploid calls (``GT=1``) are padded by scikit-allel to a second ``-1``
  allele, so the second haplotype row of every sample becomes spurious missing
  data and the haplotype count is doubled.
* Pseudo-diploid homozygous calls (``GT=1/1``, the convention many callers emit
  for haploid regions such as the male X) load as two identical haplotypes per
  sample, inflating homozygosity and biasing the SFS, pi, and Tajima's D.

Neither is distinguishable from valid diploid data by array shape alone --
read_vcf returns a ploidy-2 layout in both cases -- so the checks here inspect
the allele values. The unambiguous case (a genuine haploid call is present)
raises; the ambiguous case (a sample homozygous at every variant it carries,
which is equally what an inbred diploid line looks like) warns.
"""
import warnings

import numpy as np

ISSUE_URL = "https://github.com/kr-colab/pg_gpu/issues/121"


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


def _format_sample_list(names, idx, limit=6):
    """Render up to ``limit`` flagged samples as a readable string."""
    flagged = [str(names[i]) for i in idx]
    head = ", ".join(flagged[:limit])
    if len(flagged) > limit:
        head += f", ... (+{len(flagged) - limit} more)"
    return head


def _require_diploid_ploidy(ploidy, source):
    """Raise unless the genotype ploidy axis is 2.

    Factored out so the streaming reader (which knows the ploidy axis from
    store metadata, without reading any genotypes) and the value-based
    detector share one message and one tracking link.
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

    # scikit-allel pads a true haploid call (GT=1) as [allele, -1]: the first
    # allele present, the second missing, never the reverse. A partially
    # missing diploid call (GT=1/.) produces the same [allele, -1] shape, so
    # the padding shape alone is not proof. A sample is haploid only if its
    # first allele appears somewhere yet its second allele is missing at every
    # site -- any fully called or reverse half-call genotype (which padding can
    # never produce) means the second allele was called and rules it out.
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

    # Ambiguous case: the data are polymorphic but no genotype is ever
    # heterozygous. True for a hemizygous sample set encoded as pseudo-diploid
    # homozygotes (e.g. a male-only X chromosome), and also for a fully inbred
    # panel -- so warn rather than raise. Checked dataset-wide because a single
    # real heterozygote anywhere rules the encoding out, which keeps this from
    # firing on ordinary diploid data where some individuals are incidentally
    # homozygous throughout a small region.
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
