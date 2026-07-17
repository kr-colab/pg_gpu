"""Parity tests for the diploid Patterson/GCTA PCA (pca_dosage).

pca_dosage / randomized_pca_dosage standardize biallelic diploid dosages the way
scikit-allel does (center by the per-variant mean, scale by sqrt(p (1 - p)) with
p = mean / 2), then eigendecompose the individual-by-individual Gram. These tests
pin that against allel.pca, check the randomized approximation, the ploidy-1/2
type split (GenotypeMatrix only), and the biallelic requirement.
"""
import numpy as np
import pytest
import allel

from pg_gpu import (GenotypeMatrix, HaplotypeMatrix, pca_dosage,
                    randomized_pca_dosage)


def _polymorphic_gm(seed=0, n_ind=30, n_var=200):
    """Biallelic diploid dosages with every site polymorphic (a 0 and a 2 forced
    per variant), so scikit-allel's unguarded 1/sqrt(p(1-p)) never divides by 0."""
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 3, (n_ind, n_var)).astype(np.int8)
    geno[0, :] = 0
    geno[1, :] = 2
    return GenotypeMatrix(geno.copy(), np.arange(n_var) * 100), geno


class TestPCADosageParity:

    def test_pca_dosage_matches_allel(self):
        gm, geno = _polymorphic_gm(seed=3)
        coords, evr = pca_dosage(gm, n_components=6)
        # scikit-allel takes gn as (n_variants, n_samples)
        acoords, model = allel.pca(geno.T.astype('f8'), n_components=6,
                                   scaler='patterson', ploidy=2)
        np.testing.assert_allclose(evr, model.explained_variance_ratio_,
                                   atol=1e-8, rtol=1e-6)
        # coordinates equal up to a per-component sign flip
        for j in range(6):
            sign = np.sign(np.dot(coords[:, j], acoords[:, j]))
            np.testing.assert_allclose(coords[:, j], sign * acoords[:, j],
                                       atol=1e-6, rtol=1e-5)

    def test_randomized_pca_dosage_matches_pca_dosage(self):
        # Three groups at distinct frequencies give a well-separated leading
        # 2D subspace where the randomized approximation is accurate. Compare via
        # canonical correlations so within-subspace rotation is not penalized.
        rng = np.random.RandomState(5)
        n_ind, n_var = 45, 500
        groups = [slice(0, 15), slice(15, 30), slice(30, 45)]
        freqs = [0.2, 0.5, 0.8]
        geno = np.empty((n_ind, n_var), dtype=np.int8)
        for v in range(n_var):
            for g, pg in zip(groups, freqs):
                geno[g, v] = rng.binomial(2, pg, 15)
        gm = GenotypeMatrix(geno, np.arange(n_var) * 100)
        coords, evr = pca_dosage(gm, n_components=2)
        rcoords, revr = randomized_pca_dosage(gm, n_components=2, n_iter=7,
                                              random_state=0)
        np.testing.assert_allclose(revr, evr, atol=1e-3, rtol=1e-2)
        qa, _ = np.linalg.qr(coords)
        qb, _ = np.linalg.qr(rcoords)
        canonical = np.linalg.svd(qa.T @ qb, compute_uv=False)
        assert canonical.min() > 0.999

    def test_population_subset(self):
        rng = np.random.RandomState(7)
        geno = rng.randint(0, 3, (20, 150)).astype(np.int8)
        geno[0, :] = 0
        geno[1, :] = 2
        sets = {'A': list(range(10)), 'B': list(range(10, 20))}
        gm = GenotypeMatrix(geno.copy(), np.arange(150) * 100, sample_sets=sets)
        coords, _ = pca_dosage(gm, n_components=3, population='A')
        assert coords.shape == (10, 3)


class TestPCADosageTypeGuards:

    def test_pca_dosage_rejects_haplotype_matrix(self):
        hm = HaplotypeMatrix(np.random.RandomState(0).randint(0, 2, (10, 50)).astype(np.int8),
                             np.arange(50) * 100, 0, 5000)
        with pytest.raises(TypeError, match="HaplotypeMatrix"):
            pca_dosage(hm)

    def test_randomized_pca_dosage_rejects_haplotype_matrix(self):
        hm = HaplotypeMatrix(np.random.RandomState(0).randint(0, 2, (10, 50)).astype(np.int8),
                             np.arange(50) * 100, 0, 5000)
        with pytest.raises(TypeError, match="HaplotypeMatrix"):
            randomized_pca_dosage(hm)

    def test_pca_dosage_rejects_non_biallelic(self):
        # A dosage > 2 cannot arise from a biallelic diploid site.
        geno = np.random.RandomState(0).randint(0, 3, (10, 50)).astype(np.int8)
        geno[0, 0] = 3
        gm = GenotypeMatrix(geno, np.arange(50) * 100)
        with pytest.raises(ValueError, match="biallelic"):
            pca_dosage(gm)
