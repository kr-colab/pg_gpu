"""GenotypeMatrix property setters reset the accessible-mask filter cache."""
import cupy as cp
import numpy as np

from pg_gpu import GenotypeMatrix


def _masked_gm():
    # 4 sites at positions 10,20,30,40; a mask makes columns 0 and 2 accessible.
    g = np.array([[0, 1, 2, 0], [1, 1, 0, 2]], dtype=np.int8)
    gm = GenotypeMatrix(g, np.array([10, 20, 30, 40]))
    mask = np.zeros(41, dtype=bool)
    mask[[10, 30]] = True
    gm.set_accessible_mask(mask)
    return gm, g


def test_genotypes_setter_resets_filter_cache():
    gm, g = _masked_gm()
    np.testing.assert_array_equal(cp.asnumpy(gm.genotypes), g[:, [0, 2]])  # cache built
    g2 = (g + 1) % 3
    gm.genotypes = cp.asarray(g2)
    # getter re-derives from the new array through the same mask, not the cache.
    np.testing.assert_array_equal(cp.asnumpy(gm.genotypes), g2[:, [0, 2]])


def test_positions_setter_resets_filter_cache():
    gm, _ = _masked_gm()
    np.testing.assert_array_equal(cp.asnumpy(gm.positions), [10, 30])  # cache built
    gm.positions = cp.asarray(np.array([11, 21, 31, 41]))
    np.testing.assert_array_equal(cp.asnumpy(gm.positions), [11, 31])
