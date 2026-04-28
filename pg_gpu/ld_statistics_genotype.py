"""
Genotype-based LD statistics (DD, Dz, pi2) for diploid data.

Formulas transcribed from moments.LD.stats_from_genotype_counts,
vectorized for CuPy GPU computation. The polynomial numerators
are Mathematica-derived unbiased estimators operating on 9-way
genotype configuration counts.

Module status
-------------
All polynomial functions in this module are RETAINED AS REFERENCE
but no longer called in the production path. They have been
superseded by fused CUDA RawKernels in genotype_kernels.py that
compute each statistic in a single kernel launch with all
arithmetic in GPU registers. The CuPy-based formulas here serve
as:
  - A readable reference for the exact polynomial expressions
  - A potential CPU-fallback path (CuPy ops work without CUDA)
  - A validation target for the fused kernels (parity tests in
    tests/test_genotype_kernels_dd.py and friends assert the
    kernels match these polynomials to ~1e-12)
"""
import cupy as cp


def _safe_div(numer, denom, valid):
    """Element-wise division returning 0 where *valid* is False."""
    return cp.where(valid, numer / cp.maximum(denom, 1), 0.0)


# ===========================================================================
# DD  (D^2 and D1*D2)  --  REFERENCE ONLY: superseded by fused CUDA kernels
#         genotype_kernels._DD_GENO_SINGLE_KERN, _DD_GENO_BETWEEN_KERN
# ===========================================================================


def dd_geno_single(p):
    """DD for single population from _PopDataGeno."""
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = (
        p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9,
    )
    n = p.n
    numer = (
        g2 * g4
        - g2 ** 2 * g4
        + 4 * g3 * g4
        - 4 * g2 * g3 * g4
        - 4 * g3 ** 2 * g4
        - g2 * g4 ** 2
        - 4 * g3 * g4 ** 2
        + g1 * g5
        - g1 ** 2 * g5
        + g3 * g5
        + 2 * g1 * g3 * g5
        - g3 ** 2 * g5
        - 4 * g3 * g4 * g5
        - g1 * g5 ** 2
        - g3 * g5 ** 2
        + 4 * g1 * g6
        - 4 * g1 ** 2 * g6
        + g2 * g6
        - 4 * g1 * g2 * g6
        - g2 ** 2 * g6
        + 2 * g2 * g4 * g6
        - 4 * g1 * g5 * g6
        - 4 * g1 * g6 ** 2
        - g2 * g6 ** 2
        + 4 * g2 * g7
        - 4 * g2 ** 2 * g7
        + 16 * g3 * g7
        - 16 * g2 * g3 * g7
        - 16 * g3 ** 2 * g7
        - 4 * g2 * g4 * g7
        - 16 * g3 * g4 * g7
        + g5 * g7
        + 2 * g1 * g5 * g7
        - 4 * g2 * g5 * g7
        - 18 * g3 * g5 * g7
        - g5 ** 2 * g7
        + 4 * g6 * g7
        + 8 * g1 * g6 * g7
        - 16 * g3 * g6 * g7
        - 4 * g5 * g6 * g7
        - 4 * g6 ** 2 * g7
        - 4 * g2 * g7 ** 2
        - 16 * g3 * g7 ** 2
        - g5 * g7 ** 2
        - 4 * g6 * g7 ** 2
        + 4 * g1 * g8
        - 4 * g1 ** 2 * g8
        + 4 * g3 * g8
        + 8 * g1 * g3 * g8
        - 4 * g3 ** 2 * g8
        + g4 * g8
        - 4 * g1 * g4 * g8
        + 2 * g2 * g4 * g8
        - g4 ** 2 * g8
        - 4 * g1 * g5 * g8
        - 4 * g3 * g5 * g8
        + g6 * g8
        + 2 * g2 * g6 * g8
        - 4 * g3 * g6 * g8
        + 2 * g4 * g6 * g8
        - g6 ** 2 * g8
        - 16 * g3 * g7 * g8
        - 4 * g6 * g7 * g8
        - 4 * g1 * g8 ** 2
        - 4 * g3 * g8 ** 2
        - g4 * g8 ** 2
        - g6 * g8 ** 2
        + 16 * g1 * g9
        - 16 * g1 ** 2 * g9
        + 4 * g2 * g9
        - 16 * g1 * g2 * g9
        - 4 * g2 ** 2 * g9
        + 4 * g4 * g9
        - 16 * g1 * g4 * g9
        + 8 * g3 * g4 * g9
        - 4 * g4 ** 2 * g9
        + g5 * g9
        - 18 * g1 * g5 * g9
        - 4 * g2 * g5 * g9
        + 2 * g3 * g5 * g9
        - 4 * g4 * g5 * g9
        - g5 ** 2 * g9
        - 16 * g1 * g6 * g9
        - 4 * g2 * g6 * g9
        + 8 * g2 * g7 * g9
        + 2 * g5 * g7 * g9
        - 16 * g1 * g8 * g9
        - 4 * g4 * g8 * g9
        - 16 * g1 * g9 ** 2
        - 4 * g2 * g9 ** 2
        - 4 * g4 * g9 ** 2
        - g5 * g9 ** 2
    ) / 16.0 + (
        -(
            (g2 / 2.0 + g3 + g5 / 4.0 + g6 / 2.0)
            * (g4 / 2.0 + g5 / 4.0 + g7 + g8 / 2.0)
        )
        + (g1 + g2 / 2.0 + g4 / 2.0 + g5 / 4.0)
        * (g5 / 4.0 + g6 / 2.0 + g8 / 2.0 + g9)
    ) ** 2
    denom = n * (n - 1) * (n - 2) * (n - 3)
    valid = n >= 4
    return 4.0 * _safe_div(numer, denom, valid)


def dd_geno_between(pi, pj):
    """DD between two populations from _PopDataGeno objects."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        -(
            (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
        )
        + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
        * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
    ) * (
        -(
            (g22 / 2.0 + g23 + g25 / 4.0 + g26 / 2.0)
            * (g24 / 2.0 + g25 / 4.0 + g27 + g28 / 2.0)
        )
        + (g21 + g22 / 2.0 + g24 / 2.0 + g25 / 4.0)
        * (g25 / 4.0 + g26 / 2.0 + g28 / 2.0 + g29)
    )
    denom = n1 * (n1 - 1) * n2 * (n2 - 1)
    valid = (n1 >= 2) & (n2 >= 2)
    return 4.0 * _safe_div(numer, denom, valid)


# ===========================================================================
# Dz  --  REFERENCE ONLY: superseded by fused CUDA kernels in
#         genotype_kernels._DZ_*_KERN
# ===========================================================================


def dz_geno_single(p):
    """Dz for single population (Dz(i,i,i)) from _PopDataGeno."""
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = (
        p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9,
    )
    n = p.n
    numer = (
        -(g2 * g4)
        + 3 * g1 * g2 * g4
        + g2 ** 2 * g4
        + 2 * g3 * g4
        + 4 * g1 * g3 * g4
        - g2 * g3 * g4
        - 4 * g3 ** 2 * g4
        + g2 * g4 ** 2
        + 2 * g3 * g4 ** 2
        + 2 * g1 * g5
        - 3 * g1 ** 2 * g5
        - g1 * g2 * g5
        + 2 * g3 * g5
        + 2 * g1 * g3 * g5
        - g2 * g3 * g5
        - 3 * g3 ** 2 * g5
        - g1 * g4 * g5
        + g3 * g4 * g5
        + 2 * g1 * g6
        - 4 * g1 ** 2 * g6
        - g2 * g6
        - g1 * g2 * g6
        + g2 ** 2 * g6
        + 4 * g1 * g3 * g6
        + 3 * g2 * g3 * g6
        - 2 * g1 * g4 * g6
        - 2 * g2 * g4 * g6
        - 2 * g3 * g4 * g6
        + g1 * g5 * g6
        - g3 * g5 * g6
        + 2 * g1 * g6 ** 2
        + g2 * g6 ** 2
        + 2 * g2 * g7
        + 4 * g1 * g2 * g7
        + 2 * g2 ** 2 * g7
        + 8 * g3 * g7
        + 4 * g1 * g3 * g7
        - 4 * g3 ** 2 * g7
        - g2 * g4 * g7
        + 2 * g5 * g7
        + 2 * g1 * g5 * g7
        + g2 * g5 * g7
        + 2 * g3 * g5 * g7
        - g4 * g5 * g7
        + 2 * g6 * g7
        - g2 * g6 * g7
        - 2 * g4 * g6 * g7
        + g5 * g6 * g7
        + 2 * g6 ** 2 * g7
        - 4 * g2 * g7 ** 2
        - 4 * g3 * g7 ** 2
        - 3 * g5 * g7 ** 2
        - 4 * g6 * g7 ** 2
        + 2 * g1 * g8
        - 4 * g1 ** 2 * g8
        - 2 * g1 * g2 * g8
        + 2 * g3 * g8
        - 2 * g2 * g3 * g8
        - 4 * g3 ** 2 * g8
        - g4 * g8
        - g1 * g4 * g8
        - 2 * g2 * g4 * g8
        - g3 * g4 * g8
        + g4 ** 2 * g8
        + g1 * g5 * g8
        + g3 * g5 * g8
        - g6 * g8
        - g1 * g6 * g8
        - 2 * g2 * g6 * g8
        - g3 * g6 * g8
        - 2 * g4 * g6 * g8
        + g6 ** 2 * g8
        + 4 * g1 * g7 * g8
        - 2 * g2 * g7 * g8
        + 3 * g4 * g7 * g8
        - g5 * g7 * g8
        - g6 * g7 * g8
        + 2 * g1 * g8 ** 2
        + 2 * g3 * g8 ** 2
        + g4 * g8 ** 2
        + g6 * g8 ** 2
        + 8 * g1 * g9
        - 4 * g1 ** 2 * g9
        + 2 * g2 * g9
        + 2 * g2 ** 2 * g9
        + 4 * g1 * g3 * g9
        + 4 * g2 * g3 * g9
        + 2 * g4 * g9
        - g2 * g4 * g9
        + 2 * g4 ** 2 * g9
        + 2 * g5 * g9
        + 2 * g1 * g5 * g9
        + g2 * g5 * g9
        + 2 * g3 * g5 * g9
        + g4 * g5 * g9
        - g2 * g6 * g9
        - 2 * g4 * g6 * g9
        - g5 * g6 * g9
        + 4 * g1 * g7 * g9
        + 4 * g3 * g7 * g9
        + 4 * g4 * g7 * g9
        + 2 * g5 * g7 * g9
        + 4 * g6 * g7 * g9
        - 2 * g2 * g8 * g9
        + 4 * g3 * g8 * g9
        - g4 * g8 * g9
        - g5 * g8 * g9
        + 3 * g6 * g8 * g9
        - 4 * g1 * g9 ** 2
        - 4 * g2 * g9 ** 2
        - 4 * g4 * g9 ** 2
        - 3 * g5 * g9 ** 2
    ) / 4.0 + (-g1 + g3 - g4 + g6 - g7 + g9) * (-g1 - g2 - g3 + g7 + g8 + g9) * (
        -(
            (g2 / 2.0 + g3 + g5 / 4.0 + g6 / 2.0)
            * (g4 / 2.0 + g5 / 4.0 + g7 + g8 / 2.0)
        )
        + (g1 + g2 / 2.0 + g4 / 2.0 + g5 / 4.0)
        * (g5 / 4.0 + g6 / 2.0 + g8 / 2.0 + g9)
    )
    denom = n * (n - 1) * (n - 2) * (n - 3)
    valid = n >= 4
    return 2.0 * _safe_div(numer, denom, valid)


def dz_geno_p1p2(pi, pj):
    """Dz(i,i,j): pop1==pop2, pop3 different. pi is the repeated pop."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (
                2 * g12 * g17
                + 4 * g13 * g17
                + 2 * g15 * g17
                + 4 * g16 * g17
                - 2 * g11 * g18
                + 2 * g13 * g18
                - 2 * g14 * g18
                + 2 * g16 * g18
                - 4 * g11 * g19
                - 2 * g12 * g19
                - 4 * g14 * g19
                - 2 * g15 * g19
            )
            * (-g21 + g23 - g24 + g26 - g27 + g29)
        )
        / 4.0
        + (
            -(
                (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
                * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
            )
            + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
            * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
        )
        * (-g21 + g23 - g24 + g26 - g27 + g29)
        + (-g11 - g12 - g13 + g17 + g18 + g19)
        * (
            -(
                (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
                * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
            )
            + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
            * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
        )
        * (-g21 + g23 - g24 + g26 - g27 + g29)
    )
    denom = n2 * n1 * (n1 - 1.0) * (n1 - 2)
    valid = (n1 >= 3) & (n2 >= 1)
    return 2.0 * _safe_div(numer, denom, valid)


def dz_geno_p1p3(pi, pj):
    """Dz(i,j,i): pop1==pop3, pop2 different. pi is the repeated pop."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (
                2 * g13 * g14
                + 2 * g13 * g15
                - 2 * g11 * g16
                - 2 * g12 * g16
                + 4 * g13 * g17
                + 2 * g16 * g17
                + 4 * g13 * g18
                + 2 * g16 * g18
                - 4 * g11 * g19
                - 4 * g12 * g19
                - 2 * g14 * g19
                - 2 * g15 * g19
            )
            * (-g21 - g22 - g23 + g27 + g28 + g29)
        )
        / 4.0
        + (
            -(
                (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
                * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
            )
            + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
            * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
        )
        * (-g21 - g22 - g23 + g27 + g28 + g29)
        + (-g11 + g13 - g14 + g16 - g17 + g19)
        * (
            -(
                (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
                * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
            )
            + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
            * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
        )
        * (-g21 - g22 - g23 + g27 + g28 + g29)
    )
    denom = n2 * n1 * (n1 - 1.0) * (n1 - 2)
    valid = (n1 >= 3) & (n2 >= 1)
    return 2.0 * _safe_div(numer, denom, valid)


def dz_geno_p2p3(pi, pj):
    """Dz(i,j,j): pop2==pop3, pi is pop1, pj is the repeated pop."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        -(
            (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
        )
        + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
        * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
    ) * (-g21 + g23 + g27 - g29) + (
        -(
            (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
        )
        + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
        * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
    ) * (
        -g21 + g23 - g24 + g26 - g27 + g29
    ) * (
        -g21 - g22 - g23 + g27 + g28 + g29
    )
    denom = n2 * (n2 - 1) * n1 * (n1 - 1.0)
    valid = (n1 >= 2) & (n2 >= 2)
    return 2.0 * _safe_div(numer, denom, valid)


def dz_geno_alldiff(pi, pj, pk):
    """Dz(i,j,k): all three populations different."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    g31, g32, g33, g34, g35, g36, g37, g38, g39 = (
        pk.g1, pk.g2, pk.g3, pk.g4, pk.g5, pk.g6, pk.g7, pk.g8, pk.g9,
    )
    n1 = pi.n
    n2 = pj.n
    n3 = pk.n
    numer = (
        (
            -(
                (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
                * (g14 / 2.0 + g15 / 4.0 + g17 + g18 / 2.0)
            )
            + (g11 + g12 / 2.0 + g14 / 2.0 + g15 / 4.0)
            * (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
        )
        * (-g21 - g22 - g23 + g27 + g28 + g29)
        * (-g31 + g33 - g34 + g36 - g37 + g39)
    )
    denom = n1 * (n1 - 1) * n2 * n3
    valid = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    return 2.0 * _safe_div(numer, denom, valid)


# ===========================================================================
# pi2  --  REFERENCE ONLY: superseded by fused CUDA kernels in
#          genotype_kernels._PI2_*_KERN
# ===========================================================================


def pi2_geno_single(p):
    """pi2(i,i,i,i): all four population indices the same."""
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = (
        p.g1, p.g2, p.g3, p.g4, p.g5, p.g6, p.g7, p.g8, p.g9,
    )
    n = p.n
    numer = (g1 + g2 + g3 + g4 / 2.0 + g5 / 2.0 + g6 / 2.0) * (
        g1 + g2 / 2.0 + g4 + g5 / 2.0 + g7 + g8 / 2.0
    ) * (g2 / 2.0 + g3 + g5 / 2.0 + g6 + g8 / 2.0 + g9) * (
        g4 / 2.0 + g5 / 2.0 + g6 / 2.0 + g7 + g8 + g9
    ) + (
        13 * g2 * g4
        - 16 * g1 * g2 * g4
        - 11 * g2 ** 2 * g4
        + 16 * g3 * g4
        - 28 * g1 * g3 * g4
        - 24 * g2 * g3 * g4
        - 8 * g3 ** 2 * g4
        - 11 * g2 * g4 ** 2
        - 20 * g3 * g4 ** 2
        - 6 * g5
        + 12 * g1 * g5
        - 4 * g1 ** 2 * g5
        + 17 * g2 * g5
        - 20 * g1 * g2 * g5
        - 11 * g2 ** 2 * g5
        + 12 * g3 * g5
        - 28 * g1 * g3 * g5
        - 20 * g2 * g3 * g5
        - 4 * g3 ** 2 * g5
        + 17 * g4 * g5
        - 20 * g1 * g4 * g5
        - 32 * g2 * g4 * g5
        - 40 * g3 * g4 * g5
        - 11 * g4 ** 2 * g5
        + 11 * g5 ** 2
        - 16 * g1 * g5 ** 2
        - 17 * g2 * g5 ** 2
        - 16 * g3 * g5 ** 2
        - 17 * g4 * g5 ** 2
        - 6 * g5 ** 3
        + 16 * g1 * g6
        - 8 * g1 ** 2 * g6
        + 13 * g2 * g6
        - 24 * g1 * g2 * g6
        - 11 * g2 ** 2 * g6
        - 28 * g1 * g3 * g6
        - 16 * g2 * g3 * g6
        + 24 * g4 * g6
        - 36 * g1 * g4 * g6
        - 38 * g2 * g4 * g6
        - 36 * g3 * g4 * g6
        - 20 * g4 ** 2 * g6
        + 17 * g5 * g6
        - 40 * g1 * g5 * g6
        - 32 * g2 * g5 * g6
        - 20 * g3 * g5 * g6
        - 42 * g4 * g5 * g6
        - 17 * g5 ** 2 * g6
        - 20 * g1 * g6 ** 2
        - 11 * g2 * g6 ** 2
        - 20 * g4 * g6 ** 2
        - 11 * g5 * g6 ** 2
        + 16 * g2 * g7
        - 28 * g1 * g2 * g7
        - 20 * g2 ** 2 * g7
        + 16 * g3 * g7
        - 48 * g1 * g3 * g7
        - 44 * g2 * g3 * g7
        - 16 * g3 ** 2 * g7
        - 24 * g2 * g4 * g7
        - 44 * g3 * g4 * g7
        + 12 * g5 * g7
        - 28 * g1 * g5 * g7
        - 40 * g2 * g5 * g7
        - 48 * g3 * g5 * g7
        - 20 * g4 * g5 * g7
        - 16 * g5 ** 2 * g7
        + 16 * g6 * g7
        - 48 * g1 * g6 * g7
        - 48 * g2 * g6 * g7
        - 44 * g3 * g6 * g7
        - 36 * g4 * g6 * g7
        - 40 * g5 * g6 * g7
        - 20 * g6 ** 2 * g7
        - 8 * g2 * g7 ** 2
        - 16 * g3 * g7 ** 2
        - 4 * g5 * g7 ** 2
        - 8 * g6 * g7 ** 2
        + 16 * g1 * g8
        - 8 * g1 ** 2 * g8
        + 24 * g2 * g8
        - 36 * g1 * g2 * g8
        - 20 * g2 ** 2 * g8
        + 16 * g3 * g8
        - 48 * g1 * g3 * g8
        - 36 * g2 * g3 * g8
        - 8 * g3 ** 2 * g8
        + 13 * g4 * g8
        - 24 * g1 * g4 * g8
        - 38 * g2 * g4 * g8
        - 48 * g3 * g4 * g8
        - 11 * g4 ** 2 * g8
        + 17 * g5 * g8
        - 40 * g1 * g5 * g8
        - 42 * g2 * g5 * g8
        - 40 * g3 * g5 * g8
        - 32 * g4 * g5 * g8
        - 17 * g5 ** 2 * g8
        + 13 * g6 * g8
        - 48 * g1 * g6 * g8
        - 38 * g2 * g6 * g8
        - 24 * g3 * g6 * g8
        - 38 * g4 * g6 * g8
        - 32 * g5 * g6 * g8
        - 11 * g6 ** 2 * g8
        - 28 * g1 * g7 * g8
        - 36 * g2 * g7 * g8
        - 44 * g3 * g7 * g8
        - 16 * g4 * g7 * g8
        - 20 * g5 * g7 * g8
        - 24 * g6 * g7 * g8
        - 20 * g1 * g8 ** 2
        - 20 * g2 * g8 ** 2
        - 20 * g3 * g8 ** 2
        - 11 * g4 * g8 ** 2
        - 11 * g5 * g8 ** 2
        - 11 * g6 * g8 ** 2
        + 16 * g1 * g9
        - 16 * g1 ** 2 * g9
        + 16 * g2 * g9
        - 44 * g1 * g2 * g9
        - 20 * g2 ** 2 * g9
        - 48 * g1 * g3 * g9
        - 28 * g2 * g3 * g9
        + 16 * g4 * g9
        - 44 * g1 * g4 * g9
        - 48 * g2 * g4 * g9
        - 48 * g3 * g4 * g9
        - 20 * g4 ** 2 * g9
        + 12 * g5 * g9
        - 48 * g1 * g5 * g9
        - 40 * g2 * g5 * g9
        - 28 * g3 * g5 * g9
        - 40 * g4 * g5 * g9
        - 16 * g5 ** 2 * g9
        - 44 * g1 * g6 * g9
        - 24 * g2 * g6 * g9
        - 36 * g4 * g6 * g9
        - 20 * g5 * g6 * g9
        - 48 * g1 * g7 * g9
        - 48 * g2 * g7 * g9
        - 48 * g3 * g7 * g9
        - 28 * g4 * g7 * g9
        - 28 * g5 * g7 * g9
        - 28 * g6 * g7 * g9
        - 44 * g1 * g8 * g9
        - 36 * g2 * g8 * g9
        - 28 * g3 * g8 * g9
        - 24 * g4 * g8 * g9
        - 20 * g5 * g8 * g9
        - 16 * g6 * g8 * g9
        - 16 * g1 * g9 ** 2
        - 8 * g2 * g9 ** 2
        - 8 * g4 * g9 ** 2
        - 4 * g5 * g9 ** 2
    ) / 16.0
    denom = n * (n - 1) * (n - 2) * (n - 3)
    valid = n >= 4
    return _safe_div(numer, denom, valid)


def pi2_geno_triple_123(pi, pj):
    """pi2(i,i,i,j) or pi2(i,i,j,i): triple pop is pi, single is pj.

    Covers pop1==pop2==pop3 (with pop4 different) and
    pop1==pop2==pop4 (with pop3 different).
    """
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (
                (-3 * g14) / 2.0
                - (3 * g15) / 2.0
                - (3 * g16) / 2.0
                - 2 * g17
                - 2 * g18
                - 2 * g19
            )
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        )
        / 2.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (
                (-5 * g14) / 2.0
                - (5 * g15) / 2.0
                - (5 * g16) / 2.0
                - 4 * g17
                - 4 * g18
                - 4 * g19
            )
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 2.0
        + (
            (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (
                g15 * g21
                + 2 * g16 * g21
                + 2 * g18 * g21
                + 4 * g19 * g21
                + 2 * g22
                - g12 * g22
                - 2 * g13 * g22
                + 2 * g14 * g22
                + g15 * g22
                + 4 * g17 * g22
                + 3 * g18 * g22
                + 2 * g19 * g22
                + 4 * g23
                - 2 * g12 * g23
                - 4 * g13 * g23
                + 4 * g14 * g23
                + g15 * g23
                - 2 * g16 * g23
                + 8 * g17 * g23
                + 4 * g18 * g23
                + g15 * g24
                + 2 * g16 * g24
                + 2 * g18 * g24
                + 4 * g19 * g24
                + 2 * g25
                - g12 * g25
                - 2 * g13 * g25
                + 2 * g14 * g25
                + g15 * g25
                + 4 * g17 * g25
                + 3 * g18 * g25
                + 2 * g19 * g25
                + 4 * g26
                - 2 * g12 * g26
                - 4 * g13 * g26
                + 4 * g14 * g26
                + g15 * g26
                - 2 * g16 * g26
                + 8 * g17 * g26
                + 4 * g18 * g26
                + g15 * g27
                + 2 * g16 * g27
                + 2 * g18 * g27
                + 4 * g19 * g27
                + 2 * g28
                - g12 * g28
                - 2 * g13 * g28
                + 2 * g14 * g28
                + g15 * g28
                + 4 * g17 * g28
                + 3 * g18 * g28
                + 2 * g19 * g28
                + 4 * g29
                - 2 * g12 * g29
                - 4 * g13 * g29
                + 4 * g14 * g29
                + g15 * g29
                - 2 * g16 * g29
                + 8 * g17 * g29
                + 4 * g18 * g29
            )
        )
        / 8.0
        + (
            g15 * g21
            + 2 * g16 * g21
            - 2 * g17 * g22
            - 2 * g18 * g22
            - 2 * g19 * g22
            - g15 * g23
            - 2 * g16 * g23
            - 4 * g17 * g23
            - 4 * g18 * g23
            - 4 * g19 * g23
            + g15 * g24
            + 2 * g16 * g24
            - 2 * g17 * g25
            - 2 * g18 * g25
            - 2 * g19 * g25
            - g15 * g26
            - 2 * g16 * g26
            - 4 * g17 * g26
            - 4 * g18 * g26
            - 4 * g19 * g26
            + g15 * g27
            + 2 * g16 * g27
            - 2 * g17 * g28
            - 2 * g18 * g28
            - 2 * g19 * g28
            - g15 * g29
            - 2 * g16 * g29
            - 4 * g17 * g29
            - 4 * g18 * g29
            - 4 * g19 * g29
        )
        / 8.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (
                -(g15 * g21)
                - 2 * g16 * g21
                - 2 * g18 * g21
                - 4 * g19 * g21
                + g15 * g23
                + 2 * g16 * g23
                + 2 * g18 * g23
                + 4 * g19 * g23
                - g15 * g24
                - 2 * g16 * g24
                - 2 * g18 * g24
                - 4 * g19 * g24
                + g15 * g26
                + 2 * g16 * g26
                + 2 * g18 * g26
                + 4 * g19 * g26
                - g15 * g27
                - 2 * g16 * g27
                - 2 * g18 * g27
                - 4 * g19 * g27
                + g15 * g29
                + 2 * g16 * g29
                + 2 * g18 * g29
                + 4 * g19 * g29
            )
        )
        / 8.0
    )
    denom = n2 * n1 * (n1 - 1) * (n1 - 2)
    valid = (n1 >= 3) & (n2 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_triple_134(pi, pj):
    """pi2(i,j,i,i) or pi2(j,i,i,i): triple pop is pi, single is pj.

    Covers pop1==pop3==pop4 (with pop2 different) and
    pop2==pop3==pop4 (with pop1 different).
    """
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (
                (-3 * g12) / 2.0
                - 2 * g13
                - (3 * g15) / 2.0
                - 2 * g16
                - (3 * g18) / 2.0
                - 2 * g19
            )
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
        )
        / 2.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (
                (-5 * g12) / 2.0
                - 4 * g13
                - (5 * g15) / 2.0
                - 4 * g16
                - (5 * g18) / 2.0
                - 4 * g19
            )
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 2.0
        + (
            (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (
                g15 * g21
                + 2 * g16 * g21
                + 2 * g18 * g21
                + 4 * g19 * g21
                + g15 * g22
                + 2 * g16 * g22
                + 2 * g18 * g22
                + 4 * g19 * g22
                + g15 * g23
                + 2 * g16 * g23
                + 2 * g18 * g23
                + 4 * g19 * g23
                + 2 * g24
                + 2 * g12 * g24
                + 4 * g13 * g24
                - g14 * g24
                + g15 * g24
                + 3 * g16 * g24
                - 2 * g17 * g24
                + 2 * g19 * g24
                + 2 * g25
                + 2 * g12 * g25
                + 4 * g13 * g25
                - g14 * g25
                + g15 * g25
                + 3 * g16 * g25
                - 2 * g17 * g25
                + 2 * g19 * g25
                + 2 * g26
                + 2 * g12 * g26
                + 4 * g13 * g26
                - g14 * g26
                + g15 * g26
                + 3 * g16 * g26
                - 2 * g17 * g26
                + 2 * g19 * g26
                + 4 * g27
                + 4 * g12 * g27
                + 8 * g13 * g27
                - 2 * g14 * g27
                + g15 * g27
                + 4 * g16 * g27
                - 4 * g17 * g27
                - 2 * g18 * g27
                + 4 * g28
                + 4 * g12 * g28
                + 8 * g13 * g28
                - 2 * g14 * g28
                + g15 * g28
                + 4 * g16 * g28
                - 4 * g17 * g28
                - 2 * g18 * g28
                + 4 * g29
                + 4 * g12 * g29
                + 8 * g13 * g29
                - 2 * g14 * g29
                + g15 * g29
                + 4 * g16 * g29
                - 4 * g17 * g29
                - 2 * g18 * g29
            )
        )
        / 8.0
        + (
            g15 * g21
            + 2 * g18 * g21
            + g15 * g22
            + 2 * g18 * g22
            + g15 * g23
            + 2 * g18 * g23
            - 2 * g13 * g24
            - 2 * g16 * g24
            - 2 * g19 * g24
            - 2 * g13 * g25
            - 2 * g16 * g25
            - 2 * g19 * g25
            - 2 * g13 * g26
            - 2 * g16 * g26
            - 2 * g19 * g26
            - 4 * g13 * g27
            - g15 * g27
            - 4 * g16 * g27
            - 2 * g18 * g27
            - 4 * g19 * g27
            - 4 * g13 * g28
            - g15 * g28
            - 4 * g16 * g28
            - 2 * g18 * g28
            - 4 * g19 * g28
            - 4 * g13 * g29
            - g15 * g29
            - 4 * g16 * g29
            - 2 * g18 * g29
            - 4 * g19 * g29
        )
        / 8.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (
                -(g15 * g21)
                - 2 * g16 * g21
                - 2 * g18 * g21
                - 4 * g19 * g21
                - g15 * g22
                - 2 * g16 * g22
                - 2 * g18 * g22
                - 4 * g19 * g22
                - g15 * g23
                - 2 * g16 * g23
                - 2 * g18 * g23
                - 4 * g19 * g23
                + g15 * g27
                + 2 * g16 * g27
                + 2 * g18 * g27
                + 4 * g19 * g27
                + g15 * g28
                + 2 * g16 * g28
                + 2 * g18 * g28
                + 4 * g19 * g28
                + g15 * g29
                + 2 * g16 * g29
                + 2 * g18 * g29
                + 4 * g19 * g29
            )
        )
        / 8.0
    )
    denom = n2 * n1 * (n1 - 1) * (n1 - 2)
    valid = (n1 >= 3) & (n2 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_iikk(pi, pj):
    """pi2(i,i,j,j): pop1==pop2, pop3==pop4, two distinct populations."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (-g22 - g25 - g28)
        )
        / 4.0
        + (
            g14 * g22
            + g15 * g22
            + g16 * g22
            + g14 * g25
            + g15 * g25
            + g16 * g25
            + g14 * g28
            + g15 * g28
            + g16 * g28
        )
        / 16.0
        + (
            (-g14 - g15 - g16)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
        * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
        * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
    )
    denom = n1 * (n1 - 1) * n2 * (n2 - 1)
    valid = (n1 >= 2) & (n2 >= 2)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_ijij(pi, pj):
    """pi2(i,j,i,j) or pi2(i,j,j,i): pop1==pop3 and pop2==pop4 (or pop1==pop4 and pop2==pop3)."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    n1 = pi.n
    n2 = pj.n
    numer = (
        (
            (-g14 / 2.0 - g15 / 2.0 - g16 / 2.0 - g17 - g18 - g19)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        )
        / 4.0
        + (
            (-g15 / 4.0 - g16 / 2.0 - g18 / 2.0 - g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        )
        / 4.0
        + (
            (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
        )
        / 4.0
        + (
            (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (-g22 / 2.0 - g23 - g25 / 2.0 - g26 - g28 / 2.0 - g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (-g25 / 4.0 - g26 / 2.0 - g28 / 2.0 - g29)
        )
        / 4.0
        + (
            (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (-g25 / 4.0 - g26 / 2.0 - g28 / 2.0 - g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g25 / 4.0 + g26 / 2.0 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g25 / 4.0 + g26 / 2.0 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (-g14 / 2.0 - g15 / 2.0 - g16 / 2.0 - g17 - g18 - g19)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (1 + g12 / 2.0 + g13 - g14 / 2.0 + g16 / 2.0 - g17 - g18 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g15 / 4.0 + g16 / 2.0 + g18 / 2.0 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
        )
        / 4.0
        + (
            (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (1 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (-g12 / 2.0 - g13 - g15 / 4.0 - g16 / 2.0)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (
                -g12 / 2.0
                - g13
                - g15 / 2.0
                - g16
                - g18 / 2.0
                - g19
                - g22 / 2.0
                - g23
                - g25 / 2.0
                - g26
                - g28 / 2.0
                - g29
            )
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (g12 / 2.0 + g13 + g15 / 4.0 + g16 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
        )
        / 4.0
        + (
            g15 * g21
            + 2 * g16 * g21
            + 2 * g18 * g21
            + 4 * g19 * g21
            - g15 * g23
            - 2 * g16 * g23
            - 2 * g18 * g23
            - 4 * g19 * g23
            + g11 * g25
            - g13 * g25
            - g15 * g25
            - 2 * g16 * g25
            - g17 * g25
            - 2 * g18 * g25
            - 3 * g19 * g25
            + 2 * g11 * g26
            - 2 * g13 * g26
            - 2 * g15 * g26
            - 4 * g16 * g26
            - 2 * g17 * g26
            - 4 * g18 * g26
            - 6 * g19 * g26
            - g15 * g27
            - 2 * g16 * g27
            - 2 * g18 * g27
            - 4 * g19 * g27
            + 2 * g11 * g28
            - 2 * g13 * g28
            - 2 * g15 * g28
            - 4 * g16 * g28
            - 2 * g17 * g28
            - 4 * g18 * g28
            - 6 * g19 * g28
            + 4 * g11 * g29
            - 4 * g13 * g29
            - 3 * g15 * g29
            - 6 * g16 * g29
            - 4 * g17 * g29
            - 6 * g18 * g29
            - 8 * g19 * g29
        )
        / 16.0
    )
    denom = n1 * (n1 - 1) * n2 * (n2 - 1)
    valid = (n1 >= 2) & (n2 >= 2)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_iikl(pi, pj, pk):
    """pi2(i,i,j,k): pop1==pop2, pop3 and pop4 are distinct from each other and from pop1.

    pi is the repeated pop, pj is pop3, pk is pop4.
    """
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    g31, g32, g33, g34, g35, g36, g37, g38, g39 = (
        pk.g1, pk.g2, pk.g3, pk.g4, pk.g5, pk.g6, pk.g7, pk.g8, pk.g9,
    )
    n1 = pi.n
    n2 = pj.n
    n3 = pk.n
    numer = (
        (
            (-g14 / 2.0 - g15 / 2.0 - g16 / 2.0)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g22 / 2.0 + g23 + g25 / 2.0 + g26 + g28 / 2.0 + g29)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 2.0
        + (
            (-g14 / 2.0 - g15 / 2.0 - g16 / 2.0)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 / 2.0 + g24 + g25 / 2.0 + g27 + g28 / 2.0)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 2.0
    )
    denom = n1 * (n1 - 1) * n2 * n3
    valid = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_ijkk(pi, pj, pk):
    """pi2(i,j,k,k): pop3==pop4, pop1 and pop2 are distinct from each other and from pop3.

    pi is the repeated pop (pop3==pop4), pj is pop1, pk is pop2.
    """
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    g31, g32, g33, g34, g35, g36, g37, g38, g39 = (
        pk.g1, pk.g2, pk.g3, pk.g4, pk.g5, pk.g6, pk.g7, pk.g8, pk.g9,
    )
    n1 = pi.n
    n2 = pj.n
    n3 = pk.n
    numer = (
        (
            (-g12 / 2.0 - g15 / 2.0 - g18 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g31 + g32 + g33 + g34 / 2.0 + g35 / 2.0 + g36 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g31 + g32 + g33 + g34 / 2.0 + g35 / 2.0 + g36 / 2.0)
        )
        / 2.0
        + (
            (-g12 / 2.0 - g15 / 2.0 - g18 / 2.0)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g34 / 2.0 + g35 / 2.0 + g36 / 2.0 + g37 + g38 + g39)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g34 / 2.0 + g35 / 2.0 + g36 / 2.0 + g37 + g38 + g39)
        )
        / 2.0
    )
    denom = n1 * (n1 - 1) * n2 * n3
    valid = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_shared(pi, pj, pk):
    """pi2 with one shared pop: pi2(i,j,i,k), pi2(i,j,k,i), pi2(i,j,j,k), pi2(i,j,k,j).

    pi is the shared (repeated) pop, pj is the other locus-1 pop,
    pk is the other locus-2 pop.
    """
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    g31, g32, g33, g34, g35, g36, g37, g38, g39 = (
        pk.g1, pk.g2, pk.g3, pk.g4, pk.g5, pk.g6, pk.g7, pk.g8, pk.g9,
    )
    n1 = pi.n
    n2 = pj.n
    n3 = pk.n
    numer = (
        (
            (-g15 / 4.0 - g16 / 2.0 - g18 / 2.0 - g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 4.0
        + (
            (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 4.0
        + (
            (-g12 / 2.0 - g13 - g15 / 4.0 - g16 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g12 / 2.0 + g13 + g15 / 2.0 + g16 + g18 / 2.0 + g19)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
        )
        / 4.0
        + (
            (-g14 / 2.0 - g15 / 4.0 - g17 - g18 / 2.0)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 4.0
        + (
            (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 4.0
        + (
            (-g11 - g12 / 2.0 - g14 / 2.0 - g15 / 4.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g11 + g12 / 2.0 + g14 + g15 / 2.0 + g17 + g18 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
        )
        / 4.0
    )
    denom = n1 * (n1 - 1) * n2 * n3
    valid = (n1 >= 2) & (n2 >= 1) & (n3 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


def pi2_geno_alldiff(pi, pj, pk, pl):
    """pi2(i,j,k,l): all four population indices different."""
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = (
        pi.g1, pi.g2, pi.g3, pi.g4, pi.g5, pi.g6, pi.g7, pi.g8, pi.g9,
    )
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = (
        pj.g1, pj.g2, pj.g3, pj.g4, pj.g5, pj.g6, pj.g7, pj.g8, pj.g9,
    )
    g31, g32, g33, g34, g35, g36, g37, g38, g39 = (
        pk.g1, pk.g2, pk.g3, pk.g4, pk.g5, pk.g6, pk.g7, pk.g8, pk.g9,
    )
    g41, g42, g43, g44, g45, g46, g47, g48, g49 = (
        pl.g1, pl.g2, pl.g3, pl.g4, pl.g5, pl.g6, pl.g7, pl.g8, pl.g9,
    )
    n1 = pi.n
    n2 = pj.n
    n3 = pk.n
    n4 = pl.n
    numer = (
        (
            (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
            * (g41 + g42 / 2.0 + g44 + g45 / 2.0 + g47 + g48 / 2.0)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g32 / 2.0 + g33 + g35 / 2.0 + g36 + g38 / 2.0 + g39)
            * (g41 + g42 / 2.0 + g44 + g45 / 2.0 + g47 + g48 / 2.0)
        )
        / 4.0
        + (
            (g14 / 2.0 + g15 / 2.0 + g16 / 2.0 + g17 + g18 + g19)
            * (g21 + g22 + g23 + g24 / 2.0 + g25 / 2.0 + g26 / 2.0)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
            * (g42 / 2.0 + g43 + g45 / 2.0 + g46 + g48 / 2.0 + g49)
        )
        / 4.0
        + (
            (g11 + g12 + g13 + g14 / 2.0 + g15 / 2.0 + g16 / 2.0)
            * (g24 / 2.0 + g25 / 2.0 + g26 / 2.0 + g27 + g28 + g29)
            * (g31 + g32 / 2.0 + g34 + g35 / 2.0 + g37 + g38 / 2.0)
            * (g42 / 2.0 + g43 + g45 / 2.0 + g46 + g48 / 2.0 + g49)
        )
        / 4.0
    )
    denom = n1 * n2 * n3 * n4
    valid = (n1 >= 1) & (n2 >= 1) & (n3 >= 1) & (n4 >= 1)
    return 1.0 * _safe_div(numer, denom, valid)


# ===========================================================================
# Pairwise sigma_d^2 -- single-population unbiased estimator on diploids
# ===========================================================================
#
# Public entry point: turns the polynomial DD / pi2 components above into
# a per-pair sigma_d^2 = D^2 / pi^2 estimate, computed from 9-way diploid
# genotype counts via the unbiased multinomial projection (Ragsdale &
# Gravel 2019). This is the diploid analogue of pg_gpu.ld_statistics's
# tile-based sigma_d2 path, which works on phased haplotypes.

def sigma_d2_geno(matrix, idx_i=None, idx_j=None):
    """Per-pair unbiased sigma_D^2 (D^2 / pi^2) for diploid data.

    Computes the Ragsdale & Gravel (2019) unbiased polynomial estimator
    on 9-way diploid genotype counts. Each pair returns a single scalar
    -- the ratio of the per-pair D^2 component to the per-pair pi^2
    component -- ready for downstream binning, averaging, or whatever
    aggregation the caller wants.

    Parameters
    ----------
    matrix : GenotypeMatrix or HaplotypeMatrix
        Source of 0/1/2 dosages. A HaplotypeMatrix is converted to
        diploid by pairing adjacent haplotypes (the standard pg_gpu
        convention; missing data not allowed).
    idx_i, idx_j : array-like of int, optional
        Pair indices on the variant axis with ``idx_i[k] < idx_j[k]``.
        If omitted, defaults to the full upper triangle (``m*(m-1)/2``
        pairs). Caller is responsible for chunking when m is large.

    Returns
    -------
    cp.ndarray of float64, shape (n_pairs,)
        sigma_D^2 per pair. NaN where pi^2 is zero (e.g., monomorphic
        at one or both loci, or fewer than 4 valid individuals).

    Raises
    ------
    TypeError
        If ``matrix`` is not a GenotypeMatrix or HaplotypeMatrix.
    ValueError
        If ``matrix`` contains missing values.
    """
    from .ld_pipeline import compute_genotype_counts_for_pairs
    from .genotype_kernels import _SIGMA_D2_GENO_KERN, _launch
    from .genotype_matrix import GenotypeMatrix
    from .haplotype_matrix import HaplotypeMatrix

    if isinstance(matrix, GenotypeMatrix):
        gm = matrix
    elif isinstance(matrix, HaplotypeMatrix):
        gm = GenotypeMatrix.from_haplotype_matrix(matrix)
    else:
        raise TypeError(
            f"matrix must be a GenotypeMatrix or HaplotypeMatrix, "
            f"got {type(matrix).__name__}")
    if gm.device != 'GPU':
        gm.transfer_to_gpu()

    g = gm.genotypes
    if cp.any(g < 0):
        raise ValueError(
            "sigma_d2_geno does not support missing values (-1); "
            "filter or impute before calling")
    m = g.shape[1]

    if idx_i is None and idx_j is None:
        i, j = cp.triu_indices(m, k=1)
        idx_i = i.astype(cp.int32)
        idx_j = j.astype(cp.int32)
    elif idx_i is None or idx_j is None:
        raise ValueError("idx_i and idx_j must be provided together")
    else:
        idx_i = cp.asarray(idx_i, dtype=cp.int32)
        idx_j = cp.asarray(idx_j, dtype=cp.int32)
        if idx_i.shape != idx_j.shape:
            raise ValueError("idx_i and idx_j must have the same shape")

    # compute_genotype_counts_for_pairs returns counts in our convention
    # (n00, n01, n02, n10, n11, n12, n20, n21, n22). The sigma_D^2 kernel
    # reads in moments order (g1=n22, g2=n21, ..., g9=n00), so reverse
    # the columns. Skip _PopDataGeno: its derived fields (D_geno,
    # pA/qA, B_L/B_R, C00..C11, ...) are precomputed for the moments-LD
    # multi-pop kernels but unused by the single-pop sigma_D^2 polynomial,
    # which derives everything inline from the 9 raw counts.
    counts, _ = compute_genotype_counts_for_pairs(g, idx_i, idx_j)
    g_arr = cp.ascontiguousarray(counts[:, ::-1].astype(cp.float64))
    N = g_arr.shape[0]
    out = cp.empty(N, dtype=cp.float64)
    _launch(_SIGMA_D2_GENO_KERN, (g_arr, out, N), N)
    return out
