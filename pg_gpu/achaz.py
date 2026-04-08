"""Backward-compatible re-exports from diversity module.

The FrequencySpectrum class and related utilities have been moved to
pg_gpu.diversity. This module re-exports them for backward compatibility.
"""

from .diversity import (  # noqa: F401
    FrequencySpectrum,
    compute_sigma_ij,
    project_sfs,
    WEIGHT_REGISTRY,
    _weights_watterson, _weights_pi, _weights_theta_h, _weights_theta_l,
    _weights_eta1, _weights_eta1_star, _weights_minus_eta1,
    _weights_minus_eta1_star,
    _harmonic_sums,
)
