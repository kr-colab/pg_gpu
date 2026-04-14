"""
Generate the deterministic haplotype matrix used as input to the R lostruct
reference-generation script (tests/data/generate_lostruct_reference.R).

Committed outputs:
  - tests/data/generate_lostruct_reference_input.json
  - tests/data/generate_lostruct_reference_input.npz (same data, for the
    Python parity tests to load without re-synthesizing)

Run manually from the repo root when the reference matrix needs to change:

    python tests/data/generate_lostruct_reference_input.py

Do NOT run in CI; the committed JSON/NPZ are the single source of truth.
"""

import json
import pathlib

import numpy as np


def main() -> None:
    rng = np.random.default_rng(2024)
    n_samples = 20
    n_variants = 3000
    window_size = 100   # SNP-count windows  → 30 windows
    k = 2
    n_blocks = 10
    prop = 0.2          # 6 windows per corner
    k_corners = 3

    # Random biallelic haplotypes; inject a structured block to make the
    # local-PCA signal non-trivial.
    hap = rng.integers(0, 2, size=(n_samples, n_variants), dtype=np.int8)
    # Structured region in the middle: samples [0:10] low-freq, [10:20] high-freq.
    for j in range(1200, 1800):
        hap[:10, j] = rng.binomial(1, 0.15, 10)
        hap[10:, j] = rng.binomial(1, 0.85, 10)

    positions = np.arange(n_variants, dtype=np.int64) * 1000

    out_dir = pathlib.Path(__file__).resolve().parent
    json_path = out_dir / "generate_lostruct_reference_input.json"
    npz_path = out_dir / "generate_lostruct_reference_input.npz"

    # JSON for R (rows = samples, cols = variants)
    payload = {
        "hap": hap.astype(int).tolist(),
        "positions": positions.tolist(),
        "window_size": int(window_size),
        "k": int(k),
        "n_blocks": int(n_blocks),
        "prop": float(prop),
        "k_corners": int(k_corners),
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh)

    # NPZ for the Python parity tests
    np.savez(
        npz_path,
        hap=hap,
        positions=positions,
        window_size=np.int64(window_size),
        k=np.int64(k),
        n_blocks=np.int64(n_blocks),
        prop=np.float64(prop),
        k_corners=np.int64(k_corners),
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
