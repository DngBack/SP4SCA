from __future__ import annotations

import numpy as np


def normalize_scores(scores, method: str = "zscore", stats: dict | None = None):
    arr = np.asarray(scores, dtype=float)

    if method == "zscore":
        if stats is None:
            mu = float(arr.mean())
            sigma = float(arr.std())
            sigma = sigma if sigma > 1e-12 else 1.0
            stats = {"method": method, "mu": mu, "sigma": sigma}
        norm = (arr - stats["mu"]) / stats["sigma"]
        return norm, stats

    if method == "minmax":
        if stats is None:
            lo = float(arr.min())
            hi = float(arr.max())
            denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
            stats = {"method": method, "lo": lo, "hi": hi, "denom": denom}
        norm = (arr - stats["lo"]) / stats["denom"]
        return norm, stats

    raise ValueError(f"Unsupported normalization method: {method}")
