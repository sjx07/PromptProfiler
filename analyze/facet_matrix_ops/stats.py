"""Statistical helpers for FACET matrix operators."""
from __future__ import annotations

import math
from typing import Sequence, Tuple


def trimmed_mean(values: Sequence[float], trim: float = 0.05) -> float:
    if not values:
        return float("nan")
    vals = sorted(float(v) for v in values)
    if len(vals) < 3 or trim <= 0:
        return float(sum(vals) / len(vals))
    k = int(math.floor(len(vals) * trim))
    trimmed = vals[k:len(vals) - k] if k > 0 and len(vals) - 2 * k > 0 else vals
    return float(sum(trimmed) / len(trimmed))


def bootstrap_mean_delta_ci(
    values: Sequence[float],
    *,
    center: float = 0.0,
    n_bootstrap: int = 500,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap CI for ``mean(values) - center`` and P(>0)."""
    if n_bootstrap <= 0 or len(values) < 2:
        return (float("nan"), float("nan"), float("nan"))
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(int(n_bootstrap), len(arr)))
    sampled = arr[idx].mean(axis=1) - center
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(sampled, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return (float(lo), float(hi), float((sampled > 0).mean()))


def wilson_interval(successes: int, n: int, *, ci_level: float = 0.95) -> Tuple[float, float]:
    """Wilson interval for a binary rate without scipy."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z_values = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_values.get(round(ci_level, 2), 1.959963984540054)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * n)) / n) / denom
    return (float(center - half), float(center + half))
