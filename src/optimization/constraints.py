from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


WEIGHT_TOLERANCE = 1e-7


def build_weight_bounds(
    n_assets: int,
    *,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Build feasible per-asset bounds without silently changing the mandate."""
    if n_assets < 1:
        raise ValueError("at least one asset is required.")

    upper = 1.0 if max_weight is None else float(max_weight)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("max_weight must be positive.")
    if upper * n_assets < 1.0 - WEIGHT_TOLERANCE:
        raise ValueError(
            f"max_weight={upper:.6f} is infeasible for {n_assets} assets; "
            f"it must be at least {1.0 / n_assets:.6f}."
        )

    lower = -upper if allow_short else 0.0
    return [(lower, upper) for _ in range(n_assets)]


def validate_weight_solution(
    weights: Sequence[float] | np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    tolerance: float = WEIGHT_TOLERANCE,
) -> np.ndarray:
    """Validate solver output before it can be presented as a recommendation."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size != len(bounds):
        raise ValueError("solver returned a weight vector with the wrong shape.")
    if not np.all(np.isfinite(values)):
        raise ValueError("solver returned non-finite weights.")
    if not np.isclose(float(values.sum()), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("solver weights do not sum to one.")

    for value, (lower, upper) in zip(values, bounds, strict=False):
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError("solver weights violate their bounds.")
    return values
