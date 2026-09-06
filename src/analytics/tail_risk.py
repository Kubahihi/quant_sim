"""Empirical expected shortfall, including atoms at the quantile boundary."""
import numpy as np


def empirical_expected_shortfall(losses, confidence: float = 0.95, *, axis=None):
    """Mean of the worst (1-confidence) probability mass of equal-weight losses.

    Uses the Rockafellar-Uryasev representation at an empirical VaR. Unlike
    a conditional tail average this includes only the required mass of ties.
    The sign convention is loss-positive; gains can produce negative ES.
    """
    values = np.asarray(losses, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be non-empty and finite.")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    cutoff = np.quantile(values, level, axis=axis, method="inverted_cdf", keepdims=True)
    result = np.squeeze(cutoff, axis=axis) + np.maximum(values - cutoff, 0.0).mean(axis=axis) / (1.0 - level)
    return float(result) if np.ndim(result) == 0 else result
