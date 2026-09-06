"""Explicit rate conversions shared by portfolio estimators and reports."""
import numpy as np


def annual_effective_to_arithmetic(rate: float, periods_per_year: float = 252.0) -> float:
    """Annualize the periodic simple rate, without mixing compounding conventions."""
    value, periods = float(rate), float(periods_per_year)
    if not np.isfinite(value) or value <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -100%.")
    if not np.isfinite(periods) or periods <= 0:
        raise ValueError("periods_per_year must be finite and positive.")
    return float(periods * np.expm1(np.log1p(value) / periods))
