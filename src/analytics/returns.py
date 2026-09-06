import pandas as pd
import numpy as np


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Calculate returns from price series"""
    if method == "simple":
        returns = prices.pct_change(fill_method=None)
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns.dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative simple returns."""
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise ValueError("Simple returns must be finite and at least -100%.")
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate effective annual return from periodic simple returns."""
    if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0

    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1.0).any():
        raise ValueError("Simple returns must be finite and at least -100%.")
    if (values == -1.0).any():
        return -1.0
    log_growth = float(np.log1p(values).sum())
    return float(np.expm1(log_growth * periods_per_year / n_periods))
