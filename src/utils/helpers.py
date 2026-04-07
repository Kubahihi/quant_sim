import numpy as np
import pandas as pd
from typing import Union


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize returns"""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return float(annualized)


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize volatility"""
    return float(returns.std() * np.sqrt(periods_per_year))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    return numerator / denominator
