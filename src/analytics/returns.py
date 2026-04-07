import pandas as pd
import numpy as np


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Calculate returns from price series"""
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns.dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns"""
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return"""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    return float((1 + total_return) ** (periods_per_year / n_periods) - 1)
