from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .returns import calculate_annualized_return
from .risk_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)


TRADING_DAYS = 252


def calculate_portfolio_daily_returns(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """Calculate weighted daily portfolio returns."""
    if asset_returns.empty:
        return pd.Series(dtype=float)

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size != asset_returns.shape[1]:
        raise ValueError("Weights length must match number of return columns.")

    return (asset_returns * weights_array).sum(axis=1)


def calculate_concentration_metrics(weights: np.ndarray) -> Dict[str, float]:
    """Calculate concentration metrics from portfolio weights."""
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size == 0:
        return {
            "hhi": 0.0,
            "effective_holdings": 0.0,
            "max_weight": 0.0,
        }

    hhi = float(np.square(weights_array).sum())
    effective_holdings = float(1.0 / hhi) if hhi > 0 else 0.0
    max_weight = float(weights_array.max())

    return {
        "hhi": hhi,
        "effective_holdings": effective_holdings,
        "max_weight": max_weight,
    }


def calculate_average_correlation(corr_matrix: pd.DataFrame) -> float:
    """Calculate average off-diagonal correlation."""
    if corr_matrix.empty or corr_matrix.shape[0] <= 1:
        return 0.0

    corr_values = corr_matrix.to_numpy(dtype=float)
    n = corr_values.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr_values[mask]
    if off_diag.size == 0:
        return 0.0

    return float(np.nanmean(off_diag))


def calculate_portfolio_core_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> Dict[str, float]:
    """Calculate key portfolio metrics used in scoring and reporting."""
    if portfolio_returns.empty:
        return {
            "daily_return_mean": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    total_return = float((1 + portfolio_returns).prod() - 1)
    return {
        "daily_return_mean": float(portfolio_returns.mean()),
        "annualized_return": calculate_annualized_return(portfolio_returns, TRADING_DAYS),
        "volatility": calculate_volatility(portfolio_returns, TRADING_DAYS, annualize=True),
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_returns, risk_free_rate, TRADING_DAYS),
        "max_drawdown": calculate_max_drawdown(portfolio_returns),
        "total_return": total_return,
    }


def build_portfolio_timeseries(
    portfolio_returns: pd.Series,
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """Build indexed portfolio value and drawdown time series."""
    if portfolio_returns.empty:
        return pd.DataFrame(columns=["value", "cumulative_return", "drawdown"])

    cumulative_growth = (1 + portfolio_returns).cumprod()
    value = cumulative_growth * initial_value
    running_max = value.cummax()
    drawdown = (value - running_max) / running_max

    return pd.DataFrame({
        "value": value,
        "cumulative_return": cumulative_growth - 1,
        "drawdown": drawdown,
    })
