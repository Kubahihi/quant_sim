"""Frequency-aware analysis for imported histories, isolated from daily models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.custom_history import ImportedHistory
from .portfolio_metrics import build_portfolio_timeseries
from .returns import calculate_annualized_return
from .risk_metrics import calculate_max_drawdown, calculate_sharpe_ratio, calculate_volatility


def analyze_history(history: ImportedHistory, weights: np.ndarray, risk_free_rate: float) -> dict:
    returns = history.returns
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (returns.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Provide a non-negative, finite weight for each asset.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Portfolio weights must total 100%.")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("The annual risk-free rate must be finite and greater than -100%.")
    periods = history.periods_per_year
    portfolio = returns @ weights

    def metrics(series):
        return {
            "Total return": float((1 + series).prod() - 1),
            "Annualized return": calculate_annualized_return(series, periods),
            "Annualized volatility": calculate_volatility(series, periods),
            "Sharpe ratio": calculate_sharpe_ratio(series, risk_free_rate, periods),
            "Maximum drawdown": calculate_max_drawdown(series),
        }

    return {
        "metrics": metrics(portfolio),
        "assets": pd.DataFrame({name: metrics(returns[name]) for name in returns}).T,
        "portfolio_returns": portfolio,
        "timeseries": build_portfolio_timeseries(portfolio),
        "correlation": returns.corr(),
    }


def bootstrap_history(
    history: ImportedHistory, weights: np.ndarray, *, years: int = 5,
    simulations: int = 2000, block_length: int = 1, seed: int = 42,
) -> pd.DataFrame:
    """Joint circular block bootstrap, rebalanced once per observation period."""
    analyze_history(history, weights, 0.0)  # shared input validation
    if not 1 <= years <= 30 or not 100 <= simulations <= 10000:
        raise ValueError("Choose 1–30 years and 100–10,000 simulations.")
    if not 1 <= block_length <= len(history.returns):
        raise ValueError("Block length must fit within the imported history.")
    portfolio = history.returns.to_numpy() @ np.asarray(weights, dtype=float)
    steps = years * history.periods_per_year
    rng = np.random.default_rng(seed)
    wealth = np.ones(simulations)
    rows = [[0.0, 1.0, 1.0, 1.0]]
    # Resample whole cross-asset observations jointly. Accumulate only the
    # current paths, avoiding a potentially large steps × simulations matrix.
    for step in range(steps):
        if step % block_length == 0:
            starts = rng.integers(0, len(portfolio), size=simulations)
        sample = (starts + step % block_length) % len(portfolio)
        wealth *= 1 + portfolio[sample]
        if not np.isfinite(wealth).all():
            raise ValueError("Simulation overflowed. Check returns and units or shorten the horizon.")
        rows.append([(step + 1) / history.periods_per_year, *np.quantile(wealth, [0.1, 0.5, 0.9])])
    return pd.DataFrame(rows, columns=["Years", "10th percentile", "Median", "90th percentile"]).set_index("Years")
