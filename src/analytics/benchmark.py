from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .correlation import calculate_alpha, calculate_beta
from .returns import calculate_annualized_return


TRADING_DAYS = 252


def _empty_active_metrics(benchmark_ticker: str, reason: str = "") -> Dict[str, float | int | bool | str]:
    return {
        "benchmark_ticker": benchmark_ticker,
        "benchmark_available": False,
        "benchmark_obs": 0,
        "benchmark_total_return": 0.0,
        "benchmark_annualized_return": 0.0,
        "active_return_total": 0.0,
        "active_return_annualized": 0.0,
        "tracking_error": 0.0,
        "information_ratio": 0.0,
        "beta_to_benchmark": 0.0,
        "alpha_to_benchmark": 0.0,
        "up_capture": np.nan,
        "down_capture": np.nan,
        "active_hit_rate": 0.0,
        "reason": reason,
    }


def _capture_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    market_up: bool,
) -> float:
    mask = benchmark_returns > 0 if market_up else benchmark_returns < 0
    if int(mask.sum()) == 0:
        return np.nan

    portfolio_slice = portfolio_returns[mask]
    benchmark_slice = benchmark_returns[mask]

    benchmark_compound = float((1.0 + benchmark_slice).prod() - 1.0)
    if np.isclose(benchmark_compound, 0.0):
        return np.nan

    portfolio_compound = float((1.0 + portfolio_slice).prod() - 1.0)
    return float(portfolio_compound / benchmark_compound)


def calculate_active_risk_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_ticker: str,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, float | int | bool | str]:
    """
    Compute benchmark-relative metrics used for active portfolio management.

    Metrics are calculated on date-aligned daily returns:
    - active return (total + annualized)
    - tracking error and information ratio
    - beta and Jensen alpha relative to benchmark
    - up/down market capture
    - active hit rate (share of periods with positive active return)
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return _empty_active_metrics(benchmark_ticker, reason="missing_return_series")

    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna(how="any")

    if aligned.shape[0] < 2:
        return _empty_active_metrics(benchmark_ticker, reason="insufficient_overlap")

    port = aligned["portfolio"].astype(float)
    bench = aligned["benchmark"].astype(float)
    active_daily = port - bench

    benchmark_total_return = float((1.0 + bench).prod() - 1.0)
    benchmark_annualized_return = calculate_annualized_return(bench, periods_per_year)

    portfolio_total_return = float((1.0 + port).prod() - 1.0)
    portfolio_annualized_return = calculate_annualized_return(port, periods_per_year)

    active_return_total = portfolio_total_return - benchmark_total_return
    active_return_annualized = portfolio_annualized_return - benchmark_annualized_return

    tracking_error = float(active_daily.std() * np.sqrt(periods_per_year))
    information_ratio = (
        float(active_return_annualized / tracking_error) if tracking_error > 0 else 0.0
    )

    beta_to_benchmark = float(calculate_beta(port, bench))
    alpha_to_benchmark = float(
        calculate_alpha(
            asset_returns=port,
            market_returns=bench,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
    )

    up_capture = _capture_ratio(port, bench, market_up=True)
    down_capture = _capture_ratio(port, bench, market_up=False)
    active_hit_rate = float((active_daily > 0).mean())

    return {
        "benchmark_ticker": benchmark_ticker,
        "benchmark_available": True,
        "benchmark_obs": int(aligned.shape[0]),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "active_return_total": float(active_return_total),
        "active_return_annualized": float(active_return_annualized),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta_to_benchmark": beta_to_benchmark,
        "alpha_to_benchmark": alpha_to_benchmark,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "active_hit_rate": active_hit_rate,
        "reason": "",
    }


def calculate_return_contribution(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Build simple arithmetic return attribution from weighted daily returns.

    This representation is robust and additive in arithmetic space:
    contribution_i ~= sum_t(w_i * r_i,t)
    """
    output_columns = [
        "Ticker",
        "Weight",
        "TotalContributionApprox",
        "AnnualizedContributionApprox",
        "ContributionShare",
        "MeanDailyContribution",
    ]
    if asset_returns.empty:
        return pd.DataFrame(columns=output_columns)

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size != asset_returns.shape[1]:
        raise ValueError("Weights length must match number of return columns.")

    weighted_returns = asset_returns.multiply(weights_array, axis=1)
    portfolio_daily = weighted_returns.sum(axis=1)
    total_arithmetic = float(portfolio_daily.sum())

    rows: list[dict[str, float | str]] = []
    for idx, ticker in enumerate(asset_returns.columns):
        ticker_series = weighted_returns.iloc[:, idx]
        total_contribution = float(ticker_series.sum())
        mean_daily = float(ticker_series.mean())
        annualized_contribution = float(mean_daily * periods_per_year)
        contribution_share = (
            float(total_contribution / total_arithmetic)
            if not np.isclose(total_arithmetic, 0.0)
            else 0.0
        )
        rows.append(
            {
                "Ticker": ticker,
                "Weight": float(weights_array[idx]),
                "TotalContributionApprox": total_contribution,
                "AnnualizedContributionApprox": annualized_contribution,
                "ContributionShare": contribution_share,
                "MeanDailyContribution": mean_daily,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=output_columns)

    return result.sort_values(
        by="TotalContributionApprox",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def calculate_risk_contribution(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Decompose annualized portfolio volatility into per-asset contributions."""
    output_columns = [
        "Ticker",
        "Weight",
        "MarginalVolatility",
        "RiskContribution",
        "RiskBudgetPct",
    ]
    if asset_returns.empty:
        return pd.DataFrame(columns=output_columns)

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size != asset_returns.shape[1]:
        raise ValueError("Weights length must match number of return columns.")

    cov_matrix = asset_returns.cov().to_numpy(dtype=float) * periods_per_year
    portfolio_variance = float(weights_array.T @ cov_matrix @ weights_array)
    portfolio_volatility = float(np.sqrt(max(portfolio_variance, 0.0)))

    if np.isclose(portfolio_volatility, 0.0):
        return pd.DataFrame(
            {
                "Ticker": asset_returns.columns.tolist(),
                "Weight": [float(value) for value in weights_array],
                "MarginalVolatility": [0.0] * len(weights_array),
                "RiskContribution": [0.0] * len(weights_array),
                "RiskBudgetPct": [0.0] * len(weights_array),
            }
        )

    marginal_volatility = (cov_matrix @ weights_array) / portfolio_volatility
    risk_contribution = weights_array * marginal_volatility
    risk_budget_pct = risk_contribution / portfolio_volatility

    result = pd.DataFrame(
        {
            "Ticker": asset_returns.columns.tolist(),
            "Weight": [float(value) for value in weights_array],
            "MarginalVolatility": [float(value) for value in marginal_volatility],
            "RiskContribution": [float(value) for value in risk_contribution],
            "RiskBudgetPct": [float(value) for value in risk_budget_pct],
        }
    )
    return result.sort_values(by="RiskBudgetPct", ascending=False).reset_index(drop=True)
