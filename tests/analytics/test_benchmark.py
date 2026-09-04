from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    calculate_active_risk_metrics,
    calculate_return_contribution,
    calculate_risk_contribution,
    calculate_portfolio_daily_returns,
)
from src.analytics.portfolio_metrics import build_portfolio_timeseries


def _build_sample_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    return pd.DataFrame(
        {
            "AAA": [0.010, -0.004, 0.006, 0.003, -0.002, 0.005, -0.001, 0.004],
            "BBB": [0.004, -0.002, 0.003, 0.002, -0.001, 0.002, 0.000, 0.001],
        },
        index=dates,
    )


def test_active_risk_metrics_are_computed_when_benchmark_is_available():
    returns = _build_sample_asset_returns()
    weights = np.array([0.6, 0.4], dtype=float)
    portfolio_returns = calculate_portfolio_daily_returns(returns, weights)
    benchmark_returns = pd.Series(
        [0.008, -0.003, 0.005, 0.001, -0.002, 0.004, -0.001, 0.002],
        index=returns.index,
        dtype=float,
        name="SPY",
    )

    metrics = calculate_active_risk_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        benchmark_ticker="SPY",
    )

    assert metrics["benchmark_available"] is True
    assert metrics["benchmark_ticker"] == "SPY"
    assert int(metrics["benchmark_obs"]) == len(returns.index)
    assert float(metrics["tracking_error"]) >= 0.0
    assert np.isfinite(float(metrics["information_ratio"]))
    assert np.isfinite(float(metrics["up_capture"]))
    assert np.isfinite(float(metrics["down_capture"]))

    active = portfolio_returns - benchmark_returns
    assert np.isclose(
        float(metrics["active_return_annualized"]),
        float(active.mean() * 252),
    )
    expected_ir = float(active.mean() * 252 / (active.std() * np.sqrt(252)))
    assert np.isclose(float(metrics["information_ratio"]), expected_ir)


def test_portfolio_returns_reject_missing_values_instead_of_treating_them_as_zero():
    returns = _build_sample_asset_returns()
    returns.loc[returns.index[0], "AAA"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "finite values"):
        calculate_portfolio_daily_returns(returns, np.array([0.6, 0.4]))


def test_portfolio_timeseries_drawdown_includes_initial_value_peak():
    result = build_portfolio_timeseries(pd.Series([-0.20, 0.10]), initial_value=100.0)

    assert result["value"].tolist() == pytest.approx([80.0, 88.0])
    assert result["drawdown"].tolist() == pytest.approx([-0.20, -0.12])


def test_active_risk_metrics_gracefully_handle_missing_overlap():
    portfolio_returns = pd.Series(dtype=float)
    benchmark_returns = pd.Series(dtype=float)
    metrics = calculate_active_risk_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        benchmark_ticker="SPY",
    )

    assert metrics["benchmark_available"] is False
    assert metrics["reason"] == "missing_return_series"
    assert int(metrics["benchmark_obs"]) == 0


def test_return_and_risk_contribution_tables_are_consistent():
    returns = _build_sample_asset_returns()
    weights = np.array([0.6, 0.4], dtype=float)

    return_contribution = calculate_return_contribution(returns, weights)
    assert list(return_contribution["Ticker"]) == ["AAA", "BBB"]
    assert "TotalContributionApprox" in return_contribution.columns

    approx_total = float(return_contribution["TotalContributionApprox"].sum())
    expected_total = float((returns * weights).sum(axis=1).sum())
    assert np.isclose(approx_total, expected_total, atol=1e-12)

    risk_contribution = calculate_risk_contribution(returns, weights)
    assert "RiskBudgetPct" in risk_contribution.columns
    assert np.isclose(float(risk_contribution["RiskBudgetPct"].sum()), 1.0, atol=1e-8)
