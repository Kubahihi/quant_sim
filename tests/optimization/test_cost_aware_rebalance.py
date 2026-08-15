from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import estimate_portfolio_inputs, optimize_cost_aware_rebalance


def _sample_returns(periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "AAA": rng.normal(0.0007, 0.011, periods),
            "BBB": rng.normal(0.0004, 0.009, periods),
            "CCC": rng.normal(0.0005, 0.010, periods),
        },
        index=dates,
    )


def _large_sample_returns(assets: int, periods: int = 504) -> pd.DataFrame:
    rng = np.random.default_rng(12345 + assets)
    factors = rng.normal(0.0002, 0.006, size=(periods, 3))
    loadings = rng.normal(0.5, 0.2, size=(3, assets))
    residuals = rng.normal(0.0001, 0.008, size=(periods, assets))
    return pd.DataFrame(
        factors @ loadings + residuals,
        columns=[f"A{index}" for index in range(assets)],
    )


def test_cost_aware_rebalance_respects_constraints():
    returns = _sample_returns()
    current_weights = np.array([0.65, 0.25, 0.10], dtype=float)

    result = optimize_cost_aware_rebalance(
        returns=returns,
        current_weights=current_weights,
        max_weight=0.60,
        turnover_limit=0.40,
        transaction_cost_bps=12.0,
        risk_aversion=2.5,
    )

    assert result["success"] is True
    weights = np.asarray(result["weights"], dtype=float)
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)
    assert np.all(weights >= -1e-10)
    assert np.max(weights) <= float(result["max_weight"]) + 1e-8
    assert float(result["turnover"]) <= float(result["turnover_limit"]) + 1e-6


def test_cost_aware_rebalance_rejects_infeasible_max_weight():
    returns = _sample_returns()
    current_weights = np.array([0.5, 0.3, 0.2], dtype=float)

    with pytest.raises(ValueError, match="infeasible"):
        optimize_cost_aware_rebalance(
            returns=returns,
            current_weights=current_weights,
            max_weight=0.10,
            turnover_limit=2.0,
            transaction_cost_bps=0.0,
            risk_aversion=1.0,
        )


def test_cost_aware_rebalance_handles_empty_returns():
    with pytest.raises(ValueError, match="returns are empty"):
        optimize_cost_aware_rebalance(
            returns=pd.DataFrame(),
            current_weights=np.array([]),
        )


@pytest.mark.parametrize("assets", [50, 100])
def test_cost_aware_rebalance_converges_for_large_universe(assets: int):
    returns = _large_sample_returns(assets)
    estimates = estimate_portfolio_inputs(returns)
    current_weights = np.full(assets, 1.0 / assets, dtype=float)
    max_weight = 0.10
    turnover_limit = 0.50
    transaction_cost_bps = 12.0
    risk_aversion = 2.5

    result = optimize_cost_aware_rebalance(
        returns=returns,
        current_weights=current_weights,
        max_weight=max_weight,
        turnover_limit=turnover_limit,
        transaction_cost_bps=transaction_cost_bps,
        risk_aversion=risk_aversion,
        portfolio_estimates=estimates,
    )

    assert result["success"] is True, result["message"]
    assert "optimal" in result["message"]
    weights = np.asarray(result["weights"], dtype=float)
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-7, rtol=0.0)
    assert np.min(weights) >= -1e-7
    assert np.max(weights) <= max_weight + 1e-7
    assert result["turnover"] <= turnover_limit + 1e-6

    expected_return = float(weights @ estimates.mean_returns)
    variance = float(weights @ estimates.covariance @ weights)
    transaction_cost_drag = (
        transaction_cost_bps / 10_000.0 * float(np.abs(weights - current_weights).sum())
    )
    assert result["expected_return"] == pytest.approx(expected_return)
    assert result["transaction_cost_drag"] == pytest.approx(transaction_cost_drag)
    assert result["utility_score"] == pytest.approx(
        expected_return - risk_aversion * variance - transaction_cost_drag
    )
