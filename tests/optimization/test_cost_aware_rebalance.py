from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization import optimize_cost_aware_rebalance


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


def test_cost_aware_rebalance_auto_relaxes_infeasible_max_weight():
    returns = _sample_returns()
    current_weights = np.array([0.5, 0.3, 0.2], dtype=float)

    result = optimize_cost_aware_rebalance(
        returns=returns,
        current_weights=current_weights,
        max_weight=0.10,
        turnover_limit=2.0,
        transaction_cost_bps=0.0,
        risk_aversion=1.0,
    )

    assert result["success"] is True
    assert float(result["max_weight"]) >= (1.0 / returns.shape[1])
    weights = np.asarray(result["weights"], dtype=float)
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-8)


def test_cost_aware_rebalance_handles_empty_returns():
    result = optimize_cost_aware_rebalance(
        returns=pd.DataFrame(),
        current_weights=np.array([]),
    )

    assert result["success"] is False
    assert "returns are empty" in str(result.get("message", ""))
