from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def optimize_cost_aware_rebalance(
    returns: pd.DataFrame,
    current_weights: np.ndarray | list[float],
    risk_free_rate: float = 0.03,
    max_weight: float = 0.35,
    turnover_limit: float = 0.30,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 3.0,
) -> Dict[str, Any]:
    """
    Rebalance optimizer with explicit turnover and transaction-cost awareness.

    Objective (maximize):
      expected_return - risk_aversion * variance - transaction_cost * turnover
    """
    if returns.empty:
        return {
            "success": False,
            "message": "returns are empty",
            "weights": np.array([]),
            "symbols": [],
        }

    n_assets = int(returns.shape[1])
    raw_weights = np.asarray(current_weights, dtype=float)
    if raw_weights.size != n_assets:
        raise ValueError("Current weights length must match number of return columns.")

    total_weight = float(raw_weights.sum())
    if np.isclose(total_weight, 0.0):
        base_weights = np.array([1.0 / n_assets] * n_assets, dtype=float)
    else:
        base_weights = raw_weights / total_weight

    ann_mean_returns = returns.mean().to_numpy(dtype=float) * TRADING_DAYS
    ann_cov = returns.cov().to_numpy(dtype=float) * TRADING_DAYS
    tx_cost_rate = max(0.0, float(transaction_cost_bps)) / 10_000.0
    risk_penalty = max(0.0, float(risk_aversion))

    max_w_requested = float(max_weight)
    min_feasible = 1.0 / n_assets
    max_w_effective = max(max_w_requested, min_feasible)
    bounds = [(0.0, max_w_effective) for _ in range(n_assets)]

    turnover_cap = max(0.0, float(turnover_limit))

    def _turnover(weights: np.ndarray) -> float:
        return float(np.sum(np.abs(weights - base_weights)))

    def _objective(weights: np.ndarray) -> float:
        expected_return = float(weights @ ann_mean_returns)
        variance = float(weights.T @ ann_cov @ weights)
        turnover = _turnover(weights)
        transaction_cost_drag = tx_cost_rate * turnover
        utility = expected_return - risk_penalty * variance - transaction_cost_drag
        return -utility

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if turnover_cap > 0:
        constraints.append({"type": "ineq", "fun": lambda w: float(turnover_cap - _turnover(w))})

    result = minimize(
        _objective,
        x0=base_weights.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    optimized_weights = np.asarray(result.x, dtype=float) if result.success else base_weights
    optimized_weights = np.clip(optimized_weights, 0.0, None)
    weight_sum = float(optimized_weights.sum())
    if not np.isclose(weight_sum, 0.0):
        optimized_weights = optimized_weights / weight_sum

    expected_return = float(optimized_weights @ ann_mean_returns)
    variance = float(optimized_weights.T @ ann_cov @ optimized_weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    turnover = _turnover(optimized_weights)
    transaction_cost_drag = float(tx_cost_rate * turnover)
    sharpe_ratio = (
        float((expected_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
    )

    utility = expected_return - risk_penalty * variance - transaction_cost_drag
    status_message = result.message if not result.success else "ok"
    if max_w_effective > max_w_requested + 1e-12:
        status_message = (
            f"{status_message}; max_weight raised to {max_w_effective:.4f} for feasibility"
        )

    return {
        "weights": optimized_weights,
        "symbols": returns.columns.tolist(),
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "turnover": float(turnover),
        "turnover_limit": turnover_cap,
        "max_weight": max_w_effective,
        "transaction_cost_bps": float(transaction_cost_bps),
        "transaction_cost_drag": transaction_cost_drag,
        "risk_aversion": risk_penalty,
        "utility_score": float(utility),
        "success": bool(result.success),
        "message": str(status_message),
    }
