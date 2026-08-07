from __future__ import annotations

from typing import Any, Optional

from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import build_weight_bounds, validate_weight_solution
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    PortfolioEstimates,
    resolve_portfolio_estimates,
)


def optimize_cost_aware_rebalance(
    returns: pd.DataFrame,
    current_weights: np.ndarray | list[float],
    risk_free_rate: float = 0.03,
    max_weight: float = 0.35,
    turnover_limit: float = 0.30,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 3.0,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
) -> dict[str, Any]:
    """Optimize a long-only rebalance after turnover and proportional costs."""
    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    n_assets = len(estimates.symbols)
    raw_weights = np.asarray(current_weights, dtype=float)
    if raw_weights.ndim != 1 or raw_weights.size != n_assets:
        raise ValueError("Current weights length must match number of return columns.")
    if not np.all(np.isfinite(raw_weights)):
        raise ValueError("Current weights must be finite.")
    if np.any(raw_weights < 0):
        raise ValueError("Current weights must be non-negative for a long-only rebalance.")

    total_weight = float(raw_weights.sum())
    if total_weight <= 0:
        base_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    else:
        base_weights = raw_weights / total_weight

    bounds = build_weight_bounds(
        n_assets,
        allow_short=False,
        max_weight=max_weight,
    )
    expected_returns = estimates.mean_returns
    covariance = estimates.covariance
    tx_cost_rate = max(0.0, float(transaction_cost_bps)) / 10_000.0
    risk_penalty = max(0.0, float(risk_aversion))
    turnover_cap = float(turnover_limit)
    if not np.isfinite(turnover_cap) or turnover_cap < 0:
        raise ValueError("turnover_limit must be non-negative.")

    def turnover(weights: np.ndarray) -> float:
        return float(np.sum(np.abs(weights - base_weights)))

    def objective(weights: np.ndarray) -> float:
        expected_return = float(weights @ expected_returns)
        variance = float(weights @ covariance @ weights)
        transaction_cost_drag = tx_cost_rate * turnover(weights)
        utility = expected_return - risk_penalty * variance - transaction_cost_drag
        return -utility

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    ]
    constraints.append({
        "type": "ineq",
        "fun": lambda weights: float(turnover_cap - turnover(weights)),
    })

    # SLSQP clips an infeasible starting point to the bounds. Equal weights are
    # feasible by construction and give the solver a safe alternative when the
    # current portfolio breaches a newly introduced position cap.
    equal_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    x0 = base_weights.copy()
    if any(
        value < lower or value > upper
        for value, (lower, upper) in zip(x0, bounds, strict=False)
    ):
        x0 = equal_weights

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning(f"Cost-aware optimization failed: {result.message}")
        return {
            "weights": np.array([], dtype=float),
            "current_weights": base_weights,
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": str(result.message),
            "estimation": estimates.metadata(),
        }

    try:
        optimized_weights = validate_weight_solution(result.x, bounds, tolerance=1e-6)
    except ValueError as exc:
        logger.warning(f"Cost-aware solution rejected: {exc}")
        return {
            "weights": np.array([], dtype=float),
            "current_weights": base_weights,
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": str(exc),
            "estimation": estimates.metadata(),
        }

    realized_turnover = turnover(optimized_weights)
    if realized_turnover > turnover_cap + 1e-6:
        message = "solver weights violate the turnover limit."
        logger.warning(f"Cost-aware solution rejected: {message}")
        return {
            "weights": np.array([], dtype=float),
            "current_weights": base_weights,
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": message,
            "estimation": estimates.metadata(),
        }

    expected_return = float(optimized_weights @ expected_returns)
    variance = float(optimized_weights @ covariance @ optimized_weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    transaction_cost_drag = float(tx_cost_rate * realized_turnover)
    sharpe_ratio = (
        (expected_return - float(risk_free_rate)) / volatility
        if volatility > 0
        else 0.0
    )
    utility = expected_return - risk_penalty * variance - transaction_cost_drag

    return {
        "weights": optimized_weights,
        "current_weights": base_weights,
        "symbols": list(estimates.symbols),
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "turnover": float(realized_turnover),
        "turnover_limit": turnover_cap,
        "max_weight": float(max_weight),
        "transaction_cost_bps": float(transaction_cost_bps),
        "transaction_cost_drag": transaction_cost_drag,
        "risk_aversion": risk_penalty,
        "utility_score": float(utility),
        "success": True,
        "message": str(result.message),
        "estimation": estimates.metadata(),
    }
