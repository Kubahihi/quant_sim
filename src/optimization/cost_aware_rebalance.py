from __future__ import annotations

from typing import Any, Optional

import cvxpy as cp
from loguru import logger
import numpy as np
import pandas as pd

from .constraints import build_weight_bounds, validate_weight_solution
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    PortfolioEstimates,
    resolve_portfolio_estimates,
)


def _solve_cost_aware_problem(problem: cp.Problem) -> tuple[Optional[str], str]:
    """Solve the convex rebalance with deterministic, robust fallbacks."""
    installed = set(cp.installed_solvers())
    errors: list[str] = []
    for solver in ("CLARABEL", "OSQP", "SCS"):
        if solver not in installed:
            continue
        kwargs: dict[str, Any] = {
            "solver": solver,
            "warm_start": True,
            "verbose": False,
        }
        if solver == "OSQP":
            kwargs.update({
                "eps_abs": 1e-9,
                "eps_rel": 1e-9,
                "max_iter": 100_000,
                "polishing": True,
            })
        elif solver == "SCS":
            kwargs.update({"eps": 1e-7, "max_iters": 100_000})
        try:
            problem.solve(**kwargs)
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return solver, f"{solver}: {problem.status}"
        errors.append(f"{solver}: status={problem.status}")

    message = "; ".join(errors)
    if not message:
        message = "no compatible CVXPY solver is installed."
    return None, message


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

    def turnover(values: np.ndarray) -> float:
        return float(np.sum(np.abs(values - base_weights)))

    lower_bounds = np.asarray([lower for lower, _ in bounds], dtype=float)
    upper_bounds = np.asarray([upper for _, upper in bounds], dtype=float)
    weights = cp.Variable(n_assets, name="cost_aware_weights")
    trades = weights - base_weights
    turnover_expression = cp.norm1(trades)
    risk = cp.quad_form(weights, cp.psd_wrap(covariance))
    utility = (
        expected_returns @ weights
        - risk_penalty * risk
        - tx_cost_rate * turnover_expression
    )
    problem = cp.Problem(
        cp.Maximize(utility),
        [
            cp.sum(weights) == 1.0,
            weights >= lower_bounds,
            weights <= upper_bounds,
            turnover_expression <= turnover_cap,
        ],
    )

    # Seed the conic solver with a bounded point when the current portfolio
    # breaches a newly introduced position cap.
    equal_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    x0 = base_weights.copy()
    if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
        x0 = equal_weights
    weights.value = x0

    solver, solver_message = _solve_cost_aware_problem(problem)
    if solver is None or weights.value is None:
        logger.warning(f"Cost-aware optimization failed: {solver_message}")
        return {
            "weights": np.array([], dtype=float),
            "current_weights": base_weights,
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": solver_message,
            "estimation": estimates.metadata(),
        }

    try:
        optimized_weights = validate_weight_solution(
            np.asarray(weights.value, dtype=float).reshape(-1),
            bounds,
            tolerance=1e-6,
        )
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
        "message": solver_message,
        "estimation": estimates.metadata(),
    }
