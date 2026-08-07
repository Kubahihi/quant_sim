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


def optimize_maximum_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
) -> dict[str, Any]:
    """Optimize maximum Sharpe using the shared conservative input estimates."""
    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    n_assets = len(estimates.symbols)
    mean_returns = estimates.mean_returns
    covariance = estimates.covariance
    risk_free = float(risk_free_rate)
    bounds = build_weight_bounds(
        n_assets,
        allow_short=allow_short,
        max_weight=max_weight,
    )

    def negative_sharpe(weights: np.ndarray) -> float:
        expected_return = float(weights @ mean_returns)
        variance = float(weights @ covariance @ weights)
        volatility = np.sqrt(max(variance, 0.0))
        if volatility <= 1e-14:
            return 1e10
        return -((expected_return - risk_free) / volatility)

    def negative_sharpe_gradient(weights: np.ndarray) -> np.ndarray:
        expected_excess_return = float(weights @ mean_returns) - risk_free
        variance = max(float(weights @ covariance @ weights), 1e-28)
        volatility = np.sqrt(variance)
        gradient = (
            mean_returns / volatility
            - expected_excess_return * (covariance @ weights) / (volatility ** 3)
        )
        return -gradient

    ones = np.ones(n_assets, dtype=float)
    constraints = [{
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda _weights: ones,
    }]
    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    result = minimize(
        negative_sharpe,
        initial_weights,
        jac=negative_sharpe_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning(f"Maximum-Sharpe optimization failed: {result.message}")
        return {
            "weights": np.array([], dtype=float),
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": str(result.message),
            "estimation": estimates.metadata(),
        }

    try:
        optimal_weights = validate_weight_solution(result.x, bounds)
    except ValueError as exc:
        logger.warning(f"Maximum-Sharpe solution rejected: {exc}")
        return {
            "weights": np.array([], dtype=float),
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "message": str(exc),
            "estimation": estimates.metadata(),
        }

    expected_return = float(optimal_weights @ mean_returns)
    variance = float(optimal_weights @ covariance @ optimal_weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (
        (expected_return - risk_free) / volatility if volatility > 0 else 0.0
    )
    return {
        "weights": optimal_weights,
        "symbols": list(estimates.symbols),
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "success": True,
        "message": str(result.message),
        "estimation": estimates.metadata(),
    }
