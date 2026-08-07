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


# Treat portfolios with effectively zero sample volatility as zero-Sharpe.
_VOLATILITY_EPS: float = 1e-8


def optimize_minimum_variance(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
) -> dict[str, Any]:
    """Optimize minimum variance using the shared, auditable estimates."""
    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    n_assets = len(estimates.symbols)
    covariance = estimates.covariance

    effective_max_weight = max_weight
    if max_weight is not None:
        requested_max_weight = float(max_weight)
        if not np.isfinite(requested_max_weight) or requested_max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        if not allow_short and requested_max_weight * n_assets < 1.0:
            effective_max_weight = 1.0 / n_assets
        else:
            effective_max_weight = requested_max_weight
    if effective_max_weight != max_weight:
        logger.warning(
            f"max_weight={max_weight} is infeasible for {n_assets} assets "
            f"and was relaxed to {float(effective_max_weight):.6f}."
        )
    bounds = build_weight_bounds(
        n_assets,
        allow_short=allow_short,
        max_weight=effective_max_weight,
    )

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def portfolio_variance_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * covariance @ weights

    ones = np.ones(n_assets, dtype=float)
    constraints = [{
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda _weights: ones,
    }]
    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    result = minimize(
        portfolio_variance,
        initial_weights,
        jac=portfolio_variance_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        logger.warning(f"Minimum-variance optimization failed: {result.message}")
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
        logger.warning(f"Minimum-variance solution rejected: {exc}")
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

    expected_return = float(optimal_weights @ estimates.mean_returns)
    variance = portfolio_variance(optimal_weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sample_variance = float(
        optimal_weights @ estimates.sample_covariance @ optimal_weights
    )
    sample_volatility = float(np.sqrt(max(sample_variance, 0.0)))
    sharpe_ratio = (
        (expected_return - float(risk_free_rate)) / volatility
        if sample_volatility > _VOLATILITY_EPS and volatility > _VOLATILITY_EPS
        else 0.0
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
