from __future__ import annotations

from typing import Any, Optional

from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from src.utils.rates import annual_effective_to_arithmetic

from .constraints import build_weight_bounds, validate_weight_solution
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    PortfolioEstimates,
    resolve_portfolio_estimates,
)


def _risk_free_rate(value: float) -> float:
    """Normalize the annual effective risk-free rate used by Sharpe ratios."""
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk_free_rate must be a finite number greater than -100%.") from exc
    if not np.isfinite(rate) or rate <= -1.0:
        raise ValueError("risk_free_rate must be a finite number greater than -100%.")
    return rate


def _zero_variance_candidate(
    estimates: PortfolioEstimates,
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, float] | None:
    """Find a feasible all-deterministic allocation, if the mandate permits one.

    A declared cash proxy has exactly zero estimated variance.  It must be
    considered explicitly because a smooth Sharpe objective cannot represent
    the zero-over-zero Sharpe of an allocation earning exactly the risk-free
    rate.  Without this candidate, SLSQP can incorrectly return a negative
    Sharpe risky mix even though a zero-Sharpe cash allocation is feasible.
    """
    deterministic_names = set(estimates.deterministic_assets)
    deterministic_indices = np.asarray(
        [
            index
            for index, symbol in enumerate(estimates.symbols)
            if symbol in deterministic_names
        ],
        dtype=int,
    )
    if deterministic_indices.size == 0:
        return None

    deterministic_bounds = [bounds[index] for index in deterministic_indices]
    result = linprog(
        c=-estimates.mean_returns[deterministic_indices],
        A_eq=np.ones((1, deterministic_indices.size), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=deterministic_bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        return None

    weights = np.zeros(len(estimates.symbols), dtype=float)
    weights[deterministic_indices] = np.asarray(result.x, dtype=float)
    weights = validate_weight_solution(weights, bounds)
    expected_return = float(weights @ estimates.mean_returns)
    return weights, expected_return


def _optimization_result(
    *,
    estimates: PortfolioEstimates,
    success: bool,
    message: str,
    weights: np.ndarray | None = None,
    risk_free_rate: float | None = None,
) -> dict[str, Any]:
    """Build one consistent public optimizer result."""
    if not success or weights is None:
        return {
            "weights": np.array([], dtype=float),
            "symbols": list(estimates.symbols),
            "expected_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "success": False,
            "status": "failed",
            "message": message,
            "estimation": estimates.metadata(),
        }

    expected_return = float(weights @ estimates.mean_returns)
    variance = float(weights @ estimates.covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (
        (expected_return - float(risk_free_rate)) / volatility
        if volatility > 0.0
        else 0.0
    )
    return {
        "weights": weights,
        "symbols": list(estimates.symbols),
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "success": True,
        "status": "optimal",
        "message": message,
        "estimation": estimates.metadata(),
    }


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
    risk_free = annual_effective_to_arithmetic(_risk_free_rate(risk_free_rate), estimates.trading_days)
    bounds = build_weight_bounds(
        n_assets,
        allow_short=allow_short,
        max_weight=max_weight,
    )

    deterministic_candidate = _zero_variance_candidate(estimates, bounds)
    rate_tolerance = 1e-10 * max(1.0, abs(risk_free))
    if deterministic_candidate is not None:
        _, deterministic_return = deterministic_candidate
        if deterministic_return > risk_free + rate_tolerance:
            return _optimization_result(
                estimates=estimates,
                success=False,
                message=(
                    "A feasible zero-volatility allocation has an expected return above "
                    "risk_free_rate, so its finite maximum Sharpe ratio is undefined. "
                    "Align the risk-free rate with the cash proxy before optimizing."
                ),
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
        if (
            deterministic_candidate is not None
            and abs(deterministic_candidate[1] - risk_free) <= rate_tolerance
        ):
            fallback = _optimization_result(
                estimates=estimates,
                success=True,
                message=(
                    "Feasible zero-volatility allocation selected after the numerical "
                    "tangent-portfolio solve did not converge."
                ),
                weights=deterministic_candidate[0],
                risk_free_rate=risk_free,
            )
            fallback.update({
                "success": False,
                "status": "fallback_feasible",
                "fallback_weights": fallback["weights"].copy(),
                "weights": np.array([], dtype=float),
                "solver_message": str(result.message),
            })
            return fallback
        return _optimization_result(
            estimates=estimates,
            success=False,
            message=str(result.message),
        )

    try:
        optimal_weights = validate_weight_solution(result.x, bounds)
    except ValueError as exc:
        logger.warning(f"Maximum-Sharpe solution rejected: {exc}")
        return _optimization_result(
            estimates=estimates,
            success=False,
            message=str(exc),
        )

    candidate = _optimization_result(
        estimates=estimates,
        success=True,
        message=str(result.message),
        weights=optimal_weights,
        risk_free_rate=risk_free,
    )
    if (
        deterministic_candidate is not None
        and abs(deterministic_candidate[1] - risk_free) <= rate_tolerance
        and float(candidate["sharpe_ratio"]) < 0.0
    ):
        return _optimization_result(
            estimates=estimates,
            success=True,
            message=(
                "Feasible zero-volatility allocation selected over a negative-Sharpe "
                "risky allocation."
            ),
            weights=deterministic_candidate[0],
            risk_free_rate=risk_free,
        )
    return candidate
