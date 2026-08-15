from __future__ import annotations

from typing import Any, Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .constraints import build_weight_bounds, validate_weight_solution
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    PortfolioEstimates,
    resolve_portfolio_estimates,
)


def _calculate_diversification_metrics(weights: np.ndarray) -> tuple[float, float]:
    concentration = float(np.sum(np.square(weights)))
    if concentration <= 0:
        return 0.0, 0.0
    effective_holdings = 1.0 / concentration
    diversification_score = effective_holdings / len(weights)
    return diversification_score, effective_holdings


def _format_top_holdings(
    weights: np.ndarray,
    symbols: Sequence[str],
    top_n: int = 3,
) -> str:
    ranked_idx = np.argsort(np.abs(weights))[::-1][:top_n]
    return ", ".join(
        f"{symbols[idx]} {weights[idx]:.0%}"
        for idx in ranked_idx
        if abs(weights[idx]) > 1e-12
    )


def calculate_portfolio_statistics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.03,
    symbols: Optional[list[str]] = None,
) -> dict[str, object]:
    """Calculate comparable metrics from an explicit set of portfolio inputs."""
    values = np.asarray(weights, dtype=float)
    means = np.asarray(mean_returns, dtype=float)
    covariance = np.asarray(cov_matrix, dtype=float)
    if values.ndim != 1 or means.shape != values.shape:
        raise ValueError("weights and mean_returns must be aligned one-dimensional arrays.")
    if covariance.shape != (values.size, values.size):
        raise ValueError("cov_matrix shape must match the weight vector.")

    portfolio_return = float(values @ means)
    variance = float(values @ covariance @ values)
    portfolio_volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (
        (portfolio_return - float(risk_free_rate)) / portfolio_volatility
        if portfolio_volatility > 0
        else 0.0
    )
    diversification_score, effective_holdings = _calculate_diversification_metrics(values)
    metrics: dict[str, object] = {
        "return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "diversification_score": diversification_score,
        "effective_holdings": effective_holdings,
        "max_weight": float(np.max(np.abs(values))),
    }
    if symbols is not None:
        metrics["top_holdings"] = _format_top_holdings(values, symbols)
    return metrics


def _project_to_capped_simplex(
    weights: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """Project rows onto the long-only simplex with a uniform upper bound."""
    upper = float(max_weight)
    n_assets = weights.shape[1]
    build_weight_bounds(n_assets, allow_short=False, max_weight=upper)
    if upper >= 1.0:
        return weights
    if np.isclose(upper, 1.0 / n_assets, atol=1e-12, rtol=0.0):
        return np.full_like(weights, 1.0 / n_assets)

    lower_lambda = np.min(weights - upper, axis=1)
    upper_lambda = np.max(weights, axis=1)
    for _ in range(64):
        midpoint = (lower_lambda + upper_lambda) * 0.5
        projected_sum = np.clip(
            weights - midpoint[:, None], 0.0, upper
        ).sum(axis=1)
        too_large = projected_sum > 1.0
        lower_lambda = np.where(too_large, midpoint, lower_lambda)
        upper_lambda = np.where(too_large, upper_lambda, midpoint)
    projected = np.clip(weights - upper_lambda[:, None], 0.0, upper)
    row_sums = projected.sum(axis=1, keepdims=True)
    return np.divide(
        projected,
        row_sums,
        out=np.full_like(projected, 1.0 / n_assets),
        where=row_sums > 0,
    )


def sample_portfolio_cloud(
    returns: pd.DataFrame,
    n_samples: int = 2500,
    risk_free_rate: float = 0.03,
    random_seed: Optional[int] = 42,
    max_weight: Optional[float] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
) -> pd.DataFrame:
    """Sample long-only portfolios using the same inputs as the optimizers."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    n_assets = len(estimates.symbols)
    rng = np.random.default_rng(random_seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_samples)
    if max_weight is not None:
        weights = _project_to_capped_simplex(weights, float(max_weight))
    portfolio_returns = weights @ estimates.mean_returns
    weighted_covariance = weights @ estimates.covariance
    portfolio_variance = np.einsum(
        "ij,ij->i", weighted_covariance, weights, optimize=True
    )
    portfolio_volatility = np.sqrt(
        np.clip(portfolio_variance, a_min=0.0, a_max=None)
    )
    sharpe_ratio = np.divide(
        portfolio_returns - float(risk_free_rate),
        portfolio_volatility,
        out=np.zeros_like(portfolio_returns),
        where=portfolio_volatility > 0,
    )
    concentration = np.sum(np.square(weights), axis=1)
    effective_holdings = np.divide(
        1.0,
        concentration,
        out=np.zeros_like(concentration),
        where=concentration > 0,
    )
    cloud = pd.DataFrame({
        "expected_return": portfolio_returns,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "diversification_score": effective_holdings / n_assets,
        "effective_holdings": effective_holdings,
        "max_weight": np.max(weights, axis=1),
        "top_holdings": [
            _format_top_holdings(sample_weights, estimates.symbols)
            for sample_weights in weights
        ],
    })
    cloud.attrs["estimation"] = estimates.metadata()
    logger.info(f"Sampled {len(cloud)} portfolios for trade-off visualization")
    return cloud


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    risk_free_rate: float = 0.03,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
) -> list[dict[str, Any]]:
    """Calculate only the efficient branch, beginning at global min variance."""
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2.")

    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    n_assets = len(estimates.symbols)
    mean_returns = estimates.mean_returns
    covariance = estimates.covariance
    symbols = list(estimates.symbols)
    bounds = build_weight_bounds(
        n_assets,
        allow_short=allow_short,
        max_weight=max_weight,
    )

    if n_assets == 1:
        weights = validate_weight_solution(np.ones(1, dtype=float), bounds)
        metrics = calculate_portfolio_statistics(
            weights,
            mean_returns,
            covariance,
            risk_free_rate=risk_free_rate,
            symbols=symbols,
        )
        return [{
            "weights": weights,
            **metrics,
            "estimation": estimates.metadata(),
        }]

    ones = np.ones(n_assets, dtype=float)

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    def portfolio_variance_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * covariance @ weights

    fully_invested_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda _weights: ones,
    }
    equal_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    minimum_variance_result = minimize(
        portfolio_variance,
        equal_weights,
        jac=portfolio_variance_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=(fully_invested_constraint,),
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not minimum_variance_result.success:
        raise RuntimeError(
            f"minimum-variance anchor failed: {minimum_variance_result.message}"
        )
    minimum_variance_weights = validate_weight_solution(
        minimum_variance_result.x, bounds
    )
    minimum_variance_return = float(minimum_variance_weights @ mean_returns)

    maximum_return_result = linprog(
        c=-mean_returns,
        A_eq=ones.reshape(1, -1),
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not maximum_return_result.success:
        raise RuntimeError(
            f"maximum-return endpoint failed: {maximum_return_result.message}"
        )
    maximum_return_weights = validate_weight_solution(maximum_return_result.x, bounds)
    maximum_return = float(maximum_return_weights @ mean_returns)

    target_returns = np.linspace(
        minimum_variance_return,
        maximum_return,
        n_points,
    )
    frontier_points: list[dict[str, Any]] = []
    initial_weights = minimum_variance_weights

    # Every interior frontier point has the same quadratic objective and
    # constraint matrix; only the target-return equality changes.  Preparing
    # OSQP once lets it reuse its factorization and the previous point as a
    # warm start instead of asking SLSQP to rebuild the problem for every
    # target.  Keep the established SLSQP path as a per-point fallback so a
    # missing optional solver or a numerically difficult endpoint cannot make
    # the frontier less robust.
    frontier_solver: Any = None
    frontier_lower: Optional[np.ndarray] = None
    frontier_upper: Optional[np.ndarray] = None
    if len(target_returns) > 2:
        try:
            import osqp
            from scipy import sparse

            lower_bounds = np.asarray([lower for lower, _ in bounds], dtype=float)
            upper_bounds = np.asarray([upper for _, upper in bounds], dtype=float)
            constraint_matrix = sparse.vstack(
                (
                    sparse.csc_matrix(ones.reshape(1, -1)),
                    sparse.csc_matrix(mean_returns.reshape(1, -1)),
                    sparse.eye(n_assets, format="csc"),
                ),
                format="csc",
            )
            first_target = float(target_returns[1])
            frontier_lower = np.concatenate(
                (np.array([1.0, first_target]), lower_bounds)
            )
            frontier_upper = np.concatenate(
                (np.array([1.0, first_target]), upper_bounds)
            )
            frontier_solver = osqp.OSQP()
            frontier_solver.setup(
                P=sparse.csc_matrix(2.0 * covariance),
                q=np.zeros(n_assets, dtype=float),
                A=constraint_matrix,
                l=frontier_lower,
                u=frontier_upper,
                eps_abs=1e-9,
                eps_rel=1e-9,
                max_iter=100_000,
                polishing=True,
                verbose=False,
            )
            frontier_solver.warm_start(x=minimum_variance_weights)
        except Exception as exc:
            logger.debug(f"Reusable frontier solver unavailable: {exc}")
            frontier_solver = None

    for index, target_return in enumerate(target_returns):
        if index == 0:
            weights = minimum_variance_weights
        elif index == len(target_returns) - 1:
            weights = maximum_return_weights
        else:
            weights = None
            if (
                frontier_solver is not None
                and frontier_lower is not None
                and frontier_upper is not None
            ):
                frontier_lower[1] = float(target_return)
                frontier_upper[1] = float(target_return)
                try:
                    frontier_solver.update(l=frontier_lower, u=frontier_upper)
                    frontier_solver.warm_start(x=initial_weights)
                    solver_result = frontier_solver.solve(raise_error=False)
                    if solver_result.info.status_val in {1, 2}:
                        candidate = validate_weight_solution(
                            solver_result.x,
                            bounds,
                            tolerance=1e-6,
                        )
                        if np.isclose(
                            float(candidate @ mean_returns),
                            target_return,
                            atol=1e-6,
                            rtol=0.0,
                        ):
                            weights = candidate
                except Exception as exc:
                    logger.debug(
                        f"Reusable frontier solve failed at {target_return:.6f}: {exc}"
                    )

            if weights is None:
                target_constraint = {
                    "type": "eq",
                    "fun": lambda values, target=target_return: float(
                        values @ mean_returns - target
                    ),
                    "jac": lambda _values: mean_returns,
                }
                result = minimize(
                    portfolio_variance,
                    initial_weights,
                    jac=portfolio_variance_gradient,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=(fully_invested_constraint, target_constraint),
                    options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
                )
                if not result.success:
                    logger.warning(
                        f"Frontier target {target_return:.6f} failed: {result.message}"
                    )
                    continue
                weights = validate_weight_solution(
                    result.x,
                    bounds,
                    tolerance=1e-6,
                )
            if not np.isclose(
                float(weights @ mean_returns), target_return, atol=1e-6, rtol=0.0
            ):
                logger.warning(
                    f"Frontier target {target_return:.6f} rejected for residual error"
                )
                continue

        initial_weights = weights
        metrics = calculate_portfolio_statistics(
            weights,
            mean_returns,
            covariance,
            risk_free_rate=risk_free_rate,
            symbols=symbols,
        )
        frontier_points.append({
            "weights": weights,
            **metrics,
            "estimation": estimates.metadata(),
        })

    if not frontier_points:
        raise RuntimeError("efficient frontier produced no valid portfolios.")
    logger.info(f"Calculated {len(frontier_points)} efficient frontier points")
    return frontier_points
