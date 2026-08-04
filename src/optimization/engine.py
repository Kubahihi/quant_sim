from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import cvxpy as cp
from loguru import logger
import numpy as np
import pandas as pd

from .constraint_sets import (
    PortfolioConstraintSet,
    build_constraint_report,
    build_constraint_set,
    validate_constraint_solution,
)
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    PortfolioEstimates,
    resolve_portfolio_estimates,
)


SUPPORTED_OBJECTIVES = {
    "minimum_variance",
    "maximum_utility",
    "target_volatility",
    "minimum_cvar",
    "minimum_tracking_error",
}


def _aligned_vector(
    values: Sequence[float] | np.ndarray | Mapping[str, float],
    symbols: Sequence[str],
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = [symbol for symbol in symbols if symbol not in values]
        if missing:
            raise ValueError(f"{name} is missing values for: {', '.join(missing)}.")
        vector = np.asarray([values[symbol] for symbol in symbols], dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size != len(symbols):
        raise ValueError(f"{name} length must match the return columns.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def _transaction_cost_rates(
    transaction_cost_bps: float | Sequence[float] | Mapping[str, float],
    symbols: Sequence[str],
) -> np.ndarray:
    if isinstance(transaction_cost_bps, Mapping):
        rates = _aligned_vector(transaction_cost_bps, symbols, "transaction_cost_bps")
    elif np.isscalar(transaction_cost_bps):
        rates = np.full(len(symbols), float(transaction_cost_bps), dtype=float)
    else:
        rates = _aligned_vector(transaction_cost_bps, symbols, "transaction_cost_bps")
    if np.any(rates < 0):
        raise ValueError("transaction_cost_bps must be non-negative.")
    return rates / 10_000.0


def _solve_problem(problem: cp.Problem, objective_name: str) -> tuple[Optional[str], list[str]]:
    installed = set(cp.installed_solvers())
    errors: list[str] = []
    solver_order = ["CLARABEL"]
    if objective_name != "target_volatility":
        solver_order.append("OSQP")
    solver_order.append("SCS")

    for solver in solver_order:
        if solver not in installed:
            continue
        try:
            kwargs: dict[str, Any] = {"solver": solver, "warm_start": True}
            if solver == "OSQP":
                kwargs.update({"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 100_000})
            elif solver == "SCS":
                kwargs.update({"eps": 1e-6, "max_iters": 100_000})
            problem.solve(**kwargs)
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return solver, errors
        errors.append(f"{solver}: status={problem.status}")
    return None, errors


def optimize_portfolio(
    returns: pd.DataFrame,
    *,
    objective: str = "maximum_utility",
    constraint_set: Optional[PortfolioConstraintSet] = None,
    strategy: Optional[Mapping[str, Any]] = None,
    asset_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    current_weights: Optional[Sequence[float] | np.ndarray] = None,
    max_weight: Optional[float] = None,
    allow_short: bool = False,
    turnover_limit: Optional[float] = None,
    transaction_cost_bps: float | Sequence[float] | Mapping[str, float] = 0.0,
    risk_free_rate: float = 0.03,
    risk_aversion: float = 3.0,
    target_volatility: Optional[float] = None,
    minimum_expected_return: Optional[float] = None,
    cvar_confidence: float = 0.95,
    benchmark_weights: Optional[Sequence[float] | np.ndarray | Mapping[str, float]] = None,
    expected_returns: Optional[
        Sequence[float] | np.ndarray | Mapping[str, float]
    ] = None,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
    expected_return_model: Optional[str] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
) -> dict[str, Any]:
    """Solve a mandate-aware convex portfolio construction problem."""
    objective_name = str(objective).strip().lower()
    if objective_name not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(SUPPORTED_OBJECTIVES)}.")

    estimates = resolve_portfolio_estimates(
        returns,
        portfolio_estimates=portfolio_estimates,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    symbols = list(estimates.symbols)
    n_assets = len(symbols)
    if constraint_set is None:
        constraints_spec = build_constraint_set(
            symbols,
            strategy=strategy,
            asset_metadata=asset_metadata,
            allow_short=allow_short,
            max_weight=max_weight,
            current_weights=current_weights,
            turnover_limit=turnover_limit,
        )
    else:
        constraints_spec = constraint_set
        if constraints_spec.symbols != tuple(symbols):
            raise ValueError("constraint_set symbols must match the return columns in order.")

    mean_returns = (
        estimates.mean_returns
        if expected_returns is None
        else _aligned_vector(expected_returns, symbols, "expected_returns")
    )
    resolved_return_model = (
        str(expected_return_model)
        if expected_return_model is not None
        else estimates.expected_return_method
        if expected_returns is None
        else "custom"
    )
    covariance = estimates.covariance
    cost_rates = _transaction_cost_rates(transaction_cost_bps, symbols)
    confidence_for_metrics = float(cvar_confidence)
    if not np.isfinite(confidence_for_metrics) or not 0.5 < confidence_for_metrics < 1.0:
        raise ValueError("cvar_confidence must be between 0.5 and 1.")
    current = constraints_spec.current_weights
    if current is None and np.any(cost_rates > 0):
        raise ValueError("transaction costs require current_weights.")

    weights = cp.Variable(n_assets, name="weights")
    risk = cp.quad_form(weights, cp.psd_wrap(covariance))
    cvx_constraints: list[cp.Constraint] = [
        cp.sum(weights) == 1.0,
        weights >= constraints_spec.lower_bounds,
        weights <= constraints_spec.upper_bounds,
    ]
    for group in constraints_spec.groups:
        group_weight = cp.sum(weights[list(group.indices)])
        if group.minimum is not None:
            cvx_constraints.append(group_weight >= group.minimum)
        if group.maximum is not None:
            cvx_constraints.append(group_weight <= group.maximum)
    if constraints_spec.beta is not None:
        portfolio_beta = constraints_spec.beta @ weights
        if constraints_spec.minimum_beta is not None:
            cvx_constraints.append(portfolio_beta >= constraints_spec.minimum_beta)
        if constraints_spec.maximum_beta is not None:
            cvx_constraints.append(portfolio_beta <= constraints_spec.maximum_beta)

    turnover_expression: cp.Expression | float = 0.0
    transaction_cost_expression: cp.Expression | float = 0.0
    if current is not None:
        trades = weights - current
        turnover_expression = cp.norm1(trades)
        transaction_cost_expression = cp.sum(cp.multiply(cost_rates, cp.abs(trades)))
        if constraints_spec.turnover_limit is not None:
            cvx_constraints.append(
                turnover_expression <= constraints_spec.turnover_limit
            )
    if minimum_expected_return is not None:
        cvx_constraints.append(mean_returns @ weights >= float(minimum_expected_return))

    cvar_expression: Optional[cp.Expression] = None
    benchmark: Optional[np.ndarray] = None
    if objective_name == "minimum_variance":
        problem_objective = cp.Minimize(risk + transaction_cost_expression)
    elif objective_name == "maximum_utility":
        risk_penalty = float(risk_aversion)
        if not np.isfinite(risk_penalty) or risk_penalty < 0:
            raise ValueError("risk_aversion must be non-negative.")
        problem_objective = cp.Maximize(
            mean_returns @ weights
            - risk_penalty * risk
            - transaction_cost_expression
        )
    elif objective_name == "target_volatility":
        if target_volatility is None or not np.isfinite(float(target_volatility)):
            raise ValueError("target_volatility objective requires a finite target.")
        target = float(target_volatility)
        if target <= 0:
            raise ValueError("target_volatility must be positive.")
        cvx_constraints.append(risk <= target ** 2)
        problem_objective = cp.Maximize(
            mean_returns @ weights - transaction_cost_expression
        )
    elif objective_name == "minimum_cvar":
        confidence = confidence_for_metrics
        value_at_risk = cp.Variable(name="value_at_risk")
        scenario_losses = -estimates.returns.to_numpy(dtype=float) @ weights
        cvar_expression = value_at_risk + cp.sum(
            cp.pos(scenario_losses - value_at_risk)
        ) / (len(estimates.returns) * (1.0 - confidence))
        problem_objective = cp.Minimize(
            cvar_expression + transaction_cost_expression
        )
    else:
        if benchmark_weights is None:
            raise ValueError("minimum_tracking_error requires benchmark_weights.")
        benchmark = _aligned_vector(
            benchmark_weights, symbols, "benchmark_weights"
        )
        if np.any(benchmark < 0) or not np.isclose(float(benchmark.sum()), 1.0, atol=1e-8):
            raise ValueError("benchmark_weights must be non-negative and sum to one.")
        active_weights = weights - benchmark
        tracking_variance = cp.quad_form(
            active_weights, cp.psd_wrap(covariance)
        )
        problem_objective = cp.Minimize(
            tracking_variance + transaction_cost_expression
        )

    problem = cp.Problem(problem_objective, cvx_constraints)
    solver, solver_errors = _solve_problem(problem, objective_name)
    if solver is None or weights.value is None:
        message = "; ".join(solver_errors) or f"status={problem.status}"
        logger.warning(f"Portfolio optimization failed: {message}")
        return {
            "success": False,
            "weights": np.array([], dtype=float),
            "symbols": symbols,
            "objective": objective_name,
            "status": str(problem.status),
            "message": message,
            "warnings": list(constraints_spec.warnings),
            "estimation": estimates.metadata(),
        }

    try:
        optimal_weights = validate_constraint_solution(
            np.asarray(weights.value, dtype=float).reshape(-1),
            constraints_spec,
            tolerance=2e-5,
        )
    except ValueError as exc:
        logger.warning(f"Portfolio solution rejected: {exc}")
        return {
            "success": False,
            "weights": np.array([], dtype=float),
            "symbols": symbols,
            "objective": objective_name,
            "status": str(problem.status),
            "message": str(exc),
            "warnings": list(constraints_spec.warnings),
            "estimation": estimates.metadata(),
        }

    expected_return = float(optimal_weights @ mean_returns)
    variance = float(optimal_weights @ covariance @ optimal_weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe_ratio = (
        (expected_return - float(risk_free_rate)) / volatility
        if volatility > 0
        else 0.0
    )
    turnover = (
        float(np.sum(np.abs(optimal_weights - current)))
        if current is not None
        else 0.0
    )
    transaction_cost = (
        float(np.sum(cost_rates * np.abs(optimal_weights - current)))
        if current is not None
        else 0.0
    )
    daily_losses = -estimates.returns.to_numpy(dtype=float) @ optimal_weights
    cutoff = float(np.quantile(daily_losses, confidence_for_metrics))
    tail = daily_losses[daily_losses >= cutoff]
    historical_cvar = float(np.mean(tail)) if tail.size else cutoff
    tracking_error = None
    if benchmark is not None:
        active = optimal_weights - benchmark
        tracking_error = float(np.sqrt(max(float(active @ covariance @ active), 0.0)))

    return {
        "success": True,
        "weights": optimal_weights,
        "symbols": symbols,
        "objective": objective_name,
        "status": str(problem.status),
        "solver": solver,
        "message": "ok",
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "historical_cvar_daily": historical_cvar,
        "tracking_error": tracking_error,
        "turnover": turnover,
        "current_weights": current.copy() if current is not None else None,
        "transaction_cost_drag": transaction_cost,
        "expected_return_model": resolved_return_model,
        "constraint_report": build_constraint_report(
            optimal_weights, constraints_spec
        ),
        "warnings": list(constraints_spec.warnings),
        "estimation": estimates.metadata(),
    }
