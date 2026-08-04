from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .constraints import build_weight_bounds
from .estimators import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_RETURN_SHRINKAGE,
    TRADING_DAYS,
    clean_returns,
    estimate_black_litterman_inputs,
)
from .engine import optimize_portfolio
from .maximum_sharpe import optimize_maximum_sharpe
from .minimum_variance import optimize_minimum_variance


SUPPORTED_OPTIMIZERS = {
    "maximum_sharpe",
    "minimum_variance",
    "maximum_utility",
    "target_volatility",
    "minimum_cvar",
    "minimum_tracking_error",
}


def _normalize_initial_weights(
    initial_weights: Optional[np.ndarray | list[float]],
    n_assets: int,
) -> np.ndarray:
    if initial_weights is None:
        return np.full(n_assets, 1.0 / n_assets, dtype=float)
    values = np.asarray(initial_weights, dtype=float)
    if values.ndim != 1 or values.size != n_assets:
        raise ValueError("initial_weights length must match the return columns.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("initial_weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("initial_weights must have a positive sum.")
    return values / total


def _drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> tuple[float, np.ndarray]:
    gross_assets = 1.0 + asset_returns
    gross_portfolio = float(weights @ gross_assets)
    portfolio_return = gross_portfolio - 1.0
    if gross_portfolio <= 0:
        return portfolio_return, weights.copy()
    drifted = weights * gross_assets / gross_portfolio
    return portfolio_return, drifted


def _performance_metrics(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    turnover: pd.Series,
    transaction_costs: pd.Series,
) -> dict[str, float]:
    clean = pd.Series(returns, dtype=float).dropna()
    if clean.empty:
        return {
            "observations": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_turnover": float(turnover.sum()),
            "transaction_cost_drag": float(transaction_costs.sum()),
        }
    wealth = (1.0 + clean).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (TRADING_DAYS / len(clean)) - 1.0)
    volatility = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(clean) > 1 else 0.0
    sharpe_ratio = (
        (float(clean.mean()) * TRADING_DAYS - float(risk_free_rate)) / volatility
        if volatility > 0
        else 0.0
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "observations": float(len(clean)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(drawdown.min()),
        "total_turnover": float(turnover.sum()),
        "transaction_cost_drag": float(transaction_costs.sum()),
    }


def run_optimization_walk_forward(
    returns: pd.DataFrame,
    *,
    optimizer: str = "maximum_sharpe",
    train_periods: int = 252,
    rebalance_periods: int = 21,
    initial_weights: Optional[np.ndarray | list[float]] = None,
    max_weight: Optional[float] = None,
    risk_free_rate: float = 0.03,
    transaction_cost_bps: float = 10.0,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    strategy: Optional[dict[str, Any]] = None,
    asset_metadata: Optional[dict[str, dict[str, Any]]] = None,
    turnover_limit: Optional[float] = None,
    risk_aversion: float = 3.0,
    target_volatility: Optional[float] = None,
    cvar_confidence: float = 0.95,
    benchmark_weights: Optional[np.ndarray | list[float] | dict[str, float]] = None,
    expected_return_model: str = "shrunk_historical",
    black_litterman_views: Optional[dict[str, float]] = None,
    black_litterman_confidence: float = 0.60,
) -> dict[str, Any]:
    """Causally re-estimate, optimize, and apply weights in rolling OOS windows."""
    method = str(optimizer).strip().lower()
    if method not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(SUPPORTED_OPTIMIZERS)}."
        )
    if method == "maximum_sharpe" and (
        strategy is not None
        or turnover_limit is not None
        or str(expected_return_model).strip().lower() == "black_litterman"
    ):
        raise ValueError(
            "maximum_sharpe walk-forward does not support mandate, turnover, "
            "or Black-Litterman inputs; use maximum_utility instead."
        )
    if not isinstance(train_periods, int) or train_periods < 20:
        raise ValueError("train_periods must be an integer of at least 20.")
    if not isinstance(rebalance_periods, int) or rebalance_periods < 1:
        raise ValueError("rebalance_periods must be a positive integer.")
    transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
    if not np.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    clean = clean_returns(returns)
    if not clean.index.is_monotonic_increasing or clean.index.has_duplicates:
        raise ValueError("returns index must be unique and increasing.")
    if len(clean) <= train_periods:
        raise ValueError("returns must extend beyond the training window.")

    n_assets = clean.shape[1]
    build_weight_bounds(n_assets, allow_short=False, max_weight=max_weight)
    current_weights = _normalize_initial_weights(initial_weights, n_assets)
    equal_target = np.full(n_assets, 1.0 / n_assets, dtype=float)
    equal_weights = equal_target.copy()

    optimized_gross: dict[Any, float] = {}
    optimized_net: dict[Any, float] = {}
    equal_weight_net: dict[Any, float] = {}
    turnover_by_date: dict[Any, float] = {}
    costs_by_date: dict[Any, float] = {}
    equal_turnover_by_date: dict[Any, float] = {}
    equal_costs_by_date: dict[Any, float] = {}
    weights_history: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []

    for test_start in range(train_periods, len(clean), rebalance_periods):
        test_end = min(test_start + rebalance_periods, len(clean))
        training = clean.iloc[test_start - train_periods:test_start]
        use_legacy_optimizer = method == "maximum_sharpe" or (
            method == "minimum_variance"
            and strategy is None
            and turnover_limit is None
        )
        if use_legacy_optimizer:
            optimizer_fn = (
                optimize_maximum_sharpe
                if method == "maximum_sharpe"
                else optimize_minimum_variance
            )
            result = optimizer_fn(
                training,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                covariance_shrinkage=covariance_shrinkage,
                return_shrinkage=return_shrinkage,
            )
        else:
            portfolio_estimates = None
            if str(expected_return_model) == "black_litterman":
                views = dict(black_litterman_views or {})
                portfolio_estimates = estimate_black_litterman_inputs(
                    training,
                    market_weights=current_weights,
                    views=views,
                    view_confidences={
                        symbol: float(black_litterman_confidence)
                        for symbol in views
                    },
                    risk_aversion=risk_aversion,
                    covariance_shrinkage=covariance_shrinkage,
                    return_shrinkage=return_shrinkage,
                )
            result = optimize_portfolio(
                training,
                objective=method,
                strategy=strategy,
                asset_metadata=asset_metadata,
                current_weights=current_weights,
                max_weight=max_weight,
                turnover_limit=turnover_limit,
                transaction_cost_bps=transaction_cost_bps,
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                target_volatility=(
                    target_volatility if method == "target_volatility" else None
                ),
                cvar_confidence=cvar_confidence,
                benchmark_weights=(
                    benchmark_weights
                    if method == "minimum_tracking_error"
                    else None
                ),
                portfolio_estimates=portfolio_estimates,
                covariance_shrinkage=covariance_shrinkage,
                return_shrinkage=return_shrinkage,
            )
        optimizer_success = bool(result.get("success", False))
        target_weights = (
            np.asarray(result["weights"], dtype=float)
            if optimizer_success
            else current_weights.copy()
        )
        turnover = float(np.sum(np.abs(target_weights - current_weights)))
        transaction_cost = transaction_cost_rate * turnover

        equal_turnover = float(np.sum(np.abs(equal_target - equal_weights)))
        equal_transaction_cost = transaction_cost_rate * equal_turnover
        decision_date = clean.index[test_start]
        turnover_by_date[decision_date] = turnover
        costs_by_date[decision_date] = transaction_cost
        equal_turnover_by_date[decision_date] = equal_turnover
        equal_costs_by_date[decision_date] = equal_transaction_cost
        weights_history.append({
            "date": decision_date,
            **{
                str(symbol): float(weight)
                for symbol, weight in zip(clean.columns, target_weights, strict=False)
            },
        })
        windows.append({
            "train_start": clean.index[test_start - train_periods],
            "train_end": clean.index[test_start - 1],
            "test_start": decision_date,
            "test_end": clean.index[test_end - 1],
            "optimizer_success": optimizer_success,
            "message": str(result.get("message", "")),
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        })

        current_weights = target_weights
        equal_weights = equal_target.copy()
        for position in range(test_start, test_end):
            date = clean.index[position]
            daily_asset_returns = clean.iloc[position].to_numpy(dtype=float)
            gross_return, current_weights = _drift_weights(
                current_weights, daily_asset_returns
            )
            equal_return, equal_weights = _drift_weights(
                equal_weights, daily_asset_returns
            )
            optimized_gross[date] = gross_return
            optimized_net[date] = gross_return - (
                transaction_cost if position == test_start else 0.0
            )
            equal_weight_net[date] = equal_return - (
                equal_transaction_cost if position == test_start else 0.0
            )

    gross_series = pd.Series(optimized_gross, dtype=float, name="gross_return")
    net_series = pd.Series(optimized_net, dtype=float, name="net_return")
    equal_series = pd.Series(equal_weight_net, dtype=float, name="equal_weight_return")
    turnover_series = pd.Series(turnover_by_date, dtype=float, name="turnover")
    cost_series = pd.Series(costs_by_date, dtype=float, name="transaction_cost")
    equal_turnover_series = pd.Series(
        equal_turnover_by_date, dtype=float, name="turnover"
    )
    equal_cost_series = pd.Series(
        equal_costs_by_date, dtype=float, name="transaction_cost"
    )
    history = pd.DataFrame(weights_history).set_index("date")

    return {
        "success": bool(windows) and all(
            bool(window["optimizer_success"]) for window in windows
        ),
        "validation_type": "rolling_reoptimization_out_of_sample",
        "causal": True,
        "optimizer": method,
        "train_periods": train_periods,
        "rebalance_periods": rebalance_periods,
        "symbols": [str(column) for column in clean.columns],
        "weights_history": history,
        "gross_returns": gross_series,
        "net_returns": net_series,
        "turnover": turnover_series,
        "transaction_costs": cost_series,
        "windows": windows,
        "metrics": _performance_metrics(
            net_series,
            risk_free_rate=risk_free_rate,
            turnover=turnover_series,
            transaction_costs=cost_series,
        ),
        "equal_weight_returns": equal_series,
        "equal_weight_metrics": _performance_metrics(
            equal_series,
            risk_free_rate=risk_free_rate,
            turnover=equal_turnover_series,
            transaction_costs=equal_cost_series,
        ),
    }
