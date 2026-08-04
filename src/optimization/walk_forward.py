from __future__ import annotations

from typing import Any, Mapping, Optional

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
from .execution import estimate_trade_costs
from .maximum_sharpe import optimize_maximum_sharpe
from .minimum_variance import optimize_minimum_variance
from .universe import align_point_in_time_membership


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
    universe_membership: Optional[pd.DataFrame] = None,
    membership_lag_periods: int = 1,
    average_daily_dollar_volume: Optional[
        pd.DataFrame | Mapping[str, float]
    ] = None,
    portfolio_value: float = 1_000_000.0,
    half_spread_bps: float = 0.0,
    market_impact_bps: float = 0.0,
    max_adv_participation: Optional[float] = None,
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
        or average_daily_dollar_volume is not None
        or float(half_spread_bps) > 0
        or float(market_impact_bps) > 0
        or max_adv_participation is not None
    ):
        raise ValueError(
            "maximum_sharpe walk-forward does not support mandate, turnover, "
            "Black-Litterman, or liquidity inputs; use maximum_utility instead."
        )
    if not isinstance(train_periods, int) or train_periods < 20:
        raise ValueError("train_periods must be an integer of at least 20.")
    if not isinstance(rebalance_periods, int) or rebalance_periods < 1:
        raise ValueError("rebalance_periods must be a positive integer.")
    transaction_cost_rate = float(transaction_cost_bps) / 10_000.0
    if not np.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    if not isinstance(membership_lag_periods, int) or membership_lag_periods < 1:
        raise ValueError("membership_lag_periods must be a positive integer.")
    capital = float(portfolio_value)
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("portfolio_value must be positive.")
    point_in_time = universe_membership is not None
    if point_in_time:
        clean = pd.DataFrame(returns).copy()
        if clean.shape[1] < 1 or clean.columns.has_duplicates:
            raise ValueError("returns must have unique, non-empty columns.")
        clean = clean.apply(pd.to_numeric, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if len(clean) < 2:
            raise ValueError("returns must contain at least two observations.")
    else:
        clean = clean_returns(returns)
    if not clean.index.is_monotonic_increasing or clean.index.has_duplicates:
        raise ValueError("returns index must be unique and increasing.")
    if len(clean) <= train_periods:
        raise ValueError("returns must extend beyond the training window.")

    n_assets = clean.shape[1]
    if not point_in_time:
        build_weight_bounds(n_assets, allow_short=False, max_weight=max_weight)
    current_weights = _normalize_initial_weights(initial_weights, n_assets)
    equal_target = np.full(n_assets, 1.0 / n_assets, dtype=float)
    equal_weights = equal_target.copy()

    aligned_membership: Optional[pd.DataFrame] = None
    if universe_membership is not None:
        aligned_membership = align_point_in_time_membership(
            universe_membership,
            return_index=clean.index,
            symbols=[str(column) for column in clean.columns],
        )
        if initial_weights is None:
            initially_active = aligned_membership.iloc[0].to_numpy(dtype=bool)
            if not np.any(initially_active):
                raise ValueError("point-in-time universe has no active assets at the start.")
            current_weights = initially_active.astype(float) / float(initially_active.sum())
            equal_target = current_weights.copy()
            equal_weights = current_weights.copy()

    aligned_adv: Optional[pd.DataFrame] = None
    constant_adv: Optional[dict[str, float]] = None
    if isinstance(average_daily_dollar_volume, pd.DataFrame):
        missing_adv_columns = [
            str(column)
            for column in clean.columns
            if str(column) not in average_daily_dollar_volume.columns
        ]
        if missing_adv_columns:
            raise ValueError(
                "average_daily_dollar_volume is missing columns for: "
                + ", ".join(missing_adv_columns)
                + "."
            )
        adv_source = average_daily_dollar_volume[
            [str(column) for column in clean.columns]
        ].copy()
        adv_source.index = pd.to_datetime(adv_source.index)
        adv_source = adv_source.sort_index().apply(pd.to_numeric, errors="coerce")
        combined_index = adv_source.index.union(pd.DatetimeIndex(clean.index)).sort_values()
        aligned_adv = adv_source.reindex(combined_index).ffill().reindex(clean.index)
    elif average_daily_dollar_volume is not None:
        constant_adv = {
            str(symbol): float(average_daily_dollar_volume[str(symbol)])
            for symbol in clean.columns
        }

    optimized_gross: dict[Any, float] = {}
    optimized_net: dict[Any, float] = {}
    equal_weight_net: dict[Any, float] = {}
    turnover_by_date: dict[Any, float] = {}
    costs_by_date: dict[Any, float] = {}
    equal_turnover_by_date: dict[Any, float] = {}
    equal_costs_by_date: dict[Any, float] = {}
    weights_history: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []

    all_symbols = [str(column) for column in clean.columns]
    has_liquidity_model = bool(
        average_daily_dollar_volume is not None
        or float(half_spread_bps) > 0
        or float(market_impact_bps) > 0
        or max_adv_participation is not None
    )

    for test_start in range(train_periods, len(clean), rebalance_periods):
        test_end = min(test_start + rebalance_periods, len(clean))
        membership_position = max(0, test_start - membership_lag_periods)
        if aligned_membership is None:
            active_indices = list(range(n_assets))
        else:
            membership_row = aligned_membership.iloc[membership_position].to_numpy(dtype=bool)
            active_indices = np.flatnonzero(membership_row).tolist()
        if not active_indices:
            raise ValueError(
                f"point-in-time universe has no active assets as of {clean.index[membership_position]}."
            )
        active_symbols = [all_symbols[index] for index in active_indices]
        build_weight_bounds(
            len(active_indices), allow_short=False, max_weight=max_weight
        )
        training = clean.iloc[
            test_start - train_periods:test_start,
            active_indices,
        ]
        training.columns = active_symbols
        training = clean_returns(training)
        if len(training) < 20:
            raise ValueError(
                "point-in-time training window has fewer than 20 complete observations "
                f"for: {', '.join(active_symbols)}."
            )

        active_current_raw = current_weights[active_indices]
        active_current_total = float(active_current_raw.sum())
        active_current = (
            active_current_raw / active_current_total
            if active_current_total > 1e-12
            else np.full(len(active_indices), 1.0 / len(active_indices), dtype=float)
        )
        active_metadata = (
            {
                symbol: dict((asset_metadata or {}).get(symbol, {}))
                for symbol in active_symbols
            }
            if asset_metadata is not None
            else None
        )
        full_adv_snapshot: Optional[dict[str, float]] = None
        if aligned_adv is not None:
            row = aligned_adv.iloc[membership_position]
            full_adv_snapshot = {
                symbol: float(row[symbol])
                for symbol in all_symbols
                if pd.notna(row[symbol])
            }
        elif constant_adv is not None:
            full_adv_snapshot = dict(constant_adv)
        active_adv = (
            {symbol: float(full_adv_snapshot[symbol]) for symbol in active_symbols}
            if full_adv_snapshot is not None
            else None
        )

        use_legacy_optimizer = method == "maximum_sharpe" or (
            method == "minimum_variance"
            and strategy is None
            and turnover_limit is None
            and not has_liquidity_model
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
                views = {
                    symbol: value
                    for symbol, value in dict(black_litterman_views or {}).items()
                    if symbol in active_symbols
                }
                portfolio_estimates = estimate_black_litterman_inputs(
                    training,
                    market_weights=active_current,
                    views=views,
                    view_confidences={
                        symbol: float(black_litterman_confidence)
                        for symbol in views
                    },
                    risk_aversion=risk_aversion,
                    covariance_shrinkage=covariance_shrinkage,
                    return_shrinkage=return_shrinkage,
                )
            active_benchmark = None
            if method == "minimum_tracking_error":
                if benchmark_weights is None:
                    raise ValueError("minimum_tracking_error requires benchmark_weights.")
                if isinstance(benchmark_weights, Mapping):
                    benchmark_array = np.asarray(
                        [float(benchmark_weights[symbol]) for symbol in all_symbols],
                        dtype=float,
                    )
                else:
                    benchmark_array = np.asarray(benchmark_weights, dtype=float)
                if benchmark_array.ndim != 1 or benchmark_array.size != n_assets:
                    raise ValueError("benchmark_weights length must match return columns.")
                active_benchmark = benchmark_array[active_indices]
                active_benchmark_total = float(active_benchmark.sum())
                if active_benchmark_total <= 0:
                    raise ValueError("active point-in-time benchmark has zero weight.")
                active_benchmark = active_benchmark / active_benchmark_total
            result = optimize_portfolio(
                training,
                objective=method,
                strategy=strategy,
                asset_metadata=active_metadata,
                current_weights=active_current,
                max_weight=max_weight,
                turnover_limit=turnover_limit,
                transaction_cost_bps=transaction_cost_bps,
                half_spread_bps=half_spread_bps,
                market_impact_bps=market_impact_bps,
                average_daily_dollar_volume=active_adv,
                portfolio_value=capital,
                max_adv_participation=max_adv_participation,
                risk_free_rate=risk_free_rate,
                risk_aversion=risk_aversion,
                target_volatility=(
                    target_volatility if method == "target_volatility" else None
                ),
                cvar_confidence=cvar_confidence,
                benchmark_weights=active_benchmark,
                portfolio_estimates=portfolio_estimates,
                covariance_shrinkage=covariance_shrinkage,
                return_shrinkage=return_shrinkage,
            )
        optimizer_success = bool(result.get("success", False))
        target_weights = current_weights.copy()
        if optimizer_success:
            target_weights = np.zeros(n_assets, dtype=float)
            target_weights[active_indices] = np.asarray(result["weights"], dtype=float)
        turnover = float(np.sum(np.abs(target_weights - current_weights)))
        configured_limits = [
            float(value)
            for value in (
                turnover_limit,
                (strategy or {}).get("max_turnover"),
            )
            if value is not None
        ]
        effective_turnover_limit = min(configured_limits) if configured_limits else None
        if (
            optimizer_success
            and effective_turnover_limit is not None
            and turnover > effective_turnover_limit + 2e-5
        ):
            optimizer_success = False
            result["message"] = (
                "point-in-time membership changes make the full-universe turnover "
                "limit infeasible for this window."
            )
            target_weights = current_weights.copy()
            turnover = 0.0

        transaction_cost_model = estimate_trade_costs(
            target_weights - current_weights,
            all_symbols,
            portfolio_value=capital,
            transaction_cost_bps=transaction_cost_bps,
            half_spread_bps=half_spread_bps,
            market_impact_bps=market_impact_bps,
            average_daily_dollar_volume=full_adv_snapshot,
        )
        transaction_cost = float(transaction_cost_model["total_drag"])

        equal_target = np.zeros(n_assets, dtype=float)
        equal_target[active_indices] = 1.0 / len(active_indices)
        equal_turnover = float(np.sum(np.abs(equal_target - equal_weights)))
        equal_cost_model = estimate_trade_costs(
            equal_target - equal_weights,
            all_symbols,
            portfolio_value=capital,
            transaction_cost_bps=transaction_cost_bps,
            half_spread_bps=half_spread_bps,
            market_impact_bps=market_impact_bps,
            average_daily_dollar_volume=full_adv_snapshot,
        )
        equal_transaction_cost = float(equal_cost_model["total_drag"])
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
            "transaction_cost_breakdown": {
                key: float(transaction_cost_model[key])
                for key in (
                    "commission_drag",
                    "spread_drag",
                    "market_impact_drag",
                    "total_drag",
                )
            },
            "active_symbols": active_symbols,
            "membership_as_of": clean.index[membership_position],
        })

        current_weights = target_weights
        equal_weights = equal_target.copy()
        for position in range(test_start, test_end):
            date = clean.index[position]
            daily_asset_returns = clean.iloc[position].to_numpy(dtype=float)
            held = (current_weights > 1e-10) | (equal_weights > 1e-10)
            missing_held = held & ~np.isfinite(daily_asset_returns)
            if np.any(missing_held):
                missing_symbols = [
                    all_symbols[index]
                    for index in np.flatnonzero(missing_held)
                ]
                raise ValueError(
                    "missing out-of-sample return for held point-in-time asset(s) on "
                    f"{date}: {', '.join(missing_symbols)}. Supply delisting returns or "
                    "shorten the holding window."
                )
            daily_asset_returns = np.nan_to_num(
                daily_asset_returns, nan=0.0, posinf=0.0, neginf=0.0
            )
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
        "validation_type": (
            "point_in_time_rolling_reoptimization_out_of_sample"
            if point_in_time
            else "rolling_reoptimization_out_of_sample"
        ),
        "causal": True,
        "point_in_time_universe": point_in_time,
        "survivorship_bias_controlled": point_in_time,
        "membership_lag_periods": membership_lag_periods if point_in_time else None,
        "warnings": (
            []
            if point_in_time
            else [
                "Universe membership was not supplied point-in-time; results may contain survivorship bias."
            ]
        ),
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
