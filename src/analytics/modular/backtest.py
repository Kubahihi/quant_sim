from __future__ import annotations

from numbers import Integral, Real
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0


def _validated_simple_returns(portfolio_returns: pd.Series) -> pd.Series:
    """Return numeric simple returns, rejecting unsafe or ambiguous inputs."""
    try:
        returns = pd.Series(portfolio_returns).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("portfolio_returns must contain numeric simple returns.") from exc

    if not returns.index.is_unique:
        raise ValueError("portfolio_returns index must be unique.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("portfolio_returns index must be monotonically increasing.")

    values = returns.to_numpy(dtype=float, copy=False)
    if not bool(np.isfinite(values).all()):
        raise ValueError("portfolio_returns must contain only finite simple returns.")
    if bool((values <= -1.0).any()):
        raise ValueError("Simple returns must be greater than -100%.")
    return returns


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _equity_and_drawdown(strategy_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Build wealth and drawdown paths with initial wealth 1 included as a peak."""
    equity_curve = (1.0 + strategy_returns).cumprod()
    rolling_max = equity_curve.cummax().clip(lower=1.0)
    drawdown = equity_curve / rolling_max - 1.0
    return equity_curve, drawdown


def _backtest_metrics(strategy_returns: pd.Series, position: pd.Series, costs: pd.Series) -> Dict[str, float]:
    equity_curve, drawdown = _equity_and_drawdown(strategy_returns)
    observations = int(len(strategy_returns))
    total_return = float(equity_curve.iloc[-1] - 1.0) if observations else 0.0
    if observations and 1.0 + total_return > 0:
        annualized_return = float((1.0 + total_return) ** (TRADING_DAYS / observations) - 1.0)
    else:
        annualized_return = -1.0 if observations else 0.0
    daily_std = float(strategy_returns.std()) if observations > 1 else 0.0
    volatility = float(daily_std * np.sqrt(TRADING_DAYS))
    sharpe_ratio = (
        float(strategy_returns.mean() / daily_std * np.sqrt(TRADING_DAYS))
        if daily_std > 0
        else 0.0
    )
    max_drawdown = float(drawdown.min()) if observations else 0.0
    calmar_ratio = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    active = position > 1e-12
    win_rate = float((strategy_returns[active] > 0).mean()) if bool(active.any()) else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        # Backward-compatible alias used by older run-history records.
        "sharpe": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "mean_position": float(position.mean()) if observations else 0.0,
        "turnover": float(position.diff().abs().fillna(position.abs()).sum()),
        "transaction_cost_drag": float(costs.sum()) if observations else 0.0,
        "observations": float(observations),
    }


def deterministic_signal_backtest(
    portfolio_returns: pd.Series,
    composite_signal: float,
    risk_signal: float,
    confidence: float,
    *,
    signals_known_before_evaluation: bool = False,
) -> Dict[str, object]:
    """Replay a signal supplied *ex ante* with one-period-lagged exposure.

    This helper is deterministic and causal only when its three scalar inputs
    were fixed before the evaluated return period and the caller explicitly
    supplies ``signals_known_before_evaluation=True``. The production pipeline
    uses :func:`walk_forward_baseline_backtest` because full-sample signals
    cannot be presented as out-of-sample evidence.
    """
    returns = _validated_simple_returns(portfolio_returns)
    composite_signal = _finite_real(composite_signal, "composite_signal")
    risk_signal = _finite_real(risk_signal, "risk_signal")
    confidence = _finite_real(confidence, "confidence")
    if not isinstance(signals_known_before_evaluation, bool):
        raise ValueError("signals_known_before_evaluation must be a boolean.")
    validation_type = (
        "ex_ante_signal_replay"
        if signals_known_before_evaluation
        else "illustrative_fixed_signal_replay"
    )
    scope = (
        "Caller attests that scalar inputs were frozen before the evaluation window."
        if signals_known_before_evaluation
        else "Signal provenance is unverified; this replay is illustrative, not causal or OOS evidence."
    )
    if returns.empty:
        empty = pd.Series(dtype=float)
        return {
            "strategy_returns": empty,
            "equity_curve": empty,
            "drawdown": empty,
            "position": empty,
            "metrics": _backtest_metrics(empty, empty, empty),
            "lookahead_safe": signals_known_before_evaluation,
            "validation_type": validation_type,
            "scope": scope,
        }

    raw_position = np.clip(0.5 + 0.5 * composite_signal + 0.25 * risk_signal, 0.0, 1.0)
    scaled_position = float(np.clip(raw_position * max(0.1, min(1.0, confidence + 0.1)), 0.0, 1.0))

    position = pd.Series(index=returns.index, data=scaled_position, dtype=float)
    lagged_position = position.shift(1).fillna(0.0)

    strategy_returns = lagged_position * returns
    equity_curve, drawdown = _equity_and_drawdown(strategy_returns)

    costs = pd.Series(0.0, index=returns.index, dtype=float)
    metrics = _backtest_metrics(strategy_returns, lagged_position, costs)

    return {
        "strategy_returns": strategy_returns,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "position": lagged_position,
        "metrics": metrics,
        "lookahead_safe": signals_known_before_evaluation,
        "validation_type": validation_type,
        "scope": scope,
    }


def walk_forward_baseline_backtest(
    portfolio_returns: pd.Series,
    short_window: int = 20,
    long_window: int = 60,
    transaction_cost_bps: float = 10.0,
) -> Dict[str, object]:
    """Run a causal walk-forward trend/risk baseline.

    Every position at date *t* uses returns available strictly before *t*.
    The result is useful as an out-of-sample process check and benchmark, but
    it is deliberately not described as validation of the full model bundle.
    """
    returns = _validated_simple_returns(portfolio_returns)
    short_window = _positive_integer(short_window, "short_window")
    long_window = _positive_integer(long_window, "long_window")
    transaction_cost_bps = _finite_real(transaction_cost_bps, "transaction_cost_bps")
    if short_window < 2 or long_window <= short_window:
        raise ValueError("Require 2 <= short_window < long_window.")
    if not 0.0 <= transaction_cost_bps < 10_000.0:
        raise ValueError("transaction_cost_bps must be in [0, 10000).")
    if returns.empty:
        empty = pd.Series(dtype=float)
        return {
            "strategy_returns": empty,
            "equity_curve": empty,
            "drawdown": empty,
            "position": empty,
            "metrics": _backtest_metrics(empty, empty, empty),
            "lookahead_safe": True,
            "validation_type": "walk_forward_causal_baseline",
            "scope": "Causal baseline only; it does not validate the full model ensemble.",
        }

    lagged = returns.shift(1)
    short_mean = lagged.rolling(short_window, min_periods=short_window).mean()
    short_vol = lagged.rolling(short_window, min_periods=short_window).std()
    long_vol = lagged.rolling(long_window, min_periods=long_window).std()

    trend_t = short_mean / (short_vol / np.sqrt(float(short_window))).replace(0.0, np.nan)
    trend_score = np.tanh(trend_t.fillna(0.0) / 2.0)
    risk_score = ((long_vol - short_vol) / long_vol.replace(0.0, np.nan)).clip(-1.0, 1.0).fillna(0.0)
    position = (0.50 + 0.35 * trend_score + 0.15 * risk_score).clip(0.0, 1.0)
    position = position.where(long_vol.notna(), 0.0).astype(float)

    turnover = position.diff().abs().fillna(position.abs())
    cost_rates = turnover * (transaction_cost_bps / 10_000.0)
    gross_strategy_returns = position * returns
    strategy_returns = (1.0 + gross_strategy_returns) * (1.0 - cost_rates) - 1.0
    costs = gross_strategy_returns - strategy_returns
    equity_curve, drawdown = _equity_and_drawdown(strategy_returns)

    realized_direction = np.sign(returns)
    forecast_direction = np.sign(position - 0.5)
    directional_mask = (forecast_direction != 0) & long_vol.notna()
    directional_hit_rate = (
        float((forecast_direction[directional_mask] == realized_direction[directional_mask]).mean())
        if bool(directional_mask.any())
        else 0.0
    )
    metrics = _backtest_metrics(strategy_returns, position, costs)
    metrics["directional_hit_rate"] = directional_hit_rate

    return {
        "strategy_returns": strategy_returns,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "position": position,
        "metrics": metrics,
        "lookahead_safe": True,
        "validation_type": "walk_forward_causal_baseline",
        "scope": "Causal baseline only; it does not validate the full model ensemble.",
        "parameters": {
            "short_window": short_window,
            "long_window": long_window,
            "transaction_cost_bps": transaction_cost_bps,
        },
    }
