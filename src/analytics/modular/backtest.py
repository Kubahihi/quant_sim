from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def deterministic_signal_backtest(
    portfolio_returns: pd.Series,
    composite_signal: float,
    risk_signal: float,
    confidence: float,
) -> Dict[str, object]:
    """Simple deterministic, no-look-ahead backtest using lagged position sizing."""
    returns = pd.Series(portfolio_returns).dropna().astype(float)
    if returns.empty:
        return {
            "strategy_returns": pd.Series(dtype=float),
            "equity_curve": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "position": pd.Series(dtype=float),
            "metrics": {
                "total_return": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            },
            "lookahead_safe": True,
        }

    raw_position = np.clip(0.5 + 0.5 * composite_signal + 0.25 * risk_signal, 0.0, 1.0)
    scaled_position = float(np.clip(raw_position * max(0.1, min(1.0, confidence + 0.1)), 0.0, 1.0))

    position = pd.Series(index=returns.index, data=scaled_position, dtype=float)
    lagged_position = position.shift(1).fillna(0.0)

    strategy_returns = lagged_position * returns
    equity_curve = (1.0 + strategy_returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0

    vol = float(strategy_returns.std() * np.sqrt(252.0)) if len(strategy_returns) > 1 else 0.0
    sharpe = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252.0)) if strategy_returns.std() > 0 else 0.0

    metrics = {
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_position": float(lagged_position.mean()),
    }

    return {
        "strategy_returns": strategy_returns,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "position": lagged_position,
        "metrics": metrics,
        "lookahead_safe": True,
    }
