"""Shared transparent technical statistics (no trading recommendations)."""
import numpy as np
import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14) -> float:
    """Simple rolling RSI, retaining the screener convention (not Wilder smoothing)."""
    delta = close.diff().tail(window)
    if len(delta) < window or delta.isna().any():
        return float('nan')
    gain, loss = delta.clip(lower=0).mean(), -delta.clip(upper=0).mean()
    if not np.isfinite(gain + loss):
        return float('nan')
    return float(50 if gain + loss == 0 else 100 * gain / (gain + loss))
