"""Daily exchange-session alignment and bounded missing-price handling."""
import numpy as np
import pandas as pd


def normalize_daily_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        # Retain the exchange's local session date, not its UTC midnight.
        result.index = result.index.tz_localize(None).normalize()
        if result.index.has_duplicates:
            raise ValueError("Daily price data contains duplicate exchange-session dates.")
    return result.sort_index()


def align_daily_prices(prices: pd.DataFrame, *, max_stale_sessions: int = 3, max_stale_days: int = 7) -> pd.DataFrame:
    """Permit short holiday gaps, but never silently invent long flat histories."""
    frame = pd.DataFrame(prices).sort_index().copy()
    if frame.index.has_duplicates:
        raise ValueError("Daily price dates must be unique.")
    if max_stale_sessions < 0 or max_stale_days < 0:
        raise ValueError("Staleness limits must be non-negative.")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    invalid = frame.notna() & (~np.isfinite(frame) | (frame <= 0))
    if invalid.any().any():
        raise ValueError("Observed prices must be finite and positive.")
    filled = frame.ffill(limit=max_stale_sessions) if max_stale_sessions else frame.copy()
    for symbol in frame:
        observed = frame[symbol].notna()
        if not observed.any():
            raise ValueError(f"No observed prices for {symbol}.")
        started = observed.cummax()
        stale = started & filled[symbol].isna()
        if isinstance(frame.index, pd.DatetimeIndex):
            dates = pd.Series(frame.index, index=frame.index)
            age = dates - dates.where(observed).ffill()
            stale |= started & (age > pd.Timedelta(days=max_stale_days))
        if stale.any():
            first = frame.index[stale][0]
            raise ValueError(f"Stale prices for {symbol} at {first}; refresh or shorten the requested window.")
    filled.attrs["forward_filled_observations"] = int((frame.isna() & filled.notna()).sum().sum())
    filled.attrs["price_alignment_policy"] = "local_exchange_session_dates_bounded_forward_fill"
    return filled
