from __future__ import annotations

from collections import Counter
import threading

import numpy as np
import pandas as pd
import pytest

from src.data.fetchers import yahoo_fetcher
from src.data.fetchers.yahoo_fetcher import FetchResult, YahooFetcher


def test_fetch_result_preserves_dataframe_read_compatibility() -> None:
    frame = pd.DataFrame({"close": [10.0, 11.0]})
    result = FetchResult(data=frame, success=True)

    assert result.empty is False
    pd.testing.assert_series_equal(result["close"], frame["close"])
    assert FetchResult(data=pd.DataFrame(), success=False).empty is True


def test_close_and_liquidity_fetch_reuses_each_ohlcv_download_once(monkeypatch) -> None:
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    calls: list[str] = []

    def fake_fetch(symbol, start_date, end_date, interval="1d"):
        calls.append(symbol)
        return FetchResult(
            data=pd.DataFrame({
                "open": np.full(10, 10.0),
                "high": np.full(10, 10.5),
                "low": np.full(10, 9.5),
                "close": np.full(10, 10.0 if symbol == "A" else 20.0),
                "volume": np.arange(1, 11, dtype=float) * 1_000.0,
            }, index=dates),
            success=True,
        )

    fetcher = YahooFetcher()
    monkeypatch.setattr(fetcher, "fetch_prices", fake_fetch)

    result = fetcher.fetch_close_prices_with_liquidity(
        ["A", "B"], dates[0], dates[-1], adv_window=5
    )

    assert Counter(calls) == Counter(["A", "B"])
    assert result["prices"].columns.tolist() == ["A", "B"]
    assert result["adv_history"].shape == (10, 2)
    assert result["average_daily_dollar_volume"]["B"] == pytest.approx(
        20.0 * np.mean(np.arange(6, 11) * 1_000.0)
    )


def test_close_and_liquidity_fetch_is_bounded_concurrent_and_ordered(monkeypatch) -> None:
    symbols = ["D", "B", "A", "C", "H", "F", "E", "G"]
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    barrier = threading.Barrier(yahoo_fetcher.MAX_MULTI_FETCH_WORKERS)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_fetch(symbol, start_date, end_date, interval="1d"):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=5)
            value = float(ord(symbol) - ord("A") + 1)
            return FetchResult(
                data=pd.DataFrame({
                    "open": np.full(10, value),
                    "high": np.full(10, value + 0.5),
                    "low": np.full(10, value - 0.5),
                    "close": np.full(10, value),
                    "volume": np.arange(1, 11, dtype=float) * 1_000.0,
                }, index=dates),
                success=True,
            )
        finally:
            with state_lock:
                active -= 1

    fetcher = YahooFetcher()
    monkeypatch.setattr(fetcher, "fetch_prices", fake_fetch)

    result = fetcher.fetch_close_prices_with_liquidity(
        symbols, dates[0], dates[-1], adv_window=5
    )

    assert result["prices"].columns.tolist() == symbols
    assert result["adv_history"].columns.tolist() == symbols
    assert list(result["average_daily_dollar_volume"]) == symbols
    assert max_active == yahoo_fetcher.MAX_MULTI_FETCH_WORKERS


def test_close_and_liquidity_fetch_preserves_successes_after_partial_failure(
    monkeypatch,
) -> None:
    symbols = ["A", "BROKEN", "C"]
    dates = pd.date_range("2026-01-02", periods=10, freq="B")

    def fake_fetch(symbol, start_date, end_date, interval="1d"):
        if symbol == "BROKEN":
            return FetchResult(
                data=pd.DataFrame(),
                success=False,
                error="simulated provider failure",
            )
        value = 10.0 if symbol == "A" else 30.0
        return FetchResult(
            data=pd.DataFrame({
                "open": np.full(10, value),
                "high": np.full(10, value),
                "low": np.full(10, value),
                "close": np.full(10, value),
                "volume": np.arange(1, 11, dtype=float) * 1_000.0,
            }, index=dates),
            success=True,
        )

    fetcher = YahooFetcher()
    monkeypatch.setattr(fetcher, "fetch_prices", fake_fetch)

    result = fetcher.fetch_close_prices_with_liquidity(
        symbols, dates[0], dates[-1], adv_window=5
    )

    assert result["prices"].columns.tolist() == ["A", "C"]
    assert result["adv_history"].columns.tolist() == ["A", "C"]
    assert list(result["average_daily_dollar_volume"]) == ["A", "C"]
