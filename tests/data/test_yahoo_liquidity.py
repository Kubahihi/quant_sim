from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.fetchers.yahoo_fetcher import FetchResult, YahooFetcher


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

    assert calls == ["A", "B"]
    assert result["prices"].columns.tolist() == ["A", "B"]
    assert result["adv_history"].shape == (10, 2)
    assert result["average_daily_dollar_volume"]["B"] == pytest.approx(
        20.0 * np.mean(np.arange(6, 11) * 1_000.0)
    )
