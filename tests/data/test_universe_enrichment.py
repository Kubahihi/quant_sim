from __future__ import annotations

import yfinance as yf

from src.data import universe_enrichment


def test_build_yfinance_session_bypasses_blackhole_proxy(monkeypatch):
    for key in universe_enrichment.PROXY_ENV_KEYS:
        monkeypatch.setenv(key, "http://127.0.0.1:9")

    monkeypatch.setattr(yf.config.network, "proxy", None, raising=False)

    session = universe_enrichment._build_yfinance_session()

    assert universe_enrichment._has_blackhole_proxy_env() is True
    assert session.proxies == universe_enrichment.YF_PROXY_BYPASS
    assert yf.config.network.proxy == universe_enrichment.YF_PROXY_BYPASS


def test_select_fast_info_tickers_prefers_priced_liquid_symbols():
    frame = universe_enrichment.pd.DataFrame(
        [
            {"Ticker": "AAA", "SourceCount": 3, "Price": None, "AvgVolume": None, "MarketCap": None, "Beta": None},
            {"Ticker": "BBB", "SourceCount": 2, "Price": 50.0, "AvgVolume": 2_000_000, "MarketCap": None, "Beta": None},
            {"Ticker": "CCC", "SourceCount": 2, "Price": 10.0, "AvgVolume": 500_000, "MarketCap": None, "Beta": None},
            {"Ticker": "DDD", "SourceCount": 3, "Price": 80.0, "AvgVolume": 1_500_000, "MarketCap": 1_000_000_000, "Beta": 1.1},
        ]
    )

    selected = universe_enrichment._select_fast_info_tickers(frame, max_symbols=3)

    assert selected == ["BBB", "CCC"]
