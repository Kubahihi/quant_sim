from __future__ import annotations

from contextlib import nullcontext

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


def test_select_detail_tickers_prioritizes_uncached_symbols():
    frame = universe_enrichment.pd.DataFrame(
        [
            {
                "Ticker": "CACHED",
                "Price": 100.0,
                "SourceCount": 2,
                "MarketCap": 10_000_000_000,
                "AvgVolume": 3_000_000,
                "Sector": None,
                "Industry": None,
                "PE": None,
                "ForwardPE": None,
                "PEG": None,
                "ROE": None,
                "ROA": None,
                "RevenueGrowth": None,
                "EarningsGrowth": None,
                "DividendYield": None,
            },
            {
                "Ticker": "NEW1",
                "Price": 90.0,
                "SourceCount": 1,
                "MarketCap": 5_000_000_000,
                "AvgVolume": 2_000_000,
                "Sector": None,
                "Industry": None,
                "PE": None,
                "ForwardPE": None,
                "PEG": None,
                "ROE": None,
                "ROA": None,
                "RevenueGrowth": None,
                "EarningsGrowth": None,
                "DividendYield": None,
            },
            {
                "Ticker": "NEW2",
                "Price": 80.0,
                "SourceCount": 1,
                "MarketCap": 4_000_000_000,
                "AvgVolume": 1_500_000,
                "Sector": None,
                "Industry": None,
                "PE": None,
                "ForwardPE": None,
                "PEG": None,
                "ROE": None,
                "ROA": None,
                "RevenueGrowth": None,
                "EarningsGrowth": None,
                "DividendYield": None,
            },
        ]
    )

    selected = universe_enrichment._select_detail_tickers(
        frame,
        detail_columns=[
            "Sector",
            "Industry",
            "PE",
            "ForwardPE",
            "PEG",
            "ROE",
            "ROA",
            "RevenueGrowth",
            "EarningsGrowth",
            "DividendYield",
        ],
        max_symbols=2,
        cache={"CACHED": {"Sector": "Technology"}},
    )

    assert selected == ["NEW1", "NEW2"]


def test_build_cached_updates_supports_legacy_lowercase_cache():
    updates = universe_enrichment._build_cached_updates_for_tickers(
        tickers=["AAPL", "MSFT"],
        cache={
            "AAPL": {
                "company_name": "Apple Inc.",
                "sector": "Technology",
                "marketCap": 1_000_000,
                "returnOnEquity": 0.25,
            }
        },
    )

    assert "AAPL" in updates
    assert updates["AAPL"]["Company"] == "Apple Inc."
    assert updates["AAPL"]["Sector"] == "Technology"
    assert updates["AAPL"]["ROE"] == 0.25
    assert "MSFT" not in updates


def test_fetch_detail_info_uses_legacy_lowercase_cache_without_network(monkeypatch):
    cache_payload = {
        "AAPL": {
            "company_name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 1_000_000_000,
            "beta": 1.2,
        }
    }
    monkeypatch.setattr(universe_enrichment, "_load_fundamental_cache", lambda: cache_payload)
    monkeypatch.setattr(universe_enrichment, "_save_fundamental_cache", lambda data: None)

    ticker_called = {"value": 0}

    class _NetworkTicker:
        def __init__(self, symbol: str, session=None):
            ticker_called["value"] += 1

        def get_info(self):
            raise AssertionError("Network should not be called when cache contains usable data")

    monkeypatch.setattr(universe_enrichment.yf, "Ticker", _NetworkTicker)

    result = universe_enrichment._fetch_detail_info(["AAPL"])

    assert ticker_called["value"] == 0
    assert "AAPL" in result
    assert result["AAPL"]["Company"] == "Apple Inc."
    assert result["AAPL"]["MarketCap"] == 1_000_000_000
    assert result["AAPL"]["Sector"] == "Technology"


def test_fetch_detail_info_counts_failures_once_per_symbol(monkeypatch):
    call_count = {"value": 0}

    class _AlwaysFailTicker:
        def __init__(self, symbol: str, session=None):
            call_count["value"] += 1

        def get_info(self):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(universe_enrichment, "_load_fundamental_cache", lambda: {})
    monkeypatch.setattr(universe_enrichment, "_save_fundamental_cache", lambda data: None)
    monkeypatch.setattr(universe_enrichment.yf, "Ticker", _AlwaysFailTicker)

    tickers = [f"T{index:03d}" for index in range(100)]
    universe_enrichment._fetch_detail_info(
        tickers,
        probe_size=5,
        min_probe_success_ratio=0.0,
    )

    # max_consecutive_failures is 50, so we should attempt exactly 50 symbols.
    assert call_count["value"] == 50


def test_enrichment_requests_detail_before_fast_info(monkeypatch):
    call_order: list[str] = []

    candidates = universe_enrichment.pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "sector": None,
                "industry": None,
                "source": "seed",
                "source_count": 1,
            }
        ]
    )

    monkeypatch.setattr(universe_enrichment, "_build_yfinance_session", lambda: object())
    monkeypatch.setattr(universe_enrichment, "_temporary_proxy_bypass", lambda: nullcontext())
    monkeypatch.setattr(universe_enrichment, "_load_fundamental_cache", lambda: {})

    def _fake_price_snapshot(*args, **kwargs):
        call_order.append("price")
        return {"AAPL": {"Price": 190.0, "AvgVolume": 10_000_000}}

    def _fake_detail_info(*args, **kwargs):
        call_order.append("detail")
        return {"AAPL": {"PE": 25.0, "ForwardPE": 22.0, "ROE": 0.3, "RevenueGrowth": 0.1}}

    def _fake_fast_info(*args, **kwargs):
        call_order.append("fast")
        return {"AAPL": {"MarketCap": 1_000_000_000, "Beta": 1.1}}

    monkeypatch.setattr(universe_enrichment, "_fetch_price_volume_snapshot", _fake_price_snapshot)
    monkeypatch.setattr(universe_enrichment, "_fetch_detail_info", _fake_detail_info)
    monkeypatch.setattr(universe_enrichment, "_fetch_fast_info", _fake_fast_info)

    result = universe_enrichment.enrich_universe_candidates(
        candidates,
        fast_info_limit=1,
        detail_limit=1,
        report_coverage=False,
    )

    assert not result.empty
    assert call_order[0] == "price"
    assert call_order.index("detail") < call_order.index("fast")
