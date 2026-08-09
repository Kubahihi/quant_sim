from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import runpy
import threading

import pandas as pd
import pytest

import main as example_main
from src.data.fetchers import yahoo_fetcher
from src.data.fetchers.yahoo_fetcher import FetchResult, YahooFetcher


def test_yfinance_cache_is_project_local_overridable_and_thread_safe(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "project"
    override_dir = tmp_path / "override"
    configured_paths: list[str] = []
    calls_lock = threading.Lock()

    def fake_set_cache(path: str) -> None:
        with calls_lock:
            configured_paths.append(path)

    monkeypatch.setattr(yahoo_fetcher, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(yahoo_fetcher, "_configured_yfinance_cache", None)
    monkeypatch.setattr(yahoo_fetcher.yf, "set_tz_cache_location", fake_set_cache)
    monkeypatch.delenv(yahoo_fetcher.YFINANCE_CACHE_ENV, raising=False)

    default_path = yahoo_fetcher._configure_yfinance_cache()
    assert default_path == (project_root / "data" / "cache" / "yfinance").resolve()
    assert default_path.is_dir()

    monkeypatch.setenv(yahoo_fetcher.YFINANCE_CACHE_ENV, str(override_dir))
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(lambda _: yahoo_fetcher._configure_yfinance_cache(), range(16)))

    expected_override = override_dir.resolve()
    assert paths == [expected_override] * 16
    assert expected_override.is_dir()
    assert configured_paths == [str(default_path), str(expected_override)]


@pytest.mark.parametrize("method_name", ["fetch_multiple", "fetch_close_prices"])
def test_multi_ticker_fetch_is_bounded_concurrent_and_ordered(monkeypatch, method_name) -> None:
    symbols = ["D", "B", "A", "C", "H", "F", "E", "G"]
    dates = pd.date_range("2026-01-02", periods=2, freq="B")
    barrier = threading.Barrier(yahoo_fetcher.MAX_MULTI_FETCH_WORKERS)
    state_lock = threading.Lock()
    calls: list[tuple[str, str]] = []
    active = 0
    max_active = 0

    def fake_fetch(symbol, start_date, end_date, interval="1d"):
        nonlocal active, max_active
        with state_lock:
            calls.append((symbol, interval))
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=5)
            value = float(ord(symbol) - ord("A") + 1)
            return FetchResult(
                data=pd.DataFrame({"close": [value, value + 0.5]}, index=dates),
                success=True,
            )
        finally:
            with state_lock:
                active -= 1

    fetcher = YahooFetcher()
    monkeypatch.setattr(fetcher, "fetch_prices", fake_fetch)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 1, 10)

    if method_name == "fetch_multiple":
        result = fetcher.fetch_multiple(symbols, start_date, end_date, interval="1wk")
        assert list(result) == symbols
        expected_interval = "1wk"
    else:
        result = fetcher.fetch_close_prices(symbols, start_date, end_date)
        assert result.columns.tolist() == symbols
        expected_interval = "1d"

    assert Counter(symbol for symbol, _ in calls) == Counter(symbols)
    assert {interval for _, interval in calls} == {expected_interval}
    assert max_active == yahoo_fetcher.MAX_MULTI_FETCH_WORKERS


def test_main_returns_nonzero_when_no_market_data(monkeypatch, capsys) -> None:
    class EmptyFetcher:
        def fetch_close_prices(self, symbols, start_date, end_date):
            return pd.DataFrame()

    monkeypatch.setattr(example_main, "YahooFetcher", EmptyFetcher)

    assert example_main.main() == 1
    assert "Error: No data fetched" in capsys.readouterr().out


def test_main_script_exits_with_failure_code_when_no_market_data(monkeypatch) -> None:
    class EmptyFetcher:
        def fetch_close_prices(self, symbols, start_date, end_date):
            return pd.DataFrame()

    monkeypatch.setattr(yahoo_fetcher, "YahooFetcher", EmptyFetcher)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(example_main.__file__, run_name="__main__")

    assert exc_info.value.code == 1
