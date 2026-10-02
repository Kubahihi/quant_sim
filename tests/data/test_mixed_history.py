from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.data.custom_history import ImportedHistory, prepare_history
from src.data.mixed_history import (
    MarketHistory, combine_histories, download_market_history, parse_market_tickers,
)


def uploaded(frequency="Monthly", baseline=True):
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"])
    returns = pd.DataFrame({"Uploaded": [0.02, -0.01, 0.03, 0.01]}, index=dates)
    returns.index.name = "Date"
    return ImportedHistory(returns, frequency, [], pd.Timestamp("2023-12-29") if baseline else None)


def market(dates=None, values=None, currency="USD"):
    dates = dates if dates is not None else ["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]
    values = values if values is not None else [100, 110, 99, 118.8, 124.74]
    return MarketHistory(pd.DataFrame({"SPY": values}, index=pd.to_datetime(dates)), {"SPY": currency}, "2026-09-26T00:00:00+00:00")


def test_monthly_matching_preserves_uploaded_values_and_first_return():
    history, detail = combine_histories(uploaded(), market(), tickers=["SPY"], currency="USD")
    pd.testing.assert_series_equal(history.returns.Uploaded, uploaded().returns.Uploaded)
    np.testing.assert_allclose(history.returns.SPY, [0.1, -0.1, 0.2, 0.05])
    assert history.periods_per_year == 12
    assert detail["trimmed_for_common_coverage"] == 0
    assert detail["endpoint_sessions"]["SPY"]["2023-12-29"] == "2023-12-29"


def test_multiple_market_assets_keep_requested_order_and_names():
    prices = market()
    prices.prices["AGG"] = [100, 101, 103, 102, 105]
    prices.currencies["AGG"] = "USD"
    result, detail = combine_histories(uploaded(), prices, tickers=["AGG", "SPY"], currency="USD", market_names=["Bonds", "Equities"])
    assert list(result.returns) == ["Uploaded", "Bonds", "Equities"]
    assert detail["asset_names"] == {"AGG": "Bonds", "SPY": "Equities"}
    assert result.returns.Bonds.iloc[0] == pytest.approx(0.01)
    assert result.returns.Equities.iloc[0] == pytest.approx(0.1)


def test_different_month_end_dates_use_no_future_prices():
    history = uploaded()
    history.returns.index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    prices = market()
    prices.prices.loc[pd.Timestamp("2024-04-01"), "SPY"] = 900
    result, detail = combine_histories(history, prices, tickers=["SPY"], currency="USD")
    assert result.returns.SPY.iloc[2] == pytest.approx(0.2)
    assert detail["endpoint_sessions"]["SPY"]["2024-03-31"] == "2024-03-28"


def test_unknown_baseline_drops_only_first_return():
    result, detail = combine_histories(uploaded(baseline=False), market(), tickers=["SPY"], currency="USD")
    assert len(result.returns) == 3
    assert result.returns.index[0] == pd.Timestamp("2024-02-29")
    assert detail["first_return_without_baseline_excluded"]
    assert any("starting date is unknown" in warning for warning in result.warnings)


def test_ipo_coverage_trims_prefix_and_retains_monthly_frequency():
    prices = market()
    prices.prices = prices.prices.iloc[2:]
    result, detail = combine_histories(uploaded(), prices, tickers=["SPY"], currency="USD")
    assert len(result.returns) == 2
    assert detail["trimmed_for_common_coverage"] == 2
    assert result.baseline_date == pd.Timestamp("2024-02-29")
    assert result.periods_per_year == 12


def test_stale_endpoint_and_interior_gap_are_rejected():
    history = uploaded()
    history.returns.loc[pd.Timestamp("2024-05-31")] = 0.01
    prices = market()
    prices.prices = prices.prices.drop(index=pd.Timestamp("2024-02-29"))
    prices.prices.loc[pd.Timestamp("2024-05-31")] = 130
    with pytest.raises(ValueError, match="inside the common period"):
        combine_histories(history, prices, tickers=["SPY"], currency="USD")


def test_daily_exact_dates_do_not_fill_missing_sessions():
    history = ImportedHistory(pd.DataFrame({"Uploaded": [0.01] * 5}, index=pd.bdate_range("2024-01-02", periods=5)), "Daily", [], pd.Timestamp("2024-01-01"))
    prices = market(pd.bdate_range("2024-01-01", periods=6), [100, 101, 102, 103, 104, 105])
    result, detail = combine_histories(history, prices, tickers=["SPY"], currency="USD")
    np.testing.assert_allclose(result.returns.SPY, [1 / 100, 1 / 101, 1 / 102, 1 / 103, 1 / 104])
    assert detail["alignment"] == "exact_daily_dates"
    prices.prices = prices.prices.drop(index=pd.Timestamp("2024-01-03"))
    with pytest.raises(ValueError, match="inside the common period"):
        combine_histories(history, prices, tickers=["SPY"], currency="USD")


@pytest.mark.parametrize("fault,match", [
    ("currency", "Missing historical FX rates"), ("unknown_currency", "unknown currency"),
    ("duplicate", "unique names"), ("missing", "Missing downloaded"),
    ("zero", "finite and positive"), ("infinity", "finite and positive"),
    ("too_short", "fewer than two common"),
])
def test_rejects_invalid_market_inputs(fault, match):
    prices = market()
    kwargs = {}
    if fault == "currency":
        prices.currencies["SPY"] = "EUR"
    elif fault == "unknown_currency":
        prices.currencies = {}
    elif fault == "duplicate":
        kwargs["market_names"] = ["uploaded"]
    elif fault == "missing":
        prices.prices = prices.prices.rename(columns={"SPY": "OTHER"})
    elif fault == "zero":
        prices.prices.iloc[0, 0] = 0
    elif fault == "infinity":
        prices.prices.iloc[0, 0] = np.inf
    else:
        prices.prices = prices.prices.iloc[-2:]
    with pytest.raises(ValueError, match=match):
        combine_histories(uploaded(), prices, tickers=["SPY"], currency="USD", **kwargs)


def test_baseline_preserved_by_importer_for_prices_and_return_baseline():
    table = pd.DataFrame({"Date": pd.to_datetime(["2023-12-29", "2024-01-31", "2024-02-29"]), "A": [100.0, 101.0, 102.0]})
    options = dict(date_column="Date", asset_columns=["A"], frequency="Monthly")
    assert prepare_history(table, value_type="Prices / index levels", **options).baseline_date == pd.Timestamp("2023-12-29")
    table["A"] = [np.nan, 0.01, 0.02]
    assert prepare_history(table, value_type="Returns", **options).baseline_date == pd.Timestamp("2023-12-29")


def test_ticker_parser():
    assert parse_market_tickers("spy, agg\nBRK-B;^GSPC") == ["SPY", "AGG", "BRK-B", "^GSPC"]
    with pytest.raises(ValueError, match="unique"):
        parse_market_tickers("spy,SPY")
    with pytest.raises(ValueError, match="valid Yahoo"):
        parse_market_tickers("AAPL<script>")


def test_download_provider_contract(monkeypatch):
    from src.data.fetchers.yahoo_fetcher import YahooFetcher
    import yfinance as yf

    fetch = Mock(return_value=market().prices)
    monkeypatch.setattr(YahooFetcher, "fetch_close_prices", fetch)
    monkeypatch.setattr(yf, "Ticker", lambda ticker: Mock(get_history_metadata=lambda: {"currency": "USD"}))
    result = download_market_history(["SPY"], pd.Timestamp("2023-12-22"), pd.Timestamp("2024-04-30"))
    assert result.currencies == {"SPY": "USD"}
    assert fetch.call_args.args[2] == pd.Timestamp("2024-05-01")
    fetch.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No price history returned for: SPY"):
        download_market_history(["SPY"], pd.Timestamp("2023-12-22"), pd.Timestamp("2024-04-30"))


def test_eur_to_usd_includes_currency_return_without_changing_upload():
    prices = market(currency="EUR")
    prices.fx_prices = pd.DataFrame({"EURUSD=X": [1.0, 1.05, 1.02, 1.08, 1.10]}, index=prices.prices.index)
    result, detail = combine_histories(uploaded(), prices, tickers=["SPY"], currency="USD")
    assert result.returns.SPY.iloc[0] == pytest.approx(0.155)  # 1.10 × 1.05 − 1
    expected = (prices.prices.SPY * prices.fx_prices["EURUSD=X"]).pct_change().iloc[1:]
    np.testing.assert_allclose(result.returns.SPY, expected)
    pd.testing.assert_series_equal(result.returns.Uploaded, uploaded().returns.Uploaded)
    assert detail["fx_conversion"]["converted_from"] == ["EUR"]
    assert detail["reporting_currency"] == "USD"


@pytest.mark.parametrize("source,target", [("USD", "EUR"), ("GBP", "EUR"), ("JPY", "USD")])
def test_inverse_and_cross_currency_directions(source, target):
    prices = market(currency=source)
    quotes = pd.DataFrame({"EURUSD=X": [1.0, 1.05, 1.02, 1.08, 1.10],
                           "GBPUSD=X": [1.2, 1.25, 1.22, 1.3, 1.32],
                           "JPY=X": [140, 145, 142, 146, 148]}, index=prices.prices.index)
    prices.fx_prices = quotes
    usd = {"USD": pd.Series(1.0, index=quotes.index), "EUR": quotes["EURUSD=X"], "GBP": quotes["GBPUSD=X"], "JPY": 1 / quotes["JPY=X"]}
    expected = (prices.prices.SPY * usd[source] / usd[target]).pct_change().iloc[1:]
    result, _ = combine_histories(uploaded(), prices, tickers=["SPY"], currency=target)
    np.testing.assert_allclose(result.returns.SPY, expected)


def test_fx_missing_interior_blocks_and_never_uses_future_quote():
    history = uploaded()
    history.returns.loc[pd.Timestamp("2024-05-31")] = 0.01
    prices = market(currency="EUR")
    prices.prices.loc[pd.Timestamp("2024-05-31")] = 130
    prices.fx_prices = pd.DataFrame({"EURUSD=X": [1.0, 1.05, 1.02, 1.08, 1.10, 1.11]}, index=prices.prices.index)
    prices.fx_prices = prices.fx_prices.rename(index={pd.Timestamp("2024-02-29"): pd.Timestamp("2024-03-01")})
    with pytest.raises(ValueError, match="FX observations inside"):
        combine_histories(history, prices, tickers=["SPY"], currency="USD")


def test_fx_holiday_uses_previous_close_and_records_date():
    prices = market(currency="EUR")
    prices.fx_prices = pd.DataFrame({"EURUSD=X": [1.0, 1.05, 1.02, 1.08, 1.10]}, index=prices.prices.index)
    prices.fx_prices = prices.fx_prices.rename(index={pd.Timestamp("2024-02-29"): pd.Timestamp("2024-02-28")})
    result, detail = combine_histories(uploaded(), prices, tickers=["SPY"], currency="USD")
    assert len(result.returns) == 4
    assert detail["fx_conversion"]["endpoint_sessions"]["EURUSD=X"]["2024-02-29"] == "2024-02-28"


@pytest.mark.parametrize("value", [0, -1, np.inf])
def test_invalid_fx_rate_is_not_silently_used(value):
    prices = market(currency="EUR")
    prices.fx_prices = pd.DataFrame({"EURUSD=X": [1.0, value, 1.02, 1.08, 1.10]}, index=prices.prices.index)
    with pytest.raises(ValueError, match="finite and positive"):
        combine_histories(uploaded(), prices, tickers=["SPY"], currency="USD")


def test_downloader_fetches_required_fx_only(monkeypatch):
    from src.data.fetchers.yahoo_fetcher import YahooFetcher
    import yfinance as yf

    prices = market(currency="EUR").prices
    fx = pd.DataFrame({"EURUSD=X": [1.0, 1.05, 1.02, 1.08, 1.1]}, index=prices.index)
    fetch = Mock(side_effect=[prices, fx])
    monkeypatch.setattr(YahooFetcher, "fetch_close_prices", fetch)
    monkeypatch.setattr(yf, "Ticker", lambda ticker: Mock(get_history_metadata=lambda: {"currency": "EUR"}))
    result = download_market_history(["SPY"], pd.Timestamp("2023-12-22"), pd.Timestamp("2024-04-30"), base_currency="USD")
    assert fetch.call_args_list[1].args[0] == ["EURUSD=X"]
    pd.testing.assert_frame_equal(result.fx_prices, fx)
    fetch.side_effect = [prices, pd.DataFrame()]
    with pytest.raises(ValueError, match="Missing historical FX rates"):
        download_market_history(["SPY"], pd.Timestamp("2023-12-22"), pd.Timestamp("2024-04-30"), base_currency="USD")
