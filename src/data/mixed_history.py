"""Combine uploaded returns and adjusted market prices over matching intervals."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re

import numpy as np
import pandas as pd

from .custom_history import ImportedHistory
from .price_alignment import normalize_daily_series


@dataclass
class MarketHistory:
    prices: pd.DataFrame
    currencies: dict[str, str]
    downloaded_at: str
    fx_prices: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_market_tickers(text: str) -> list[str]:
    tickers = [item.upper() for item in re.split(r"[\s,;]+", text.strip()) if item]
    if len(tickers) > 20:
        raise ValueError("Add no more than 20 market tickers per import.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Market tickers must be unique.")
    if any(not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=_-]{0,39}", item) for item in tickers):
        raise ValueError("Enter valid Yahoo Finance symbols separated by commas or new lines.")
    return tickers


def download_market_history(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
    *, base_currency: str | None = None,
) -> MarketHistory:
    """Use the existing adjusted-price provider, with explicit quote currencies.

    ``end`` is inclusive here; the underlying Yahoo fetcher uses an exclusive
    end. Downloads happen only on a user action, never as part of a UI rerun.
    """
    import yfinance as yf
    from .fetchers.yahoo_fetcher import YahooFetcher

    prices = YahooFetcher().fetch_close_prices(
        tickers, pd.Timestamp(start).to_pydatetime(),
        (pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime(),
    )
    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
    if missing:
        raise ValueError("No price history returned for: " + ", ".join(missing) + ". Check the symbols and requested dates.")
    currencies = {}
    for ticker in tickers:
        try:
            currency = yf.Ticker(ticker).get_history_metadata().get("currency")
        except Exception as exc:
            raise ValueError(f"Unable to verify the quote currency for {ticker}. Try downloading again.") from exc
        if not currency:
            raise ValueError(f"No quote currency returned for {ticker}. Choose a listing with a known currency.")
        # Minor-unit quotes have identical returns to their parent currency.
        currencies[ticker] = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}.get(str(currency), str(currency).upper())
    fx_prices = pd.DataFrame()
    if base_currency and any(code != base_currency.strip().upper() for code in currencies.values()):
        from src.analytics.currency_risk import required_fx_symbols

        symbols = required_fx_symbols(currencies.values(), base_currency)
        fx_prices = YahooFetcher().fetch_close_prices(
            list(symbols), pd.Timestamp(start).to_pydatetime(),
            (pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime(),
        )
        missing_fx = [symbol for symbol in symbols if symbol not in fx_prices or fx_prices[symbol].dropna().empty]
        if missing_fx:
            raise ValueError("Missing historical FX rates: " + ", ".join(missing_fx) + ". Try downloading again.")
    return MarketHistory(prices[tickers], currencies, datetime.now(timezone.utc).isoformat(), fx_prices)


def _endpoint_levels(series: pd.Series, endpoints: pd.DatetimeIndex, max_age: int, label: str):
    series = normalize_daily_series(series)
    series = pd.to_numeric(series, errors="raise").dropna()
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"No dated market prices available for {label}.")
    if not np.isfinite(series.to_numpy(dtype=float)).all() or (series <= 0).any():
        raise ValueError(f"Market prices for {label} must be finite and positive.")
    locations = series.index.get_indexer(endpoints, method="pad", tolerance=pd.Timedelta(days=max_age))
    levels = np.full(len(endpoints), np.nan)
    valid = locations >= 0
    levels[valid] = series.to_numpy(dtype=float)[locations[valid]]
    return levels, (locations, series.index)


def combine_histories(
    imported: ImportedHistory, market: MarketHistory, *, tickers: list[str],
    currency: str, market_names: list[str] | None = None,
) -> tuple[ImportedHistory, dict]:
    """Match price endpoints to uploaded intervals; never upsample returns.

    Daily inputs require exact session dates. Lower-frequency endpoints use
    the latest observed close on or before the uploaded date, at most seven
    calendar days earlier. Only incomplete leading/trailing overlap is trimmed;
    missing interior periods cause an explicit error.
    """
    if not currency.strip():
        raise ValueError("Specify the common data currency before adding market data.")
    labels = [str(name).strip() for name in (market_names if market_names is not None else tickers)]
    all_labels = [*imported.returns.columns, *labels]
    if len(labels) != len(tickers) or not all(labels) or len({str(x).casefold() for x in all_labels}) != len(all_labels):
        raise ValueError("Uploaded and downloaded assets must have unique names. Rename any overlapping assets.")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Select unique market tickers.")
    missing = [ticker for ticker in tickers if ticker not in market.prices]
    if missing:
        raise ValueError("Missing downloaded assets: " + ", ".join(missing))
    currency = currency.strip().upper()
    for ticker in tickers:
        if not market.currencies.get(ticker, "").strip():
            raise ValueError(f"{ticker} is quoted in an unknown currency. Download verified currency metadata before combining histories.")

    uploaded = imported.returns.copy()
    warnings = list(imported.warnings)
    # Drop timezone offsets while retaining local observation dates.
    uploaded.index = pd.DatetimeIndex(uploaded.index).tz_localize(None).normalize()
    if not uploaded.index.is_monotonic_increasing or uploaded.index.has_duplicates:
        raise ValueError("Uploaded observations must have sorted, unique dates.")
    baseline = imported.baseline_date
    if baseline is None:
        endpoints = uploaded.index
        target = uploaded.iloc[1:]
        warnings.append("The first uploaded return was excluded because its starting date is unknown. Include a dated empty baseline row to retain it in mixed analysis.")
    else:
        baseline = pd.Timestamp(baseline).tz_localize(None).normalize()
        if baseline >= uploaded.index[0]:
            raise ValueError("The baseline date must precede the first return observation.")
        codes = {"Weekly": "W-SUN", "Monthly": "M", "Quarterly": "Q", "Annual": "Y"}
        if imported.frequency in codes:
            code = codes[imported.frequency]
            if uploaded.index[0].to_period(code).ordinal - baseline.to_period(code).ordinal != 1:
                raise ValueError("The baseline must be in the period immediately before the first return.")
        elif (uploaded.index[0] - baseline).days > 7:
            raise ValueError("The daily baseline is more than seven days before the first return.")
        endpoints = pd.DatetimeIndex([baseline, *uploaded.index])
        target = uploaded
    if len(target) < 2:
        raise ValueError("At least two common return observations are required. Add more history or a dated baseline row.")

    market_returns = pd.DataFrame(index=target.index)
    endpoint_details = {}
    fx_endpoint_details = {}
    fx_rates = {}
    foreign_currencies = {market.currencies[ticker].upper() for ticker in tickers} - {currency}
    if foreign_currencies:
        from src.analytics.currency_risk import FX_USD_QUOTES, required_fx_symbols

        symbols = required_fx_symbols(foreign_currencies, currency)
        missing_fx = [symbol for symbol in symbols if symbol not in market.fx_prices]
        if missing_fx:
            raise ValueError("Missing historical FX rates: " + ", ".join(missing_fx) + ". Download market data again to convert currencies.")
        usd_levels = {"USD": np.ones(len(endpoints))}
        for code in foreign_currencies | {currency}:
            if code == "USD":
                continue
            symbol, invert = FX_USD_QUOTES[code]
            quote, detail = _endpoint_levels(market.fx_prices[symbol], endpoints, 7, symbol)
            usd_levels[code] = 1.0 / quote if invert else quote
            fx_endpoint_details[symbol] = detail
        fx_rates = {code: usd_levels[code] / usd_levels[currency] for code in foreign_currencies}
    coverage = {}
    for ticker, label in zip(tickers, labels):
        max_age = 0 if imported.frequency == "Daily" else 7
        levels, detail = _endpoint_levels(market.prices[ticker], endpoints, max_age, ticker)
        quote_currency = market.currencies[ticker].upper()
        if quote_currency != currency:
            levels = levels * fx_rates[quote_currency]
        periodic = levels[1:] / levels[:-1] - 1
        if np.isinf(periodic).any():
            raise ValueError(f"Return calculation overflowed for {ticker}. Check prices and FX rates.")
        market_returns[label] = periodic
        valid_returns = target.index[np.isfinite(periodic)]
        coverage[ticker] = {
            "first_return": str(valid_returns.min().date()) if len(valid_returns) else None,
            "last_return": str(valid_returns.max().date()) if len(valid_returns) else None,
            "available_returns": len(valid_returns), "currency": market.currencies[ticker],
            "reporting_currency": currency,
        }
        endpoint_details[ticker] = detail
    combined = pd.concat([target, market_returns], axis=1)
    complete = np.isfinite(combined.to_numpy(dtype=float)).all(axis=1)
    positions = np.flatnonzero(complete)
    if len(positions) < 2:
        raise ValueError("The uploaded and downloaded assets have fewer than two common return observations.")
    first, last = positions[0], positions[-1]
    if not complete[first:last + 1].all():
        bad_date = combined.index[first:last + 1][~complete[first:last + 1]][0]
        unavailable = combined.columns[~np.isfinite(combined.loc[bad_date].to_numpy(dtype=float))].tolist()
        raise ValueError(f"Missing market or FX observations inside the common period at {bad_date.date()}: {', '.join(unavailable)}. Repair the source data or choose another ticker; interior periods are not dropped.")
    result = combined.iloc[first:last + 1].copy()
    if (result <= -1).any().any():
        raise ValueError("Combined returns must be greater than -100%.")
    result.index.name = "Date"
    trimmed = len(target) - len(result)
    if trimmed:
        warnings.append(f"Common coverage excludes {trimmed} uploaded return observations at the beginning or end of the history.")
    warnings.append(
        "Downloaded daily returns use exact uploaded session dates."
        if imported.frequency == "Daily" else
        "Downloaded returns use adjusted closes on or before each uploaded observation date (up to seven calendar days earlier)."
    )
    # Audit which actual sessions supplied every retained market endpoint.
    sessions = {}
    for ticker, (locations, index) in endpoint_details.items():
        sessions[ticker] = {
            str(endpoints[i].date()): str(index[locations[i]].date())
            for i in range(first, last + 2)
        }
    fx_sessions = {
        symbol: {str(endpoints[i].date()): str(index[locations[i]].date()) for i in range(first, last + 2)}
        for symbol, (locations, index) in fx_endpoint_details.items()
    }
    if foreign_currencies:
        warnings.append(f"Market assets in {', '.join(sorted(foreign_currencies))} were converted to {currency} with historical FX rates. Returns include currency movements; no additional currency hedge or conversion fee is modeled.")
    metadata = {
        "provider": "Yahoo Finance", "downloaded_at": market.downloaded_at,
        "tickers": tickers, "asset_names": dict(zip(tickers, labels)),
        "adjusted_prices": True, "currencies": market.currencies,
        "coverage": coverage, "endpoint_sessions": sessions,
        "reporting_currency": currency,
        "fx_conversion": {
            "converted_from": sorted(foreign_currencies), "to": currency,
            "symbols": sorted(fx_endpoint_details), "endpoint_sessions": fx_sessions,
            "rate_convention": "reporting currency units per one quote currency unit",
            "return_formula": "(1 + local_return) * (fx_end / fx_start) - 1",
            "alignment": "previous_FX_close_max_7_calendar_days",
        },
        "alignment": "exact_daily_dates" if imported.frequency == "Daily" else "previous_adjusted_close_max_7_calendar_days",
        "first_return_without_baseline_excluded": baseline is None,
        "trimmed_for_common_coverage": int(trimmed),
    }
    return ImportedHistory(result, imported.frequency, warnings, endpoints[first]), metadata
