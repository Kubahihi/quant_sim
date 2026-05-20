from __future__ import annotations

from contextlib import contextmanager
from io import StringIO
from typing import Callable, Iterable
from urllib.request import Request, urlopen
import json
import os
import re

import pandas as pd


USER_AGENT = "quant-sim-universe/1.0 (research contact: local-app)"
VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,9}$")
SEC_NOISE_SUFFIXES = {"F", "Y", "Q", "W", "U", "R"}

# Heuristic keyword blocklist to remove obvious non-common-stock instruments.
NOISE_KEYWORDS = (
    " ETF",
    " ETN",
    " TRUST",
    " FUND",
    " INDEX",
    " PREFERRED",
    " DEPOSITARY",
    " WARRANT",
    " RIGHTS",
    " UNIT",
    " BOND",
    " NOTE",
    " INCOME SHARES",
    " MUTUAL",
    " CLOSED END",
)

NASDAQ_EXCHANGE_MAP = {
    "Q": "NASDAQ",
    "N": "NYSE",
    "A": "NYSE AMERICAN",
    "P": "NYSE ARCA",
    "Z": "CBOE",
    "V": "IEX",
}

PROXY_ENV_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]


def _is_blackhole_proxy(value: str | None) -> bool:
    if not value:
        return False
    lowered = str(value).strip().lower()
    return "127.0.0.1:9" in lowered


@contextmanager
def _temporary_proxy_bypass():
    original: dict[str, str] = {}
    removed: list[str] = []
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if _is_blackhole_proxy(value):
            original[key] = str(value)
            os.environ.pop(key, None)
            removed.append(key)
    try:
        yield
    finally:
        for key in removed:
            os.environ[key] = original[key]


def normalize_symbol(symbol: str) -> str:
    """Normalize ticker symbols to a yfinance-friendly US equity form."""
    cleaned = (symbol or "").strip().upper()
    cleaned = cleaned.replace(".", "-").replace("/", "-").replace("^", "-")
    cleaned = re.sub(r"[^A-Z0-9\-]", "", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned


def _is_likely_us_common_symbol(symbol: str) -> bool:
    """
    Coarse symbol-level filter to cut obvious SEC feed noise.

    The SEC ticker file includes many OTC tickers, rights, units and warrant
    artifacts that materially degrade Yahoo enrichment reliability.
    """
    text = normalize_symbol(symbol)
    if not text or not VALID_SYMBOL_RE.match(text):
        return False

    if len(text) > 5 and "-" not in text:
        return False

    if len(text) == 5 and text[-1] in SEC_NOISE_SUFFIXES:
        return False

    return True


def _first_non_empty(values: Iterable[object]) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "nan":
            return text
    return None


def _is_likely_common_stock(name: str) -> bool:
    text = f" {str(name or '').upper()} "
    return not any(keyword in text for keyword in NOISE_KEYWORDS)


def _download_text(url: str, timeout: int = 25) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with _temporary_proxy_bypass():
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _clean_nasdaq_footer_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_col = str(df.columns[0])
    mask = ~df[first_col].astype(str).str.contains("File Creation Time", na=False)
    return df.loc[mask].copy()


def _collect_from_nasdaq_trader() -> pd.DataFrame:
    """Collect symbols from NASDAQ Trader listing files."""
    rows: list[dict[str, object]] = []

    source_specs = [
        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
            "nasdaqtrader_nasdaq",
        ),
        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            "nasdaqtrader_other",
        ),
    ]

    for url, source_name in source_specs:
        text = _download_text(url)
        table = pd.read_csv(StringIO(text), sep="|", dtype=str).fillna("")
        table = _clean_nasdaq_footer_rows(table)
        if table.empty:
            continue

        symbol_col = _find_column(table, ["Symbol", "NASDAQ Symbol", "ACT Symbol", "CQS Symbol"])
        name_col = _find_column(table, ["Security Name"])
        exchange_col = _find_column(table, ["Listing Exchange", "Exchange"])
        etf_col = _find_column(table, ["ETF"])
        test_issue_col = _find_column(table, ["Test Issue"])

        if symbol_col is None:
            continue

        for _, record in table.iterrows():
            raw_symbol = str(record.get(symbol_col, "")).strip()
            if not raw_symbol:
                continue

            symbol = normalize_symbol(raw_symbol)
            if not symbol or not VALID_SYMBOL_RE.match(symbol):
                continue

            is_test_issue = str(record.get(test_issue_col, "")).strip().upper() == "Y"
            is_etf = str(record.get(etf_col, "")).strip().upper() == "Y"
            if is_test_issue or is_etf:
                continue

            company_name = str(record.get(name_col, "")).strip()
            if company_name and not _is_likely_common_stock(company_name):
                continue

            exchange_code = str(record.get(exchange_col, "")).strip().upper()
            exchange = NASDAQ_EXCHANGE_MAP.get(exchange_code, exchange_code or None)

            rows.append({
                "ticker": symbol,
                "company_name": company_name or None,
                "exchange": exchange,
                "sector": None,
                "industry": None,
                "source": source_name,
            })

    return pd.DataFrame(rows)


def _collect_from_sec_company_tickers() -> pd.DataFrame:
    """Collect symbols from SEC company ticker feed."""
    text = _download_text("https://www.sec.gov/files/company_tickers.json")
    payload = json.loads(text)

    rows: list[dict[str, object]] = []
    for item in payload.values():
        symbol = normalize_symbol(str(item.get("ticker", "")))
        if not _is_likely_us_common_symbol(symbol):
            continue

        company_name = str(item.get("title", "")).strip()
        if company_name and not _is_likely_common_stock(company_name):
            continue

        rows.append({
            "ticker": symbol,
            "company_name": company_name or None,
            "exchange": None,
            "sector": None,
            "industry": None,
            "source": "sec_company_tickers",
        })

    return pd.DataFrame(rows)


def _collect_from_sp500_wikipedia() -> pd.DataFrame:
    """Collect S&P 500 members (adds high-quality overlap symbols)."""
    with _temporary_proxy_bypass():
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    if table.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, record in table.iterrows():
        symbol = normalize_symbol(str(record.get("Symbol", "")))
        if not symbol or not VALID_SYMBOL_RE.match(symbol):
            continue

        rows.append({
            "ticker": symbol,
            "company_name": str(record.get("Security", "")).strip() or None,
            "exchange": str(record.get("Exchange", "")).strip() or None,
            "sector": str(record.get("GICS Sector", "")).strip() or None,
            "industry": str(record.get("GICS Sub-Industry", "")).strip() or None,
            "source": "wikipedia_sp500",
        })
    return pd.DataFrame(rows)


def _collect_from_sector_mapper() -> pd.DataFrame:
    """
    Collect sector/industry classifications from the dedicated sector mapper.

    This provides GICS sector data for ~500+ major tickers from static mappings
    and Wikipedia S&P 500, serving as a reliable sector source independent of
    yfinance API availability.
    """
    try:
        from src.data.sector_mapper import gather_sector_classifications
        sector_df = gather_sector_classifications()
        if sector_df.empty:
            return pd.DataFrame()
        # Add placeholder columns to match the standard candidate schema
        sector_df = sector_df.copy()
        sector_df["company_name"] = None
        sector_df = sector_df[["ticker", "company_name", "sector", "industry", "source"]]
        return sector_df
    except Exception:
        return pd.DataFrame()


def _fill_sectors_from_mapper(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sector/industry values from the sector mapper.

    This is called after the main aggregation to backfill any remaining
    sector gaps for tickers that exist in our static/curated mappings.
    """
    try:
        from src.data.sector_mapper import gather_sector_classifications
        sector_map = gather_sector_classifications()
        if sector_map.empty:
            return combined

        sector_lookup = dict(zip(sector_map["ticker"], sector_map["sector"]))
        industry_lookup = dict(zip(sector_map["ticker"], sector_map["industry"]))

        for idx in combined.index:
            ticker = str(combined.at[idx, "ticker"] or "").strip().upper()
            if not ticker:
                continue

            current_sector = combined.at[idx, "sector"]
            current_industry = combined.at[idx, "industry"]

            # Only fill if currently missing
            sector_missing = pd.isna(current_sector) or str(current_sector).strip() == ""
            industry_missing = pd.isna(current_industry) or str(current_industry).strip() == ""

            if sector_missing and ticker in sector_lookup:
                combined.at[idx, "sector"] = sector_lookup[ticker]
            if industry_missing and ticker in industry_lookup:
                combined.at[idx, "industry"] = industry_lookup[ticker]
    except Exception:
        pass

    return combined


def gather_universe_candidates() -> pd.DataFrame:
    """
    Gather and normalize US equity universe candidates from multiple sources.

    Returns a dataframe with columns:
    - ticker
    - company_name
    - exchange
    - sector
    - industry
    - source
    - source_count
    """
    collectors: list[Callable[[], pd.DataFrame]] = [
        _collect_from_nasdaq_trader,
        _collect_from_sec_company_tickers,
        _collect_from_sp500_wikipedia,
        _collect_from_sector_mapper,
    ]

    collected_frames: list[pd.DataFrame] = []
    for collector in collectors:
        try:
            frame = collector()
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        collected_frames.append(frame)

    if not collected_frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "exchange",
                "sector",
                "industry",
                "source",
                "source_count",
            ]
        )

    combined = pd.concat(collected_frames, ignore_index=True)
    combined["ticker"] = combined["ticker"].map(normalize_symbol)
    combined = combined[combined["ticker"].map(lambda value: bool(VALID_SYMBOL_RE.match(value or "")))]
    combined = combined.dropna(subset=["ticker"]).copy()

    # For sector and industry, prefer non-empty values using _first_non_empty
    # which will naturally prefer values from sources that provide them
    grouped = (
        combined.groupby("ticker", as_index=False)
        .agg({
            "company_name": _first_non_empty,
            "exchange": _first_non_empty,
            "sector": _first_non_empty,
            "industry": _first_non_empty,
            "source": lambda values: ",".join(sorted({str(item).strip() for item in values if str(item).strip()})),
        })
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    # Post-aggregation: fill remaining sector/industry gaps from the sector mapper
    grouped = _fill_sectors_from_mapper(grouped)

    grouped["source_count"] = grouped["source"].map(
        lambda value: len([item for item in str(value).split(",") if item.strip()])
    )
    return grouped
