from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
import time

import pandas as pd
import yfinance as yf
from loguru import logger

from .base_fetcher import BaseFetcher


PROJECT_ROOT = Path(__file__).resolve().parents[3]
YFINANCE_CACHE_ENV = "YFINANCE_CACHE_DIR"
MAX_MULTI_FETCH_WORKERS = 4

_yfinance_cache_lock = threading.Lock()
_configured_yfinance_cache: Path | None = None


def _yfinance_cache_dir() -> Path:
    override = str(os.getenv(YFINANCE_CACHE_ENV, "") or "").strip()
    requested = Path(os.path.expandvars(override)).expanduser() if override else (
        PROJECT_ROOT / "data" / "cache" / "yfinance"
    )
    return requested.resolve()


def _configure_yfinance_cache() -> Path:
    """Configure yfinance's SQLite caches once for the requested location."""
    global _configured_yfinance_cache

    cache_dir = _yfinance_cache_dir()
    if _configured_yfinance_cache == cache_dir:
        return cache_dir

    with _yfinance_cache_lock:
        if _configured_yfinance_cache == cache_dir:
            return cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
        _configured_yfinance_cache = cache_dir
    return cache_dir


@dataclass
class FetchResult:
    data: pd.DataFrame
    success: bool
    error: str = ""

    @property
    def empty(self) -> bool:
        """Preserve read compatibility with callers that expected a DataFrame."""
        return bool(self.data.empty)

    def __getitem__(self, key):
        """Forward column reads to the wrapped DataFrame."""
        return self.data[key]


class YahooFetcher(BaseFetcher):
    """Yahoo Finance data fetcher"""

    def _fetch_many(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> list[tuple[str, FetchResult]]:
        """Fetch symbols concurrently while returning results in input order."""
        ordered_symbols = list(symbols)
        if not ordered_symbols:
            return []
        if len(ordered_symbols) == 1:
            symbol = ordered_symbols[0]
            return [(symbol, self.fetch_prices(symbol, start_date, end_date, interval))]

        def fetch_one(symbol: str) -> FetchResult:
            return self.fetch_prices(symbol, start_date, end_date, interval)

        workers = min(MAX_MULTI_FETCH_WORKERS, len(ordered_symbols))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yahoo-fetch") as executor:
            fetch_results = executor.map(fetch_one, ordered_symbols)
            return list(zip(ordered_symbols, fetch_results))
    
    def fetch_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> FetchResult:
        """Fetch OHLCV price data from Yahoo Finance with retry backoff"""
        max_retries = 3

        try:
            _configure_yfinance_cache()
        except Exception as exc:
            msg = f"Unable to configure yfinance cache for {symbol}: {exc}"
            logger.error(msg)
            return FetchResult(data=pd.DataFrame(), success=False, error=msg)
        
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                )
                
                if data.empty:
                    msg = f"No data returned for {symbol}"
                    logger.warning(msg)
                    return FetchResult(data=pd.DataFrame(), success=False, error=msg)
                
                data = data.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
                
                data = data[["open", "high", "low", "close", "volume"]]
                
                logger.info(f"Fetched {len(data)} rows for {symbol}")
                return FetchResult(data=data, success=True)
                
            except Exception as e:
                msg = f"Error fetching {symbol} (Attempt {attempt + 1}/{max_retries}): {e}"
                logger.error(msg)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return FetchResult(data=pd.DataFrame(), success=False, error=msg)
        
        return FetchResult(data=pd.DataFrame(), success=False, error="Max retries reached")
    
    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        result: dict[str, pd.DataFrame] = {}

        for symbol, fetch_result in self._fetch_many(symbols, start_date, end_date, interval):
            if fetch_result.success and not fetch_result.data.empty:
                result[symbol] = fetch_result.data
        
        return result
    
    def fetch_close_prices(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch close prices for multiple symbols as DataFrame"""
        prices: dict[str, pd.Series] = {}

        for symbol, fetch_result in self._fetch_many(symbols, start_date, end_date):
            if fetch_result.success and not fetch_result.data.empty:
                prices[symbol] = fetch_result.data["close"]
        
        return pd.DataFrame(prices)

    def fetch_close_prices_with_liquidity(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        *,
        adv_window: int = 30,
    ) -> dict[str, object]:
        """Fetch close prices and causal rolling average daily dollar volume."""
        if not isinstance(adv_window, int) or adv_window < 1:
            raise ValueError("adv_window must be a positive integer.")
        prices: dict[str, pd.Series] = {}
        adv_history: dict[str, pd.Series] = {}
        latest_adv: dict[str, float] = {}
        for symbol, fetch_result in self._fetch_many(symbols, start_date, end_date):
            if not fetch_result.success or fetch_result.data.empty:
                continue
            data = fetch_result.data
            close = pd.to_numeric(data["close"], errors="coerce")
            volume = pd.to_numeric(data["volume"], errors="coerce")
            dollar_volume = (close * volume).where(lambda values: values > 0)
            rolling_adv = dollar_volume.rolling(
                window=adv_window,
                min_periods=min(5, adv_window),
            ).mean()
            prices[symbol] = close
            adv_history[symbol] = rolling_adv
            valid_adv = rolling_adv.dropna()
            if not valid_adv.empty:
                latest_adv[symbol] = float(valid_adv.iloc[-1])
        return {
            "prices": pd.DataFrame(prices),
            "average_daily_dollar_volume": latest_adv,
            "adv_history": pd.DataFrame(adv_history),
            "adv_window": adv_window,
        }
