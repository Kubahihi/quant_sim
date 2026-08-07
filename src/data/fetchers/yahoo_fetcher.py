from datetime import datetime
import pandas as pd
import yfinance as yf
from loguru import logger
import time
from dataclasses import dataclass

from .base_fetcher import BaseFetcher


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
    
    def fetch_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> FetchResult:
        """Fetch OHLCV price data from Yahoo Finance with retry backoff"""
        max_retries = 3
        
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
        result = {}
        
        for symbol in symbols:
            fetch_result = self.fetch_prices(symbol, start_date, end_date, interval)
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
        prices = {}
        
        for symbol in symbols:
            fetch_result = self.fetch_prices(symbol, start_date, end_date)
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
        for symbol in symbols:
            fetch_result = self.fetch_prices(symbol, start_date, end_date)
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
