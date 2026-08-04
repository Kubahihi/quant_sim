from datetime import datetime
from typing import Optional
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
