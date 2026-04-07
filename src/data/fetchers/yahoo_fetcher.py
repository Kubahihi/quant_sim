from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf
from loguru import logger

from .base_fetcher import BaseFetcher


class YahooFetcher(BaseFetcher):
    """Yahoo Finance data fetcher"""
    
    def fetch_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            data = data.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            
            data = data[["open", "high", "low", "close", "volume"]]
            
            logger.info(f"Fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
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
            data = self.fetch_prices(symbol, start_date, end_date, interval)
            if not data.empty:
                result[symbol] = data
        
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
            data = self.fetch_prices(symbol, start_date, end_date)
            if not data.empty:
                prices[symbol] = data["close"]
        
        return pd.DataFrame(prices)
