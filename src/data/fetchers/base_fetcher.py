from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd


class BaseFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    @abstractmethod
    def fetch_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV price data"""
        pass
    
    @abstractmethod
    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        pass
