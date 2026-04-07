from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pandas as pd
from loguru import logger
from typing import Optional


class CacheManager:
    """Manage local SQLite cache for market data"""
    
    def __init__(self, db_path: str = "data/cache/market_data.db", expiry_hours: int = 24):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.expiry_hours = expiry_hours
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    fetch_timestamp DATETIME,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_fetch DATETIME,
                    earliest_date DATE,
                    latest_date DATE
                )
            """)
            
            conn.commit()
    
    def get_cached_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and fresh"""
        metadata = self._get_cache_metadata(symbol)
        
        if metadata is None:
            return None
        
        last_fetch = datetime.fromisoformat(metadata["last_fetch"])
        if datetime.now() - last_fetch > timedelta(hours=self.expiry_hours):
            logger.info(f"Cache expired for {symbol}")
            return None
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            data = pd.read_sql_query(
                query,
                conn,
                params=(symbol, start_date.date(), end_date.date()),
                index_col="date",
                parse_dates=["date"],
            )
        
        if data.empty:
            return None
        
        logger.info(f"Retrieved {len(data)} cached rows for {symbol}")
        return data
    
    def save_data(self, symbol: str, data: pd.DataFrame):
        """Save data to cache"""
        if data.empty:
            return
        
        data_to_save = data.copy()
        data_to_save["symbol"] = symbol
        data_to_save["fetch_timestamp"] = datetime.now()
        data_to_save = data_to_save.reset_index()
        data_to_save = data_to_save.rename(columns={"index": "date"})
        
        with sqlite3.connect(self.db_path) as conn:
            data_to_save.to_sql(
                "prices",
                conn,
                if_exists="replace",
                index=False,
                method="multi",
            )
            
            self._update_metadata(symbol, data_to_save)
        
        logger.info(f"Cached {len(data)} rows for {symbol}")
    
    def _get_cache_metadata(self, symbol: str) -> Optional[dict]:
        """Get cache metadata for symbol"""
        query = "SELECT * FROM cache_metadata WHERE symbol = ?"
        
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql_query(query, conn, params=(symbol,))
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    def _update_metadata(self, symbol: str, data: pd.DataFrame):
        """Update cache metadata"""
        metadata = {
            "symbol": symbol,
            "last_fetch": datetime.now().isoformat(),
            "earliest_date": data["date"].min(),
            "latest_date": data["date"].max(),
        }
        
        with sqlite3.connect(self.db_path) as conn:
            pd.DataFrame([metadata]).to_sql(
                "cache_metadata",
                conn,
                if_exists="replace",
                index=False,
            )
