from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import pandas as pd
from loguru import logger
from typing import Optional


class CacheManager:
    """Manage local SQLite cache for market data.

    The underlying SQLite schema uses a composite PRIMARY KEY (symbol, date)
    on the ``prices`` table and a simple PRIMARY KEY (symbol) on
    ``cache_metadata``. Price rows use ``INSERT OR REPLACE`` and metadata uses
    ``ON CONFLICT ... DO UPDATE`` so that:

    * Existing rows for *other* symbols are never touched.
    * Rows for the current symbol are updated or inserted atomically inside a
      single transaction.
    * Incremental saves expand, rather than overwrite, the cached date range.
    * Repeated calls with the same data are idempotent.
    """

    def __init__(self, db_path: str = "data/cache/market_data.db", expiry_hours: int = 24):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.expiry_hours = expiry_hours
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema (idempotent)."""
        # ``sqlite3.Connection`` commits/rolls back in its context manager but
        # does not close the underlying handle. Close explicitly so short-lived
        # cache managers do not leave the database locked on Windows.
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT NOT NULL,
                    date   TEXT NOT NULL,
                    open   REAL,
                    high   REAL,
                    low    REAL,
                    close  REAL,
                    volume INTEGER,
                    fetch_timestamp TEXT,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol        TEXT PRIMARY KEY NOT NULL,
                    last_fetch    TEXT,
                    earliest_date TEXT,
                    latest_date   TEXT
                )
            """)
            conn.commit()

    def get_cached_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and fresh."""
        metadata = self._get_cache_metadata(symbol)

        if metadata is None:
            return None

        try:
            last_fetch = datetime.fromisoformat(str(metadata["last_fetch"]))
        except (TypeError, ValueError):
            logger.warning(f"Invalid cache metadata timestamp for {symbol}")
            return None
        now = datetime.now(last_fetch.tzinfo) if last_fetch.tzinfo else datetime.now()
        if now - last_fetch > timedelta(hours=self.expiry_hours):
            logger.info(f"Cache expired for {symbol}")
            return None

        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """

        with closing(sqlite3.connect(self.db_path)) as conn:
            data = pd.read_sql_query(
                query,
                conn,
                params=(symbol, str(start_date.date()), str(end_date.date())),
                index_col="date",
                parse_dates=["date"],
            )

        if data.empty:
            return None

        logger.info(f"Retrieved {len(data)} cached rows for {symbol}")
        return data

    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Upsert price rows for *symbol* without touching any other symbol.

        Uses ``INSERT OR REPLACE INTO`` which maps directly to the
        ``PRIMARY KEY (symbol, date)`` constraint, making the operation
        idempotent and non-destructive for other tickers.
        """
        if data.empty:
            return

        # Normalise: reset index so 'date' becomes a plain column.
        data_to_save = data.copy()
        data_to_save.index.name = "date"  # ensure the index is named before reset
        data_to_save = data_to_save.reset_index()
        # rename index column regardless of its original name
        if "index" in data_to_save.columns and "date" not in data_to_save.columns:
            data_to_save = data_to_save.rename(columns={"index": "date"})

        data_to_save["symbol"] = symbol
        fetch_ts = datetime.now(timezone.utc).isoformat()
        data_to_save["fetch_timestamp"] = fetch_ts

        # Ensure date is stored as a plain ISO string, not a Timestamp object.
        data_to_save["date"] = data_to_save["date"].astype(str)

        rows = data_to_save[
            ["symbol", "date", "open", "high", "low", "close", "volume", "fetch_timestamp"]
        ].itertuples(index=False, name=None)

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO prices
                    (symbol, date, open, high, low, close, volume, fetch_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            # Update metadata for this symbol only.
            conn.execute(
                """
                INSERT INTO cache_metadata
                    (symbol, last_fetch, earliest_date, latest_date)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_fetch = excluded.last_fetch,
                    earliest_date = MIN(
                        COALESCE(cache_metadata.earliest_date, excluded.earliest_date),
                        excluded.earliest_date
                    ),
                    latest_date = MAX(
                        COALESCE(cache_metadata.latest_date, excluded.latest_date),
                        excluded.latest_date
                    )
                """,
                (
                    symbol,
                    fetch_ts,
                    str(data_to_save["date"].min()),
                    str(data_to_save["date"].max()),
                ),
            )
            conn.commit()

        logger.info(f"Cached {len(data)} rows for {symbol}")

    def _get_cache_metadata(self, symbol: str) -> Optional[dict]:
        """Get cache metadata for a single symbol."""
        query = "SELECT * FROM cache_metadata WHERE symbol = ?"
        with closing(sqlite3.connect(self.db_path)) as conn:
            result = pd.read_sql_query(query, conn, params=(symbol,))

        if result.empty:
            return None

        return result.iloc[0].to_dict()

    def _update_metadata(self, symbol: str, data: pd.DataFrame) -> None:
        """Deprecated: metadata is now written atomically inside save_data.

        Kept as a no-op so external callers that reference this private method
        do not break immediately.  Will be removed in a future cleanup pass.
        """
        logger.warning(
            "_update_metadata() is deprecated and no longer performs any work; "
            "metadata is written by save_data()."
        )
