"""
Regression tests for CacheManager.

Covers the three bugs fixed:
  1. save_data previously used if_exists='replace', destroying all other tickers.
  2. _update_metadata had the same bug, wiping all metadata on every save.
  3. Date values were stored as Timestamp objects (undefined SQLite encoding).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path: Path) -> CacheManager:
    """Return a CacheManager backed by a temporary SQLite database."""
    return CacheManager(db_path=str(tmp_path / "test_cache.db"), expiry_hours=24)


def _make_df(prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    dates = pd.date_range(start=start, periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "open":   prices,
            "high":   [p * 1.01 for p in prices],
            "low":    [p * 0.99 for p in prices],
            "close":  prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# BUG 1 + BUG 2: Multi-ticker isolation
# ---------------------------------------------------------------------------

class TestMultiTickerIsolation:
    """Saving ticker B must not delete ticker A's rows or metadata."""

    def test_prices_for_first_ticker_survive_second_save(self, cache: CacheManager):
        df_aapl = _make_df([150.0, 151.0, 152.0])
        df_msft = _make_df([300.0, 301.0, 302.0])

        cache.save_data("AAPL", df_aapl)
        cache.save_data("MSFT", df_msft)

        with sqlite3.connect(cache.db_path) as conn:
            symbols = {
                row[0]
                for row in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()
            }

        assert "AAPL" in symbols, "AAPL rows were deleted when MSFT was saved"
        assert "MSFT" in symbols

    def test_metadata_for_first_ticker_survives_second_save(self, cache: CacheManager):
        df_aapl = _make_df([150.0, 151.0, 152.0])
        df_msft = _make_df([300.0, 301.0, 302.0])

        cache.save_data("AAPL", df_aapl)
        cache.save_data("MSFT", df_msft)

        with sqlite3.connect(cache.db_path) as conn:
            symbols = {
                row[0]
                for row in conn.execute(
                    "SELECT symbol FROM cache_metadata"
                ).fetchall()
            }

        assert "AAPL" in symbols, "AAPL metadata was deleted when MSFT was saved"
        assert "MSFT" in symbols

    def test_three_tickers_all_persist(self, cache: CacheManager):
        for ticker, price in [("AAPL", 150.0), ("MSFT", 300.0), ("GOOG", 2800.0)]:
            cache.save_data(ticker, _make_df([price, price + 1.0]))

        with sqlite3.connect(cache.db_path) as conn:
            price_symbols = {
                r[0] for r in conn.execute(
                    "SELECT DISTINCT symbol FROM prices"
                ).fetchall()
            }
            meta_symbols = {
                r[0] for r in conn.execute(
                    "SELECT symbol FROM cache_metadata"
                ).fetchall()
            }

        assert price_symbols == {"AAPL", "MSFT", "GOOG"}
        assert meta_symbols  == {"AAPL", "MSFT", "GOOG"}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Repeated saves of the same data must not create duplicate rows."""

    def test_repeated_save_does_not_duplicate_rows(self, cache: CacheManager):
        df = _make_df([100.0, 101.0, 102.0])
        cache.save_data("AAPL", df)
        cache.save_data("AAPL", df)

        with sqlite3.connect(cache.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE symbol = 'AAPL'"
            ).fetchone()[0]

        assert count == 3, f"Expected 3 rows, got {count}"

    def test_update_row_with_new_data(self, cache: CacheManager):
        df_old = _make_df([100.0, 101.0])
        cache.save_data("AAPL", df_old)

        # New data for the same dates with different close prices.
        df_new = _make_df([110.0, 111.0])
        cache.save_data("AAPL", df_new)

        with sqlite3.connect(cache.db_path) as conn:
            rows = conn.execute(
                "SELECT close FROM prices WHERE symbol = 'AAPL' ORDER BY date"
            ).fetchall()

        closes = [r[0] for r in rows]
        assert closes == [110.0, 111.0], f"Expected updated closes, got {closes}"

    def test_incremental_saves_preserve_full_metadata_range(self, cache: CacheManager):
        cache.save_data("AAPL", _make_df([100.0, 101.0], start="2024-01-08"))
        cache.save_data("AAPL", _make_df([98.0, 99.0], start="2024-01-02"))
        cache.save_data("AAPL", _make_df([102.0, 103.0], start="2024-01-15"))

        metadata = cache._get_cache_metadata("AAPL")

        assert metadata is not None
        assert metadata["earliest_date"] == "2024-01-02"
        assert metadata["latest_date"] == "2024-01-16"


# ---------------------------------------------------------------------------
# BUG 3: Date serialisation
# ---------------------------------------------------------------------------

class TestDateSerialisation:
    """Dates must be stored as plain strings, not Python objects."""

    def test_dates_are_stored_as_strings(self, cache: CacheManager):
        df = _make_df([150.0, 151.0])
        cache.save_data("AAPL", df)

        with sqlite3.connect(cache.db_path) as conn:
            dates = conn.execute(
                "SELECT date FROM prices WHERE symbol = 'AAPL'"
            ).fetchall()

        for (date_val,) in dates:
            assert isinstance(date_val, str), (
                f"Expected str, got {type(date_val).__name__}: {date_val!r}"
            )

    def test_metadata_dates_are_strings(self, cache: CacheManager):
        df = _make_df([150.0])
        cache.save_data("AAPL", df)

        with sqlite3.connect(cache.db_path) as conn:
            row = conn.execute(
                "SELECT earliest_date, latest_date FROM cache_metadata WHERE symbol = 'AAPL'"
            ).fetchone()

        assert row is not None
        for val in row:
            assert isinstance(val, str), (
                f"Expected str, got {type(val).__name__}: {val!r}"
            )


# ---------------------------------------------------------------------------
# Cache retrieval
# ---------------------------------------------------------------------------

class TestCacheRetrieval:
    def test_get_cached_data_returns_correct_rows(self, cache: CacheManager):
        df = _make_df([150.0, 151.0, 152.0], start="2024-01-02")
        cache.save_data("AAPL", df)

        start = datetime(2024, 1, 1)
        end   = datetime(2024, 1, 31)
        result = cache.get_cached_data("AAPL", start, end)

        assert result is not None
        assert len(result) == 3

    def test_get_cached_data_returns_none_for_unknown_symbol(self, cache: CacheManager):
        result = cache.get_cached_data("UNKNOWN", datetime(2024, 1, 1), datetime(2024, 1, 31))
        assert result is None

    def test_get_cached_data_returns_none_after_expiry(self, cache: CacheManager):
        df = _make_df([150.0])
        cache.save_data("AAPL", df)

        # Artificially expire the cache by setting expiry_hours to 0.
        cache.expiry_hours = 0
        result = cache.get_cached_data("AAPL", datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert result is None

    def test_save_empty_dataframe_is_noop(self, cache: CacheManager):
        cache.save_data("AAPL", pd.DataFrame())

        with sqlite3.connect(cache.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM prices WHERE symbol = 'AAPL'"
            ).fetchone()[0]
        assert count == 0

    def test_operations_release_sqlite_file_handle(self, cache: CacheManager, tmp_path: Path):
        cache.save_data("AAPL", _make_df([150.0, 151.0]))
        assert cache.get_cached_data(
            "AAPL",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        ) is not None

        moved_path = tmp_path / "moved_cache.db"
        cache.db_path.rename(moved_path)
        moved_path.rename(cache.db_path)
