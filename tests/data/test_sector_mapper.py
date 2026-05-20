"""Tests for the sector mapper module."""

from __future__ import annotations

import pandas as pd

from src.data.sector_mapper import (
    GICS_SECTORS,
    KNOWN_SECTOR_MAP,
    gather_sector_classifications,
    get_sector_for_ticker,
    enrich_with_sectors,
    normalize_symbol,
)


def test_known_sector_map_contains_major_tickers():
    """Verify that the known sector map contains major tickers."""
    major_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "XOM"]
    for ticker in major_tickers:
        assert ticker in KNOWN_SECTOR_MAP, f"Missing ticker: {ticker}"


def test_known_sector_map_has_valid_sectors():
    """Verify that all sectors in the known map are valid GICS sectors."""
    for ticker, (sector, industry) in KNOWN_SECTOR_MAP.items():
        assert sector in GICS_SECTORS, f"Invalid sector for {ticker}: {sector}"
        assert industry and len(industry) > 0, f"Empty industry for {ticker}"


def test_gather_sector_classifications_returns_data():
    """Test that gather_sector_classifications returns a non-empty DataFrame."""
    result = gather_sector_classifications()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "ticker" in result.columns
    assert "sector" in result.columns
    assert "industry" in result.columns


def test_gather_sector_classifications_has_expected_columns():
    """Test that the result has the expected columns."""
    result = gather_sector_classifications()
    expected_columns = ["ticker", "sector", "industry", "source"]
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"


def test_get_sector_for_ticker_returns_valid_data():
    """Test that get_sector_for_ticker returns valid sector data for known tickers."""
    # Test a few known tickers
    test_cases = [
        ("AAPL", "Information Technology"),
        ("JPM", "Financials"),
        ("JNJ", "Health Care"),
        ("XOM", "Energy"),
    ]

    for ticker, expected_sector in test_cases:
        sector, industry = get_sector_for_ticker(ticker)
        assert sector == expected_sector, f"Expected {expected_sector} for {ticker}, got {sector}"
        assert industry is not None, f"Industry should not be None for {ticker}"


def test_get_sector_for_ticker_returns_none_for_unknown():
    """Test that get_sector_for_ticker returns None for unknown tickers."""
    sector, industry = get_sector_for_ticker("UNKNOWNXYZ")
    assert sector is None
    assert industry is None


def test_enrich_with_sectors_fills_missing_data():
    """Test that enrich_with_sectors fills missing sector data."""
    # Create a test DataFrame with some missing sectors
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "UNKNOWN"],
        "sector": [None, None, None],
        "industry": [None, None, None],
    })

    result = enrich_with_sectors(df)

    # AAPL and MSFT should have sectors filled
    assert result.loc[0, "sector"] == "Information Technology"
    assert result.loc[1, "sector"] == "Information Technology"
    # UNKNOWN should remain None
    assert pd.isna(result.loc[2, "sector"]) or result.loc[2, "sector"] is None


def test_enrich_with_sectors_preserves_existing_data():
    """Test that enrich_with_sectors doesn't overwrite existing sector data."""
    df = pd.DataFrame({
        "ticker": ["AAPL", "CUSTOM"],
        "sector": ["Custom Sector", None],
        "industry": ["Custom Industry", None],
    })

    result = enrich_with_sectors(df)

    # AAPL should keep its custom sector (existing data preserved)
    assert result.loc[0, "sector"] == "Custom Sector"
    # CUSTOM should get filled from the mapper
    assert result.loc[1, "sector"] is None or pd.isna(result.loc[1, "sector"])  # Not in known map


def test_normalize_symbol_handles_various_formats():
    """Test that normalize_symbol handles various ticker formats."""
    test_cases = [
        ("aapl", "AAPL"),
        ("AAPL", "AAPL"),
        ("BRK.B", "BRK-B"),  # Dots are converted to dashes for yfinance compatibility
        ("BRK.B ", "BRK-B"),
        (" brk.b ", "BRK-B"),
    ]

    for input_symbol, expected in test_cases:
        result = normalize_symbol(input_symbol)
        assert result == expected, f"normalize_symbol('{input_symbol}') = '{result}', expected '{expected}'"


def test_sector_mapper_covers_all_gics_sectors():
    """Test that the sector mapper covers all GICS sectors."""
    result = gather_sector_classifications()
    covered_sectors = set(result["sector"].unique())

    # Check that most GICS sectors are covered
    # (at least 10 out of 11, allowing for some sectors to have no mapped tickers)
    covered_count = len(covered_sectors & set(GICS_SECTORS))
    assert covered_count >= 10, f"Only {covered_count} GICS sectors covered: {covered_sectors}"