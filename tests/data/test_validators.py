"""
Regression tests for PriceValidator.

Covers bugs fixed:
  1. validate_ohlc_logic raised KeyError when 'open' or 'close' was missing.
  2. validate_data aggregated results using `ohlc_valid and missing_issues`
     (a list), so datasets with excessive missing data were incorrectly valid.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data.validators import PriceValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(**overrides) -> pd.DataFrame:
    """Return a minimal 3-row OHLCV DataFrame, with optional column overrides."""
    base = {
        "open":   [10.0, 11.0, 12.0],
        "high":   [11.0, 12.0, 13.0],
        "low":    [ 9.0, 10.0, 11.0],
        "close":  [10.5, 11.5, 12.5],
        "volume": [1000,  2000,  3000],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# BUG 1: Missing column guard — no KeyError
# ---------------------------------------------------------------------------

class TestMissingColumnGuard:
    """validate_ohlc_logic must never raise KeyError."""

    @pytest.mark.parametrize("drop_col", ["open", "high", "low", "close"])
    def test_missing_single_column_returns_failure_not_keyerror(self, drop_col: str):
        df = _ohlcv().drop(columns=[drop_col])
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False, "Should be invalid when a required column is missing"
        assert any("Missing" in issue for issue in issues), (
            f"Expected a 'Missing' issue description, got: {issues}"
        )

    def test_all_ohlc_columns_missing_returns_failure(self):
        df = pd.DataFrame({"volume": [1000, 2000, 3000]})
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False
        assert issues  # must have at least one descriptive issue

    def test_empty_dataframe_with_no_columns_returns_failure(self):
        valid, issues = PriceValidator.validate_ohlc_logic(pd.DataFrame())
        assert valid is False
        assert issues

    def test_complete_ohlc_valid_data_passes(self):
        df = _ohlcv()
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is True
        assert issues == []

    def test_only_high_low_missing_returns_failure_without_keyerror(self):
        """Earlier guard only checked high and low — open/close access would KeyError."""
        df = _ohlcv().drop(columns=["high", "low"])
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False

    def test_high_low_present_open_close_missing(self):
        df = pd.DataFrame({"high": [11.0, 12.0], "low": [9.0, 10.0]})
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False
        assert issues


# ---------------------------------------------------------------------------
# OHLC logic validation
# ---------------------------------------------------------------------------

class TestOHLCLogicValidation:
    def test_high_below_max_open_close_is_detected(self):
        df = _ohlcv(high=[9.0, 12.0, 13.0])  # first row: high < open
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False
        assert any("high" in issue for issue in issues)

    def test_low_above_min_open_close_is_detected(self):
        df = _ohlcv(low=[11.0, 10.0, 11.0])  # first row: low > open=10
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False
        assert any("low" in issue for issue in issues)

    def test_both_errors_reported(self):
        df = _ohlcv(high=[9.0, 12.0, 13.0], low=[11.0, 10.0, 11.0])
        valid, issues = PriceValidator.validate_ohlc_logic(df)
        assert valid is False
        assert len(issues) == 2


# ---------------------------------------------------------------------------
# BUG 2: validate_data aggregation logic
# ---------------------------------------------------------------------------

class TestValidateDataAggregation:
    """validate_data was using `ohlc_valid and missing_issues` (list as bool)."""

    def test_excessive_missing_data_makes_valid_false(self):
        """If >5% of values are missing, validate_data must return valid=False."""
        df = _ohlcv()
        # Inject >5% NaN into 'close' (1/3 = 33%)
        df.loc[0, "close"] = np.nan
        overall_valid, results = PriceValidator.validate_data(df)
        assert overall_valid is False, (
            "validate_data incorrectly returned valid=True despite missing data"
        )
        assert results["valid"] is False
        assert results["issues"]  # must contain at least one issue description

    def test_clean_data_is_valid(self):
        df = _ohlcv()
        overall_valid, results = PriceValidator.validate_data(df)
        assert overall_valid is True
        assert results["valid"] is True
        assert results["issues"] == []

    def test_empty_dataframe_is_invalid(self):
        overall_valid, results = PriceValidator.validate_data(pd.DataFrame())
        assert overall_valid is False
        assert "Empty dataframe" in results["issues"]

    def test_ohlc_violation_and_missing_both_reported(self):
        df = _ohlcv(high=[9.0, 12.0, 13.0])  # OHLC violation
        df.loc[0, "close"] = np.nan            # missing data
        overall_valid, results = PriceValidator.validate_data(df)
        assert overall_valid is False
        assert len(results["issues"]) >= 2


# ---------------------------------------------------------------------------
# validate_missing_data edge cases
# ---------------------------------------------------------------------------

class TestValidateMissingData:
    def test_empty_dataframe_does_not_crash(self):
        valid, issues = PriceValidator.validate_missing_data(pd.DataFrame())
        assert valid is True
        assert issues == []

    def test_all_null_column_reported(self):
        df = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})
        valid, issues = PriceValidator.validate_missing_data(df)
        assert valid is False
        assert issues

    def test_below_threshold_passes(self):
        df = pd.DataFrame({"close": [1.0, np.nan, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0]})  # 1/20 = 5% — at threshold
        valid, issues = PriceValidator.validate_missing_data(df, threshold=0.05)
        # At exactly the threshold (not above) it should pass
        assert valid is True
