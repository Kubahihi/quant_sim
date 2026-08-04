from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple

# All four columns must be present for OHLC validation to proceed.
_REQUIRED_OHLC_COLS: tuple[str, ...] = ("open", "high", "low", "close")


class PriceValidator:
    """Validate OHLCV price data."""

    @staticmethod
    def validate_ohlc_logic(data: pd.DataFrame) -> Tuple[bool, list[str]]:
        """Validate OHLC price relationships.

        Returns ``(is_valid, issues)`` where *issues* is a list of human-readable
        strings describing every problem found.  Never raises ``KeyError``.

        The function first checks that all required columns are present.
        If any are missing it returns a descriptive error immediately rather
        than proceeding and crashing on column access.
        """
        issues: list[str] = []

        missing = [col for col in _REQUIRED_OHLC_COLS if col not in data.columns]
        if missing:
            issues.append(f"Missing required OHLC columns: {missing}")
            return False, issues

        # Both comparisons use vectorised operations — no Python-level loops.
        high_valid = data["high"] >= data[["open", "close"]].max(axis=1)
        low_valid  = data["low"]  <= data[["open", "close"]].min(axis=1)

        if not high_valid.all():
            n_invalid = int((~high_valid).sum())
            issues.append(f"{n_invalid} row(s) where high < max(open, close)")

        if not low_valid.all():
            n_invalid = int((~low_valid).sum())
            issues.append(f"{n_invalid} row(s) where low > min(open, close)")

        return len(issues) == 0, issues

    @staticmethod
    def validate_missing_data(
        data: pd.DataFrame, threshold: float = 0.05
    ) -> Tuple[bool, list[str]]:
        """Check for excessive missing data per column.

        Returns ``(is_valid, issues)``.  Safe to call on an empty DataFrame.
        """
        if data.empty:
            return True, []

        issues: list[str] = []
        missing_pct = data.isna().sum() / len(data)

        for col, pct in missing_pct.items():
            if pct > threshold:
                issues.append(f"{col}: {pct:.2%} missing data")

        return len(issues) == 0, issues

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, dict]:
        """Run all validation checks and aggregate results.

        Returns ``(overall_valid, results_dict)`` where *results_dict* has
        keys ``"valid"`` (bool) and ``"issues"`` (list[str]).
        """
        results: dict = {
            "valid": True,
            "issues": [],
        }

        if data.empty:
            results["valid"] = False
            results["issues"].append("Empty dataframe")
            return False, results

        ohlc_valid, ohlc_issues = PriceValidator.validate_ohlc_logic(data)
        # FIX: was `ohlc_valid and missing_issues` — a non-empty list is
        # truthy, so datasets with missing data were incorrectly marked valid.
        missing_valid, missing_issues = PriceValidator.validate_missing_data(data)

        results["valid"] = ohlc_valid and missing_valid
        results["issues"].extend(ohlc_issues)
        results["issues"].extend(missing_issues)

        if not results["valid"]:
            logger.warning(f"Validation issues: {results['issues']}")

        return results["valid"], results
