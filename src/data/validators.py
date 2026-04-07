import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple


class PriceValidator:
    """Validate OHLCV price data"""
    
    @staticmethod
    def validate_ohlc_logic(data: pd.DataFrame) -> Tuple[bool, list[str]]:
        """Validate OHLC price relationships"""
        issues = []
        
        if "high" not in data.columns or "low" not in data.columns:
            return True, []
        
        high_valid = data["high"] >= data[["open", "close"]].max(axis=1)
        low_valid = data["low"] <= data[["open", "close"]].min(axis=1)
        
        if not high_valid.all():
            n_invalid = (~high_valid).sum()
            issues.append(f"{n_invalid} rows with high < max(open, close)")
        
        if not low_valid.all():
            n_invalid = (~low_valid).sum()
            issues.append(f"{n_invalid} rows with low > min(open, close)")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_missing_data(data: pd.DataFrame, threshold: float = 0.05) -> Tuple[bool, list[str]]:
        """Check for missing data"""
        issues = []
        
        missing_pct = data.isna().sum() / len(data)
        
        for col, pct in missing_pct.items():
            if pct > threshold:
                issues.append(f"{col}: {pct:.2%} missing data")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, dict]:
        """Run all validation checks"""
        results = {
            "valid": True,
            "issues": [],
        }
        
        if data.empty:
            results["valid"] = False
            results["issues"].append("Empty dataframe")
            return False, results
        
        ohlc_valid, ohlc_issues = PriceValidator.validate_ohlc_logic(data)
        missing_valid, missing_issues = PriceValidator.validate_missing_data(data)
        
        results["valid"] = ohlc_valid and missing_valid
        results["issues"].extend(ohlc_issues)
        results["issues"].extend(missing_issues)
        
        if not results["valid"]:
            logger.warning(f"Validation issues: {results['issues']}")
        
        return results["valid"], results
