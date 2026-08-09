"""
Tests for optimize_minimum_variance.

Covers both the original tests (preserved) and regression tests for every
bug fixed in the refactor:
  1. max_weight infeasibility fails closed instead of changing the mandate.
  2. Singular / near-zero-variance covariance matrix (no sqrt of negative).
  3. 'message' key present in return dict (API consistency with maximum_sharpe).
  4. allow_short=True + max_weight respected together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.minimum_variance import optimize_minimum_variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns() -> pd.DataFrame:
    np.random.seed(42)
    data = np.random.normal(0.001, 0.02, (100, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


@pytest.fixture
def low_corr_returns() -> pd.DataFrame:
    """Returns with clearly different variances to produce a non-trivial MV portfolio."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "LOW_VOL":  np.random.normal(0.0005, 0.005, 120),
            "MED_VOL":  np.random.normal(0.0008, 0.015, 120),
            "HIGH_VOL": np.random.normal(0.0012, 0.030, 120),
        }
    )


# ---------------------------------------------------------------------------
# Original tests (preserved, adapted for annualised covariance)
# ---------------------------------------------------------------------------

class TestBasicCorrectness:
    def test_weights_sum_to_one(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert np.isclose(result["weights"].sum(), 1.0, atol=1e-6)

    def test_weights_non_negative_long_only(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert np.all(result["weights"] >= -1e-8)

    def test_success_flag(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert result["success"] is True

    def test_sharpe_ratio_formula(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns, risk_free_rate=0.03)
        assert result["success"]
        vol = result["volatility"]
        ret = result["expected_return"]
        if vol > 0:
            expected_sharpe = (ret - 0.03) / vol
            assert np.isclose(result["sharpe_ratio"], expected_sharpe, atol=1e-8)
        else:
            assert result["sharpe_ratio"] == 0.0

    def test_message_key_present(self, sample_returns: pd.DataFrame):
        """Return dict must include 'message' key for API consistency."""
        result = optimize_minimum_variance(sample_returns)
        assert "message" in result, "'message' key missing — API inconsistency with maximum_sharpe"

    def test_symbols_match_columns(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert result["symbols"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Input cleaning
# ---------------------------------------------------------------------------

class TestInputCleaning:
    def test_nan_rows_dropped(self):
        df = pd.DataFrame({"A": [0.01, 0.02, np.nan, 0.04], "B": [0.02, -0.01, 0.03, 0.01]})
        result = optimize_minimum_variance(df)
        assert result["success"]
        assert len(result["symbols"]) == 2

    def test_insufficient_observations_raises(self):
        bad = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValueError, match="at least two observations"):
            optimize_minimum_variance(bad)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError):
            optimize_minimum_variance(pd.DataFrame())

    def test_all_nan_raises(self):
        df = pd.DataFrame({"A": [np.nan, np.nan, np.nan], "B": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            optimize_minimum_variance(df)


# ---------------------------------------------------------------------------
# BUG 1: max_weight infeasibility guard
# ---------------------------------------------------------------------------

class TestMaxWeightFeasibility:
    def test_feasible_max_weight_respected(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns, max_weight=0.6)
        assert result["success"]
        assert np.all(result["weights"] <= 0.6 + 1e-6)

    def test_infeasible_max_weight_fails_closed(self, sample_returns: pd.DataFrame):
        """A client mandate must never be silently relaxed by the optimizer."""
        with pytest.raises(ValueError, match="infeasible for 3 assets"):
            optimize_minimum_variance(sample_returns, max_weight=0.1)

    def test_invalid_max_weight_zero_raises(self, sample_returns: pd.DataFrame):
        with pytest.raises(ValueError, match="max_weight must be positive"):
            optimize_minimum_variance(sample_returns, max_weight=0.0)

    def test_invalid_max_weight_negative_raises(self, sample_returns: pd.DataFrame):
        with pytest.raises(ValueError, match="max_weight must be positive"):
            optimize_minimum_variance(sample_returns, max_weight=-0.5)


# ---------------------------------------------------------------------------
# allow_short + max_weight combined
# ---------------------------------------------------------------------------

class TestAllowShort:
    def test_allow_short_without_max_weight(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns, allow_short=True)
        assert result["success"]
        # Long-only constraint is lifted — some weights may be negative.
        assert np.all(result["weights"] >= -1.0 - 1e-6)
        assert np.all(result["weights"] <=  1.0 + 1e-6)

    def test_allow_short_with_max_weight_bounds(self, sample_returns: pd.DataFrame):
        """allow_short=True + max_weight=0.5 → bounds in [-0.5, 0.5]."""
        result = optimize_minimum_variance(sample_returns, allow_short=True, max_weight=0.5)
        assert result["success"]
        assert np.all(result["weights"] >= -0.5 - 1e-5)
        assert np.all(result["weights"] <=  0.5 + 1e-5)


# ---------------------------------------------------------------------------
# BUG 2: Numerical stability — near-zero variance
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_zero_volatility_no_nan(self):
        """Constant returns → singular covariance → volatility must be ≥ 0 and not NaN.

        With floating-point arithmetic the annualised sample covariance of
        identical series is not exactly zero — residuals are ~1e-30.  The
        _VOLATILITY_EPS guard in optimize_minimum_variance clips these to
        Sharpe=0 rather than returning a ratio of order 1e+15.
        """
        df = pd.DataFrame({"A": [0.01] * 10, "B": [0.01] * 10})
        result = optimize_minimum_variance(df, risk_free_rate=0.0)

        assert not np.isnan(result["volatility"]), "volatility must not be NaN"
        assert result["volatility"] >= 0.0, "volatility must be non-negative"
        # Sharpe must be finite and well-behaved (not exploding due to ÷ ε).
        assert not np.isnan(result["sharpe_ratio"]), "sharpe_ratio must not be NaN"
        assert not np.isinf(result["sharpe_ratio"]), (
            f"sharpe_ratio is infinite ({result['sharpe_ratio']!r}); "
            "epsilon guard may be missing or set too low"
        )
        # With volatility effectively zero the guard returns Sharpe=0.
        assert result["sharpe_ratio"] == 0.0, (
            f"Expected Sharpe=0 via epsilon guard, got {result['sharpe_ratio']!r}"
        )

    def test_low_variance_returns_stable(self, low_corr_returns: pd.DataFrame):
        result = optimize_minimum_variance(low_corr_returns)
        assert result["success"]
        assert not np.isnan(result["volatility"])
        assert not np.isnan(result["sharpe_ratio"])
        # MV portfolio should heavily favour the low-vol asset.
        idx = result["symbols"].index("LOW_VOL")
        assert result["weights"][idx] > 0.5, (
            "MV portfolio should overweight LOW_VOL given its much lower variance"
        )

    def test_volatility_non_negative(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert result["volatility"] >= 0.0


# ---------------------------------------------------------------------------
# Return dict completeness
# ---------------------------------------------------------------------------

class TestReturnDictSchema:
    EXPECTED_KEYS = {"weights", "symbols", "expected_return", "volatility",
                     "sharpe_ratio", "success", "message"}

    def test_all_keys_present(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        missing = self.EXPECTED_KEYS - set(result.keys())
        assert not missing, f"Return dict missing keys: {missing}"

    def test_weights_is_numpy_array(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert isinstance(result["weights"], np.ndarray)

    def test_symbols_is_list(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert isinstance(result["symbols"], list)

    def test_scalar_fields_are_float(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        for key in ("expected_return", "volatility", "sharpe_ratio"):
            assert isinstance(result[key], float), f"{key!r} is {type(result[key]).__name__}"

    def test_success_is_bool(self, sample_returns: pd.DataFrame):
        result = optimize_minimum_variance(sample_returns)
        assert isinstance(result["success"], bool)
