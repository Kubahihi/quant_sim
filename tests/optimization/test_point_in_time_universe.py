from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    align_point_in_time_membership,
    parse_point_in_time_membership,
    run_optimization_walk_forward,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(91)
    dates = pd.date_range("2020-01-02", periods=240, freq="B")
    values = rng.normal([0.0004, 0.0003, 0.0005], [0.010, 0.009, 0.012], size=(240, 3))
    frame = pd.DataFrame(values, index=dates, columns=["A", "B", "C"])
    # A leaves the investable universe before its return history disappears.
    frame.loc[dates[160]:, "A"] = np.nan
    return frame


def _membership(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"A": True, "B": True, "C": False},
            {"A": False, "B": True, "C": True},
        ],
        index=[index[0], index[159]],
    )


def test_membership_parser_accepts_long_form_and_never_backfills() -> None:
    raw = pd.DataFrame({
        "Date": ["2024-01-03", "2024-01-03", "2024-02-01"],
        "Ticker": ["A", "B", "A"],
        "Is Member": [1, 1, 0],
    })
    parsed = parse_point_in_time_membership(raw)
    returns_index = pd.date_range("2024-01-01", "2024-02-05", freq="B")
    aligned = align_point_in_time_membership(
        parsed,
        return_index=returns_index,
        symbols=["A", "B"],
    )

    assert aligned.loc[pd.Timestamp("2024-01-01")].sum() == 0
    assert bool(aligned.loc[pd.Timestamp("2024-01-03"), "A"]) is True
    assert bool(aligned.loc[pd.Timestamp("2024-02-01"), "A"]) is False


def test_point_in_time_walk_forward_changes_active_universe_causally() -> None:
    returns = _returns()
    result = run_optimization_walk_forward(
        returns,
        optimizer="minimum_variance",
        train_periods=100,
        rebalance_periods=20,
        max_weight=0.80,
        universe_membership=_membership(returns.index),
        membership_lag_periods=1,
        transaction_cost_bps=10.0,
    )

    assert result["success"] is True
    assert result["survivorship_bias_controlled"] is True
    assert result["validation_type"] == "point_in_time_rolling_reoptimization_out_of_sample"
    assert result["windows"][0]["active_symbols"] == ["A", "B"]
    assert result["windows"][3]["active_symbols"] == ["B", "C"]
    assert result["weights_history"].iloc[3]["A"] == pytest.approx(0.0)


def test_missing_return_for_a_held_asset_fails_closed() -> None:
    returns = _returns()
    membership = pd.DataFrame(
        [{"A": True, "B": True, "C": True}],
        index=[returns.index[0]],
    )

    with pytest.raises(ValueError, match="missing out-of-sample return for held"):
        run_optimization_walk_forward(
            returns,
            optimizer="minimum_variance",
            train_periods=100,
            rebalance_periods=20,
            max_weight=0.80,
            universe_membership=membership,
        )


def test_membership_columns_must_match_return_union() -> None:
    returns = _returns()
    with pytest.raises(ValueError, match="membership columns must match"):
        run_optimization_walk_forward(
            returns,
            train_periods=100,
            universe_membership=pd.DataFrame(
                [{"A": True, "B": True}],
                index=[returns.index[0]],
            ),
        )


def test_membership_alignment_accepts_timezone_aware_market_index() -> None:
    membership = pd.DataFrame(
        [{"A": True}],
        index=[pd.Timestamp("2026-01-02")],
    )
    market_index = pd.date_range("2026-01-02", periods=3, freq="B", tz="UTC")

    aligned = align_point_in_time_membership(
        membership,
        return_index=market_index,
        symbols=["A"],
    )

    assert aligned.index.equals(market_index)
    assert aligned["A"].all()
