from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk_metrics import calculate_volatility
from src.portfolio_tracker.live_analytics import build_live_competition_analytics


def _returns(**columns: list[float]) -> pd.DataFrame:
    length = len(next(iter(columns.values()))) if columns else 0
    return pd.DataFrame(columns, index=pd.date_range("2025-01-02", periods=length, freq="B"))


def test_duplicate_lots_are_aggregated_and_actual_pnl_is_tagged_by_thesis() -> None:
    positions = [
        {"ticker": "aaa", "status": "open", "quantity": 10, "entry_price": 10},
        {"ticker": "AAA", "status": "open", "quantity": 5, "entry_price": 12},
        {
            "ticker": "AAA",
            "status": "closed",
            "quantity": 2,
            "entry_price": 10,
            "exit_price": 8,
        },
    ]
    theses = {
        "aaa": {
            "payload_json": json.dumps(
                {"sector": "Technology", "primary_goal": "Long-term growth"}
            )
        }
    }

    result = build_live_competition_analytics(
        positions,
        live_prices={"AAA": 15},
        asset_returns=_returns(AAA=[0.01, -0.01, 0.02, -0.005]),
        initial_capital=500,
        thesis_by_ticker=theses,
    )

    ticker = result["ledger_attribution_by_ticker"].set_index("Ticker").loc["AAA"]
    assert ticker["OpenLots"] == 2
    assert ticker["ClosedLots"] == 1
    assert ticker["OpenQuantity"] == pytest.approx(15)
    assert ticker["OpenCost"] == pytest.approx(160)
    assert ticker["CurrentValue"] == pytest.approx(225)
    assert ticker["UnrealizedPnL"] == pytest.approx(65)
    assert ticker["RealizedPnL"] == pytest.approx(-4)
    assert ticker["TotalPnL"] == pytest.approx(61)
    assert ticker["Sector"] == "Technology"
    assert ticker["ClientGoal"] == "Long-term growth"

    exposure = result["open_exposures"].set_index("Ticker").loc["AAA"]
    assert exposure["Quantity"] == pytest.approx(15)
    assert exposure["CurrentPrice"] == pytest.approx(15)
    assert result["ledger_attribution_by_sector"].iloc[0]["TotalPnL"] == pytest.approx(61)
    assert result["ledger_attribution_by_goal"].iloc[0]["TotalPnL"] == pytest.approx(61)


def test_current_cash_includes_realized_pnl_and_reconciles_to_equity() -> None:
    result = build_live_competition_analytics(
        positions=[
            {"ticker": "AAA", "status": "open", "quantity": 1, "entry_price": 100},
            {
                "ticker": "BBB",
                "status": "closed",
                "quantity": 1,
                "entry_price": 100,
                "exit_price": 120,
            },
        ],
        live_prices={"AAA": 110},
        asset_returns=_returns(AAA=[0.01, -0.01, 0.005]),
        initial_capital=500,
    )

    assert result["ledger_performance"]["realized_pnl"] == pytest.approx(20)
    assert result["current_cash"] == pytest.approx(420)
    assert result["current_equity"] == pytest.approx(530)
    assert result["current_cash"] + result["open_exposures"]["CurrentValue"].sum() == pytest.approx(530)
    assert result["current_weights"].sum() == pytest.approx(1.0)
    assert result["current_weights"]["CASH"] == pytest.approx(420 / 530)


def test_cash_weight_lowers_historical_risk_without_renormalising_risky_asset() -> None:
    asset_returns = _returns(AAA=[0.02, -0.01, 0.015, -0.02, 0.01, -0.005])
    result = build_live_competition_analytics(
        positions=[
            {
                "ticker": "AAA",
                "status": "open",
                "quantity": 2_500,
                "entry_price": 100,
            }
        ],
        live_prices={"AAA": 100},
        asset_returns=asset_returns,
    )

    risky_only_volatility = calculate_volatility(asset_returns["AAA"])
    assert result["risk_proxy_available"] is True
    assert result["proxy_weights"]["AAA"] == pytest.approx(0.5)
    assert result["proxy_weights"]["CASH"] == pytest.approx(0.5)
    assert result["proxy_weights"].sum() == pytest.approx(1.0)
    assert result["portfolio_returns"].equals((asset_returns["AAA"] * 0.5).rename("current_weight_proxy"))
    assert result["risk_metrics"]["volatility"] == pytest.approx(risky_only_volatility * 0.5)


def test_missing_history_is_preserved_as_unmodeled_weight_and_warned() -> None:
    result = build_live_competition_analytics(
        positions=[
            {"ticker": "AAA", "status": "open", "quantity": 1, "entry_price": 100},
            {"ticker": "BBB", "status": "open", "quantity": 1, "entry_price": 100},
        ],
        live_prices={"AAA": 100, "BBB": 100},
        asset_returns=_returns(AAA=[0.01, -0.01, 0.02, -0.005]),
        initial_capital=500,
    )

    assert result["risk_proxy_available"] is True
    assert result["risk_metrics"]["partial_coverage"] is True
    assert result["coverage"]["missing_history_tickers"] == ["BBB"]
    assert result["coverage"]["history_coverage_pct"] == pytest.approx(0.5)
    assert result["coverage"]["history_value_coverage_pct"] == pytest.approx(0.5)
    assert result["proxy_weights"]["UNMODELED"] == pytest.approx(0.2)
    assert result["proxy_weights"].sum() == pytest.approx(1.0)
    assert list(result["correlation"].columns) == ["AAA"]
    assert any("BBB" in warning and "understated" in warning for warning in result["warnings"])


def test_manual_and_entry_fallback_prices_are_reported_in_coverage() -> None:
    result = build_live_competition_analytics(
        positions=[
            {
                "ticker": "AAA",
                "status": "open",
                "quantity": 1,
                "entry_price": 100,
                "last_price": 105,
            },
            {"ticker": "BBB", "status": "open", "quantity": 1, "entry_price": 80},
        ],
        live_prices={},
        asset_returns=_returns(
            AAA=[0.01, -0.01, 0.02],
            BBB=[0.005, -0.002, 0.004],
        ),
        initial_capital=500,
    )

    assert result["coverage"]["live_price_tickers"] == 0
    assert result["coverage"]["manual_price_tickers"] == 1
    assert result["coverage"]["entry_fallback_tickers"] == 1
    assert result["coverage"]["price_coverage_pct"] == 0.0
    assert any("Manual prices" in warning for warning in result["warnings"])
    assert any("Entry-price fallbacks" in warning for warning in result["warnings"])


def test_benchmark_metrics_use_only_overlapping_proxy_observations() -> None:
    returns = _returns(AAA=[0.01, -0.005, 0.007, 0.002])
    benchmark = pd.Series(
        [0.008, -0.004, 0.006, 0.001],
        index=returns.index,
        dtype=float,
    )
    result = build_live_competition_analytics(
        positions=[{"ticker": "AAA", "status": "open", "quantity": 5, "entry_price": 100}],
        live_prices={"AAA": 100},
        asset_returns=returns,
        benchmark_returns=benchmark,
        initial_capital=500,
    )

    assert result["benchmark_metrics"]["benchmark_available"] is True
    assert result["benchmark_metrics"]["benchmark_ticker"] == "SPY"
    assert result["benchmark_metrics"]["benchmark_obs"] == 4
    assert result["coverage"]["benchmark_observations"] == 4

    no_overlap = benchmark.copy()
    no_overlap.index = pd.date_range("2026-01-02", periods=4, freq="B")
    unavailable = build_live_competition_analytics(
        positions=[{"ticker": "AAA", "status": "open", "quantity": 5, "entry_price": 100}],
        live_prices={"AAA": 100},
        asset_returns=returns,
        benchmark_returns=no_overlap,
        initial_capital=500,
    )
    assert unavailable["benchmark_metrics"]["benchmark_available"] is False
    assert unavailable["benchmark_metrics"]["reason"] == "insufficient_overlap"
    assert any("insufficient overlap" in warning for warning in unavailable["warnings"])


def test_actual_and_proxy_contributions_reconcile_to_their_respective_totals() -> None:
    result = build_live_competition_analytics(
        positions=[
            {"ticker": "AAA", "status": "open", "quantity": 2, "entry_price": 100},
            {"ticker": "BBB", "status": "open", "quantity": 1, "entry_price": 100},
        ],
        live_prices={"AAA": 110, "BBB": 90},
        asset_returns=_returns(
            AAA=[0.01, -0.005, 0.012, 0.004, -0.002],
            BBB=[0.003, 0.002, -0.004, 0.006, -0.001],
        ),
        initial_capital=500,
        thesis_by_ticker={
            "AAA": {"payload": {"sector": "Technology", "primary_goal": "Growth"}},
            "BBB": {"sector": "Healthcare", "primary_goal": "Stability"},
        },
    )

    actual = result["ledger_attribution_by_ticker"]
    expected_actual_return = result["ledger_performance"]["total_pnl"] / 500
    assert actual["ReturnContribution"].sum() == pytest.approx(expected_actual_return)
    assert actual["TotalPnL"].sum() == pytest.approx(result["ledger_performance"]["total_pnl"])
    assert actual["GrossPnLImpactPct"].sum() == pytest.approx(1.0)

    risk = result["risk_contribution"]
    assert set(risk["Ticker"]) == {"AAA", "BBB", "CASH"}
    assert risk["RiskBudgetPct"].sum() == pytest.approx(1.0)
    assert result["proxy_return_contribution"]["TotalContributionApprox"].sum() == pytest.approx(
        result["portfolio_returns"].sum()
    )


def test_empty_ledger_returns_stable_empty_outputs_and_full_cash_weight() -> None:
    result = build_live_competition_analytics(
        positions=[],
        live_prices={},
        asset_returns=pd.DataFrame(),
        initial_capital=500,
    )

    assert result["available"] is False
    assert result["risk_proxy_available"] is False
    assert result["current_equity"] == pytest.approx(500)
    assert result["current_cash"] == pytest.approx(500)
    assert result["current_weights"].to_dict() == {"CASH": 1.0}
    assert result["ledger_attribution_by_ticker"].empty
    assert result["open_exposures"].empty
    assert any("all cash" in warning for warning in result["warnings"])


def test_initial_capital_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        build_live_competition_analytics([], {}, pd.DataFrame(), initial_capital=np.nan)

