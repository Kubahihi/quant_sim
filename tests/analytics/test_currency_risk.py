from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.currency_risk import (
    aggregate_currency_exposure,
    build_fx_rate_history,
    build_fx_stress_table,
    calculate_fx_risk,
    optimize_currency_hedges,
    required_fx_symbols,
)


def _fx_history(periods: int = 300) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=periods)
    t = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "EURUSD=X": 1.08 * np.cumprod(1.0 + 0.0001 + 0.0030 * np.sin(t / 7.0)),
            "JPY=X": 145.0 * np.cumprod(1.0 - 0.00005 + 0.0020 * np.cos(t / 11.0)),
            "CZK=X": 22.5 * np.cumprod(1.0 + 0.00002 + 0.0015 * np.sin(t / 5.0)),
        },
        index=index,
    )


def test_required_symbols_and_cross_rates_have_consistent_orientation():
    prices = _fx_history()

    assert required_fx_symbols(["EUR", "JPY"], "CZK") == ("CZK=X", "EURUSD=X", "JPY=X")
    rates = build_fx_rate_history(prices, ["EUR", "JPY"], base_currency="CZK")

    expected_eurczk = prices["EURUSD=X"].iloc[-1] * prices["CZK=X"].iloc[-1]
    expected_jpyczk = prices["CZK=X"].iloc[-1] / prices["JPY=X"].iloc[-1]
    assert rates["EUR"].iloc[-1] == pytest.approx(expected_eurczk)
    assert rates["JPY"].iloc[-1] == pytest.approx(expected_jpyczk)
    assert (rates["CZK"] == 1.0).all()


def test_exposure_aggregation_preserves_net_and_gross_values():
    positions = pd.DataFrame(
        [
            {"Asset": "European equity", "Currency": "EUR", "MarketValueLocal": 100_000.0},
            {"Asset": "EUR liability", "Currency": "EUR", "MarketValueLocal": -20_000.0},
            {"Asset": "Cash", "Currency": "USD", "MarketValueLocal": 50_000.0},
        ]
    )
    exposure = aggregate_currency_exposure(positions, {"EUR": 1.1}, base_currency="USD")
    eur = exposure.set_index("Currency").loc["EUR"]

    assert eur["NetExposureBase"] == pytest.approx(88_000.0)
    assert eur["GrossExposureBase"] == pytest.approx(132_000.0)
    assert eur["PositionCount"] == 2
    assert exposure["GrossShare"].sum() == pytest.approx(1.0)


def test_risk_and_zero_cost_optimizer_reduce_currency_volatility():
    rates = build_fx_rate_history(_fx_history(), ["EUR", "JPY", "USD"], base_currency="USD")
    positions = pd.DataFrame(
        [
            {"Asset": "EUR asset", "Currency": "EUR", "MarketValueLocal": 100_000.0},
            {"Asset": "JPY asset", "Currency": "JPY", "MarketValueLocal": 12_000_000.0},
            {"Asset": "USD asset", "Currency": "USD", "MarketValueLocal": 50_000.0},
        ]
    )
    exposure = aggregate_currency_exposure(positions, rates.iloc[-1], base_currency="USD")
    risk = calculate_fx_risk(exposure, rates, confidence=0.95)
    optimized = optimize_currency_hedges(
        exposure,
        rates,
        annual_cost_bps=0.0,
        risk_aversion=1.0,
    )

    assert risk["AnnualizedVolatilityBase"] > 0
    assert risk["HistoricalVaRBase"] >= 0
    assert risk["ExpectedShortfallBase"] >= risk["HistoricalVaRBase"]
    assert risk["Contributions"]["AnnualizedRiskContributionBase"].sum() == pytest.approx(
        risk["AnnualizedVolatilityBase"]
    )
    assert optimized["Plan"]["HedgeRatio"].to_numpy() == pytest.approx([1.0, 1.0], abs=2e-5)
    assert optimized["AfterAnnualVolatilityBase"] == pytest.approx(0.0, abs=1e-4)


def test_hedge_cost_creates_partial_solution_and_stress_uses_residuals():
    rates = build_fx_rate_history(_fx_history(), ["EUR", "USD"], base_currency="USD")
    positions = pd.DataFrame(
        [{"Asset": "EUR asset", "Currency": "EUR", "MarketValueLocal": 100_000.0}]
    )
    exposure = aggregate_currency_exposure(positions, rates.iloc[-1], base_currency="USD")
    optimized = optimize_currency_hedges(
        exposure,
        rates,
        annual_cost_bps=5.0,
        risk_aversion=1.0,
    )
    ratio = float(optimized["Plan"].iloc[0]["HedgeRatio"])
    stress = build_fx_stress_table(exposure, {"EUR": -0.10}, hedge_plan=optimized["Plan"])

    assert 0.0 < ratio < 1.0
    assert abs(stress["HedgedPnLBase"].sum()) < abs(stress["UnhedgedPnLBase"].sum())
    assert optimized["EstimatedAnnualCostBase"] > 0


@pytest.mark.parametrize("shock", [-1.0, np.nan])
def test_stress_rejects_invalid_shocks(shock):
    exposure = pd.DataFrame(
        [{"Currency": "EUR", "NetExposureBase": 100.0, "GrossExposureBase": 100.0, "RateToBase": 1.1, "BaseCurrency": "USD"}]
    )
    with pytest.raises(ValueError, match="Shock for EUR"):
        build_fx_stress_table(exposure, {"EUR": shock})
