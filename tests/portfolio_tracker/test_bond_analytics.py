from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio_tracker.bond_analytics import (
    assess_bond_data_quality,
    build_fixed_income_analytics,
    build_bond_sensitivity,
    build_rate_spread_scenario_grid,
    calculate_bond_metrics,
    is_individual_bond,
    position_cost_usd,
    price_from_ytm,
    solve_ytm,
    solve_yield_to_call,
    stress_fixed_income,
    value_position,
)


def _bond(**overrides):
    values = {
        "id": 1,
        "ticker": "US0001",
        "security_type": "Bond",
        "bond_instrument_type": "individual",
        "bond_category": "Corporate",
        "isin": "US0000000001",
        "issuer": "Example Corp",
        "currency": "USD",
        "quantity": 10,
        "face_value": 1_000,
        "entry_price": 98,
        "entry_accrued_interest": 1,
        "last_price": 101,
        "accrued_interest": 0.5,
        "entry_fx_rate_to_usd": 1,
        "fx_rate_to_usd": 1,
        "coupon_income": 250,
        "coupon_rate": 0.05,
        "coupon_frequency": 2,
        "maturity_date": "2030-08-02",
        "next_coupon_date": "2027-02-02",
        "credit_rating": "A",
        "status": "open",
    }
    values.update(overrides)
    return values


def test_individual_bond_uses_par_quote_accrued_interest_fx_and_coupon_income():
    bond = _bond(entry_fx_rate_to_usd=1.1, fx_rate_to_usd=1.2)

    assert is_individual_bond(bond) is True
    assert position_cost_usd(bond) == pytest.approx(10 * 1_000 * 0.99 * 1.1)
    valuation = value_position(bond, 101)
    assert valuation["current_value"] == pytest.approx(10 * 1_000 * 1.015 * 1.2)
    assert valuation["pnl"] == pytest.approx(
        valuation["current_value"] - position_cost_usd(bond) + 250
    )


def test_bond_etf_keeps_share_price_valuation():
    etf = {
        "ticker": "BND",
        "security_type": "Bond",
        "bond_instrument_type": "etf",
        "bond_category": "Government",
        "quantity": 20,
        "entry_price": 70,
    }

    assert is_individual_bond(etf) is False
    assert position_cost_usd(etf) == pytest.approx(1_400)
    assert value_position(etf, 72)["current_value"] == pytest.approx(1_440)


def test_par_bond_ytm_duration_convexity_and_dv01_are_calculated():
    bond = _bond(
        quantity=1,
        entry_price=100,
        entry_accrued_interest=0,
        last_price=100,
        accrued_interest=0,
        coupon_income=0,
    )

    ytm = solve_ytm(bond, 100, as_of="2026-08-02")
    metrics = calculate_bond_metrics(bond, 100, as_of="2026-08-02")

    assert ytm == pytest.approx(0.05, abs=0.001)
    assert metrics["yield_to_maturity"] == pytest.approx(0.05, abs=0.001)
    assert 3 < metrics["modified_duration"] < 4.5
    assert metrics["convexity"] > 0
    assert metrics["dv01_usd"] == pytest.approx(
        metrics["market_value_usd"] * metrics["modified_duration"] * 0.0001
    )
    assert price_from_ytm(bond, 0.05, as_of="2026-08-02") == pytest.approx(100, abs=0.1)

    sensitivity = build_bond_sensitivity(
        bond, 100, as_of="2026-08-02", shocks_bps=[-100, 0, 100]
    ).set_index("ShockBps")
    assert sensitivity.loc[-100, "CleanPrice"] > 100
    assert sensitivity.loc[0, "CleanPrice"] == pytest.approx(100, abs=0.1)
    assert sensitivity.loc[100, "CleanPrice"] < 100
    assert set(sensitivity["Method"]) == {"exact cash-flow repricing"}


def test_portfolio_fixed_income_outputs_cashflows_ladders_and_stress():
    bond = _bond()
    etf = {
        "id": 2,
        "ticker": "BND",
        "security_type": "Bond",
        "bond_instrument_type": "etf",
        "bond_category": "Government",
        "quantity": 100,
        "entry_price": 70,
        "last_price": 72,
        "yield_to_maturity": 0.04,
        "modified_duration": 6,
        "credit_rating": "AAA",
        "status": "open",
    }
    performance = [
        {**bond, "current_price": 101, "current_value": 10_150, "price_source": "manual"},
        {**etf, "current_price": 72, "current_value": 7_200, "price_source": "live"},
    ]

    result = build_fixed_income_analytics([bond, etf], performance, as_of="2026-08-02")

    assert result["position_count"] == 2
    assert result["individual_bond_count"] == 1
    assert result["bond_etf_count"] == 1
    assert result["market_value_usd"] == pytest.approx(17_350)
    assert not result["cashflows"].empty
    assert result["cashflows"].iloc[-1]["PrincipalLocal"] == pytest.approx(10_000)
    assert set(result["maturity_ladder"]["MaturityBucket"]) == {"3-5Y", "No maturity / ETF"}

    stress = stress_fixed_income(result, curve_shock_bps=100, credit_spread_shock_bps=50)
    assert isinstance(stress, pd.DataFrame)
    assert stress.set_index("Ticker").loc["US0001", "YieldShockBps"] == pytest.approx(150)
    assert stress.set_index("Ticker").loc["BND", "YieldShockBps"] == pytest.approx(100)
    assert stress["EstimatedPnLUSD"].sum() < 0


def test_callable_credit_and_data_quality_metrics_support_decision_analysis():
    bond = _bond(
        entry_price=105,
        last_price=105,
        coupon_rate=0.08,
        accrued_interest=0,
        entry_accrued_interest=0,
        callable=1,
        call_date="2028-08-02",
        call_price=100,
        benchmark_name="Treasury",
        benchmark_yield=0.035,
        default_probability=0.01,
        recovery_rate=0.4,
        valuation_source="Primary dealer quote",
        price_observed_at="2026-08-02",
    )

    ytc = solve_yield_to_call(bond, 105, as_of="2026-08-02")
    metrics = calculate_bond_metrics(bond, 105, as_of="2026-08-02")

    assert ytc is not None
    assert metrics["yield_to_call"] == pytest.approx(ytc)
    assert metrics["yield_to_worst"] == min(metrics["yield_to_maturity"], ytc)
    assert metrics["spread_to_benchmark"] == pytest.approx(metrics["yield_to_worst"] - 0.035)
    assert metrics["expected_loss_rate"] == pytest.approx(0.006)
    assert metrics["expected_loss_usd"] == pytest.approx(metrics["market_value_usd"] * 0.006)
    assert metrics["breakeven_yield_rise_bps"] is not None

    callable_sensitivity = build_bond_sensitivity(
        bond, 105, as_of="2026-08-02", shocks_bps=[0, 100]
    ).set_index("ShockBps")
    assert callable_sensitivity.loc[0, "CleanPrice"] == pytest.approx(105, abs=0.1)
    assert callable_sensitivity.loc[100, "CleanPrice"] < callable_sensitivity.loc[0, "CleanPrice"]
    assert set(callable_sensitivity["Method"]) == {"exact cash-flow repricing to call"}

    grid = build_rate_spread_scenario_grid(
        bond,
        105,
        as_of="2026-08-02",
        curve_shocks_bps=[0, 100],
        spread_shocks_bps=[0, 200],
        horizon_years=1,
    )
    assert len(grid) == 4
    assert grid.loc[grid["TotalYieldShockBps"] == 0, "CarryUSD"].iloc[0] > 0
    assert grid.loc[grid["TotalYieldShockBps"] == 300, "ExpectedTotalPnLUSD"].iloc[0] < grid.loc[grid["TotalYieldShockBps"] == 0, "ExpectedTotalPnLUSD"].iloc[0]

    quality = assess_bond_data_quality(bond, as_of="2026-08-02")
    assert quality["score"] == 100
    assert quality["status"] == "ready"
