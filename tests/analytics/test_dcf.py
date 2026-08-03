from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from src.analytics.dcf import (
    build_dcf_sensitivity,
    build_multistage_dcf_scenarios,
    calculate_multistage_dcf,
    calculate_wacc,
    default_multistage_dcf_assumptions,
    prepare_dcf_inputs,
    solve_reverse_dcf,
)


def _prepared_inputs() -> dict:
    return {
        "reported": {
            "cash": 5_000_000_000.0,
            "debt": 2_000_000_000.0,
            "shares_outstanding": 1_000_000_000.0,
            "current_price": 120.0,
        },
        "normalized": {
            "fcff": 10_000_000_000.0,
            "cash_flow_basis": "fundamental_fcff",
        },
        "observed_growth": {
            "revenue_growth": 0.20,
            "earnings_growth": 0.24,
            "quarterly_earnings_growth": 0.18,
        },
        "wacc": {"wacc": 0.09},
        "quality": {"warnings": []},
    }


def test_wacc_uses_adjusted_beta_market_weights_and_tax_shield():
    result = calculate_wacc(
        market_equity=800.0,
        debt=200.0,
        beta=1.2,
        risk_free_rate=0.04,
        equity_risk_premium=0.05,
        pre_tax_cost_of_debt=0.06,
        tax_rate=0.25,
    )

    adjusted_beta = 0.67 * 1.2 + 0.33
    expected = 0.8 * (0.04 + adjusted_beta * 0.05) + 0.2 * 0.06 * (1.0 - 0.25)
    assert result["adjusted_beta"] == pytest.approx(adjusted_beta)
    assert result["wacc"] == pytest.approx(expected)
    assert result["cost_of_debt_source"] == "reported"


def test_prepare_inputs_builds_fcff_from_operating_statements():
    period = pd.Timestamp("2025-12-31")
    snapshot = {
        "ticker": "TEST",
        "info": {
            "marketCap": 800.0,
            "totalDebt": 200.0,
            "totalCash": 50.0,
            "sharesOutstanding": 100.0,
            "currentPrice": 12.0,
            "beta": 1.1,
        },
        "income_statement": pd.DataFrame(
            {period: [1_000.0, 100.0, 20.0, 100.0, -10.0]},
            index=["Total Revenue", "Operating Income", "Tax Provision", "Pretax Income", "Interest Expense"],
        ),
        "cash_flow": pd.DataFrame(
            {period: [10.0, -20.0, -5.0]},
            index=["Depreciation And Amortization", "Capital Expenditure", "Change In Working Capital"],
        ),
    }

    result = prepare_dcf_inputs(snapshot)

    # NOPAT 80 + D&A 10 - capex 20 + cash-flow statement WC effect (-5).
    assert result["normalized"]["fcff"] == pytest.approx(65.0)
    assert result["normalized"]["cash_flow_basis"] == "fundamental_fcff"
    assert result["history"][0]["fundamental_fcff"] == pytest.approx(65.0)
    assert result["reported"]["cash"] == 50.0


def test_prepare_inputs_labels_reported_fcf_proxy_when_components_are_missing():
    result = prepare_dcf_inputs({
        "ticker": "TEST",
        "info": {
            "freeCashflow": 4_000_000_000.0,
            "marketCap": 100_000_000_000.0,
            "sharesOutstanding": 1_000_000_000.0,
        },
    })

    assert result["normalized"]["fcff"] == 4_000_000_000.0
    assert result["normalized"]["cash_flow_basis"] == "reported_fcf_proxy"
    assert any("FCFF proxy" in warning for warning in result["quality"]["warnings"])


def test_prepare_inputs_preserves_a_genuine_zero_observed_tax_rate():
    period = pd.Timestamp("2025-12-31")
    result = prepare_dcf_inputs({
        "info": {
            "marketCap": 800.0,
            "totalDebt": 200.0,
            "sharesOutstanding": 100.0,
            "beta": 1.0,
        },
        "income_statement": pd.DataFrame(
            {period: [1_000.0, 100.0, 0.0, 100.0]},
            index=["Total Revenue", "Operating Income", "Tax Provision", "Pretax Income"],
        ),
        "cash_flow": pd.DataFrame(
            {period: [10.0, -20.0, -5.0]},
            index=["Depreciation And Amortization", "Capital Expenditure", "Change In Working Capital"],
        ),
    })

    assert result["wacc"]["tax_rate"] == 0.0
    assert result["normalized"]["fcff"] == pytest.approx(85.0)


def test_multistage_dcf_accepts_zero_growth_and_fades_continuously():
    inputs = _prepared_inputs()
    assumptions = {
        **default_multistage_dcf_assumptions(inputs),
        "initial_growth_rate": 0.0,
        "near_term_years": 2,
        "fade_years": 3,
        "terminal_growth_rate": 0.02,
        "discount_rate": 0.09,
    }

    result = calculate_multistage_dcf(inputs, assumptions)

    assert result["available"] is True
    assert len(result["projected"]) == 5
    assert result["projected"][0]["phase"] == "near_term"
    assert result["projected"][-1]["growth_rate"] == pytest.approx(0.02)
    assert result["diagnostics"]["continuity_gap"] == pytest.approx(0.0)
    assert result["equity_value"] == pytest.approx(
        result["enterprise_value"] + assumptions["cash"] - assumptions["debt"]
    )


def test_scenarios_are_ordered_and_do_not_mutate_manual_assumptions():
    inputs = _prepared_inputs()
    assumptions = {
        **default_multistage_dcf_assumptions(inputs),
        "initial_growth_rate": 0.18,
    }
    original = deepcopy(assumptions)

    scenarios = build_multistage_dcf_scenarios(inputs, assumptions)

    assert assumptions == original
    assert all(result["available"] for result in scenarios.values())
    assert scenarios["Bear"]["fair_value_per_share"] < scenarios["Base"]["fair_value_per_share"]
    assert scenarios["Base"]["fair_value_per_share"] < scenarios["Bull"]["fair_value_per_share"]


def test_bull_scenario_keeps_a_valid_terminal_spread_at_low_wacc():
    inputs = _prepared_inputs()
    assumptions = {
        **default_multistage_dcf_assumptions(inputs),
        "discount_rate": 0.06,
        "terminal_growth_rate": 0.035,
    }

    scenarios = build_multistage_dcf_scenarios(inputs, assumptions)

    assert all(result["available"] for result in scenarios.values())
    bull = scenarios["Bull"]["assumptions"]
    assert bull["discount_rate"] - bull["terminal_growth_rate"] >= 0.02 - 1e-12


def test_sensitivity_center_matches_base_and_value_falls_as_wacc_rises():
    inputs = _prepared_inputs()
    assumptions = default_multistage_dcf_assumptions(inputs)
    sensitivity = build_dcf_sensitivity(inputs, assumptions)
    center = calculate_multistage_dcf(inputs, assumptions)

    assert sensitivity["center"]["fair_value_per_share"] == pytest.approx(center["fair_value_per_share"])
    base_terminal_row = sensitivity["values"][2]
    assert all(value is not None for value in base_terminal_row)
    assert base_terminal_row == sorted(base_terminal_row, reverse=True)


def test_reverse_dcf_recovers_growth_used_to_create_target_price():
    inputs = _prepared_inputs()
    assumptions = default_multistage_dcf_assumptions(inputs)
    target_growth = 0.23
    target = calculate_multistage_dcf(
        inputs,
        {**assumptions, "initial_growth_rate": target_growth},
    )["fair_value_per_share"]

    reverse = solve_reverse_dcf(inputs, assumptions, target_price=target)

    assert reverse["available"] is True
    assert reverse["implied_initial_growth_rate"] == pytest.approx(target_growth, abs=1e-8)


def test_invalid_terminal_spread_and_zero_shares_are_rejected():
    inputs = _prepared_inputs()
    assumptions = default_multistage_dcf_assumptions(inputs)

    result = calculate_multistage_dcf(inputs, {
        **assumptions,
        "discount_rate": 0.05,
        "terminal_growth_rate": 0.04,
        "shares_outstanding": 0.0,
    })

    assert result["available"] is False
    assert "Shares outstanding must be positive" in result["error"]
    assert "at least 2 percentage points" in result["error"]
