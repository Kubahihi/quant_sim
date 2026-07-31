"""Auditable multi-stage discounted cash-flow valuation.

The v2 engine deliberately keeps reported values, normalized cash flow and
forecast judgments separate.  It is vendor-agnostic: callers pass the company
snapshot already used by the research UI.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DCF_SCHEMA_VERSION = 2
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_EQUITY_RISK_PREMIUM = 0.05
DEFAULT_TAX_RATE = 0.21


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _normalise_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _info_from_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    info = snapshot.get("info")
    return info if isinstance(info, Mapping) else snapshot


def _statement_series(frame: Any, *names: str) -> dict[str, float]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    lookup = {_normalise_label(index): index for index in frame.index}
    selected = next((lookup[_normalise_label(name)] for name in names if _normalise_label(name) in lookup), None)
    if selected is None:
        return {}
    output: dict[str, float] = {}
    row = frame.loc[selected]
    for column, raw in row.items():
        value = _finite(raw)
        if value is None:
            continue
        try:
            label = pd.Timestamp(column).date().isoformat()
        except Exception:
            label = str(column)
        output[label] = float(value)
    return output


def _series_value(series: Mapping[str, float], period: str) -> float | None:
    return _finite(series.get(period))


def _robust_weighted_average(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    if len(clean) >= 3:
        median = float(np.median(clean))
        mad = float(np.median(np.abs(np.asarray(clean) - median)))
        if mad > 0:
            lower, upper = median - 3.0 * mad, median + 3.0 * mad
            clean = [_clip(value, lower, upper) for value in clean]
    weights = np.asarray([0.60**index for index in range(len(clean))], dtype=float)
    weights /= weights.sum()
    return float(np.dot(np.asarray(clean, dtype=float), weights))


def calculate_wacc(
    *,
    market_equity: float,
    debt: float,
    beta: float | None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    pre_tax_cost_of_debt: float | None = None,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> dict[str, Any]:
    """Calculate market-value weighted WACC with source-labelled fallbacks."""
    warnings: list[str] = []
    raw_beta = _finite(beta)
    beta_source = "reported"
    if raw_beta is None or raw_beta <= 0:
        raw_beta, beta_source = 1.0, "fallback"
        warnings.append("Beta was unavailable; a market beta of 1.0 was used.")
    # Bloomberg-style mean reversion prevents one noisy historical beta from
    # dominating the entire valuation while retaining company-specific risk.
    adjusted_beta = _clip((0.67 * raw_beta) + 0.33, 0.50, 2.00)
    rf = _clip(float(risk_free_rate), 0.0, 0.15)
    erp = _clip(float(equity_risk_premium), 0.02, 0.10)
    cost_of_equity = rf + adjusted_beta * erp

    debt_value = max(float(_finite(debt, 0.0) or 0.0), 0.0)
    equity_value = max(float(_finite(market_equity, 0.0) or 0.0), 0.0)
    cost_of_debt = _finite(pre_tax_cost_of_debt)
    debt_cost_source = "reported"
    if cost_of_debt is None or cost_of_debt <= 0:
        leverage = debt_value / (debt_value + equity_value) if debt_value + equity_value > 0 else 0.0
        cost_of_debt = rf + 0.015 + 0.025 * leverage
        debt_cost_source = "fallback"
        warnings.append("Cost of debt was unavailable; a deterministic risk-free-plus-spread proxy was used.")
    cost_of_debt = _clip(cost_of_debt, 0.01, 0.25)
    tax = _clip(float(tax_rate), 0.0, 0.40)
    capital = equity_value + debt_value
    if capital <= 0:
        equity_weight, debt_weight = 1.0, 0.0
        warnings.append("Capital structure was unavailable; WACC uses 100% equity weight.")
    else:
        equity_weight, debt_weight = equity_value / capital, debt_value / capital
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1.0 - tax)
    wacc = _clip(wacc, 0.05, 0.25)
    return {
        "wacc": wacc,
        "risk_free_rate": rf,
        "equity_risk_premium": erp,
        "raw_beta": raw_beta,
        "adjusted_beta": adjusted_beta,
        "beta_source": beta_source,
        "cost_of_equity": cost_of_equity,
        "pre_tax_cost_of_debt": cost_of_debt,
        "cost_of_debt_source": debt_cost_source,
        "tax_rate": tax,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "warnings": warnings,
    }


def prepare_dcf_inputs(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize reported statements into an auditable FCFF starting point."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("DCF snapshot must be a mapping.")
    info = _info_from_snapshot(snapshot)
    cash_flow = snapshot.get("cash_flow")
    income = snapshot.get("income_statement")
    warnings: list[str] = []

    revenue_series = _statement_series(income, "Total Revenue", "Operating Revenue", "Revenue")
    ebit_series = _statement_series(income, "Operating Income", "EBIT")
    tax_series = _statement_series(income, "Tax Provision", "Income Tax Expense")
    pretax_series = _statement_series(income, "Pretax Income", "Income Before Tax")
    interest_series = _statement_series(income, "Interest Expense", "Interest Expense Non Operating")
    depreciation_series = _statement_series(
        cash_flow, "Depreciation And Amortization", "Depreciation Amortization Depletion"
    )
    capex_series = _statement_series(cash_flow, "Capital Expenditure", "Capital Expenditures")
    working_capital_series = _statement_series(cash_flow, "Change In Working Capital", "Changes In Cash Working Capital")
    reported_fcf_series = _statement_series(cash_flow, "Free Cash Flow")

    periods = sorted(
        set(revenue_series) | set(ebit_series) | set(reported_fcf_series),
        reverse=True,
    )
    history: list[dict[str, Any]] = []
    fundamental_fcff_values: list[float] = []
    bridged_fcff_values: list[float] = []
    tax_rates: list[float] = []
    for period in periods[:5]:
        revenue = _series_value(revenue_series, period)
        ebit = _series_value(ebit_series, period)
        tax_expense = _series_value(tax_series, period)
        pretax_income = _series_value(pretax_series, period)
        tax_rate = None
        if tax_expense is not None and pretax_income is not None and pretax_income > 0:
            tax_rate = _clip(tax_expense / pretax_income, 0.0, 0.40)
            tax_rates.append(tax_rate)
        tax_rate = tax_rate if tax_rate is not None else DEFAULT_TAX_RATE
        depreciation = _series_value(depreciation_series, period)
        capex_raw = _series_value(capex_series, period)
        capex = abs(capex_raw) if capex_raw is not None else None
        change_wc = _series_value(working_capital_series, period)
        interest_raw = _series_value(interest_series, period)
        interest = abs(interest_raw) if interest_raw is not None else None
        reported_fcf = _series_value(reported_fcf_series, period)
        fundamental_fcff = None
        if None not in (ebit, depreciation, capex, change_wc):
            fundamental_fcff = ebit * (1.0 - tax_rate) + depreciation - capex + change_wc
            fundamental_fcff_values.append(float(fundamental_fcff))
        bridged_fcff = None
        if reported_fcf is not None and interest is not None:
            bridged_fcff = reported_fcf + interest * (1.0 - tax_rate)
            bridged_fcff_values.append(float(bridged_fcff))
        history.append({
            "period": period,
            "revenue": revenue,
            "ebit": ebit,
            "tax_rate": tax_rate,
            "depreciation": depreciation,
            "capex": capex,
            "change_working_capital": change_wc,
            "interest_expense": interest,
            "reported_fcf": reported_fcf,
            "fundamental_fcff": fundamental_fcff,
            "bridged_fcff": bridged_fcff,
        })

    normalized_tax_value = _robust_weighted_average(tax_rates)
    normalized_tax = DEFAULT_TAX_RATE if normalized_tax_value is None else normalized_tax_value
    reported_fcf_ttm = _finite(info.get("freeCashflow"))
    interest_ttm = _finite(info.get("interestExpense"))
    ttm_fcff_proxy = None
    if reported_fcf_ttm is not None and interest_ttm is not None:
        ttm_fcff_proxy = reported_fcf_ttm + abs(interest_ttm) * (1.0 - normalized_tax)

    if fundamental_fcff_values:
        historical_normalized = _robust_weighted_average(fundamental_fcff_values)
        cash_flow_basis = "fundamental_fcff"
        starting_fcff = historical_normalized
        if ttm_fcff_proxy is not None and historical_normalized is not None:
            starting_fcff = 0.70 * ttm_fcff_proxy + 0.30 * historical_normalized
        method = "recency_weighted_fundamental_fcff"
    elif bridged_fcff_values or ttm_fcff_proxy is not None:
        candidates = ([ttm_fcff_proxy] if ttm_fcff_proxy is not None else []) + bridged_fcff_values
        starting_fcff = _robust_weighted_average(candidates)
        cash_flow_basis = "fcf_plus_after_tax_interest"
        method = "reported_fcf_interest_bridge"
        warnings.append("FCFF uses a bridge from reported FCF plus after-tax interest because full components were unavailable.")
    else:
        starting_fcff = reported_fcf_ttm
        if starting_fcff is None:
            starting_fcff = _robust_weighted_average(list(reported_fcf_series.values()))
        cash_flow_basis = "reported_fcf_proxy"
        method = "reported_fcf_proxy"
        warnings.append("Reported FCF is used as an FCFF proxy; confirm its cash-flow basis before relying on the valuation.")

    revenue = _finite(info.get("totalRevenue"))
    if revenue is None and revenue_series:
        revenue = next(iter(revenue_series.values()))
    market_cap = max(float(_finite(info.get("marketCap"), 0.0) or 0.0), 0.0)
    debt = max(float(_finite(info.get("totalDebt"), 0.0) or 0.0), 0.0)
    pre_tax_cost_of_debt = None
    latest_interest = abs(next(iter(interest_series.values()))) if interest_series else None
    if latest_interest is not None and debt > 0:
        pre_tax_cost_of_debt = latest_interest / debt
    wacc = calculate_wacc(
        market_equity=market_cap,
        debt=debt,
        beta=_finite(info.get("beta")),
        pre_tax_cost_of_debt=pre_tax_cost_of_debt,
        tax_rate=normalized_tax,
    )
    warnings.extend(wacc["warnings"])
    starting_fcff_value = float(starting_fcff or 0.0)
    if starting_fcff_value <= 0:
        warnings.append("Normalized FCFF is unavailable or non-positive; a cash-flow-growth DCF cannot be calculated reliably.")
    return {
        "schema_version": DCF_SCHEMA_VERSION,
        "ticker": str(snapshot.get("ticker") or info.get("symbol") or "").upper(),
        "valuation_date": datetime.now(timezone.utc).date().isoformat(),
        "reported": {
            "revenue": float(revenue or 0.0),
            "free_cash_flow": float(reported_fcf_ttm or 0.0),
            "cash": max(float(_finite(info.get("totalCash"), 0.0) or 0.0), 0.0),
            "debt": debt,
            "shares_outstanding": max(float(_finite(info.get("sharesOutstanding"), 0.0) or 0.0), 0.0),
            "current_price": max(float(_finite(info.get("currentPrice") or info.get("regularMarketPrice"), 0.0) or 0.0), 0.0),
            "market_cap": market_cap,
        },
        "normalized": {
            "fcff": starting_fcff_value,
            "method": method,
            "cash_flow_basis": cash_flow_basis,
            "history_years": max(len(fundamental_fcff_values), len(bridged_fcff_values), len(reported_fcf_series)),
        },
        "history": history,
        "wacc": wacc,
        "observed_growth": {
            "revenue_growth": _finite(info.get("revenueGrowth")),
            "earnings_growth": _finite(info.get("earningsGrowth")),
            "quarterly_earnings_growth": _finite(info.get("earningsQuarterlyGrowth")),
        },
        "quality": {
            "warnings": list(dict.fromkeys(warnings)),
            "statement_periods": len(history),
            "cash_flow_basis": cash_flow_basis,
        },
    }


def default_multistage_dcf_assumptions(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Create reproducible lifecycle-aware assumptions from prepared inputs."""
    observed = inputs.get("observed_growth", {}) if isinstance(inputs.get("observed_growth"), Mapping) else {}
    candidates = [
        _finite(observed.get("revenue_growth")),
        _finite(observed.get("earnings_growth")),
        _finite(observed.get("quarterly_earnings_growth")),
    ]
    growth_values = [value for value in candidates if value is not None and -0.50 < value < 2.0]
    observed_growth = float(np.median(growth_values)) if growth_values else 0.05
    # Shrink volatile point-in-time growth toward a sustainable anchor.
    initial_growth = _clip((0.70 * observed_growth) + 0.30 * 0.05, -0.15, 0.35)
    if initial_growth >= 0.18:
        lifecycle, near_term_years, fade_years = "high_growth", 4, 6
    elif initial_growth >= 0.08:
        lifecycle, near_term_years, fade_years = "transition", 3, 5
    elif initial_growth >= 0.0:
        lifecycle, near_term_years, fade_years = "mature", 3, 3
    else:
        lifecycle, near_term_years, fade_years = "contracting", 2, 4
    discount_rate = float(inputs.get("wacc", {}).get("wacc", 0.10))
    terminal_growth = _clip(0.02 + max(initial_growth, 0.0) * 0.025, 0.01, 0.03)
    terminal_growth = min(terminal_growth, discount_rate - 0.025)
    reported = inputs.get("reported", {}) if isinstance(inputs.get("reported"), Mapping) else {}
    normalized = inputs.get("normalized", {}) if isinstance(inputs.get("normalized"), Mapping) else {}
    return {
        "schema_version": DCF_SCHEMA_VERSION,
        "model_kind": "fcff_growth_fade",
        "lifecycle": lifecycle,
        "starting_fcff": float(normalized.get("fcff") or 0.0),
        "initial_growth_rate": initial_growth,
        "near_term_years": near_term_years,
        "fade_years": fade_years,
        "discount_rate": discount_rate,
        "terminal_growth_rate": terminal_growth,
        "midyear_convention": True,
        "cash": float(reported.get("cash") or 0.0),
        "debt": float(reported.get("debt") or 0.0),
        "shares_outstanding": float(reported.get("shares_outstanding") or 0.0),
        "current_price": float(reported.get("current_price") or 0.0),
    }


def _validate_assumptions(assumptions: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    finite_keys = ("starting_fcff", "initial_growth_rate", "discount_rate", "terminal_growth_rate", "shares_outstanding")
    for key in finite_keys:
        if _finite(assumptions.get(key)) is None:
            errors.append(f"{key} must be finite.")
    if float(_finite(assumptions.get("starting_fcff"), 0.0) or 0.0) <= 0:
        errors.append("Normalized starting FCFF must be positive.")
    if float(_finite(assumptions.get("shares_outstanding"), 0.0) or 0.0) <= 0:
        errors.append("Shares outstanding must be positive.")
    growth_value = _finite(assumptions.get("initial_growth_rate"), -1.0)
    growth = float(-1.0 if growth_value is None else growth_value)
    if growth <= -1.0 or growth > 1.0:
        errors.append("Initial FCFF growth must be above -100% and no greater than 100%.")
    discount_value = _finite(assumptions.get("discount_rate"), 0.0)
    terminal_value = _finite(assumptions.get("terminal_growth_rate"), 0.0)
    discount = float(0.0 if discount_value is None else discount_value)
    terminal = float(0.0 if terminal_value is None else terminal_value)
    if discount <= 0.0 or discount > 0.50:
        errors.append("WACC must be above 0% and no greater than 50%.")
    if terminal <= -1.0:
        errors.append("Terminal growth must be above -100%.")
    if discount - terminal < 0.02 - 1e-12:
        errors.append("WACC must exceed terminal growth by at least 2 percentage points.")
    try:
        near, fade = int(assumptions.get("near_term_years", 0)), int(assumptions.get("fade_years", 0))
    except (TypeError, ValueError):
        near = fade = 0
    if near < 1 or near > 10 or fade < 1 or fade > 15 or near + fade > 20:
        errors.append("Forecast stages must total 2 to 20 years with at least one year per stage.")
    return errors


def calculate_multistage_dcf(
    inputs: Mapping[str, Any],
    assumptions: Mapping[str, Any],
) -> dict[str, Any]:
    """Value FCFF through a near-term stage and a smooth competitive fade."""
    base = {**default_multistage_dcf_assumptions(inputs), **dict(assumptions)}
    errors = _validate_assumptions(base)
    if errors:
        return {"available": False, "schema_version": DCF_SCHEMA_VERSION, "error": " ".join(errors), "errors": errors}
    starting_fcff = float(base["starting_fcff"])
    initial_growth = float(base["initial_growth_rate"])
    near_years, fade_years = int(base["near_term_years"]), int(base["fade_years"])
    wacc, terminal_growth = float(base["discount_rate"]), float(base["terminal_growth_rate"])
    midyear = bool(base.get("midyear_convention", True))
    growth_path = [initial_growth] * near_years + [
        initial_growth + (terminal_growth - initial_growth) * (step / fade_years)
        for step in range(1, fade_years + 1)
    ]
    projected: list[dict[str, Any]] = []
    fcff = starting_fcff
    pv_explicit = 0.0
    for year, growth in enumerate(growth_path, start=1):
        fcff *= 1.0 + growth
        exponent = year - 0.5 if midyear else year
        discount_factor = 1.0 / ((1.0 + wacc) ** exponent)
        present_value = fcff * discount_factor
        pv_explicit += present_value
        projected.append({
            "year": year,
            "phase": "near_term" if year <= near_years else "fade",
            "growth_rate": growth,
            "free_cash_flow": fcff,
            "discount_rate": wacc,
            "discount_factor": discount_factor,
            "present_value": present_value,
        })
    terminal_fcff = fcff * (1.0 + terminal_growth)
    terminal_value = terminal_fcff / (wacc - terminal_growth)
    terminal_present_value = terminal_value / ((1.0 + wacc) ** len(growth_path))
    enterprise_value = pv_explicit + terminal_present_value
    equity_value = enterprise_value + float(base.get("cash", 0.0)) - float(base.get("debt", 0.0))
    shares = float(base["shares_outstanding"])
    fair_value = equity_value / shares
    current_price = float(base.get("current_price", 0.0))
    upside = fair_value / current_price - 1.0 if current_price > 0 else None
    terminal_share = terminal_present_value / enterprise_value if enterprise_value else 0.0
    warnings = list(inputs.get("quality", {}).get("warnings", []))
    if terminal_share > 0.75:
        warnings.append("More than 75% of enterprise value comes from terminal value.")
    if fair_value < 0:
        warnings.append("The equity bridge produces a negative equity value under these assumptions.")
    return {
        "available": True,
        "schema_version": DCF_SCHEMA_VERSION,
        "model_kind": "fcff_growth_fade",
        "projected": projected,
        "pv_explicit": pv_explicit,
        "terminal_fcff": terminal_fcff,
        "terminal_value": terminal_value,
        "terminal_present_value": terminal_present_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value,
        "current_price": current_price,
        "upside_pct": upside,
        "terminal_value_share": terminal_share,
        "terminal_fcf_multiple": terminal_value / fcff if fcff else None,
        "bridge": {
            "enterprise_value": enterprise_value,
            "cash": float(base.get("cash", 0.0)),
            "debt": float(base.get("debt", 0.0)),
            "equity_value": equity_value,
            "shares_outstanding": shares,
        },
        "assumptions": base,
        "diagnostics": {
            "warnings": list(dict.fromkeys(warnings)),
            "cash_flow_basis": inputs.get("normalized", {}).get("cash_flow_basis"),
            "last_explicit_growth": growth_path[-1],
            "terminal_growth_rate": terminal_growth,
            "continuity_gap": abs(growth_path[-1] - terminal_growth),
        },
    }


def build_multistage_dcf_scenarios(
    inputs: Mapping[str, Any],
    assumptions: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    base = {**default_multistage_dcf_assumptions(inputs), **dict(assumptions or {})}
    growth = float(base["initial_growth_rate"])
    shock = max(0.03, abs(growth) * 0.20)
    bull_wacc = max(0.05, float(base["discount_rate"]) - 0.010)
    bull_terminal = min(
        0.04,
        float(base["terminal_growth_rate"]) + 0.0025,
        bull_wacc - 0.020,
    )
    scenarios = {
        "Bear": {
            "starting_fcff": float(base["starting_fcff"]) * 0.95,
            "initial_growth_rate": max(-0.30, growth - shock),
            "near_term_years": max(1, int(base["near_term_years"]) - 1),
            "discount_rate": min(0.30, float(base["discount_rate"]) + 0.015),
            "terminal_growth_rate": max(-0.01, float(base["terminal_growth_rate"]) - 0.005),
        },
        "Base": {},
        "Bull": {
            "starting_fcff": float(base["starting_fcff"]) * 1.05,
            "initial_growth_rate": min(0.80, growth + shock),
            "near_term_years": min(10, int(base["near_term_years"]) + 1),
            "discount_rate": bull_wacc,
            "terminal_growth_rate": bull_terminal,
        },
    }
    return {
        name: calculate_multistage_dcf(inputs, {**deepcopy(base), **override})
        for name, override in scenarios.items()
    }


def build_dcf_sensitivity(
    inputs: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    wacc_offsets: Sequence[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
    terminal_growth_offsets: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> dict[str, Any]:
    base = {**default_multistage_dcf_assumptions(inputs), **dict(assumptions)}
    wacc_values = [float(base["discount_rate"]) + float(offset) for offset in wacc_offsets]
    terminal_values = [float(base["terminal_growth_rate"]) + float(offset) for offset in terminal_growth_offsets]
    grid: list[list[float | None]] = []
    for terminal_growth in terminal_values:
        row: list[float | None] = []
        for wacc in wacc_values:
            result = calculate_multistage_dcf(
                inputs,
                {**base, "discount_rate": wacc, "terminal_growth_rate": terminal_growth},
            )
            row.append(float(result["fair_value_per_share"]) if result.get("available") else None)
        grid.append(row)
    return {
        "wacc_values": wacc_values,
        "terminal_growth_values": terminal_values,
        "values": grid,
        "center": calculate_multistage_dcf(inputs, base),
    }


def solve_reverse_dcf(
    inputs: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    target_price: float | None = None,
    bounds: tuple[float, float] = (-0.20, 0.80),
) -> dict[str, Any]:
    """Solve the near-term FCFF growth implied by a target share price."""
    base = {**default_multistage_dcf_assumptions(inputs), **dict(assumptions)}
    target = float(target_price if target_price is not None else base.get("current_price", 0.0))
    if target <= 0:
        return {"available": False, "error": "A positive target price is required."}
    low, high = bounds
    low_result = calculate_multistage_dcf(inputs, {**base, "initial_growth_rate": low})
    high_result = calculate_multistage_dcf(inputs, {**base, "initial_growth_rate": high})
    if not low_result.get("available") or not high_result.get("available"):
        return {"available": False, "error": "Reverse DCF bounds could not be valued."}
    low_price, high_price = float(low_result["fair_value_per_share"]), float(high_result["fair_value_per_share"])
    if target < low_price or target > high_price:
        return {
            "available": False,
            "error": "Target price lies outside the configured reverse-DCF growth bounds.",
            "price_bounds": [low_price, high_price],
            "growth_bounds": [low, high],
        }
    for _ in range(80):
        midpoint = (low + high) / 2.0
        result = calculate_multistage_dcf(inputs, {**base, "initial_growth_rate": midpoint})
        if float(result["fair_value_per_share"]) < target:
            low = midpoint
        else:
            high = midpoint
    implied = (low + high) / 2.0
    return {
        "available": True,
        "target_price": target,
        "implied_initial_growth_rate": implied,
        "base_initial_growth_rate": float(base["initial_growth_rate"]),
        "growth_gap": implied - float(base["initial_growth_rate"]),
    }


__all__ = [
    "DCF_SCHEMA_VERSION",
    "build_dcf_sensitivity",
    "build_multistage_dcf_scenarios",
    "calculate_multistage_dcf",
    "calculate_wacc",
    "default_multistage_dcf_assumptions",
    "prepare_dcf_inputs",
    "solve_reverse_dcf",
]
