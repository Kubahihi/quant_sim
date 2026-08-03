"""Pure fixed-income valuation, risk, and cash-flow helpers.

The competition ledger historically treated every instrument as shares times
price.  That remains correct for equities and bond ETFs.  Individual bonds,
however, are normally quoted as a percentage of par and require accrued
interest, FX conversion, coupon income, and maturity-aware risk measures.

All rates are decimals (``0.05`` means 5%) and bond prices/accrued interest are
quote points per 100 of face value.  Monetary outputs are converted to USD.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import calendar
import math
from typing import Any

import pandas as pd


_INDIVIDUAL_BOND_KINDS = {
    "individual",
    "individual bond",
    "corporate",
    "government",
    "municipal",
    "sovereign",
}
_BOND_SECURITY_TYPES = {"bond", "bonds", "fixed income", "fixed income security"}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(value: Any) -> float | None:
    number = _optional_finite(value)
    if number is None:
        return None
    return number / 100.0 if abs(number) > 1.0 else number


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "callable"}


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def is_bond_security(position: Mapping[str, Any]) -> bool:
    security_type = str(position.get("security_type") or "").strip().casefold()
    return security_type in _BOND_SECURITY_TYPES


def is_individual_bond(position: Mapping[str, Any]) -> bool:
    """Return True only for par-quoted bonds, never for bond ETFs."""
    if not is_bond_security(position):
        return False
    kind = str(position.get("bond_instrument_type") or "").strip().casefold()
    if kind in {"etf", "fund", "bond etf"}:
        return False
    if kind in _INDIVIDUAL_BOND_KINDS:
        return True
    return bool(
        str(position.get("isin") or "").strip()
        or str(position.get("maturity_date") or "").strip()
        or _optional_finite(position.get("face_value")) is not None
    )


def quote_value_usd(
    position: Mapping[str, Any],
    clean_price: Any,
    *,
    accrued_interest: Any = 0.0,
    fx_rate_to_usd: Any = 1.0,
) -> float:
    """Convert a clean price quoted per 100 of par to full dirty USD value."""
    quantity = _finite(position.get("quantity"))
    face_value = _finite(position.get("face_value"), 1000.0)
    if face_value <= 0:
        face_value = 1000.0
    price = _finite(clean_price)
    accrued = _finite(accrued_interest)
    fx = _finite(fx_rate_to_usd, 1.0)
    if fx <= 0:
        fx = 1.0
    return quantity * face_value * (price + accrued) / 100.0 * fx


def position_cost_usd(position: Mapping[str, Any]) -> float:
    if not is_individual_bond(position):
        return _finite(position.get("quantity")) * _finite(position.get("entry_price"))
    return quote_value_usd(
        position,
        position.get("entry_price"),
        accrued_interest=position.get("entry_accrued_interest"),
        fx_rate_to_usd=position.get("entry_fx_rate_to_usd") or position.get("fx_rate_to_usd") or 1.0,
    )


def value_position(
    position: Mapping[str, Any],
    current_price: Any,
    *,
    closed: bool = False,
) -> dict[str, float]:
    """Return cost, value, income, and P/L using the instrument's quote convention."""
    cost = position_cost_usd(position)
    coupon_income = _finite(position.get("coupon_income"))
    if is_individual_bond(position):
        current_value = quote_value_usd(
            position,
            current_price,
            accrued_interest=(
                position.get("exit_accrued_interest")
                if closed
                else position.get("accrued_interest")
            ),
            fx_rate_to_usd=(
                position.get("exit_fx_rate_to_usd")
                if closed
                else position.get("fx_rate_to_usd")
            )
            or 1.0,
        )
    else:
        current_value = _finite(position.get("quantity")) * _finite(current_price)
    pnl = current_value - cost + coupon_income
    return {
        "cost": cost,
        "current_value": current_value,
        "coupon_income": coupon_income,
        "pnl": pnl,
        "return_pct": pnl / cost * 100.0 if cost else 0.0,
    }


def coupon_dates(
    maturity_date: Any,
    settlement_date: Any,
    frequency: Any = 2,
    next_coupon_date: Any = None,
) -> list[date]:
    maturity = _as_date(maturity_date)
    settlement = _as_date(settlement_date) or date.today()
    freq = int(_finite(frequency, 2.0))
    if maturity is None or maturity <= settlement or freq not in {1, 2, 4, 12}:
        return []
    months = 12 // freq
    next_coupon = _as_date(next_coupon_date)
    if next_coupon and settlement < next_coupon <= maturity:
        dates: list[date] = []
        cursor = next_coupon
        while cursor < maturity and len(dates) < 1000:
            dates.append(cursor)
            cursor = _add_months(cursor, months)
        if not dates or dates[-1] != maturity:
            dates.append(maturity)
        return dates

    dates = [maturity]
    cursor = maturity
    while len(dates) < 1000:
        previous = _add_months(cursor, -months)
        if previous <= settlement:
            break
        dates.append(previous)
        cursor = previous
    return sorted(set(dates))


def _cash_flows_per_unit(
    position: Mapping[str, Any],
    settlement: date,
) -> list[tuple[date, float]]:
    maturity = _as_date(position.get("maturity_date"))
    if maturity is None:
        return []
    face_value = _finite(position.get("face_value"), 1000.0)
    if face_value <= 0:
        return []
    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if frequency not in {1, 2, 4, 12}:
        frequency = 2
    coupon_rate_input = _rate(position.get("coupon_rate"))
    coupon_rate = coupon_rate_input or 0.0
    coupon = face_value * coupon_rate / frequency
    dates = coupon_dates(
        maturity,
        settlement,
        frequency,
        position.get("next_coupon_date"),
    )
    flows: list[tuple[date, float]] = []
    for payment_date in dates:
        amount = coupon + (face_value if payment_date == maturity else 0.0)
        flows.append((payment_date, amount))
    return flows


def _present_value(
    flows: Sequence[tuple[date, float]],
    settlement: date,
    annual_yield: float,
    frequency: int,
) -> float:
    base = 1.0 + annual_yield / frequency
    if base <= 0:
        return math.inf
    return sum(
        amount / base ** (frequency * max((payment_date - settlement).days / 365.25, 0.0))
        for payment_date, amount in flows
    )


def solve_ytm(
    position: Mapping[str, Any],
    clean_price: Any,
    *,
    as_of: Any = None,
) -> float | None:
    """Solve nominal annual YTM from the dirty market price by bisection."""
    settlement = _as_date(as_of) or date.today()
    flows = _cash_flows_per_unit(position, settlement)
    face_value = _finite(position.get("face_value"), 1000.0)
    if not flows or face_value <= 0:
        return None
    dirty_quote = _finite(clean_price) + _finite(position.get("accrued_interest"))
    target = face_value * dirty_quote / 100.0
    if target <= 0:
        return None
    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if frequency not in {1, 2, 4, 12}:
        frequency = 2
    low, high = -0.95, 5.0
    low_value = _present_value(flows, settlement, low, frequency) - target
    high_value = _present_value(flows, settlement, high, frequency) - target
    if not math.isfinite(low_value) or low_value * high_value > 0:
        return None
    for _ in range(120):
        mid = (low + high) / 2.0
        difference = _present_value(flows, settlement, mid, frequency) - target
        if abs(difference) < 1e-10:
            return mid
        if difference > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def solve_yield_to_call(
    position: Mapping[str, Any],
    clean_price: Any,
    *,
    as_of: Any = None,
) -> float | None:
    """Solve yield to the first supplied call date and call price."""
    settlement = _as_date(as_of) or date.today()
    call_date = _as_date(position.get("call_date"))
    maturity = _as_date(position.get("maturity_date"))
    if call_date is None or call_date <= settlement or (maturity and call_date >= maturity):
        return None
    call_position = dict(position)
    call_position["maturity_date"] = call_date.isoformat()
    next_coupon = _as_date(position.get("next_coupon_date"))
    if next_coupon and next_coupon > call_date:
        call_position["next_coupon_date"] = None
    flows = _cash_flows_per_unit(call_position, settlement)
    face_value = _finite(position.get("face_value"), 1000.0)
    if not flows or face_value <= 0:
        return None
    call_price = _finite(position.get("call_price"), 100.0)
    final_date, final_amount = flows[-1]
    flows[-1] = (final_date, final_amount - face_value + face_value * call_price / 100.0)
    dirty_quote = _finite(clean_price) + _finite(position.get("accrued_interest"))
    target = face_value * dirty_quote / 100.0
    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if target <= 0 or frequency not in {1, 2, 4, 12}:
        return None
    low, high = -0.95, 5.0
    low_value = _present_value(flows, settlement, low, frequency) - target
    high_value = _present_value(flows, settlement, high, frequency) - target
    if not math.isfinite(low_value) or low_value * high_value > 0:
        return None
    for _ in range(120):
        mid = (low + high) / 2.0
        difference = _present_value(flows, settlement, mid, frequency) - target
        if abs(difference) < 1e-10:
            return mid
        if difference > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def price_from_ytm(
    position: Mapping[str, Any],
    annual_yield: Any,
    *,
    as_of: Any = None,
    clean: bool = True,
) -> float | None:
    """Return a bond quote per 100 of par from a nominal annual yield."""
    settlement = _as_date(as_of) or date.today()
    ytm = _optional_finite(annual_yield)
    flows = _cash_flows_per_unit(position, settlement)
    face_value = _finite(position.get("face_value"), 1000.0)
    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if ytm is None or not flows or face_value <= 0 or frequency not in {1, 2, 4, 12}:
        return None
    dirty_value = _present_value(flows, settlement, ytm, frequency)
    if not math.isfinite(dirty_value):
        return None
    dirty_quote = dirty_value / face_value * 100.0
    return dirty_quote - _finite(position.get("accrued_interest")) if clean else dirty_quote


def price_from_yield_to_call(
    position: Mapping[str, Any],
    annual_yield: Any,
    *,
    as_of: Any = None,
    clean: bool = True,
) -> float | None:
    """Return a quote per 100 assuming redemption at the first call."""
    settlement = _as_date(as_of) or date.today()
    ytc = _optional_finite(annual_yield)
    call_date = _as_date(position.get("call_date"))
    maturity = _as_date(position.get("maturity_date"))
    if ytc is None or call_date is None or call_date <= settlement or (maturity and call_date >= maturity):
        return None
    call_position = dict(position)
    call_position["maturity_date"] = call_date.isoformat()
    if (_as_date(position.get("next_coupon_date")) or call_date) > call_date:
        call_position["next_coupon_date"] = None
    flows = _cash_flows_per_unit(call_position, settlement)
    face_value = _finite(position.get("face_value"), 1000.0)
    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if not flows or face_value <= 0 or frequency not in {1, 2, 4, 12}:
        return None
    call_price = _finite(position.get("call_price"), 100.0)
    final_date, final_amount = flows[-1]
    flows[-1] = (final_date, final_amount - face_value + face_value * call_price / 100.0)
    dirty_value = _present_value(flows, settlement, ytc, frequency)
    if not math.isfinite(dirty_value):
        return None
    dirty_quote = dirty_value / face_value * 100.0
    return dirty_quote - _finite(position.get("accrued_interest")) if clean else dirty_quote


def calculate_bond_metrics(
    position: Mapping[str, Any],
    current_clean_price: Any = None,
    *,
    as_of: Any = None,
) -> dict[str, Any]:
    """Calculate yield, duration, convexity, DV01, and maturity metadata."""
    settlement = _as_date(as_of) or date.today()
    price = _optional_finite(current_clean_price)
    if price is None:
        price = _optional_finite(position.get("last_price"))
    if price is None:
        price = _optional_finite(position.get("entry_price"))
    price = price or 0.0
    maturity = _as_date(position.get("maturity_date"))
    years_to_maturity = (
        max((maturity - settlement).days / 365.25, 0.0) if maturity else None
    )
    face_value = _finite(position.get("face_value"), 1000.0)
    coupon_rate_input = _rate(position.get("coupon_rate"))
    coupon_rate = coupon_rate_input or 0.0
    supplied_ytm = _rate(position.get("yield_to_maturity"))
    ytm = supplied_ytm if supplied_ytm is not None else solve_ytm(position, price, as_of=settlement)
    income_yield_override = _rate(position.get("income_yield"))
    current_yield = (
        coupon_rate * 100.0 / price
        if price > 0 and is_individual_bond(position)
        else income_yield_override
        if income_yield_override is not None
        else coupon_rate * 100.0 / price
        if price > 0 and coupon_rate_input is not None
        else None
    )

    frequency = int(_finite(position.get("coupon_frequency"), 2.0))
    if frequency not in {1, 2, 4, 12}:
        frequency = 2
    flows = _cash_flows_per_unit(position, settlement)
    macaulay: float | None = None
    modified: float | None = None
    convexity: float | None = None
    supplied_duration = _optional_finite(position.get("modified_duration"))
    supplied_convexity = _optional_finite(position.get("convexity"))
    if ytm is not None and flows:
        base = 1.0 + ytm / frequency
        if base > 0:
            present_values: list[tuple[float, float]] = []
            for payment_date, amount in flows:
                years = max((payment_date - settlement).days / 365.25, 0.0)
                pv = amount / base ** (frequency * years)
                present_values.append((years, pv))
            dirty_value = sum(pv for _, pv in present_values)
            if dirty_value > 0:
                macaulay = sum(years * pv for years, pv in present_values) / dirty_value
                modified = macaulay / base
                convexity = (
                    sum(years * (years + 1.0 / frequency) * pv for years, pv in present_values)
                    / dirty_value
                    / (base * base)
                )
    if supplied_duration is not None:
        modified = supplied_duration
    if supplied_convexity is not None:
        convexity = supplied_convexity

    valuation = value_position(position, price)
    market_value = valuation["current_value"]
    dv01 = abs(market_value) * modified * 0.0001 if modified is not None else None
    callable_flag = _truthy(position.get("callable"))
    ytc = solve_yield_to_call(position, price, as_of=settlement) if callable_flag else None
    yield_candidates = [value for value in (ytm, ytc) if value is not None]
    yield_to_worst = min(yield_candidates) if yield_candidates else None
    benchmark_yield = _rate(position.get("benchmark_yield"))
    spread_to_benchmark = (
        yield_to_worst - benchmark_yield
        if yield_to_worst is not None and benchmark_yield is not None
        else None
    )
    default_probability = _rate(position.get("default_probability"))
    recovery_rate = _rate(position.get("recovery_rate"))
    if default_probability is not None:
        default_probability = min(max(default_probability, 0.0), 1.0)
    if recovery_rate is not None:
        recovery_rate = min(max(recovery_rate, 0.0), 1.0)
    expected_loss_rate = (
        default_probability * (1.0 - recovery_rate)
        if default_probability is not None and recovery_rate is not None
        else None
    )
    expected_loss_usd = (
        abs(market_value) * expected_loss_rate if expected_loss_rate is not None else None
    )
    annual_income_usd = (
        _finite(position.get("quantity"))
        * face_value
        * coupon_rate
        * max(_finite(position.get("fx_rate_to_usd"), 1.0), 0.0)
        if is_individual_bond(position)
        else abs(market_value) * current_yield
        if current_yield is not None
        else None
    )
    net_carry_rate = (
        current_yield - expected_loss_rate
        if current_yield is not None and expected_loss_rate is not None
        else current_yield
    )
    breakeven_yield_rise_bps = (
        net_carry_rate / modified * 10_000.0
        if net_carry_rate is not None and modified is not None and modified > 0
        else None
    )
    return {
        "clean_price": price,
        "dirty_price": price + _finite(position.get("accrued_interest")),
        "coupon_rate": coupon_rate if is_individual_bond(position) or coupon_rate_input is not None else None,
        "current_yield": current_yield,
        "yield_to_maturity": ytm,
        "yield_to_call": ytc,
        "yield_to_worst": yield_to_worst,
        "benchmark_yield": benchmark_yield,
        "spread_to_benchmark": spread_to_benchmark,
        "macaulay_duration": macaulay,
        "modified_duration": modified,
        "convexity": convexity,
        "dv01_usd": dv01,
        "maturity_date": maturity.isoformat() if maturity else None,
        "years_to_maturity": years_to_maturity,
        "market_value_usd": market_value,
        "face_value": face_value,
        "annual_income_usd": annual_income_usd,
        "default_probability": default_probability,
        "recovery_rate": recovery_rate,
        "expected_loss_rate": expected_loss_rate,
        "expected_loss_usd": expected_loss_usd,
        "breakeven_yield_rise_bps": breakeven_yield_rise_bps,
    }


def _maturity_bucket(years: float | None) -> str:
    if years is None:
        return "No maturity / ETF"
    if years <= 1:
        return "0-1Y"
    if years <= 3:
        return "1-3Y"
    if years <= 5:
        return "3-5Y"
    if years <= 10:
        return "5-10Y"
    return "10Y+"


def build_fixed_income_analytics(
    positions: Sequence[Mapping[str, Any]] | None,
    performance_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    as_of: Any = None,
) -> dict[str, Any]:
    """Build portfolio-level fixed-income tables and weighted risk metrics."""
    settlement = _as_date(as_of) or date.today()
    performance_by_id = {
        item.get("id"): dict(item)
        for item in performance_rows or []
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    performance_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for item in performance_rows or []:
        if not isinstance(item, Mapping):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if ticker:
            performance_by_ticker.setdefault(ticker, []).append(dict(item))
    overview: list[dict[str, Any]] = []
    cashflows: list[dict[str, Any]] = []
    for raw in positions or []:
        if not isinstance(raw, Mapping) or not is_bond_security(raw):
            continue
        position = dict(raw)
        if str(position.get("status") or "open").strip().lower() == "closed":
            continue
        ticker = str(position.get("ticker") or "").strip().upper()
        performance = performance_by_id.get(position.get("id"))
        if performance is None:
            candidates = performance_by_ticker.get(ticker, [])
            performance = candidates.pop(0) if candidates else {}
        clean_price = performance.get("current_price", position.get("last_price"))
        metrics = calculate_bond_metrics(position, clean_price, as_of=settlement)
        market_value = _finite(performance.get("current_value"), metrics["market_value_usd"])
        instrument_type = "Individual" if is_individual_bond(position) else "ETF"
        overview.append(
            {
                "Ticker": str(position.get("ticker") or "").upper(),
                "ISIN": str(position.get("isin") or ""),
                "Instrument": instrument_type,
                "BondCategory": str(position.get("bond_category") or "Unspecified"),
                "Issuer": str(position.get("issuer") or "Unassigned"),
                "Currency": str(position.get("currency") or "USD").upper(),
                "MarketValueUSD": market_value,
                "CleanPrice": metrics["clean_price"],
                "CouponRate": metrics["coupon_rate"],
                "CurrentYield": metrics["current_yield"],
                "YieldToMaturity": metrics["yield_to_maturity"],
                "YieldToCall": metrics["yield_to_call"],
                "YieldToWorst": metrics["yield_to_worst"],
                "BenchmarkYield": metrics["benchmark_yield"],
                "BenchmarkName": str(position.get("benchmark_name") or ""),
                "SpreadToBenchmark": metrics["spread_to_benchmark"],
                "ModifiedDuration": metrics["modified_duration"],
                "DV01USD": metrics["dv01_usd"],
                "Convexity": metrics["convexity"],
                "MaturityDate": metrics["maturity_date"],
                "YearsToMaturity": metrics["years_to_maturity"],
                "MaturityBucket": _maturity_bucket(metrics["years_to_maturity"]),
                "CreditRating": str(position.get("credit_rating") or "Unrated"),
                "Callable": _truthy(position.get("callable")),
                "CallDate": str(position.get("call_date") or ""),
                "CallPrice": _optional_finite(position.get("call_price")),
                "ExpectedLossUSD": metrics["expected_loss_usd"],
                "BreakevenYieldRiseBps": metrics["breakeven_yield_rise_bps"],
                "Seniority": str(position.get("seniority") or "Unspecified"),
                "PriceSource": str(performance.get("price_source") or "manual/entry"),
                "ValuationSource": str(position.get("valuation_source") or "Unspecified"),
                "SourceReference": str(position.get("source_url") or ""),
                "PriceObservedAt": str(position.get("price_observed_at") or ""),
            }
        )
        if is_individual_bond(position):
            quantity = _finite(position.get("quantity"))
            fx = _finite(position.get("fx_rate_to_usd"), 1.0)
            if fx <= 0:
                fx = 1.0
            maturity = _as_date(position.get("maturity_date"))
            face_value = _finite(position.get("face_value"), 1000.0)
            coupon_rate = _rate(position.get("coupon_rate")) or 0.0
            frequency = int(_finite(position.get("coupon_frequency"), 2.0))
            for payment_date, amount_per_unit in _cash_flows_per_unit(position, settlement):
                principal = face_value if payment_date == maturity else 0.0
                coupon = face_value * coupon_rate / frequency
                cashflows.append(
                    {
                        "Date": payment_date.isoformat(),
                        "Ticker": str(position.get("ticker") or "").upper(),
                        "Currency": str(position.get("currency") or "USD").upper(),
                        "CouponLocal": coupon * quantity,
                        "PrincipalLocal": principal * quantity,
                        "TotalLocal": amount_per_unit * quantity,
                        "TotalUSD": amount_per_unit * quantity * fx,
                    }
                )

    overview_frame = pd.DataFrame(overview)
    cashflow_frame = pd.DataFrame(cashflows)
    if not cashflow_frame.empty:
        cashflow_frame = cashflow_frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    total_value = float(overview_frame["MarketValueUSD"].sum()) if not overview_frame.empty else 0.0

    def weighted(column: str) -> float | None:
        if overview_frame.empty:
            return None
        valid = overview_frame.dropna(subset=[column]).copy()
        denominator = float(valid["MarketValueUSD"].abs().sum())
        if denominator <= 0:
            return None
        return float((valid[column] * valid["MarketValueUSD"].abs()).sum() / denominator)

    def exposure(group: str) -> pd.DataFrame:
        if overview_frame.empty:
            return pd.DataFrame(columns=[group, "MarketValueUSD", "Weight"])
        result = overview_frame.groupby(group, dropna=False)["MarketValueUSD"].sum().reset_index()
        result["Weight"] = result["MarketValueUSD"] / total_value if total_value else 0.0
        return result.sort_values("MarketValueUSD", ascending=False).reset_index(drop=True)

    return {
        "available": not overview_frame.empty,
        "as_of": settlement.isoformat(),
        "position_count": len(overview_frame),
        "individual_bond_count": int((overview_frame.get("Instrument") == "Individual").sum()) if not overview_frame.empty else 0,
        "bond_etf_count": int((overview_frame.get("Instrument") == "ETF").sum()) if not overview_frame.empty else 0,
        "market_value_usd": total_value,
        "weighted_yield_to_maturity": weighted("YieldToMaturity"),
        "weighted_yield_to_worst": weighted("YieldToWorst"),
        "weighted_spread_to_benchmark": weighted("SpreadToBenchmark"),
        "weighted_current_yield": weighted("CurrentYield"),
        "weighted_modified_duration": weighted("ModifiedDuration"),
        "portfolio_dv01_usd": float(overview_frame["DV01USD"].fillna(0.0).sum()) if not overview_frame.empty else 0.0,
        "expected_credit_loss_usd": float(overview_frame["ExpectedLossUSD"].fillna(0.0).sum()) if not overview_frame.empty else 0.0,
        "overview": overview_frame,
        "cashflows": cashflow_frame,
        "maturity_ladder": exposure("MaturityBucket"),
        "currency_exposure": exposure("Currency"),
        "issuer_exposure": exposure("Issuer"),
        "rating_exposure": exposure("CreditRating"),
    }


def build_bond_sensitivity(
    position: Mapping[str, Any],
    current_clean_price: Any = None,
    *,
    as_of: Any = None,
    shocks_bps: Sequence[float] = (-200, -100, -50, 0, 50, 100, 200),
) -> pd.DataFrame:
    """Build exact (individual) or duration-convexity (ETF) yield sensitivity."""
    metrics = calculate_bond_metrics(position, current_clean_price, as_of=as_of)
    current_price = _finite(metrics.get("clean_price"))
    current_value = _finite(metrics.get("market_value_usd"))
    yield_to_call = _optional_finite(metrics.get("yield_to_call"))
    yield_to_worst = _optional_finite(metrics.get("yield_to_worst"))
    worst_is_call = (
        yield_to_call is not None
        and yield_to_worst is not None
        and math.isclose(yield_to_call, yield_to_worst, rel_tol=1e-10, abs_tol=1e-12)
    )
    base_yield = yield_to_worst if worst_is_call else _optional_finite(metrics.get("yield_to_maturity"))
    duration = _optional_finite(metrics.get("modified_duration"))
    convexity = _optional_finite(metrics.get("convexity")) or 0.0
    individual = is_individual_bond(position)
    rows: list[dict[str, Any]] = []
    for shock_bps in shocks_bps:
        shock = _finite(shock_bps) / 10_000.0
        shocked_yield = base_yield + shock if base_yield is not None else None
        shocked_price: float | None = None
        shocked_value: float | None = None
        method = "unavailable"
        if individual and shocked_yield is not None:
            shocked_price = (
                price_from_yield_to_call(position, shocked_yield, as_of=as_of, clean=True)
                if worst_is_call
                else price_from_ytm(position, shocked_yield, as_of=as_of, clean=True)
            )
            if shocked_price is not None:
                shocked_value = quote_value_usd(
                    position,
                    shocked_price,
                    accrued_interest=position.get("accrued_interest"),
                    fx_rate_to_usd=position.get("fx_rate_to_usd") or 1.0,
                )
                method = (
                    "exact cash-flow repricing to call"
                    if worst_is_call
                    else "exact cash-flow repricing"
                )
        elif duration is not None:
            price_change = -duration * shock + 0.5 * convexity * shock * shock
            shocked_price = current_price * (1.0 + price_change)
            shocked_value = current_value * (1.0 + price_change)
            method = "duration + convexity"
        rows.append(
            {
                "ShockBps": _finite(shock_bps),
                "YieldToMaturity": shocked_yield,
                "CleanPrice": shocked_price,
                "MarketValueUSD": shocked_value,
                "PriceChange": (
                    shocked_price / current_price - 1.0
                    if shocked_price is not None and current_price > 0
                    else None
                ),
                "PnLUSD": shocked_value - current_value if shocked_value is not None else None,
                "Method": method,
            }
        )
    return pd.DataFrame(rows)


def build_rate_spread_scenario_grid(
    position: Mapping[str, Any],
    current_clean_price: Any = None,
    *,
    as_of: Any = None,
    curve_shocks_bps: Sequence[float] = (-100, 0, 100, 200),
    spread_shocks_bps: Sequence[float] = (0, 100, 300),
    horizon_years: float = 1.0,
) -> pd.DataFrame:
    """Combine rate/spread price risk, carry, and expected credit loss."""
    horizon = min(max(_finite(horizon_years, 1.0), 0.0), 10.0)
    metrics = calculate_bond_metrics(position, current_clean_price, as_of=as_of)
    current_value = _finite(metrics.get("market_value_usd"))
    annual_income = _optional_finite(metrics.get("annual_income_usd")) or 0.0
    expected_loss_rate = _optional_finite(metrics.get("expected_loss_rate")) or 0.0
    carry_usd = annual_income * horizon
    expected_credit_loss = abs(current_value) * min(expected_loss_rate * horizon, 1.0)
    rows: list[dict[str, Any]] = []
    for curve_shock in curve_shocks_bps:
        for spread_shock in spread_shocks_bps:
            total_shock = _finite(curve_shock) + _finite(spread_shock)
            sensitivity = build_bond_sensitivity(
                position,
                current_clean_price,
                as_of=as_of,
                shocks_bps=[total_shock],
            )
            shocked = sensitivity.iloc[0].to_dict() if not sensitivity.empty else {}
            price_pnl = _optional_finite(shocked.get("PnLUSD"))
            total_pnl = (
                price_pnl + carry_usd - expected_credit_loss
                if price_pnl is not None
                else None
            )
            rows.append(
                {
                    "CurveShockBps": _finite(curve_shock),
                    "SpreadShockBps": _finite(spread_shock),
                    "TotalYieldShockBps": total_shock,
                    "HorizonYears": horizon,
                    "ShockedCleanPrice": shocked.get("CleanPrice"),
                    "PricePnLUSD": price_pnl,
                    "CarryUSD": carry_usd,
                    "ExpectedCreditLossUSD": expected_credit_loss,
                    "ExpectedTotalPnLUSD": total_pnl,
                    "ExpectedReturn": total_pnl / current_value if total_pnl is not None and current_value else None,
                    "RepricingMethod": shocked.get("Method", "unavailable"),
                }
            )
    return pd.DataFrame(rows)


def assess_bond_data_quality(
    position: Mapping[str, Any],
    *,
    as_of: Any = None,
) -> dict[str, Any]:
    """Score analytical input completeness and return actionable issues."""
    settlement = _as_date(as_of) or date.today()
    individual = is_individual_bond(position)
    issues: list[dict[str, str]] = []
    score = 100

    def issue(severity: str, message: str, penalty: int) -> None:
        nonlocal score
        issues.append({"severity": severity, "message": message})
        score -= penalty

    if not str(position.get("ticker") or position.get("isin") or "").strip():
        issue("error", "Identifier is missing.", 25)
    if _finite(position.get("last_price")) <= 0:
        issue("error", "Current price is missing or non-positive.", 25)
    if not str(position.get("valuation_source") or "").strip():
        issue("warning", "Valuation source is not documented.", 8)
    observed = _as_date(position.get("price_observed_at"))
    if observed is None:
        issue("warning", "Price observation date is missing.", 6)
    elif observed > settlement:
        issue("error", "Price observation date is after the analysis date.", 12)
    elif (settlement - observed).days > 14:
        issue("warning", f"Price is {(settlement - observed).days} days old.", 8)
    if not str(position.get("credit_rating") or "").strip():
        issue("info", "Credit rating is not supplied.", 3)
    if _rate(position.get("benchmark_yield")) is None:
        issue("info", "Benchmark yield is missing, so spread cannot be calculated.", 5)
    if _rate(position.get("default_probability")) is None or _rate(position.get("recovery_rate")) is None:
        issue("info", "Default probability or recovery is missing, so expected credit loss is unavailable.", 5)

    if individual:
        if not str(position.get("isin") or "").strip():
            issue("warning", "ISIN is missing.", 5)
        if _finite(position.get("face_value")) <= 0:
            issue("error", "Face value is missing or non-positive.", 20)
        maturity = _as_date(position.get("maturity_date"))
        if maturity is None or maturity <= settlement:
            issue("error", "A future maturity date is required.", 20)
        if _rate(position.get("coupon_rate")) is None:
            issue("warning", "Coupon rate is missing; use zero explicitly for a zero-coupon bond.", 5)
        if int(_finite(position.get("coupon_frequency"))) not in {1, 2, 4, 12}:
            issue("error", "Coupon frequency must be 1, 2, 4, or 12.", 10)
        callable_flag = _truthy(position.get("callable"))
        if callable_flag and _as_date(position.get("call_date")) is None:
            issue("error", "Callable bond is missing its call date.", 15)
    elif _optional_finite(position.get("modified_duration")) is None:
        issue("error", "Bond ETF duration is required for sensitivity analysis.", 20)

    score = max(score, 0)
    status = "ready" if score >= 90 and not any(x["severity"] == "error" for x in issues) else "review" if score >= 65 else "insufficient"
    return {"score": score, "status": status, "issues": issues}


def stress_fixed_income(
    analytics: Mapping[str, Any],
    *,
    curve_shock_bps: float = 0.0,
    credit_spread_shock_bps: float = 0.0,
) -> pd.DataFrame:
    """Estimate price P/L with duration-convexity for parallel curve/spread shocks."""
    overview = analytics.get("overview")
    if not isinstance(overview, pd.DataFrame) or overview.empty:
        return pd.DataFrame(
            columns=["Ticker", "MarketValueUSD", "YieldShockBps", "EstimatedPriceChange", "EstimatedPnLUSD"]
        )
    rows: list[dict[str, Any]] = []
    for item in overview.to_dict("records"):
        duration = _optional_finite(item.get("ModifiedDuration"))
        convexity = _optional_finite(item.get("Convexity")) or 0.0
        value = _finite(item.get("MarketValueUSD"))
        rating = str(item.get("CreditRating") or "Unrated").upper()
        category = str(item.get("BondCategory") or "").strip().casefold()
        government_like = category in {"government", "sovereign"} or rating in {
            "GOVERNMENT", "SOVEREIGN"
        }
        total_bps = float(curve_shock_bps) + (0.0 if government_like else float(credit_spread_shock_bps))
        shock = total_bps / 10_000.0
        price_change = (-duration * shock + 0.5 * convexity * shock * shock) if duration is not None else None
        rows.append(
            {
                "Ticker": item.get("Ticker"),
                "MarketValueUSD": value,
                "YieldShockBps": total_bps,
                "EstimatedPriceChange": price_change,
                "EstimatedPnLUSD": value * price_change if price_change is not None else None,
                "Method": "duration + convexity" if duration is not None else "missing duration",
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "build_fixed_income_analytics",
    "build_bond_sensitivity",
    "build_rate_spread_scenario_grid",
    "calculate_bond_metrics",
    "assess_bond_data_quality",
    "coupon_dates",
    "is_bond_security",
    "is_individual_bond",
    "position_cost_usd",
    "price_from_ytm",
    "price_from_yield_to_call",
    "quote_value_usd",
    "solve_ytm",
    "solve_yield_to_call",
    "stress_fixed_income",
    "value_position",
]
