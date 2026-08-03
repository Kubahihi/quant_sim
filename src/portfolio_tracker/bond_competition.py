"""Competition-oriented fixed-income decision support.

The helpers in this module turn bond analytics into an auditable client case.
They deliberately separate analytical completeness from competition eligibility:
an instrument is eligible only when the team records a current rule source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import math
from typing import Any

import pandas as pd

from .bond_analytics import assess_bond_data_quality, calculate_bond_metrics


ELIGIBILITY_PENDING = "Pending verification"
ELIGIBILITY_VERIFIED = "Verified eligible"
ELIGIBILITY_INELIGIBLE = "Verified ineligible"


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _pct(value: Any) -> str:
    number = _finite(value)
    return f"{number:.2%}" if number is not None else "Not available"


def _money(value: Any) -> str:
    number = _finite(value)
    return f"${number:,.2f}" if number is not None else "Not available"


def _number(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    return f"{number:,.{digits}f}" if number is not None else "Not available"


def assess_bond_competition_case(
    position: Mapping[str, Any],
    metrics: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    competition_case: Mapping[str, Any] | None = None,
    *,
    scenario_grid: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assess evidence and client fit without using performance as a score input."""
    case = dict(competition_case or {})
    eligibility = str(
        case.get("eligibility_status")
        or position.get("competition_eligibility_status")
        or ELIGIBILITY_PENDING
    ).strip()
    eligibility_source = str(
        case.get("eligibility_source") or position.get("eligibility_source") or ""
    ).strip()
    eligibility_date = _as_date(
        case.get("eligibility_checked_at") or position.get("eligibility_checked_at")
    )
    proposed_weight = _finite(case.get("proposed_weight"))
    max_weight = _finite(case.get("max_position_weight"))
    category = str(position.get("bond_category") or "").strip().casefold()
    is_credit = category in {"corporate", "municipal", "sovereign"}
    callable_flag = _truthy(position.get("callable"))
    quality_score = int(_finite(data_quality.get("score")) or 0)
    quality_errors = [
        item for item in data_quality.get("issues", [])
        if isinstance(item, Mapping) and str(item.get("severity") or "").lower() == "error"
    ]
    has_scenario = isinstance(scenario_grid, pd.DataFrame) and not scenario_grid.empty

    checks = [
        ("Client fit", "Client goal is explicit", 10, _present(case.get("client_goal"))),
        ("Client fit", "Portfolio role is explicit", 8, _present(case.get("portfolio_role"))),
        ("Thesis", "Core bond thesis is documented", 10, _present(case.get("thesis"))),
        ("Thesis", "Why-now rationale is documented", 6, _present(case.get("why_now"))),
        ("Thesis", "Observable invalidation condition is documented", 7, _present(case.get("invalidation"))),
        ("Thesis", "Sell / review discipline is documented", 5, _present(case.get("sell_discipline"))),
        ("Evidence", "Valuation source or URL is recorded", 8, _present(position.get("valuation_source")) or _present(position.get("source_url"))),
        ("Evidence", "Input quality is at least 75/100", 6, quality_score >= 75),
        ("Evidence", "Competition eligibility is verified with a source and date", 12, eligibility == ELIGIBILITY_VERIFIED and bool(eligibility_source) and eligibility_date is not None),
        ("Risk", "Rate risk is quantified", 6, _finite(metrics.get("modified_duration")) is not None and _finite(metrics.get("dv01_usd")) is not None),
        ("Risk", "Curve/spread downside scenario is quantified", 5, has_scenario),
        ("Risk", "Key risks are written in plain language", 5, _present(case.get("risks"))),
        ("Risk", "Benchmark spread is available", 4, _finite(metrics.get("spread_to_benchmark")) is not None),
        ("Risk", "Credit loss assumptions are supplied when relevant", 4, (not is_credit) or (_finite(metrics.get("default_probability")) is not None and _finite(metrics.get("recovery_rate")) is not None)),
        ("Risk", "Call risk is quantified when relevant", 4, (not callable_flag) or _finite(metrics.get("yield_to_call")) is not None),
        ("Execution", "Proposed weight is positive and within the team limit", 4, proposed_weight is not None and proposed_weight > 0 and max_weight is not None and max_weight > 0 and proposed_weight <= max_weight),
    ]
    total_weight = sum(weight for _, _, weight, _ in checks)
    earned = sum(weight for _, _, weight, passed in checks if passed)
    score = round(100.0 * earned / total_weight) if total_weight else 0
    rows = [
        {"Category": group, "Check": label, "Weight": weight, "Status": "Complete" if passed else "Gap"}
        for group, label, weight, passed in checks
    ]
    blockers: list[str] = []
    if eligibility != ELIGIBILITY_VERIFIED:
        blockers.append("Competition eligibility has not been verified against the current official trading rules.")
    elif not eligibility_source or eligibility_date is None:
        blockers.append("Eligibility is marked verified but its source or verification date is missing.")
    if eligibility == ELIGIBILITY_INELIGIBLE:
        blockers.append("The instrument is marked ineligible for the competition simulator.")
    if proposed_weight is not None and max_weight is not None and proposed_weight > max_weight:
        blockers.append("Proposed position weight exceeds the team's stated position limit.")
    if quality_errors:
        blockers.append("The underlying bond analysis contains blocking input errors.")

    if blockers:
        status = "Do not trade"
    elif score >= 85:
        status = "Ready for team approval"
    elif score >= 65:
        status = "Needs evidence"
    else:
        status = "Case incomplete"

    worst_scenario = None
    if has_scenario and "ExpectedReturn" in scenario_grid.columns:
        values = pd.to_numeric(scenario_grid["ExpectedReturn"], errors="coerce")
        if values.notna().any():
            worst_scenario = float(values.min())
    return {
        "score": score,
        "status": status,
        "checks": rows,
        "blockers": blockers,
        "eligibility_status": eligibility,
        "eligibility_source": eligibility_source,
        "eligibility_checked_at": eligibility_date.isoformat() if eligibility_date else None,
        "worst_scenario_return": worst_scenario,
    }


def build_bond_pitch_questions(
    position: Mapping[str, Any],
    metrics: Mapping[str, Any],
    readiness: Mapping[str, Any],
    competition_case: Mapping[str, Any] | None = None,
) -> list[str]:
    """Create security-specific oral-defense prompts from the recorded case."""
    case = dict(competition_case or {})
    ticker = str(position.get("ticker") or "this instrument").upper()
    questions = [
        f"Why does {ticker} fit the client's stated goal better than cash, a Treasury ETF, or another bond?",
        f"What observable evidence supports the yield and price used for {ticker}, and how recent is it?",
        f"What happens to the position if yields rise 100 bp, given duration {_number(metrics.get('modified_duration'))} and DV01 {_money(metrics.get('dv01_usd'))}?",
        f"Why is the selected benchmark appropriate in currency, maturity, and credit quality, and what explains the {_pct(metrics.get('spread_to_benchmark'))} spread?",
        f"Which assumption would invalidate the {ticker} thesis, and who is responsible for monitoring it?",
        "How does the proposed position size affect issuer, duration, credit-rating, and currency concentration?",
        "What is the expected return decomposition: carry, pull-to-par, rate move, spread move, credit loss, FX, and fees?",
    ]
    if position.get("callable"):
        questions.append(
            f"If the issuer calls the bond, why is yield to worst {_pct(metrics.get('yield_to_worst'))} still attractive and how will proceeds be reinvested?"
        )
    if _finite(metrics.get("default_probability")) is not None:
        questions.append(
            f"What evidence supports default probability {_pct(metrics.get('default_probability'))} and recovery {_pct(metrics.get('recovery_rate'))}?"
        )
    for item in readiness.get("checks", []):
        if isinstance(item, Mapping) and item.get("Status") == "Gap":
            questions.append(f"What evidence will close this case gap: {item.get('Check')}?")
    if _present(case.get("counter_thesis")):
        questions.append("What evidence would make the recorded counter-thesis more likely than the base case?")
    return list(dict.fromkeys(questions))[:14]


def generate_bond_competition_memo(
    position: Mapping[str, Any],
    metrics: Mapping[str, Any],
    readiness: Mapping[str, Any],
    competition_case: Mapping[str, Any] | None = None,
    *,
    questions: Sequence[str] = (),
    generated_on: date | None = None,
) -> str:
    """Generate a concise working memo; the team remains responsible for final prose."""
    case = dict(competition_case or {})
    ticker = str(position.get("ticker") or "Unidentified instrument").upper()
    spread = _finite(metrics.get("spread_to_benchmark"))
    spread_text = f"{spread * 10_000:+.0f} bp" if spread is not None else "Not available"
    lines = [
        f"# Bond Competition Case: {ticker}",
        "",
        f"Generated: {(generated_on or date.today()).isoformat()}",
        f"Decision status: {readiness.get('status', 'Not assessed')} ({readiness.get('score', 0)}/100 completeness)",
        f"Competition eligibility: {readiness.get('eligibility_status') or ELIGIBILITY_PENDING}",
        "",
        "## Client fit",
        "",
        f"- Client goal: {case.get('client_goal') or 'Not documented'}",
        f"- Portfolio role: {case.get('portfolio_role') or 'Not documented'}",
        f"- Proposed / maximum weight: {_pct(case.get('proposed_weight'))} / {_pct(case.get('max_position_weight'))}",
        "",
        "## Thesis and timing",
        "",
        str(case.get("thesis") or "Not documented"),
        "",
        f"Why now: {case.get('why_now') or 'Not documented'}",
        "",
        "## Fixed-income evidence",
        "",
        f"- Issuer: {position.get('issuer') or 'Not documented'}",
        f"- Instrument / category: {position.get('bond_instrument_type') or 'Unknown'} / {position.get('bond_category') or 'Unknown'}",
        f"- Price / dirty price: {_number(metrics.get('clean_price'), 3)} / {_number(metrics.get('dirty_price'), 3)}",
        f"- Yield to maturity / call / worst: {_pct(metrics.get('yield_to_maturity'))} / {_pct(metrics.get('yield_to_call'))} / {_pct(metrics.get('yield_to_worst'))}",
        f"- Benchmark / spread: {position.get('benchmark_name') or 'Not documented'} / {spread_text}",
        f"- Modified duration / DV01: {_number(metrics.get('modified_duration'))} / {_money(metrics.get('dv01_usd'))}",
        f"- Annual income / expected credit loss: {_money(metrics.get('annual_income_usd'))} / {_money(metrics.get('expected_loss_usd'))}",
        f"- Worst modeled scenario return: {_pct(readiness.get('worst_scenario_return'))}",
        "",
        "## Risks and decision discipline",
        "",
        f"- Key risks: {case.get('risks') or 'Not documented'}",
        f"- Counter-thesis: {case.get('counter_thesis') or 'Not documented'}",
        f"- Invalidation condition: {case.get('invalidation') or 'Not documented'}",
        f"- Sell / review discipline: {case.get('sell_discipline') or 'Not documented'}",
        "",
        "## Eligibility and sources",
        "",
        f"- Eligibility source: {readiness.get('eligibility_source') or 'Not documented'}",
        f"- Eligibility checked: {readiness.get('eligibility_checked_at') or 'Not documented'}",
        f"- Valuation source: {position.get('valuation_source') or 'Not documented'}",
        f"- Source URL/reference: {position.get('source_url') or 'Not documented'}",
        f"- Price observed: {position.get('price_observed_at') or 'Not documented'}",
        "",
        "## Open blockers",
        "",
    ]
    blockers = readiness.get("blockers", [])
    lines.extend([f"- {item}" for item in blockers] or ["- None recorded."])
    lines.extend(["", "## Pitch-defense questions", ""])
    lines.extend([f"{index}. {question}" for index, question in enumerate(questions, start=1)] or ["1. Not generated."])
    lines.extend([
        "",
        "> Working analytical draft only. Verify every number and source. The student team must write and cite the final submission in its own voice.",
        "",
    ])
    return "\n".join(lines)


def build_bond_relative_value_table(
    positions: Sequence[Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]] = (),
    *,
    as_of: Any = None,
) -> pd.DataFrame:
    """Build a transparent shortlist table without an opaque buy/sell score."""
    by_id = {
        item.get("id"): item for item in performance_rows
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    rows: list[dict[str, Any]] = []
    for position in positions:
        if str(position.get("status") or "open").lower() != "open":
            continue
        performance = by_id.get(position.get("id"), {})
        price = performance.get("current_price", position.get("last_price"))
        metrics = calculate_bond_metrics(position, price, as_of=as_of)
        quality = assess_bond_data_quality(position, as_of=as_of)
        current_yield = _finite(metrics.get("current_yield"))
        expected_loss = _finite(metrics.get("expected_loss_rate"))
        carry_after_loss = current_yield - expected_loss if current_yield is not None and expected_loss is not None else current_yield
        duration = _finite(metrics.get("modified_duration"))
        carry_per_duration = carry_after_loss / duration if carry_after_loss is not None and duration and duration > 0 else None
        eligibility = str(position.get("competition_eligibility_status") or ELIGIBILITY_PENDING)
        rows.append({
            "Identifier": str(position.get("ticker") or "").upper(),
            "Issuer": str(position.get("issuer") or "Unassigned"),
            "Instrument": str(position.get("bond_instrument_type") or "Unknown").title(),
            "Eligibility": eligibility,
            "YieldToWorst": metrics.get("yield_to_worst"),
            "CurrentYield": current_yield,
            "SpreadBps": (_finite(metrics.get("spread_to_benchmark")) * 10_000 if _finite(metrics.get("spread_to_benchmark")) is not None else None),
            "ModifiedDuration": duration,
            "DV01USD": metrics.get("dv01_usd"),
            "ExpectedLossRate": expected_loss,
            "CarryAfterExpectedLoss": carry_after_loss,
            "CarryPerUnitDuration": carry_per_duration,
            "DataQuality": int(quality.get("score") or 0),
            "EvidenceReady": eligibility == ELIGIBILITY_VERIFIED and int(quality.get("score") or 0) >= 75,
        })
    columns = [
        "Identifier", "Issuer", "Instrument", "Eligibility", "YieldToWorst", "CurrentYield",
        "SpreadBps", "ModifiedDuration", "DV01USD", "ExpectedLossRate",
        "CarryAfterExpectedLoss", "CarryPerUnitDuration", "DataQuality", "EvidenceReady",
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "ELIGIBILITY_INELIGIBLE", "ELIGIBILITY_PENDING", "ELIGIBILITY_VERIFIED",
    "assess_bond_competition_case", "build_bond_pitch_questions",
    "build_bond_relative_value_table", "generate_bond_competition_memo",
]
