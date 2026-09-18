"""Versioned case facts and submission checks from the supplied assessment.

The assessment is a team-supplied secondary source dated 2026-09-17, not a
live official rules snapshot. Workbook return assumptions are illustrations.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


CASE_ID = "laura_gao_2026_2027"
SOURCE_DATE = "2026-09-17"
SOURCES = {
    "assessment": "Laura_Gao_assessment_CS.pdf",
    "cashflow": "Laura_Gao_Cashflow.xlsx",
    "origin": "Team-supplied assessment and illustrative workbook; not a primary-source rules snapshot",
    "as_of": SOURCE_DATE,
    "assessment_sha256": "fb7c5a8d21d482e42dfcf6f64bf94e5405533319ab15670148f4cf4f0d192b7a",
    "cashflow_sha256": "fb69a5943628f89edb72f1a0328d0aad5a656ac981b2daac4e64cbf2b85de0ec",
    "official_rules": "https://globalyouth.wharton.upenn.edu/competitions/investment-competition/rules-roles/",
    "official_faq": "https://globalyouth.wharton.upenn.edu/competitions/investment-competition/faq/",
}

REQUIREMENTS = (
    ("K01", "Two deposits: $300,000 in 2027 and $150,000 in 2028", "Fixed cash-flow calendar"),
    ("K02", "No other cash flows before 2033", "Fixed cash-flow calendar"),
    ("K03", "All flows occur at the beginning of the year", "Dated path engine"),
    ("K04", "Ten nominal $50,000 payments, 2033–2042", "Ten-payment ledger"),
    ("K05", "Operations funded solely by this portfolio", "No borrowing or external rescue"),
    ("K06", "Reserve before the first payment in 2033", "Reserve allocation includes the first payment"),
    ("K07", "Reserve composition and its evolution", "Annual reserve weights; team must justify instruments"),
    ("K08", "Define and evaluate high confidence", "Joint payment success, reserve gap and team threshold"),
    ("K09", "Responsible facility contribution", "Surplus rule after reserve and minimum flexibility"),
    ("K10", "Preserve financial flexibility", "Retained amount and explicit flexibility deficit"),
    ("K11", "Communicate a 2033 contribution range in 2031", "Separate beginning-of-2031 conditional projection"),
    ("K12", "State confidence in the whole interval", "Coverage, lower-bound risk and joint payment success"),
    ("K13", "Explain favorable and adverse markets", "Paired scenarios and stress comparisons"),
    ("K14", "Prepare fundraising material", "Exportable conditional draft; team review required"),
    ("K15", "Explain all assumptions", "Saved policy, allocation, history and model disclosures"),
    ("K16", "Separate WInS from the client projection", "Fixed client deposits; no WInS balance or P&L input"),
    ("K17", "One strategy across Trading Notes, IPS and Final Report", "Existing rulebook/evidence workflow; team consistency review"),
)


def cashflow_calendar() -> list[dict[str, Any]]:
    return [
        {"calendar_year": year, "model_year": year - 2026,
         "deposit_usd": {2027: 300_000, 2028: 150_000}.get(year, 0),
         "operating_payment_usd": 50_000 if 2033 <= year <= 2042 else 0,
         "timing": "Beginning of year",
         "milestone": {2031: "Partner communication; valuation date is a team choice",
                       2033: "Reserve first, then facility/flexibility; first payment is inside reserve",
                       2042: "Final payment; model ends immediately afterwards"}.get(year, "")}
        for year in range(2026, 2043)
    ]


def deliverables_calendar() -> list[dict[str, Any]]:
    rows = [
        ("Team Roster", "2026-10-09", "Full names, unique emails, dates of birth, graduation years; team and school details"),
        ("Trading Notes Analysis", "2026-10-23", "Three executed WInS trades; exact original notes; each reflection at most 100 words"),
        ("Investment Policy Statement", "2026-11-06", "Pitch <=50 words; IPS <=500 words; trading and strategy lock"),
        ("Final Report + separate school documentation", "2026-12-04", "Full client analysis; exact final format and school template still require confirmation"),
    ]
    result = []
    for title, day, details in rows:
        due = datetime.fromisoformat(day + "T17:00:00").replace(tzinfo=ZoneInfo("America/New_York"))
        result.append({"deliverable": title, "deadline_et": due.isoformat(),
                       "deadline_prague": due.astimezone(ZoneInfo("Europe/Prague")).isoformat(),
                       "requirements": details, "source": "Supplied assessment, pp. 10–11"})
    return result


def word_count(text: str) -> int:
    """Whitespace-delimited count, with no punctuation-based limit workaround."""
    return len(str(text or "").split())


def check_trading_notes(notes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks = [{"check": "Exactly three executed trades", "passed": len(notes) == 3}]
    refs = [str(note.get("execution_reference") or "").strip() for note in notes]
    checks.append({"check": "Three distinct execution evidence references", "passed": len(refs) == 3 and all(refs) and len(set(refs)) == 3})
    for i, note in enumerate(notes, 1):
        count = word_count(str(note.get("reflection") or ""))
        checks.extend([
            {"check": f"Trade {i}: original WInS note present", "passed": bool(str(note.get("note") or "").strip())},
            {"check": f"Trade {i}: exact note and execution confirmed by team", "passed": note.get("verbatim_confirmed") is True},
            {"check": f"Trade {i}: reflection ({count}/100 words)", "passed": 0 < count <= 100},
        ])
    return checks


def check_ips_text(pitch: str, ips: str) -> list[dict[str, Any]]:
    pitch_words, ips_words = word_count(pitch), word_count(ips)
    return [
        {"check": f"Elevator pitch ({pitch_words}/50 words)", "passed": 0 < pitch_words <= 50},
        {"check": f"IPS ({ips_words}/500 words)", "passed": 0 < ips_words <= 500},
        {"check": "No external links in pitch or IPS", "passed": not bool(re.search(r"https?://|www\.", pitch + " " + ips, re.I))},
    ]


def save_case_section(connection: Any, section: str, values: Mapping[str, Any], *, updated_by: str) -> dict[str, Any]:
    """Merge into the latest canonical mandate without replacing team decisions."""
    from src.portfolio_tracker.strategy_store import load_client_mandate, save_client_mandate

    if section not in {"planning", "deliverables"}:
        raise ValueError("Unknown Laura workspace section.")
    record = load_client_mandate(connection)
    mandate = deepcopy((record or {}).get("payload") or {})
    case = dict(mandate.get("laura_case") or {})
    case.update({"case_id": CASE_ID, "sources": dict(SOURCES), section: deepcopy(dict(values))})
    mandate["laura_case"] = case
    if not record:
        mandate.update({"client_name": "Laura Gao", "case_status": "Official case entered",
                        "base_currency": "USD", "risk_tolerance": "Not specified",
                        "investable_capital": 0.0,
                        "mandate_summary": "Taiwan creative residency: prioritize ten operating payments before the facility contribution. Two future deposits; no WInS capital transfer."})
    return save_client_mandate(connection, mandate, updated_by=updated_by)


def fundraising_draft(interval: Mapping[str, Any], *, state_description: str) -> str:
    """A conditional working text, not a promise or an automatically sent message."""
    return (
        f"Planning basis: {state_description}; valuation at the beginning of 2031. "
        f"Our modeled contribution at the beginning of 2033 ranges from "
        f"${interval['lower_usd']:,.0f} to ${interval['upper_usd']:,.0f}. "
        f"The unrounded interval contains {interval['empirical_interval_coverage']:.1%} of modeled outcomes; "
        f"the risk of falling below its lower bound is {interval['below_lower_probability']:.1%}. "
        f"The modeled joint chance of falling in that range AND making all ten operating payments is "
        f"{interval['interval_and_all_payments_probability']:.1%}. "
        "The ten annual $50,000 operating payments in 2033–2042 take priority. "
        "The contribution is determined only after allocating the operating reserve and retaining flexibility. "
        "Market losses, reserve costs, fees or revised assumptions may reduce the contribution, including to zero. "
        "This is a conditional model estimate, not a binding commitment or guarantee. "
        "We will update the range using the observed portfolio and reserve prices before committing funds."
    )
