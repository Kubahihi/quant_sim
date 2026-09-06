"""Transparent readiness diagnostics for an investment-competition workflow."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import re
from typing import Any, Iterable, Mapping, Sequence


def _payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("payload")
    return dict(nested) if isinstance(nested, Mapping) else dict(value)


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


_MEASURABLE_SELECTION_TEXT = re.compile(
    r"(?:[<>]=?|\b(?:above|below|at least|at most|no more than|minimum|maximum|"
    r"positive|negative|top quartile|bottom quartile)\b|\d)",
    flags=re.IGNORECASE,
)


def _measurable_selection_value(value: Any) -> bool:
    """Return whether a selection criterion contains an observable pass/fail rule."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(value.strip() and _MEASURABLE_SELECTION_TEXT.search(value))
    if isinstance(value, Mapping):
        keys = {str(key).strip().casefold() for key in value}
        threshold_keys = {
            "threshold", "minimum", "maximum", "min", "max", "operator",
            "criterion", "test", "target", "value",
        }
        if keys & threshold_keys and any(
            _present(value.get(key))
            for key in value
            if str(key).strip().casefold() in threshold_keys
        ):
            return True
        return any(_measurable_selection_value(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_measurable_selection_value(item) for item in value)
    return False


def _has_measurable_selection_rules(strategy: Mapping[str, Any]) -> bool:
    for field in ("selection_rules", "selection_factors", "selection_process"):
        value = strategy.get(field)
        if isinstance(value, Mapping):
            if any(_measurable_selection_value(item) for item in value.values()):
                return True
        elif _measurable_selection_value(value):
            return True
    return False


def _assessment(checks: Sequence[tuple[str, str, bool]]) -> dict[str, Any]:
    completed = [{"key": key, "label": label} for key, label, passed in checks if passed]
    missing = [{"key": key, "label": label} for key, label, passed in checks if not passed]
    score = round(100.0 * len(completed) / len(checks)) if checks else 0
    status = "Competition ready" if score >= 90 else "Developing" if score >= 65 else "Material gaps"
    return {"score": score, "status": status, "completed": completed, "missing": missing, "total": len(checks)}


def _build_priority_actions(
    constitution: Mapping[str, Any],
    dossiers: Sequence[Mapping[str, Any]],
    governance: Mapping[str, Any],
    operating_gates: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """Return one dependency-ordered action per workflow area."""
    actions: list[dict[str, Any]] = []
    constitution_missing = list(constitution.get("missing", []))
    if constitution_missing:
        actions.append({
            "key": "strategy_constitution",
            "area": "Client & Policy",
            "action": (
                "Complete the strategy constitution "
                f"({len(constitution_missing)} required field group(s) remain)."
            ),
        })

    incomplete_dossiers = [item for item in dossiers if item.get("missing")]
    if not dossiers:
        actions.append({
            "key": "security_dossiers",
            "area": "Research",
            "action": "Create the first canonical Security Dossier with primary evidence.",
        })
    elif incomplete_dossiers:
        weakest = min(incomplete_dossiers, key=lambda item: float(item.get("score") or 0))
        actions.append({
            "key": "security_dossiers",
            "area": "Research",
            "action": (
                f"Close gaps in {len(incomplete_dossiers)} Security Dossier(s); "
                f"start with {weakest.get('ticker') or 'the lowest-scoring holding'}."
            ),
        })

    supporting_governance_keys = {"reviews", "red_team", "qa", "rules", "ai"}
    supporting_gaps = [
        item for item in governance.get("missing", [])
        if item.get("key") in supporting_governance_keys
    ]
    if supporting_gaps:
        actions.append({
            "key": "governance_evidence",
            "area": "Evidence & Defense",
            "action": (
                "Complete the review, rules, AI-disclosure, and oral-defense trail "
                f"({len(supporting_gaps)} gate(s) remain)."
            ),
        })

    gate_actions = {
        "investment_committee": (
            "Investment Committee",
            "Complete independent pre/post votes and final sign-off for an approved case.",
        ),
        "wins_reconciliation": (
            "Portfolio / WInS",
            "Import the authoritative WInS snapshot and close every reconciliation exception.",
        ),
        "report_frozen_to_snapshot": (
            "Report & Pitch",
            "Freeze the report against the clean, reconciled WInS portfolio snapshot.",
        ),
    }
    for key, (area, action) in gate_actions.items():
        if not operating_gates.get(key, False):
            actions.append({"key": key, "area": area, "action": action})

    return actions


def assess_strategy_constitution(mandate_record: Any, strategy_record: Any) -> dict[str, Any]:
    """Score explicit client and process rules; no market performance enters the score."""
    mandate = _payload(mandate_record)
    strategy = _payload(strategy_record)
    behavior = mandate.get("behavioral_profile") if isinstance(mandate.get("behavioral_profile"), Mapping) else {}
    behavior_answers = behavior.get("answers") if isinstance(behavior.get("answers"), Mapping) else {}
    behavior_actions = behavior.get("drawdown_actions") if isinstance(behavior.get("drawdown_actions"), Mapping) else {}
    checks = [
        ("client", "Named client and case status", _present(mandate.get("client_name")) and _present(mandate.get("case_status"))),
        ("goals", "Measurable client goal buckets", _present(mandate.get("goals"))),
        ("risk_tolerance", "Risk tolerance", _present(mandate.get("risk_tolerance")) and mandate.get("risk_tolerance") != "Not specified"),
        ("risk_capacity", "Risk capacity", _present(mandate.get("risk_capacity")) and mandate.get("risk_capacity") != "Not specified"),
        ("drawdown", "Maximum tolerated drawdown and response", _present(mandate.get("max_tolerated_drawdown")) and _present(mandate.get("drawdown_response"))),
        ("financial_picture", "Client total financial picture", _present(mandate.get("total_financial_picture"))),
        ("horizon", "Investment horizon", float(mandate.get("horizon_years") or 0) > 0),
        ("liquidity", "Liquidity need explicitly quantified", mandate.get("liquidity_need_pct") is not None),
        ("constraints", "Values and investment constraints", _present(mandate.get("values_constraints")) or _present(mandate.get("values_constraints_text"))),
        ("benchmark", "Policy benchmark and rationale", _present(mandate.get("policy_benchmark")) and _present(mandate.get("policy_benchmark_rationale"))),
        (
            "behavior",
            "Behavioral profile, drawdown actions, and decision protocol",
            len(behavior_answers) >= 10 and len(behavior_actions) >= 3 and _present(behavior.get("decision_protocol")),
        ),
        ("thesis", "One-sentence strategy thesis", any(_present(strategy.get(key)) for key in ("thesis", "strategy_thesis", "one_sentence_thesis"))),
        (
            "selection",
            "Measurable security-selection rules",
            _has_measurable_selection_rules(strategy),
        ),
        ("sizing", "Position and sector sizing rules", float(strategy.get("max_position_weight") or 0) > 0 and float(strategy.get("max_sector_weight") or 0) > 0),
        (
            "sell",
            "Explicit sell / thesis-break discipline",
            any(_present(strategy.get(key)) for key in ("sell_discipline", "sell_rules")),
        ),
        (
            "rebalance",
            "Explicit rebalancing policy",
            _present(strategy.get("rebalance_policy")),
        ),
    ]
    return _assessment(checks)


def assess_security_dossier(
    ticker: str,
    thesis_record: Any,
    sources: Iterable[Mapping[str, Any]] = (),
    catalysts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    thesis = _payload(thesis_record)
    code = str(ticker or thesis_record.get("ticker") if isinstance(thesis_record, Mapping) else ticker or "").strip().upper()
    linked_sources = [item for item in sources if not item.get("ticker") or str(item.get("ticker")).upper() == code]
    primary_sources = [item for item in linked_sources if bool(item.get("primary_source"))]
    linked_catalysts = [item for item in catalysts if str(item.get("ticker") or "").upper() == code]
    dossier_catalysts = thesis.get("catalysts")
    dossier_catalyst_count = (
        len(dossier_catalysts)
        if isinstance(dossier_catalysts, Sequence) and not isinstance(dossier_catalysts, (str, bytes))
        else int(_present(dossier_catalysts))
    )
    fair_values = thesis.get("fair_value_scenarios") if isinstance(thesis.get("fair_value_scenarios"), Mapping) else {}
    checks = [
        ("role", "Portfolio role and client goal", _present(thesis.get("portfolio_role")) and _present(thesis.get("primary_goal"))),
        ("why_now", "Why now / timing", _present(thesis.get("why_now"))),
        ("thesis", "Core investment thesis", _present(thesis.get("investment_thesis"))),
        ("drivers", "Value drivers and monitoring KPIs", _present(thesis.get("value_drivers")) and _present(thesis.get("monitoring_kpis"))),
        ("scenarios", "Bear/base/bull operating cases", all(_present(thesis.get(key)) for key in ("bear_case", "base_case", "bull_case"))),
        ("valuation", "Bear/base/bull fair values", all(_present(fair_values.get(key)) for key in ("bear", "base", "bull"))),
        ("margin", "Margin of safety", thesis.get("margin_of_safety") is not None),
        ("counter", "Strongest counter-thesis", _present(thesis.get("counter_thesis"))),
        ("risks", "Key risks", _present(thesis.get("risks"))),
        ("invalidation", "Observable invalidation condition", _present(thesis.get("invalidation"))),
        ("catalysts", "Dated catalyst / thesis test", bool(linked_catalysts) or dossier_catalyst_count > 0),
        ("evidence", "At least one primary source", bool(primary_sources)),
        ("review", "Next review date", _present(thesis.get("review_date")) or _present(thesis_record.get("next_review_at") if isinstance(thesis_record, Mapping) else None)),
    ]
    result = _assessment(checks)
    result.update({
        "ticker": code,
        "source_count": len(linked_sources),
        "primary_source_count": len(primary_sources),
        "catalyst_count": len(linked_catalysts) + dossier_catalyst_count,
    })
    return result


def build_competition_readiness(
    *,
    mandate: Any,
    strategy: Any,
    theses: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]] = (),
    catalysts: Sequence[Mapping[str, Any]] = (),
    decisions: Sequence[Mapping[str, Any]] = (),
    thesis_reviews: Sequence[Mapping[str, Any]] = (),
    decision_reviews: Sequence[Mapping[str, Any]] = (),
    red_team_reviews: Sequence[Mapping[str, Any]] = (),
    ai_usage: Sequence[Mapping[str, Any]] = (),
    investment_cases: Sequence[Mapping[str, Any]] = (),
    reconciliation: Mapping[str, Any] | None = None,
    report_workspace: Mapping[str, Any] | None = None,
    qa_sessions: Sequence[Mapping[str, Any]] = (),
    rules_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    constitution = assess_strategy_constitution(mandate, strategy)
    dossiers = [assess_security_dossier(str(item.get("ticker") or ""), item, sources, catalysts) for item in theses if item.get("ticker")]
    dossier_score = round(sum(item["score"] for item in dossiers) / len(dossiers)) if dossiers else 0
    approved_cases = [
        item for item in investment_cases
        if str(item.get("state") or item.get("status") or item.get("stage") or "").lower()
        in {
            "approved", "sized", "executed", "reconciled", "closed",
            "sizing", "wins_execution", "reconciliation", "active", "exited",
        }
    ]

    def _round_complete(item: Mapping[str, Any], round_name: str) -> bool:
        canonical = item.get(f"{round_name}_vote")
        if isinstance(canonical, Mapping):
            return bool(
                canonical.get("revealed")
                and int(canonical.get("submitted_count") or 0) >= 2
            )
        legacy = item.get(f"{round_name}_votes")
        return isinstance(legacy, Mapping) and len(legacy) >= 2

    independently_voted = [
        item for item in approved_cases
        if _round_complete(item, "pre")
        and _round_complete(item, "post")
        and bool(
            item.get("final_approval_complete")
            or _present(item.get("final_approvals"))
            or _present(item.get("sign_offs"))
        )
    ]
    canonical_position_reviews = [
        review
        for case in investment_cases
        for review in (
            case.get("position_reviews")
            if isinstance(case.get("position_reviews"), Sequence)
            else []
        )
        if isinstance(review, Mapping)
    ]
    reconciliation_data = dict(reconciliation or {})
    reconciliation_clean = (
        str(reconciliation_data.get("status") or "").lower() in {"clean", "reconciled"}
        and not reconciliation_data.get("open_exceptions")
        and not reconciliation_data.get("exceptions")
    )
    report_data = dict(report_workspace or {})
    report_frozen = bool(
        report_data.get("frozen")
        or str(report_data.get("status") or "").lower() in {"frozen", "approved", "final"}
    )
    nested_report_snapshot = (
        report_data.get("portfolio_snapshot")
        if isinstance(report_data.get("portfolio_snapshot"), Mapping)
        else {}
    )
    report_has_snapshot = _present(
        report_data.get("portfolio_snapshot_id")
        or report_data.get("as_of_snapshot")
        or nested_report_snapshot.get("snapshot_id")
    )
    rules_data = dict(rules_snapshot or {})
    rules_acknowledged = bool(
        rules_data
        and _present(rules_data.get("content_hash") or rules_data.get("hash"))
        and bool(rules_data.get("all_acknowledged") or rules_data.get("acknowledged_by"))
    )
    governance_checks = [
        (
            "decisions",
            "Canonical Investment Committee lifecycle is populated",
            bool(investment_cases),
        ),
        (
            "reviews",
            "Theses or canonical positions have append-only reviews",
            bool(thesis_reviews or canonical_position_reviews),
        ),
        ("red_team", "Independent red-team challenge is recorded", bool(red_team_reviews)),
        (
            "investment_committee",
            "An approved case has independent pre/post votes and final sign-off",
            bool(independently_voted),
        ),
        (
            "reconciliation",
            "Latest WInS reconciliation is clean with no open exceptions",
            reconciliation_clean,
        ),
        (
            "report",
            "Report is frozen against an as-of portfolio snapshot",
            report_frozen and report_has_snapshot and reconciliation_clean,
        ),
        ("qa", "At least one scored Q&A rehearsal is recorded", bool(qa_sessions)),
        ("rules", "Latest official-rules snapshot is hashed and acknowledged", rules_acknowledged),
        ("sources", "Research evidence register is populated", bool(sources)),
        ("ai", "AI usage/disclosure log is recorded", bool(ai_usage)),
    ]
    governance = _assessment(governance_checks)
    overall = round(0.40 * constitution["score"] + 0.40 * dossier_score + 0.20 * governance["score"])
    operating_gates = {
        "investment_committee": bool(independently_voted),
        "wins_reconciliation": reconciliation_clean,
        "report_frozen_to_snapshot": report_frozen and report_has_snapshot and reconciliation_clean,
    }
    if not all(operating_gates.values()):
        overall = min(overall, 89)
    priority_actions = _build_priority_actions(
        constitution,
        dossiers,
        governance,
        operating_gates,
    )
    return {
        "overall_score": overall,
        "status": "Pitch ready" if overall >= 90 else "Evidence build" if overall >= 65 else "Foundation incomplete",
        "constitution": constitution,
        "dossiers": dossiers,
        "dossier_score": dossier_score,
        "governance": governance,
        "operating_gates": operating_gates,
        "priority_actions": priority_actions,
        "next_action": priority_actions[0] if priority_actions else None,
    }


def build_pitch_question_bank(readiness: Mapping[str, Any]) -> list[str]:
    """Create an evidence-oriented oral-defense checklist from current gaps."""
    questions = [
        "Why is this strategy the best fit for this client, rather than merely a good portfolio?",
        "Which single assumption would most damage the portfolio if it proved wrong?",
        "How does the policy benchmark reflect the client's actual goals and constraints?",
    ]
    priority_actions = [
        item for item in readiness.get("priority_actions", [])
        if isinstance(item, Mapping) and item.get("action")
    ]
    questions.extend(
        f"What evidence proves this action is complete: {item['action']}"
        for item in priority_actions[:6]
    )
    return questions


def generate_competition_brief(
    readiness: Mapping[str, Any],
    *,
    mandate: Any,
    strategy: Any,
    sources: Sequence[Mapping[str, Any]] = (),
    ai_usage: Sequence[Mapping[str, Any]] = (),
    generated_on: date | None = None,
) -> str:
    """Generate a concise working brief; students remain responsible for final prose."""
    m = _payload(mandate)
    s = _payload(strategy)
    lines = [
        "# Competition Readiness Brief",
        "",
        f"Generated: {(generated_on or date.today()).isoformat()}",
        f"Readiness: {readiness.get('overall_score', 0)}/100 — {readiness.get('status', 'Not assessed')}",
        "",
        "## Priority actions",
        "",
    ]
    priority_actions = [
        item for item in readiness.get("priority_actions", [])
        if isinstance(item, Mapping) and item.get("action")
    ]
    if priority_actions:
        lines.extend(
            f"- {item.get('area')}: {item.get('action')}"
            for item in priority_actions
        )
    else:
        lines.append("- No readiness blockers are currently recorded.")
    lines.extend([
        "",
        "## Client and investment policy",
        "",
        f"- Client: {m.get('client_name') or 'Not documented'}",
        f"- Mandate: {m.get('mandate_summary') or 'Not documented'}",
        f"- Risk tolerance / capacity: {m.get('risk_tolerance') or '—'} / {m.get('risk_capacity') or '—'}",
        f"- Policy benchmark: {m.get('policy_benchmark') or 'Not documented'}",
        "",
        "## Strategy constitution",
        "",
        str(s.get("thesis") or s.get("strategy_thesis") or s.get("one_sentence_thesis") or "Not documented"),
        "",
        "## Security dossier readiness",
        "",
    ])
    for dossier in readiness.get("dossiers", []):
        gaps = ", ".join(item["label"] for item in dossier.get("missing", [])) or "None"
        lines.append(f"- {dossier.get('ticker')}: {dossier.get('score')}%; gaps: {gaps}")
    lines.extend(["", "## Works cited register", ""])
    if sources:
        for item in sources:
            lines.append(f"- {item.get('title') or 'Untitled'} — {item.get('publisher') or 'Unknown publisher'} — {item.get('url') or 'No URL'}")
    else:
        lines.append("- No research sources recorded.")
    lines.extend(["", "## AI usage disclosure", ""])
    if ai_usage:
        for item in ai_usage:
            lines.append(f"- {item.get('tool_name')}: {item.get('purpose')} | Used: {item.get('output_used') or 'Not stated'} | Verified: {item.get('verification_notes') or 'Not stated'}")
    else:
        lines.append("- No AI usage recorded. Confirm whether that is accurate before submission.")
    lines.extend(["", "> Working evidence brief only. The team must verify every claim and write the final submission in its own voice.", ""])
    return "\n".join(lines)


__all__ = [
    "assess_security_dossier", "assess_strategy_constitution", "build_competition_readiness",
    "build_pitch_question_bank", "generate_competition_brief",
]
