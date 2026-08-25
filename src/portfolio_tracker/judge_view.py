"""Pure, read-only projection of competition data for external judges.

The judge model deliberately selects fields instead of returning stored records.
Besides keeping the surface compact, this prevents committee ballots and other
write-workflow internals from leaking into the external view.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any


_DEFAULT_QUESTIONS = (
    "Why is this strategy the best fit for this client, rather than merely a good portfolio?",
    "Which single assumption would most damage the portfolio if it proved wrong?",
    "What evidence shows that the portfolio and the report use the same reconciled snapshot?",
)

_APP_MODULES = (
    (
        "overview_tasks",
        "Overview & Tasks",
        "Team coordination, ownership, deadlines, files, discussion, and workspace map.",
    ),
    (
        "competition_readiness",
        "Competition Readiness",
        "Current readiness assessment, evidence coverage, open gaps, and judge prompts.",
    ),
    (
        "mandate_strategy",
        "Mandate & Strategy",
        "Client mandate, strategy versions, holding theses, and approved universe inputs.",
    ),
    (
        "research_workspace",
        "Research Workspace",
        "Company research, sources, catalysts, thesis reviews, macro snapshots, and AI disclosure.",
    ),
    (
        "security_dossiers",
        "Security Dossiers",
        "Canonical dossier versions, investment theses, KPI definitions, and observations.",
    ),
    (
        "investment_committee",
        "Investment Committee",
        "Decision log, revision trail, outcome reviews, red-team work, and lifecycle summaries.",
    ),
    (
        "portfolio_overview",
        "Portfolio Overview",
        "All tracked holdings, marks, valuation provenance, performance, and portfolio authority.",
    ),
    (
        "wins_reconciliation",
        "WInS & Reconciliation",
        "Saved WInS workspace, selected canonical snapshot, reconciliation, and reporting gate.",
    ),
    (
        "risk_scenarios",
        "Risk & Scenarios",
        "Saved Quant runs and the analytical snapshots attached to investment decisions.",
    ),
    (
        "report_pitch",
        "Report & Pitch",
        "Report workspace, validation, evidence, Q&A workspace, and oral-defense prompts.",
    ),
    (
        "rules_compliance",
        "Rules & Compliance",
        "Competition settings, rule checks, official-rules provenance, and integrity controls.",
    ),
)

_RESTRICTED_RECORD_KEYS = {
    "ballot",
    "ballots",
    "dissent",
    "dissents",
    "file_path",
    "password",
    "password_hash",
    "rationale",
    "strongest_objection",
}
_RESTRICTED_KEY_MARKERS = (
    "api_key",
    "credential",
    "password",
    "secret",
    "session_token",
    "token_hash",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _payload(value: Any) -> dict[str, Any]:
    """Return a record payload, while accepting already-unwrapped mappings."""
    record = _mapping(value)
    nested = record.get("payload")
    if not isinstance(nested, Mapping):
        return record
    merged = {key: item for key, item in record.items() if key != "payload"}
    merged.update(nested)
    return merged


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        candidates = value.values()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = value
    else:
        return []
    return [dict(item) for item in candidates if isinstance(item, Mapping)]


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    result = " ".join(str(value).strip().split())
    return result or default


def _first_text(record: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        result = _text(record.get(key))
        if result:
            return result
    return default


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any, default: int = 0) -> int:
    number = _number(value)
    return int(number) if number is not None else default


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1", "ready", "approved", "clean"}:
            return True
        if normalized in {"false", "no", "n", "0", "none", "pending", "blocked"}:
            return False
    return default


def _fraction(value: Any) -> float | None:
    """Normalize either a decimal weight or a 0-100 percentage to a fraction."""
    number = _number(value)
    if number is None or number < 0:
        return None
    if number > 1.0:
        number /= 100.0
    return min(number, 1.0)


def _percentage(value: Any, *, decimal_hint: bool = False) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if decimal_hint and -1.0 <= number <= 1.0:
        return number * 100.0
    return number


def _first_mapping(record: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_rows(record: Mapping[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        result = _rows(record.get(key))
        if result:
            return result
    return []


def _normalise_goals(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = value
    elif value:
        candidates = [value]
    else:
        candidates = []
    for item in candidates:
        if isinstance(item, Mapping):
            goal = dict(item)
            result.append(
                {
                    "name": _first_text(goal, "name", "goal", "label"),
                    "description": _first_text(goal, "description", "success_condition"),
                    "target_weight": _fraction(
                        goal.get("target_weight", goal.get("allocation_pct"))
                    ),
                    "target_amount": _number(
                        goal.get("target_wealth", goal.get("target_amount"))
                    ),
                    "horizon_years": _number(goal.get("horizon_years")),
                    "priority": _number(goal.get("priority")),
                }
            )
        else:
            label = _text(item)
            if label:
                result.append(
                    {
                        "name": label,
                        "description": "",
                        "target_weight": None,
                        "target_amount": None,
                        "horizon_years": None,
                        "priority": None,
                    }
                )
    return result


def _normalise_mandate(record: Any) -> dict[str, Any]:
    mandate = _payload(record)
    constraints = mandate.get("values_constraints")
    if isinstance(constraints, Mapping):
        constraint_text = "; ".join(
            f"{_text(key)}: {_text(value)}"
            for key, value in constraints.items()
            if _text(value)
        )
    else:
        constraint_text = _text(
            constraints or mandate.get("values_constraints_text") or mandate.get("constraints")
        )
    return {
        "available": bool(mandate),
        "client_name": _first_text(mandate, "client_name", "name"),
        "case_status": _first_text(mandate, "case_status", "status"),
        "mandate_summary": _first_text(mandate, "mandate_summary", "summary"),
        "goals": _normalise_goals(mandate.get("goals")),
        "risk_tolerance": _first_text(mandate, "risk_tolerance"),
        "risk_capacity": _first_text(mandate, "risk_capacity"),
        "max_tolerated_drawdown": _percentage(
            mandate.get("max_tolerated_drawdown"), decimal_hint=True
        ),
        "drawdown_response": _first_text(mandate, "drawdown_response"),
        "horizon_years": _number(mandate.get("horizon_years")),
        "liquidity_need_pct": _percentage(
            mandate.get("liquidity_need_pct"), decimal_hint=True
        ),
        "constraints": constraint_text,
        "policy_benchmark": _first_text(mandate, "policy_benchmark"),
        "policy_benchmark_rationale": _first_text(
            mandate, "policy_benchmark_rationale", "benchmark_rationale"
        ),
    }


def _compact_value(value: Any) -> str | list[str]:
    if isinstance(value, Mapping):
        return [
            f"{_text(key)}: {_text(item)}"
            for key, item in value.items()
            if _text(item)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_text(item) for item in value if _text(item)]
    return _text(value)


def _normalise_strategy(record: Any) -> dict[str, Any]:
    strategy = _payload(record)
    return {
        "available": bool(strategy),
        "name": _first_text(strategy, "name", "strategy_name", "title"),
        "status": _first_text(strategy, "status"),
        "version": strategy.get("version"),
        "thesis": _first_text(
            strategy, "thesis", "strategy_thesis", "one_sentence_thesis"
        ),
        "process": _first_text(
            strategy, "process", "selection_process", "investment_process"
        ),
        "selection_factors": _compact_value(strategy.get("selection_factors")),
        "sell_discipline": _first_text(strategy, "sell_discipline", "sell_rules"),
        "rebalance_policy": _first_text(strategy, "rebalance_policy"),
        "max_position_weight": _fraction(strategy.get("max_position_weight")),
        "max_sector_weight": _fraction(strategy.get("max_sector_weight")),
        "min_cash_weight": _fraction(strategy.get("min_cash_weight")),
        "max_cash_weight": _fraction(strategy.get("max_cash_weight")),
        "cash_target": _fraction(
            strategy.get("cash_target", strategy.get("target_cash_weight"))
        ),
        "drift_limit": _fraction(
            strategy.get(
                "drift_limit",
                strategy.get("max_goal_drift", strategy.get("max_sector_drift")),
            )
        ),
    }


def _normalise_positions(value: Any, *, omit_closed: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _rows(value):
        status = _first_text(raw, "status", default="open").casefold()
        if omit_closed and status in {"closed", "exited", "sold"}:
            continue
        ticker = _first_text(raw, "ticker", "security_id", "symbol").upper()
        if not ticker:
            continue
        quantity = _number(raw.get("quantity", raw.get("shares")))
        price = _number(
            raw.get("current_price", raw.get("last_price", raw.get("price")))
        )
        market_value = _number(
            raw.get(
                "market_value",
                raw.get("current_value", raw.get("position_value", raw.get("value"))),
            )
        )
        if market_value is None and quantity is not None and price is not None:
            market_value = quantity * price
        decimal_weight = raw.get("weight")
        if decimal_weight is None:
            decimal_weight = raw.get("portfolio_weight")
        if decimal_weight is not None:
            weight = _fraction(decimal_weight)
        else:
            percentage_weight = _number(raw.get("weight_pct"))
            weight = (
                min(max(percentage_weight, 0.0), 100.0) / 100.0
                if percentage_weight is not None
                else None
            )
        return_pct = _percentage(raw.get("return_pct"))
        if return_pct is None:
            return_pct = _percentage(raw.get("return"), decimal_hint=True)
        rows.append(
            {
                "ticker": ticker,
                "security_type": _first_text(
                    raw, "security_type", "asset_type", "type", default="Security"
                ),
                "status": status or "open",
                "quantity": quantity,
                "current_price": price,
                "market_value": market_value,
                "weight": weight,
                "weight_pct": None if weight is None else weight * 100.0,
                "cost": _number(raw.get("cost", raw.get("cost_basis"))),
                "pnl": _number(
                    raw.get("pnl", raw.get("total_pnl", raw.get("unrealized_pnl")))
                ),
                "return_pct": return_pct,
                "sector": _first_text(raw, "sector", default="Unassigned"),
                "client_goal": _first_text(
                    raw, "client_goal", "primary_goal", default="Unassigned"
                ),
                "lifecycle_state": _first_text(raw, "lifecycle_state", "stage"),
                "currency": _first_text(raw, "currency", default="USD").upper(),
                "price_source": _first_text(raw, "price_source"),
                "valuation_source": _first_text(raw, "valuation_source"),
                "price_observed_at": _first_text(
                    raw,
                    "price_observed_at",
                    "valuation_as_of",
                )
                or None,
            }
        )

    return rows


def _fill_missing_position_weights(
    positions: Sequence[dict[str, Any]],
    *,
    denominator: float | None,
) -> None:
    total_market_value = sum(
        float(row["market_value"] or 0.0)
        for row in positions
        if row["market_value"] is not None
    )
    weight_denominator = (
        float(denominator)
        if denominator is not None and denominator > 0
        else total_market_value
    )
    if weight_denominator <= 0:
        return
    for row in positions:
        if row["weight"] is None and row["market_value"] is not None:
            row["weight"] = float(row["market_value"]) / weight_denominator
            row["weight_pct"] = row["weight"] * 100.0


def _snapshot_metrics(
    snapshot: Mapping[str, Any],
    positions: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    payload = _first_mapping(snapshot, "payload") or dict(snapshot)
    invested = _number(payload.get("invested_value"))
    if invested is None:
        values = [row.get("market_value") for row in positions]
        invested = (
            sum(float(value) for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )
    cash = _number(payload.get("cash_value", payload.get("cash")))
    total = _number(payload.get("total_value", payload.get("equity")))
    if total is None and invested is not None and cash is not None:
        total = invested + cash
    return invested, cash, total


def _portfolio_model(
    raw: Mapping[str, Any],
    report: Mapping[str, Any],
    report_validation: Any = None,
) -> dict[str, Any]:
    report_snapshot = _first_mapping(report, "portfolio_snapshot")
    validation = _mapping(report_validation)
    validation_issue_codes = {
        _first_text(item, "code")
        for item in _rows(validation.get("issues"))
        if _first_text(item, "code")
    }
    raw_issue_codes = validation.get("issue_codes")
    if isinstance(raw_issue_codes, Sequence) and not isinstance(
        raw_issue_codes,
        (str, bytes, bytearray),
    ):
        validation_issue_codes.update(
            _text(item) for item in raw_issue_codes if _text(item)
        )
    pipeline = _payload(raw.get("pipeline"))
    canonical = _first_mapping(pipeline, "canonical_snapshot")
    tracker = _payload(
        raw.get("tracker_performance", raw.get("performance", raw.get("tracker")))
    )

    source_tier = "none"
    source_label = "No portfolio snapshot available"
    snapshot: dict[str, Any] = {}
    positions: list[dict[str, Any]] = []
    reconciled = False
    as_of = ""
    snapshot_id = ""
    total_return_pct: float | None = None
    total_pnl: float | None = None
    invested: float | None = None
    cash: float | None = None
    total: float | None = None
    last_known_good = False
    snapshot_fresh: bool | None = None
    reporting_ready: bool | None = None
    reporting_blockers: list[str] = []
    valuation_sources: list[str] = []
    return_suppressed = False
    valuation_note = ""

    if (
        report_snapshot
        and _boolean(report_snapshot.get("reconciled"))
        and _first_text(report_snapshot, "snapshot_id")
        and "portfolio_snapshot_hash" not in validation_issue_codes
    ):
        source_tier = "report_bound"
        source_label = "Report-bound reconciled snapshot"
        snapshot = report_snapshot
        positions = _normalise_positions(report_snapshot.get("positions"))
        reconciled = True
        snapshot_id = _first_text(report_snapshot, "snapshot_id")
        as_of = _first_text(report_snapshot, "as_of", "observed_at", "updated_at")
        invested, cash, total = _snapshot_metrics(report_snapshot, positions)
        _fill_missing_position_weights(
            positions,
            denominator=total if total is not None else invested,
        )
        attribution = _first_mapping(report, "performance_attribution")
        total_return_pct = _percentage(
            attribution.get("portfolio_return"), decimal_hint=True
        )
    elif (
        _first_text(pipeline, "authority").casefold() == "wins_reconciled"
        and canonical
        and _first_text(canonical, "snapshot_id")
    ):
        source_tier = "wins_reconciled"
        source_label = "WInS reconciled canonical snapshot"
        snapshot = canonical
        payload = _first_mapping(canonical, "payload") or canonical
        positions = _normalise_positions(payload.get("positions"))
        reconciled = True
        snapshot_id = _first_text(canonical, "snapshot_id")
        as_of = _first_text(canonical, "observed_at", "as_of", "received_at")
        invested, cash, total = _snapshot_metrics(canonical, positions)
        _fill_missing_position_weights(
            positions,
            denominator=total if total is not None else invested,
        )
        selection = _first_mapping(pipeline, "selection")
        last_known_good = _boolean(pipeline.get("last_known_good"))
        if "is_fresh" in selection:
            snapshot_fresh = _boolean(selection.get("is_fresh"))
        reporting_binding = _first_mapping(
            _first_mapping(pipeline, "consumer_bindings"),
            "reporting",
        )
        if reporting_binding:
            reporting_ready = _boolean(reporting_binding.get("allowed"))
            reporting_blockers = [
                _text(item)
                for item in reporting_binding.get("blockers", [])
                if _text(item)
            ]
    else:
        tracker_rows = _rows(tracker.get("positions"))
        tracker_has_capital = any(
            _number(tracker.get(key)) is not None
            for key in ("equity", "cash", "cash_before_pnl", "initial_capital")
        )
        if tracker_rows or tracker_has_capital:
            source_tier = "tracker_provisional"
            source_label = "Tracker fallback — provisional"
            positions = _normalise_positions(tracker_rows, omit_closed=True)
            snapshot_id = _first_text(tracker, "snapshot_id")
            as_of = _first_text(tracker, "as_of", "observed_at", "updated_at")
            invested = sum(
                float(row["market_value"] or 0.0)
                for row in positions
                if row["market_value"] is not None
            )
            cash = _number(tracker.get("cash"))
            if cash is None:
                cash_before_pnl = _number(tracker.get("cash_before_pnl"))
                realised_pnl = _number(tracker.get("realized_pnl"))
                open_cash_income = _number(tracker.get("open_cash_income"))
                if cash_before_pnl is not None:
                    cash = (
                        cash_before_pnl
                        + (realised_pnl or 0.0)
                        + (open_cash_income or 0.0)
                    )
            total = _number(tracker.get("equity"))
            if total is None and cash is not None:
                total = invested + cash
            _fill_missing_position_weights(
                positions,
                denominator=total if total is not None else invested,
            )
            valuation_sources = sorted(
                {
                    (
                        f"{row['valuation_source']} ({row['price_source']})"
                        if row.get("valuation_source")
                        and row.get("price_source")
                        and str(row["valuation_source"]).casefold()
                        != str(row["price_source"]).casefold()
                        else _first_text(row, "valuation_source", "price_source")
                    )
                    for row in positions
                    if _first_text(row, "valuation_source", "price_source")
                }
            )
            observed_times = [
                str(row["price_observed_at"])
                for row in positions
                if row.get("price_observed_at")
            ]
            if not as_of and observed_times:
                as_of = max(observed_times)
            return_suppressed = any(
                str(row.get("price_source") or "").casefold() == "entry fallback"
                for row in positions
            )
            if return_suppressed:
                valuation_note = (
                    "Portfolio return is withheld because at least one open holding "
                    "uses its entry price as a valuation fallback."
                )
                total_return_pct = None
            else:
                total_return_pct = _percentage(tracker.get("total_return_pct"))
            total_pnl = _number(tracker.get("total_pnl"))

    return {
        "available": source_tier != "none",
        "source_tier": source_tier,
        "source_label": source_label,
        "snapshot_id": snapshot_id or None,
        "as_of": as_of or None,
        "reconciled": reconciled,
        "position_count": len(positions),
        "positions": positions,
        "invested_value": invested,
        "cash": cash,
        "cash_value": cash,
        "equity": total,
        "total_return_pct": total_return_pct,
        "total_pnl": total_pnl,
        "last_known_good": last_known_good,
        "snapshot_fresh": snapshot_fresh,
        "reporting_ready": reporting_ready,
        "reporting_blockers": reporting_blockers,
        "valuation_sources": valuation_sources,
        "return_suppressed": return_suppressed,
        "valuation_note": valuation_note,
    }


def _readiness_model(value: Any) -> dict[str, Any]:
    readiness = _mapping(value)
    constitution = _first_mapping(readiness, "constitution")
    governance = _first_mapping(readiness, "governance")
    gates = _first_mapping(readiness, "operating_gates")
    priority_actions = [
        {
            "key": _first_text(item, "key"),
            "area": _first_text(item, "area", default="Workflow"),
            "action": _first_text(item, "action"),
        }
        for item in _rows(readiness.get("priority_actions"))
        if _first_text(item, "action")
    ]
    next_action = _first_mapping(readiness, "next_action")
    return {
        "available": bool(readiness),
        "overall_score": _integer(readiness.get("overall_score")),
        "status": _first_text(readiness, "status", default="Not assessed"),
        "constitution_score": _integer(constitution.get("score")),
        "dossier_score": _integer(readiness.get("dossier_score")),
        "governance_score": _integer(governance.get("score")),
        "operating_gates": {str(key): _boolean(value) for key, value in gates.items()},
        "priority_actions": priority_actions,
        "next_action": (
            {
                "area": _first_text(next_action, "area", default="Workflow"),
                "action": _first_text(next_action, "action"),
            }
            if next_action
            else None
        ),
        "returns_excluded": True,
    }


def _fair_values(thesis: Mapping[str, Any]) -> dict[str, float | None]:
    nested = _first_mapping(thesis, "fair_value_scenarios", "fair_values")
    return {
        case: _number(
            nested.get(
                case,
                thesis.get(
                    f"{case}_fair_value",
                    thesis.get(f"fair_value_{case}", thesis.get(f"{case}_value")),
                ),
            )
        )
        for case in ("bear", "base", "bull")
    }


def _dossier_model(raw: Mapping[str, Any], readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    assessments = {
        _first_text(item, "ticker").upper(): item
        for item in _rows(readiness.get("dossiers"))
        if _first_text(item, "ticker")
    }
    theses = _rows(raw.get("theses"))
    thesis_by_ticker = {
        _first_text(item, "ticker").upper(): item
        for item in theses
        if _first_text(item, "ticker")
    }
    universe_by_ticker = {
        _first_text(item, "ticker").upper(): item
        for item in _rows(raw.get("approved_securities"))
        if _first_text(item, "ticker")
    }
    result: list[dict[str, Any]] = []
    for ticker in sorted(set(thesis_by_ticker) | set(assessments)):
        source_record = thesis_by_ticker.get(ticker, {})
        thesis = _payload(source_record)
        assessment = assessments.get(ticker, {})
        universe_record = universe_by_ticker.get(ticker, {})
        universe_payload = _payload(universe_record)
        missing = [
            _first_text(item, "label", "key")
            for item in _rows(assessment.get("missing"))
            if _first_text(item, "label", "key")
        ]
        result.append(
            {
                "ticker": ticker,
                "name": _first_text(thesis, "company_name", "name"),
                "dossier_status": _first_text(
                    source_record,
                    "canonical_dossier_status",
                    "status",
                    default="Not recorded",
                ),
                "dossier_source": _first_text(
                    source_record,
                    "source_system",
                    default="legacy_or_unspecified",
                ),
                "dossier_version": source_record.get("canonical_dossier_version"),
                "content_hash_present": bool(
                    _first_text(source_record, "content_hash")
                ),
                "eligibility": _first_text(
                    universe_record,
                    "eligibility",
                    "status",
                    default=_first_text(
                        universe_payload,
                        "eligibility",
                        default="not listed",
                    ),
                ),
                "universe_approved": (
                    _boolean(universe_record.get("approved"))
                    if universe_record
                    else None
                ),
                "universe_provenance": _first_text(
                    universe_record,
                    "provenance_status",
                    default=_first_text(universe_payload, "provenance_status"),
                ),
                "universe_source": _first_text(
                    universe_payload,
                    "source_name",
                    default=_first_text(universe_record, "source_system"),
                ),
                "universe_as_of": _first_text(
                    universe_payload,
                    "source_as_of",
                    default=_first_text(universe_record, "updated_at"),
                )
                or None,
                "portfolio_role": _first_text(thesis, "portfolio_role"),
                "client_goal": _first_text(thesis, "primary_goal", "client_goal"),
                "why_now": _first_text(thesis, "why_now"),
                "thesis": _first_text(thesis, "investment_thesis", "thesis"),
                "value_drivers": _compact_value(thesis.get("value_drivers")),
                "monitoring_kpis": _compact_value(thesis.get("monitoring_kpis")),
                "bear_case": _first_text(thesis, "bear_case"),
                "base_case": _first_text(thesis, "base_case"),
                "bull_case": _first_text(thesis, "bull_case"),
                "fair_values": _fair_values(thesis),
                "fair_value_currency": _first_text(
                    thesis,
                    "fair_value_currency",
                    "valuation_currency",
                ),
                "margin_of_safety": _percentage(
                    thesis.get("margin_of_safety"), decimal_hint=True
                ),
                "counter_thesis": _first_text(thesis, "counter_thesis"),
                "risks": _compact_value(thesis.get("risks")),
                "invalidation": _first_text(
                    thesis, "invalidation", "invalidation_condition"
                ),
                "catalysts": _compact_value(thesis.get("catalysts")),
                "review_date": _first_text(
                    thesis, "review_date", default=_first_text(source_record, "next_review_at")
                ),
                "readiness_score": _integer(assessment.get("score")),
                "readiness_status": _first_text(
                    assessment, "status", default="Not assessed"
                ),
                "missing": missing,
                "source_count": _integer(assessment.get("source_count")),
                "primary_source_count": _integer(
                    assessment.get("primary_source_count")
                ),
                "catalyst_count": _integer(assessment.get("catalyst_count")),
            }
        )
    return result


def _vote_round_summary(value: Any) -> dict[str, Any]:
    round_data = _mapping(value)
    return {
        "status": _first_text(round_data, "status", default="not_open"),
        "opened": _boolean(round_data.get("opened")),
        "revealed": _boolean(round_data.get("revealed")),
        "eligible_count": _integer(round_data.get("eligible_count")),
        "submitted_count": _integer(round_data.get("submitted_count")),
        "remaining_count": _integer(round_data.get("remaining_count")),
        "result": _first_text(round_data, "outcome", "result") or None,
    }


def _reconciliation_summary(value: Any) -> dict[str, Any]:
    reconciliation = _payload(value)
    nested = _first_mapping(reconciliation, "reconciliation")
    exception_value = reconciliation.get(
        "open_exceptions", reconciliation.get("exceptions")
    )
    explicit_exception_count = _number(reconciliation.get("open_exception_count"))
    if explicit_exception_count is not None:
        exception_count: int | None = max(0, int(explicit_exception_count))
    elif isinstance(exception_value, Mapping):
        exception_count = len(exception_value)
    elif isinstance(exception_value, Sequence) and not isinstance(
        exception_value, (str, bytes, bytearray)
    ):
        exception_count = len(exception_value)
    else:
        exception_count = 0 if reconciliation else None
    return {
        "available": bool(reconciliation),
        "status": _first_text(reconciliation, "status", default="Not recorded"),
        "reconciliation_id": _first_text(
            reconciliation, "reconciliation_id", default=_first_text(nested, "reconciliation_id")
        )
        or None,
        "snapshot_id": _first_text(
            reconciliation,
            "wins_snapshot_id",
            "snapshot_id",
            default=_first_text(nested, "wins_snapshot_id", "snapshot_id"),
        )
        or None,
        "open_exception_count": exception_count,
        "as_of": _first_text(
            reconciliation, "as_of", "updated_at", "created_at", "completed_at"
        )
        or None,
    }


def _decision_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    cases = _first_rows(raw, "investment_cases", "lifecycles", "decisions")
    items: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    fully_voted = 0
    approved_states = {
        "approved",
        "sizing",
        "wins_execution",
        "reconciliation",
        "active",
        "exited",
        "closed",
    }
    for case in cases:
        state = _first_text(case, "state", "status", "stage", default="unknown")
        state_counts[state] += 1
        pre = _vote_round_summary(case.get("pre_vote"))
        post = _vote_round_summary(case.get("post_vote"))
        final_approvals = _rows(case.get("final_approvals"))
        final_complete = _boolean(case.get("final_approval_complete"))
        if (
            pre["revealed"]
            and pre["submitted_count"] >= 2
            and post["revealed"]
            and post["submitted_count"] >= 2
            and final_complete
        ):
            fully_voted += 1
        audit = _mapping(case.get("audit"))
        audit_events = _rows(case.get("audit_events"))
        proposal = _mapping(case.get("proposal"))
        items.append(
            {
                "id": case.get("id"),
                "ticker": _first_text(case, "ticker", "security_ticker").upper(),
                "state": state,
                "owner": _first_text(case, "owner_id", "owner"),
                "challenger": _first_text(case, "challenger_id", "challenger"),
                "updated_at": _first_text(case, "updated_at") or None,
                "proposal_summary": _first_text(
                    proposal, "summary", "rationale", "investment_case"
                ),
                "proposed_weight_pct": (
                    _percentage(proposal.get("proposed_weight_pct"))
                    if proposal.get("proposed_weight_pct") is not None
                    else _percentage(
                        proposal.get("proposed_weight"),
                        decimal_hint=True,
                    )
                ),
                "pre_vote": pre,
                "post_vote": post,
                "final_approval_complete": final_complete,
                "approval_count": sum(
                    _first_text(item, "decision").casefold() == "approve"
                    for item in final_approvals
                ),
                "reconciliation": _reconciliation_summary(
                    case.get("latest_reconciliation")
                ),
                "audit_valid": (
                    _boolean(audit.get("valid")) if "valid" in audit else None
                ),
                "audit_event_count": len(audit_events),
            }
        )
    return {
        "count": len(items),
        "approved_or_later_count": sum(
            _first_text(item, "state").casefold() in approved_states for item in items
        ),
        "fully_voted_count": fully_voted,
        "by_state": dict(sorted(state_counts.items())),
        "items": items,
    }


def _evidence_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    sources = _first_rows(raw, "sources", "research_sources")
    catalysts = _rows(raw.get("catalysts"))
    thesis_reviews = _rows(raw.get("thesis_reviews"))
    red_team_reviews = _rows(raw.get("red_team_reviews"))
    ai_usage = _rows(raw.get("ai_usage"))
    qa_rounds = _first_rows(raw, "qa_rounds", "qa_sessions")
    approved = _rows(raw.get("approved_securities"))
    source_items = [
        {
            "ticker": _first_text(item, "ticker").upper(),
            "title": _first_text(item, "title", "name", "source_name"),
            "source_type": _first_text(item, "source_type", "type"),
            "primary_source": _boolean(item.get("primary_source")),
            "citation": _first_text(item, "citation"),
            "url": _first_text(item, "url", "link"),
            "as_of": _first_text(item, "as_of", "published_at", "created_at") or None,
            "verified_by": _first_text(item, "verified_by", "recorded_by"),
        }
        for item in sources
    ]
    evidence_times = [
        timestamp
        for item in [
            *sources,
            *catalysts,
            *thesis_reviews,
            *red_team_reviews,
            *ai_usage,
            *qa_rounds,
        ]
        if (
            timestamp := _first_text(
                item,
                "as_of",
                "published_at",
                "updated_at",
                "created_at",
                "recorded_at",
            )
        )
    ]
    return {
        "source_count": len(sources),
        "primary_source_count": sum(
            _boolean(item.get("primary_source")) for item in sources
        ),
        "catalyst_count": len(catalysts),
        "thesis_review_count": len(thesis_reviews),
        "red_team_review_count": len(red_team_reviews),
        "ai_disclosure_count": len(ai_usage),
        "qa_round_count": len(qa_rounds),
        "approved_security_count": len(approved),
        "as_of": max(evidence_times) if evidence_times else None,
        "sources": source_items,
    }


def _report_model(report: Mapping[str, Any], validation_value: Any) -> dict[str, Any]:
    validation = _mapping(validation_value)
    sections = _first_mapping(report, "sections")
    section_rows = [item for item in sections.values() if isinstance(item, Mapping)]
    approvals = _first_mapping(report, "approvals")
    snapshot = _first_mapping(report, "portfolio_snapshot")
    freeze = _first_mapping(report, "freeze")
    attribution = _first_mapping(report, "performance_attribution")
    contribution_rows = [
        {
            "label": _first_text(item, "label", "id"),
            "contribution_pct": _percentage(
                item.get("contribution"), decimal_hint=True
            ),
        }
        for item in _rows(attribution.get("contributions"))
    ]
    issues = [
        {
            "code": _first_text(item, "code"),
            "message": _first_text(item, "message"),
        }
        for item in _rows(validation.get("issues"))
    ]
    return {
        "available": bool(report),
        "report_id": _first_text(report, "report_id") or None,
        "title": _first_text(report, "title"),
        "report_type": _first_text(report, "report_type"),
        "status": _first_text(report, "status", default="Not started"),
        "schema_version": _first_text(report, "schema_version"),
        "updated_at": _first_text(report, "updated_at") or None,
        "finalised_at": _first_text(report, "finalised_at") or None,
        "frozen": bool(freeze) or _first_text(report, "status").casefold() in {
            "frozen",
            "approved",
            "final",
            "finalised",
        },
        "section_count": len(section_rows),
        "ready_section_count": sum(
            _first_text(item, "status").casefold() == "ready" for item in section_rows
        ),
        "approval_count": len(approvals),
        "snapshot_id": _first_text(snapshot, "snapshot_id") or None,
        "snapshot_reconciled": _boolean(snapshot.get("reconciled")),
        "validation_ready": (
            _boolean(validation.get("is_ready"))
            if "is_ready" in validation
            else None
        ),
        "issue_count": _integer(validation.get("issue_count"), default=len(issues)),
        "issues": issues,
        "page_budget": _number(validation.get("page_budget", report.get("page_budget"))),
        "estimated_pages": _number(validation.get("estimated_pages")),
        "claim_count": _integer(
            validation.get("claim_count"), default=len(_first_mapping(report, "claims"))
        ),
        "evidence_count": _integer(
            validation.get("evidence_count"), default=len(_first_mapping(report, "evidence"))
        ),
        "figure_count": _integer(
            validation.get("figure_count"), default=len(_first_mapping(report, "figures"))
        ),
        "case_study_count": _integer(
            validation.get("case_study_count"),
            default=len(_first_mapping(report, "case_studies")),
        ),
        "performance_attribution": {
            "available": bool(attribution),
            "as_of": _first_text(attribution, "as_of") or None,
            "benchmark": _first_text(attribution, "benchmark"),
            "portfolio_return_pct": _percentage(
                attribution.get("portfolio_return"), decimal_hint=True
            ),
            "benchmark_return_pct": _percentage(
                attribution.get("benchmark_return"), decimal_hint=True
            ),
            "active_return_pct": _percentage(
                attribution.get("active_return"), decimal_hint=True
            ),
            "attributed_return_pct": _percentage(
                attribution.get("attributed_return"), decimal_hint=True
            ),
            "residual_pct": _percentage(
                attribution.get("residual"), decimal_hint=True
            ),
            "methodology": _first_text(attribution, "methodology"),
            "contributions": contribution_rows,
        },
    }


def _compliance_model(value: Any) -> dict[str, Any]:
    checks = _rows(value)
    items = [
        {
            "status": _first_text(item, "status", default="pending").casefold(),
            "rule": _first_text(item, "rule", "name"),
            "detail": _first_text(item, "detail", "message"),
        }
        for item in checks
    ]
    status_counts = Counter(item["status"] for item in items)
    return {
        "check_count": len(items),
        "pass_count": status_counts.get("pass", 0),
        "fail_count": status_counts.get("fail", 0),
        "pending_count": status_counts.get("pending", 0),
        "all_clear": (
            bool(items)
            and status_counts.get("fail", 0) == 0
            and status_counts.get("pending", 0) == 0
        ),
        "items": items,
    }


def _integrity_model(
    raw: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    report: Mapping[str, Any],
    decisions: Mapping[str, Any],
    dossiers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pipeline = _payload(raw.get("pipeline"))
    contexts = _first_mapping(pipeline, "context_status")
    bindings = _first_mapping(pipeline, "consumer_bindings", "bindings")
    reporting_binding = _mapping(bindings.get("reporting"))
    reconciliation = _reconciliation_summary(raw.get("reconciliation_record"))
    rules = _payload(raw.get("rules_record"))
    rules_available = bool(rules)
    decision_items = _rows(decisions.get("items"))
    audited = [item for item in decision_items if item.get("audit_valid") is not None]
    canonical = _first_mapping(pipeline, "canonical_snapshot")
    canonical_id = _first_text(canonical, "snapshot_id") or None
    report_snapshot_id = report.get("snapshot_id")
    holding_tickers = {
        _first_text(item, "ticker").upper()
        for item in _rows(portfolio.get("positions"))
        if _first_text(item, "ticker")
    }
    dossier_tickers = {
        _first_text(item, "ticker").upper()
        for item in dossiers
        if _first_text(item, "ticker")
    }
    universe_by_ticker = {
        _first_text(item, "ticker").upper(): item
        for item in _rows(raw.get("approved_securities"))
        if _first_text(item, "ticker")
    }
    eligible_tickers = {
        ticker
        for ticker, item in universe_by_ticker.items()
        if _boolean(item.get("approved"))
        or _first_text(item, "eligibility", "status").casefold() == "eligible"
    }
    holdings_without_dossier = sorted(holding_tickers - dossier_tickers)
    holdings_without_universe_record = sorted(
        holding_tickers - set(universe_by_ticker)
    )
    holdings_not_eligible = sorted(
        ticker
        for ticker in holding_tickers & set(universe_by_ticker)
        if ticker not in eligible_tickers
    )
    holding_count = len(holding_tickers)
    return {
        "pipeline_status": _first_text(pipeline, "status", default="Not available"),
        "pipeline_authority": _first_text(pipeline, "authority", default="none"),
        "reporting_binding_allowed": (
            _boolean(reporting_binding.get("allowed"))
            if reporting_binding
            else None
        ),
        "pipeline_reason_codes": [
            _text(item) for item in pipeline.get("reason_codes", []) if _text(item)
        ]
        if isinstance(pipeline.get("reason_codes"), Sequence)
        and not isinstance(pipeline.get("reason_codes"), (str, bytes, bytearray))
        else [],
        "context_status": {str(key): _boolean(value) for key, value in contexts.items()},
        "reconciliation": reconciliation,
        "rules": {
            "available": rules_available,
            "version": rules.get("version"),
            "captured_at": _first_text(
                rules, "captured_at", "observed_at", "updated_at"
            )
            or None,
            "content_hash_present": (
                bool(_first_text(rules, "content_hash", "hash"))
                if rules_available
                else None
            ),
            "all_acknowledged": (
                _boolean(rules.get("all_acknowledged"))
                or bool(rules.get("acknowledged_by"))
                if rules_available
                else None
            ),
            "acknowledged_count": len(rules.get("acknowledged_by", {}))
            if isinstance(rules.get("acknowledged_by"), Mapping)
            else len(rules.get("acknowledged_by", []))
            if isinstance(rules.get("acknowledged_by"), Sequence)
            and not isinstance(rules.get("acknowledged_by"), (str, bytes, bytearray))
            else 0,
        },
        "audited_decision_count": len(audited),
        "valid_audit_count": sum(item.get("audit_valid") is True for item in audited),
        "invalid_audit_count": sum(item.get("audit_valid") is False for item in audited),
        "report_snapshot_matches_pipeline": (
            str(report_snapshot_id) == canonical_id
            if report_snapshot_id and canonical_id
            else None
        ),
        "selected_portfolio_source": portfolio.get("source_tier", "none"),
        "holding_coverage": {
            "holding_count": holding_count,
            "dossier_coverage_pct": (
                len(holding_tickers & dossier_tickers) / holding_count * 100.0
                if holding_count
                else None
            ),
            "eligible_universe_coverage_pct": (
                len(holding_tickers & eligible_tickers) / holding_count * 100.0
                if holding_count
                else None
            ),
            "holdings_without_dossier": holdings_without_dossier,
            "holdings_without_universe_record": holdings_without_universe_record,
            "holdings_not_eligible": holdings_not_eligible,
            "all_holdings_covered": (
                not (
                    holdings_without_dossier
                    or holdings_without_universe_record
                    or holdings_not_eligible
                )
                if holding_count
                else None
            ),
        },
    }


def _question_model(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        supplied = [_text(item) for item in value if _text(item)]
    else:
        supplied = []
    return list(dict.fromkeys([*_DEFAULT_QUESTIONS, *supplied]))


def _safe_judge_value(value: Any) -> Any:
    """Return a detached JSON-like value with private workflow fields removed."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalised_key = key.strip().casefold()
            if normalised_key in _RESTRICTED_RECORD_KEYS or any(
                marker in normalised_key for marker in _RESTRICTED_KEY_MARKERS
            ):
                continue
            result[key] = _safe_judge_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_safe_judge_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _text(value)


def _record_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return 1 if value else 0
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value)
    return 0 if value in (None, "") else 1


def _group(
    label: str,
    value: Any,
    *,
    count_as_saved: bool = True,
) -> dict[str, Any]:
    safe_value = _safe_judge_value(value)
    return {
        "label": label,
        "record_count": _record_count(safe_value) if count_as_saved else 0,
        "value": safe_value,
    }


def _evidence_available(evidence: Mapping[str, Any]) -> bool:
    return any(
        _integer(evidence.get(key)) > 0
        for key in (
            "source_count",
            "catalyst_count",
            "thesis_review_count",
            "red_team_review_count",
            "ai_disclosure_count",
            "qa_round_count",
            "approved_security_count",
        )
    )


def _integrity_available(integrity: Mapping[str, Any]) -> bool:
    reconciliation = _mapping(integrity.get("reconciliation"))
    rules = _mapping(integrity.get("rules"))
    return bool(
        reconciliation.get("available")
        or rules.get("available")
        or _integer(integrity.get("audited_decision_count"))
        or _first_text(integrity, "selected_portfolio_source", default="none")
        != "none"
        or _first_text(integrity, "pipeline_authority", default="none") != "none"
    )


def _module_record(
    key: str,
    label: str,
    description: str,
    groups: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    group_rows = [dict(item) for item in groups if isinstance(item, Mapping)]
    count = sum(_integer(item.get("record_count")) for item in group_rows)
    return {
        "key": key,
        "label": label,
        "description": description,
        "status": "Available" if count else "Not used yet",
        "record_count": count,
        "groups": group_rows,
    }


def _app_record_model(
    source: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    decisions: Mapping[str, Any],
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
    compliance: Mapping[str, Any],
    integrity: Mapping[str, Any],
    questions: Sequence[str],
) -> dict[str, Any]:
    """Mirror every team workspace as a complete, safe, read-only record."""
    saved = _mapping(source.get("app_record"))
    collaboration = _mapping(saved.get("collaboration"))
    strategy = _mapping(saved.get("strategy"))
    research = _mapping(saved.get("research"))
    governance = _mapping(saved.get("governance"))
    operations = _mapping(saved.get("operations"))

    module_groups: dict[str, list[dict[str, Any]]] = {
        "overview_tasks": [
            _group("Tasks", collaboration.get("tasks")),
            _group("Subprojects", collaboration.get("subprojects")),
            _group("Subproject links", collaboration.get("subproject_files")),
            _group("Uploaded file register", collaboration.get("files")),
            _group("Team discussion", collaboration.get("chat")),
            _group("Workspace map", collaboration.get("mindmap")),
        ],
        "competition_readiness": [
            _group(
                "Readiness assessment",
                readiness if _boolean(readiness.get("available")) else {},
            ),
            _group(
                "Evidence coverage",
                evidence if _evidence_available(evidence) else {},
            ),
            _group("Judge question bank", list(questions), count_as_saved=False),
        ],
        "mandate_strategy": [
            _group("Client mandate record", saved.get("mandate_record")),
            _group("Strategy version history", strategy.get("versions")),
            _group("Holding theses", strategy.get("holding_theses")),
            _group("Approved securities", strategy.get("approved_securities")),
            _group("Authoritative universe", strategy.get("authoritative_universe")),
            _group("Universe snapshot history", strategy.get("universe_history")),
        ],
        "research_workspace": [
            _group("Company research", research.get("company_research")),
            _group("Research sources", research.get("sources")),
            _group("Catalyst calendar", research.get("catalysts")),
            _group("Thesis reviews", research.get("thesis_reviews")),
            _group("Macro snapshots", research.get("macro_snapshots")),
            _group("AI-use disclosures", research.get("ai_usage")),
        ],
        "security_dossiers": [
            _group("Canonical dossier register", research.get("security_dossiers")),
        ],
        "investment_committee": [
            _group("Decision log", governance.get("decision_log")),
            _group("Decision edit history", governance.get("decision_edits")),
            _group("Decision outcome reviews", governance.get("decision_reviews")),
            _group("Red-team reviews", research.get("red_team_reviews")),
            _group(
                "Canonical lifecycle summaries",
                decisions if _integer(decisions.get("count")) else {},
            ),
        ],
        "portfolio_overview": [
            _group(
                "Canonical judge portfolio",
                portfolio if _boolean(portfolio.get("available")) else {},
            ),
            _group("Competition position ledger", operations.get("tracker_positions")),
            _group(
                "Tracker performance",
                operations.get("tracker_performance")
                if _record_count(operations.get("tracker_positions"))
                else {},
            ),
        ],
        "wins_reconciliation": [
            _group("Saved WInS workspace", operations.get("pipeline_record")),
            _group(
                "Resolved portfolio pipeline",
                operations.get("pipeline") if operations.get("pipeline_record") else {},
            ),
            _group("Latest reconciliation", operations.get("reconciliation_record")),
        ],
        "risk_scenarios": [
            _group("Saved Quant run history", operations.get("quant_runs")),
            _group("Decision-linked Quant snapshots", operations.get("quant_snapshots")),
            _group("Other current analytical records", operations.get("analytical_records")),
        ],
        "report_pitch": [
            _group("Report workspace", operations.get("report_record")),
            _group("Report validation", operations.get("report_validation")),
            _group("Q&A workspace", operations.get("qa_workspace")),
            _group("Q&A session records", operations.get("qa_rounds")),
            _group("Oral-defense prompts", list(questions), count_as_saved=False),
        ],
        "rules_compliance": [
            _group("Competition settings", operations.get("compliance_settings")),
            _group(
                "Compliance checks",
                compliance if _integer(compliance.get("check_count")) else {},
            ),
            _group("Official rules snapshot", operations.get("rules_record")),
            _group(
                "Integrity assessment",
                integrity if _integrity_available(integrity) else {},
            ),
            _group("Operating-system event register", operations.get("operating_events")),
        ],
    }

    modules = [
        _module_record(key, label, description, module_groups.get(key, []))
        for key, label, description in _APP_MODULES
    ]
    return {
        "title": "Complete App Record",
        "scope_note": (
            "This read-only record mirrors every team workspace and all saved competition "
            "data. Credentials, server file paths, and unrevealed individual voting material "
            "are excluded."
        ),
        "module_count": len(modules),
        "available_module_count": sum(
            item["status"] == "Available" for item in modules
        ),
        "modules": modules,
    }


def build_judge_view_model(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a stable, non-mutating judge projection from heterogeneous records.

    Portfolio authority is strict: a reconciled snapshot attached to the report
    wins, followed by the canonical reconciled WInS pipeline snapshot.  The team
    tracker is only a provisional fallback and is labelled as such.
    """
    source = _mapping(raw)
    report_payload = _payload(source.get("report_record", source.get("report_workspace")))
    readiness_source = _mapping(source.get("readiness"))
    portfolio = _portfolio_model(
        source,
        report_payload,
        source.get("report_validation"),
    )
    readiness = _readiness_model(readiness_source)
    decisions = _decision_model(source)
    report = _report_model(report_payload, source.get("report_validation"))
    dossiers = _dossier_model(source, readiness_source)
    evidence = _evidence_model(source)
    compliance = _compliance_model(source.get("compliance_checks"))
    questions = _question_model(source.get("questions"))
    integrity = _integrity_model(
        source,
        portfolio,
        report,
        decisions,
        dossiers,
    )
    return {
        "mandate": _normalise_mandate(source.get("mandate_record", source.get("mandate"))),
        "strategy": _normalise_strategy(source.get("strategy_record", source.get("strategy"))),
        "readiness": readiness,
        "portfolio": portfolio,
        "dossiers": dossiers,
        "decisions": decisions,
        "evidence": evidence,
        "report": report,
        "compliance": compliance,
        "integrity": integrity,
        "app_record": _app_record_model(
            source,
            readiness=readiness,
            portfolio=portfolio,
            decisions=decisions,
            evidence=evidence,
            report=report,
            compliance=compliance,
            integrity=integrity,
            questions=questions,
        ),
        "questions": questions,
    }


__all__ = ["build_judge_view_model"]
