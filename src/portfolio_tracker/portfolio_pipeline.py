"""Canonical live-portfolio pipeline for tracker, quant, risk, and reporting.

The pipeline selects one reconciled WInS snapshot and binds every downstream
consumer to that same immutable snapshot ID.  When the current WInS state is
not clean it can continue analysis from the last known good snapshot, but the
reporting gate remains blocked and the degradation is explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import re
from typing import Any

from src.data.reliability import (
    assess_snapshot,
    create_data_snapshot,
    select_reliable_snapshot,
)
from src.portfolio_tracker.reconciliation_ledger import (
    materialize_reconciliation,
    migrate_reconciliation_ledger,
    reconciliation_readiness_gate,
)


ANALYSIS_CONSUMERS = (
    "tracker",
    "quant",
    "risk",
    "factors",
    "scenarios",
    "fx",
    "reporting",
)


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only finite JSON values.") from exc


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text or text.casefold() in {"n/a", "na", "none", "null", "-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[$€£,%\s,]", "", text)
    try:
        number = float(text)
    except ValueError:
        return None
    if negative:
        number = -number
    return number if math.isfinite(number) else None


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    keys = {
        re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_"): value
        for key, value in row.items()
    }
    for name in names:
        canonical = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
        if canonical in keys and keys[canonical] not in (None, ""):
            return keys[canonical]
    return None


def _position_rows(rows: Any) -> list[Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        if _first(rows, ("ticker", "ticker_symbol", "symbol", "security_symbol")) is not None:
            return [rows]
        result: list[Mapping[str, Any]] = []
        for ticker, value in rows.items():
            if isinstance(value, Mapping):
                result.append({"ticker": ticker, **value})
            else:
                result.append({"ticker": ticker, "quantity": value})
        return result
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def normalize_portfolio_positions(rows: Any) -> list[dict[str, Any]]:
    """Normalise common WInS/tracker position shapes and aggregate lots."""
    normalised: list[dict[str, Any]] = []
    for row in _position_rows(rows):
        ticker = " ".join(
            str(_first(row, ("ticker", "ticker_symbol", "symbol", "security_symbol")) or "")
            .strip()
            .split()
        ).upper()
        if not ticker:
            continue
        quantity = _number(_first(row, ("quantity", "shares", "units", "qty")))
        price = _number(_first(row, ("current_price", "market_price", "last_price", "price")))
        market_value = _number(
            _first(row, ("market_value", "current_value", "position_value", "value"))
        )
        if market_value is None and quantity is not None and price is not None:
            market_value = quantity * price
        if price is None and market_value is not None and quantity not in (None, 0.0):
            price = market_value / quantity

        total_cost = _number(_first(row, ("total_cost", "total_cost_basis", "book_value")))
        unit_cost = _number(_first(row, ("unit_cost", "average_cost", "entry_price", "cost_basis")))
        if total_cost is None and unit_cost is not None and quantity is not None:
            total_cost = unit_cost * quantity
        if unit_cost is None and total_cost is not None and quantity not in (None, 0.0):
            unit_cost = total_cost / quantity
        asset_type = " ".join(
            str(
                _first(row, ("asset_type", "security_type", "instrument_type", "type")) or "Unknown"
            )
            .strip()
            .split()
        )
        raw_currency = _first(row, ("currency", "trading_currency", "market_currency"))
        currency = (
            " ".join(str(raw_currency).strip().split()).upper()
            if raw_currency not in (None, "")
            else None
        )
        normalised.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "unit_cost": unit_cost,
                "total_cost": total_cost,
                "current_price": price,
                "market_value": market_value,
                "asset_type": asset_type,
                "currency": currency,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normalised:
        grouped.setdefault(row["ticker"], []).append(row)
    result: list[dict[str, Any]] = []
    for ticker in sorted(grouped):
        lots = grouped[ticker]

        def complete_sum(field: str) -> float | None:
            values = [lot.get(field) for lot in lots]
            return (
                sum(float(value) for value in values)
                if all(value is not None for value in values)
                else None
            )

        quantity = complete_sum("quantity")
        total_cost = complete_sum("total_cost")
        market_value = complete_sum("market_value")
        currencies = sorted({lot["currency"] for lot in lots}, key=lambda item: str(item))
        asset_types = sorted({str(lot["asset_type"]) for lot in lots})
        result.append(
            {
                "ticker": ticker,
                "quantity": quantity,
                "unit_cost": (
                    total_cost / quantity
                    if total_cost is not None and quantity not in (None, 0.0)
                    else None
                ),
                "total_cost": total_cost,
                "current_price": (
                    market_value / quantity
                    if market_value is not None and quantity not in (None, 0.0)
                    else None
                ),
                "market_value": market_value,
                "asset_type": asset_types[0] if len(asset_types) == 1 else "Mixed",
                "currency": currencies[0] if len(currencies) == 1 else "MIXED",
                "source_lot_count": len(lots),
            }
        )
    return result


def create_portfolio_snapshot(
    positions: Any,
    *,
    provider: str,
    observed_at: datetime | str,
    received_at: datetime | str | None = None,
    method: str = "live",
    source_reference: str = "",
    imported_by: str = "",
    cash_value: float = 0.0,
    base_currency: str = "USD",
    expected_tickers: Sequence[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Build a weighted competition-portfolio snapshot with provenance."""
    rows = normalize_portfolio_positions(positions)
    cash = _number(cash_value)
    if cash is None:
        raise ValueError("cash_value must be a finite number.")
    invested_values = [row["market_value"] for row in rows]
    invested_value = (
        sum(float(value) for value in invested_values)
        if all(value is not None for value in invested_values)
        else None
    )
    total_value = invested_value + cash if invested_value is not None else None
    for row in rows:
        row["weight"] = (
            float(row["market_value"]) / total_value
            if row["market_value"] is not None and total_value not in (None, 0.0)
            else None
        )

    fx_exposures: dict[str, float | None] = {}
    currency_labels = {str(row["currency"] or "UNKNOWN") for row in rows}
    for currency in sorted(currency_labels):
        values = [
            row["market_value"] for row in rows if str(row["currency"] or "UNKNOWN") == currency
        ]
        fx_exposures[currency] = (
            sum(float(value) for value in values)
            if all(value is not None for value in values)
            else None
        )
    base = " ".join(str(base_currency or "").strip().split()).upper()
    if not base:
        raise ValueError("base_currency is required.")
    if base not in fx_exposures:
        fx_exposures[base] = cash
    elif fx_exposures[base] is not None:
        fx_exposures[base] = float(fx_exposures[base]) + cash
    payload = {
        "positions": rows,
        "cash_value": cash,
        "cash_weight": cash / total_value if total_value not in (None, 0.0) else None,
        "invested_value": invested_value,
        "total_value": total_value,
        "base_currency": base,
        "fx_exposures": fx_exposures,
    }
    return create_data_snapshot(
        payload,
        dataset="competition_portfolio",
        provider=provider,
        observed_at=observed_at,
        received_at=received_at,
        method=method,
        source_reference=source_reference,
        records_path="positions",
        required_fields=("ticker", "quantity", "market_value", "weight", "currency"),
        expected_keys=expected_tickers,
        imported_by=imported_by,
        notes=notes,
    )


def _context_active(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if value.get("active") is False:
        return False
    status = str(value.get("status") or "").strip().casefold()
    return status not in {"draft", "inactive", "archived", "superseded"}


def _is_wins_snapshot(snapshot: Mapping[str, Any]) -> bool:
    source = snapshot.get("source") if isinstance(snapshot.get("source"), Mapping) else {}
    provider = str(source.get("provider") or "").casefold()
    reference = str(source.get("reference") or "").casefold()
    return "wins" in provider or "wins" in reference


def _approved_reconciliation_snapshot_ids(ledger: Mapping[str, Any]) -> set[str]:
    approved: set[str] = set()
    for record in ledger.get("reconciliations", []):
        if not isinstance(record, Mapping) or not record.get("reconciliation_id"):
            continue
        current = materialize_reconciliation(ledger, str(record["reconciliation_id"]))
        sign_off = current.get("sign_off")
        if (
            current.get("base_is_clean")
            and current.get("all_exceptions_closed")
            and isinstance(sign_off, Mapping)
            and sign_off.get("decision") == "approved"
        ):
            approved.add(str(current.get("wins_snapshot_id")))
    return approved


def _latest_wins_snapshot_id(snapshots: Sequence[Mapping[str, Any]]) -> str | None:
    wins = [snapshot for snapshot in snapshots if _is_wins_snapshot(snapshot)]
    if not wins:
        return None
    valid: list[tuple[datetime, Mapping[str, Any]]] = []
    for snapshot in wins:
        text = str(snapshot.get("observed_at") or "")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            observed = datetime.fromisoformat(text)
        except ValueError:
            continue
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        else:
            observed = observed.astimezone(timezone.utc)
        valid.append((observed, snapshot))
    if not valid:
        return None
    return str(max(valid, key=lambda item: item[0])[1].get("snapshot_id") or "") or None


def _binding(
    *, snapshot_id: str | None, requirements: Sequence[tuple[str, bool]]
) -> dict[str, Any]:
    blockers = [name for name, met in requirements if not met]
    return {
        "snapshot_id": snapshot_id,
        "allowed": not blockers,
        "blockers": blockers,
    }


def build_live_portfolio_pipeline(
    portfolio_snapshots: Sequence[Mapping[str, Any]],
    reconciliation_ledger: Mapping[str, Any] | None,
    *,
    mandate: Mapping[str, Any] | None,
    rulebook: Mapping[str, Any] | None,
    expected_return_assumptions: Mapping[str, Any] | None,
    now: datetime | str | None = None,
    max_age_seconds: float = 86_400,
    min_completeness_pct: float = 100.0,
) -> dict[str, Any]:
    """Select one canonical snapshot and create strict downstream bindings."""
    snapshots = [_json_copy(item, field="portfolio snapshot") for item in portfolio_snapshots]
    ledger = migrate_reconciliation_ledger(reconciliation_ledger)
    mandate_copy = _json_copy(mandate or {}, field="mandate")
    rulebook_copy = _json_copy(rulebook or {}, field="rulebook")
    returns_copy = _json_copy(
        expected_return_assumptions or {}, field="expected_return_assumptions"
    )
    approved_ids = _approved_reconciliation_snapshot_ids(ledger)
    authoritative = [
        snapshot
        for snapshot in snapshots
        if str(snapshot.get("snapshot_id")) in approved_ids and _is_wins_snapshot(snapshot)
    ]
    selected = select_reliable_snapshot(
        authoritative,
        now=now,
        max_age_seconds=max_age_seconds,
        min_completeness_pct=min_completeness_pct,
        provider_priority=("WInS", "wins"),
        allow_last_known_good=True,
    )
    authority = "wins_reconciled"
    if selected["snapshot"] is None:
        selected = select_reliable_snapshot(
            snapshots,
            now=now,
            max_age_seconds=max_age_seconds,
            min_completeness_pct=min_completeness_pct,
            provider_priority=("WInS", "wins", "tracker", "manual"),
            allow_last_known_good=True,
        )
        authority = "provisional"

    canonical = selected.get("snapshot")
    snapshot_id = str(canonical.get("snapshot_id")) if canonical else None
    latest_wins_id = _latest_wins_snapshot_id(snapshots)
    gate = reconciliation_readiness_gate(
        ledger,
        now=now,
        max_age_seconds=max_age_seconds,
        expected_wins_snapshot_id=latest_wins_id,
    )
    snapshot_available = canonical is not None
    snapshot_fresh = bool((selected.get("selection") or {}).get("is_fresh"))
    mandate_active = _context_active(mandate_copy)
    rulebook_active = _context_active(rulebook_copy)
    returns_active = _context_active(returns_copy)
    canonical_is_latest_clean = (
        snapshot_available
        and authority == "wins_reconciled"
        and gate.get("ready") is True
        and gate.get("wins_snapshot_id") == snapshot_id
        and latest_wins_id == snapshot_id
    )

    common = [
        ("portfolio_snapshot_unavailable", snapshot_available),
        ("active_mandate_missing", mandate_active),
        ("active_rulebook_missing", rulebook_active),
    ]
    bindings = {
        "tracker": _binding(
            snapshot_id=snapshot_id,
            requirements=[("portfolio_snapshot_unavailable", snapshot_available)],
        ),
        "quant": _binding(
            snapshot_id=snapshot_id,
            requirements=common + [("expected_return_assumptions_missing", returns_active)],
        ),
        "risk": _binding(snapshot_id=snapshot_id, requirements=common),
        "factors": _binding(snapshot_id=snapshot_id, requirements=common),
        "scenarios": _binding(snapshot_id=snapshot_id, requirements=common),
        "fx": _binding(snapshot_id=snapshot_id, requirements=common),
        "reporting": _binding(
            snapshot_id=snapshot_id,
            requirements=common
            + [
                ("expected_return_assumptions_missing", returns_active),
                ("snapshot_stale", snapshot_fresh),
                ("latest_wins_reconciliation_not_clean", canonical_is_latest_clean),
            ],
        ),
    }
    reason_codes = sorted(
        {blocker for binding in bindings.values() for blocker in binding["blockers"]}
    )
    status = (
        "blocked"
        if not snapshot_available
        else "ready"
        if bindings["reporting"]["allowed"]
        else "degraded"
    )
    return {
        "status": status,
        "authority": authority if snapshot_available else "none",
        "canonical_snapshot": canonical,
        "selection": selected.get("selection"),
        "last_known_good": bool(
            (selected.get("selection") or {}).get("used_last_known_good")
            or (
                snapshot_available
                and not canonical_is_latest_clean
                and authority == "wins_reconciled"
            )
        ),
        "latest_wins_snapshot_id": latest_wins_id,
        "reconciliation_gate": gate,
        "contexts": {
            "mandate": mandate_copy,
            "rulebook": rulebook_copy,
            "expected_return_assumptions": returns_copy,
        },
        "context_status": {
            "mandate_active": mandate_active,
            "rulebook_active": rulebook_active,
            "expected_return_assumptions_active": returns_active,
        },
        "consumer_bindings": bindings,
        "reason_codes": reason_codes,
    }


def materialize_consumer_input(
    pipeline: Mapping[str, Any],
    consumer: str,
) -> dict[str, Any]:
    """Materialise a consumer payload while preserving the shared snapshot ID."""
    name = str(consumer or "").strip().lower()
    if name not in ANALYSIS_CONSUMERS:
        raise ValueError(f"consumer must be one of {list(ANALYSIS_CONSUMERS)}.")
    bindings = pipeline.get("consumer_bindings")
    if not isinstance(bindings, Mapping) or name not in bindings:
        raise ValueError("pipeline does not contain consumer bindings.")
    binding = _json_copy(bindings[name], field="consumer binding")
    snapshot = pipeline.get("canonical_snapshot")
    contexts = pipeline.get("contexts") if isinstance(pipeline.get("contexts"), Mapping) else {}
    if snapshot and binding.get("snapshot_id") != snapshot.get("snapshot_id"):
        raise ValueError("Consumer binding does not match the canonical portfolio snapshot.")
    return {
        "consumer": name,
        "allowed": bool(binding.get("allowed")),
        "blockers": list(binding.get("blockers") or []),
        "portfolio_snapshot_id": binding.get("snapshot_id"),
        "portfolio_as_of": snapshot.get("observed_at") if snapshot else None,
        "portfolio_source": snapshot.get("source") if snapshot else None,
        "portfolio": _json_copy(snapshot.get("payload"), field="portfolio") if snapshot else None,
        "mandate": _json_copy(contexts.get("mandate", {}), field="mandate"),
        "rulebook": _json_copy(contexts.get("rulebook", {}), field="rulebook"),
        "expected_return_assumptions": _json_copy(
            contexts.get("expected_return_assumptions", {}),
            field="expected_return_assumptions",
        ),
        "reconciliation_gate": _json_copy(
            pipeline.get("reconciliation_gate", {}), field="reconciliation_gate"
        ),
    }


def snapshot_badges(
    snapshot: Mapping[str, Any],
    *,
    now: datetime | str | None = None,
    max_age_seconds: float = 86_400,
) -> dict[str, Any]:
    """Convenience badge payload for any portfolio/risk/report surface."""
    assessment = assess_snapshot(snapshot, now=now, max_age_seconds=max_age_seconds)
    return {
        "snapshot_id": assessment["snapshot_id"],
        "as_of": snapshot.get("observed_at"),
        "source": assessment["source_badge"],
        "freshness": assessment["freshness"],
        "age_seconds": assessment["age_seconds"],
        "completeness_pct": assessment["completeness_pct"],
        "integrity_valid": assessment["integrity_valid"],
        "status": "ready" if assessment["is_fresh"] and assessment["usable"] else "degraded",
    }


__all__ = [
    "ANALYSIS_CONSUMERS",
    "build_live_portfolio_pipeline",
    "create_portfolio_snapshot",
    "materialize_consumer_input",
    "normalize_portfolio_positions",
    "snapshot_badges",
]
