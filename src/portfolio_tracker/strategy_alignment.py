"""Deterministic client-mandate and strategy-alignment analytics.

The module deliberately has no UI, persistence, or network dependencies.  Its
public functions accept dictionaries/lists that can be edited directly by a
Streamlit form and return JSON-serialisable dictionaries for presentation.

Allocation conventions
----------------------
Goal and sector allocations use invested market value as their denominator.
Cash weight and position limits use total portfolio value.  This keeps a cash
reserve from making a fully allocated goal or sector plan look artificially
underweight while still checking liquidity constraints against the portfolio.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any


DEFAULT_SCORE_WEIGHTS = {
    "goal_allocation": 30.0,
    "sector_alignment": 20.0,
    "concentration": 20.0,
    "cash": 15.0,
    "holding_rules": 15.0,
}

_CASH_TICKERS = {"CASH", "CASH USD", "CASH_USD"}
_BAD_THESIS_STATUSES = {
    "broken",
    "closed",
    "invalid",
    "invalidated",
    "rejected",
    "sell",
}
_MISSING_THESIS_STATUSES = {
    "missing",
    "n/a",
    "none",
    "not available",
    "not_available",
    "unknown",
}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = _finite_number(value, math.nan)
    return number if math.isfinite(number) else None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"false", "no", "n", "off", "0"}:
            return False
        if text in {"true", "yes", "y", "on", "1"}:
            return True
    return bool(value)


def _weight(value: Any, default: float = 0.0, *, clamp: bool = True) -> float:
    """Return a fraction, forgiving common form inputs expressed as 0..100."""
    number = _finite_number(value, default)
    if abs(number) > 1.0:
        number /= 100.0
    if clamp:
        return min(1.0, max(0.0, number))
    return number


def _optional_weight(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return _weight(value)


def _normal_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _slug(value: Any, fallback: str) -> str:
    text = _normal_key(value)
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return slug or fallback


def _display_name(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text or fallback


def _string_list(value: Any, *, upper: bool = False, lower: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = re.split(r"[,;\n]", value)
    elif isinstance(value, Mapping):
        raw_values = [key for key, enabled in value.items() if bool(enabled)]
    elif _is_sequence(value):
        raw_values = list(value)
    else:
        raw_values = [value]

    cleaned: dict[str, str] = {}
    for item in raw_values:
        text = " ".join(str(item or "").strip().split())
        if not text:
            continue
        if upper:
            text = text.upper()
        elif lower:
            text = text.casefold()
        cleaned.setdefault(text.casefold(), text)
    return sorted(cleaned.values(), key=str.casefold)


def _mapping_rows(raw: Any, *, name_field: str) -> list[dict[str, Any]]:
    """Convert either editable mappings or sequences to independent row dicts."""
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault(name_field, key)
                row.setdefault("id", key)
            else:
                row = {name_field: key, "id": key, "target_weight": value}
            rows.append(row)
    elif _is_sequence(raw):
        for value in raw:
            if isinstance(value, Mapping):
                rows.append(dict(value))
            elif value is not None:
                rows.append({name_field: value})
    return rows


def _normalise_weighted_rows(
    raw: Any,
    *,
    name_field: str,
    kind: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _mapping_rows(raw, name_field=name_field)
    warnings: list[str] = []
    normalised: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        name = _display_name(
            row.get(name_field) or row.get("name") or row.get("id"),
            f"{kind.title()} {index}",
        )
        base_id = _slug(row.get("id") or name, f"{kind}-{index}")
        row_id = base_id
        suffix = 2
        while row_id in used_ids:
            row_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(row_id)

        target = _weight(row.get("target_weight", row.get("weight", 0.0)))
        minimum = _optional_weight(row.get("min_weight"))
        maximum = _optional_weight(row.get("max_weight"))
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum
            warnings.append(f"{name}: min_weight and max_weight were reversed.")

        item: dict[str, Any] = {
            "id": row_id,
            "name": name,
            "target_weight": target,
            "min_weight": minimum,
            "max_weight": maximum,
        }
        if kind == "sector":
            item["sector"] = name
        if kind == "goal":
            priority = int(round(_finite_number(row.get("priority"), 3.0)))
            item.update(
                {
                    "priority": min(5, max(1, priority)),
                    "horizon": str(row.get("horizon") or "").strip(),
                    "description": str(row.get("description") or "").strip(),
                    "allowed_sectors": _string_list(row.get("allowed_sectors")),
                    "allowed_asset_types": _string_list(row.get("allowed_asset_types"), lower=True),
                    "required_tags": _string_list(row.get("required_tags"), lower=True),
                    "max_position_weight": _optional_weight(row.get("max_position_weight")),
                    "aliases": _string_list(row.get("aliases")),
                }
            )
        normalised.append(item)

    total = sum(item["target_weight"] for item in normalised)
    if normalised and total <= 0:
        equal_weight = 1.0 / len(normalised)
        for item in normalised:
            item["target_weight"] = equal_weight
        warnings.append(f"{kind.title()} targets were empty; equal target weights were applied.")
    elif total > 0 and not math.isclose(total, 1.0, abs_tol=1e-9):
        scale = 1.0 / total
        for item in normalised:
            item["target_weight"] *= scale
            if item["min_weight"] is not None:
                item["min_weight"] = min(1.0, item["min_weight"] * scale)
            if item["max_weight"] is not None:
                item["max_weight"] = min(1.0, item["max_weight"] * scale)
        warnings.append(
            f"{kind.title()} target weights summed to {total:.2%} and were normalised to 100%."
        )
    return normalised, warnings


def normalize_client_mandate(mandate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a validated, stable client-mandate payload.

    ``goals`` accepts either a list of row dictionaries or a mapping such as
    ``{"Capital growth": 0.7, "Liquidity": 0.3}``.
    """
    raw = dict(mandate or {})
    goals, warnings = _normalise_weighted_rows(
        raw.get("goals", raw.get("goal_buckets", [])),
        name_field="name",
        kind="goal",
    )

    constraints_raw = raw.get("values_constraints", {})
    if isinstance(constraints_raw, Mapping):
        constraints = dict(constraints_raw)
        notes = _string_list(constraints.get("notes"))
    else:
        constraints = {}
        notes = _string_list(constraints_raw)

    liquidity = _weight(
        raw.get(
            "liquidity_need_pct",
            raw.get("liquidity_need_weight", constraints.get("min_cash_weight", 0.0)),
        )
    )
    horizon = max(0.0, _finite_number(raw.get("horizon_years"), 0.0))

    values_constraints = {
        "excluded_sectors": _string_list(
            constraints.get("excluded_sectors", constraints.get("restricted_sectors"))
        ),
        "excluded_tickers": _string_list(
            constraints.get(
                "excluded_tickers",
                constraints.get("prohibited_tickers", constraints.get("excluded_companies")),
            ),
            upper=True,
        ),
        "allowed_sectors": _string_list(constraints.get("allowed_sectors")),
        "allowed_asset_types": _string_list(constraints.get("allowed_asset_types"), lower=True),
        "required_tags": _string_list(constraints.get("required_tags"), lower=True),
        "notes": notes,
    }

    return {
        "client_name": _display_name(raw.get("client_name"), "Client"),
        "case_status": str(raw.get("case_status") or "draft").strip().casefold(),
        "risk_tolerance": str(raw.get("risk_tolerance") or "unspecified").strip().casefold(),
        "horizon_years": horizon,
        "liquidity_need_pct": liquidity,
        "base_currency": str(raw.get("base_currency") or "USD").strip().upper(),
        "values_constraints": values_constraints,
        "goals": goals,
        "normalization_warnings": warnings,
    }


def _intersect_or_choose(first: list[str], second: list[str]) -> list[str]:
    if not first:
        return list(second)
    if not second:
        return list(first)
    allowed = {item.casefold() for item in second}
    return [item for item in first if item.casefold() in allowed]


def normalize_strategy_rulebook(
    strategy: Mapping[str, Any] | None,
    mandate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalised strategy rulebook with effective mandate constraints."""
    raw = dict(strategy or {})
    normalised_mandate = normalize_client_mandate(mandate)
    mandate_constraints = normalised_mandate["values_constraints"]
    sector_targets, warnings = _normalise_weighted_rows(
        raw.get("sector_targets", []),
        name_field="sector",
        kind="sector",
    )

    configured: list[str] = []
    recognised_fields = (
        "max_position_weight",
        "max_sector_weight",
        "min_cash_weight",
        "max_cash_weight",
        "target_holdings",
        "min_holdings",
        "max_holdings",
        "max_turnover",
        "sector_targets",
        "max_goal_drift",
        "max_sector_drift",
        "allowed_sectors",
        "excluded_sectors",
        "prohibited_tickers",
        "allowed_asset_types",
        "required_tags",
        "require_approved",
        "min_beta",
        "max_beta",
    )
    configured.extend(field for field in recognised_fields if field in raw)
    if normalised_mandate["liquidity_need_pct"] > 0:
        configured.append("client_liquidity_need")
    if any(mandate_constraints[key] for key in mandate_constraints if key != "notes"):
        configured.append("client_values_constraints")

    min_cash = max(
        _weight(raw.get("min_cash_weight", 0.0)),
        normalised_mandate["liquidity_need_pct"],
    )
    max_cash = _weight(raw.get("max_cash_weight", 1.0))
    if max_cash < min_cash:
        max_cash = min_cash
        warnings.append("max_cash_weight was below the client liquidity floor and was raised.")

    target_holdings_raw = _optional_number(raw.get("target_holdings"))
    target_holdings = max(1, int(round(target_holdings_raw))) if target_holdings_raw else None
    min_holdings_raw = _optional_number(raw.get("min_holdings"))
    max_holdings_raw = _optional_number(raw.get("max_holdings"))
    min_holdings = max(0, int(round(min_holdings_raw))) if min_holdings_raw is not None else None
    max_holdings = max(0, int(round(max_holdings_raw))) if max_holdings_raw is not None else None
    if min_holdings is not None and max_holdings is not None and min_holdings > max_holdings:
        min_holdings, max_holdings = max_holdings, min_holdings
        warnings.append("min_holdings and max_holdings were reversed.")

    strategy_allowed_sectors = _string_list(raw.get("allowed_sectors"))
    strategy_allowed_assets = _string_list(raw.get("allowed_asset_types"), lower=True)
    allowed_sectors = _intersect_or_choose(
        strategy_allowed_sectors,
        mandate_constraints["allowed_sectors"],
    )
    allowed_asset_types = _intersect_or_choose(
        strategy_allowed_assets,
        mandate_constraints["allowed_asset_types"],
    )
    if strategy_allowed_sectors and mandate_constraints["allowed_sectors"] and not allowed_sectors:
        warnings.append("Strategy and client allowed-sector lists have no overlap.")
    if strategy_allowed_assets and mandate_constraints["allowed_asset_types"] and not allowed_asset_types:
        warnings.append("Strategy and client allowed-asset lists have no overlap.")

    excluded_sectors = _string_list(
        _string_list(raw.get("excluded_sectors")) + mandate_constraints["excluded_sectors"]
    )
    prohibited_tickers = _string_list(
        _string_list(raw.get("prohibited_tickers"), upper=True)
        + mandate_constraints["excluded_tickers"],
        upper=True,
    )
    required_tags = _string_list(
        _string_list(raw.get("required_tags"), lower=True)
        + mandate_constraints["required_tags"],
        lower=True,
    )

    score_weights_raw = raw.get("score_weights", DEFAULT_SCORE_WEIGHTS)
    score_weights: dict[str, float] = {}
    for component, default in DEFAULT_SCORE_WEIGHTS.items():
        value = default
        if isinstance(score_weights_raw, Mapping):
            value = max(0.0, _finite_number(score_weights_raw.get(component), default))
        score_weights[component] = value

    min_beta = _optional_number(raw.get("min_beta"))
    max_beta = _optional_number(raw.get("max_beta"))
    if min_beta is not None and max_beta is not None and min_beta > max_beta:
        min_beta, max_beta = max_beta, min_beta
        warnings.append("min_beta and max_beta were reversed.")

    return {
        "name": _display_name(raw.get("name"), "Investment strategy"),
        "thesis": str(raw.get("thesis") or "").strip(),
        "max_position_weight": _weight(raw.get("max_position_weight", 1.0)),
        "max_sector_weight": _weight(raw.get("max_sector_weight", 1.0)),
        "min_cash_weight": min_cash,
        "max_cash_weight": max_cash,
        "target_holdings": target_holdings,
        "min_holdings": min_holdings,
        "max_holdings": max_holdings,
        "max_turnover": _optional_weight(raw.get("max_turnover")),
        "current_turnover": _optional_weight(raw.get("current_turnover")),
        "max_goal_drift": _weight(raw.get("max_goal_drift", 0.10)),
        "max_sector_drift": _weight(raw.get("max_sector_drift", 0.10)),
        "sector_targets": sector_targets,
        "allowed_sectors": allowed_sectors,
        "excluded_sectors": excluded_sectors,
        "prohibited_tickers": prohibited_tickers,
        "allowed_asset_types": allowed_asset_types,
        "required_tags": required_tags,
        "require_approved": _as_bool(raw.get("require_approved"), False),
        "long_only": _as_bool(raw.get("long_only"), True),
        "min_beta": min_beta,
        "max_beta": max_beta,
        "score_weights": score_weights,
        "configured_rules": sorted(set(configured)),
        "normalization_warnings": warnings,
    }


def _holding_rows(holdings: Any) -> list[dict[str, Any]]:
    if isinstance(holdings, Mapping):
        rows: list[dict[str, Any]] = []
        for key, value in holdings.items():
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("ticker", key)
            else:
                row = {"ticker": key, "market_value": value}
            rows.append(row)
        return rows
    if _is_sequence(holdings):
        return [dict(item) for item in holdings if isinstance(item, Mapping)]
    return []


def _extract_value(row: Mapping[str, Any], portfolio_value: float | None) -> tuple[float, str]:
    for field in ("market_value", "current_value", "value", "position_value"):
        if row.get(field) not in (None, ""):
            return _finite_number(row.get(field)), field

    quantity = _optional_number(row.get("quantity"))
    if quantity is not None:
        for price_field in ("current_price", "last_price", "price", "entry_price"):
            price = _optional_number(row.get(price_field))
            if price is not None:
                return quantity * price, f"quantity*{price_field}"

    raw_weight = row.get("portfolio_weight", row.get("weight"))
    if raw_weight not in (None, "") and portfolio_value is not None:
        return _weight(raw_weight, clamp=False) * portfolio_value, "weight*portfolio_value"
    return 0.0, "missing"


def _goal_assignments(row: Mapping[str, Any]) -> list[tuple[str, float]]:
    raw = row.get("goals")
    if raw in (None, "", []):
        raw = row.get("primary_goal", row.get("goal", row.get("goal_bucket")))

    assignments: list[tuple[str, float]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                name = value.get("id") or value.get("name") or key
                weight = value.get("weight", value.get("allocation", 0.0))
            else:
                name, weight = key, value
            assignments.append((str(name), max(0.0, _finite_number(weight))))
    elif _is_sequence(raw):
        for value in raw:
            if isinstance(value, Mapping):
                name = value.get("id") or value.get("name") or value.get("goal")
                weight = value.get("weight", value.get("allocation", 1.0))
            else:
                name, weight = value, 1.0
            if name not in (None, ""):
                assignments.append((str(name), max(0.0, _finite_number(weight))))
    elif raw not in (None, ""):
        assignments.append((str(raw), 1.0))

    total = sum(weight for _, weight in assignments)
    if assignments and total <= 0:
        equal = 1.0 / len(assignments)
        return [(name, equal) for name, _ in assignments]
    return [(name, weight / total) for name, weight in assignments if weight > 0]


def _issue(
    code: str,
    message: str,
    *,
    severity: str = "medium",
    scope: str = "portfolio",
    subject: str = "Portfolio",
    actual: Any = None,
    limit: Any = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "scope": scope,
        "subject": subject,
        "message": message,
        "actual": actual,
        "limit": limit,
    }


def _check(code: str, passed: bool, message: str) -> dict[str, Any]:
    return {"code": code, "passed": bool(passed), "message": message}


def _allocation_score(rows: Sequence[Mapping[str, Any]]) -> float:
    total_absolute_drift = sum(abs(_finite_number(row.get("drift"))) for row in rows)
    return 100.0 * max(0.0, 1.0 - min(1.0, total_absolute_drift / 2.0))


def _range_score(actual: float, minimum: float, maximum: float) -> float:
    if minimum <= actual <= maximum:
        return 100.0
    if actual < minimum:
        return 100.0 * max(0.0, actual / minimum) if minimum > 0 else 0.0
    room = 1.0 - maximum
    return 100.0 * max(0.0, 1.0 - (actual - maximum) / room) if room > 0 else 0.0


def _rating(score: float) -> str:
    if score >= 90:
        return "Excellent alignment"
    if score >= 75:
        return "Strong alignment"
    if score >= 60:
        return "Partial alignment"
    if score >= 40:
        return "Weak alignment"
    return "Misaligned"


def _thesis_is_active(status: str) -> bool:
    return bool(status) and status not in _BAD_THESIS_STATUSES | _MISSING_THESIS_STATUSES


def _consolidate_prepared_holdings(
    prepared: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine open lots into one analytical position per ticker.

    Position limits, HHI, and holding counts describe securities rather than
    transaction lots.  Metadata is therefore combined conservatively: explicit
    approval conflicts resolve to ``False``, broken theses take precedence,
    tags are unioned, and beta/goal assignments are exposure weighted.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in prepared:
        grouped.setdefault(str(item["ticker"]), []).append(item)

    consolidated: list[dict[str, Any]] = []
    for ticker, lots in grouped.items():
        ordered = sorted(
            lots,
            key=lambda item: (
                str(item.get("_original", {}).get("id") or ""),
                int(item.get("_index", 0)),
            ),
        )
        first = ordered[0]
        gross_lot_value = sum(_finite_number(item.get("exposure_value")) for item in ordered)
        market_value = sum(_finite_number(item.get("market_value")) for item in ordered)

        def exposure_choice(field: str, fallback: str) -> str:
            candidates: dict[str, tuple[float, int, str]] = {}
            for item in ordered:
                value = _display_name(item.get(field), fallback)
                key = _normal_key(value)
                exposure, first_index, _ = candidates.get(key, (0.0, int(item.get("_index", 0)), value))
                candidates[key] = (
                    exposure + _finite_number(item.get("exposure_value")),
                    min(first_index, int(item.get("_index", 0))),
                    value,
                )
            return sorted(candidates.values(), key=lambda value: (-value[0], value[1], value[2].casefold()))[0][2]

        beta_weight = sum(
            _finite_number(item.get("exposure_value"))
            for item in ordered
            if item.get("beta") is not None
        )
        beta = (
            sum(
                _finite_number(item.get("exposure_value")) * _finite_number(item.get("beta"))
                for item in ordered
                if item.get("beta") is not None
            )
            / beta_weight
            if beta_weight > 0
            else None
        )

        approvals = [item.get("approved") for item in ordered if item.get("approved") is not None]
        approved = None if not approvals else all(bool(value) for value in approvals)

        statuses = [_normal_key(item.get("thesis_status")) for item in ordered]
        statuses = [status for status in statuses if status]
        bad_statuses = [status for status in statuses if status in _BAD_THESIS_STATUSES]
        active_statuses = [status for status in statuses if _thesis_is_active(status)]
        missing_statuses = [status for status in statuses if status in _MISSING_THESIS_STATUSES]
        thesis_status = (
            bad_statuses[0]
            if bad_statuses
            else active_statuses[0]
            if active_statuses
            else missing_statuses[0]
            if missing_statuses
            else ""
        )

        goal_values: dict[str, tuple[str, float]] = {}
        unassigned_goal_value = 0.0
        for item in ordered:
            exposure = _finite_number(item.get("exposure_value"))
            assignments = list(item.get("goal_assignments_raw") or [])
            if not assignments:
                unassigned_goal_value += exposure
                continue
            for goal_name, allocation in assignments:
                key = _normal_key(goal_name)
                display, value = goal_values.get(key, (str(goal_name), 0.0))
                goal_values[key] = (display, value + exposure * _finite_number(allocation))

        if gross_lot_value > 0:
            goal_assignments = [
                (display, value / gross_lot_value)
                for display, value in goal_values.values()
                if value > 0
            ]
            unassigned_goal_fraction = unassigned_goal_value / gross_lot_value
        else:
            goal_assignments = list(first.get("goal_assignments_raw") or [])
            unassigned_goal_fraction = 0.0 if goal_assignments else 1.0

        tags = sorted(
            {
                str(tag)
                for item in ordered
                for tag in item.get("tags", [])
                if str(tag).strip()
            },
            key=str.casefold,
        )
        original = dict(first.get("_original") or {})
        consolidated.append(
            {
                "_index": min(int(item.get("_index", 0)) for item in ordered),
                "_original": original,
                "ticker": ticker,
                "name": next(
                    (
                        str(item["name"])
                        for item in ordered
                        if _normal_key(item.get("name")) not in {"", _normal_key(ticker)}
                    ),
                    str(first.get("name") or ticker),
                ),
                "sector": exposure_choice("sector", "Unclassified"),
                "asset_type": exposure_choice("asset_type", "equity"),
                "market_value": market_value,
                "exposure_value": abs(market_value),
                "value_source": first.get("value_source") if len(ordered) == 1 else "consolidated_lots",
                "beta": beta,
                "thesis_status": thesis_status,
                "approved": approved,
                "tags": tags,
                "goal_assignments_raw": goal_assignments,
                "unassigned_goal_fraction": unassigned_goal_fraction,
                "lot_count": len(ordered),
            }
        )

    return sorted(consolidated, key=lambda item: (item["ticker"], item["_index"]))


def analyze_strategy_alignment(
    holdings: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    mandate: Mapping[str, Any] | None = None,
    strategy: Mapping[str, Any] | None = None,
    *,
    cash_value: float | None = None,
    portfolio_value: float | None = None,
) -> dict[str, Any]:
    """Evaluate holdings against an editable client mandate and rulebook.

    The result contains normalised input payloads, component scores, allocation
    tables, structured violations/warnings, and per-holding rule outcomes.
    No timestamp is included, so identical inputs produce identical output.
    """
    normalised_mandate = normalize_client_mandate(mandate)
    normalised_strategy = normalize_strategy_rulebook(strategy, normalised_mandate)
    rows = _holding_rows(holdings)
    requested_portfolio_value = _optional_number(portfolio_value)
    if requested_portfolio_value is not None:
        requested_portfolio_value = max(0.0, requested_portfolio_value)

    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for message in normalised_mandate["normalization_warnings"]:
        warnings.append(_issue("mandate_normalized", message, severity="low", scope="input"))
    for message in normalised_strategy["normalization_warnings"]:
        warnings.append(_issue("strategy_normalized", message, severity="low", scope="input"))

    prepared: list[dict[str, Any]] = []
    inferred_cash = 0.0
    for index, original in enumerate(rows):
        status = _normal_key(original.get("status") or "open")
        if status in {"closed", "sold", "inactive"}:
            continue
        ticker = str(original.get("ticker") or original.get("symbol") or f"POSITION-{index + 1}").strip().upper()
        asset_type = _normal_key(original.get("asset_type") or original.get("security_type") or "equity")
        value, value_source = _extract_value(original, requested_portfolio_value)
        is_cash = asset_type in {"cash", "currency", "money market", "money-market"} or ticker in _CASH_TICKERS
        if is_cash:
            inferred_cash += max(0.0, value)
            continue

        if value_source == "missing":
            warnings.append(
                _issue(
                    "missing_market_value",
                    f"{ticker} has no usable market value and is treated as zero.",
                    severity="medium",
                    scope="holding",
                    subject=ticker,
                )
            )
        if value < 0 and normalised_strategy["long_only"]:
            violations.append(
                _issue(
                    "short_position",
                    f"{ticker} has a negative market value in a long-only strategy.",
                    severity="high",
                    scope="holding",
                    subject=ticker,
                    actual=value,
                    limit=0.0,
                )
            )

        prepared.append(
            {
                "_index": index,
                "_original": dict(original),
                "ticker": ticker,
                "name": _display_name(original.get("name") or original.get("company_name"), ticker),
                "sector": _display_name(original.get("sector"), "Unclassified"),
                "asset_type": asset_type or "equity",
                "market_value": value,
                "exposure_value": abs(value),
                "value_source": value_source,
                "beta": _optional_number(original.get("beta")),
                "thesis_status": _normal_key(original.get("thesis_status")),
                "approved": (
                    _as_bool(original.get("approved"))
                    if original.get("approved") not in (None, "")
                    else None
                ),
                "tags": _string_list(original.get("tags"), lower=True),
                "goal_assignments_raw": _goal_assignments(original),
            }
        )

    prepared = _consolidate_prepared_holdings(prepared)
    invested_value = sum(item["exposure_value"] for item in prepared)

    explicit_cash = _optional_number(cash_value)
    if explicit_cash is not None:
        cash = max(0.0, explicit_cash)
        if inferred_cash and not math.isclose(cash, inferred_cash, rel_tol=1e-6, abs_tol=1e-6):
            warnings.append(
                _issue(
                    "cash_override",
                    "Explicit cash_value overrides cash rows in holdings.",
                    severity="low",
                    scope="input",
                    actual=cash,
                    limit=inferred_cash,
                )
            )
    elif inferred_cash:
        cash = inferred_cash
    elif requested_portfolio_value is not None:
        cash = max(0.0, requested_portfolio_value - invested_value)
    else:
        cash = 0.0

    if requested_portfolio_value is not None:
        total_value = requested_portfolio_value
        accounted = invested_value + cash
        tolerance = max(1e-6, total_value * 0.001)
        if accounted > total_value + tolerance:
            warnings.append(
                _issue(
                    "portfolio_value_mismatch",
                    "Holdings plus cash exceed the supplied portfolio value.",
                    severity="medium",
                    scope="input",
                    actual=accounted,
                    limit=total_value,
                )
            )
    else:
        total_value = invested_value + cash
    unallocated_value = max(0.0, total_value - invested_value - cash)

    cash_weight = cash / total_value if total_value > 0 else 0.0
    for item in prepared:
        item["invested_weight"] = item["exposure_value"] / invested_value if invested_value > 0 else 0.0
        item["portfolio_weight"] = item["exposure_value"] / total_value if total_value > 0 else 0.0

    goal_definitions = normalised_mandate["goals"]
    goal_lookup: dict[str, dict[str, Any]] = {}
    for goal in goal_definitions:
        for alias in [goal["id"], goal["name"], *goal["aliases"]]:
            goal_lookup[_normal_key(alias)] = goal

    goal_values = {goal["id"]: 0.0 for goal in goal_definitions}
    goal_counts = {goal["id"]: 0 for goal in goal_definitions}
    unassigned_goal_value = 0.0
    for item in prepared:
        resolved: list[dict[str, Any]] = []
        unknown_names: list[str] = []
        for goal_name, allocation in item["goal_assignments_raw"]:
            goal = goal_lookup.get(_normal_key(goal_name))
            if goal is None:
                unknown_names.append(goal_name)
                unassigned_goal_value += item["exposure_value"] * allocation
            else:
                goal_values[goal["id"]] += item["exposure_value"] * allocation
                goal_counts[goal["id"]] += 1
                resolved.append(
                    {
                        "id": goal["id"],
                        "name": goal["name"],
                        "allocation": allocation,
                    }
                )
        if goal_definitions:
            unassigned_goal_value += item["exposure_value"] * _finite_number(
                item.get(
                    "unassigned_goal_fraction",
                    0.0 if item["goal_assignments_raw"] else 1.0,
                )
            )
        item["goals"] = resolved
        item["unknown_goals"] = unknown_names

    goal_allocation: list[dict[str, Any]] = []
    goal_tolerance = normalised_strategy["max_goal_drift"]
    for goal in goal_definitions:
        actual = goal_values[goal["id"]] / invested_value if invested_value > 0 else 0.0
        target = goal["target_weight"]
        drift = actual - target
        minimum = goal["min_weight"]
        maximum = goal["max_weight"]
        below = actual < minimum - 1e-12 if minimum is not None else drift < -goal_tolerance - 1e-12
        above = actual > maximum + 1e-12 if maximum is not None else drift > goal_tolerance + 1e-12
        status = "underweight" if below else "overweight" if above else "aligned"
        allocation_row = {
            "goal_id": goal["id"],
            "goal_name": goal["name"],
            "priority": goal["priority"],
            "market_value": goal_values[goal["id"]],
            "holding_count": goal_counts[goal["id"]],
            "target_weight": target,
            "actual_weight": actual,
            "drift": drift,
            "abs_drift": abs(drift),
            "status": status,
        }
        goal_allocation.append(allocation_row)
        if status != "aligned":
            violations.append(
                _issue(
                    f"goal_{status}",
                    f"{goal['name']} is {abs(drift):.1%} {status} versus its target.",
                    severity="medium",
                    scope="goal",
                    subject=goal["name"],
                    actual=actual,
                    limit={"target": target, "tolerance": goal_tolerance},
                )
            )
    if goal_definitions and unassigned_goal_value > 1e-12:
        actual = unassigned_goal_value / invested_value if invested_value > 0 else 0.0
        goal_allocation.append(
            {
                "goal_id": "unassigned",
                "goal_name": "Unassigned",
                "priority": None,
                "market_value": unassigned_goal_value,
                "holding_count": sum(
                    1 for item in prepared if not item["goals"] or item["unknown_goals"]
                ),
                "target_weight": 0.0,
                "actual_weight": actual,
                "drift": actual,
                "abs_drift": actual,
                "status": "unassigned",
            }
        )
        violations.append(
            _issue(
                "unassigned_goal_exposure",
                f"{actual:.1%} of invested value is not mapped to a valid client goal.",
                severity="high",
                scope="goal",
                subject="Unassigned",
                actual=actual,
                limit=0.0,
            )
        )

    sector_values: dict[str, float] = {}
    sector_names: dict[str, str] = {}
    sector_counts: dict[str, int] = {}
    for item in prepared:
        key = _normal_key(item["sector"])
        sector_names.setdefault(key, item["sector"])
        sector_values[key] = sector_values.get(key, 0.0) + item["exposure_value"]
        sector_counts[key] = sector_counts.get(key, 0) + 1

    sector_targets = normalised_strategy["sector_targets"]
    target_by_key = {_normal_key(item["name"]): item for item in sector_targets}
    sector_keys = list(target_by_key)
    sector_keys.extend(sorted((key for key in sector_values if key not in target_by_key), key=str.casefold))
    sector_allocation: list[dict[str, Any]] = []
    sector_tolerance = normalised_strategy["max_sector_drift"]
    for key in sector_keys:
        definition = target_by_key.get(key)
        name = definition["name"] if definition else sector_names.get(key, "Unclassified")
        actual = sector_values.get(key, 0.0) / invested_value if invested_value > 0 else 0.0
        target = definition["target_weight"] if definition else (0.0 if sector_targets else None)
        drift = actual - target if target is not None else None
        minimum = definition["min_weight"] if definition else None
        maximum = definition["max_weight"] if definition else None
        lower = minimum if minimum is not None else max(0.0, target - sector_tolerance) if target is not None else 0.0
        upper = maximum if maximum is not None else min(1.0, target + sector_tolerance) if target is not None else 1.0
        upper = min(upper, normalised_strategy["max_sector_weight"])
        status = "underweight" if actual < lower - 1e-12 else "overweight" if actual > upper + 1e-12 else "aligned"
        sector_allocation.append(
            {
                "sector": name,
                "market_value": sector_values.get(key, 0.0),
                "holding_count": sector_counts.get(key, 0),
                "target_weight": target,
                "actual_weight": actual,
                "drift": drift,
                "abs_drift": abs(drift) if drift is not None else None,
                "min_weight": lower,
                "max_weight": upper,
                "status": status,
            }
        )
        if status != "aligned":
            violations.append(
                _issue(
                    f"sector_{status}",
                    f"{name} is outside its permitted allocation range.",
                    severity="medium",
                    scope="sector",
                    subject=name,
                    actual=actual,
                    limit={"min": lower, "max": upper},
                )
            )

    holding_results: list[dict[str, Any]] = []
    max_position = normalised_strategy["max_position_weight"]
    max_position_configured = "max_position_weight" in normalised_strategy["configured_rules"]
    allowed_sector_keys = {_normal_key(item) for item in normalised_strategy["allowed_sectors"]}
    excluded_sector_keys = {_normal_key(item) for item in normalised_strategy["excluded_sectors"]}
    allowed_assets = {_normal_key(item) for item in normalised_strategy["allowed_asset_types"]}
    prohibited_tickers = set(normalised_strategy["prohibited_tickers"])
    required_tags = set(normalised_strategy["required_tags"])

    for item in prepared:
        checks: list[dict[str, Any]] = []
        ticker = item["ticker"]
        sector_key = _normal_key(item["sector"])
        holding_tags = set(item["tags"])

        if normalised_strategy["long_only"]:
            long_only_ok = item["market_value"] >= 0
            checks.append(_check("long_only", long_only_ok, "Position has non-negative long exposure."))

        if goal_definitions:
            known_goals = bool(item["goals"]) and not item["unknown_goals"]
            checks.append(_check("valid_goal", known_goals, "Position is mapped only to recognised client goals."))
            if not known_goals:
                violations.append(
                    _issue(
                        "holding_goal_missing",
                        f"{ticker} is missing a valid client-goal assignment.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                    )
                )

        sector_allowed = sector_key not in excluded_sector_keys and (
            not allowed_sector_keys or sector_key in allowed_sector_keys
        )
        if excluded_sector_keys or allowed_sector_keys:
            checks.append(_check("sector_allowed", sector_allowed, "Sector complies with strategy and client values."))
            if not sector_allowed:
                violations.append(
                    _issue(
                        "holding_sector_restricted",
                        f"{ticker} is in restricted sector {item['sector']}.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                        actual=item["sector"],
                        limit=normalised_strategy["allowed_sectors"],
                    )
                )

        ticker_allowed = ticker not in prohibited_tickers
        if prohibited_tickers:
            checks.append(_check("ticker_allowed", ticker_allowed, "Security is not prohibited."))
            if not ticker_allowed:
                violations.append(
                    _issue(
                        "holding_prohibited",
                        f"{ticker} is prohibited by the mandate or strategy.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                    )
                )

        asset_allowed = not allowed_assets or item["asset_type"] in allowed_assets
        if allowed_assets:
            checks.append(_check("asset_type_allowed", asset_allowed, "Asset type is permitted."))
            if not asset_allowed:
                violations.append(
                    _issue(
                        "holding_asset_type_restricted",
                        f"{ticker} uses restricted asset type {item['asset_type']}.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                        actual=item["asset_type"],
                        limit=normalised_strategy["allowed_asset_types"],
                    )
                )

        tags_ok = required_tags.issubset(holding_tags)
        if required_tags:
            checks.append(_check("required_tags", tags_ok, "Position contains all required strategy tags."))
            if not tags_ok:
                missing_tags = sorted(required_tags - holding_tags)
                violations.append(
                    _issue(
                        "holding_tags_missing",
                        f"{ticker} is missing required tags: {', '.join(missing_tags)}.",
                        severity="medium",
                        scope="holding",
                        subject=ticker,
                        actual=item["tags"],
                        limit=sorted(required_tags),
                    )
                )

        position_ok = item["portfolio_weight"] <= max_position + 1e-12
        if max_position_configured:
            checks.append(_check("position_limit", position_ok, "Position is within the maximum weight."))
            if not position_ok:
                violations.append(
                    _issue(
                        "position_limit_exceeded",
                        f"{ticker} exceeds the {max_position:.1%} position limit.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                        actual=item["portfolio_weight"],
                        limit=max_position,
                    )
                )
            elif max_position > 0 and item["portfolio_weight"] >= max_position * 0.9:
                warnings.append(
                    _issue(
                        "position_near_limit",
                        f"{ticker} is within 10% of its position limit.",
                        severity="low",
                        scope="holding",
                        subject=ticker,
                        actual=item["portfolio_weight"],
                        limit=max_position,
                    )
                )

        approved_value = item["approved"]
        if normalised_strategy["require_approved"]:
            approved = bool(approved_value)
            checks.append(_check("approved", approved, "Position has strategy approval."))
            if not approved:
                violations.append(
                    _issue(
                        "holding_not_approved",
                        f"{ticker} has not been approved under the strategy process.",
                        severity="high",
                        scope="holding",
                        subject=ticker,
                    )
                )

        if item["thesis_status"]:
            thesis_ok = _thesis_is_active(item["thesis_status"])
            checks.append(_check("thesis_active", thesis_ok, "Investment thesis remains active."))
            if not thesis_ok:
                thesis_missing = item["thesis_status"] in _MISSING_THESIS_STATUSES
                violations.append(
                    _issue(
                        "holding_thesis_missing" if thesis_missing else "holding_thesis_invalidated",
                        f"{ticker} has thesis status '{item['thesis_status']}'.",
                        severity="medium" if thesis_missing else "high",
                        scope="holding",
                        subject=ticker,
                        actual=item["thesis_status"],
                        limit="active thesis",
                    )
                )

        for goal_link in item["goals"]:
            goal = next(goal for goal in goal_definitions if goal["id"] == goal_link["id"])
            goal_allowed_sectors = {_normal_key(value) for value in goal["allowed_sectors"]}
            if goal_allowed_sectors:
                passed = sector_key in goal_allowed_sectors
                checks.append(_check(f"goal_sector:{goal['id']}", passed, f"Sector fits {goal['name']}."))
                if not passed:
                    violations.append(
                        _issue(
                            "goal_sector_mismatch",
                            f"{ticker}'s sector does not fit goal {goal['name']}.",
                            severity="medium",
                            scope="holding",
                            subject=ticker,
                            actual=item["sector"],
                            limit=goal["allowed_sectors"],
                        )
                    )
            goal_required_tags = set(goal["required_tags"])
            if goal_required_tags:
                passed = goal_required_tags.issubset(holding_tags)
                checks.append(_check(f"goal_tags:{goal['id']}", passed, f"Tags fit {goal['name']}."))
                if not passed:
                    violations.append(
                        _issue(
                            "goal_tags_missing",
                            f"{ticker} is missing tags required by goal {goal['name']}.",
                            severity="medium",
                            scope="holding",
                            subject=ticker,
                            actual=item["tags"],
                            limit=sorted(goal_required_tags),
                        )
                    )
            goal_allowed_assets = {_normal_key(value) for value in goal["allowed_asset_types"]}
            if goal_allowed_assets:
                passed = item["asset_type"] in goal_allowed_assets
                checks.append(_check(f"goal_asset:{goal['id']}", passed, f"Asset type fits {goal['name']}."))
                if not passed:
                    violations.append(
                        _issue(
                            "goal_asset_type_mismatch",
                            f"{ticker}'s asset type does not fit goal {goal['name']}.",
                            severity="medium",
                            scope="holding",
                            subject=ticker,
                            actual=item["asset_type"],
                            limit=goal["allowed_asset_types"],
                        )
                    )
            if goal["max_position_weight"] is not None:
                passed = item["portfolio_weight"] <= goal["max_position_weight"] + 1e-12
                checks.append(_check(f"goal_position:{goal['id']}", passed, f"Position size fits {goal['name']}."))
                if not passed:
                    violations.append(
                        _issue(
                            "goal_position_limit_exceeded",
                            f"{ticker} exceeds the position limit for goal {goal['name']}.",
                            severity="high",
                            scope="holding",
                            subject=ticker,
                            actual=item["portfolio_weight"],
                            limit=goal["max_position_weight"],
                        )
                    )

        passed_count = sum(1 for check in checks if check["passed"])
        alignment = 100.0 * passed_count / len(checks) if checks else 100.0
        failed = [check for check in checks if not check["passed"]]
        holding_results.append(
            {
                "ticker": ticker,
                "name": item["name"],
                "sector": item["sector"],
                "asset_type": item["asset_type"],
                "market_value": item["market_value"],
                "exposure_value": item["exposure_value"],
                "portfolio_weight": item["portfolio_weight"],
                "invested_weight": item["invested_weight"],
                "beta": item["beta"],
                "thesis_status": item["thesis_status"] or None,
                "approved": item["approved"],
                "lot_count": item["lot_count"],
                "goals": item["goals"],
                "unknown_goals": item["unknown_goals"],
                "alignment_score": alignment,
                "status": "aligned" if not failed else "review" if alignment >= 60 else "misaligned",
                "checks": checks,
                "failed_rules": [check["code"] for check in failed],
            }
        )

    hhi = sum(item["invested_weight"] ** 2 for item in prepared)
    effective_holdings = 1.0 / hhi if hhi > 0 else 0.0
    largest_position_weight = max((item["portfolio_weight"] for item in prepared), default=0.0)
    largest_invested_weight = max((item["invested_weight"] for item in prepared), default=0.0)
    weighted_beta_numerator = sum(
        item["invested_weight"] * item["beta"] for item in prepared if item["beta"] is not None
    )
    beta_coverage = sum(item["invested_weight"] for item in prepared if item["beta"] is not None)
    weighted_beta = weighted_beta_numerator / beta_coverage if beta_coverage > 0 else None
    goal_assignment_coverage = (
        max(0.0, 1.0 - unassigned_goal_value / invested_value)
        if goal_definitions and invested_value > 0
        else None
    )
    classified_sector_coverage = (
        sum(
            item["invested_weight"]
            for item in prepared
            if _normal_key(item["sector"]) not in {"", "unclassified", "unknown", "n/a"}
        )
        if invested_value > 0
        else 0.0
    )
    approval_data_coverage = (
        sum(item["invested_weight"] for item in prepared if item["approved"] is not None)
        if invested_value > 0
        else 0.0
    )
    active_thesis_coverage = (
        sum(
            item["invested_weight"]
            for item in prepared
            if _thesis_is_active(item["thesis_status"])
        )
        if invested_value > 0
        else 0.0
    )

    position_count = len([item for item in prepared if item["exposure_value"] > 0])
    min_holdings = normalised_strategy["min_holdings"]
    max_holdings = normalised_strategy["max_holdings"]
    target_holdings = normalised_strategy["target_holdings"]
    if min_holdings is not None and position_count < min_holdings:
        violations.append(
            _issue(
                "too_few_holdings",
                f"Portfolio has {position_count} holdings; minimum is {min_holdings}.",
                severity="medium",
                actual=position_count,
                limit=min_holdings,
            )
        )
    if max_holdings is not None and position_count > max_holdings:
        violations.append(
            _issue(
                "too_many_holdings",
                f"Portfolio has {position_count} holdings; maximum is {max_holdings}.",
                severity="medium",
                actual=position_count,
                limit=max_holdings,
            )
        )

    min_beta = normalised_strategy["min_beta"]
    max_beta = normalised_strategy["max_beta"]
    portfolio_rule_checks: list[dict[str, Any]] = []
    if (min_beta is not None or max_beta is not None) and weighted_beta is None:
        warnings.append(
            _issue(
                "beta_not_available",
                "A portfolio beta rule is configured, but no holding beta data is available.",
                severity="low",
                scope="input",
            )
        )
    elif (min_beta is not None or max_beta is not None) and beta_coverage < 0.90:
        warnings.append(
            _issue(
                "beta_coverage_low",
                f"Portfolio beta covers only {beta_coverage:.1%} of invested value.",
                severity="medium",
                scope="input",
                actual=beta_coverage,
                limit=0.90,
            )
        )
    if weighted_beta is not None and min_beta is not None and weighted_beta < min_beta:
        portfolio_rule_checks.append(
            _check("portfolio_beta_minimum", False, "Portfolio beta meets the strategy minimum.")
        )
        violations.append(
            _issue(
                "portfolio_beta_below_minimum",
                "Weighted portfolio beta is below the strategy minimum.",
                severity="medium",
                actual=weighted_beta,
                limit=min_beta,
            )
        )
    elif weighted_beta is not None and min_beta is not None:
        portfolio_rule_checks.append(
            _check("portfolio_beta_minimum", True, "Portfolio beta meets the strategy minimum.")
        )
    if weighted_beta is not None and max_beta is not None and weighted_beta > max_beta:
        portfolio_rule_checks.append(
            _check("portfolio_beta_maximum", False, "Portfolio beta meets the strategy maximum.")
        )
        violations.append(
            _issue(
                "portfolio_beta_above_maximum",
                "Weighted portfolio beta exceeds the strategy maximum.",
                severity="medium",
                actual=weighted_beta,
                limit=max_beta,
            )
        )
    elif weighted_beta is not None and max_beta is not None:
        portfolio_rule_checks.append(
            _check("portfolio_beta_maximum", True, "Portfolio beta meets the strategy maximum.")
        )

    max_turnover = normalised_strategy["max_turnover"]
    current_turnover = normalised_strategy["current_turnover"]
    if max_turnover is not None:
        if current_turnover is None:
            warnings.append(
                _issue(
                    "turnover_not_available",
                    "A turnover limit is configured, but current_turnover was not supplied.",
                    severity="low",
                    scope="input",
                    limit=max_turnover,
                )
            )
        elif current_turnover > max_turnover + 1e-12:
            portfolio_rule_checks.append(
                _check("turnover_limit", False, "Portfolio turnover is within the strategy limit.")
            )
            violations.append(
                _issue(
                    "turnover_limit_exceeded",
                    "Current turnover exceeds the strategy limit.",
                    severity="medium",
                    actual=current_turnover,
                    limit=max_turnover,
                )
            )
        else:
            portfolio_rule_checks.append(
                _check("turnover_limit", True, "Portfolio turnover is within the strategy limit.")
            )

    min_cash = normalised_strategy["min_cash_weight"]
    max_cash = normalised_strategy["max_cash_weight"]
    cash_ok = min_cash - 1e-12 <= cash_weight <= max_cash + 1e-12
    if not cash_ok:
        violations.append(
            _issue(
                "cash_outside_range",
                f"Cash weight {cash_weight:.1%} is outside the {min_cash:.1%}-{max_cash:.1%} range.",
                severity="high" if cash_weight < min_cash else "medium",
                actual=cash_weight,
                limit={"min": min_cash, "max": max_cash},
            )
        )

    if goal_definitions:
        goal_score = _allocation_score(goal_allocation) if invested_value > 0 else 0.0
    else:
        goal_score = 100.0
    if sector_targets:
        sector_score = _allocation_score(sector_allocation) if invested_value > 0 else 0.0
    else:
        sector_excess = sum(
            max(0.0, row["actual_weight"] - normalised_strategy["max_sector_weight"])
            for row in sector_allocation
        )
        sector_score = 100.0 * max(0.0, 1.0 - sector_excess)

    concentration_scores: list[float] = []
    if max_position_configured:
        excess = sum(max(0.0, item["portfolio_weight"] - max_position) for item in prepared)
        denominator = max(max_position, 1e-12)
        concentration_scores.append(100.0 * max(0.0, 1.0 - excess / denominator))
    if target_holdings:
        concentration_scores.append(100.0 * min(1.0, effective_holdings / target_holdings))
    if min_holdings is not None and min_holdings > 0:
        concentration_scores.append(100.0 * min(1.0, position_count / min_holdings))
    if max_holdings is not None and position_count > max_holdings:
        concentration_scores.append(100.0 * max_holdings / position_count)
    concentration_score = sum(concentration_scores) / len(concentration_scores) if concentration_scores else 100.0
    cash_score = _range_score(cash_weight, min_cash, max_cash)
    holding_rule_scores: list[float] = []
    if holding_results and invested_value > 0:
        holding_rule_scores.append(sum(
            row["alignment_score"] * row["invested_weight"] for row in holding_results
        ))
    if portfolio_rule_checks:
        holding_rule_scores.append(
            100.0
            * sum(1 for check in portfolio_rule_checks if check["passed"])
            / len(portfolio_rule_checks)
        )
    holding_score = sum(holding_rule_scores) / len(holding_rule_scores) if holding_rule_scores else 100.0

    component_applicability = {
        "goal_allocation": bool(goal_definitions),
        "sector_alignment": bool(sector_targets)
        or "max_sector_weight" in normalised_strategy["configured_rules"],
        "concentration": bool(concentration_scores),
        "cash": bool(
            {"min_cash_weight", "max_cash_weight", "client_liquidity_need"}
            & set(normalised_strategy["configured_rules"])
        ),
        "holding_rules": any(row["checks"] for row in holding_results) or bool(portfolio_rule_checks),
    }
    raw_component_scores = {
        "goal_allocation": goal_score,
        "sector_alignment": sector_score,
        "concentration": concentration_score,
        "cash": cash_score,
        "holding_rules": holding_score,
    }
    components: dict[str, dict[str, Any]] = {}
    applicable_weight = sum(
        normalised_strategy["score_weights"][name]
        for name, applies in component_applicability.items()
        if applies
    )
    configured_weight_total = sum(normalised_strategy["score_weights"].values())
    scoring_coverage = (
        applicable_weight / configured_weight_total if configured_weight_total > 0 else 0.0
    )
    weighted_score = 0.0
    for name, score in raw_component_scores.items():
        applies = component_applicability[name]
        configured_weight = normalised_strategy["score_weights"][name]
        effective_weight = configured_weight / applicable_weight if applies and applicable_weight > 0 else 0.0
        components[name] = {
            "score": score,
            "applicable": applies,
            "configured_weight": configured_weight,
            "effective_weight": effective_weight,
        }
        if name == "holding_rules":
            components[name]["portfolio_checks"] = portfolio_rule_checks
        weighted_score += score * effective_weight
    alignment_score = weighted_score if applicable_weight > 0 else 100.0

    severity_order = {"high": 0, "medium": 1, "low": 2}
    violations.sort(
        key=lambda item: (
            severity_order.get(item["severity"], 9),
            item["scope"],
            item["code"],
            item["subject"],
        )
    )
    warnings.sort(
        key=lambda item: (
            severity_order.get(item["severity"], 9),
            item["scope"],
            item["code"],
            item["subject"],
        )
    )

    return {
        "alignment_score": alignment_score,
        "score": alignment_score,
        "rating": _rating(alignment_score),
        "scoring_coverage": scoring_coverage,
        "mandate": normalised_mandate,
        "strategy": normalised_strategy,
        "portfolio_summary": {
            "total_value": total_value,
            "invested_value": invested_value,
            "cash_value": cash,
            "unallocated_value": unallocated_value,
            "cash_weight": cash_weight,
            "position_count": position_count,
            "concentration_hhi": hhi,
            "effective_holdings": effective_holdings,
            "largest_position_weight": largest_position_weight,
            "largest_invested_weight": largest_invested_weight,
            "weighted_beta": weighted_beta,
            "beta_coverage": beta_coverage,
            "goal_assignment_coverage": goal_assignment_coverage,
            "sector_classification_coverage": classified_sector_coverage,
            "approval_data_coverage": approval_data_coverage,
            "active_thesis_coverage": active_thesis_coverage,
        },
        "goal_allocation_basis": "invested_value",
        "goal_allocation": goal_allocation,
        "sector_allocation_basis": "invested_value",
        "sector_allocation": sector_allocation,
        "holdings": holding_results,
        "portfolio_rule_checks": portfolio_rule_checks,
        "components": components,
        "violations": violations,
        "warnings": warnings,
        "violation_count": len(violations),
        "warning_count": len(warnings),
    }


def analyze_portfolio_alignment(
    holdings: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    mandate: Mapping[str, Any] | None = None,
    strategy: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-friendly alias for :func:`analyze_strategy_alignment`."""
    return analyze_strategy_alignment(holdings, mandate, strategy, **kwargs)


__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "analyze_portfolio_alignment",
    "analyze_strategy_alignment",
    "normalize_client_mandate",
    "normalize_strategy_rulebook",
]
