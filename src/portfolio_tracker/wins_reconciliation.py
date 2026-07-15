"""Pure helpers for reconciling WInS exports with tracked open positions.

The functions in this module deliberately perform no file I/O, persistence, or
UI work.  They accept ordinary mappings (for example records produced by a CSV
parser) and return JSON-serialisable dictionaries.

All monetary differences use the convention ``WInS - tracked``.  Cost means
total historical cost, while value means current market value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any


_TICKER_ALIASES = (
    "ticker",
    "symbol",
    "ticker symbol",
    "security symbol",
    "stock symbol",
    "instrument symbol",
)
_QUANTITY_ALIASES = (
    "quantity",
    "shares",
    "share quantity",
    "qty",
    "units",
    "position quantity",
)
_CURRENT_PRICE_ALIASES = (
    "current price",
    "market price",
    "last price",
    "last trade price",
    "closing price",
    "close price",
    "price",
)
_UNIT_COST_ALIASES = (
    "average cost",
    "average price",
    "avg cost",
    "avg price",
    "unit cost",
    "cost per share",
    "cost per unit",
    "entry price",
    "purchase price",
)
_TOTAL_COST_ALIASES = (
    "total cost",
    "total cost basis",
    "cost basis",
    "book value",
    "purchase value",
    "invested amount",
    "cost",
)
_CURRENT_VALUE_ALIASES = (
    "current value",
    "market value",
    "position value",
    "total value",
    "net value",
    "value",
)
_SECURITY_TYPE_ALIASES = (
    "security type",
    "asset type",
    "instrument type",
    "security category",
    "type",
)
_STATUS_ALIASES = ("status", "position status", "trade status")

_CLOSED_STATUSES = {
    "closed",
    "closed position",
    "exited",
    "liquidated",
    "sell",
    "sold",
}

_EQUITY_TYPES = {"common stock", "equity", "equities", "share", "shares", "stock"}
_ETF_TYPES = {"exchange traded fund", "exchange traded funds", "etf", "etfs"}
_BOND_TYPES = {"bond", "bonds", "fixed income", "fixed income security"}


def _canonical_key(value: Any) -> str:
    """Make common CSV/header spelling differences irrelevant."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().casefold()).strip()


def _normalised_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {_canonical_key(key): value for key, value in row.items()}


def _first(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        value = row.get(_canonical_key(alias))
        if value not in (None, ""):
            return value
    return None


def _number(value: Any) -> float | None:
    """Parse finite numbers commonly found in spreadsheet/CSV exports."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if not text or text.casefold() in {"-", "--", "n/a", "na", "none", "null"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = (
        text.replace("\u00a0", "")
        .replace(" ", "")
        .replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
    )
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if negative:
        number = -number
    return number if math.isfinite(number) else None


def _display_text(value: Any, default: str = "") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or default


def _ticker(value: Any) -> str:
    return _display_text(value).upper()


def _security_type_key(value: Any) -> str:
    key = _canonical_key(value)
    if key in _EQUITY_TYPES:
        return "equity"
    if key in _ETF_TYPES:
        return "etf"
    if key in _BOND_TYPES:
        return "bond"
    return key


def _row_sequence(rows: Any) -> list[dict[str, Any]]:
    """Accept either records or a mapping keyed by ticker without mutation."""
    if isinstance(rows, Mapping):
        normalised_keys = {_canonical_key(key) for key in rows}
        ticker_keys = {_canonical_key(alias) for alias in _TICKER_ALIASES}
        if normalised_keys & ticker_keys:
            return [dict(rows)]

        records: list[dict[str, Any]] = []
        for key, value in rows.items():
            if isinstance(value, Mapping):
                record = dict(value)
                if not ({_canonical_key(item) for item in record} & ticker_keys):
                    record["ticker"] = key
            else:
                record = {"ticker": key, "quantity": value}
            records.append(record)
        return records

    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _normalise_single_row(
    raw_row: Mapping[str, Any],
    *,
    tracked: bool,
) -> dict[str, Any] | None:
    row = _normalised_mapping(raw_row)
    ticker = _ticker(_first(row, _TICKER_ALIASES))
    if not ticker:
        return None

    quantity = _number(_first(row, _QUANTITY_ALIASES))
    current_price = _number(_first(row, _CURRENT_PRICE_ALIASES))
    current_value = _number(_first(row, _CURRENT_VALUE_ALIASES))

    # The app's generic portfolio manager stores ``cost_basis`` as a per-unit
    # amount, whereas a broker/WInS export normally uses it as a total.  Keep
    # those two input conventions separate.
    unit_cost_aliases = list(_UNIT_COST_ALIASES)
    total_cost_aliases = list(_TOTAL_COST_ALIASES)
    if tracked:
        unit_cost_aliases.extend(("cost basis", "cost_basis"))
        total_cost_aliases = [
            alias for alias in total_cost_aliases if _canonical_key(alias) != "cost basis"
        ]

    unit_cost = _number(_first(row, unit_cost_aliases))
    total_cost = _number(_first(row, total_cost_aliases))

    if current_value is None and quantity is not None and current_price is not None:
        current_value = quantity * current_price
    if current_price is None and current_value is not None and quantity not in (None, 0.0):
        current_price = current_value / quantity
    if total_cost is None and quantity is not None and unit_cost is not None:
        total_cost = quantity * unit_cost
    if unit_cost is None and total_cost is not None and quantity not in (None, 0.0):
        unit_cost = total_cost / quantity

    security_type = _display_text(_first(row, _SECURITY_TYPE_ALIASES), "Unknown")
    return {
        "ticker": ticker,
        "security_type": security_type,
        "quantity": quantity,
        "unit_cost": unit_cost,
        "total_cost": total_cost,
        "current_price": current_price,
        "current_value": current_value,
        "source_rows": 1,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["ticker"], []).append(row)

    result: list[dict[str, Any]] = []
    for ticker in sorted(grouped):
        group = grouped[ticker]

        def complete_total(field: str) -> float | None:
            values = [row[field] for row in group]
            if any(value is None for value in values):
                return None
            return sum(float(value) for value in values)

        quantity = complete_total("quantity")
        total_cost = complete_total("total_cost")
        current_value = complete_total("current_value")
        unit_cost = total_cost / quantity if total_cost is not None and quantity not in (None, 0) else None
        current_price = (
            current_value / quantity
            if current_value is not None and quantity not in (None, 0)
            else None
        )
        security_types_by_key: dict[str, str] = {}
        for row in group:
            key = _canonical_key(row["security_type"])
            if key not in {"", "unknown"}:
                security_types_by_key.setdefault(key, row["security_type"])
        security_types = [
            security_types_by_key[key] for key in sorted(security_types_by_key)
        ]
        security_type = (
            security_types[0]
            if len(security_types) == 1
            else "Mixed"
            if security_types
            else "Unknown"
        )
        result.append(
            {
                "ticker": ticker,
                "security_type": security_type,
                "quantity": quantity,
                "unit_cost": unit_cost,
                "total_cost": total_cost,
                "current_price": current_price,
                "current_value": current_value,
                "source_rows": len(group),
            }
        )
    return result


def normalize_wins_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalise and aggregate WInS/export position rows by ticker.

    Header matching is case- and punctuation-insensitive.  Rows without a
    ticker are ignored.  A field stays ``None`` when at least one duplicate row
    lacks enough data to calculate a complete aggregate; this prevents a
    partial monetary total from being presented as exact.
    """
    normalised: list[dict[str, Any]] = []
    for row in _row_sequence(rows):
        keyed = _normalised_mapping(row)
        status = _canonical_key(_first(keyed, _STATUS_ALIASES) or "open")
        if status in _CLOSED_STATUSES:
            continue
        item = _normalise_single_row(row, tracked=False)
        if item is not None:
            normalised.append(item)
    return _aggregate_rows(normalised)


def _normalize_tracked_open_positions(rows: Any) -> list[dict[str, Any]]:
    open_rows: list[dict[str, Any]] = []
    for row in _row_sequence(rows):
        keyed = _normalised_mapping(row)
        status = _canonical_key(_first(keyed, _STATUS_ALIASES) or "open")
        if status in _CLOSED_STATUSES:
            continue
        item = _normalise_single_row(row, tracked=True)
        if item is not None:
            open_rows.append(item)
    return _aggregate_rows(open_rows)


def _difference(wins_value: float | None, tracked_value: float | None) -> float | None:
    if wins_value is None or tracked_value is None:
        return None
    return wins_value - tracked_value


def _within(value: float | None, tolerance: float) -> bool | None:
    if value is None:
        return None
    return abs(value) <= tolerance


def _comparison_row(
    wins: Mapping[str, Any],
    tracked: Mapping[str, Any],
    *,
    quantity_tolerance: float,
    currency_tolerance: float,
) -> dict[str, Any]:
    differences = {
        "quantity": _difference(wins.get("quantity"), tracked.get("quantity")),
        "total_cost": _difference(wins.get("total_cost"), tracked.get("total_cost")),
        "current_value": _difference(wins.get("current_value"), tracked.get("current_value")),
    }
    field_matches = {
        "quantity": _within(differences["quantity"], quantity_tolerance),
        "total_cost": _within(differences["total_cost"], currency_tolerance),
        "current_value": _within(differences["current_value"], currency_tolerance),
    }
    wins_type = _security_type_key(wins.get("security_type"))
    tracked_type = _security_type_key(tracked.get("security_type"))
    security_type_match: bool | None = (
        None
        if wins_type in {"", "unknown"} or tracked_type in {"", "unknown"}
        else wins_type == tracked_type
    )
    comparable = list(field_matches.values()) + [security_type_match]
    if any(value is False for value in comparable):
        status = "difference"
    elif any(value is None for value in comparable):
        status = "partial"
    else:
        status = "matched"

    return {
        "ticker": wins["ticker"],
        "status": status,
        "wins": dict(wins),
        "tracked": dict(tracked),
        "quantity_difference": differences["quantity"],
        "cost_difference": differences["total_cost"],
        "value_difference": differences["current_value"],
        "differences": differences,
        "field_matches": field_matches,
        "security_type_match": security_type_match,
    }


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    for field in ("quantity", "total_cost", "current_value"):
        values = [row.get(field) for row in rows]
        totals[field] = (
            sum(float(value) for value in values)
            if all(value is not None for value in values)
            else None
        )
    return totals


def reconcile_wins_positions(
    wins_rows: Any,
    tracked_positions: Any,
    *,
    quantity_tolerance: float = 1e-8,
    currency_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Compare a WInS/export snapshot with the app's tracked open positions.

    ``matched`` contains every ticker present on both sides, including rows
    whose values differ. ``missing`` contains tracked positions absent from
    WInS, while ``extra`` contains WInS positions absent from the tracker.
    Coverage is the percentage of tracked tickers found in WInS.  The stricter
    two-way coverage uses the union of both ticker sets.
    """
    quantity_tolerance = max(0.0, float(quantity_tolerance))
    currency_tolerance = max(0.0, float(currency_tolerance))
    wins = normalize_wins_rows(wins_rows)
    tracked = _normalize_tracked_open_positions(tracked_positions)
    wins_by_ticker = {row["ticker"]: row for row in wins}
    tracked_by_ticker = {row["ticker"]: row for row in tracked}
    wins_tickers = set(wins_by_ticker)
    tracked_tickers = set(tracked_by_ticker)
    common_tickers = sorted(wins_tickers & tracked_tickers)

    matched = [
        _comparison_row(
            wins_by_ticker[ticker],
            tracked_by_ticker[ticker],
            quantity_tolerance=quantity_tolerance,
            currency_tolerance=currency_tolerance,
        )
        for ticker in common_tickers
    ]

    zero = {
        "quantity": 0.0,
        "total_cost": 0.0,
        "current_value": 0.0,
    }
    missing = []
    for ticker in sorted(tracked_tickers - wins_tickers):
        row = tracked_by_ticker[ticker]
        missing.append(
            {
                **row,
                "quantity_difference": _difference(0.0, row.get("quantity")),
                "cost_difference": _difference(0.0, row.get("total_cost")),
                "value_difference": _difference(0.0, row.get("current_value")),
            }
        )
    extra = []
    for ticker in sorted(wins_tickers - tracked_tickers):
        row = wins_by_ticker[ticker]
        extra.append(
            {
                **row,
                "quantity_difference": _difference(row.get("quantity"), zero["quantity"]),
                "cost_difference": _difference(row.get("total_cost"), zero["total_cost"]),
                "value_difference": _difference(row.get("current_value"), zero["current_value"]),
            }
        )

    wins_totals = _totals(wins)
    tracked_totals = _totals(tracked)
    total_differences = {
        field: _difference(wins_totals[field], tracked_totals[field])
        for field in ("quantity", "total_cost", "current_value")
    }

    tracker_count = len(tracked_tickers)
    union_count = len(wins_tickers | tracked_tickers)
    coverage_pct = 100.0 * len(common_tickers) / tracker_count if tracker_count else (
        100.0 if not wins_tickers else 0.0
    )
    two_way_coverage_pct = 100.0 * len(common_tickers) / union_count if union_count else 100.0

    differences = [row for row in matched if row["status"] == "difference"]
    partial = [row for row in matched if row["status"] == "partial"]
    if not wins and not tracked:
        status = "no_data"
    elif missing or extra or differences:
        status = "differences"
    elif partial:
        status = "partial"
    else:
        status = "reconciled"

    return {
        "status": status,
        "is_reconciled": status == "reconciled",
        "coverage_pct": coverage_pct,
        "two_way_coverage_pct": two_way_coverage_pct,
        "summary": {
            "wins_positions": len(wins),
            "tracked_open_positions": len(tracked),
            "matched_positions": len(matched),
            "exact_matches": sum(row["status"] == "matched" for row in matched),
            "mismatched_positions": len(differences),
            "partial_positions": len(partial),
            "missing_positions": len(missing),
            "extra_positions": len(extra),
            "coverage_pct": coverage_pct,
            "two_way_coverage_pct": two_way_coverage_pct,
        },
        "totals": {
            "wins": wins_totals,
            "tracked": tracked_totals,
            "difference": total_differences,
        },
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "wins_positions": wins,
        "tracked_open_positions": tracked,
    }


__all__ = ["normalize_wins_rows", "reconcile_wins_positions"]
