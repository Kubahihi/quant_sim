"""Pure pre-trade simulation and strategy-impact analytics.

The functions in this module deliberately perform no persistence, network, or
UI work.  Prices, competition-position lots, strategy metadata, and proposed
trades are supplied by the caller.  A trade plan is evaluated atomically: when
any blocker is present, the public ``after_positions`` state remains identical
to the input state while ``projected_positions`` retains the partial sequential
simulation for diagnostics.

Turnover convention
-------------------
``incremental_turnover`` is gross proposed trade notional divided by pre-trade
portfolio equity.  It is not the one-way/half-turnover convention used by some
institutional reports, and it is kept separate from historical turnover unless
the caller explicitly supplies a historical ``current_turnover`` value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from .strategy_alignment import analyze_strategy_alignment
from .wharton_competition import (
    INITIAL_CAPITAL_USD,
    calculate_portfolio_performance,
)


TURNOVER_DEFINITION = "gross proposed trade notional / pre-trade equity"
_BUY_ACTIONS = {"buy", "add", "purchase"}
_SELL_ACTIONS = {"sell", "trim", "reduce"}
_EPSILON = 1e-9


def _finite_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normal_ticker(value: Any) -> str:
    return "".join(str(value or "").strip().upper().split())


def _normal_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


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


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _rows(value: Any, *, keyed_field: str, record_markers: set[str]) -> list[dict[str, Any]]:
    """Return copied row dictionaries from common list or keyed-map inputs."""
    if isinstance(value, Mapping):
        if record_markers & {str(key) for key in value}:
            return [deepcopy(dict(value))]
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                row = deepcopy(dict(item))
                row.setdefault(keyed_field, key)
            else:
                row = {keyed_field: key, "value": deepcopy(item)}
                if keyed_field == "ticker" and isinstance(item, bool):
                    row["approved"] = item
            rows.append(row)
        return rows
    if _is_sequence(value):
        return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _position_rows(positions: Any) -> list[dict[str, Any]]:
    return _rows(
        positions,
        keyed_field="ticker",
        record_markers={"ticker", "quantity", "entry_price", "status"},
    )


def _trade_rows(trades: Any) -> list[dict[str, Any]]:
    return _rows(
        trades,
        keyed_field="ticker",
        record_markers={"ticker", "action", "quantity", "shares", "units"},
    )


def _metadata_rows(value: Any) -> list[dict[str, Any]]:
    return _rows(
        value,
        keyed_field="ticker",
        record_markers={"ticker", "payload", "approved", "status"},
    )


def _payload(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    raw = record.get("payload")
    return deepcopy(dict(raw)) if isinstance(raw, Mapping) else deepcopy(dict(record))


def _normal_live_prices(live_prices: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in (live_prices or {}).items():
        ticker = _normal_ticker(key)
        price = _finite_number(value)
        if ticker and price is not None and price > 0:
            result[ticker] = price
    return result


def _status(row: Mapping[str, Any]) -> str:
    return _normal_key(row.get("status") or "open")


def _is_open(row: Mapping[str, Any]) -> bool:
    return _status(row) != "closed"


def _positive_quantity(row: Mapping[str, Any]) -> float:
    quantity = _finite_number(row.get("quantity"))
    return max(0.0, quantity or 0.0)


def _weighted_choice(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    default: str,
) -> str:
    choices: dict[str, tuple[float, int, str]] = {}
    for index, row in enumerate(rows):
        display = " ".join(str(row.get(field) or default).strip().split()) or default
        key = _normal_key(display)
        exposure = abs(_finite_number(row.get("current_value")) or 0.0)
        prior_exposure, prior_index, prior_display = choices.get(
            key, (0.0, index, display)
        )
        choices[key] = (
            prior_exposure + exposure,
            min(prior_index, index),
            prior_display,
        )
    if not choices:
        return default
    return sorted(choices.values(), key=lambda item: (-item[0], item[1], item[2].casefold()))[0][2]


def _weighted_price(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        quantity = _positive_quantity(row)
        price = _finite_number(row.get(field))
        if quantity > 0 and price is not None and price > 0:
            numerator += quantity * price
            denominator += quantity
    return numerator / denominator if denominator > 0 else None


def _true_cash(performance: Mapping[str, Any]) -> float:
    return float(performance.get("cash_before_pnl") or 0.0) + float(
        performance.get("realized_pnl") or 0.0
    )


def build_competition_strategy_snapshot(
    positions: Any,
    live_prices: Mapping[str, Any] | None = None,
    *,
    theses: Any = (),
    approved_securities: Any = (),
    initial_capital: float = INITIAL_CAPITAL_USD,
) -> dict[str, Any]:
    """Build an aggregated competition snapshot suitable for strategy analysis.

    Open transaction lots are aggregated to one holding per ticker.  Cash is
    derived from the complete lot ledger as initial capital less open cost plus
    realised P/L, which correctly includes proceeds from closed positions.
    """
    position_rows = _position_rows(positions)
    prices = _normal_live_prices(live_prices)
    performance = calculate_portfolio_performance(
        position_rows,
        prices,
        initial_capital=float(initial_capital),
    )

    thesis_rows = _metadata_rows(theses)
    thesis_by_ticker: dict[str, dict[str, Any]] = {}
    for item in thesis_rows:
        ticker = _normal_ticker(item.get("ticker"))
        if ticker:
            thesis_by_ticker[ticker] = item

    approved_rows = _metadata_rows(approved_securities)
    approved_by_ticker: dict[str, bool] = {}
    for item in approved_rows:
        ticker = _normal_ticker(item.get("ticker"))
        if ticker:
            approved_by_ticker[ticker] = _as_bool(item.get("approved"), True)
    universe_configured = bool(approved_by_ticker)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in performance.get("positions", []):
        if not isinstance(raw, Mapping) or not _is_open(raw):
            continue
        ticker = _normal_ticker(raw.get("ticker"))
        if not ticker or _positive_quantity(raw) <= 0:
            continue
        grouped.setdefault(ticker, []).append(deepcopy(dict(raw)))

    holdings: list[dict[str, Any]] = []
    open_quantities: dict[str, float] = {}
    price_by_ticker: dict[str, float] = {}
    price_source_by_ticker: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []

    for ticker in sorted(grouped):
        lots = grouped[ticker]
        quantity = sum(_positive_quantity(row) for row in lots)
        market_value = sum(float(row.get("current_value") or 0.0) for row in lots)
        cost_value = sum(float(row.get("cost") or 0.0) for row in lots)
        current_price = market_value / quantity if quantity > 0 else 0.0
        sources = sorted({str(row.get("price_source") or "unknown") for row in lots})
        price_source = sources[0] if len(sources) == 1 else "mixed"
        thesis_record = thesis_by_ticker.get(ticker, {})
        thesis = _payload(thesis_record)
        beta = _finite_number(thesis.get("beta"))
        tags_raw = thesis.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [part.strip().casefold() for part in tags_raw.replace(";", ",").split(",") if part.strip()]
        elif _is_sequence(tags_raw):
            tags = [str(part).strip().casefold() for part in tags_raw if str(part).strip()]
        else:
            tags = []
        goals = deepcopy(thesis.get("goals") or [])
        security_type = _weighted_choice(lots, "security_type", "Stock")
        holding = {
            "ticker": ticker,
            "name": str(thesis.get("company_name") or thesis.get("name") or ticker),
            "quantity": quantity,
            "market_value": market_value,
            "current_value": market_value,
            "cost_value": cost_value,
            "current_price": current_price,
            "price_source": price_source,
            "lot_count": len(lots),
            "sector": thesis.get("sector") or "Unassigned",
            "primary_goal": thesis.get("primary_goal") or "",
            "goals": goals,
            "thesis_status": thesis_record.get("status") or thesis.get("thesis_status") or "missing",
            "beta": beta,
            "approved": approved_by_ticker.get(ticker, False) if universe_configured else None,
            "asset_type": security_type,
            "security_type": security_type,
            "tags": sorted(set(tags)),
        }
        holdings.append(holding)
        open_quantities[ticker] = quantity
        price_by_ticker[ticker] = current_price
        price_source_by_ticker[ticker] = price_source
        if price_source in {"entry fallback", "mixed"} and any(
            str(row.get("price_source") or "") == "entry fallback" for row in lots
        ):
            warnings.append(
                {
                    "code": "entry_price_fallback",
                    "severity": "low",
                    "scope": "holding",
                    "subject": ticker,
                    "message": f"{ticker} uses an entry-price fallback for at least one open lot.",
                }
            )

    cash_value = _true_cash(performance)
    portfolio_value = float(performance.get("equity") or 0.0)
    return {
        "performance": deepcopy(performance),
        "holdings": holdings,
        "open_quantities": open_quantities,
        "price_by_ticker": price_by_ticker,
        "price_source_by_ticker": price_source_by_ticker,
        "cash_value": cash_value,
        "portfolio_value": portfolio_value,
        "invested_value": sum(float(item["market_value"]) for item in holdings),
        "initial_capital": float(initial_capital),
        "approved_universe_configured": universe_configured,
        "source_position_count": len(position_rows),
        "open_position_count": len(holdings),
        "warnings": warnings,
    }


def _blocker(
    code: str,
    message: str,
    *,
    trade_index: int,
    subject: str,
    actual: Any = None,
    limit: Any = None,
    scope: str = "trade",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "blocker",
        "scope": scope,
        "subject": subject,
        "message": message,
        "actual": actual,
        "limit": limit,
        "trade_index": trade_index,
    }


def _fallback_trade_price(
    ticker: str,
    trade: Mapping[str, Any],
    positions: Sequence[Mapping[str, Any]],
    live_prices: Mapping[str, float],
) -> tuple[float | None, str, str | None]:
    has_explicit = "price" in trade or "execution_price" in trade
    explicit_raw = trade.get("price", trade.get("execution_price"))
    if has_explicit and explicit_raw not in (None, "", 0, 0.0):
        explicit = _finite_number(explicit_raw)
        if explicit is None or explicit <= 0:
            return None, "invalid", "invalid_price"
        return explicit, "explicit", None

    live = live_prices.get(ticker)
    if live is not None and live > 0:
        return live, "live", None

    open_lots = [
        row for row in positions
        if _is_open(row) and _normal_ticker(row.get("ticker")) == ticker
    ]
    stored = _weighted_price(open_lots, "last_price")
    if stored is not None and stored > 0:
        return stored, "stored", None
    entry = _weighted_price(open_lots, "entry_price")
    if entry is not None and entry > 0:
        return entry, "entry", None
    return None, "missing", "missing_price"


def _existing_security_type(
    positions: Sequence[Mapping[str, Any]],
    ticker: str,
) -> str:
    for row in positions:
        if _is_open(row) and _normal_ticker(row.get("ticker")) == ticker:
            value = str(row.get("security_type") or "").strip()
            if value:
                return value
    return "Stock"


def _fifo_open_indices(
    positions: Sequence[Mapping[str, Any]],
    ticker: str,
) -> list[int]:
    candidates: list[tuple[tuple[Any, ...], int]] = []
    for index, row in enumerate(positions):
        if not _is_open(row) or _normal_ticker(row.get("ticker")) != ticker:
            continue
        if _positive_quantity(row) <= 0:
            continue
        sequence = row.get("_pretrade_sequence")
        if sequence is not None:
            key: tuple[Any, ...] = (1, int(sequence), index)
        else:
            raw_id = row.get("id")
            numeric_id = _finite_number(raw_id)
            id_key: tuple[Any, ...] = (
                (0, numeric_id) if numeric_id is not None else (1, str(raw_id or ""))
            )
            key = (
                0,
                str(row.get("entry_date") or ""),
                id_key,
                index,
            )
        candidates.append((key, index))
    return [index for _, index in sorted(candidates)]


def _apply_buy(
    positions: list[dict[str, Any]],
    trade: Mapping[str, Any],
    *,
    trade_index: int,
    ticker: str,
    quantity: float,
    price: float,
    security_type: str,
) -> None:
    positions.append(
        {
            "id": f"pretrade-buy-{trade_index}",
            "ticker": ticker,
            "security_type": security_type,
            "quantity": quantity,
            "entry_price": price,
            "entry_date": str(trade.get("trade_date") or trade.get("date") or "pretrade"),
            "opened_by": "pretrade",
            "opened_at": "pretrade",
            "last_price": price,
            "notes": str(trade.get("notes") or "Proposed trade"),
            "status": "open",
            "_pretrade_sequence": trade_index,
        }
    )


def _apply_fifo_sell(
    positions: list[dict[str, Any]],
    trade: Mapping[str, Any],
    *,
    trade_index: int,
    ticker: str,
    quantity: float,
    price: float,
) -> None:
    remaining = quantity
    closed_splits: list[dict[str, Any]] = []
    split_number = 0
    for lot_index in _fifo_open_indices(positions, ticker):
        if remaining <= _EPSILON:
            break
        lot = positions[lot_index]
        lot_quantity = _positive_quantity(lot)
        consumed = min(lot_quantity, remaining)
        trade_date = str(trade.get("trade_date") or trade.get("date") or "pretrade")
        if consumed >= lot_quantity - _EPSILON:
            lot["status"] = "closed"
            lot["exit_price"] = price
            lot["exit_date"] = trade_date
            lot["closed_by"] = "pretrade"
        else:
            lot["quantity"] = lot_quantity - consumed
            split_number += 1
            closed = deepcopy(lot)
            closed.update(
                {
                    "id": f"pretrade-sell-{trade_index}-lot-{split_number}",
                    "quantity": consumed,
                    "status": "closed",
                    "exit_price": price,
                    "exit_date": trade_date,
                    "closed_by": "pretrade",
                    "_pretrade_sequence": trade_index,
                }
            )
            closed_splits.append(closed)
        remaining -= consumed
    positions.extend(closed_splits)


def _clean_internal_positions(positions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in positions:
        row = deepcopy(dict(raw))
        row.pop("_pretrade_sequence", None)
        result.append(row)
    return result


def simulate_trade_plan(
    positions: Any,
    trades: Any,
    live_prices: Mapping[str, Any] | None = None,
    *,
    initial_capital: float = INITIAL_CAPITAL_USD,
) -> dict[str, Any]:
    """Sequentially validate a proposed buy/sell plan and simulate lot changes.

    Sells consume open lots FIFO by entry date and id; partial sells split the
    consumed quantity into a synthetic closed lot.  The plan is atomic in the
    public ``after_positions`` state.
    """
    before_positions = _position_rows(positions)
    proposal_rows = _trade_rows(trades)
    prices = _normal_live_prices(live_prices)
    before_performance = calculate_portfolio_performance(
        before_positions,
        prices,
        initial_capital=float(initial_capital),
    )
    before_cash = _true_cash(before_performance)
    before_equity = float(before_performance.get("equity") or 0.0)
    running_cash = before_cash
    working = deepcopy(before_positions)

    trade_results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    gross_notional = 0.0
    validated_notional = 0.0

    if not proposal_rows:
        warnings.append(
            {
                "code": "empty_trade_plan",
                "severity": "low",
                "scope": "plan",
                "subject": "Trade plan",
                "message": "No proposed trades were supplied.",
            }
        )

    for offset, raw_trade in enumerate(proposal_rows):
        trade_index = offset + 1
        ticker = _normal_ticker(raw_trade.get("ticker"))
        action_raw = _normal_key(raw_trade.get("action"))
        action = "buy" if action_raw in _BUY_ACTIONS else "sell" if action_raw in _SELL_ACTIONS else ""
        quantity = _finite_number(
            raw_trade.get("quantity", raw_trade.get("shares", raw_trade.get("units")))
        )
        row_blockers: list[dict[str, Any]] = []
        subject = ticker or f"Trade {trade_index}"

        if not ticker:
            row_blockers.append(
                _blocker(
                    "invalid_ticker",
                    "A proposed trade is missing a ticker.",
                    trade_index=trade_index,
                    subject=subject,
                    actual=raw_trade.get("ticker"),
                )
            )
        if not action:
            row_blockers.append(
                _blocker(
                    "invalid_action",
                    "Trade action must be buy or sell.",
                    trade_index=trade_index,
                    subject=subject,
                    actual=raw_trade.get("action"),
                    limit=["buy", "sell"],
                )
            )
        if quantity is None or quantity <= 0:
            row_blockers.append(
                _blocker(
                    "invalid_quantity",
                    "Trade quantity must be a positive finite number.",
                    trade_index=trade_index,
                    subject=subject,
                    actual=raw_trade.get("quantity", raw_trade.get("shares", raw_trade.get("units"))),
                    limit="> 0",
                )
            )

        price: float | None = None
        price_source = "missing"
        price_error: str | None = None
        if ticker:
            price, price_source, price_error = _fallback_trade_price(
                ticker,
                raw_trade,
                working,
                prices,
            )
            if price_error == "invalid_price":
                row_blockers.append(
                    _blocker(
                        "invalid_price",
                        "Explicit trade price must be a positive finite number.",
                        trade_index=trade_index,
                        subject=subject,
                        actual=raw_trade.get("price", raw_trade.get("execution_price")),
                        limit="> 0",
                    )
                )
            elif price_error == "missing_price":
                row_blockers.append(
                    _blocker(
                        "missing_price",
                        "No explicit, live, stored, or entry price is available for this trade.",
                        trade_index=trade_index,
                        subject=subject,
                    )
                )

        notional = (
            float(quantity) * float(price)
            if quantity is not None and quantity > 0 and price is not None and price > 0
            else None
        )
        if notional is not None and action:
            gross_notional += notional

        if not row_blockers and action == "buy" and notional is not None:
            if running_cash + _EPSILON < notional:
                row_blockers.append(
                    _blocker(
                        "insufficient_cash",
                        f"The ordered plan has only ${running_cash:,.2f} available before this buy.",
                        trade_index=trade_index,
                        subject=subject,
                        actual=notional,
                        limit=max(0.0, running_cash),
                    )
                )
        if not row_blockers and action == "sell" and quantity is not None:
            available = sum(
                _positive_quantity(row)
                for row in working
                if _is_open(row) and _normal_ticker(row.get("ticker")) == ticker
            )
            if available + _EPSILON < quantity:
                row_blockers.append(
                    _blocker(
                        "oversell",
                        f"The ordered plan can sell at most {available:g} units of {ticker} at this step.",
                        trade_index=trade_index,
                        subject=subject,
                        actual=quantity,
                        limit=available,
                    )
                )

        security_type = str(
            raw_trade.get("security_type")
            or _existing_security_type(working, ticker)
            or "Stock"
        ).strip() or "Stock"
        if not row_blockers and notional is not None and quantity is not None and price is not None:
            if action == "buy":
                _apply_buy(
                    working,
                    raw_trade,
                    trade_index=trade_index,
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    security_type=security_type,
                )
                running_cash -= notional
            else:
                _apply_fifo_sell(
                    working,
                    raw_trade,
                    trade_index=trade_index,
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                )
                running_cash += notional
            validated_notional += notional

        blockers.extend(row_blockers)
        message = (
            "; ".join(item["message"] for item in row_blockers)
            if row_blockers
            else f"Trade {trade_index} is feasible in the supplied order."
        )
        checks.append(
            {
                "code": f"trade_{trade_index}_feasible",
                "passed": not row_blockers,
                "message": message,
                "trade_index": trade_index,
            }
        )
        trade_results.append(
            {
                "trade_index": trade_index,
                "ticker": ticker,
                "action": action or action_raw,
                "quantity": quantity,
                "price": price,
                "price_source": price_source,
                "security_type": security_type,
                "notional": notional,
                "status": "blocked" if row_blockers else "validated",
                "blocker_codes": [item["code"] for item in row_blockers],
            }
        )

    plan_valid = not blockers
    projected_positions = _clean_internal_positions(working)
    after_positions = projected_positions if plan_valid else deepcopy(before_positions)
    after_performance = calculate_portfolio_performance(
        after_positions,
        prices,
        initial_capital=float(initial_capital),
    )
    after_cash = _true_cash(after_performance)
    incremental_turnover = gross_notional / before_equity if before_equity > 0 else None

    return {
        "status": "pass" if plan_valid else "blocked",
        "plan_valid": plan_valid,
        "checks": checks,
        "blockers": blockers,
        "warnings": warnings,
        "trades": trade_results,
        "before_positions": deepcopy(before_positions),
        "projected_positions": projected_positions,
        "after_positions": deepcopy(after_positions),
        "before_performance": deepcopy(before_performance),
        "after_performance": deepcopy(after_performance),
        "before_cash_value": before_cash,
        "projected_cash_value": running_cash,
        "after_cash_value": after_cash,
        "gross_proposed_notional": gross_notional,
        "validated_notional": validated_notional,
        "incremental_turnover": incremental_turnover,
        "turnover_definition": TURNOVER_DEFINITION,
    }


def _issue_identity(issue: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _normal_key(issue.get("code")),
        _normal_key(issue.get("scope")),
        _normal_key(issue.get("subject")),
    )


def _identity_text(identity: tuple[str, str, str]) -> str:
    return "|".join(identity)


def _compare_violations(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    before_map = {
        _issue_identity(item): deepcopy(dict(item))
        for item in before.get("violations", [])
        if isinstance(item, Mapping)
    }
    after_map = {
        _issue_identity(item): deepcopy(dict(item))
        for item in after.get("violations", [])
        if isinstance(item, Mapping)
    }
    new: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    persistent: list[dict[str, Any]] = []
    for identity in sorted(after_map.keys() - before_map.keys()):
        item = after_map[identity]
        item["identity"] = _identity_text(identity)
        new.append(item)
    for identity in sorted(before_map.keys() - after_map.keys()):
        item = before_map[identity]
        item["identity"] = _identity_text(identity)
        resolved.append(item)
    for identity in sorted(before_map.keys() & after_map.keys()):
        after_item = after_map[identity]
        persistent.append(
            {
                "identity": _identity_text(identity),
                "code": after_item.get("code"),
                "scope": after_item.get("scope"),
                "subject": after_item.get("subject"),
                "before": before_map[identity],
                "after": after_item,
            }
        )
    return {"new": new, "resolved": resolved, "persistent": persistent}


def _number_or_zero(value: Any) -> float:
    return _finite_number(value) or 0.0


def _allocation_changes(
    before_rows: Any,
    after_rows: Any,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    before_list = [dict(item) for item in before_rows or [] if isinstance(item, Mapping)]
    after_list = [dict(item) for item in after_rows or [] if isinstance(item, Mapping)]
    if kind == "goal":
        def key(item: Mapping[str, Any]) -> str:
            return _normal_key(item.get("goal_id") or item.get("goal_name"))

        def name(item: Mapping[str, Any]) -> str:
            return str(item.get("goal_name") or item.get("goal_id") or "Unassigned")
    else:
        def key(item: Mapping[str, Any]) -> str:
            return _normal_key(item.get("sector") or item.get("sector_name"))

        def name(item: Mapping[str, Any]) -> str:
            return str(item.get("sector") or item.get("sector_name") or "Unclassified")

    before_map = {key(item): item for item in before_list if key(item)}
    after_map = {key(item): item for item in after_list if key(item)}
    changes: list[dict[str, Any]] = []
    for identity in sorted(before_map.keys() | after_map.keys()):
        old = before_map.get(identity, {})
        new = after_map.get(identity, {})
        before_actual = _number_or_zero(old.get("actual_weight"))
        after_actual = _number_or_zero(new.get("actual_weight"))
        before_drift_raw = _finite_number(old.get("drift"))
        after_drift_raw = _finite_number(new.get("drift"))
        before_drift = before_drift_raw if before_drift_raw is not None else 0.0
        after_drift = after_drift_raw if after_drift_raw is not None else 0.0
        row = {
            f"{kind}_key": identity,
            f"{kind}_name": name(new or old),
            "target_weight": new.get("target_weight", old.get("target_weight")),
            "before_actual_weight": before_actual,
            "after_actual_weight": after_actual,
            "actual_weight_delta": after_actual - before_actual,
            "before_drift": before_drift_raw,
            "after_drift": after_drift_raw,
            "drift_delta": after_drift - before_drift,
            "before_abs_drift": abs(before_drift),
            "after_abs_drift": abs(after_drift),
            "abs_drift_delta": abs(after_drift) - abs(before_drift),
            "before_status": old.get("status"),
            "after_status": new.get("status"),
        }
        if kind == "goal":
            row["goal_id"] = new.get("goal_id", old.get("goal_id"))
        changes.append(row)
    return changes


def _holding_changes(
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    before_alignment: Mapping[str, Any],
    after_alignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_holdings = {
        _normal_ticker(item.get("ticker")): dict(item)
        for item in before_snapshot.get("holdings", [])
        if isinstance(item, Mapping) and _normal_ticker(item.get("ticker"))
    }
    after_holdings = {
        _normal_ticker(item.get("ticker")): dict(item)
        for item in after_snapshot.get("holdings", [])
        if isinstance(item, Mapping) and _normal_ticker(item.get("ticker"))
    }
    before_aligned = {
        _normal_ticker(item.get("ticker")): dict(item)
        for item in before_alignment.get("holdings", [])
        if isinstance(item, Mapping) and _normal_ticker(item.get("ticker"))
    }
    after_aligned = {
        _normal_ticker(item.get("ticker")): dict(item)
        for item in after_alignment.get("holdings", [])
        if isinstance(item, Mapping) and _normal_ticker(item.get("ticker"))
    }
    changes: list[dict[str, Any]] = []
    for ticker in sorted(before_holdings.keys() | after_holdings.keys()):
        old = before_holdings.get(ticker, {})
        new = after_holdings.get(ticker, {})
        old_alignment = before_aligned.get(ticker, {})
        new_alignment = after_aligned.get(ticker, {})
        before_quantity = _number_or_zero(old.get("quantity"))
        after_quantity = _number_or_zero(new.get("quantity"))
        before_value = _number_or_zero(old.get("market_value"))
        after_value = _number_or_zero(new.get("market_value"))
        before_weight = _number_or_zero(old_alignment.get("portfolio_weight"))
        after_weight = _number_or_zero(new_alignment.get("portfolio_weight"))
        changes.append(
            {
                "ticker": ticker,
                "before_quantity": before_quantity,
                "after_quantity": after_quantity,
                "quantity_delta": after_quantity - before_quantity,
                "before_market_value": before_value,
                "after_market_value": after_value,
                "market_value_delta": after_value - before_value,
                "before_weight": before_weight,
                "after_weight": after_weight,
                "weight_delta": after_weight - before_weight,
                "before_price": old.get("current_price"),
                "after_price": new.get("current_price"),
            }
        )
    return changes


def _absolute_drift(rows: Any) -> float:
    total = 0.0
    for item in rows or []:
        if isinstance(item, Mapping):
            drift = _finite_number(item.get("drift"))
            if drift is not None:
                total += abs(drift)
    return total


def _metric_delta(before: Any, after: Any) -> float:
    return _number_or_zero(after) - _number_or_zero(before)


def analyze_pretrade_impact(
    positions: Any,
    trades: Any,
    mandate: Mapping[str, Any] | None = None,
    strategy: Mapping[str, Any] | None = None,
    *,
    live_prices: Mapping[str, Any] | None = None,
    theses: Any = (),
    approved_securities: Any = (),
    initial_capital: float = INITIAL_CAPITAL_USD,
    current_turnover: float | None = None,
) -> dict[str, Any]:
    """Compare strategy alignment before and after an ordered trade plan."""
    simulation = simulate_trade_plan(
        positions,
        trades,
        live_prices,
        initial_capital=initial_capital,
    )
    before_snapshot = build_competition_strategy_snapshot(
        simulation["before_positions"],
        live_prices,
        theses=theses,
        approved_securities=approved_securities,
        initial_capital=initial_capital,
    )
    after_snapshot = build_competition_strategy_snapshot(
        simulation["after_positions"],
        live_prices,
        theses=theses,
        approved_securities=approved_securities,
        initial_capital=initial_capital,
    )

    before_strategy = deepcopy(dict(strategy or {}))
    after_strategy = deepcopy(dict(strategy or {}))
    supplied_turnover = _finite_number(current_turnover)
    stored_turnover = _finite_number(before_strategy.get("current_turnover"))
    base_turnover = supplied_turnover if supplied_turnover is not None else stored_turnover
    if base_turnover is not None:
        before_strategy["current_turnover"] = max(0.0, base_turnover)
        incremental = simulation.get("incremental_turnover")
        after_strategy["current_turnover"] = max(0.0, base_turnover) + (
            float(incremental or 0.0) if simulation["plan_valid"] else 0.0
        )

    before_alignment = analyze_strategy_alignment(
        before_snapshot["holdings"],
        deepcopy(dict(mandate or {})),
        before_strategy,
        cash_value=before_snapshot["cash_value"],
        portfolio_value=before_snapshot["portfolio_value"],
    )
    after_alignment = analyze_strategy_alignment(
        after_snapshot["holdings"],
        deepcopy(dict(mandate or {})),
        after_strategy,
        cash_value=after_snapshot["cash_value"],
        portfolio_value=after_snapshot["portfolio_value"],
    )

    violation_changes = _compare_violations(before_alignment, after_alignment)
    goal_changes = _allocation_changes(
        before_alignment.get("goal_allocation", []),
        after_alignment.get("goal_allocation", []),
        kind="goal",
    )
    sector_changes = _allocation_changes(
        before_alignment.get("sector_allocation", []),
        after_alignment.get("sector_allocation", []),
        kind="sector",
    )
    holding_changes = _holding_changes(
        before_snapshot,
        after_snapshot,
        before_alignment,
        after_alignment,
    )

    before_summary = before_alignment.get("portfolio_summary", {})
    after_summary = after_alignment.get("portfolio_summary", {})
    before_goal_drift = _absolute_drift(before_alignment.get("goal_allocation", []))
    after_goal_drift = _absolute_drift(after_alignment.get("goal_allocation", []))
    before_sector_drift = _absolute_drift(before_alignment.get("sector_allocation", []))
    after_sector_drift = _absolute_drift(after_alignment.get("sector_allocation", []))
    deltas = {
        "alignment_score": _metric_delta(
            before_alignment.get("alignment_score"), after_alignment.get("alignment_score")
        ),
        "cash_value": _metric_delta(before_snapshot.get("cash_value"), after_snapshot.get("cash_value")),
        "cash_weight": _metric_delta(before_summary.get("cash_weight"), after_summary.get("cash_weight")),
        "largest_position_weight": _metric_delta(
            before_summary.get("largest_position_weight"),
            after_summary.get("largest_position_weight"),
        ),
        "effective_holdings": _metric_delta(
            before_summary.get("effective_holdings"), after_summary.get("effective_holdings")
        ),
        "position_count": _metric_delta(
            before_summary.get("position_count"), after_summary.get("position_count")
        ),
        "goal_absolute_drift": after_goal_drift - before_goal_drift,
        "sector_absolute_drift": after_sector_drift - before_sector_drift,
        "gross_proposed_notional": simulation["gross_proposed_notional"],
        "incremental_turnover": simulation["incremental_turnover"],
    }

    plan_valid = bool(simulation["plan_valid"])
    alignment_worsened = deltas["alignment_score"] < -1e-9
    status = (
        "blocked"
        if not plan_valid
        else "review"
        if violation_changes["new"] or alignment_worsened
        else "pass"
    )
    top_warnings = (
        deepcopy(simulation.get("warnings", []))
        + deepcopy(after_snapshot.get("warnings", []))
        + deepcopy(after_alignment.get("warnings", []))
    )

    return {
        "status": status,
        "plan_valid": plan_valid,
        "checks": deepcopy(simulation["checks"]),
        "blockers": deepcopy(simulation["blockers"]),
        "warnings": top_warnings,
        "trades": deepcopy(simulation["trades"]),
        "gross_proposed_notional": simulation["gross_proposed_notional"],
        "incremental_turnover": simulation["incremental_turnover"],
        "turnover_definition": TURNOVER_DEFINITION,
        "before": {
            "snapshot": before_snapshot,
            "alignment": before_alignment,
        },
        "after": {
            "snapshot": after_snapshot,
            "alignment": after_alignment,
        },
        "deltas": deltas,
        "violation_changes": violation_changes,
        "holding_changes": holding_changes,
        "goal_changes": goal_changes,
        "sector_changes": sector_changes,
    }


__all__ = [
    "TURNOVER_DEFINITION",
    "analyze_pretrade_impact",
    "build_competition_strategy_snapshot",
    "simulate_trade_plan",
]
