from __future__ import annotations

from datetime import date
from typing import Iterable

from .models import SwingTrade


def calculate_position_size(
    *,
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    if account_size <= 0:
        raise ValueError("Account size must be greater than zero.")
    if risk_percent <= 0:
        raise ValueError("Risk percent must be greater than zero.")
    stop_distance = abs(float(entry_price) - float(stop_loss))
    if stop_distance <= 0:
        raise ValueError("Stop distance must be greater than zero.")
    risk_budget = float(account_size) * (float(risk_percent) / 100.0)
    return float(round(risk_budget / stop_distance, 6))


def calculate_holding_days(
    entry_date: date | None,
    end_date: date | None,
) -> int | None:
    if entry_date is None or end_date is None:
        return None
    days = (end_date - entry_date).days
    return max(0, int(days))


def calculate_realized_pnl(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    position_size: float,
) -> float:
    normalized_direction = str(direction).strip().lower()
    if normalized_direction == "long":
        pnl = (float(exit_price) - float(entry_price)) * float(position_size)
    else:
        pnl = (float(entry_price) - float(exit_price)) * float(position_size)
    return float(round(pnl, 6))


def calculate_realized_r_multiple(
    *,
    realized_pnl: float,
    entry_price: float,
    stop_loss: float,
    position_size: float,
) -> float:
    total_risk = abs(float(entry_price) - float(stop_loss)) * float(position_size)
    if total_risk <= 0:
        return 0.0
    return float(round(float(realized_pnl) / total_risk, 6))


def _is_overdue(actual_holding_days: int | None, planned_holding_days: int | None, time_stop_days: int | None) -> bool:
    if actual_holding_days is None:
        return False
    if planned_holding_days is not None and actual_holding_days > planned_holding_days:
        return True
    if time_stop_days is not None and actual_holding_days > time_stop_days:
        return True
    return False


def determine_trade_status(trade: SwingTrade, as_of: date | None = None) -> str:
    normalized = trade.normalized()
    base_status = normalized.status

    if base_status in {"closed", "invalidated"}:
        return base_status

    if base_status == "planned" and normalized.entry_date is not None:
        base_status = "open"

    if base_status not in {"open", "overdue"}:
        return base_status

    reference_end = normalized.exit_date or as_of or date.today()
    actual_days = calculate_holding_days(normalized.entry_date, reference_end)
    if _is_overdue(actual_days, normalized.planned_holding_days, normalized.time_stop_days):
        return "overdue"
    return "open"


def calculate_discipline_score(trade: SwingTrade, as_of: date | None = None) -> float:
    normalized = trade.normalized()
    status = determine_trade_status(normalized, as_of=as_of)
    score = 100.0

    if not normalized.stop_rationale:
        score -= 35.0

    if normalized.status == "invalidated":
        score -= 20.0

    if status == "overdue":
        score -= 25.0

    reference_end = normalized.exit_date or as_of or date.today()
    actual_days = calculate_holding_days(normalized.entry_date, reference_end)
    if (
        actual_days is not None
        and normalized.planned_holding_days is not None
        and actual_days > normalized.planned_holding_days
    ):
        score -= 10.0
    if (
        actual_days is not None
        and normalized.time_stop_days is not None
        and actual_days > normalized.time_stop_days
    ):
        score -= 10.0

    if normalized.status == "closed":
        if not normalized.exit_reason:
            score -= 12.0
        if normalized.exit_price is None:
            score -= 15.0

    if normalized.risk_percent > 2.0:
        score -= min(15.0, (normalized.risk_percent - 2.0) * 5.0)

    if normalized.position_size <= 0:
        score -= 25.0

    return float(round(max(0.0, min(100.0, score)), 1))


def apply_trade_lifecycle(trade: SwingTrade, as_of: date | None = None) -> SwingTrade:
    normalized = trade.normalized()
    final = normalized.normalized()

    final.status = determine_trade_status(normalized, as_of=as_of)  # type: ignore[assignment]

    end_for_holding = final.exit_date or as_of or date.today()
    final.actual_holding_days = calculate_holding_days(final.entry_date, end_for_holding)

    if final.status in {"closed", "invalidated"} and final.exit_price is not None:
        final.realized_pnl = calculate_realized_pnl(
            direction=final.direction,
            entry_price=final.entry_price,
            exit_price=final.exit_price,
            position_size=final.position_size,
        )
        final.realized_r_multiple = calculate_realized_r_multiple(
            realized_pnl=float(final.realized_pnl),
            entry_price=final.entry_price,
            stop_loss=final.stop_loss,
            position_size=final.position_size,
        )
    elif final.status not in {"closed", "invalidated"}:
        final.realized_pnl = None
        final.realized_r_multiple = None

    final.discipline_score = calculate_discipline_score(final, as_of=as_of)
    return final


def refresh_trade_book(
    trades: Iterable[SwingTrade],
    as_of: date | None = None,
) -> tuple[list[SwingTrade], bool]:
    refreshed: list[SwingTrade] = []
    changed = False
    for trade in trades:
        updated = apply_trade_lifecycle(trade, as_of=as_of)
        if updated.to_dict() != trade.to_dict():
            changed = True
        refreshed.append(updated)
    return refreshed, changed


def compute_capital_trapped_overdue(trades: Iterable[SwingTrade]) -> float:
    total = 0.0
    for trade in trades:
        normalized = trade.normalized()
        if normalized.status != "overdue":
            continue
        if normalized.exit_date is not None:
            continue
        total += abs(normalized.entry_price * normalized.position_size)
    return float(round(total, 2))


def compute_holding_delta(trade: SwingTrade) -> int | None:
    normalized = trade.normalized()
    if normalized.actual_holding_days is None or normalized.planned_holding_days is None:
        return None
    return int(normalized.actual_holding_days - normalized.planned_holding_days)


def summarize_trade_book(trades: Iterable[SwingTrade]) -> dict[str, float]:
    rows = [item.normalized() for item in trades]
    total = float(len(rows))
    open_count = float(sum(1 for item in rows if item.status == "open"))
    overdue_count = float(sum(1 for item in rows if item.status == "overdue"))
    planned_count = float(sum(1 for item in rows if item.status == "planned"))
    closed = [item for item in rows if item.status in {"closed", "invalidated"}]
    closed_count = float(len(closed))
    wins = [item for item in closed if (item.realized_pnl or 0.0) > 0]
    win_rate = (float(len(wins)) / closed_count) if closed_count > 0 else 0.0
    avg_r = (
        float(sum((item.realized_r_multiple or 0.0) for item in closed) / closed_count)
        if closed_count > 0
        else 0.0
    )
    avg_discipline = (
        float(sum((item.discipline_score or 0.0) for item in rows) / total)
        if total > 0
        else 0.0
    )
    trapped_capital = compute_capital_trapped_overdue(rows)

    return {
        "total_trades": total,
        "planned_trades": planned_count,
        "open_trades": open_count,
        "overdue_trades": overdue_count,
        "closed_trades": closed_count,
        "win_rate": win_rate,
        "avg_r_multiple": avg_r,
        "avg_discipline_score": avg_discipline,
        "capital_trapped_overdue": float(trapped_capital),
    }
