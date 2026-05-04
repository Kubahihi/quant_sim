from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .logic import (
    apply_trade_lifecycle,
    calculate_position_size,
    compute_holding_delta,
    refresh_trade_book,
    summarize_trade_book,
)
from .models import SwingTrade
from .stop_logic import validate_stop_rationale
from .stop_logic import validate_stop_loss_side


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Legacy swing tracker path (for backward compatibility) - also exported for __init__.py
LEGACY_SWING_TRACKER_PATH = PROJECT_ROOT / "data" / "swing_tracker" / "trades.json"
SWING_TRACKER_PATH = LEGACY_SWING_TRACKER_PATH  # Alias for backward compatibility


def _get_swing_tracker_path(user_id: int | None = None) -> Path:
    """
    Get the swing tracker file path for a user.
    
    If user_id is provided, returns user-specific path.
    Otherwise, returns the legacy path for backward compatibility.
    """
    if user_id is not None:
        user_dir = PROJECT_ROOT / "data" / "users" / str(user_id) / "swing_tracker"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "trades.json"
    return LEGACY_SWING_TRACKER_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_storage(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _default_payload() -> dict[str, Any]:
    return {
        "updated_at": _utc_now_iso(),
        "trades": [],
    }


def _resolve_storage_path(storage_path: str | Path | None = None, user_id: int | None = None) -> Path:
    """
    Resolve the storage path for swing tracker data.
    
    If storage_path is explicitly provided, use it (for testing/custom paths).
    If user_id is provided, use user-specific path.
    Otherwise, use legacy path for backward compatibility.
    """
    if storage_path is not None:
        return Path(storage_path)
    return _get_swing_tracker_path(user_id)


def load_trade_book(storage_path: str | Path | None = None, user_id: int | None = None) -> list[SwingTrade]:
    """
    Load trade book from storage.
    
    Args:
        storage_path: Explicit storage path (overrides user_id).
        user_id: User ID for user-specific storage (used if storage_path not provided).
    """
    path = _ensure_storage(_resolve_storage_path(storage_path, user_id))
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    raw_trades = payload.get("trades", [])
    if not isinstance(raw_trades, list):
        return []

    trades: list[SwingTrade] = []
    for item in raw_trades:
        if not isinstance(item, dict):
            continue
        trade = SwingTrade.from_dict(item)
        if not trade.id:
            continue
        trades.append(trade)
    return trades


def save_trade_book(
    trades: Iterable[SwingTrade],
    storage_path: str | Path | None = None,
    user_id: int | None = None,
) -> Path:
    """
    Save trade book to storage.
    
    Args:
        trades: The trades to save.
        storage_path: Explicit storage path (overrides user_id).
        user_id: User ID for user-specific storage (used if storage_path not provided).
    """
    path = _ensure_storage(_resolve_storage_path(storage_path, user_id))
    refreshed, _ = refresh_trade_book(list(trades))
    payload = _default_payload()
    payload["trades"] = [item.to_dict() for item in refreshed]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def new_trade_id() -> str:
    return f"swing_{uuid4().hex[:12]}"


def validate_trade(trade: SwingTrade) -> list[str]:
    errors = trade.validate()
    if not validate_stop_rationale(trade.stop_rationale):
        errors.append("Stop-loss rationale must be provided.")
    try:
        validate_stop_loss_side(
            direction=trade.direction,
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
        )
    except Exception as exc:
        errors.append(str(exc))
    return errors


def create_trade(
    *,
    ticker: str,
    direction: str,
    setup_type: str,
    thesis: str,
    entry_price: float,
    stop_loss: float,
    stop_type: str,
    stop_rationale: str,
    target_price: float | None,
    targets: list[float] | None,
    time_stop_days: int | None,
    planned_holding_days: int | None,
    risk_percent: float,
    position_size: float,
    status: str,
    entry_date: date | None,
    notes: str = "",
    trade_id: str | None = None,
) -> SwingTrade:
    trade = SwingTrade(
        id=(trade_id or new_trade_id()),
        ticker=ticker,
        direction=direction.lower(),  # type: ignore[arg-type]
        setup_type=setup_type,
        thesis=thesis,
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        stop_type=stop_type.lower(),  # type: ignore[arg-type]
        stop_rationale=stop_rationale,
        target_price=target_price,
        targets=targets or [],
        time_stop_days=time_stop_days,
        planned_holding_days=planned_holding_days,
        risk_percent=float(risk_percent),
        position_size=float(position_size),
        status=status.lower(),  # type: ignore[arg-type]
        entry_date=entry_date,
        notes=notes,
    )
    trade = apply_trade_lifecycle(trade)
    errors = validate_trade(trade)
    if errors:
        raise ValueError(" | ".join(errors))
    return trade


def upsert_trade(trades: Iterable[SwingTrade], trade: SwingTrade) -> list[SwingTrade]:
    rows: list[SwingTrade] = []
    replaced = False
    for item in trades:
        if item.id == trade.id:
            rows.append(trade)
            replaced = True
        else:
            rows.append(item)
    if not replaced:
        rows.append(trade)
    return rows


def close_trade(
    trades: Iterable[SwingTrade],
    *,
    trade_id: str,
    exit_price: float,
    exit_date: date,
    exit_reason: str,
    notes: str = "",
    final_status: str = "closed",
) -> tuple[list[SwingTrade], SwingTrade]:
    updated_rows: list[SwingTrade] = []
    closed_trade: SwingTrade | None = None
    normalized_final_status = "invalidated" if str(final_status).strip().lower() == "invalidated" else "closed"

    for item in trades:
        if item.id != trade_id:
            updated_rows.append(item)
            continue

        if item.status in {"closed", "invalidated"}:
            raise ValueError("Trade is already closed or invalidated.")

        merged_notes = str(item.notes or "").strip()
        new_note_text = str(notes or "").strip()
        if new_note_text:
            merged_notes = f"{merged_notes}\n{new_note_text}".strip()

        updated = SwingTrade(
            id=item.id,
            ticker=item.ticker,
            direction=item.direction,
            setup_type=item.setup_type,
            thesis=item.thesis,
            entry_price=item.entry_price,
            stop_loss=item.stop_loss,
            stop_type=item.stop_type,
            stop_rationale=item.stop_rationale,
            target_price=item.target_price,
            targets=item.targets,
            time_stop_days=item.time_stop_days,
            planned_holding_days=item.planned_holding_days,
            risk_percent=item.risk_percent,
            position_size=item.position_size,
            status=normalized_final_status,  # type: ignore[arg-type]
            entry_date=item.entry_date,
            exit_date=exit_date,
            exit_price=float(exit_price),
            exit_reason=str(exit_reason or "").strip(),
            notes=merged_notes,
        )
        updated = apply_trade_lifecycle(updated, as_of=exit_date)
        updated_rows.append(updated)
        closed_trade = updated

    if closed_trade is None:
        raise KeyError(f"Trade id not found: {trade_id}")

    return updated_rows, closed_trade


def open_trade_rows(trades: Iterable[SwingTrade]) -> list[SwingTrade]:
    return [item for item in trades if item.status in {"open", "overdue"}]


def historical_trade_rows(trades: Iterable[SwingTrade]) -> list[SwingTrade]:
    return [item for item in trades if item.status in {"closed", "invalidated"}]


def trade_to_row(trade: SwingTrade) -> dict[str, Any]:
    normalized = trade.normalized()
    holding_delta = compute_holding_delta(normalized)
    risk_per_share = abs(normalized.entry_price - normalized.stop_loss)
    risk_amount = risk_per_share * normalized.position_size
    notional = normalized.entry_price * normalized.position_size
    return {
        "ID": normalized.id,
        "Ticker": normalized.ticker,
        "Direction": normalized.direction,
        "Setup": normalized.setup_type,
        "Status": normalized.status,
        "EntryDate": normalized.entry_date.isoformat() if normalized.entry_date else "",
        "Entry": normalized.entry_price,
        "Stop": normalized.stop_loss,
        "Target": normalized.target_price,
        "RiskPct": normalized.risk_percent,
        "PositionSize": normalized.position_size,
        "Notional": notional,
        "RiskAmount": risk_amount,
        "TimeStopDays": normalized.time_stop_days,
        "PlannedHoldDays": normalized.planned_holding_days,
        "ActualHoldDays": normalized.actual_holding_days,
        "HoldDelta": holding_delta,
        "ExitDate": normalized.exit_date.isoformat() if normalized.exit_date else "",
        "Exit": normalized.exit_price,
        "ExitReason": normalized.exit_reason,
        "RealizedPnL": normalized.realized_pnl,
        "RMultiple": normalized.realized_r_multiple,
        "DisciplineScore": normalized.discipline_score,
        "StopType": normalized.stop_type,
        "StopRationale": normalized.stop_rationale,
        "Thesis": normalized.thesis,
        "Notes": normalized.notes,
    }


def trades_to_rows(trades: Iterable[SwingTrade]) -> list[dict[str, Any]]:
    return [trade_to_row(item) for item in trades]


def refresh_and_persist(
    storage_path: str | Path | None = None,
    user_id: int | None = None,
) -> list[SwingTrade]:
    """
    Refresh trade book and persist changes.
    
    Args:
        storage_path: Explicit storage path (overrides user_id).
        user_id: User ID for user-specific storage (used if storage_path not provided).
    """
    trades = load_trade_book(storage_path=storage_path, user_id=user_id)
    refreshed, changed = refresh_trade_book(trades)
    if changed:
        save_trade_book(refreshed, storage_path=storage_path, user_id=user_id)
    return refreshed


def build_discipline_overview(trades: Iterable[SwingTrade]) -> dict[str, float]:
    return summarize_trade_book(trades)


def calculate_position_size_for_trade(
    *,
    account_size: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    return calculate_position_size(
        account_size=account_size,
        risk_percent=risk_percent,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )
