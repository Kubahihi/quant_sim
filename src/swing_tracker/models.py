from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Literal, Mapping


TradeDirection = Literal["long", "short"]
TradeStatus = Literal["planned", "open", "closed", "invalidated", "overdue"]
StopType = Literal["structural", "atr", "fixed_risk", "time_stop"]

VALID_DIRECTIONS = {"long", "short"}
VALID_STATUSES = {"planned", "open", "closed", "invalidated", "overdue"}
VALID_STOP_TYPES = {"structural", "atr", "fixed_risk", "time_stop"}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return date.fromisoformat(text)
    except Exception:
        return None


def _parse_targets(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    elif isinstance(value, Iterable):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        return []

    targets: list[float] = []
    for part in parts:
        parsed = _to_optional_float(part)
        if parsed is not None and parsed > 0:
            targets.append(float(parsed))
    return targets


@dataclass
class SwingTrade:
    id: str
    ticker: str
    direction: TradeDirection
    setup_type: str
    thesis: str
    entry_price: float
    stop_loss: float
    stop_type: StopType
    stop_rationale: str
    target_price: float | None = None
    targets: list[float] = field(default_factory=list)
    time_stop_days: int | None = None
    planned_holding_days: int | None = None
    risk_percent: float = 0.0
    position_size: float = 0.0
    status: TradeStatus = "planned"
    entry_date: date | None = None
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    realized_pnl: float | None = None
    realized_r_multiple: float | None = None
    actual_holding_days: int | None = None
    discipline_score: float | None = None
    notes: str = ""

    def normalized(self) -> "SwingTrade":
        direction = str(self.direction).strip().lower()
        status = str(self.status).strip().lower()
        stop_type = str(self.stop_type).strip().lower()
        normalized_targets = [float(item) for item in self.targets if float(item) > 0]

        target_price = self.target_price
        if target_price is None and normalized_targets:
            target_price = float(normalized_targets[0])

        return SwingTrade(
            id=str(self.id).strip(),
            ticker=str(self.ticker).strip().upper(),
            direction=direction if direction in VALID_DIRECTIONS else "long",
            setup_type=str(self.setup_type or "").strip(),
            thesis=str(self.thesis or "").strip(),
            entry_price=float(self.entry_price),
            stop_loss=float(self.stop_loss),
            stop_type=stop_type if stop_type in VALID_STOP_TYPES else "structural",
            stop_rationale=str(self.stop_rationale or "").strip(),
            target_price=(None if target_price in (None, "") else float(target_price)),
            targets=normalized_targets,
            time_stop_days=(None if self.time_stop_days is None else int(self.time_stop_days)),
            planned_holding_days=(
                None if self.planned_holding_days is None else int(self.planned_holding_days)
            ),
            risk_percent=float(self.risk_percent),
            position_size=float(self.position_size),
            status=status if status in VALID_STATUSES else "planned",
            entry_date=self.entry_date,
            exit_date=self.exit_date,
            exit_price=self.exit_price,
            exit_reason=str(self.exit_reason or "").strip(),
            realized_pnl=self.realized_pnl,
            realized_r_multiple=self.realized_r_multiple,
            actual_holding_days=self.actual_holding_days,
            discipline_score=self.discipline_score,
            notes=str(self.notes or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "id": normalized.id,
            "ticker": normalized.ticker,
            "direction": normalized.direction,
            "setup_type": normalized.setup_type,
            "thesis": normalized.thesis,
            "entry_price": normalized.entry_price,
            "stop_loss": normalized.stop_loss,
            "stop_type": normalized.stop_type,
            "stop_rationale": normalized.stop_rationale,
            "target_price": normalized.target_price,
            "targets": [float(item) for item in normalized.targets],
            "time_stop_days": normalized.time_stop_days,
            "planned_holding_days": normalized.planned_holding_days,
            "risk_percent": normalized.risk_percent,
            "position_size": normalized.position_size,
            "status": normalized.status,
            "entry_date": normalized.entry_date.isoformat() if normalized.entry_date else None,
            "exit_date": normalized.exit_date.isoformat() if normalized.exit_date else None,
            "exit_price": normalized.exit_price,
            "exit_reason": normalized.exit_reason,
            "realized_pnl": normalized.realized_pnl,
            "realized_r_multiple": normalized.realized_r_multiple,
            "actual_holding_days": normalized.actual_holding_days,
            "discipline_score": normalized.discipline_score,
            "notes": normalized.notes,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SwingTrade":
        return SwingTrade(
            id=str(payload.get("id", "")).strip(),
            ticker=str(payload.get("ticker", "")).strip().upper(),
            direction=str(payload.get("direction", "long")).strip().lower(),  # type: ignore[arg-type]
            setup_type=str(payload.get("setup_type", "")).strip(),
            thesis=str(payload.get("thesis", "")).strip(),
            entry_price=_to_float(payload.get("entry_price"), default=0.0),
            stop_loss=_to_float(payload.get("stop_loss"), default=0.0),
            stop_type=str(payload.get("stop_type", "structural")).strip().lower(),  # type: ignore[arg-type]
            stop_rationale=str(payload.get("stop_rationale", "")).strip(),
            target_price=_to_optional_float(payload.get("target_price")),
            targets=_parse_targets(payload.get("targets", [])),
            time_stop_days=_to_optional_int(payload.get("time_stop_days")),
            planned_holding_days=_to_optional_int(payload.get("planned_holding_days")),
            risk_percent=_to_float(payload.get("risk_percent"), default=0.0),
            position_size=_to_float(payload.get("position_size"), default=0.0),
            status=str(payload.get("status", "planned")).strip().lower(),  # type: ignore[arg-type]
            entry_date=_parse_date(payload.get("entry_date")),
            exit_date=_parse_date(payload.get("exit_date")),
            exit_price=_to_optional_float(payload.get("exit_price")),
            exit_reason=str(payload.get("exit_reason", "")).strip(),
            realized_pnl=_to_optional_float(payload.get("realized_pnl")),
            realized_r_multiple=_to_optional_float(payload.get("realized_r_multiple")),
            actual_holding_days=_to_optional_int(payload.get("actual_holding_days")),
            discipline_score=_to_optional_float(payload.get("discipline_score")),
            notes=str(payload.get("notes", "")).strip(),
        ).normalized()

    def validate(self) -> list[str]:
        normalized = self.normalized()
        errors: list[str] = []

        if not normalized.id:
            errors.append("Trade id is required.")
        if not normalized.ticker:
            errors.append("Ticker is required.")
        if normalized.direction not in VALID_DIRECTIONS:
            errors.append("Direction must be long or short.")
        if normalized.entry_price <= 0:
            errors.append("Entry price must be greater than zero.")
        if normalized.stop_loss <= 0:
            errors.append("Stop-loss must be greater than zero.")
        if normalized.stop_type not in VALID_STOP_TYPES:
            errors.append("Stop type is invalid.")
        if normalized.status not in VALID_STATUSES:
            errors.append("Trade status is invalid.")
        if not normalized.stop_rationale:
            errors.append("Stop-loss rationale is required.")
        if normalized.position_size <= 0:
            errors.append("Position size must be greater than zero.")
        if normalized.risk_percent <= 0:
            errors.append("Risk percent must be greater than zero.")
        if normalized.time_stop_days is not None and normalized.time_stop_days <= 0:
            errors.append("Time stop days must be positive when provided.")
        if (
            normalized.planned_holding_days is not None
            and normalized.planned_holding_days <= 0
        ):
            errors.append("Planned holding days must be positive when provided.")
        if normalized.exit_price is not None and normalized.exit_price <= 0:
            errors.append("Exit price must be greater than zero when provided.")
        if normalized.target_price is not None and normalized.target_price <= 0:
            errors.append("Target price must be greater than zero when provided.")

        return errors
