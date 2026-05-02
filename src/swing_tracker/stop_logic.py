from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StopComputation:
    stop_loss: float
    stop_type: str
    rationale: str
    details: dict[str, Any]


def validate_stop_loss_side(direction: str, entry_price: float, stop_loss: float) -> None:
    normalized_direction = str(direction).strip().lower()
    if normalized_direction == "long" and stop_loss >= entry_price:
        raise ValueError("For long trades, stop-loss must be below entry price.")
    if normalized_direction == "short" and stop_loss <= entry_price:
        raise ValueError("For short trades, stop-loss must be above entry price.")


def calculate_stop_loss(
    *,
    direction: str,
    entry_price: float,
    stop_type: str,
    structural_price: float | None = None,
    atr_value: float | None = None,
    atr_multiple: float | None = None,
    fixed_risk_percent: float | None = None,
    manual_stop_loss: float | None = None,
) -> float:
    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")

    normalized_direction = str(direction).strip().lower()
    normalized_stop_type = str(stop_type).strip().lower()

    if normalized_direction not in {"long", "short"}:
        raise ValueError("Direction must be long or short.")
    if normalized_stop_type not in {"structural", "atr", "fixed_risk", "time_stop"}:
        raise ValueError("Unsupported stop type.")

    stop_loss = 0.0
    if normalized_stop_type == "structural":
        if structural_price is None or structural_price <= 0:
            raise ValueError("Structural stop requires a valid structural price.")
        stop_loss = float(structural_price)
    elif normalized_stop_type == "atr":
        if atr_value is None or atr_value <= 0:
            raise ValueError("ATR stop requires a positive ATR value.")
        multiple = float(atr_multiple) if atr_multiple is not None else 2.0
        if multiple <= 0:
            raise ValueError("ATR multiple must be positive.")
        distance = float(atr_value) * multiple
        stop_loss = entry_price - distance if normalized_direction == "long" else entry_price + distance
    elif normalized_stop_type == "fixed_risk":
        if fixed_risk_percent is None or fixed_risk_percent <= 0:
            raise ValueError("Fixed-risk stop requires a positive risk percentage.")
        distance = entry_price * (float(fixed_risk_percent) / 100.0)
        stop_loss = entry_price - distance if normalized_direction == "long" else entry_price + distance
    else:
        if manual_stop_loss is not None and manual_stop_loss > 0:
            stop_loss = float(manual_stop_loss)
        elif fixed_risk_percent is not None and fixed_risk_percent > 0:
            distance = entry_price * (float(fixed_risk_percent) / 100.0)
            stop_loss = entry_price - distance if normalized_direction == "long" else entry_price + distance
        else:
            raise ValueError(
                "Time-stop entries still require a hard stop. Provide manual stop-loss or fixed risk percent."
            )

    validate_stop_loss_side(normalized_direction, float(entry_price), float(stop_loss))
    return float(round(stop_loss, 6))


def build_stop_rationale(
    *,
    direction: str,
    stop_type: str,
    entry_price: float,
    stop_loss: float,
    time_stop_days: int | None = None,
    atr_value: float | None = None,
    atr_multiple: float | None = None,
    fixed_risk_percent: float | None = None,
) -> str:
    normalized_direction = str(direction).strip().lower()
    normalized_stop_type = str(stop_type).strip().lower()
    risk_distance = abs(float(entry_price) - float(stop_loss))
    risk_pct = (risk_distance / float(entry_price) * 100.0) if entry_price > 0 else 0.0

    if normalized_stop_type == "structural":
        return (
            f"Structural {normalized_direction} stop is set at {stop_loss:.2f}, below/above key structure. "
            f"Risk distance is {risk_distance:.2f} ({risk_pct:.2f}%)."
        )
    if normalized_stop_type == "atr":
        atr_text = "n/a" if atr_value is None else f"{float(atr_value):.4f}"
        multiple_text = "n/a" if atr_multiple is None else f"{float(atr_multiple):.2f}x"
        return (
            f"ATR-based {normalized_direction} stop uses ATR {atr_text} at {multiple_text}, "
            f"placing stop at {stop_loss:.2f}. Risk distance is {risk_distance:.2f} ({risk_pct:.2f}%)."
        )
    if normalized_stop_type == "fixed_risk":
        fixed_text = "n/a" if fixed_risk_percent is None else f"{float(fixed_risk_percent):.2f}%"
        return (
            f"Fixed-risk {normalized_direction} stop uses {fixed_text} of entry, "
            f"setting stop at {stop_loss:.2f}. Risk distance is {risk_distance:.2f} ({risk_pct:.2f}%)."
        )

    time_stop_text = (
        f" Time stop is {int(time_stop_days)} day(s)." if time_stop_days is not None and time_stop_days > 0 else ""
    )
    return (
        f"Time-stop anchored {normalized_direction} plan keeps hard stop at {stop_loss:.2f}. "
        f"Risk distance is {risk_distance:.2f} ({risk_pct:.2f}%).{time_stop_text}"
    )


def validate_stop_rationale(rationale: str) -> bool:
    return bool(str(rationale or "").strip())
