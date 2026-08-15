"""Small DB-API and JSON helpers shared by governance stores.

The helpers intentionally depend only on the Python standard library and a
minimal DB-API surface.  This keeps the stores compatible with both sqlite3
and the libSQL/Turso connection used by the application.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def iso_date(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{name} must not be empty.")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD).") from exc


def text(value: Any, name: str, *, required: bool = False, limit: int = 10_000) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


def ticker(value: Any) -> str:
    result = text(value, "Ticker", required=True, limit=32).upper()
    return result


def enum(value: Any, name: str, allowed: Iterable[str]) -> str:
    choices = frozenset(allowed)
    result = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if result not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}.")
    return result


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        number = int(value)
        original = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if number <= 0 or not math.isfinite(original) or original != float(number):
        raise ValueError(f"{name} must be a positive integer.")
    return number


def finite_number(
    value: Any,
    name: str,
    *,
    optional: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{name} must be numeric.")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return number


def json_object(value: Mapping[str, Any] | None, name: str = "Payload") -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    try:
        copied = deepcopy(dict(value))
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only valid finite JSON values.") from exc
    return copied, encoded


def json_array(value: Any, name: str = "Value") -> tuple[list[Any], str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON array.")
    try:
        copied = deepcopy(list(value))
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only valid finite JSON values.") from exc
    return copied, encoded


def decode_object(value: Any) -> dict[str, Any] | None:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def decode_array(value: Any) -> list[Any] | None:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, list) else None


def canonical_hash(value: Mapping[str, Any] | list[Any]) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Hash input must contain only valid finite JSON values.") from exc
    return hashlib.sha256(encoded).hexdigest()


def row_value(row: Any, name: str, index: int) -> Any:
    try:
        keys = row.keys()
    except (AttributeError, TypeError):
        keys = ()
    if keys:
        by_lower_name = {str(key).lower(): key for key in keys}
        actual = by_lower_name.get(name.lower())
        if actual is not None:
            try:
                return row[actual]
            except (KeyError, TypeError, IndexError):
                pass
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return row[index]


def ensure_schema(connection: Any, statements: Iterable[str]) -> None:
    for statement in statements:
        connection.execute(statement)


def commit_and_sync(connection: Any) -> None:
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


def inserted_id(connection: Any, cursor: Any, table: str) -> int:
    candidate = getattr(cursor, "lastrowid", None)
    try:
        identifier = int(candidate)
    except (TypeError, ValueError):
        identifier = 0
    if identifier <= 0:
        row = connection.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        identifier = int(row_value(row, "MAX(id)", 0) or 0)
    if identifier <= 0:
        raise RuntimeError(f"Could not determine inserted id for {table}.")
    return identifier


__all__ = [
    "boolean",
    "canonical_hash",
    "commit_and_sync",
    "decode_array",
    "decode_object",
    "ensure_schema",
    "enum",
    "finite_number",
    "inserted_id",
    "iso_date",
    "json_array",
    "json_object",
    "positive_int",
    "row_value",
    "text",
    "ticker",
    "utc_timestamp",
]
