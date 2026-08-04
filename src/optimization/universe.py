from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


_DATE_COLUMNS = ("date", "as_of", "effective_date", "membership_date")
_SYMBOL_COLUMNS = ("ticker", "symbol", "asset")
_MEMBER_COLUMNS = ("is_member", "member", "included", "active")


def _normalized_columns(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        str(column).strip().casefold().replace(" ", "_"): column
        for column in frame.columns
    }


def _find_column(columns: dict[str, Any], candidates: Sequence[str]) -> Any | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _parse_membership_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        if float(value) in {0.0, 1.0}:
            return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "y", "active", "included", "member"}:
        return True
    if normalized in {"0", "false", "no", "n", "inactive", "excluded", "not_member"}:
        return False
    raise ValueError(f"cannot parse membership value {value!r}.")


def parse_point_in_time_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse long- or wide-form historical universe membership data."""
    if frame is None or frame.empty:
        raise ValueError("point-in-time membership data is empty.")
    columns = _normalized_columns(frame)
    date_column = _find_column(columns, _DATE_COLUMNS)
    if date_column is None:
        raise ValueError("point-in-time membership data requires a date column.")
    symbol_column = _find_column(columns, _SYMBOL_COLUMNS)
    member_column = _find_column(columns, _MEMBER_COLUMNS)

    source = frame.copy()
    source[date_column] = pd.to_datetime(source[date_column], errors="raise").dt.normalize()
    if symbol_column is not None and member_column is not None:
        source[symbol_column] = source[symbol_column].astype(str).str.strip().str.upper()
        if (source[symbol_column] == "").any():
            raise ValueError("point-in-time membership contains an empty symbol.")
        source[member_column] = source[member_column].map(_parse_membership_value)
        membership = source.pivot_table(
            index=date_column,
            columns=symbol_column,
            values=member_column,
            aggfunc="last",
        )
    else:
        value_columns = [column for column in source.columns if column != date_column]
        if not value_columns:
            raise ValueError("wide point-in-time membership has no symbol columns.")
        membership = source.set_index(date_column)[value_columns]
        membership.columns = [str(column).strip().upper() for column in membership.columns]
        membership = membership.apply(
            lambda column: column.map(_parse_membership_value)
        )

    if membership.columns.duplicated().any():
        raise ValueError("point-in-time membership symbols must be unique.")
    membership = membership.sort_index()
    membership = membership[~membership.index.duplicated(keep="last")]
    return membership.astype("boolean")


def align_point_in_time_membership(
    membership: pd.DataFrame,
    *,
    return_index: pd.Index,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Align membership causally: forward-fill known states, never back-fill."""
    if membership is None or membership.empty:
        raise ValueError("point-in-time membership data is empty.")
    names = [str(symbol) for symbol in symbols]
    supplied = [str(column) for column in membership.columns]
    missing = [symbol for symbol in names if symbol not in supplied]
    extra = [symbol for symbol in supplied if symbol not in names]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("not in returns: " + ", ".join(extra))
        raise ValueError("membership columns must match return columns (" + "; ".join(details) + ").")

    target_index = pd.DatetimeIndex(pd.to_datetime(return_index)).normalize()
    if target_index.tz is not None:
        target_index = target_index.tz_localize(None)
    if target_index.has_duplicates or not target_index.is_monotonic_increasing:
        raise ValueError("return index must be unique and increasing.")
    source = membership.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index)).normalize()
    if source.index.tz is not None:
        source.index = source.index.tz_localize(None)
    source = source.sort_index()[names]
    source = source[~source.index.duplicated(keep="last")]
    combined_index = source.index.union(target_index).sort_values()
    aligned = source.reindex(combined_index).ffill().reindex(target_index).fillna(False)
    aligned.index = return_index
    return aligned.astype(bool)
