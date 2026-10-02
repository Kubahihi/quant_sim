"""Explicit, validated imports of periodic investment histories.

Workbook cells are data only. Formulas are never evaluated; XLSX reads use
saved values. Missing observations are not interpolated or silently filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


FREQUENCIES = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annual": 1}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000


@dataclass
class ImportedHistory:
    returns: pd.DataFrame
    frequency: str
    warnings: list[str]
    baseline_date: pd.Timestamp | None = None

    @property
    def periods_per_year(self) -> int:
        return FREQUENCIES[self.frequency]


def _check_file(content: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise ValueError("Choose a CSV or XLSX file.")
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Choose a non-empty file no larger than 20 MB.")
    if suffix == ".xlsx":
        try:
            with ZipFile(BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("Expanded workbook exceeds the 100 MB limit.")
        except BadZipFile as exc:
            raise ValueError("The file is not a valid XLSX workbook.") from exc
    return suffix


def workbook_sheets(content: bytes, filename: str) -> list[str]:
    if _check_file(content, filename) == ".csv":
        return []
    import openpyxl

    workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
    try:
        return workbook.sheetnames
    finally:
        workbook.close()


def read_history_table(
    content: bytes, filename: str, *, sheet: str | None = None,
    header_row: int = 1, separator: str = ",", decimal: str = ".",
) -> pd.DataFrame:
    suffix = _check_file(content, filename)
    if not 1 <= header_row <= 100:
        raise ValueError("Header row must be between 1 and 100.")
    if suffix == ".xlsx":
        raw = pd.read_excel(
            BytesIO(content), sheet_name=sheet or 0, header=None,
            skiprows=header_row - 1, nrows=MAX_ROWS + 2, engine="openpyxl",
        )
    else:
        raw = pd.read_csv(
            BytesIO(content), header=None, skiprows=header_row - 1,
            nrows=MAX_ROWS + 2, sep=separator, decimal=decimal, encoding="utf-8-sig",
        )
    raw = raw.dropna(axis=1, how="all")
    if len(raw) > MAX_ROWS + 1:
        raise ValueError(f"The table exceeds {MAX_ROWS:,} data rows.")
    if raw.empty or raw.shape[1] < 2:
        raise ValueError("Select a header row with a date column and at least one asset.")
    headers = raw.iloc[0]
    if headers.isna().any():
        raise ValueError("Every populated column needs a header.")
    names = headers.astype(str).str.strip()
    if names.eq("").any() or names.duplicated().any():
        raise ValueError("Column headers must be non-empty and unique.")
    table = raw.iloc[1:].copy()
    table.columns = names.tolist()
    table = table.dropna(how="all").reset_index(drop=True)
    # CSV headers force object dtype; apply the selected decimal convention
    # without changing date strings or requiring thousands separators.
    if suffix == ".csv" and decimal == ",":
        for name in table:
            converted = pd.to_numeric(table[name].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if converted.notna().sum() == table[name].notna().sum():
                table[name] = converted
    return table


def prepare_history(
    table: pd.DataFrame, *, date_column: str, asset_columns: list[str],
    frequency: str, value_type: str, date_format: str = "ISO / Excel dates",
    return_unit: str = "Decimal", names: list[str] | None = None,
) -> ImportedHistory:
    if frequency not in FREQUENCIES or value_type not in {"Returns", "Prices / index levels"}:
        raise ValueError("Choose a supported frequency and data type.")
    if return_unit not in {"Decimal", "Percent"}:
        raise ValueError("Choose decimal or percent returns.")
    if date_format not in {"ISO / Excel dates", "Day / month / year", "Month / day / year"}:
        raise ValueError("Choose a supported date format.")
    if not asset_columns or len(set(asset_columns)) != len(asset_columns):
        raise ValueError("Select at least one unique asset column.")
    if date_column in asset_columns or any(c not in table for c in [date_column, *asset_columns]):
        raise ValueError("Date and asset columns must be separate and present in the table.")
    if table.empty:
        raise ValueError("Provide at least two return observations (or three price observations).")
    labels = [str(name).strip() for name in (names if names is not None else asset_columns)]
    if len(labels) != len(asset_columns) or not all(labels) or len(set(labels)) != len(labels):
        raise ValueError("Asset names must be non-empty and unique.")
    raw_dates = table[date_column]
    if pd.api.types.is_numeric_dtype(raw_dates) or raw_dates.map(
        lambda value: isinstance(value, (int, float, np.number)) and not pd.isna(value)
    ).any():
        raise ValueError("Use formatted Excel dates or date strings, not numeric date codes.")
    formats = {"ISO / Excel dates": "ISO8601", "Day / month / year": "%d/%m/%Y", "Month / day / year": "%m/%d/%Y"}
    dates = pd.to_datetime(raw_dates, format=formats[date_format], errors="coerce")
    if dates.isna().any():
        raise ValueError("Some dates are missing or invalid. Check the date format and selected rows.")
    try:
        index = pd.DatetimeIndex(dates).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Use dates in one consistent timezone.") from exc
    if index.has_duplicates:
        raise ValueError("Duplicate dates found. Keep one observation per date.")
    values = table[asset_columns].apply(pd.to_numeric, errors="coerce")
    values.index = index
    values.columns = labels
    warnings = []
    if not index.is_monotonic_increasing:
        warnings.append("Rows were sorted from oldest to newest.")
    values = values.sort_index()
    # A common source convention includes a dated initial baseline with no
    # return. Only this single all-empty leading row is safely removable.
    leading_source_row = table.iloc[int(np.argmin(index.asi8))][asset_columns]
    baseline_date = None
    if value_type == "Returns" and len(values) and leading_source_row.isna().all():
        baseline_date = values.index[0]
        values = values.iloc[1:]
        warnings.append("Excluded the initial baseline row, which has no returns.")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Selected assets contain missing or non-numeric values. No gaps are filled automatically.")
    if len(values) < (3 if value_type == "Prices / index levels" else 2):
        raise ValueError("Provide at least two return observations (or three price observations).")
    period_codes = {"Weekly": "W-SUN", "Monthly": "M", "Quarterly": "Q", "Annual": "Y"}
    if frequency in period_codes:
        ordinal = values.index.to_period(period_codes[frequency]).asi8
        if not np.all(np.diff(ordinal) == 1):
            raise ValueError(f"Dates must contain exactly one observation per consecutive {frequency.lower()} period.")
    else:
        gaps = np.diff(values.index.values).astype("timedelta64[D]").astype(int)
        if np.median(gaps) > 3:
            raise ValueError("Dates do not look daily. Select the matching frequency.")
        if gaps.max() > 7:
            raise ValueError("Daily history contains a gap longer than seven days. Supply a continuous history.")
        warnings.append("Daily annualization assumes 252 trading sessions per year; exchange holidays are not independently verified.")
    if value_type == "Prices / index levels":
        if (values <= 0).any().any():
            raise ValueError("Prices and index levels must be strictly positive.")
        baseline_date = values.index[0]
        returns = values.pct_change(fill_method=None).iloc[1:]
    else:
        returns = values / (100.0 if return_unit == "Percent" else 1.0)
    if (returns <= -1).any().any():
        raise ValueError("Returns must be greater than -100%. Check units and data type.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("The computed returns are not finite. Check the supplied values.")
    returns.index.name = "Date"
    return ImportedHistory(returns.astype(float), frequency, warnings, baseline_date)
