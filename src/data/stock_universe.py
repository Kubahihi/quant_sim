from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
import inspect
import json

import pandas as pd
import streamlit as st

from src.data.universe_enrichment import (
    NUMERIC_COLUMNS,
    STANDARD_COLUMNS,
    TEXT_COLUMNS,
    enrich_universe_candidates,
)
from src.data.universe_sources import gather_universe_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_DIR = PROJECT_ROOT / "data" / "cache" / "universe"
RAW_UNIVERSE_PATH = UNIVERSE_DIR / "raw_universe.csv"
SNAPSHOT_PARQUET_PATH = UNIVERSE_DIR / "universe_snapshot.parquet"
SNAPSHOT_CSV_PATH = UNIVERSE_DIR / "universe_snapshot.csv"
METADATA_PATH = UNIVERSE_DIR / "universe_metadata.json"


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    progress: float,
    stage: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return

    safe_progress = max(0.0, min(1.0, float(progress)))
    payload: dict[str, Any] = {
        "progress": safe_progress,
        "stage": stage,
        "message": message,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    progress_callback(payload)


def _empty_snapshot() -> pd.DataFrame:
    frame = pd.DataFrame(columns=STANDARD_COLUMNS)
    return _normalize_snapshot_columns(frame)


def _ensure_storage_dirs() -> None:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_metadata_from_disk() -> dict:
    if not METADATA_PATH.exists():
        return {}

    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(metadata: dict) -> None:
    _ensure_storage_dirs()
    payload = {**metadata, "updated_at": _safe_timestamp_now()}
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_universe_snapshot_from_disk() -> pd.DataFrame:
    if SNAPSHOT_PARQUET_PATH.exists():
        try:
            data = pd.read_parquet(SNAPSHOT_PARQUET_PATH)
            return _normalize_snapshot_columns(data)
        except Exception:
            pass

    if SNAPSHOT_CSV_PATH.exists():
        try:
            data = pd.read_csv(SNAPSHOT_CSV_PATH)
            return _normalize_snapshot_columns(data)
        except Exception:
            pass

    return _empty_snapshot()


def _normalize_snapshot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[STANDARD_COLUMNS]

    for column in TEXT_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype("object")

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def _save_snapshot_to_disk(snapshot: pd.DataFrame) -> None:
    _ensure_storage_dirs()
    data = _normalize_snapshot_columns(snapshot)

    parquet_ok = False
    try:
        data.to_parquet(SNAPSHOT_PARQUET_PATH, index=False)
        parquet_ok = True
    except Exception:
        parquet_ok = False

    data.to_csv(SNAPSHOT_CSV_PATH, index=False)
    _write_metadata({
        "last_refresh": _safe_timestamp_now(),
        "rows": int(len(data)),
        "columns": list(data.columns),
        "storage": "parquet+csv" if parquet_ok else "csv",
        "status": "ok",
    })


@st.cache_data(ttl=300, show_spinner=False)
def load_universe_metadata() -> dict:
    """Load universe snapshot metadata from persistent storage."""
    return _read_metadata_from_disk()


@st.cache_data(ttl=900, show_spinner=False)
def load_universe_snapshot() -> pd.DataFrame:
    """Load the latest universe snapshot from persistent local storage."""
    return _load_universe_snapshot_from_disk()


def build_universe_snapshot(progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
    """
    Build and persist a new daily universe snapshot.

    The function is fault tolerant and always writes a metadata record.
    """
    _emit_progress(progress_callback, 0.02, "bootstrap", "Initializing universe build")
    _ensure_storage_dirs()
    previous_snapshot = _load_universe_snapshot_from_disk()

    _emit_progress(progress_callback, 0.06, "raw_sources", "Collecting ticker universe from sources")
    candidates = gather_universe_candidates()
    if candidates.empty:
        raise RuntimeError("No ticker candidates were collected from any source.")
    _emit_progress(
        progress_callback,
        0.10,
        "raw_sources",
        f"Collected {len(candidates):,} candidate symbols",
        current=len(candidates),
        total=len(candidates),
    )

    try:
        candidates.to_csv(RAW_UNIVERSE_PATH, index=False)
    except Exception:
        # Raw layer write failure should not block snapshot generation.
        pass

    _emit_progress(progress_callback, 0.14, "enrichment", "Starting enrichment stage")
    enrich_signature = inspect.signature(enrich_universe_candidates)
    if "progress_callback" in enrich_signature.parameters:
        enriched = enrich_universe_candidates(
            candidates,
            previous_snapshot=previous_snapshot,
            progress_callback=progress_callback,
        )
    else:
        # Compatibility path when an older enrichment module is still loaded
        # in the current process.
        enriched = enrich_universe_candidates(
            candidates,
            previous_snapshot=previous_snapshot,
        )
    if enriched.empty:
        raise RuntimeError("Universe enrichment produced an empty snapshot.")

    _emit_progress(progress_callback, 0.99, "persist", "Saving snapshot to disk")
    _save_snapshot_to_disk(enriched)
    load_universe_snapshot.clear()
    load_universe_metadata.clear()
    get_universe.clear()
    _emit_progress(progress_callback, 1.0, "done", f"Universe refresh completed ({len(enriched):,} symbols)")
    return enriched


def _is_snapshot_fresh(metadata: dict, max_age_hours: int) -> bool:
    last_refresh_raw = str(metadata.get("last_refresh", "")).strip()
    if not last_refresh_raw:
        return False

    try:
        last_refresh = datetime.fromisoformat(last_refresh_raw.replace("Z", "+00:00"))
    except Exception:
        return False

    age = datetime.now(timezone.utc) - last_refresh.astimezone(timezone.utc)
    return age <= timedelta(hours=max_age_hours)


def refresh_universe_if_stale(
    max_age_hours: int = 24,
    force_refresh: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Return a fresh universe snapshot when needed.

    - Uses existing snapshot instantly when it is still fresh.
    - Rebuilds when stale or when forced.
    - Falls back to the last valid snapshot if rebuild fails.
    """
    metadata = _read_metadata_from_disk()
    current_snapshot = _load_universe_snapshot_from_disk()

    should_refresh = force_refresh
    if current_snapshot.empty:
        should_refresh = True
    elif not force_refresh:
        should_refresh = not _is_snapshot_fresh(metadata, max_age_hours=max_age_hours)

    if not should_refresh:
        _emit_progress(progress_callback, 1.0, "cached", "Using fresh cached universe snapshot")
        return current_snapshot

    try:
        return build_universe_snapshot(progress_callback=progress_callback)
    except Exception as exc:
        _write_metadata({
            **metadata,
            "status": "fallback",
            "fallback_reason": str(exc),
            "fallback_at": _safe_timestamp_now(),
            "rows": int(len(current_snapshot)),
        })
        _emit_progress(progress_callback, 1.0, "fallback", "Refresh failed, using last valid snapshot")
        if not current_snapshot.empty:
            return current_snapshot
        return _empty_snapshot()


@st.cache_data(ttl=1800, show_spinner=False)
def get_universe(max_age_hours: int = 24, force_refresh: bool = False) -> pd.DataFrame:
    """Main runtime accessor for the cached stock universe snapshot."""
    return refresh_universe_if_stale(max_age_hours=max_age_hours, force_refresh=force_refresh)
