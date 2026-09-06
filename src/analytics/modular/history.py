from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from .results import RunRecord
from src.utils.redaction import redact_sensitive_data, sanitize_run_config

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Legacy run history directory (for backward compatibility)
LEGACY_HISTORY_DIR = PROJECT_ROOT / "data" / "run_history"


def _get_history_dir(user_id: int | None = None) -> Path:
    """
    Get the run history directory for a user.
    
    If user_id is provided, returns user-specific directory.
    Otherwise, returns the legacy directory for backward compatibility.
    """
    if user_id is not None:
        user_dir = PROJECT_ROOT / "data" / "users" / str(user_id) / "run_history"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    LEGACY_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return LEGACY_HISTORY_DIR


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def ensure_history_dir(base_dir: str | Path | None = None, user_id: int | None = None) -> Path:
    """
    Ensure the run history directory exists.
    
    If base_dir is explicitly provided, use it (for testing/custom paths).
    If user_id is provided, use user-specific directory.
    Otherwise, use legacy directory for backward compatibility.
    """
    if base_dir is not None:
        path = Path(base_dir)
    else:
        path = _get_history_dir(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode_record(content: str) -> Dict[str, Any]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("History record must be an object.")
    payload = redact_sensitive_data(payload)
    if isinstance(payload.get("config"), dict):
        payload["config"] = sanitize_run_config(payload["config"])
    return payload


def save_run_record(record: RunRecord, base_dir: str | Path | None = None, user_id: int | None = None, *, team_connection_factory=None) -> Path | str:
    content = json.dumps(redact_sensitive_data(record.to_dict()), indent=2, default=_json_default)
    if team_connection_factory is not None:
        if user_id is not None:
            raise ValueError("Team history and personal user_id are mutually exclusive.")
        with team_connection_factory() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS team_quant_run_history (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, content_json TEXT NOT NULL)")
            conn.execute("INSERT INTO team_quant_run_history (run_id, created_at, content_json) VALUES (?, ?, ?)",
                         (record.run_id, record.timestamp, content))
            conn.commit()
            if hasattr(conn, "sync"):
                conn.sync()
        return f"db://wharton_team/run_history/{record.run_id}"
    if user_id is not None:
        from src.auth.database import save_user_data
        file_name = f"{record.run_id}.json"
        save_user_data(user_id, "run_history", file_name, content)
        return f"db://user_{user_id}/run_history/{file_name}"
    else:
        history_dir = ensure_history_dir(base_dir)
        target = history_dir / f"{record.run_id}.json"
        target.write_text(content, encoding="utf-8")
        return target


def load_run_record(run_id: str, base_dir: str | Path | None = None, user_id: int | None = None, *, team_connection_factory=None) -> Dict[str, Any]:
    if team_connection_factory is not None:
        if user_id is not None:
            raise ValueError("Team history and personal user_id are mutually exclusive.")
        with team_connection_factory() as conn:
            exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_quant_run_history'").fetchone()
            row = conn.execute("SELECT content_json FROM team_quant_run_history WHERE run_id=?", (run_id,)).fetchone() if exists else None
        if row is None:
            raise FileNotFoundError(f"Run id not found in team DB: {run_id}")
        return _decode_record(row[0])
    if user_id is not None:
        from src.auth.database import load_user_data
        content = load_user_data(user_id, "run_history", f"{run_id}.json")
        if content:
            return _decode_record(content)
        raise FileNotFoundError(f"Run id not found in DB: {run_id}")
    else:
        path = (Path(base_dir) if base_dir is not None else LEGACY_HISTORY_DIR) / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run id not found: {run_id}")
        return _decode_record(path.read_text(encoding="utf-8"))


def list_run_records(base_dir: str | Path | None = None, limit: int = 50, user_id: int | None = None, *, team_connection_factory=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if team_connection_factory is not None:
        if user_id is not None:
            raise ValueError("Team history and personal user_id are mutually exclusive.")
        with team_connection_factory() as conn:
            exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='team_quant_run_history'").fetchone()
            if not exists:
                return []
            records = conn.execute("SELECT content_json FROM team_quant_run_history ORDER BY created_at DESC, run_id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        for record in records:
            try:
                rows.append(_decode_record(record[0]))
            except (TypeError, ValueError):
                continue
        return rows
    
    if user_id is not None:
        from src.auth.database import list_user_data, load_user_data
        file_names = sorted(list_user_data(user_id, "run_history"), reverse=True)
        for fname in file_names:
            content = load_user_data(user_id, "run_history", fname)
            if content:
                try:
                    rows.append(_decode_record(content))
                except Exception:
                    continue
        return sorted(rows, key=lambda row: (str(row.get("timestamp", "")), str(row.get("run_id", ""))), reverse=True)[:max(1, int(limit))]

    history_dir = Path(base_dir) if base_dir is not None else LEGACY_HISTORY_DIR
    for file in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            rows.append(_decode_record(file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(rows, key=lambda row: (str(row.get("timestamp", "")), str(row.get("run_id", ""))), reverse=True)[:max(1, int(limit))]


def compare_runs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_metrics = left.get("metrics", {})
    right_metrics = right.get("metrics", {})
    keys = sorted(set(left_metrics.keys()) | set(right_metrics.keys()))
    diff = {}
    for key in keys:
        lval = left_metrics.get(key)
        rval = right_metrics.get(key)
        if isinstance(lval, (int, float)) and isinstance(rval, (int, float)):
            diff[key] = float(rval - lval)
        else:
            diff[key] = None

    left_summary = left.get("summary", {})
    right_summary = right.get("summary", {})
    summary_diff = {
        "composite_score": float(right_summary.get("composite_score", 0.0) or 0.0)
        - float(left_summary.get("composite_score", 0.0) or 0.0),
        "confidence": float(right_summary.get("confidence", 0.0) or 0.0)
        - float(left_summary.get("confidence", 0.0) or 0.0),
        "news_sentiment": float(right_summary.get("news_sentiment", 0.0) or 0.0)
        - float(left_summary.get("news_sentiment", 0.0) or 0.0),
    }

    return {
        "left_run_id": left.get("run_id"),
        "right_run_id": right.get("run_id"),
        "metric_diff": diff,
        "summary_diff": summary_diff,
        "left_summary": left_summary,
        "right_summary": right_summary,
    }
