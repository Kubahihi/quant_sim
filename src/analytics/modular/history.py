from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from .results import RunRecord


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def ensure_history_dir(base_dir: str | Path = "data/run_history") -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_record(record: RunRecord, base_dir: str | Path = "data/run_history") -> Path:
    history_dir = ensure_history_dir(base_dir)
    target = history_dir / f"{record.run_id}.json"
    target.write_text(json.dumps(record.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    return target


def load_run_record(run_id: str, base_dir: str | Path = "data/run_history") -> Dict[str, Any]:
    path = ensure_history_dir(base_dir) / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run id not found: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_records(base_dir: str | Path = "data/run_history", limit: int = 50) -> List[Dict[str, Any]]:
    history_dir = ensure_history_dir(base_dir)
    rows: List[Dict[str, Any]] = []
    for file in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            rows.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            continue
        if len(rows) >= max(1, int(limit)):
            break
    return rows


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
