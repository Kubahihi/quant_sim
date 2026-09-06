"""Inspect or remove credential fields from local QuantSim analysis history.

Only history JSON and history rows in local SQLite databases are inspected.
Output includes counts and locations, never credential values. Remote databases,
provider credentials, authentication records, and backups elsewhere are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.redaction import redact_sensitive_data


def sanitize_local_history(data_dir: Path, *, apply: bool = False) -> dict:
    counts = Counter()
    changed_locations: list[str] = []

    def sanitized(content: str, location: str) -> str | None:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("History record must be an object.")
        counts["records_scanned"] += 1
        clean = redact_sensitive_data(payload)
        if clean == payload:
            return None
        counts["records_requiring_redaction"] += 1
        changed_locations.append(location)
        return json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=True) + "\n"

    for path in sorted(data_dir.rglob("*.json")):
        if "run_history" not in path.parts:
            continue
        try:
            clean = sanitized(path.read_text(encoding="utf-8"), str(path))
            if apply and clean is not None:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                     prefix=".redacted-", delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(clean)
                    os.replace(temporary, path)
                    counts["records_redacted"] += 1
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
        except (OSError, ValueError):
            counts["unreadable_records"] += 1

    for path in sorted(data_dir.glob("*.db*")):
        if path.name.endswith(("-wal", "-shm", "-journal")) or not path.is_file():
            continue
        mode = "rw" if apply else "ro"
        connection = None
        try:
            connection = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True)
            if apply:
                connection.execute("PRAGMA secure_delete=ON")
                connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts["local_databases_scanned"] += 1
            updated = 0
            for table, condition in (("user_data", " WHERE data_type='run_history'"),
                                     ("team_quant_run_history", "")):
                if table not in tables:
                    continue
                # Table names and predicates are fixed constants, never user input.
                for row_id, content in connection.execute(f"SELECT rowid, content_json FROM {table}{condition}").fetchall():
                    try:
                        clean = sanitized(content, f"{path}:{table}:{row_id}")
                    except (TypeError, ValueError):
                        counts["unreadable_records"] += 1
                        continue
                    if apply and clean is not None:
                        connection.execute(f"UPDATE {table} SET content_json=? WHERE rowid=?", (clean, row_id))
                        updated += 1
            if apply:
                connection.commit()
                counts["records_redacted"] += updated
        except (sqlite3.Error, OSError):
            if connection is not None:
                connection.rollback()
            counts["unreadable_databases"] += 1
        finally:
            if connection is not None:
                connection.close()

    return {"mode": "apply" if apply else "inspect", "scope": "local_analysis_history",
            "counts": dict(counts), "locations": changed_locations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--apply", action="store_true", help="Remove sensitive fields; otherwise only inspect.")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error("data directory does not exist")
    report = sanitize_local_history(args.data_dir.resolve(), apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    counts = report["counts"]
    return 1 if counts.get("unreadable_records", 0) or counts.get("unreadable_databases", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
