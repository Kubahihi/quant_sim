"""Append-only competition audit records.

The audit is intentionally separate from investment conclusions: it records how
AI was used and how a decision was challenged, without rewriting the underlying
thesis or decision history.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS analytical_ai_usage_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        purpose TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        prompt_summary TEXT NOT NULL,
        output_used TEXT NOT NULL,
        verification_notes TEXT NOT NULL,
        citation TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (length(trim(purpose)) > 0),
        CHECK (length(trim(tool_name)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analytical_red_team_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        decision_id INTEGER,
        strongest_counterargument TEXT NOT NULL,
        disconfirming_evidence TEXT NOT NULL,
        rejected_alternative TEXT NOT NULL,
        verdict TEXT NOT NULL,
        reviewed_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK (ticker IS NOT NULL OR decision_id IS NOT NULL),
        CHECK (decision_id IS NULL OR decision_id > 0),
        CHECK (length(trim(strongest_counterargument)) > 0),
        CHECK (verdict IN ('proceed', 'revise', 'reject', 'monitor')),
        CHECK (length(trim(reviewed_by)) > 0)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_usage_created ON analytical_ai_usage_log(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_red_team_ticker_created ON analytical_red_team_reviews(ticker, created_at)",
)


def _now(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _text(value: Any, name: str, *, required: bool = False, limit: int = 10_000) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


def _ensure(connection: Any) -> None:
    for statement in _SCHEMA:
        connection.execute(statement)


def _commit(connection: Any) -> None:
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


def _row(row: Any, name: str, index: int) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return row[index]


def _last_id(connection: Any, cursor: Any, table: str) -> int:
    value = getattr(cursor, "lastrowid", None)
    try:
        identifier = int(value)
    except (TypeError, ValueError):
        identifier = 0
    if identifier <= 0:
        identifier = int(connection.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0])
    return identifier


def init_competition_audit_tables(connection: Any) -> None:
    _ensure(connection)
    _commit(connection)


def append_ai_usage(
    connection: Any,
    purpose: str,
    tool_name: str,
    *,
    prompt_summary: str = "",
    output_used: str = "",
    verification_notes: str = "",
    citation: str = "",
    recorded_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    _ensure(connection)
    values = (
        _text(purpose, "Purpose", required=True, limit=500),
        _text(tool_name, "Tool name", required=True, limit=200),
        _text(prompt_summary, "Prompt summary"),
        _text(output_used, "Output used"),
        _text(verification_notes, "Verification notes"),
        _text(citation, "Citation", limit=2_000),
        _text(recorded_by, "Recorded by", required=True, limit=200),
        _now(now),
    )
    cursor = connection.execute(
        """INSERT INTO analytical_ai_usage_log
        (purpose, tool_name, prompt_summary, output_used, verification_notes,
         citation, recorded_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        values,
    )
    identifier = _last_id(connection, cursor, "analytical_ai_usage_log")
    _commit(connection)
    return next(item for item in list_ai_usage(connection) if item["id"] == identifier)


def list_ai_usage(connection: Any) -> list[dict[str, Any]]:
    _ensure(connection)
    rows = connection.execute(
        """SELECT id, purpose, tool_name, prompt_summary, output_used,
        verification_notes, citation, recorded_by, created_at
        FROM analytical_ai_usage_log ORDER BY id DESC"""
    ).fetchall()
    fields = ("id", "purpose", "tool_name", "prompt_summary", "output_used",
              "verification_notes", "citation", "recorded_by", "created_at")
    return [{name: _row(row, name, index) for index, name in enumerate(fields)} for row in rows]


def append_red_team_review(
    connection: Any,
    *,
    ticker: str | None = None,
    decision_id: int | None = None,
    strongest_counterargument: str,
    disconfirming_evidence: str = "",
    rejected_alternative: str = "",
    verdict: str = "monitor",
    reviewed_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    code = _text(ticker, "Ticker", limit=32).upper() or None
    linked_decision = None if decision_id in (None, "") else int(decision_id)
    if code is None and linked_decision is None:
        raise ValueError("Provide a ticker or decision id.")
    if linked_decision is not None and linked_decision <= 0:
        raise ValueError("Decision id must be positive.")
    outcome = str(verdict or "").strip().lower()
    if outcome not in {"proceed", "revise", "reject", "monitor"}:
        raise ValueError("Verdict must be proceed, revise, reject, or monitor.")
    _ensure(connection)
    values = (
        code,
        linked_decision,
        _text(strongest_counterargument, "Strongest counterargument", required=True),
        _text(disconfirming_evidence, "Disconfirming evidence"),
        _text(rejected_alternative, "Rejected alternative"),
        outcome,
        _text(reviewed_by, "Reviewed by", required=True, limit=200),
        _now(now),
    )
    cursor = connection.execute(
        """INSERT INTO analytical_red_team_reviews
        (ticker, decision_id, strongest_counterargument, disconfirming_evidence,
         rejected_alternative, verdict, reviewed_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        values,
    )
    identifier = _last_id(connection, cursor, "analytical_red_team_reviews")
    _commit(connection)
    return next(item for item in list_red_team_reviews(connection) if item["id"] == identifier)


def list_red_team_reviews(connection: Any, *, ticker: str | None = None) -> list[dict[str, Any]]:
    _ensure(connection)
    query = """SELECT id, ticker, decision_id, strongest_counterargument,
        disconfirming_evidence, rejected_alternative, verdict, reviewed_by, created_at
        FROM analytical_red_team_reviews"""
    parameters: tuple[Any, ...] = ()
    if ticker is not None:
        query += " WHERE ticker = ?"
        parameters = (_text(ticker, "Ticker", required=True, limit=32).upper(),)
    rows = connection.execute(query + " ORDER BY id DESC", parameters).fetchall()
    fields = ("id", "ticker", "decision_id", "strongest_counterargument",
              "disconfirming_evidence", "rejected_alternative", "verdict",
              "reviewed_by", "created_at")
    return [{name: _row(row, name, index) for index, name in enumerate(fields)} for row in rows]


__all__ = [
    "append_ai_usage", "append_red_team_review", "init_competition_audit_tables",
    "list_ai_usage", "list_red_team_reviews",
]
