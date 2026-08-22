"""Maintenance operations for the deprecated decision journal."""

from __future__ import annotations

from typing import Any


def _commit_and_sync(connection: Any) -> None:
    connection.commit()
    sync = getattr(connection, "sync", None)
    if callable(sync):
        sync()


def delete_legacy_decision(connection: Any, decision_id: int) -> bool:
    """Delete one legacy decision and records that exist only to describe it.

    Canonical Investment Committee lifecycles are stored in separate tables and
    are intentionally outside the scope of this maintenance operation.
    """
    record_id = int(decision_id)
    if record_id <= 0:
        raise ValueError("Decision id must be positive.")

    existing = connection.execute(
        "SELECT id FROM decision_log WHERE id = ?",
        (record_id,),
    ).fetchone()
    if existing is None:
        return False

    cursor = connection.execute(
        "DELETE FROM decision_log WHERE id = ?",
        (record_id,),
    )
    changed = int(getattr(cursor, "rowcount", 0) or 0) > 0
    if changed:
        connection.execute(
            "DELETE FROM decision_edit_log WHERE decision_id = ?",
            (record_id,),
        )
        connection.execute(
            "DELETE FROM analytical_decision_reviews WHERE decision_id = ?",
            (record_id,),
        )
    _commit_and_sync(connection)
    return changed


__all__ = ["delete_legacy_decision"]
