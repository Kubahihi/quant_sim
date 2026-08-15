"""Immutable official-rules snapshots with hashes, diffs, and acknowledgements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import difflib
import hashlib
import json
from typing import Any


DEFAULT_RULESETS = (
    "rules_and_roles",
    "trading_rules",
    "case_study",
    "deliverables",
)


def _text(value: Any, name: str, *, required: bool = False, limit: int = 20_000) -> str:
    result = " ".join(str(value or "").strip().split())
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


def _content(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Rules content must be text.")
    if not value.strip():
        raise ValueError("Rules content must not be empty.")
    if len(value) > 5_000_000:
        raise ValueError("Rules content must be at most 5,000,000 characters.")
    return value


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True, limit=160)
    if any(character.isspace() for character in result):
        raise ValueError(f"{name} must not contain whitespace.")
    return result


def _timestamp(value: date | datetime | str | None = None, *, name: str = "Timestamp") -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        raw = _text(value, name, required=True, limit=80)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            try:
                return date.fromisoformat(raw).isoformat()
            except ValueError:
                raise ValueError(f"{name} must be an ISO date or timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _json_copy(value: Any, name: str = "Value") -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serialisable.") from exc
    return json.loads(encoded)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _member(value: Any, watch: Mapping[str, Any], name: str) -> str:
    member = _text(value, name, required=True, limit=200)
    if member not in watch.get("team_members", []):
        raise ValueError(f"{name} must be a registered team member.")
    return member


def _snapshot(watch: Mapping[str, Any], snapshot_id: str) -> dict[str, Any]:
    identifier = _identifier(snapshot_id, "Snapshot id")
    try:
        return watch["snapshots"][identifier]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown rules snapshot id: {identifier}.") from exc


def create_official_rules_watch(
    watch_id: str,
    *,
    competition: str,
    team_members: Sequence[str],
    created_by: str,
    rulesets: Sequence[str] = DEFAULT_RULESETS,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    if isinstance(team_members, (str, bytes, bytearray)):
        raise ValueError("Team members must be a sequence.")
    members: list[str] = []
    for raw in team_members:
        member = _text(raw, "Team member", required=True, limit=200)
        if member in members:
            raise ValueError(f"Duplicate team member: {member}.")
        members.append(member)
    if not members:
        raise ValueError("At least one team member is required.")
    creator = _text(created_by, "Created by", required=True, limit=200)
    if creator not in members:
        raise ValueError("Created by must be a registered team member.")
    if isinstance(rulesets, (str, bytes, bytearray)):
        raise ValueError("Rulesets must be a sequence.")
    rule_keys: list[str] = []
    for raw in rulesets:
        key = _identifier(raw, "Ruleset id")
        if key in rule_keys:
            raise ValueError(f"Duplicate ruleset id: {key}.")
        rule_keys.append(key)
    if not rule_keys:
        raise ValueError("At least one ruleset is required.")
    return {
        "watch_id": _identifier(watch_id, "Watch id"),
        "competition": _text(competition, "Competition", required=True, limit=500),
        "team_members": members,
        "rulesets": {key: {"ruleset_id": key, "snapshot_ids": []} for key in rule_keys},
        "snapshots": {},
        "acknowledgements": {},
        "created_by": creator,
        "created_at": _timestamp(now),
    }


def capture_rules_snapshot(
    watch: Mapping[str, Any],
    ruleset_id: str,
    *,
    source_url: str,
    content: str,
    captured_by: str,
    published_at: date | datetime | str | None = None,
    retrieved_at: date | datetime | str | None = None,
    snapshot_id: str | None = None,
    source_title: str = "",
) -> dict[str, Any]:
    """Append a changed official source; previous snapshot records remain untouched."""
    result = _json_copy(watch, "Rules watch")
    ruleset_key = _identifier(ruleset_id, "Ruleset id")
    if ruleset_key not in result.get("rulesets", {}):
        raise ValueError(f"Unknown ruleset id: {ruleset_key}.")
    actor = _member(captured_by, result, "Captured by")
    body = _content(content)
    content_hash = _sha256(body)
    history = result["rulesets"][ruleset_key]["snapshot_ids"]
    previous_id = history[-1] if history else None
    if previous_id and result["snapshots"][previous_id]["content_hash"] == content_hash:
        raise ValueError("Rules content is unchanged from the latest immutable snapshot.")
    version = len(history) + 1
    identifier = _identifier(snapshot_id or f"{ruleset_key}:v{version}", "Snapshot id")
    if identifier in result["snapshots"]:
        raise ValueError(f"Snapshot id already exists: {identifier}.")
    record = {
        "snapshot_id": identifier,
        "ruleset_id": ruleset_key,
        "version": version,
        "source_title": _text(source_title, "Source title", limit=500),
        "source_url": _text(source_url, "Source URL", required=True, limit=4_000),
        "content": body,
        "content_hash": content_hash,
        "previous_snapshot_id": previous_id,
        "published_at": _timestamp(published_at, name="Published at")
        if published_at is not None
        else None,
        "retrieved_at": _timestamp(retrieved_at, name="Retrieved at"),
        "captured_by": actor,
    }
    result["snapshots"][identifier] = record
    history.append(identifier)
    result["acknowledgements"][identifier] = {}
    return result


def latest_rules_snapshot(
    watch: Mapping[str, Any], ruleset_id: str
) -> dict[str, Any] | None:
    value = _json_copy(watch, "Rules watch")
    key = _identifier(ruleset_id, "Ruleset id")
    if key not in value.get("rulesets", {}):
        raise ValueError(f"Unknown ruleset id: {key}.")
    history = value["rulesets"][key]["snapshot_ids"]
    return value["snapshots"][history[-1]] if history else None


def diff_rules_snapshots(
    watch: Mapping[str, Any],
    from_snapshot_id: str,
    to_snapshot_id: str,
    *,
    context_lines: int = 3,
) -> dict[str, Any]:
    value = _json_copy(watch, "Rules watch")
    before = _snapshot(value, from_snapshot_id)
    after = _snapshot(value, to_snapshot_id)
    if before["ruleset_id"] != after["ruleset_id"]:
        raise ValueError("Rules snapshots must belong to the same ruleset.")
    if isinstance(context_lines, bool) or int(context_lines) != context_lines or context_lines < 0:
        raise ValueError("Context lines must be a non-negative integer.")
    before_lines = before["content"].splitlines()
    after_lines = after["content"].splitlines()
    unified = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before["snapshot_id"],
            tofile=after["snapshot_id"],
            lineterm="",
            n=int(context_lines),
        )
    )
    added = [line[1:] for line in unified if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in unified if line.startswith("-") and not line.startswith("---")]
    return {
        "ruleset_id": before["ruleset_id"],
        "from_snapshot_id": before["snapshot_id"],
        "to_snapshot_id": after["snapshot_id"],
        "from_hash": before["content_hash"],
        "to_hash": after["content_hash"],
        "changed": before["content_hash"] != after["content_hash"],
        "added_line_count": len(added),
        "removed_line_count": len(removed),
        "added_lines": added,
        "removed_lines": removed,
        "unified_diff": unified,
    }


def acknowledge_rules_snapshot(
    watch: Mapping[str, Any],
    snapshot_id: str,
    *,
    member_id: str,
    note: str = "",
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(watch, "Rules watch")
    snapshot = _snapshot(result, snapshot_id)
    if snapshot["content_hash"] != _sha256(snapshot["content"]):
        raise ValueError("Rules snapshot failed its content hash integrity check.")
    member = _member(member_id, result, "Member")
    acknowledgements = result.setdefault("acknowledgements", {}).setdefault(
        snapshot["snapshot_id"], {}
    )
    if member in acknowledgements:
        raise ValueError("Member has already acknowledged this rules snapshot.")
    acknowledgements[member] = {
        "member_id": member,
        "snapshot_id": snapshot["snapshot_id"],
        "content_hash": snapshot["content_hash"],
        "acknowledged_at": _timestamp(now),
        "note": _text(note, "Acknowledgement note", limit=4_000),
    }
    return result


def rules_acknowledgement_status(
    watch: Mapping[str, Any],
    snapshot_id: str,
    *,
    required_members: Sequence[str] | None = None,
) -> dict[str, Any]:
    value = _json_copy(watch, "Rules watch")
    snapshot = _snapshot(value, snapshot_id)
    if isinstance(required_members, (str, bytes, bytearray)):
        raise ValueError("Required members must be a sequence.")
    members = [
        _member(item, value, "Required member")
        for item in (required_members or value["team_members"])
    ]
    members = list(dict.fromkeys(members))
    records = value.get("acknowledgements", {}).get(snapshot["snapshot_id"], {})
    valid = [
        member
        for member in members
        if records.get(member, {}).get("content_hash") == snapshot["content_hash"]
    ]
    missing = [member for member in members if member not in valid]
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "ruleset_id": snapshot["ruleset_id"],
        "content_hash": snapshot["content_hash"],
        "is_fully_acknowledged": not missing,
        "required_members": members,
        "acknowledged_members": valid,
        "missing_members": missing,
        "acknowledged_count": len(valid),
        "required_count": len(members),
    }


def current_rules_watch_status(watch: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate acknowledgements for every latest official-rules snapshot."""
    value = _json_copy(watch, "Rules watch")
    statuses: list[dict[str, Any]] = []
    missing_rulesets: list[str] = []
    for ruleset_id in value.get("rulesets", {}):
        history = value["rulesets"][ruleset_id]["snapshot_ids"]
        if not history:
            missing_rulesets.append(ruleset_id)
            continue
        statuses.append(rules_acknowledgement_status(value, history[-1]))
    unacknowledged = [
        item["ruleset_id"] for item in statuses if not item["is_fully_acknowledged"]
    ]
    return {
        "is_current": not missing_rulesets and not unacknowledged,
        "missing_rulesets": missing_rulesets,
        "unacknowledged_rulesets": unacknowledged,
        "latest_snapshots": statuses,
    }


def verify_rules_watch_integrity(watch: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(watch, "Rules watch")
    issues: list[dict[str, str]] = []
    referenced: list[str] = []
    for ruleset_id, ruleset in value.get("rulesets", {}).items():
        previous: str | None = None
        for expected_version, snapshot_id in enumerate(ruleset.get("snapshot_ids", []), start=1):
            referenced.append(snapshot_id)
            snapshot = value.get("snapshots", {}).get(snapshot_id)
            if snapshot is None:
                issues.append({"code": "missing_snapshot", "snapshot_id": snapshot_id})
                continue
            if snapshot.get("ruleset_id") != ruleset_id:
                issues.append({"code": "ruleset_mismatch", "snapshot_id": snapshot_id})
            if snapshot.get("version") != expected_version:
                issues.append({"code": "version_mismatch", "snapshot_id": snapshot_id})
            if snapshot.get("previous_snapshot_id") != previous:
                issues.append({"code": "chain_mismatch", "snapshot_id": snapshot_id})
            if snapshot.get("content_hash") != _sha256(snapshot.get("content", "")):
                issues.append({"code": "hash_mismatch", "snapshot_id": snapshot_id})
            previous = snapshot_id
    orphaned = sorted(set(value.get("snapshots", {})) - set(referenced))
    for snapshot_id in orphaned:
        issues.append({"code": "orphaned_snapshot", "snapshot_id": snapshot_id})
    return {
        "is_valid": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "snapshot_count": len(value.get("snapshots", {})),
    }


__all__ = [
    "DEFAULT_RULESETS",
    "acknowledge_rules_snapshot",
    "capture_rules_snapshot",
    "create_official_rules_watch",
    "current_rules_watch_status",
    "diff_rules_snapshots",
    "latest_rules_snapshot",
    "rules_acknowledgement_status",
    "verify_rules_watch_integrity",
]
