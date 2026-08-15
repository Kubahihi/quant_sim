"""Linked work items for client goals, security dossiers, and decisions.

Tasks inherit omitted context links from a parent subproject, but every stored
item is materialised with concrete links.  This avoids anonymous checklist
items that cannot be traced to the investment process they are meant to serve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
import json
from typing import Any


WORK_ITEM_KINDS = frozenset({"task", "subproject"})
WORK_ITEM_STATUSES = frozenset({"planned", "in_progress", "blocked", "done", "cancelled"})
LINK_FIELDS = ("client_goal_id", "dossier_id", "decision_id")

_TRANSITIONS = {
    "planned": frozenset({"in_progress", "cancelled"}),
    "in_progress": frozenset({"blocked", "done", "cancelled"}),
    "blocked": frozenset({"in_progress", "cancelled"}),
    "done": frozenset(),
    "cancelled": frozenset(),
}


def _text(value: Any, name: str, *, required: bool = False, limit: int = 20_000) -> str:
    result = " ".join(str(value or "").strip().split())
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


def _identifier(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value in (None, ""):
        return None
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
        parsed = datetime.combine(value, time.max, tzinfo=timezone.utc)
    else:
        raw = _text(value, name, required=True, limit=80)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            try:
                parsed = datetime.combine(
                    date.fromisoformat(raw), time.max, tzinfo=timezone.utc
                )
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


def _catalog(values: Sequence[Any], name: str) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence.")
    result: list[str] = []
    for value in values:
        identifier = _identifier(value, name)
        assert identifier is not None
        if identifier in result:
            raise ValueError(f"Duplicate {name.lower()}: {identifier}.")
        result.append(identifier)
    return result


def _member(value: Any, registry: Mapping[str, Any], name: str) -> str:
    result = _text(value, name, required=True, limit=200)
    if result not in registry.get("team_members", []):
        raise ValueError(f"{name} must be a registered team member.")
    return result


def _item(registry: Mapping[str, Any], work_item_id: str) -> dict[str, Any]:
    identifier = _identifier(work_item_id, "Work item id")
    try:
        return registry["items"][identifier]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown work item id: {identifier}.") from exc


def create_work_item_registry(
    registry_id: str,
    *,
    team_members: Sequence[str],
    client_goal_ids: Sequence[str] = (),
    dossier_ids: Sequence[str] = (),
    decision_ids: Sequence[str] = (),
    required_links: Sequence[str] = LINK_FIELDS,
    created_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    members = _catalog(team_members, "Team member")
    if not members:
        raise ValueError("At least one team member is required.")
    creator = _text(created_by, "Created by", required=True, limit=200)
    if creator not in members:
        raise ValueError("Created by must be a registered team member.")
    if isinstance(required_links, (str, bytes, bytearray)):
        raise ValueError("Required links must be a sequence.")
    required: list[str] = []
    for value in required_links:
        field = _text(value, "Required link", required=True, limit=50)
        if field not in LINK_FIELDS:
            raise ValueError(f"Required link must be one of: {', '.join(LINK_FIELDS)}.")
        if field not in required:
            required.append(field)
    if not required:
        raise ValueError("At least one contextual link type must be required.")
    return {
        "registry_id": _identifier(registry_id, "Registry id"),
        "team_members": members,
        "references": {
            "client_goal_id": _catalog(client_goal_ids, "Client goal id"),
            "dossier_id": _catalog(dossier_ids, "Dossier id"),
            "decision_id": _catalog(decision_ids, "Decision id"),
        },
        "required_links": required,
        "items": {},
        "item_order": [],
        "created_by": creator,
        "created_at": _timestamp(now),
    }


def register_work_reference(
    registry: Mapping[str, Any],
    reference_type: str,
    reference_id: str,
) -> dict[str, Any]:
    result = _json_copy(registry, "Work item registry")
    field = _text(reference_type, "Reference type", required=True, limit=50)
    if field not in LINK_FIELDS:
        raise ValueError(f"Reference type must be one of: {', '.join(LINK_FIELDS)}.")
    identifier = _identifier(reference_id, "Reference id")
    if identifier in result["references"][field]:
        raise ValueError(f"Reference already exists: {identifier}.")
    result["references"][field].append(identifier)
    return result


def add_work_item(
    registry: Mapping[str, Any],
    work_item_id: str,
    *,
    title: str,
    kind: str,
    owner: str,
    due_at: date | datetime | str,
    client_goal_id: str | None = None,
    dossier_id: str | None = None,
    decision_id: str | None = None,
    parent_id: str | None = None,
    description: str = "",
    created_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(registry, "Work item registry")
    identifier = _identifier(work_item_id, "Work item id")
    if identifier in result.get("items", {}):
        raise ValueError(f"Work item id already exists: {identifier}.")
    item_kind = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if item_kind not in WORK_ITEM_KINDS:
        raise ValueError("Work item kind must be task or subproject.")
    parent_key = _identifier(parent_id, "Parent id", optional=True)
    parent: Mapping[str, Any] | None = None
    if parent_key is not None:
        parent = _item(result, parent_key)
        if parent.get("kind") != "subproject":
            raise ValueError("A parent work item must be a subproject.")
        if parent.get("status") in {"done", "cancelled"}:
            raise ValueError("Cannot add work beneath a closed subproject.")

    supplied = {
        "client_goal_id": client_goal_id,
        "dossier_id": dossier_id,
        "decision_id": decision_id,
    }
    links: dict[str, str | None] = {}
    explicit_links: list[str] = []
    for field in LINK_FIELDS:
        raw = supplied[field]
        if raw not in (None, ""):
            link = _identifier(raw, field.replace("_", " ").title())
            explicit_links.append(field)
            if parent is not None and parent.get("links", {}).get(field) != link:
                raise ValueError(
                    f"Child {field} must match its parent subproject context."
                )
        elif parent is not None:
            link = parent.get("links", {}).get(field)
        else:
            link = None
        if link is not None and link not in result.get("references", {}).get(field, []):
            raise ValueError(f"Unknown {field}: {link}.")
        links[field] = link
    missing = [field for field in result.get("required_links", []) if not links.get(field)]
    if missing:
        raise ValueError("Work item is missing required links: " + ", ".join(missing) + ".")

    creator = _member(created_by, result, "Created by")
    timestamp = _timestamp(now)
    item = {
        "work_item_id": identifier,
        "title": _text(title, "Title", required=True, limit=500),
        "description": _text(description, "Description", limit=20_000),
        "kind": item_kind,
        "status": "planned",
        "owner": _member(owner, result, "Owner"),
        "due_at": _timestamp(due_at, name="Deadline"),
        "parent_id": parent_key,
        "links": links,
        "explicit_links": explicit_links,
        "created_by": creator,
        "created_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "completion_note": "",
        "history": [
            {
                "from_status": None,
                "to_status": "planned",
                "actor": creator,
                "note": "Work item created.",
                "at": timestamp,
            }
        ],
    }
    result["items"][identifier] = item
    result["item_order"].append(identifier)
    return result


def transition_work_item(
    registry: Mapping[str, Any],
    work_item_id: str,
    new_status: str,
    *,
    actor: str,
    note: str = "",
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(registry, "Work item registry")
    item = _item(result, work_item_id)
    target = str(new_status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if target not in WORK_ITEM_STATUSES:
        raise ValueError("Unknown work item status.")
    prior = item["status"]
    if target not in _TRANSITIONS[prior]:
        raise ValueError(f"Invalid work item transition: {prior} -> {target}.")
    actor_id = _member(actor, result, "Actor")
    clean_note = _text(note, "Transition note", required=target in {"blocked", "done", "cancelled"}, limit=10_000)
    timestamp = _timestamp(now)
    item["status"] = target
    item["updated_at"] = timestamp
    item["history"].append(
        {
            "from_status": prior,
            "to_status": target,
            "actor": actor_id,
            "note": clean_note,
            "at": timestamp,
        }
    )
    if target == "done":
        open_children = [
            child["work_item_id"]
            for child in result["items"].values()
            if child.get("parent_id") == item["work_item_id"]
            and child.get("status") not in {"done", "cancelled"}
        ]
        if open_children:
            raise ValueError(
                "Cannot complete a subproject with open children: " + ", ".join(open_children) + "."
            )
        item["completed_at"] = timestamp
        item["completion_note"] = clean_note
    return result


def work_items_for_context(
    registry: Mapping[str, Any],
    *,
    client_goal_id: str | None = None,
    dossier_id: str | None = None,
    decision_id: str | None = None,
    owner: str | None = None,
    include_closed: bool = False,
) -> list[dict[str, Any]]:
    value = _json_copy(registry, "Work item registry")
    filters = {
        "client_goal_id": _identifier(client_goal_id, "Client goal id", optional=True),
        "dossier_id": _identifier(dossier_id, "Dossier id", optional=True),
        "decision_id": _identifier(decision_id, "Decision id", optional=True),
    }
    owner_value = _member(owner, value, "Owner") if owner is not None else None
    records: list[dict[str, Any]] = []
    for item_id in value.get("item_order", []):
        item = value["items"][item_id]
        if not include_closed and item["status"] in {"done", "cancelled"}:
            continue
        if owner_value is not None and item["owner"] != owner_value:
            continue
        if any(link is not None and item["links"].get(field) != link for field, link in filters.items()):
            continue
        records.append(item)
    return records


def work_item_dashboard(
    registry: Mapping[str, Any],
    *,
    as_of: date | datetime | str | None = None,
) -> dict[str, Any]:
    value = _json_copy(registry, "Work item registry")
    timestamp = _timestamp(as_of)
    active = [
        item for item in value.get("items", {}).values()
        if item.get("status") not in {"done", "cancelled"}
    ]
    overdue = sorted(
        [item for item in active if item["due_at"] < timestamp],
        key=lambda item: (item["due_at"], item["work_item_id"]),
    )
    by_status = {
        status: sum(item.get("status") == status for item in value.get("items", {}).values())
        for status in sorted(WORK_ITEM_STATUSES)
    }
    by_owner = {
        member: sum(
            item.get("owner") == member and item.get("status") not in {"done", "cancelled"}
            for item in value.get("items", {}).values()
        )
        for member in value.get("team_members", [])
    }
    return {
        "as_of": timestamp,
        "item_count": len(value.get("items", {})),
        "active_count": len(active),
        "overdue_count": len(overdue),
        "overdue_items": overdue,
        "by_status": by_status,
        "active_by_owner": by_owner,
    }


def validate_work_item_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(registry, "Work item registry")
    issues: list[dict[str, str]] = []
    order = value.get("item_order", [])
    if len(order) != len(set(order)) or set(order) != set(value.get("items", {})):
        issues.append({"code": "item_order", "work_item_id": ""})
    for item_id, item in value.get("items", {}).items():
        if item.get("owner") not in value.get("team_members", []):
            issues.append({"code": "unknown_owner", "work_item_id": item_id})
        for field in value.get("required_links", []):
            link = item.get("links", {}).get(field)
            if not link:
                issues.append({"code": "missing_link", "work_item_id": item_id})
            elif link not in value.get("references", {}).get(field, []):
                issues.append({"code": "unknown_link", "work_item_id": item_id})
        parent_id = item.get("parent_id")
        if parent_id:
            parent = value.get("items", {}).get(parent_id)
            if parent is None or parent.get("kind") != "subproject":
                issues.append({"code": "invalid_parent", "work_item_id": item_id})
        if not item.get("due_at"):
            issues.append({"code": "missing_deadline", "work_item_id": item_id})
    return {
        "is_valid": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "item_count": len(value.get("items", {})),
    }


__all__ = [
    "LINK_FIELDS",
    "WORK_ITEM_KINDS",
    "WORK_ITEM_STATUSES",
    "add_work_item",
    "create_work_item_registry",
    "register_work_reference",
    "transition_work_item",
    "validate_work_item_registry",
    "work_item_dashboard",
    "work_items_for_context",
]
