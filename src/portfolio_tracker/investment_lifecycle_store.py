"""Canonical Investment Committee lifecycle with blind two-round voting.

The store enforces the operational sequence from proposal to reconciled WInS
position.  Every mutation writes one append-only, hash-chained audit event.
Ballots are immutable and the public read API withholds every ballot until all
eligible, present and non-conflicted members have submitted the round.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping

from ._governance_utils import (
    boolean,
    canonical_hash,
    commit_and_sync,
    decode_array,
    decode_object,
    ensure_schema,
    enum,
    finite_number,
    inserted_id,
    json_array,
    json_object,
    positive_int,
    row_value,
    text,
    ticker,
    utc_timestamp,
)
from .authoritative_universe_store import check_security_eligibility
from .security_dossier_store import get_dossier_version, verify_frozen_dossier


LIFECYCLE_STATES = (
    "proposal",
    "dossier_frozen",
    "pre_vote",
    "discussion",
    "post_vote",
    "rule_check",
    "final_approval",
    "sizing",
    "wins_execution",
    "reconciliation",
    "active",
    "rejected",
    "withdrawn",
    "exited",
)
VOTE_ROUNDS = frozenset({"pre", "post"})
VOTE_DECISIONS = frozenset({"buy", "watch", "reject"})
VOTE_SCOPES = frozenset({"investment", "advisory", "observer"})
COMMITTEE_ROLES = frozenset(
    {"owner", "challenger", "member", "clarity_reviewer", "client_fit_reviewer", "observer"}
)
FINAL_APPROVAL_DECISIONS = frozenset({"approve", "reject"})
PROPOSAL_ACTIONS = frozenset({"buy", "add", "trim", "sell"})

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_lifecycles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        dossier_id INTEGER NOT NULL,
        dossier_version INTEGER NOT NULL,
        locked_dossier_hash TEXT NOT NULL DEFAULT '',
        universe_snapshot_id INTEGER NOT NULL,
        state TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        challenger_id TEXT NOT NULL,
        quorum INTEGER NOT NULL,
        proposal_json TEXT NOT NULL,
        vote_policy_json TEXT NOT NULL,
        current_position_id TEXT,
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (length(trim(ticker)) > 0),
        CHECK (dossier_id > 0),
        CHECK (dossier_version > 0),
        CHECK (universe_snapshot_id > 0),
        CHECK (state IN (
            'proposal', 'dossier_frozen', 'pre_vote', 'discussion',
            'post_vote', 'rule_check', 'final_approval', 'sizing',
            'wins_execution', 'reconciliation', 'active', 'rejected',
            'withdrawn', 'exited'
        )),
        CHECK (owner_id <> challenger_id),
        CHECK (quorum > 0),
        CHECK (length(trim(created_by)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_lifecycle_ticker_state
    ON canonical_investment_lifecycles(ticker, state, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_committee_members (
        lifecycle_id INTEGER NOT NULL,
        member_id TEXT NOT NULL,
        display_name TEXT NOT NULL,
        committee_role TEXT NOT NULL,
        vote_scope TEXT NOT NULL,
        present INTEGER NOT NULL,
        conflicted INTEGER NOT NULL,
        conflict_reason TEXT NOT NULL,
        required_approver INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (lifecycle_id, member_id),
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (committee_role IN (
            'owner', 'challenger', 'member', 'clarity_reviewer',
            'client_fit_reviewer', 'observer'
        )),
        CHECK (vote_scope IN ('investment', 'advisory', 'observer')),
        CHECK (present IN (0, 1)),
        CHECK (conflicted IN (0, 1)),
        CHECK (required_approver IN (0, 1)),
        CHECK (conflicted = 0 OR length(trim(conflict_reason)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_votes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        vote_round TEXT NOT NULL,
        member_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        proposed_weight_pct REAL,
        confidence INTEGER NOT NULL,
        rationale TEXT NOT NULL,
        strongest_objection TEXT NOT NULL,
        dimensions_json TEXT NOT NULL,
        ballot_hash TEXT NOT NULL,
        submitted_at TEXT NOT NULL,
        UNIQUE (lifecycle_id, vote_round, member_id),
        FOREIGN KEY (lifecycle_id, member_id)
            REFERENCES canonical_investment_committee_members(lifecycle_id, member_id),
        CHECK (vote_round IN ('pre', 'post')),
        CHECK (decision IN ('buy', 'watch', 'reject')),
        CHECK (proposed_weight_pct IS NULL OR proposed_weight_pct BETWEEN 0 AND 100),
        CHECK (confidence BETWEEN 1 AND 5),
        CHECK (length(trim(rationale)) > 0),
        CHECK (length(trim(strongest_objection)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_investment_votes_round
    ON canonical_investment_votes(lifecycle_id, vote_round, submitted_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_discussions (
        lifecycle_id INTEGER PRIMARY KEY,
        bull_case TEXT NOT NULL,
        bear_case TEXT NOT NULL,
        q_and_a_json TEXT NOT NULL,
        notes TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (length(trim(bull_case)) > 0),
        CHECK (length(trim(bear_case)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_rule_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        rulebook_version INTEGER NOT NULL,
        mandate_version INTEGER NOT NULL,
        checks_json TEXT NOT NULL,
        failed_rules_json TEXT NOT NULL,
        override_json TEXT NOT NULL,
        passed INTEGER NOT NULL,
        effective_pass INTEGER NOT NULL,
        evaluated_by TEXT NOT NULL,
        evaluated_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (rulebook_version > 0),
        CHECK (mandate_version > 0),
        CHECK (passed IN (0, 1)),
        CHECK (effective_pass IN (0, 1)),
        CHECK (length(trim(evaluated_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_final_approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        member_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        comment TEXT NOT NULL,
        submitted_at TEXT NOT NULL,
        UNIQUE (lifecycle_id, member_id),
        FOREIGN KEY (lifecycle_id, member_id)
            REFERENCES canonical_investment_committee_members(lifecycle_id, member_id),
        CHECK (decision IN ('approve', 'reject'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_sizing (
        lifecycle_id INTEGER PRIMARY KEY,
        sizing_json TEXT NOT NULL,
        sized_by TEXT NOT NULL,
        sized_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (length(trim(sized_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_wins_executions (
        lifecycle_id INTEGER PRIMARY KEY,
        wins_transaction_id TEXT NOT NULL UNIQUE,
        execution_json TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (length(trim(wins_transaction_id)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_reconciliations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        wins_snapshot_id TEXT NOT NULL,
        position_id TEXT,
        reconciliation_json TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (status IN ('clean', 'open_exceptions')),
        CHECK (length(trim(wins_snapshot_id)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_investment_reconciliations
    ON canonical_investment_reconciliations(lifecycle_id, id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_position_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        reviewed_by TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (outcome IN ('confirmed', 'watch', 'invalidated')),
        CHECK (length(trim(reviewed_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_exits (
        lifecycle_id INTEGER PRIMARY KEY,
        wins_transaction_id TEXT NOT NULL UNIQUE,
        exit_json TEXT NOT NULL,
        recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (length(trim(wins_transaction_id)) > 0),
        CHECK (length(trim(recorded_by)) > 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS canonical_investment_audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lifecycle_id INTEGER NOT NULL,
        sequence INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        actor TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        event_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (lifecycle_id, sequence),
        FOREIGN KEY (lifecycle_id) REFERENCES canonical_investment_lifecycles(id),
        CHECK (sequence > 0),
        CHECK (length(trim(event_type)) > 0),
        CHECK (length(trim(actor)) > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_canonical_investment_audit
    ON canonical_investment_audit_events(lifecycle_id, sequence)
    """,
)


def _ensure(connection: Any) -> None:
    ensure_schema(connection, _SCHEMA)


def init_investment_lifecycle_tables(connection: Any) -> None:
    _ensure(connection)
    commit_and_sync(connection)


def _member_id(value: Any, name: str = "Member id") -> str:
    result = text(value, name, required=True, limit=100)
    if any(character.isspace() for character in result):
        raise ValueError(f"{name} must not contain whitespace.")
    return result


def _normalise_member(
    raw: Mapping[str, Any],
    *,
    owner_id: str,
    challenger_id: str,
    required_approvers: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Each committee member must be a JSON object.")
    identifier = _member_id(raw.get("member_id", raw.get("id")))
    role = enum(raw.get("committee_role", raw.get("role", "member")), "Committee role", COMMITTEE_ROLES)
    if identifier == owner_id:
        role = "owner"
    elif identifier == challenger_id:
        role = "challenger"
    scope_default = "observer" if role == "observer" else (
        "advisory" if role in {"clarity_reviewer", "client_fit_reviewer"} else "investment"
    )
    scope = enum(raw.get("vote_scope", scope_default), "Vote scope", VOTE_SCOPES)
    present = boolean(raw.get("present", True), "Present")
    conflicted = boolean(raw.get("conflicted", False), "Conflicted")
    conflict_reason = text(raw.get("conflict_reason"), "Conflict reason", limit=2_000)
    if conflicted and not conflict_reason:
        raise ValueError(f"Conflict reason is required for {identifier}.")
    extra, _ = json_object(raw.get("payload", {}), "Committee member payload")
    return {
        "member_id": identifier,
        "display_name": text(
            raw.get("display_name", raw.get("name", identifier)),
            "Display name",
            required=True,
            limit=200,
        ),
        "committee_role": role,
        "vote_scope": scope,
        "present": present,
        "conflicted": conflicted,
        "conflict_reason": conflict_reason,
        "required_approver": identifier in required_approvers,
        "payload": extra,
    }


def _normalise_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw, _ = json_object(value or {}, "Vote policy")
    fraction = finite_number(
        raw.get("minimum_buy_fraction", 0.5),
        "Minimum buy fraction",
        minimum=0,
        maximum=0.99,
    )
    reject_veto = boolean(raw.get("reject_veto", False), "Reject veto")
    return {
        **raw,
        "minimum_buy_fraction": fraction,
        "reject_veto": reject_veto,
    }


def _append_audit_event(
    connection: Any,
    lifecycle_id: int,
    *,
    event_type: str,
    from_state: str | None,
    to_state: str,
    actor: str,
    payload: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    event_name = text(event_type, "Event type", required=True, limit=200)
    event_actor = text(actor, "Actor", required=True, limit=200)
    payload_copy, payload_json = json_object(payload, "Audit payload")
    row = connection.execute(
        """
        SELECT sequence, event_hash FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? ORDER BY sequence DESC LIMIT 1
        """,
        (lifecycle_id,),
    ).fetchone()
    sequence = 1 if row is None else int(row_value(row, "sequence", 0)) + 1
    previous_hash = "" if row is None else str(row_value(row, "event_hash", 1))
    hash_input = {
        "lifecycle_id": lifecycle_id,
        "sequence": sequence,
        "event_type": event_name,
        "from_state": from_state,
        "to_state": to_state,
        "actor": event_actor,
        "payload": payload_copy,
        "previous_hash": previous_hash,
        "created_at": timestamp,
    }
    event_hash = canonical_hash(hash_input)
    cursor = connection.execute(
        """
        INSERT INTO canonical_investment_audit_events (
            lifecycle_id, sequence, event_type, from_state, to_state, actor,
            payload_json, previous_hash, event_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lifecycle_id,
            sequence,
            event_name,
            from_state,
            to_state,
            event_actor,
            payload_json,
            previous_hash,
            event_hash,
            timestamp,
        ),
    )
    event_id = inserted_id(connection, cursor, "canonical_investment_audit_events")
    return {"id": event_id, **hash_input, "event_hash": event_hash}


def _lifecycle_row(connection: Any, lifecycle_id: int) -> Any:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    row = connection.execute(
        """
        SELECT id, ticker, dossier_id, dossier_version, locked_dossier_hash,
               universe_snapshot_id, state, owner_id, challenger_id, quorum,
               proposal_json, vote_policy_json, current_position_id,
               created_by, created_at, updated_at
        FROM canonical_investment_lifecycles WHERE id = ?
        """,
        (identifier,),
    ).fetchone()
    if row is None:
        raise ValueError("Investment lifecycle does not exist.")
    return row


def _base_record(row: Any) -> dict[str, Any] | None:
    proposal = decode_object(row_value(row, "proposal_json", 10))
    policy = decode_object(row_value(row, "vote_policy_json", 11))
    if proposal is None or policy is None:
        return None
    return {
        "id": int(row_value(row, "id", 0)),
        "ticker": str(row_value(row, "ticker", 1)),
        "dossier_id": int(row_value(row, "dossier_id", 2)),
        "dossier_version": int(row_value(row, "dossier_version", 3)),
        "locked_dossier_hash": str(row_value(row, "locked_dossier_hash", 4) or ""),
        "universe_snapshot_id": int(row_value(row, "universe_snapshot_id", 5)),
        "state": str(row_value(row, "state", 6)),
        "owner_id": str(row_value(row, "owner_id", 7)),
        "challenger_id": str(row_value(row, "challenger_id", 8)),
        "quorum": int(row_value(row, "quorum", 9)),
        "proposal": proposal,
        "vote_policy": policy,
        "current_position_id": (
            None
            if row_value(row, "current_position_id", 12) is None
            else str(row_value(row, "current_position_id", 12))
        ),
        "created_by": str(row_value(row, "created_by", 13)),
        "created_at": str(row_value(row, "created_at", 14)),
        "updated_at": str(row_value(row, "updated_at", 15)),
    }


def _state(connection: Any, lifecycle_id: int) -> str:
    return str(row_value(_lifecycle_row(connection, lifecycle_id), "state", 6))


def _require_state(connection: Any, lifecycle_id: int, expected: str | Iterable[str]) -> str:
    current = _state(connection, lifecycle_id)
    allowed = {expected} if isinstance(expected, str) else set(expected)
    if current not in allowed:
        raise ValueError(
            f"Lifecycle is in state {current!r}; expected {', '.join(sorted(allowed))}."
        )
    return current


def _transition(
    connection: Any,
    lifecycle_id: int,
    *,
    expected: str | Iterable[str],
    target: str,
    event_type: str,
    actor: str,
    payload: Mapping[str, Any],
    timestamp: str,
) -> None:
    current = _require_state(connection, lifecycle_id, expected)
    enum(target, "Lifecycle target state", LIFECYCLE_STATES)
    connection.execute(
        "UPDATE canonical_investment_lifecycles SET state = ?, updated_at = ? WHERE id = ?",
        (target, timestamp, lifecycle_id),
    )
    _append_audit_event(
        connection,
        lifecycle_id,
        event_type=event_type,
        from_state=current,
        to_state=target,
        actor=actor,
        payload=payload,
        timestamp=timestamp,
    )


def _committee_members(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT member_id, display_name, committee_role, vote_scope, present,
               conflicted, conflict_reason, required_approver, payload_json
        FROM canonical_investment_committee_members
        WHERE lifecycle_id = ? ORDER BY member_id
        """,
        (lifecycle_id,),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        payload = decode_object(row_value(row, "payload_json", 8))
        if payload is None:
            continue
        result.append(
            {
                "member_id": str(row_value(row, "member_id", 0)),
                "display_name": str(row_value(row, "display_name", 1)),
                "committee_role": str(row_value(row, "committee_role", 2)),
                "vote_scope": str(row_value(row, "vote_scope", 3)),
                "present": bool(row_value(row, "present", 4)),
                "conflicted": bool(row_value(row, "conflicted", 5)),
                "conflict_reason": str(row_value(row, "conflict_reason", 6) or ""),
                "required_approver": bool(row_value(row, "required_approver", 7)),
                "payload": payload,
            }
        )
    return result


def _eligible_members(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    return [
        item
        for item in _committee_members(connection, lifecycle_id)
        if item["present"] and not item["conflicted"] and item["vote_scope"] != "observer"
    ]


def _validate_committee_ready(connection: Any, lifecycle_id: int) -> None:
    row = _lifecycle_row(connection, lifecycle_id)
    lifecycle = _base_record(row)
    if lifecycle is None:
        raise ValueError("Lifecycle data is corrupt.")
    members = _committee_members(connection, lifecycle_id)
    by_id = {item["member_id"]: item for item in members}
    protected_ids = {
        lifecycle["owner_id"],
        lifecycle["challenger_id"],
        *(item["member_id"] for item in members if item["required_approver"]),
    }
    for identifier in protected_ids:
        member = by_id.get(identifier)
        if member is None or not member["present"] or member["conflicted"]:
            raise ValueError(f"Required committee member {identifier} must be present and non-conflicted.")
        if member["vote_scope"] != "investment":
            raise ValueError(f"Required committee member {identifier} must have an investment vote.")
    investment_voters = [
        item for item in _eligible_members(connection, lifecycle_id) if item["vote_scope"] == "investment"
    ]
    if len(investment_voters) < lifecycle["quorum"]:
        raise ValueError("Present, non-conflicted investment voters do not satisfy quorum.")


def create_investment_proposal(
    connection: Any,
    *,
    security_ticker: str,
    dossier_id: int,
    dossier_version: int,
    universe_snapshot_id: int,
    proposal: Mapping[str, Any],
    committee_members: Iterable[Mapping[str, Any]],
    owner_id: str,
    challenger_id: str,
    required_approvers: Iterable[str],
    quorum: int,
    created_by: str,
    vote_policy: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create a proposal linked to one dossier version and universe snapshot."""

    code = ticker(security_ticker)
    linked_dossier = positive_int(dossier_id, "Dossier id")
    linked_version = positive_int(dossier_version, "Dossier version")
    linked_universe = positive_int(universe_snapshot_id, "Universe snapshot id")
    owner = _member_id(owner_id, "Owner id")
    challenger = _member_id(challenger_id, "Challenger id")
    if owner == challenger:
        raise ValueError("Owner and challenger must be different committee members.")
    approvers = {_member_id(item, "Required approver id") for item in required_approvers}
    if len(approvers) < 2:
        raise ValueError("At least two independent final approvers are required.")
    required_quorum = positive_int(quorum, "Quorum")
    if required_quorum < 2:
        raise ValueError("Investment Committee quorum must be at least two.")
    author = text(created_by, "Created by", required=True, limit=200)
    proposal_copy, _ = json_object(proposal, "Proposal")
    if not proposal_copy:
        raise ValueError("Proposal must contain a rationale and proposed action.")
    proposal_copy["action"] = enum(
        proposal_copy.get("action"),
        "Proposal action",
        PROPOSAL_ACTIONS,
    )
    proposal_copy["rationale"] = text(
        proposal_copy.get("rationale"),
        "Proposal rationale",
        required=True,
    )
    _, proposal_json = json_object(proposal_copy, "Proposal")
    policy = _normalise_policy(vote_policy)
    _, policy_json = json_object(policy, "Vote policy")
    timestamp = utc_timestamp(now)

    _ensure(connection)
    dossier = get_dossier_version(connection, linked_dossier, linked_version)
    if dossier is None:
        raise ValueError("Linked dossier version does not exist.")
    if dossier["ticker"] != code:
        raise ValueError("Proposal ticker does not match the linked dossier.")
    require_universe = check_security_eligibility(
        connection,
        code,
        snapshot_id=linked_universe,
    )
    if not require_universe["can_trade"]:
        raise ValueError(
            f"{code} is not eligible in the active official universe: "
            + " ".join(require_universe["reasons"])
        )
    try:
        raw_members = list(committee_members)
    except TypeError as exc:
        raise ValueError("Committee members must be an iterable of JSON objects.") from exc
    members = [
        _normalise_member(
            item,
            owner_id=owner,
            challenger_id=challenger,
            required_approvers=approvers,
        )
        for item in raw_members
    ]
    member_ids = [item["member_id"] for item in members]
    if len(member_ids) != len(set(member_ids)):
        duplicate = next(item for item in member_ids if member_ids.count(item) > 1)
        raise ValueError(f"Duplicate committee member: {duplicate}.")
    missing = ({owner, challenger} | approvers) - set(member_ids)
    if missing:
        raise ValueError("Committee roster is missing: " + ", ".join(sorted(missing)) + ".")
    if required_quorum > sum(item["vote_scope"] == "investment" for item in members):
        raise ValueError("Quorum exceeds the number of investment-voting committee members.")

    cursor = connection.execute(
        """
        INSERT INTO canonical_investment_lifecycles (
            ticker, dossier_id, dossier_version, locked_dossier_hash,
            universe_snapshot_id, state, owner_id, challenger_id, quorum,
            proposal_json, vote_policy_json, current_position_id,
            created_by, created_at, updated_at
        ) VALUES (?, ?, ?, '', ?, 'proposal', ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            code,
            linked_dossier,
            linked_version,
            linked_universe,
            owner,
            challenger,
            required_quorum,
            proposal_json,
            policy_json,
            author,
            timestamp,
            timestamp,
        ),
    )
    lifecycle_id = inserted_id(connection, cursor, "canonical_investment_lifecycles")
    for member in members:
        _, member_payload_json = json_object(member["payload"], "Committee member payload")
        connection.execute(
            """
            INSERT INTO canonical_investment_committee_members (
                lifecycle_id, member_id, display_name, committee_role, vote_scope,
                present, conflicted, conflict_reason, required_approver, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lifecycle_id,
                member["member_id"],
                member["display_name"],
                member["committee_role"],
                member["vote_scope"],
                int(member["present"]),
                int(member["conflicted"]),
                member["conflict_reason"],
                int(member["required_approver"]),
                member_payload_json,
            ),
        )
    _append_audit_event(
        connection,
        lifecycle_id,
        event_type="proposal_created",
        from_state=None,
        to_state="proposal",
        actor=author,
        payload={
            "ticker": code,
            "dossier_id": linked_dossier,
            "dossier_version": linked_version,
            "universe_snapshot_id": linked_universe,
            "proposal_hash": canonical_hash(proposal_copy),
            "committee_members": [
                {
                    "member_id": item["member_id"],
                    "role": item["committee_role"],
                    "vote_scope": item["vote_scope"],
                    "present": item["present"],
                    "conflicted": item["conflicted"],
                    "required_approver": item["required_approver"],
                }
                for item in members
            ],
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, lifecycle_id)
    if record is None:
        raise RuntimeError("The investment proposal could not be read after creation.")
    return record


def update_committee_member_status(
    connection: Any,
    lifecycle_id: int,
    member_id: str,
    *,
    present: bool | None = None,
    conflicted: bool | None = None,
    conflict_reason: str | None = None,
    updated_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record attendance/conflict handling before the blind vote opens."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    member = _member_id(member_id)
    actor = text(updated_by, "Updated by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    current = _require_state(connection, identifier, {"proposal", "dossier_frozen"})
    row = connection.execute(
        """
        SELECT present, conflicted, conflict_reason
        FROM canonical_investment_committee_members
        WHERE lifecycle_id = ? AND member_id = ?
        """,
        (identifier, member),
    ).fetchone()
    if row is None:
        raise ValueError("Committee member does not exist.")
    before = {
        "present": bool(row_value(row, "present", 0)),
        "conflicted": bool(row_value(row, "conflicted", 1)),
        "conflict_reason": str(row_value(row, "conflict_reason", 2) or ""),
    }
    after_present = before["present"] if present is None else boolean(present, "Present")
    after_conflicted = before["conflicted"] if conflicted is None else boolean(conflicted, "Conflicted")
    reason = before["conflict_reason"] if conflict_reason is None else text(
        conflict_reason,
        "Conflict reason",
        limit=2_000,
    )
    if after_conflicted and not reason:
        raise ValueError("Conflict reason is required when a member is conflicted.")
    if not after_conflicted:
        reason = ""
    after = {
        "present": after_present,
        "conflicted": after_conflicted,
        "conflict_reason": reason,
    }
    connection.execute(
        """
        UPDATE canonical_investment_committee_members
        SET present = ?, conflicted = ?, conflict_reason = ?
        WHERE lifecycle_id = ? AND member_id = ?
        """,
        (int(after_present), int(after_conflicted), reason, identifier, member),
    )
    connection.execute(
        "UPDATE canonical_investment_lifecycles SET updated_at = ? WHERE id = ?",
        (timestamp, identifier),
    )
    _append_audit_event(
        connection,
        identifier,
        event_type="committee_member_status_updated",
        from_state=current,
        to_state=current,
        actor=actor,
        payload={"member_id": member, "before": before, "after": after},
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after updating attendance.")
    return record


def lock_proposal_dossier(
    connection: Any,
    lifecycle_id: int,
    *,
    locked_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bind the proposal to an already-frozen, hash-verified dossier version."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    actor = text(locked_by, "Locked by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "proposal")
    base = _base_record(_lifecycle_row(connection, identifier))
    if base is None:
        raise ValueError("Lifecycle data is corrupt.")
    dossier = get_dossier_version(
        connection,
        base["dossier_id"],
        base["dossier_version"],
    )
    if dossier is None or dossier["status"] != "frozen":
        raise ValueError("The linked dossier version must be frozen before committee voting.")
    if not verify_frozen_dossier(dossier):
        raise ValueError("The frozen dossier content hash is invalid.")
    eligibility = check_security_eligibility(
        connection,
        base["ticker"],
        snapshot_id=base["universe_snapshot_id"],
    )
    if not eligibility["can_trade"]:
        raise ValueError(
            "The linked universe snapshot is no longer authoritative: "
            + " ".join(eligibility["reasons"])
        )
    _validate_committee_ready(connection, identifier)
    connection.execute(
        """
        UPDATE canonical_investment_lifecycles
        SET locked_dossier_hash = ? WHERE id = ?
        """,
        (dossier["content_hash"], identifier),
    )
    _transition(
        connection,
        identifier,
        expected="proposal",
        target="dossier_frozen",
        event_type="dossier_locked",
        actor=actor,
        payload={
            "dossier_id": dossier["dossier_id"],
            "dossier_version": dossier["version"],
            "content_hash": dossier["content_hash"],
            "kpi_definition_count": len(dossier["kpi_snapshot"]),
            "universe_snapshot_id": base["universe_snapshot_id"],
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after locking the dossier.")
    return record


def open_pre_vote(
    connection: Any,
    lifecycle_id: int,
    *,
    opened_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    actor = text(opened_by, "Opened by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "dossier_frozen")
    _validate_committee_ready(connection, identifier)
    eligible = _eligible_members(connection, identifier)
    _transition(
        connection,
        identifier,
        expected="dossier_frozen",
        target="pre_vote",
        event_type="pre_vote_opened",
        actor=actor,
        payload={
            "eligible_member_ids": [item["member_id"] for item in eligible],
            "investment_voter_count": sum(item["vote_scope"] == "investment" for item in eligible),
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    return get_vote_round(connection, identifier, "pre")


def _round_was_opened(connection: Any, lifecycle_id: int, vote_round: str) -> bool:
    event_type = "pre_vote_opened" if vote_round == "pre" else "discussion_recorded"
    row = connection.execute(
        """
        SELECT 1 FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? AND event_type = ? LIMIT 1
        """,
        (lifecycle_id, event_type),
    ).fetchone()
    return row is not None


def _vote_record(row: Any) -> dict[str, Any] | None:
    dimensions = decode_object(row_value(row, "dimensions_json", 10))
    if dimensions is None:
        return None
    proposed = row_value(row, "proposed_weight_pct", 6)
    return {
        "id": int(row_value(row, "id", 0)),
        "member_id": str(row_value(row, "member_id", 1)),
        "display_name": str(row_value(row, "display_name", 2)),
        "vote_scope": str(row_value(row, "vote_scope", 3)),
        "decision": str(row_value(row, "decision", 4)),
        "committee_role": str(row_value(row, "committee_role", 5)),
        "proposed_weight_pct": None if proposed is None else float(proposed),
        "confidence": int(row_value(row, "confidence", 7)),
        "rationale": str(row_value(row, "rationale", 8)),
        "strongest_objection": str(row_value(row, "strongest_objection", 9)),
        "dimensions": dimensions,
        "ballot_hash": str(row_value(row, "ballot_hash", 11)),
        "submitted_at": str(row_value(row, "submitted_at", 12)),
    }


def _vote_outcome(ballots: list[dict[str, Any]], policy: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
    investment = [item for item in ballots if item["vote_scope"] == "investment"]
    tally = {choice: sum(item["decision"] == choice for item in investment) for choice in sorted(VOTE_DECISIONS)}
    if not investment:
        return "reject", tally
    if bool(policy.get("reject_veto")) and tally["reject"]:
        return "reject", tally
    buy_ratio = tally["buy"] / len(investment)
    if buy_ratio > float(policy.get("minimum_buy_fraction", 0.5)):
        return "buy", tally
    if tally["reject"] > tally["buy"]:
        return "reject", tally
    return "watch", tally


def get_vote_round(
    connection: Any,
    lifecycle_id: int,
    vote_round: str,
) -> dict[str, Any]:
    """Read a vote round without leaking partial ballots."""

    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    round_name = enum(vote_round, "Vote round", VOTE_ROUNDS)
    base = _base_record(_lifecycle_row(connection, identifier))
    if base is None:
        raise ValueError("Lifecycle data is corrupt.")
    eligible = _eligible_members(connection, identifier)
    eligible_ids = [item["member_id"] for item in eligible]
    rows = connection.execute(
        """
        SELECT v.id, v.member_id, m.display_name, m.vote_scope, v.decision,
               m.committee_role, v.proposed_weight_pct, v.confidence,
               v.rationale, v.strongest_objection, v.dimensions_json,
               v.ballot_hash, v.submitted_at
        FROM canonical_investment_votes v
        JOIN canonical_investment_committee_members m
          ON m.lifecycle_id = v.lifecycle_id AND m.member_id = v.member_id
        WHERE v.lifecycle_id = ? AND v.vote_round = ?
        ORDER BY v.member_id
        """,
        (identifier, round_name),
    ).fetchall()
    stored_ballots = [ballot for row in rows if (ballot := _vote_record(row)) is not None]
    submitted_ids = [item["member_id"] for item in stored_ballots]
    opened = _round_was_opened(connection, identifier, round_name)
    revealed = opened and bool(eligible_ids) and set(submitted_ids) == set(eligible_ids)
    outcome: str | None = None
    tally: dict[str, int] | None = None
    dissent: list[dict[str, Any]] = []
    if revealed:
        outcome, tally = _vote_outcome(stored_ballots, base["vote_policy"])
        dissent = [
            {
                "member_id": item["member_id"],
                "decision": item["decision"],
                "strongest_objection": item["strongest_objection"],
            }
            for item in stored_ballots
            if item["vote_scope"] == "investment" and item["decision"] != outcome
        ]
    closed_event = "pre_vote_closed" if round_name == "pre" else "post_vote_closed"
    is_closed = connection.execute(
        """
        SELECT 1 FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? AND event_type = ? LIMIT 1
        """,
        (identifier, closed_event),
    ).fetchone() is not None
    status = "not_open"
    if opened:
        status = "closed" if is_closed else ("ready_to_close" if revealed else "collecting")
    return {
        "lifecycle_id": identifier,
        "round": round_name,
        "status": status,
        "opened": opened,
        "revealed": revealed,
        "eligible_count": len(eligible_ids),
        "submitted_count": len(submitted_ids),
        "remaining_count": max(0, len(eligible_ids) - len(submitted_ids)),
        "ballots": stored_ballots if revealed else [],
        "tally": tally,
        "outcome": outcome,
        "dissent": dissent if revealed else [],
    }


def has_member_submitted_vote(
    connection: Any,
    lifecycle_id: int,
    vote_round: str,
    member_id: str,
) -> bool:
    """Return an own-vote receipt without exposing any ballot content."""

    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    round_name = enum(vote_round, "Vote round", VOTE_ROUNDS)
    member = _member_id(member_id)
    _lifecycle_row(connection, identifier)
    row = connection.execute(
        """
        SELECT 1 FROM canonical_investment_votes
        WHERE lifecycle_id = ? AND vote_round = ? AND member_id = ?
        LIMIT 1
        """,
        (identifier, round_name, member),
    ).fetchone()
    return row is not None


def submit_committee_vote(
    connection: Any,
    lifecycle_id: int,
    vote_round: str,
    member_id: str,
    *,
    decision: str,
    proposed_weight_pct: float | None,
    confidence: int,
    rationale: str,
    strongest_objection: str,
    dimensions: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Submit one immutable blind ballot and return only the safe round view."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    round_name = enum(vote_round, "Vote round", VOTE_ROUNDS)
    expected_state = "pre_vote" if round_name == "pre" else "post_vote"
    member = _member_id(member_id)
    choice = enum(decision, "Vote decision", VOTE_DECISIONS)
    weight = finite_number(
        proposed_weight_pct,
        "Proposed weight",
        optional=True,
        minimum=0,
        maximum=100,
    )
    if choice == "buy" and weight is None:
        raise ValueError("A BUY vote must include a proposed position weight.")
    if isinstance(confidence, bool):
        raise ValueError("Confidence must be an integer from 1 through 5.")
    try:
        confidence_number = int(confidence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Confidence must be an integer from 1 through 5.") from exc
    if confidence_number not in range(1, 6) or float(confidence) != confidence_number:
        raise ValueError("Confidence must be an integer from 1 through 5.")
    reasoning = text(rationale, "Vote rationale", required=True, limit=10_000)
    objection = text(
        strongest_objection,
        "Strongest unresolved objection",
        required=True,
        limit=10_000,
    )
    dimensions_copy, dimensions_json = json_object(dimensions or {}, "Vote dimensions")
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, expected_state)
    eligible = {item["member_id"]: item for item in _eligible_members(connection, identifier)}
    if member not in eligible:
        raise ValueError("Member is absent, conflicted, an observer, or not on this committee.")
    required_dimension = {
        "clarity_reviewer": "clarity",
        "client_fit_reviewer": "client_fit",
    }.get(eligible[member]["committee_role"])
    if required_dimension is not None and dimensions_copy.get(required_dimension) is None:
        raise ValueError(
            f"{eligible[member]['committee_role']} must submit a {required_dimension} score."
        )
    for dimension_name in ("clarity", "client_fit"):
        if dimensions_copy.get(dimension_name) is None:
            continue
        dimension_value = finite_number(
            dimensions_copy[dimension_name],
            f"{dimension_name} score",
            minimum=1,
            maximum=5,
        )
        if not float(dimension_value).is_integer():
            raise ValueError(f"{dimension_name} score must be an integer from 1 through 5.")
        dimensions_copy[dimension_name] = int(dimension_value)
    _, dimensions_json = json_object(dimensions_copy, "Vote dimensions")
    existing = connection.execute(
        """
        SELECT 1 FROM canonical_investment_votes
        WHERE lifecycle_id = ? AND vote_round = ? AND member_id = ?
        """,
        (identifier, round_name, member),
    ).fetchone()
    if existing is not None:
        raise ValueError("This member has already submitted an immutable ballot for the round.")
    ballot_content = {
        "lifecycle_id": identifier,
        "round": round_name,
        "member_id": member,
        "decision": choice,
        "proposed_weight_pct": weight,
        "confidence": confidence_number,
        "rationale": reasoning,
        "strongest_objection": objection,
        "dimensions": dimensions_copy,
        "submitted_at": timestamp,
    }
    ballot_hash = canonical_hash(ballot_content)
    connection.execute(
        """
        INSERT INTO canonical_investment_votes (
            lifecycle_id, vote_round, member_id, decision, proposed_weight_pct,
            confidence, rationale, strongest_objection, dimensions_json,
            ballot_hash, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identifier,
            round_name,
            member,
            choice,
            weight,
            confidence_number,
            reasoning,
            objection,
            dimensions_json,
            ballot_hash,
            timestamp,
        ),
    )
    _append_audit_event(
        connection,
        identifier,
        event_type=f"{round_name}_vote_submitted",
        from_state=expected_state,
        to_state=expected_state,
        actor=member,
        payload={"member_id": member, "round": round_name, "ballot_hash": ballot_hash},
        timestamp=timestamp,
    )
    connection.execute(
        "UPDATE canonical_investment_lifecycles SET updated_at = ? WHERE id = ?",
        (timestamp, identifier),
    )
    commit_and_sync(connection)
    return get_vote_round(connection, identifier, round_name)


def close_vote_round(
    connection: Any,
    lifecycle_id: int,
    vote_round: str,
    *,
    closed_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    round_name = enum(vote_round, "Vote round", VOTE_ROUNDS)
    expected_state = "pre_vote" if round_name == "pre" else "post_vote"
    actor = text(closed_by, "Closed by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, expected_state)
    summary = get_vote_round(connection, identifier, round_name)
    if not summary["revealed"]:
        raise ValueError("Vote round cannot close until every eligible member has submitted.")
    if round_name == "pre":
        target = "discussion"
    else:
        target = "rule_check" if summary["outcome"] == "buy" else "rejected"
    _transition(
        connection,
        identifier,
        expected=expected_state,
        target=target,
        event_type=f"{round_name}_vote_closed",
        actor=actor,
        payload={
            "outcome": summary["outcome"],
            "tally": summary["tally"],
            "dissent": summary["dissent"],
            "eligible_count": summary["eligible_count"],
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after closing the vote.")
    return record


def _normalise_q_and_a(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        try:
            value = list(value)
        except TypeError as exc:
            raise ValueError("Q&A must be an iterable of JSON objects.") from exc
    rows, _ = json_array(value, "Q&A")
    if not rows:
        raise ValueError("Committee discussion must record at least one question and answer.")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Q&A item {index} must be a JSON object.")
        copy, _ = json_object(item, f"Q&A item {index}")
        question = text(copy.get("question"), f"Q&A item {index} question", required=True)
        answer = text(
            copy.get("answer", copy.get("response")),
            f"Q&A item {index} answer",
            required=True,
        )
        result.append({**copy, "question": question, "answer": answer})
    return result


def record_committee_discussion(
    connection: Any,
    lifecycle_id: int,
    *,
    bull_case: str,
    bear_case: str,
    q_and_a: Iterable[Mapping[str, Any]],
    notes: str = "",
    recorded_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record bull case, challenger-led bear case and committee Q&A."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    bull = text(bull_case, "Bull case", required=True)
    bear = text(bear_case, "Bear case", required=True)
    questions = _normalise_q_and_a(q_and_a)
    _, q_and_a_json = json_array(questions, "Q&A")
    discussion_notes = text(notes, "Discussion notes")
    actor = text(recorded_by, "Recorded by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "discussion")
    pre_round = get_vote_round(connection, identifier, "pre")
    if pre_round["status"] != "closed":
        raise ValueError("Pre-vote round must be closed before discussion is recorded.")
    connection.execute(
        """
        INSERT INTO canonical_investment_discussions (
            lifecycle_id, bull_case, bear_case, q_and_a_json, notes,
            recorded_by, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (identifier, bull, bear, q_and_a_json, discussion_notes, actor, timestamp),
    )
    _transition(
        connection,
        identifier,
        expected="discussion",
        target="post_vote",
        event_type="discussion_recorded",
        actor=actor,
        payload={
            "bull_case_hash": canonical_hash({"text": bull}),
            "bear_case_hash": canonical_hash({"text": bear}),
            "q_and_a_count": len(questions),
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after discussion.")
    return record


def _normalise_rule_checks(checks: Any) -> list[dict[str, Any]]:
    if not isinstance(checks, (list, tuple)):
        try:
            checks = list(checks)
        except TypeError as exc:
            raise ValueError("Rule checks must be an iterable of JSON objects.") from exc
    rows, _ = json_array(checks, "Rule checks")
    if not rows:
        raise ValueError("At least one rule check is required.")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Rule check {index} must be a JSON object.")
        copy, _ = json_object(item, f"Rule check {index}")
        rule_id = text(copy.get("rule_id"), f"Rule check {index} id", required=True, limit=200)
        if rule_id in seen:
            raise ValueError(f"Duplicate rule check: {rule_id}.")
        seen.add(rule_id)
        passed = boolean(copy.get("passed"), f"Rule check {rule_id} passed")
        result.append({**copy, "rule_id": rule_id, "passed": passed})
    return result


def _normalise_override(
    override: Mapping[str, Any] | None,
    *,
    required_approvers: set[str],
) -> dict[str, Any]:
    if override is None:
        return {}
    copy, _ = json_object(override, "Rule override")
    reason = text(copy.get("reason"), "Override reason", required=True)
    authorized_by = _member_id(copy.get("authorized_by"), "Override authorizer")
    if authorized_by not in required_approvers:
        raise ValueError("A rule override must be authorized by a required final approver.")
    return {**copy, "reason": reason, "authorized_by": authorized_by}


def record_rule_check(
    connection: Any,
    lifecycle_id: int,
    *,
    rulebook_version: int,
    mandate_version: int,
    checks: Iterable[Mapping[str, Any]],
    evaluated_by: str,
    override: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run mandatory strategy/mandate checks, with only explicit overrides."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rulebook = positive_int(rulebook_version, "Rulebook version")
    mandate = positive_int(mandate_version, "Mandate version")
    actor = text(evaluated_by, "Evaluated by", required=True, limit=200)
    normalised_checks = _normalise_rule_checks(checks)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "rule_check")
    base = _base_record(_lifecycle_row(connection, identifier))
    if base is None:
        raise ValueError("Lifecycle data is corrupt.")
    required = {
        item["member_id"]
        for item in _committee_members(connection, identifier)
        if item["required_approver"]
    }
    explicit_override = _normalise_override(override, required_approvers=required)

    dossier = get_dossier_version(connection, base["dossier_id"], base["dossier_version"])
    dossier_passed = bool(
        dossier
        and dossier["status"] == "frozen"
        and verify_frozen_dossier(dossier)
        and dossier["content_hash"] == base["locked_dossier_hash"]
    )
    eligibility = check_security_eligibility(
        connection,
        base["ticker"],
        snapshot_id=base["universe_snapshot_id"],
    )
    system_checks = [
        {
            "rule_id": "locked_dossier_integrity",
            "passed": dossier_passed,
            "content_hash": base["locked_dossier_hash"],
        },
        {
            "rule_id": "authoritative_universe_current",
            "passed": bool(eligibility["can_trade"]),
            "universe_snapshot_id": base["universe_snapshot_id"],
            "reasons": eligibility["reasons"],
        },
    ]
    all_checks = system_checks + normalised_checks
    failed = [item for item in all_checks if not item["passed"]]
    passed = not failed
    if explicit_override:
        raw_scope = explicit_override.get("scope")
        if not isinstance(raw_scope, (list, tuple)):
            raise ValueError("Rule override scope must list every failed rule id.")
        scope = {
            text(item, "Override scope rule id", required=True, limit=200)
            for item in raw_scope
        }
        failed_ids = {str(item["rule_id"]) for item in failed}
        if not failed_ids:
            raise ValueError("A rule override is not permitted when every rule check passes.")
        missing_scope = failed_ids - scope
        if missing_scope:
            raise ValueError(
                "Rule override scope is missing failed rules: "
                + ", ".join(sorted(missing_scope))
                + "."
            )
        explicit_override["scope"] = sorted(scope)
    effective_pass = passed or bool(explicit_override)
    _, checks_json = json_array(all_checks, "Rule checks")
    _, failed_json = json_array(failed, "Failed rules")
    _, override_json = json_object(explicit_override, "Rule override")
    cursor = connection.execute(
        """
        INSERT INTO canonical_investment_rule_checks (
            lifecycle_id, rulebook_version, mandate_version, checks_json,
            failed_rules_json, override_json, passed, effective_pass,
            evaluated_by, evaluated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            identifier,
            rulebook,
            mandate,
            checks_json,
            failed_json,
            override_json,
            int(passed),
            int(effective_pass),
            actor,
            timestamp,
        ),
    )
    check_id = inserted_id(connection, cursor, "canonical_investment_rule_checks")
    target = "final_approval" if effective_pass else "rejected"
    _transition(
        connection,
        identifier,
        expected="rule_check",
        target=target,
        event_type="rule_check_completed",
        actor=actor,
        payload={
            "rule_check_id": check_id,
            "rulebook_version": rulebook,
            "mandate_version": mandate,
            "passed": passed,
            "effective_pass": effective_pass,
            "failed_rule_ids": [item["rule_id"] for item in failed],
            "override": explicit_override,
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after rule checks.")
    return record


def _final_approval_records(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT a.id, a.member_id, m.display_name, a.decision, a.comment,
               a.submitted_at
        FROM canonical_investment_final_approvals a
        JOIN canonical_investment_committee_members m
          ON m.lifecycle_id = a.lifecycle_id AND m.member_id = a.member_id
        WHERE a.lifecycle_id = ? ORDER BY a.id
        """,
        (lifecycle_id,),
    ).fetchall()
    return [
        {
            "id": int(row_value(row, "id", 0)),
            "member_id": str(row_value(row, "member_id", 1)),
            "display_name": str(row_value(row, "display_name", 2)),
            "decision": str(row_value(row, "decision", 3)),
            "comment": str(row_value(row, "comment", 4)),
            "submitted_at": str(row_value(row, "submitted_at", 5)),
        }
        for row in rows
    ]


def submit_final_approval(
    connection: Any,
    lifecycle_id: int,
    member_id: str,
    *,
    decision: str,
    comment: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    member = _member_id(member_id)
    choice = enum(decision, "Final approval decision", FINAL_APPROVAL_DECISIONS)
    note = text(comment, "Approval comment", required=True)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "final_approval")
    committee = {item["member_id"]: item for item in _committee_members(connection, identifier)}
    approver = committee.get(member)
    if approver is None or not approver["required_approver"]:
        raise ValueError("Only a designated final approver may sign this proposal.")
    if not approver["present"] or approver["conflicted"]:
        raise ValueError("A final approver must be present and non-conflicted.")
    if connection.execute(
        """
        SELECT 1 FROM canonical_investment_final_approvals
        WHERE lifecycle_id = ? AND member_id = ?
        """,
        (identifier, member),
    ).fetchone() is not None:
        raise ValueError("This final approver has already signed; approvals are immutable.")
    connection.execute(
        """
        INSERT INTO canonical_investment_final_approvals (
            lifecycle_id, member_id, decision, comment, submitted_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (identifier, member, choice, note, timestamp),
    )
    required_ids = {
        item["member_id"] for item in committee.values() if item["required_approver"]
    }
    approvals = _final_approval_records(connection, identifier)
    approved_ids = {item["member_id"] for item in approvals if item["decision"] == "approve"}
    if choice == "reject":
        target = "rejected"
    elif approved_ids == required_ids:
        target = "sizing"
    else:
        target = "final_approval"
    if target == "final_approval":
        connection.execute(
            "UPDATE canonical_investment_lifecycles SET updated_at = ? WHERE id = ?",
            (timestamp, identifier),
        )
        _append_audit_event(
            connection,
            identifier,
            event_type="final_approval_submitted",
            from_state="final_approval",
            to_state="final_approval",
            actor=member,
            payload={"decision": choice, "remaining_approver_ids": sorted(required_ids - approved_ids)},
            timestamp=timestamp,
        )
    else:
        _transition(
            connection,
            identifier,
            expected="final_approval",
            target=target,
            event_type="final_approval_completed" if target == "sizing" else "final_approval_rejected",
            actor=member,
            payload={
                "decision": choice,
                "approvals": [
                    {"member_id": item["member_id"], "decision": item["decision"]}
                    for item in approvals
                ],
            },
            timestamp=timestamp,
        )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after final approval.")
    return record


def record_position_sizing(
    connection: Any,
    lifecycle_id: int,
    sizing: Mapping[str, Any],
    *,
    sized_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    sizing_copy, _ = json_object(sizing, "Position sizing")
    target_weight = finite_number(
        sizing_copy.get("target_weight_pct"),
        "Target weight",
        minimum=0.000001,
        maximum=100,
    )
    rationale = text(sizing_copy.get("rationale"), "Sizing rationale", required=True)
    starter = boolean(sizing_copy.get("starter_position", False), "Starter position")
    if starter and not sizing_copy.get("expansion_conditions"):
        raise ValueError("Starter positions require explicit expansion conditions.")
    canonical_sizing = {
        **sizing_copy,
        "target_weight_pct": target_weight,
        "rationale": rationale,
        "starter_position": starter,
    }
    _, sizing_json = json_object(canonical_sizing, "Position sizing")
    actor = text(sized_by, "Sized by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "sizing")
    rule_checks = list_rule_checks(connection, identifier)
    if not rule_checks or not rule_checks[0]["effective_pass"]:
        raise ValueError("Position sizing requires a successful linked rule check.")
    latest_rule_check = rule_checks[0]
    binding_limits: list[float] = []
    for check in latest_rule_check["checks"]:
        if check.get("rule_id") not in {"max_position", "max_position_pct", "position_size"}:
            continue
        for key in ("limit_pct", "max_pct", "limit"):
            if check.get(key) is not None:
                candidate = finite_number(check[key], f"{check['rule_id']} {key}", minimum=0, maximum=100)
                binding_limits.append(float(candidate))
                break
    binding_max = min(binding_limits) if binding_limits else None
    if binding_max is not None and float(target_weight) > binding_max:
        raise ValueError(
            f"Target weight {target_weight:g}% exceeds the checked maximum {binding_max:g}%."
        )
    post_round = get_vote_round(connection, identifier, "post")
    buy_weights = [
        item["proposed_weight_pct"]
        for item in post_round["ballots"]
        if item["vote_scope"] == "investment"
        and item["decision"] == "buy"
        and item["proposed_weight_pct"] is not None
    ]
    canonical_sizing["committee_buy_weight_range"] = (
        None if not buy_weights else {"min": min(buy_weights), "max": max(buy_weights)}
    )
    canonical_sizing["validated_rule_check_id"] = latest_rule_check["id"]
    canonical_sizing["binding_max_weight_pct"] = binding_max
    _, sizing_json = json_object(canonical_sizing, "Position sizing")
    connection.execute(
        """
        INSERT INTO canonical_investment_sizing (
            lifecycle_id, sizing_json, sized_by, sized_at
        ) VALUES (?, ?, ?, ?)
        """,
        (identifier, sizing_json, actor, timestamp),
    )
    _transition(
        connection,
        identifier,
        expected="sizing",
        target="wins_execution",
        event_type="position_sized",
        actor=actor,
        payload={
            "target_weight_pct": target_weight,
            "starter_position": starter,
            "committee_buy_weight_range": canonical_sizing["committee_buy_weight_range"],
            "validated_rule_check_id": latest_rule_check["id"],
            "binding_max_weight_pct": binding_max,
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after sizing.")
    return record


def _iso_event_time(value: Any, name: str) -> str:
    if isinstance(value, datetime):
        return utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    raw = text(value, name, required=True, limit=100)
    try:
        if "T" in raw or " " in raw:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return utc_timestamp(parsed)
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date or timestamp.") from exc


def record_wins_execution(
    connection: Any,
    lifecycle_id: int,
    execution: Mapping[str, Any],
    *,
    recorded_by: str,
    now: datetime | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record the actual WInS transaction linked to the approved proposal."""

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    execution_copy, _ = json_object(execution, "WInS execution")
    transaction_id = text(
        execution_copy.get("wins_transaction_id", execution_copy.get("wins_trade_id")),
        "WInS transaction id",
        required=True,
        limit=200,
    )
    side = enum(execution_copy.get("side"), "Execution side", {"buy", "sell"})
    quantity = finite_number(execution_copy.get("quantity"), "Executed quantity", minimum=0.00000001)
    price = finite_number(
        execution_copy.get("average_price", execution_copy.get("price")),
        "Average execution price",
        minimum=0.00000001,
    )
    executed_at = _iso_event_time(execution_copy.get("executed_at"), "Executed at")
    currency = text(
        execution_copy.get("currency"),
        "Execution currency",
        required=True,
        limit=16,
    ).upper()
    canonical_execution = {
        **execution_copy,
        "wins_transaction_id": transaction_id,
        "side": side,
        "quantity": quantity,
        "average_price": price,
        "executed_at": executed_at,
        "currency": currency,
    }
    _, execution_json = json_object(canonical_execution, "WInS execution")
    actor = text(recorded_by, "Recorded by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "wins_execution")
    base = _base_record(_lifecycle_row(connection, identifier))
    if base is None:
        raise ValueError("Lifecycle data is corrupt.")
    proposal_action = str(base["proposal"].get("action") or "").strip().lower()
    expected_side = {
        "buy": "buy",
        "add": "buy",
        "initiate": "buy",
        "sell": "sell",
        "trim": "sell",
        "exit": "sell",
    }.get(proposal_action)
    if expected_side is not None and side != expected_side:
        raise ValueError(
            f"Execution side {side!r} does not match proposal action {proposal_action!r}."
        )
    connection.execute(
        """
        INSERT INTO canonical_investment_wins_executions (
            lifecycle_id, wins_transaction_id, execution_json, recorded_by, recorded_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (identifier, transaction_id, execution_json, actor, timestamp),
    )
    _transition(
        connection,
        identifier,
        expected="wins_execution",
        target="reconciliation",
        event_type="wins_execution_recorded",
        actor=actor,
        payload={
            "wins_transaction_id": transaction_id,
            "side": side,
            "quantity": quantity,
            "average_price": price,
            "executed_at": executed_at,
            "currency": currency,
        },
        timestamp=timestamp,
    )
    if commit:
        commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after WInS execution.")
    return record


_CANONICAL_PIPELINE_SOURCE = "portfolio_pipeline/competition"


def _tracker_position_rows(connection: Any) -> list[dict[str, Any]]:
    """Rebuild the exact tracker input used by the canonical reconciliation ledger."""

    try:
        rows = connection.execute(
            """
            SELECT id, ticker, quantity, entry_price, last_price, security_type,
                   currency, status, lifecycle_id, entry_date
            FROM competition_positions
            WHERE lower(COALESCE(status, 'open')) IN ('open', 'pending_reconciliation')
            ORDER BY entry_date, id
            """
        ).fetchall()
    except Exception as exc:
        raise ValueError(
            "The canonical tracker projection is unavailable; stage the pending position first."
        ) from exc

    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            quantity = float(row_value(row, "quantity", 2) or 0.0)
            entry_price = float(row_value(row, "entry_price", 3) or 0.0)
            price = float(row_value(row, "last_price", 4) or entry_price or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("The canonical tracker projection contains invalid numeric data.") from exc
        currency = str(row_value(row, "currency", 6) or "").strip().upper() or None
        result.append(
            {
                "ticker": str(row_value(row, "ticker", 1) or "").strip().upper(),
                "quantity": quantity,
                "current_price": price,
                "market_value": quantity * price,
                "total_cost": quantity * entry_price,
                "asset_type": str(row_value(row, "security_type", 5) or "Unknown"),
                "currency": currency,
            }
        )
    return result


def _current_tracker_cash(connection: Any) -> float:
    """Rebuild competition cash from the complete tracker transaction history."""
    from src.portfolio_tracker.wharton_competition import calculate_portfolio_performance

    try:
        rows = connection.execute(
            "SELECT * FROM competition_positions ORDER BY entry_date, id"
        ).fetchall()
        positions = [dict(row) for row in rows]
    except Exception as exc:
        raise ValueError(
            "The tracker cash balance cannot be rebuilt from competition positions."
        ) from exc
    performance = calculate_portfolio_performance(positions, live_prices={})
    return float(
        performance["cash_before_pnl"]
        + performance["realized_pnl"]
        + performance["open_cash_income"]
    )


def _pending_tracker_position(
    connection: Any,
    lifecycle_id: int,
    *,
    lifecycle: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        rows = connection.execute(
            """
            SELECT id, ticker, quantity, currency, status, lifecycle_id
            FROM competition_positions
            WHERE lifecycle_id = ? AND lower(COALESCE(status, '')) = 'pending_reconciliation'
            ORDER BY id
            """,
            (lifecycle_id,),
        ).fetchall()
    except Exception as exc:
        raise ValueError(
            "The canonical tracker projection is unavailable; stage the pending position first."
        ) from exc
    if len(rows) != 1:
        raise ValueError(
            "Exactly one pending tracker position must be linked to this investment lifecycle."
        )
    row = rows[0]
    position_id = positive_int(row_value(row, "id", 0), "Tracker position id")
    expected_ticker = str(lifecycle.get("ticker") or "").strip().upper()
    actual_ticker = str(row_value(row, "ticker", 1) or "").strip().upper()
    if actual_ticker != expected_ticker:
        raise ValueError("The pending tracker ticker does not match the investment lifecycle.")
    try:
        pending_quantity = float(row_value(row, "quantity", 2))
        executed_quantity = float(execution["quantity"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The pending tracker position has no valid quantity.") from exc
    if abs(pending_quantity - executed_quantity) > 1e-8:
        raise ValueError("The pending tracker quantity does not match the WInS execution.")
    pending_currency = str(row_value(row, "currency", 3) or "").strip().upper()
    execution_currency = str(execution.get("currency") or "").strip().upper()
    if pending_currency != execution_currency:
        raise ValueError("The pending tracker currency does not match the WInS execution.")
    return {
        "position_id": position_id,
        "ticker": actual_ticker,
        "quantity": pending_quantity,
        "currency": pending_currency,
    }


def _canonical_clean_reconciliation(
    connection: Any,
    lifecycle_id: int,
    *,
    now: datetime | None,
    max_age_seconds: float = 86_400,
) -> dict[str, Any]:
    """Derive lifecycle activation exclusively from the persisted canonical pipeline."""

    # Local imports keep the domain stores independently initialisable and avoid
    # coupling module import order to the portfolio pipeline.
    from src.data.reliability import assess_snapshot, verify_snapshot_integrity
    from src.portfolio_tracker.operating_system_store import get_current_record
    from src.portfolio_tracker.portfolio_pipeline import build_live_portfolio_pipeline
    from src.portfolio_tracker.reconciliation_ledger import latest_reconciliation
    from src.portfolio_tracker.wins_reconciliation import reconcile_wins_positions

    lifecycle = _base_record(_lifecycle_row(connection, lifecycle_id))
    execution_record = _execution_record(connection, lifecycle_id)
    if lifecycle is None or execution_record is None:
        raise ValueError("The lifecycle execution cannot be resolved.")
    execution = execution_record.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("The lifecycle execution is corrupt.")
    pending = _pending_tracker_position(
        connection,
        lifecycle_id,
        lifecycle=lifecycle,
        execution=execution,
    )

    persisted = get_current_record(
        connection,
        "portfolio_pipeline",
        "competition",
    )
    if persisted is None or persisted.get("status") != "active":
        raise ValueError("The persisted canonical portfolio pipeline is unavailable.")
    workspace = persisted.get("payload")
    if not isinstance(workspace, Mapping):
        raise ValueError("The persisted canonical portfolio pipeline is corrupt.")
    if canonical_hash(dict(workspace)) != str(persisted.get("payload_hash") or ""):
        raise ValueError("The persisted canonical portfolio pipeline failed its integrity check.")
    snapshots = workspace.get("snapshots")
    ledger = workspace.get("ledger")
    if not isinstance(snapshots, list) or not isinstance(ledger, Mapping):
        raise ValueError("The persisted canonical portfolio pipeline is incomplete.")

    active_context = {"status": "active", "active": True}
    pipeline = build_live_portfolio_pipeline(
        snapshots,
        ledger,
        mandate=active_context,
        rulebook=active_context,
        expected_return_assumptions=active_context,
        now=now,
        max_age_seconds=max_age_seconds,
        min_completeness_pct=100.0,
    )
    gate = pipeline.get("reconciliation_gate")
    canonical_snapshot = pipeline.get("canonical_snapshot")
    if not isinstance(gate, Mapping) or not isinstance(canonical_snapshot, Mapping):
        raise ValueError("The canonical WInS reconciliation is unavailable.")
    if pipeline.get("authority") != "wins_reconciled" or gate.get("ready") is not True:
        blockers = ", ".join(str(item) for item in gate.get("blockers", []))
        detail = blockers or "no signed clean reconciliation"
        raise ValueError(f"Lifecycle activation is blocked by the canonical ledger: {detail}.")

    snapshot_id = str(canonical_snapshot.get("snapshot_id") or "")
    if (
        not snapshot_id
        or snapshot_id != str(gate.get("wins_snapshot_id") or "")
        or snapshot_id != str(pipeline.get("latest_wins_snapshot_id") or "")
    ):
        raise ValueError("Only the latest reconciled WInS snapshot can activate a lifecycle.")
    if not verify_snapshot_integrity(canonical_snapshot):
        raise ValueError("The canonical WInS snapshot failed its integrity check.")
    assessment = assess_snapshot(
        canonical_snapshot,
        now=now,
        max_age_seconds=max_age_seconds,
        min_completeness_pct=100.0,
    )
    if not assessment["is_fresh"]:
        raise ValueError("The canonical WInS snapshot is stale or has an invalid timestamp.")
    if not assessment["complete_enough"]:
        raise ValueError("The canonical WInS snapshot is incomplete.")

    current = latest_reconciliation(ledger)
    if current is None or str(current.get("reconciliation_id") or "") != str(
        gate.get("latest_reconciliation_id") or ""
    ):
        raise ValueError("The canonical reconciliation record cannot be resolved.")
    sign_off = current.get("sign_off")
    if not isinstance(sign_off, Mapping) or sign_off.get("decision") != "approved":
        raise ValueError("The canonical reconciliation requires approval.")
    if str(sign_off.get("signed_off_by") or "").strip().casefold() == str(
        current.get("owner") or ""
    ).strip().casefold():
        raise ValueError("The canonical reconciliation requires an independent approval.")
    if (
        current.get("base_is_clean") is not True
        or current.get("all_exceptions_closed") is not True
        or current.get("exceptions")
        or int(current.get("open_exception_count") or 0)
    ):
        raise ValueError("Only a clean reconciliation without exceptions can activate a lifecycle.")
    reconciliation_result = current.get("result")
    cash_comparison = (
        reconciliation_result.get("cash_comparison")
        if isinstance(reconciliation_result, Mapping)
        else None
    )
    if not isinstance(cash_comparison, Mapping) or cash_comparison.get("status") != "matched":
        raise ValueError(
            "Lifecycle activation requires a new cash-aware canonical reconciliation."
        )

    tracked_snapshot_id = str(current.get("tracked_snapshot_id") or "")
    tracked_snapshots = [
        item
        for item in snapshots
        if isinstance(item, Mapping)
        and str(item.get("snapshot_id") or "") == tracked_snapshot_id
    ]
    if len(tracked_snapshots) != 1:
        raise ValueError(
            "The exact tracker snapshot used for reconciliation is not persisted."
        )
    tracked_snapshot = tracked_snapshots[0]
    if not verify_snapshot_integrity(tracked_snapshot):
        raise ValueError("The reconciled tracker snapshot failed its integrity check.")
    tracked_source = (
        tracked_snapshot.get("source")
        if isinstance(tracked_snapshot.get("source"), Mapping)
        else {}
    )
    tracked_provider = str(tracked_source.get("provider") or "").strip().casefold()
    tracked_reference = str(tracked_source.get("reference") or "").strip().casefold()
    if "tracker" not in tracked_provider and "competition_positions" not in tracked_reference:
        raise ValueError("The reconciled tracked snapshot is not a Portfolio Tracker snapshot.")
    tracked_assessment = assess_snapshot(
        tracked_snapshot,
        now=now,
        max_age_seconds=max_age_seconds,
        min_completeness_pct=100.0,
    )
    if not tracked_assessment["complete_enough"]:
        raise ValueError("The reconciled tracker snapshot is incomplete.")
    tracked_payload = tracked_snapshot.get("payload")
    tracked_positions = (
        tracked_payload.get("positions") if isinstance(tracked_payload, Mapping) else None
    )
    if not isinstance(tracked_positions, list):
        raise ValueError("The reconciled tracker snapshot has no position list.")
    tracker_cash = tracked_payload.get("cash_value")
    try:
        frozen_tracker_cash = float(tracker_cash)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The reconciled tracker snapshot has no valid cash balance.") from exc
    tolerances = current.get("tolerances")
    if not isinstance(tolerances, Mapping):
        tolerances = {}
    cash_tolerance = float(tolerances.get("cash") or tolerances.get("currency") or 0.01)
    if abs(_current_tracker_cash(connection) - frozen_tracker_cash) > cash_tolerance:
        raise ValueError(
            "The current tracker cash differs from the snapshot used for reconciliation."
        )
    tracker_rows = _tracker_position_rows(connection)
    lifecycle_ticker = str(lifecycle.get("ticker") or "").strip().upper()
    tracked_ticker_positions = [
        item
        for item in tracked_positions
        if isinstance(item, Mapping)
        and str(item.get("ticker") or "").strip().upper() == lifecycle_ticker
    ]
    current_ticker_rows = [
        item for item in tracker_rows if item.get("ticker") == lifecycle_ticker
    ]
    if len(tracked_ticker_positions) != 1 or len(current_ticker_rows) < 1:
        raise ValueError("The reconciled tracker snapshot does not contain the lifecycle ticker.")
    try:
        source_lot_count = int(tracked_ticker_positions[0].get("source_lot_count"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The reconciled tracker snapshot has no valid lot membership.") from exc
    if source_lot_count != len(current_ticker_rows):
        raise ValueError("The reconciled tracker lot membership no longer matches the tracker.")
    tracker_comparison = reconcile_wins_positions(
        tracked_positions,
        tracker_rows,
        quantity_tolerance=float(tolerances.get("quantity") or 1e-8),
        currency_tolerance=float(tolerances.get("currency") or 0.01),
    )
    if tracker_comparison.get("is_reconciled") is not True:
        raise ValueError(
            "The current tracker positions differ from the snapshot used for reconciliation."
        )

    ticker_value = lifecycle_ticker
    payload = canonical_snapshot.get("payload")
    positions = payload.get("positions") if isinstance(payload, Mapping) else None
    if not isinstance(positions, list):
        raise ValueError("The canonical WInS snapshot has no position list.")
    snapshot_matches = [
        item
        for item in positions
        if isinstance(item, Mapping)
        and str(item.get("ticker") or "").strip().upper() == ticker_value
    ]
    matched_rows = (
        reconciliation_result.get("matched", [])
        if isinstance(reconciliation_result, Mapping)
        else []
    )
    reconciliation_matches = [
        item
        for item in matched_rows
        if isinstance(item, Mapping)
        and str(item.get("ticker") or "").strip().upper() == ticker_value
    ]
    if len(snapshot_matches) != 1 or len(reconciliation_matches) != 1:
        raise ValueError(
            "The latest reconciliation does not contain exactly one aggregate for this ticker."
        )
    snapshot_position = snapshot_matches[0]
    matched = reconciliation_matches[0]
    wins_match = matched.get("wins") if isinstance(matched.get("wins"), Mapping) else {}
    tracked_match = (
        matched.get("tracked") if isinstance(matched.get("tracked"), Mapping) else {}
    )
    try:
        snapshot_quantity = float(snapshot_position.get("quantity"))
        wins_quantity = float(wins_match.get("quantity"))
        tracked_quantity = float(tracked_match.get("quantity"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The canonical aggregate has no valid quantity.") from exc
    if (
        matched.get("status") != "matched"
        or abs(snapshot_quantity - wins_quantity) > 1e-8
        or abs(wins_quantity - tracked_quantity) > 1e-8
    ):
        raise ValueError("The canonical WInS and tracker aggregates do not match.")
    snapshot_currency = str(snapshot_position.get("currency") or "").strip().upper()
    if snapshot_currency != pending["currency"]:
        raise ValueError("The canonical WInS currency does not match the lifecycle execution.")

    return {
        "status": "clean",
        "wins_snapshot_id": snapshot_id,
        "canonical_reconciliation_id": str(current["reconciliation_id"]),
        "canonical_source": _CANONICAL_PIPELINE_SOURCE,
        "position_id": str(pending["position_id"]),
        "exceptions": [],
        "_position_ticker": pending["ticker"],
        "_position_quantity": pending["quantity"],
        "_position_currency": pending["currency"],
    }


def record_wins_reconciliation(
    connection: Any,
    lifecycle_id: int,
    reconciliation: Mapping[str, Any] | None = None,
    *,
    recorded_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append an exception or atomically activate from the persisted canonical ledger.

    Passing no reconciliation derives the clean result, snapshot and tracker
    position from ``portfolio_pipeline/competition``.  A supplied clean mapping
    remains accepted for compatibility, but every canonical identifier must
    agree with that persisted source; caller assertions alone never activate a
    position.
    """

    identifier = positive_int(lifecycle_id, "Lifecycle id")
    _ensure(connection)
    _require_state(connection, identifier, "reconciliation")
    supplied = reconciliation is not None
    if reconciliation is None:
        reconciliation_copy: dict[str, Any] = {"status": "clean"}
    else:
        reconciliation_copy, _ = json_object(reconciliation, "WInS reconciliation")
    status = enum(
        reconciliation_copy.get("status"),
        "Reconciliation status",
        {"clean", "open_exceptions"},
    )
    raw_exceptions = reconciliation_copy.get("exceptions", [])
    exceptions, _ = json_array(raw_exceptions, "Reconciliation exceptions")
    if status == "clean" and exceptions:
        raise ValueError("A clean reconciliation cannot contain open exceptions.")
    if status == "open_exceptions" and not exceptions:
        raise ValueError("Open-exception reconciliation must describe at least one exception.")

    if status == "clean":
        derived = _canonical_clean_reconciliation(connection, identifier, now=now)
        activation_ticker = str(derived.pop("_position_ticker"))
        activation_quantity = float(derived.pop("_position_quantity"))
        activation_currency = str(derived.pop("_position_currency"))
        if supplied:
            supplied_source = text(
                reconciliation_copy.get("canonical_source"),
                "Canonical source",
                required=True,
                limit=100,
            )
            supplied_reconciliation_id = text(
                reconciliation_copy.get("canonical_reconciliation_id"),
                "Canonical reconciliation id",
                required=True,
                limit=300,
            )
            supplied_snapshot_id = text(
                reconciliation_copy.get("wins_snapshot_id"),
                "WInS snapshot id",
                required=True,
                limit=300,
            )
            supplied_position_id = text(
                reconciliation_copy.get("position_id"),
                "Position id",
                required=True,
                limit=300,
            )
            supplied_bindings = {
                "canonical_source": supplied_source,
                "canonical_reconciliation_id": supplied_reconciliation_id,
                "wins_snapshot_id": supplied_snapshot_id,
                "position_id": supplied_position_id,
            }
            expected_bindings = {
                key: str(derived[key]) for key in supplied_bindings
            }
            if supplied_bindings != expected_bindings:
                raise ValueError(
                    "Caller-supplied reconciliation bindings do not match the persisted "
                    "canonical portfolio pipeline."
                )
        canonical_reconciliation = {**reconciliation_copy, **derived}
        snapshot_id = str(derived["wins_snapshot_id"])
        position_id = str(derived["position_id"])
    else:
        snapshot_id = text(
            reconciliation_copy.get("wins_snapshot_id"),
            "WInS snapshot id",
            required=True,
            limit=300,
        )
        position_id = None
        canonical_reconciliation = {
            **reconciliation_copy,
            "status": status,
            "wins_snapshot_id": snapshot_id,
            "position_id": position_id,
            "exceptions": exceptions,
        }
    canonical_reconciliation = {
        **reconciliation_copy,
        **canonical_reconciliation,
        "status": status,
        "wins_snapshot_id": snapshot_id,
        "position_id": position_id,
        "exceptions": exceptions,
    }
    _, reconciliation_json = json_object(canonical_reconciliation, "WInS reconciliation")
    actor = text(recorded_by, "Recorded by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    try:
        cursor = connection.execute(
            """
            INSERT INTO canonical_investment_reconciliations (
                lifecycle_id, status, wins_snapshot_id, position_id,
                reconciliation_json, recorded_by, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                status,
                snapshot_id,
                position_id,
                reconciliation_json,
                actor,
                timestamp,
            ),
        )
        reconciliation_id = inserted_id(
            connection, cursor, "canonical_investment_reconciliations"
        )
        if status == "clean":
            connection.execute(
                """
                UPDATE competition_positions SET status = 'open'
                WHERE id = ? AND lifecycle_id = ?
                  AND lower(COALESCE(status, '')) = 'pending_reconciliation'
                  AND upper(trim(ticker)) = ? AND abs(quantity - ?) <= 0.00000001
                  AND upper(trim(currency)) = ?
                """,
                (
                    positive_int(position_id, "Position id"),
                    identifier,
                    activation_ticker,
                    activation_quantity,
                    activation_currency,
                ),
            )
            promoted = connection.execute(
                "SELECT status FROM competition_positions WHERE id = ? AND lifecycle_id = ?",
                (positive_int(position_id, "Position id"), identifier),
            ).fetchone()
            if promoted is None or str(row_value(promoted, "status", 0)).strip().lower() != "open":
                raise RuntimeError("The pending tracker position could not be activated.")
            connection.execute(
                "UPDATE canonical_investment_lifecycles SET current_position_id = ? WHERE id = ?",
                (position_id, identifier),
            )
            _transition(
                connection,
                identifier,
                expected="reconciliation",
                target="active",
                event_type="wins_reconciliation_clean",
                actor=actor,
                payload={
                    "reconciliation_id": reconciliation_id,
                    "canonical_reconciliation_id": canonical_reconciliation[
                        "canonical_reconciliation_id"
                    ],
                    "canonical_source": canonical_reconciliation["canonical_source"],
                    "wins_snapshot_id": snapshot_id,
                    "position_id": position_id,
                },
                timestamp=timestamp,
            )
        else:
            connection.execute(
                "UPDATE canonical_investment_lifecycles SET updated_at = ? WHERE id = ?",
                (timestamp, identifier),
            )
            _append_audit_event(
                connection,
                identifier,
                event_type="wins_reconciliation_exceptions",
                from_state="reconciliation",
                to_state="reconciliation",
                actor=actor,
                payload={
                    "reconciliation_id": reconciliation_id,
                    "wins_snapshot_id": snapshot_id,
                    "exception_count": len(exceptions),
                },
                timestamp=timestamp,
            )
        commit_and_sync(connection)
    except Exception:
        connection.rollback()
        raise
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after reconciliation.")
    return record


def append_position_review(
    connection: Any,
    lifecycle_id: int,
    payload: Mapping[str, Any],
    *,
    outcome: str,
    reviewed_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    result = enum(outcome, "Review outcome", {"confirmed", "watch", "invalidated"})
    payload_copy, payload_json = json_object(payload, "Position review")
    actor = text(reviewed_by, "Reviewed by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "active")
    cursor = connection.execute(
        """
        INSERT INTO canonical_investment_position_reviews (
            lifecycle_id, outcome, payload_json, reviewed_by, reviewed_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (identifier, result, payload_json, actor, timestamp),
    )
    review_id = inserted_id(connection, cursor, "canonical_investment_position_reviews")
    connection.execute(
        "UPDATE canonical_investment_lifecycles SET updated_at = ? WHERE id = ?",
        (timestamp, identifier),
    )
    _append_audit_event(
        connection,
        identifier,
        event_type="position_review_appended",
        from_state="active",
        to_state="active",
        actor=actor,
        payload={
            "review_id": review_id,
            "outcome": result,
            "review_hash": canonical_hash(payload_copy),
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after position review.")
    return record


def record_position_exit(
    connection: Any,
    lifecycle_id: int,
    exit_record: Mapping[str, Any],
    *,
    recorded_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    exit_copy, _ = json_object(exit_record, "Position exit")
    transaction_id = text(
        exit_copy.get("wins_transaction_id", exit_copy.get("wins_trade_id")),
        "Exit WInS transaction id",
        required=True,
        limit=200,
    )
    reason = text(exit_copy.get("reason"), "Exit reason", required=True)
    executed_at = _iso_event_time(exit_copy.get("executed_at"), "Exit executed at")
    canonical_exit = {
        **exit_copy,
        "wins_transaction_id": transaction_id,
        "reason": reason,
        "executed_at": executed_at,
    }
    _, exit_json = json_object(canonical_exit, "Position exit")
    actor = text(recorded_by, "Recorded by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    _require_state(connection, identifier, "active")
    connection.execute(
        """
        INSERT INTO canonical_investment_exits (
            lifecycle_id, wins_transaction_id, exit_json, recorded_by, recorded_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (identifier, transaction_id, exit_json, actor, timestamp),
    )
    _transition(
        connection,
        identifier,
        expected="active",
        target="exited",
        event_type="position_exited",
        actor=actor,
        payload={
            "wins_transaction_id": transaction_id,
            "executed_at": executed_at,
            "reason": reason,
        },
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after exit.")
    return record


def withdraw_proposal(
    connection: Any,
    lifecycle_id: int,
    *,
    reason: str,
    withdrawn_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    explanation = text(reason, "Withdrawal reason", required=True)
    actor = text(withdrawn_by, "Withdrawn by", required=True, limit=200)
    timestamp = utc_timestamp(now)
    _ensure(connection)
    allowed = {
        "proposal",
        "dossier_frozen",
        "pre_vote",
        "discussion",
        "post_vote",
        "rule_check",
        "final_approval",
        "sizing",
    }
    current = _require_state(connection, identifier, allowed)
    _transition(
        connection,
        identifier,
        expected=current,
        target="withdrawn",
        event_type="proposal_withdrawn",
        actor=actor,
        payload={"reason": explanation},
        timestamp=timestamp,
    )
    commit_and_sync(connection)
    record = get_investment_lifecycle(connection, identifier)
    if record is None:
        raise RuntimeError("The lifecycle could not be read after withdrawal.")
    return record


def _discussion_record(connection: Any, lifecycle_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT bull_case, bear_case, q_and_a_json, notes, recorded_by, recorded_at
        FROM canonical_investment_discussions WHERE lifecycle_id = ?
        """,
        (lifecycle_id,),
    ).fetchone()
    if row is None:
        return None
    q_and_a = decode_array(row_value(row, "q_and_a_json", 2))
    if q_and_a is None:
        return None
    return {
        "bull_case": str(row_value(row, "bull_case", 0)),
        "bear_case": str(row_value(row, "bear_case", 1)),
        "q_and_a": q_and_a,
        "notes": str(row_value(row, "notes", 3) or ""),
        "recorded_by": str(row_value(row, "recorded_by", 4)),
        "recorded_at": str(row_value(row, "recorded_at", 5)),
    }


def list_rule_checks(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rows = connection.execute(
        """
        SELECT id, rulebook_version, mandate_version, checks_json,
               failed_rules_json, override_json, passed, effective_pass,
               evaluated_by, evaluated_at
        FROM canonical_investment_rule_checks
        WHERE lifecycle_id = ? ORDER BY id DESC
        """,
        (identifier,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        checks = decode_array(row_value(row, "checks_json", 3))
        failed = decode_array(row_value(row, "failed_rules_json", 4))
        override = decode_object(row_value(row, "override_json", 5))
        if checks is None or failed is None or override is None:
            continue
        records.append(
            {
                "id": int(row_value(row, "id", 0)),
                "lifecycle_id": identifier,
                "rulebook_version": int(row_value(row, "rulebook_version", 1)),
                "mandate_version": int(row_value(row, "mandate_version", 2)),
                "checks": checks,
                "failed_rules": failed,
                "override": override,
                "passed": bool(row_value(row, "passed", 6)),
                "effective_pass": bool(row_value(row, "effective_pass", 7)),
                "evaluated_by": str(row_value(row, "evaluated_by", 8)),
                "evaluated_at": str(row_value(row, "evaluated_at", 9)),
            }
        )
    return records


def _sizing_record(connection: Any, lifecycle_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT sizing_json, sized_by, sized_at
        FROM canonical_investment_sizing WHERE lifecycle_id = ?
        """,
        (lifecycle_id,),
    ).fetchone()
    if row is None:
        return None
    sizing = decode_object(row_value(row, "sizing_json", 0))
    if sizing is None:
        return None
    return {
        "sizing": sizing,
        "sized_by": str(row_value(row, "sized_by", 1)),
        "sized_at": str(row_value(row, "sized_at", 2)),
    }


def _execution_record(connection: Any, lifecycle_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT wins_transaction_id, execution_json, recorded_by, recorded_at
        FROM canonical_investment_wins_executions WHERE lifecycle_id = ?
        """,
        (lifecycle_id,),
    ).fetchone()
    if row is None:
        return None
    execution = decode_object(row_value(row, "execution_json", 1))
    if execution is None:
        return None
    return {
        "wins_transaction_id": str(row_value(row, "wins_transaction_id", 0)),
        "execution": execution,
        "recorded_by": str(row_value(row, "recorded_by", 2)),
        "recorded_at": str(row_value(row, "recorded_at", 3)),
    }


def list_wins_reconciliations(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rows = connection.execute(
        """
        SELECT id, status, wins_snapshot_id, position_id,
               reconciliation_json, recorded_by, recorded_at
        FROM canonical_investment_reconciliations
        WHERE lifecycle_id = ? ORDER BY id DESC
        """,
        (identifier,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        reconciliation = decode_object(row_value(row, "reconciliation_json", 4))
        if reconciliation is None:
            continue
        records.append(
            {
                "id": int(row_value(row, "id", 0)),
                "lifecycle_id": identifier,
                "status": str(row_value(row, "status", 1)),
                "wins_snapshot_id": str(row_value(row, "wins_snapshot_id", 2)),
                "position_id": (
                    None if row_value(row, "position_id", 3) is None else str(row_value(row, "position_id", 3))
                ),
                "reconciliation": reconciliation,
                "recorded_by": str(row_value(row, "recorded_by", 5)),
                "recorded_at": str(row_value(row, "recorded_at", 6)),
            }
        )
    return records


def list_position_reviews(connection: Any, lifecycle_id: int) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rows = connection.execute(
        """
        SELECT id, outcome, payload_json, reviewed_by, reviewed_at
        FROM canonical_investment_position_reviews
        WHERE lifecycle_id = ? ORDER BY id DESC
        """,
        (identifier,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = decode_object(row_value(row, "payload_json", 2))
        if payload is None:
            continue
        records.append(
            {
                "id": int(row_value(row, "id", 0)),
                "lifecycle_id": identifier,
                "outcome": str(row_value(row, "outcome", 1)),
                "payload": payload,
                "reviewed_by": str(row_value(row, "reviewed_by", 3)),
                "reviewed_at": str(row_value(row, "reviewed_at", 4)),
            }
        )
    return records


def _exit_record(connection: Any, lifecycle_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT wins_transaction_id, exit_json, recorded_by, recorded_at
        FROM canonical_investment_exits WHERE lifecycle_id = ?
        """,
        (lifecycle_id,),
    ).fetchone()
    if row is None:
        return None
    exit_payload = decode_object(row_value(row, "exit_json", 1))
    if exit_payload is None:
        return None
    return {
        "wins_transaction_id": str(row_value(row, "wins_transaction_id", 0)),
        "exit": exit_payload,
        "recorded_by": str(row_value(row, "recorded_by", 2)),
        "recorded_at": str(row_value(row, "recorded_at", 3)),
    }


def _audit_record(row: Any) -> dict[str, Any] | None:
    payload = decode_object(row_value(row, "payload_json", 7))
    if payload is None:
        return None
    return {
        "id": int(row_value(row, "id", 0)),
        "lifecycle_id": int(row_value(row, "lifecycle_id", 1)),
        "sequence": int(row_value(row, "sequence", 2)),
        "event_type": str(row_value(row, "event_type", 3)),
        "from_state": (
            None if row_value(row, "from_state", 4) is None else str(row_value(row, "from_state", 4))
        ),
        "to_state": str(row_value(row, "to_state", 5)),
        "actor": str(row_value(row, "actor", 6)),
        "payload": payload,
        "previous_hash": str(row_value(row, "previous_hash", 8)),
        "event_hash": str(row_value(row, "event_hash", 9)),
        "created_at": str(row_value(row, "created_at", 10)),
    }


def list_lifecycle_audit_events(
    connection: Any,
    lifecycle_id: int,
) -> list[dict[str, Any]]:
    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rows = connection.execute(
        """
        SELECT id, lifecycle_id, sequence, event_type, from_state, to_state,
               actor, payload_json, previous_hash, event_hash, created_at
        FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? ORDER BY sequence
        """,
        (identifier,),
    ).fetchall()
    return [record for row in rows if (record := _audit_record(row)) is not None]


def verify_lifecycle_audit_chain(connection: Any, lifecycle_id: int) -> dict[str, Any]:
    """Recompute every event hash and sequence link, reporting first failure."""

    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    rows = connection.execute(
        """
        SELECT id, lifecycle_id, sequence, event_type, from_state, to_state,
               actor, payload_json, previous_hash, event_hash, created_at
        FROM canonical_investment_audit_events
        WHERE lifecycle_id = ? ORDER BY sequence
        """,
        (identifier,),
    ).fetchall()
    previous_hash = ""
    for expected_sequence, row in enumerate(rows, start=1):
        event = _audit_record(row)
        if event is None:
            return {
                "lifecycle_id": identifier,
                "valid": False,
                "checked_events": expected_sequence - 1,
                "error_sequence": expected_sequence,
                "reason": "Audit payload is not a valid JSON object.",
            }
        hash_input = {
            "lifecycle_id": event["lifecycle_id"],
            "sequence": event["sequence"],
            "event_type": event["event_type"],
            "from_state": event["from_state"],
            "to_state": event["to_state"],
            "actor": event["actor"],
            "payload": event["payload"],
            "previous_hash": event["previous_hash"],
            "created_at": event["created_at"],
        }
        if event["sequence"] != expected_sequence:
            reason = "Audit event sequence is not contiguous."
        elif event["previous_hash"] != previous_hash:
            reason = "Audit previous-hash link is invalid."
        elif canonical_hash(hash_input) != event["event_hash"]:
            reason = "Audit event content hash is invalid."
        else:
            previous_hash = event["event_hash"]
            continue
        return {
            "lifecycle_id": identifier,
            "valid": False,
            "checked_events": expected_sequence - 1,
            "error_sequence": expected_sequence,
            "reason": reason,
        }
    return {
        "lifecycle_id": identifier,
        "valid": True,
        "checked_events": len(rows),
        "error_sequence": None,
        "reason": "",
        "head_hash": previous_hash,
    }


def get_investment_lifecycle(
    connection: Any,
    lifecycle_id: int,
) -> dict[str, Any] | None:
    _ensure(connection)
    identifier = positive_int(lifecycle_id, "Lifecycle id")
    try:
        row = _lifecycle_row(connection, identifier)
    except ValueError:
        return None
    base = _base_record(row)
    if base is None:
        return None
    reconciliations = list_wins_reconciliations(connection, identifier)
    rule_checks = list_rule_checks(connection, identifier)
    committee = _committee_members(connection, identifier)
    pre_vote = get_vote_round(connection, identifier, "pre")
    post_vote = get_vote_round(connection, identifier, "post")
    approvals = _final_approval_records(connection, identifier)
    audit_events = list_lifecycle_audit_events(connection, identifier)
    required_approver_ids = sorted(
        item["member_id"] for item in committee if item["required_approver"]
    )
    return {
        **base,
        "committee": committee,
        "committee_status": {
            "required_approver_ids": required_approver_ids,
            "eligible_member_ids": [
                item["member_id"]
                for item in committee
                if item["present"] and not item["conflicted"] and item["vote_scope"] != "observer"
            ],
            "conflicts": [
                {"member_id": item["member_id"], "reason": item["conflict_reason"]}
                for item in committee
                if item["conflicted"]
            ],
            "quorum": base["quorum"],
        },
        "pre_vote": pre_vote,
        "discussion": _discussion_record(connection, identifier),
        "post_vote": post_vote,
        "recorded_dissent": post_vote["dissent"],
        "latest_rule_check": None if not rule_checks else rule_checks[0],
        "final_approvals": approvals,
        "final_approval_complete": (
            bool(required_approver_ids)
            and {
                item["member_id"]
                for item in approvals
                if item["decision"] == "approve"
            }
            == set(required_approver_ids)
        ),
        "position_sizing": _sizing_record(connection, identifier),
        "wins_execution": _execution_record(connection, identifier),
        "latest_reconciliation": None if not reconciliations else reconciliations[0],
        "reconciliation_history": reconciliations,
        "position_reviews": list_position_reviews(connection, identifier),
        "exit": _exit_record(connection, identifier),
        "audit": verify_lifecycle_audit_chain(connection, identifier),
        "audit_events": audit_events,
    }


def list_investment_lifecycles(
    connection: Any,
    *,
    security_ticker: str | None = None,
    state: str | None = None,
) -> list[dict[str, Any]]:
    _ensure(connection)
    clauses: list[str] = []
    parameters: list[Any] = []
    if security_ticker is not None:
        clauses.append("ticker = ?")
        parameters.append(ticker(security_ticker))
    if state is not None:
        clauses.append("state = ?")
        parameters.append(enum(state, "Lifecycle state", LIFECYCLE_STATES))
    query = "SELECT id FROM canonical_investment_lifecycles"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC, id DESC"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    return [
        record
        for row in rows
        if (record := get_investment_lifecycle(connection, int(row_value(row, "id", 0)))) is not None
    ]


__all__ = [
    "COMMITTEE_ROLES",
    "FINAL_APPROVAL_DECISIONS",
    "LIFECYCLE_STATES",
    "PROPOSAL_ACTIONS",
    "VOTE_DECISIONS",
    "VOTE_ROUNDS",
    "VOTE_SCOPES",
    "append_position_review",
    "close_vote_round",
    "create_investment_proposal",
    "get_investment_lifecycle",
    "get_vote_round",
    "has_member_submitted_vote",
    "init_investment_lifecycle_tables",
    "list_investment_lifecycles",
    "list_lifecycle_audit_events",
    "list_position_reviews",
    "list_rule_checks",
    "list_wins_reconciliations",
    "lock_proposal_dossier",
    "open_pre_vote",
    "record_committee_discussion",
    "record_position_exit",
    "record_position_sizing",
    "record_rule_check",
    "record_wins_execution",
    "record_wins_reconciliation",
    "submit_committee_vote",
    "submit_final_approval",
    "update_committee_member_status",
    "verify_lifecycle_audit_chain",
    "withdraw_proposal",
]
