"""Evidence-linked question bank and deterministic Q&A rehearsal rounds.

The model is intentionally persistence-neutral and copy-on-write.  Question
snapshots are copied into each round, so later edits cannot rewrite what a
member was actually asked or the time limit against which they were scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import json
import math
import random
from typing import Any


SCORE_DIMENSIONS = ("clarity", "evidence", "client_fit")
ROUND_STATUSES = frozenset({"active", "completed"})


def _text(value: Any, name: str, *, required: bool = False, limit: int = 20_000) -> str:
    result = " ".join(str(value or "").strip().split())
    if required and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(result) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return result


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


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def _json_copy(value: Any, name: str = "Value") -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serialisable.") from exc
    return json.loads(encoded)


def _member(value: Any, workspace: Mapping[str, Any], name: str) -> str:
    result = _text(value, name, required=True, limit=200)
    if result not in workspace.get("team_members", []):
        raise ValueError(f"{name} must be a registered team member.")
    return result


def _round(workspace: Mapping[str, Any], round_id: str) -> dict[str, Any]:
    identifier = _identifier(round_id, "Round id")
    try:
        return workspace["rounds"][identifier]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown round id: {identifier}.") from exc


def create_qa_rehearsal_workspace(
    workspace_id: str,
    *,
    team_members: Sequence[str],
    created_by: str,
    passing_score: float = 3.5,
    minimum_dimension_score: float = 3.0,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Create a question bank with explicit scoring policy."""
    if isinstance(team_members, (str, bytes, bytearray)):
        raise ValueError("Team members must be a sequence.")
    members: list[str] = []
    for value in team_members:
        member = _text(value, "Team member", required=True, limit=200)
        if member in members:
            raise ValueError(f"Duplicate team member: {member}.")
        members.append(member)
    if len(members) < 2:
        raise ValueError("At least two team members are required for primary and backup roles.")
    creator = _text(created_by, "Created by", required=True, limit=200)
    if creator not in members:
        raise ValueError("Created by must be a registered team member.")
    threshold = _number(passing_score, "Passing score", minimum=1.0, maximum=5.0)
    dimension_threshold = _number(
        minimum_dimension_score, "Minimum dimension score", minimum=1.0, maximum=5.0
    )
    return {
        "workspace_id": _identifier(workspace_id, "Workspace id"),
        "team_members": members,
        "created_by": creator,
        "created_at": _timestamp(now),
        "scoring_policy": {
            "dimensions": list(SCORE_DIMENSIONS),
            "passing_score": threshold,
            "minimum_dimension_score": dimension_threshold,
            "must_finish_within_time": True,
        },
        "questions": {},
        "rounds": {},
        "round_order": [],
    }


def add_qa_question(
    workspace: Mapping[str, Any],
    question_id: str,
    *,
    prompt: str,
    model_answer: str,
    evidence_ids: Sequence[str],
    primary_responder: str,
    backup_responder: str,
    time_limit_seconds: int,
    category: str = "general",
    follow_ups: Sequence[str] = (),
    killer_question: bool = False,
    created_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Add a rehearsal-ready question; incomplete placeholders are rejected."""
    result = _json_copy(workspace, "Workspace")
    identifier = _identifier(question_id, "Question id")
    if identifier in result.get("questions", {}):
        raise ValueError(f"Question id already exists: {identifier}.")
    primary = _member(primary_responder, result, "Primary responder")
    backup = _member(backup_responder, result, "Backup responder")
    if primary == backup:
        raise ValueError("Primary and backup responders must be different people.")
    seconds_float = _number(
        time_limit_seconds, "Time limit", minimum=10, maximum=600
    )
    if seconds_float != int(seconds_float):
        raise ValueError("Time limit must be a whole number of seconds.")
    if isinstance(evidence_ids, (str, bytes, bytearray)):
        raise ValueError("Evidence ids must be a sequence.")
    links = list(dict.fromkeys(_identifier(value, "Evidence id") for value in evidence_ids))
    if not links:
        raise ValueError("A model answer must link to at least one evidence record.")
    if isinstance(follow_ups, (str, bytes, bytearray)):
        raise ValueError("Follow-up questions must be a sequence.")
    follow_up_values: list[str] = []
    for value in follow_ups:
        follow_up = _text(value, "Follow-up question", required=True, limit=4_000)
        if follow_up not in follow_up_values:
            follow_up_values.append(follow_up)
    creator = _member(created_by, result, "Created by")
    result["questions"][identifier] = {
        "question_id": identifier,
        "prompt": _text(prompt, "Question", required=True, limit=10_000),
        "category": _text(category, "Category", required=True, limit=100).lower(),
        "model_answer": _text(model_answer, "Model answer", required=True, limit=30_000),
        "evidence_ids": links,
        "primary_responder": primary,
        "backup_responder": backup,
        "time_limit_seconds": int(seconds_float),
        "follow_ups": follow_up_values,
        "killer_question": bool(killer_question),
        "created_by": creator,
        "created_at": _timestamp(now),
        "status": "ready",
    }
    return result


def retire_qa_question(
    workspace: Mapping[str, Any],
    question_id: str,
    *,
    retired_by: str,
    reason: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(workspace, "Workspace")
    identifier = _identifier(question_id, "Question id")
    try:
        question = result["questions"][identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown question id: {identifier}.") from exc
    if question.get("status") == "retired":
        raise ValueError("Question is already retired.")
    question["status"] = "retired"
    question["retired_by"] = _member(retired_by, result, "Retired by")
    question["retired_at"] = _timestamp(now)
    question["retirement_reason"] = _text(
        reason, "Retirement reason", required=True, limit=4_000
    )
    return result


def create_mock_round(
    workspace: Mapping[str, Any],
    round_id: str,
    *,
    started_by: str,
    participant_ids: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
    question_count: int | None = None,
    random_seed: int | str = 0,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Create a reproducibly random round and freeze each question snapshot."""
    result = _json_copy(workspace, "Workspace")
    identifier = _identifier(round_id, "Round id")
    if identifier in result.get("rounds", {}):
        raise ValueError(f"Round id already exists: {identifier}.")
    starter = _member(started_by, result, "Started by")
    if isinstance(participant_ids, (str, bytes, bytearray)):
        raise ValueError("Participant ids must be a sequence.")
    raw_participants = participant_ids or result["team_members"]
    participants: list[str] = []
    for value in raw_participants:
        member = _member(value, result, "Participant")
        if member not in participants:
            participants.append(member)
    if not participants:
        raise ValueError("At least one participant is required.")

    ready_ids = [
        key for key, question in result.get("questions", {}).items()
        if question.get("status") == "ready"
    ]
    if question_ids is None:
        selected = list(ready_ids)
    else:
        if isinstance(question_ids, (str, bytes, bytearray)):
            raise ValueError("Question ids must be a sequence.")
        selected = list(dict.fromkeys(_identifier(value, "Question id") for value in question_ids))
        unknown = [value for value in selected if value not in ready_ids]
        if unknown:
            raise ValueError(f"Unknown or inactive question ids: {', '.join(unknown)}.")
    if not selected:
        raise ValueError("No ready questions are available for the mock round.")
    if question_count is not None:
        count_float = _number(question_count, "Question count", minimum=1)
        if count_float != int(count_float):
            raise ValueError("Question count must be a positive integer.")
        count = int(count_float)
        if count > len(selected):
            raise ValueError("Question count exceeds the available questions.")
    else:
        count = len(selected)
    rng = random.Random(str(random_seed))
    rng.shuffle(selected)
    selected = selected[:count]

    slots: list[dict[str, Any]] = []
    for position, question_id in enumerate(selected, start=1):
        question = _json_copy(result["questions"][question_id])
        eligible = [
            member
            for member in (question["primary_responder"], question["backup_responder"])
            if member in participants
        ]
        if not eligible:
            raise ValueError(
                f"Question {question_id} has no designated responder in this round."
            )
        slots.append(
            {
                "position": position,
                "question_id": question_id,
                "assigned_responder": eligible[0],
                "eligible_responders": eligible,
                "question_snapshot": question,
                "response": None,
            }
        )
    timestamp = _timestamp(now)
    result["rounds"][identifier] = {
        "round_id": identifier,
        "status": "active",
        "started_by": starter,
        "started_at": timestamp,
        "completed_by": None,
        "completed_at": None,
        "participants": participants,
        "random_seed": str(random_seed),
        "slots": slots,
    }
    result["round_order"].append(identifier)
    return result


def record_qa_response(
    workspace: Mapping[str, Any],
    round_id: str,
    question_id: str,
    *,
    responder: str,
    answer: str,
    duration_seconds: float,
    scores: Mapping[str, float],
    evaluator: str,
    follow_up_answers: Mapping[str, str] | None = None,
    notes: str = "",
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(workspace, "Workspace")
    rehearsal_round = _round(result, round_id)
    if rehearsal_round.get("status") != "active":
        raise ValueError("Responses can only be recorded in an active round.")
    identifier = _identifier(question_id, "Question id")
    slot = next(
        (item for item in rehearsal_round["slots"] if item["question_id"] == identifier),
        None,
    )
    if slot is None:
        raise ValueError(f"Question {identifier} is not in this round.")
    if slot.get("response") is not None:
        raise ValueError("This question already has a response in the round.")
    actor = _member(responder, result, "Responder")
    if actor not in slot["eligible_responders"]:
        raise ValueError("Responder is not the designated primary or backup for this question.")
    reviewer = _member(evaluator, result, "Evaluator")
    if reviewer == actor:
        raise ValueError("A responder cannot evaluate their own answer.")
    if not isinstance(scores, Mapping):
        raise ValueError("Scores must be a mapping.")
    if set(scores) != set(SCORE_DIMENSIONS):
        raise ValueError("Scores must contain clarity, evidence, and client_fit exactly.")
    normalised_scores = {
        dimension: _number(scores[dimension], f"{dimension} score", minimum=1.0, maximum=5.0)
        for dimension in SCORE_DIMENSIONS
    }
    duration = _number(duration_seconds, "Duration", minimum=0.0)
    allowed_follow_ups = slot["question_snapshot"].get("follow_ups", [])
    follow_up_values: list[dict[str, str]] = []
    for prompt, response in (follow_up_answers or {}).items():
        clean_prompt = _text(prompt, "Follow-up question", required=True, limit=4_000)
        if clean_prompt not in allowed_follow_ups:
            raise ValueError("Follow-up answer does not match the frozen question snapshot.")
        follow_up_values.append(
            {
                "prompt": clean_prompt,
                "answer": _text(response, "Follow-up answer", required=True, limit=20_000),
            }
        )
    overall = sum(normalised_scores.values()) / len(normalised_scores)
    policy = result["scoring_policy"]
    within_time = duration <= slot["question_snapshot"]["time_limit_seconds"]
    passed = (
        overall >= policy["passing_score"]
        and min(normalised_scores.values()) >= policy["minimum_dimension_score"]
        and (within_time or not policy["must_finish_within_time"])
    )
    slot["response"] = {
        "responder": actor,
        "answer": _text(answer, "Answer", required=True, limit=30_000),
        "duration_seconds": duration,
        "within_time": within_time,
        "scores": normalised_scores,
        "overall_score": overall,
        "passed": passed,
        "evaluator": reviewer,
        "follow_up_answers": follow_up_values,
        "notes": _text(notes, "Notes", limit=10_000),
        "recorded_at": _timestamp(now),
    }
    return result


def complete_mock_round(
    workspace: Mapping[str, Any],
    round_id: str,
    *,
    completed_by: str,
    now: date | datetime | str | None = None,
) -> dict[str, Any]:
    result = _json_copy(workspace, "Workspace")
    rehearsal_round = _round(result, round_id)
    if rehearsal_round.get("status") != "active":
        raise ValueError("Round is not active.")
    missing = [
        item["question_id"] for item in rehearsal_round["slots"] if item.get("response") is None
    ]
    if missing:
        raise ValueError("Cannot complete a round with unanswered questions: " + ", ".join(missing))
    responses = [item["response"] for item in rehearsal_round["slots"]]
    rehearsal_round["status"] = "completed"
    rehearsal_round["completed_by"] = _member(completed_by, result, "Completed by")
    rehearsal_round["completed_at"] = _timestamp(now)
    rehearsal_round["summary"] = {
        "question_count": len(responses),
        "passed_count": sum(bool(item["passed"]) for item in responses),
        "pass_rate_pct": 100.0 * sum(bool(item["passed"]) for item in responses) / len(responses),
        "average_score": sum(float(item["overall_score"]) for item in responses) / len(responses),
        "within_time_pct": 100.0 * sum(bool(item["within_time"]) for item in responses) / len(responses),
    }
    return result


def member_qa_history(workspace: Mapping[str, Any], member_id: str) -> dict[str, Any]:
    value = _json_copy(workspace, "Workspace")
    member = _member(member_id, value, "Member")
    attempts: list[dict[str, Any]] = []
    for round_id in value.get("round_order", []):
        rehearsal_round = value["rounds"][round_id]
        for slot in rehearsal_round["slots"]:
            response = slot.get("response")
            if response and response["responder"] == member:
                attempts.append(
                    {
                        "round_id": round_id,
                        "question_id": slot["question_id"],
                        "category": slot["question_snapshot"]["category"],
                        **response,
                    }
                )
    count = len(attempts)
    averages = {
        dimension: (
            sum(item["scores"][dimension] for item in attempts) / count if count else None
        )
        for dimension in SCORE_DIMENSIONS
    }
    return {
        "member_id": member,
        "attempt_count": count,
        "passed_count": sum(bool(item["passed"]) for item in attempts),
        "pass_rate_pct": 100.0 * sum(bool(item["passed"]) for item in attempts) / count if count else None,
        "within_time_pct": 100.0 * sum(bool(item["within_time"]) for item in attempts) / count if count else None,
        "average_scores": averages,
        "attempts": attempts,
    }


def killer_question_status(workspace: Mapping[str, Any]) -> dict[str, Any]:
    """List killer questions that have or have not yet received a passing answer."""
    value = _json_copy(workspace, "Workspace")
    passed_question_ids: set[str] = set()
    attempted_question_ids: set[str] = set()
    for rehearsal_round in value.get("rounds", {}).values():
        for slot in rehearsal_round["slots"]:
            response = slot.get("response")
            if response:
                attempted_question_ids.add(slot["question_id"])
                if response["passed"]:
                    passed_question_ids.add(slot["question_id"])
    records: list[dict[str, Any]] = []
    for question_id in sorted(value.get("questions", {})):
        question = value["questions"][question_id]
        if not question.get("killer_question") or question.get("status") != "ready":
            continue
        records.append(
            {
                "question_id": question_id,
                "prompt": question["prompt"],
                "primary_responder": question["primary_responder"],
                "attempted": question_id in attempted_question_ids,
                "resolved": question_id in passed_question_ids,
            }
        )
    unresolved = [item for item in records if not item["resolved"]]
    return {
        "killer_question_count": len(records),
        "resolved_count": len(records) - len(unresolved),
        "unresolved_count": len(unresolved),
        "questions": records,
        "unresolved": unresolved,
    }


__all__ = [
    "ROUND_STATUSES",
    "SCORE_DIMENSIONS",
    "add_qa_question",
    "complete_mock_round",
    "create_mock_round",
    "create_qa_rehearsal_workspace",
    "killer_question_status",
    "member_qa_history",
    "record_qa_response",
    "retire_qa_question",
]
