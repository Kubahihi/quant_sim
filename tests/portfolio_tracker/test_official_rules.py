from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from src.portfolio_tracker.official_rules import (
    acknowledge_rules_snapshot,
    capture_rules_snapshot,
    create_official_rules_watch,
    current_rules_watch_status,
    diff_rules_snapshots,
    latest_rules_snapshot,
    rules_acknowledgement_status,
    verify_rules_watch_integrity,
)


NOW = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)


def test_changed_official_rules_create_immutable_hash_chain_and_diff():
    watch = create_official_rules_watch(
        "wharton-2027",
        competition="Wharton Investment Competition 2026-2027",
        team_members=["Anna", "Boris"],
        created_by="Anna",
        rulesets=["trading_rules"],
        now=NOW,
    )
    first = capture_rules_snapshot(
        watch,
        "trading_rules",
        source_url="https://example.org/trading-rules",
        source_title="Official trading rules",
        content="Position cap: 10%\nCash minimum: 2%\n",
        captured_by="Anna",
        retrieved_at=NOW,
    )
    immutable_first = deepcopy(first["snapshots"]["trading_rules:v1"])
    second = capture_rules_snapshot(
        first,
        "trading_rules",
        source_url="https://example.org/trading-rules",
        content="Position cap: 8%\nCash minimum: 2%\n",
        captured_by="Boris",
        retrieved_at=NOW,
    )
    change = diff_rules_snapshots(second, "trading_rules:v1", "trading_rules:v2")

    assert second["snapshots"]["trading_rules:v1"] == immutable_first
    assert second["snapshots"]["trading_rules:v2"]["previous_snapshot_id"] == "trading_rules:v1"
    assert len(second["snapshots"]["trading_rules:v2"]["content_hash"]) == 64
    assert change["changed"] is True
    assert change["added_lines"] == ["Position cap: 8%"]
    assert change["removed_lines"] == ["Position cap: 10%"]
    assert latest_rules_snapshot(second, "trading_rules")["version"] == 2
    assert verify_rules_watch_integrity(second)["is_valid"] is True
    json.dumps(second, allow_nan=False)


def test_latest_snapshot_requires_acknowledgement_from_every_member():
    watch = create_official_rules_watch(
        "rules",
        competition="Competition",
        team_members=["Anna", "Boris"],
        created_by="Anna",
        rulesets=["deliverables"],
        now=NOW,
    )
    watch = capture_rules_snapshot(
        watch,
        "deliverables",
        source_url="https://example.org/deliverables",
        content="Mid-project report\nFinal report\n",
        captured_by="Anna",
        retrieved_at=NOW,
    )
    watch = acknowledge_rules_snapshot(
        watch, "deliverables:v1", member_id="Anna", note="Read and understood.", now=NOW
    )

    partial = rules_acknowledgement_status(watch, "deliverables:v1")
    watch = acknowledge_rules_snapshot(watch, "deliverables:v1", member_id="Boris", now=NOW)

    assert partial["missing_members"] == ["Boris"]
    assert current_rules_watch_status(watch)["is_current"] is True
    with pytest.raises(ValueError, match="already acknowledged"):
        acknowledge_rules_snapshot(watch, "deliverables:v1", member_id="Boris", now=NOW)


def test_duplicate_content_is_not_snapshotted_and_tampering_is_detected():
    watch = create_official_rules_watch(
        "rules",
        competition="Competition",
        team_members=["Anna"],
        created_by="Anna",
        rulesets=["case_study"],
        now=NOW,
    )
    watch = capture_rules_snapshot(
        watch,
        "case_study",
        source_url="https://example.org/case",
        content="Exact official text\n",
        captured_by="Anna",
        retrieved_at=NOW,
    )
    with pytest.raises(ValueError, match="unchanged"):
        capture_rules_snapshot(
            watch,
            "case_study",
            source_url="https://example.org/case",
            content="Exact official text\n",
            captured_by="Anna",
            retrieved_at=NOW,
        )

    tampered = deepcopy(watch)
    tampered["snapshots"]["case_study:v1"]["content"] = "Rewritten"
    assert verify_rules_watch_integrity(tampered)["issues"][0]["code"] == "hash_mismatch"
    with pytest.raises(ValueError, match="integrity"):
        acknowledge_rules_snapshot(tampered, "case_study:v1", member_id="Anna", now=NOW)

