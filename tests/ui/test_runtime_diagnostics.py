from __future__ import annotations

from pathlib import Path

import pytest

from ui.runtime_diagnostics import (
    PerformanceTrace,
    append_trace_history,
    resolve_build_identity,
    summarize_trace_history,
)


def test_build_identity_reads_branch_and_commit_without_spawning_git(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs" / "heads" / "Matej"
    ref.parent.mkdir(parents=True)
    ref.write_text("1234567890abcdef\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/Matej\n", encoding="utf-8")

    identity = resolve_build_identity(tmp_path, environment={})

    assert identity.branch == "Matej"
    assert identity.commit == "1234567890ab"
    assert identity.label == "Matej@1234567890ab"


def test_environment_build_identity_has_priority(tmp_path: Path) -> None:
    identity = resolve_build_identity(
        tmp_path,
        environment={
            "STREAMLIT_GIT_COMMIT": "fedcba9876543210",
            "STREAMLIT_GIT_BRANCH": "Matej",
        },
    )

    assert identity.commit == "fedcba987654"
    assert identity.branch == "Matej"


def test_trace_history_is_bounded_and_reports_median_and_p95() -> None:
    history: list[dict[str, object]] = []
    for total in (10.0, 20.0, 30.0, 40.0):
        append_trace_history(history, {"total_ms": total}, limit=3)

    summary = summarize_trace_history(history)

    assert [item["total_ms"] for item in history] == [20.0, 30.0, 40.0]
    assert summary["count"] == 3
    assert summary["median_ms"] == pytest.approx(30.0)
    assert summary["p95_ms"] == pytest.approx(40.0)


def test_performance_trace_captures_named_phases(monkeypatch) -> None:
    values = iter([10.1, 10.25])
    monkeypatch.setattr("ui.runtime_diagnostics.time.perf_counter", lambda: next(values))
    trace = PerformanceTrace(started_at=10.0)

    assert trace.mark("bootstrap") == pytest.approx(100.0)
    snapshot = trace.snapshot(route="Quant Platform", stage="ready")

    assert snapshot["total_ms"] == pytest.approx(250.0)
    assert snapshot["phases_ms"] == {"bootstrap": pytest.approx(100.0)}
