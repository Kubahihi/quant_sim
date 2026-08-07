from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Mapping, MutableSequence, Optional


_COMMIT_ENV_KEYS = (
    "STREAMLIT_GIT_COMMIT",
    "GIT_COMMIT",
    "COMMIT_SHA",
    "RENDER_GIT_COMMIT",
)
_BRANCH_ENV_KEYS = (
    "STREAMLIT_GIT_BRANCH",
    "GIT_BRANCH",
    "BRANCH_NAME",
)


@dataclass(frozen=True)
class BuildIdentity:
    commit: str
    branch: str

    @property
    def label(self) -> str:
        if self.branch and self.branch != "unknown":
            return f"{self.branch}@{self.commit}"
        return self.commit


@dataclass
class PerformanceTrace:
    """Small, dependency-free server-side timing trace for one Streamlit rerun."""

    started_at: float = field(default_factory=time.perf_counter)
    last_mark_at: float = field(init=False)
    phases_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.last_mark_at = self.started_at

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        duration_ms = (now - self.last_mark_at) * 1_000.0
        self.phases_ms[str(name)] = duration_ms
        self.last_mark_at = now
        return duration_ms

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1_000.0

    def snapshot(self, *, route: str, stage: str) -> dict[str, object]:
        return {
            "route": str(route),
            "stage": str(stage),
            "total_ms": float(self.elapsed_ms),
            "phases_ms": {
                name: float(duration)
                for name, duration in self.phases_ms.items()
            },
        }


def _first_environment_value(
    keys: tuple[str, ...],
    environment: Mapping[str, str],
) -> str:
    for key in keys:
        value = str(environment.get(key) or "").strip()
        if value:
            return value
    return ""


def _read_git_identity(project_root: Path) -> tuple[str, str]:
    git_dir = project_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return "", ""
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "", ""
    if not head.startswith("ref: "):
        return head, "detached"

    reference = head.removeprefix("ref: ").strip()
    branch = reference.rsplit("/", 1)[-1]
    loose_ref = git_dir / reference
    try:
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip(), branch
    except OSError:
        pass

    packed_refs = git_dir / "packed-refs"
    try:
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, packed_reference = line.split(" ", 1)
            if packed_reference.strip() == reference:
                return commit.strip(), branch
    except (OSError, ValueError):
        pass
    return "", branch


def resolve_build_identity(
    project_root: str | Path,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> BuildIdentity:
    env = os.environ if environment is None else environment
    commit = _first_environment_value(_COMMIT_ENV_KEYS, env)
    branch = _first_environment_value(_BRANCH_ENV_KEYS, env)
    git_commit, git_branch = _read_git_identity(Path(project_root))
    commit = commit or git_commit or "unknown"
    branch = branch or git_branch or "unknown"
    return BuildIdentity(commit=commit[:12], branch=branch)


def append_trace_history(
    history: MutableSequence[dict[str, object]],
    snapshot: dict[str, object],
    *,
    limit: int = 20,
) -> None:
    if limit < 1:
        raise ValueError("limit must be positive.")
    history.append(dict(snapshot))
    del history[:-limit]


def summarize_trace_history(
    history: MutableSequence[dict[str, object]],
) -> dict[str, float]:
    totals = sorted(
        float(item["total_ms"])
        for item in history
        if item.get("total_ms") is not None
    )
    if not totals:
        return {"count": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    midpoint = len(totals) // 2
    median = (
        totals[midpoint]
        if len(totals) % 2
        else (totals[midpoint - 1] + totals[midpoint]) / 2.0
    )
    p95_index = min(len(totals) - 1, max(0, int(0.95 * len(totals) + 0.999999) - 1))
    return {
        "count": float(len(totals)),
        "median_ms": float(median),
        "p95_ms": float(totals[p95_index]),
    }
