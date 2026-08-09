"""Create or verify the deployable release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _working_tree_clean() -> bool:
    return not _git_output("status", "--porcelain", "--untracked-files=all")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a verified release manifest.")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--commit", help="Full immutable commit SHA; defaults to HEAD.")
    create.add_argument("--ref", help="Release branch or tag; defaults to the Git ref.")

    verify = subparsers.add_parser("verify", help="Verify a release manifest.")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-commit", help="Required full immutable commit SHA.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from src.operations.release_manifest import (
        ReleaseManifestError,
        create_release_manifest,
        load_release_manifest,
        verify_release_manifest,
        write_release_manifest,
    )

    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            commit = args.commit or _git_output("rev-parse", "HEAD")
            ref = args.ref or _git_output("rev-parse", "--abbrev-ref", "HEAD")
            manifest = create_release_manifest(
                PROJECT_ROOT,
                commit_sha=commit,
                ref=ref,
                working_tree_clean=_working_tree_clean(),
            )
            write_release_manifest(args.output, manifest)
            result = {
                "created": True,
                "commit_sha": manifest["commit_sha"],
                "file_count": len(manifest["files"]),
                "source_tree_sha256": manifest["source_tree_sha256"],
            }
        else:
            manifest = load_release_manifest(args.manifest)
            result = verify_release_manifest(
                PROJECT_ROOT,
                manifest,
                expected_commit=args.expected_commit,
            )
    except (OSError, subprocess.SubprocessError, ReleaseManifestError):
        result = {"valid": False, "reason": "release_manifest_failed"}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("created") or result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
