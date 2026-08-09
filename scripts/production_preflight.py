"""Run privacy-safe production and backup-restore preflight checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    from src.operations.preflight import run_production_preflight, run_restore_drill

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate configuration without remote database/storage probes.",
    )
    parser.add_argument(
        "--restore-drill",
        type=Path,
        metavar="BACKUP_DB",
        help="Also verify an exported SQLite backup in an isolated read-only copy.",
    )
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Run only the isolated restore drill (requires --restore-drill).",
    )
    args = parser.parse_args(argv)
    if args.restore_only and args.restore_drill is None:
        parser.error("--restore-only requires --restore-drill BACKUP_DB")

    if args.restore_only:
        restore_result = run_restore_drill(args.restore_drill)
        result = {
            "ready": restore_result["status"] == "healthy",
            "checks": {"restore_drill": restore_result},
        }
    else:
        result = run_production_preflight(
            live=not args.config_only,
            restore_backup=args.restore_drill,
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
