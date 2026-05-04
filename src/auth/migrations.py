"""
Data migration utilities for multi-user support.

Safely migrates existing single-user data to user-specific directories,
assigning all existing data to a default user (id=1).

Migration is idempotent - safe to run multiple times.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import (
    init_auth_database,
    user_exists,
    get_user_by_username,
    create_user,
)
from .manager import (
    hash_password,
    ensure_user_dirs,
    get_user_data_dir,
    USERS_DATA_DIR,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Migration marker file
MIGRATION_MARKER = DATA_DIR / ".migration_completed"

# Default user credentials
DEFAULT_USERNAME = "admin"
DEFAULT_EMAIL = "admin@localhost"
DEFAULT_PASSWORD = "admin123"  # Should be changed after first login


def _get_migration_info() -> Dict[str, Any]:
    """Get migration info from marker file if it exists."""
    if not MIGRATION_MARKER.exists():
        return {"completed": False}
    
    try:
        info = json.loads(MIGRATION_MARKER.read_text(encoding="utf-8"))
        return info
    except (json.JSONDecodeError, Exception):
        return {"completed": False}


def _mark_migration_complete(files_migrated: int, user_id: int) -> None:
    """Mark migration as complete with details."""
    info = {
        "completed": True,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "files_migrated": files_migrated,
        "default_user_id": user_id,
        "version": "1.0",
    }
    MIGRATION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    MIGRATION_MARKER.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _migrate_portfolio_files(user_id: int, dry_run: bool = False) -> int:
    """
    Migrate portfolio JSON files to user-specific directory.
    
    Returns number of files migrated.
    """
    old_portfolio_dir = DATA_DIR / "portfolios"
    new_portfolio_dir = ensure_user_dirs(user_id)["portfolios"]
    
    if not old_portfolio_dir.exists():
        return 0
    
    migrated = 0
    for portfolio_file in old_portfolio_dir.glob("*.json"):
        target_file = new_portfolio_dir / portfolio_file.name
        
        # Skip if already migrated (idempotent)
        if target_file.exists():
            continue
        
        if not dry_run:
            shutil.copy2(portfolio_file, target_file)
        migrated += 1
    
    return migrated


def _migrate_swing_tracker_files(user_id: int, dry_run: bool = False) -> int:
    """
    Migrate swing tracker JSON files to user-specific directory.
    
    Returns number of files migrated.
    """
    old_swing_dir = DATA_DIR / "swing_tracker"
    new_swing_dir = ensure_user_dirs(user_id)["swing_tracker"]
    
    if not old_swing_dir.exists():
        return 0
    
    migrated = 0
    for swing_file in old_swing_dir.glob("*.json"):
        target_file = new_swing_dir / swing_file.name
        
        # Skip if already migrated (idempotent)
        if target_file.exists():
            continue
        
        if not dry_run:
            shutil.copy2(swing_file, target_file)
        migrated += 1
    
    return migrated


def _migrate_run_history_files(user_id: int, dry_run: bool = False) -> int:
    """
    Migrate run history JSON files to user-specific directory.
    
    Returns number of files migrated.
    """
    old_history_dir = DATA_DIR / "run_history"
    new_history_dir = ensure_user_dirs(user_id)["run_history"]
    
    if not old_history_dir.exists():
        return 0
    
    migrated = 0
    for history_file in old_history_dir.glob("*.json"):
        target_file = new_history_dir / history_file.name
        
        # Skip if already migrated (idempotent)
        if target_file.exists():
            continue
        
        if not dry_run:
            shutil.copy2(history_file, target_file)
        migrated += 1
    
    return migrated


def create_default_user() -> Optional[Dict[str, Any]]:
    """
    Create the default admin user if it doesn't exist.
    
    Returns user dict if created or already exists.
    """
    init_auth_database()
    
    # Check if default user already exists
    user = get_user_by_username(DEFAULT_USERNAME)
    if user:
        return user
    
    # Check if username is taken by another user
    if user_exists(username=DEFAULT_USERNAME):
        return None
    
    # Create default user
    password_hash = hash_password(DEFAULT_PASSWORD)
    try:
        user = create_user(DEFAULT_USERNAME, DEFAULT_EMAIL, password_hash)
        return user
    except Exception:
        return None


def migrate_existing_data(dry_run: bool = False) -> Dict[str, Any]:
    """
    Main migration function.
    
    Migrates all existing single-user data to the default user's directory.
    This function is idempotent - safe to run multiple times.
    
    Args:
        dry_run: If True, don't actually move files, just report what would be done.
    
    Returns:
        Dict with migration results
    """
    # Check if already migrated
    migration_info = _get_migration_info()
    if migration_info.get("completed"):
        return {
            "success": True,
            "already_migrated": True,
            "migrated_at": migration_info.get("migrated_at"),
            "message": "Migration already completed",
        }
    
    # Create default user
    default_user = create_default_user()
    if not default_user:
        return {
            "success": False,
            "error": "Failed to create default user",
        }
    
    user_id = default_user["id"]
    
    # Migrate files
    portfolios_migrated = _migrate_portfolio_files(user_id, dry_run)
    swing_tracker_migrated = _migrate_swing_tracker_files(user_id, dry_run)
    run_history_migrated = _migrate_run_history_files(user_id, dry_run)
    
    total_migrated = portfolios_migrated + swing_tracker_migrated + run_history_migrated
    
    # Mark migration complete (only if not dry run)
    if not dry_run and total_migrated > 0:
        _mark_migration_complete(total_migrated, user_id)
    elif not dry_run:
        # Even if no files to migrate, mark as complete
        _mark_migration_complete(0, user_id)
    
    return {
        "success": True,
        "already_migrated": False,
        "default_user_id": user_id,
        "default_username": DEFAULT_USERNAME,
        "default_password": DEFAULT_PASSWORD if not dry_run else "***",
        "files_migrated": {
            "portfolios": portfolios_migrated,
            "swing_tracker": swing_tracker_migrated,
            "run_history": run_history_migrated,
            "total": total_migrated,
        },
        "dry_run": dry_run,
        "message": (
            f"Migration {'would be ' if dry_run else ''}completed. "
            f"{total_migrated} files {'would be ' if dry_run else ''}migrated to user '{DEFAULT_USERNAME}'."
        ),
    }


def get_migration_status() -> Dict[str, Any]:
    """Get current migration status."""
    return _get_migration_info()


def rollback_migration() -> bool:
    """
    Remove migration marker to allow re-migration.
    
    WARNING: This does NOT remove the migrated files or the default user.
    It only removes the marker, allowing migration to be run again.
    Use with caution.
    """
    if MIGRATION_MARKER.exists():
        MIGRATION_MARKER.unlink()
        return True
    return False


def get_data_layout() -> Dict[str, str]:
    """
    Get the current data directory layout description.
    
    Returns a dict explaining where different types of data are stored.
    """
    return {
        "auth_database": str(DATA_DIR / "auth.db"),
        "shared_cache": str(DATA_DIR / "cache" / "market_data.db"),
        "user_data_root": str(USERS_DATA_DIR),
        "user_data_pattern": str(USERS_DATA_DIR / "{user_id}" / ""),
        "example_user_dirs": {
            "portfolios": str(USERS_DATA_DIR / "1" / "portfolios"),
            "swing_tracker": str(USERS_DATA_DIR / "1" / "swing_tracker"),
            "run_history": str(USERS_DATA_DIR / "1" / "run_history"),
        },
        "legacy_dirs": {
            "portfolios": str(DATA_DIR / "portfolios"),
            "swing_tracker": str(DATA_DIR / "swing_tracker"),
            "run_history": str(DATA_DIR / "run_history"),
        },
    }