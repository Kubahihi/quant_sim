"""
Authentication module for multi-user support.

Provides user registration, login, logout, and session management
with SQLite-backed persistent storage.
"""

from .database import (
    init_auth_database,
    create_user,
    get_user_by_username,
    get_user_by_id,
    validate_session_token,
    create_session,
    revoke_session,
    cleanup_expired_sessions,
    get_user_by_session_token,
)
from .manager import (
    register_user,
    login_user,
    logout_user,
    get_current_user,
    is_authenticated,
    get_user_data_dir,
    ensure_user_dirs,
)
from .migrations import migrate_existing_data

__all__ = [
    # Database functions
    "init_auth_database",
    "create_user",
    "get_user_by_username",
    "get_user_by_id",
    "validate_session_token",
    "create_session",
    "revoke_session",
    "cleanup_expired_sessions",
    "get_user_by_session_token",
    # Manager functions
    "register_user",
    "login_user",
    "logout_user",
    "get_current_user",
    "is_authenticated",
    "get_user_data_dir",
    "ensure_user_dirs",
    # Migrations
    "migrate_existing_data",
]