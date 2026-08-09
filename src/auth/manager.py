"""
Authentication manager with high-level operations.

Provides user registration, login, logout, and session management
with bcrypt password hashing and input validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple

import bcrypt

from .database import (
    authenticate_user_once,
    init_auth_database,
    get_user_by_session_token,
    register_user_once,
    revoke_session,
    revoke_all_user_sessions,
)


# ---- Password hashing ----

def hash_password(password: str) -> str:
    """Hash a password using the required bcrypt dependency."""
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        raise ValueError("Password must be at most 72 UTF-8 bytes")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash, failing closed on bad input."""
    if not password_hash.startswith("$2"):
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (TypeError, ValueError):
        return False


# ---- Input validation ----

def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate a username.
    
    Requirements:
    - 3-30 characters
    - Alphanumeric and underscores only
    - Must start with a letter
    """
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(username) > 30:
        return False, "Username must be 30 characters or less"
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', username):
        return False, "Username must start with a letter and contain only letters, numbers, and underscores"
    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address.
    """
    if not email:
        return False, "Email is required"
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, "Invalid email address format"
    if len(email) > 254:
        return False, "Email address is too long"
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate a password.
    
    Requirements:
    - At least 8 characters
    - At least one letter
    - At least one number
    """
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password.encode("utf-8")) > 72:
        return False, "Password must be at most 72 UTF-8 bytes"
    if not re.search(r'[a-zA-Z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, ""


# ---- Authentication operations ----

def register_user(
    username: str,
    email: str,
    password: str,
    confirm_password: str = None,
) -> Tuple[Optional[dict[str, Any]], list[str]]:
    """
    Register a new user.
    
    Returns:
        Tuple of (user_dict, errors)
        user_dict is None if registration failed
    """
    errors = []
    
    # Clean input
    username = (username or "").strip()
    email = (email or "").strip().lower()
    
    # Validate username
    valid, msg = validate_username(username)
    if not valid:
        errors.append(msg)
    
    # Validate email
    valid, msg = validate_email(email)
    if not valid:
        errors.append(msg)
    
    # Validate password
    valid, msg = validate_password(password)
    if not valid:
        errors.append(msg)
    
    # Check password confirmation
    if confirm_password and password != confirm_password:
        errors.append("Passwords do not match")
    
    if errors:
        return None, errors

    try:
        # Initialize only after cheap input validation, and keep infrastructure
        # errors behind the same privacy-safe response as registration errors.
        init_auth_database()
        user, status = register_user_once(
            username,
            email,
            lambda: hash_password(password),
        )
        if status == "username_exists":
            return None, ["Username already exists"]
        if status == "email_exists":
            return None, ["Email already registered"]
        if user is None:
            return None, ["Registration failed"]
        return user, []
    except Exception:
        return None, ["Registration failed"]


def login_user(
    username: str,
    password: str,
    ip_address: Optional[str] = None,
) -> Tuple[Optional[str], Optional[dict[str, Any]], list[str]]:
    """
    Log in a user.
    
    Returns:
        Tuple of (session_token, user_dict, errors)
        session_token and user_dict are None if login failed
    """
    errors = []
    username = (username or "").strip()
    
    if not username:
        return None, None, ["Username is required"]
    
    if not password:
        return None, None, ["Password is required"]
    
    try:
        # Never return database or provider exception text to an authentication
        # client: connection errors can contain credential-bearing URLs.
        init_auth_database()
        token, user, status = authenticate_user_once(
            username,
            lambda password_hash: verify_password(password, password_hash),
            ip_address=ip_address,
        )
        if status == "rate_limited":
            return None, None, ["Too many failed attempts. Please try again in 10 minutes."]
        if status != "ok" or token is None or user is None:
            return None, None, ["Invalid username or password"]
        return token, user, []
    except Exception:
        return None, None, ["Login failed"]


def logout_user(session_token: str) -> None:
    """
    Log out a user by revoking their session.
    """
    if session_token:
        revoke_session(session_token)


def get_current_user(session_token: str) -> Optional[dict[str, Any]]:
    """
    Get the currently logged-in user from a session token.
    
    Returns None if not authenticated.
    """
    if not session_token:
        return None
    
    # Initialize database if needed. The database layer memoizes successful
    # schema initialization per database path.
    init_auth_database()

    return get_user_by_session_token(session_token)


def is_authenticated(session_token: str) -> bool:
    """
    Check if a session token represents a valid authenticated user.
    """
    return get_current_user(session_token) is not None


# ---- User data directory management ----

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_DATA_DIR = PROJECT_ROOT / "data" / "users"


def get_user_data_dir(user_id: int) -> Path:
    """
    Get the data directory for a specific user.
    
    Returns path like: data/users/{user_id}/
    """
    user_dir = USERS_DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def ensure_user_dirs(user_id: int) -> dict[str, Path]:
    """
    Ensure all user-specific data directories exist.
    
    Returns dict of directory names to paths:
    {
        "root": Path to user's root data dir,
        "portfolios": Path to user's portfolios dir,
        "swing_tracker": Path to user's swing tracker dir,
        "run_history": Path to user's run history dir,
    }
    """
    user_dir = get_user_data_dir(user_id)
    
    dirs = {
        "root": user_dir,
        "portfolios": user_dir / "portfolios",
        "swing_tracker": user_dir / "swing_tracker",
        "run_history": user_dir / "run_history",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_user_portfolio_dir(user_id: int) -> Path:
    """Get the portfolios directory for a user."""
    return ensure_user_dirs(user_id)["portfolios"]


def get_user_swing_tracker_dir(user_id: int) -> Path:
    """Get the swing tracker directory for a user."""
    return ensure_user_dirs(user_id)["swing_tracker"]


def get_user_run_history_dir(user_id: int) -> Path:
    """Get the run history directory for a user."""
    return ensure_user_dirs(user_id)["run_history"]
