"""
Authentication manager with high-level operations.

Provides user registration, login, logout, and session management
with bcrypt password hashing and input validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple

# Only import bcrypt if available, provide fallback for testing
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    # Fallback for environments without bcrypt (use hashlib)
    import hashlib

from .database import (
    init_auth_database,
    create_user,
    get_user_by_username,
    get_user_by_id,
    get_user_by_session_token,
    create_session,
    revoke_session,
    revoke_all_user_sessions,
    user_exists,
)


# ---- Password hashing ----

def hash_password(password: str) -> str:
    """
    Hash a password securely using bcrypt.
    
    Falls back to SHA-256 if bcrypt is not available (not recommended for production).
    """
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    else:
        # Fallback: SHA-256 with salt (less secure, for development only)
        import secrets as _secrets
        salt = _secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"sha256${salt}${hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.
    
    Handles both bcrypt and fallback SHA-256 hashes.
    """
    if BCRYPT_AVAILABLE and password_hash.startswith('$2'):
        # bcrypt hash
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )
    elif password_hash.startswith('sha256$'):
        # Fallback SHA-256 hash
        parts = password_hash.split('$')
        if len(parts) != 3:
            return False
        salt = parts[1]
        expected_hash = parts[2]
        actual_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return actual_hash == expected_hash
    else:
        # Unknown hash format
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
    
    # Initialize database first (before any DB operations)
    init_auth_database()
    
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
    
    # Check if username exists
    if user_exists(username=username):
        errors.append("Username already exists")
    
    # Check if email exists
    if user_exists(email=email):
        errors.append("Email already registered")
    
    if errors:
        return None, errors
    
    # Hash password and create user
    password_hash = hash_password(password)
    try:
        user = create_user(username, email, password_hash)
        return user, []
    except Exception as e:
        return None, [f"Registration failed: {str(e)}"]


def login_user(
    username: str,
    password: str,
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
    
    # Initialize database if needed
    init_auth_database()
    
    # Get user
    user = get_user_by_username(username)
    if not user:
        return None, None, ["Invalid username or password"]
    
    # Verify password
    if not verify_password(password, user["password_hash"]):
        return None, None, ["Invalid username or password"]
    
    # Create session
    try:
        token = create_session(user["id"])
        # Remove password_hash from returned user dict
        user_safe = {k: v for k, v in user.items() if k != "password_hash"}
        return token, user_safe, []
    except Exception as e:
        return None, None, [f"Login failed: {str(e)}"]


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
    
    # Initialize database if needed
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