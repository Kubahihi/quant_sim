"""
SQLite database layer for authentication.

Provides user and session management with proper schema,
connection handling, and data access functions.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Database path - stored in project data directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTH_DB_PATH = PROJECT_ROOT / "data" / "auth.db"

# Allow overriding via environment variable (for testing)
if os.environ.get("AUTH_TEST_DB_PATH"):
    AUTH_DB_PATH = Path(os.environ["AUTH_TEST_DB_PATH"])

# Session expiry time (24 hours)
SESSION_EXPIRY_HOURS = 24


def _row_to_dict(cursor, row) -> dict[str, Any] | None:
    """Safely convert a DB row to a dict, handling both sqlite3.Row and plain tuples."""
    if row is None:
        return None
    if hasattr(row, 'keys'):
        return dict(row)
    cols = [col[0] for col in cursor.description]
    return dict(zip(cols, row))


def _get_db_path() -> Path:
    """Get the database path, creating parent directories if needed."""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return AUTH_DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with proper settings. Uses Turso if configured."""
    turso_url = None
    turso_token = None
    try:
        import streamlit as st
        turso_url = st.secrets.get("TURSO_DATABASE_URL")
        turso_token = st.secrets.get("TURSO_AUTH_TOKEN")
    except Exception:
        pass
    
    if not turso_url:
        turso_url = os.environ.get("TURSO_DATABASE_URL")
    if not turso_token:
        turso_token = os.environ.get("TURSO_AUTH_TOKEN")

    db_path = _get_db_path()
    
    if turso_url and turso_token:
        try:
            import libsql_experimental as libsql
        except ImportError:
            try:
                import libsql
            except ImportError:
                libsql = sqlite3
                turso_url = None
        
        if turso_url:
            conn = libsql.connect(str(db_path), sync_url=turso_url, auth_token=turso_token)
            conn.sync()
            try:
                conn.row_factory = sqlite3.Row
            except AttributeError:
                pass
            return conn

    # Local SQLite fallback
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_auth_database() -> None:
    """Initialize the authentication database schema."""
    conn = _get_connection()
    try:
        conn.executescript("""
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            );

            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            -- Brute-force protection table
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,
                ip_address TEXT
            );

            -- Index for faster session lookups
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
            ON sessions(user_id);
            
            -- Index for session cleanup
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
            ON sessions(expires_at);

            -- Index for brute-force tracking
            CREATE INDEX IF NOT EXISTS idx_login_attempts_username_time 
            ON login_attempts(username, timestamp);

            -- Generic User Data table (for portfolios, swing_tracker, run_history)
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                content_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE (user_id, data_type, file_name)
            );
        """)
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
    finally:
        conn.close()


def create_user(
    username: str,
    email: str,
    password_hash: str,
) -> dict[str, Any]:
    """
    Create a new user in the database.
    
    Returns:
        dict with user info if successful
        
    Raises:
        sqlite3.IntegrityError if username or email already exists
    """
    conn = _get_connection()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, password_hash, created_at),
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
        
        user_id = cursor.lastrowid
        return {
            "id": user_id,
            "username": username,
            "email": email,
            "created_at": created_at,
        }
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    """Get a user by username."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, username, email, password_hash, created_at, is_active "
            "FROM users WHERE username = ? AND is_active = 1",
            (username,),
        )
        row = cursor.fetchone()
        if row:
            return _row_to_dict(cursor, row)
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[dict[str, Any]]:
    """Get a user by ID."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, username, email, created_at, is_active "
            "FROM users WHERE id = ? AND is_active = 1",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return _row_to_dict(cursor, row)
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    """Get a user by email."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, username, email, password_hash, created_at, is_active "
            "FROM users WHERE email = ? AND is_active = 1",
            (email,),
        )
        row = cursor.fetchone()
        if row:
            return _row_to_dict(cursor, row)
        return None
    finally:
        conn.close()


def create_session(user_id: int) -> str:
    """
    Create a new session for a user.
    
    Returns:
        Session token string
    """
    conn = _get_connection()
    try:
        # Generate secure random token
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        created_at = now.isoformat()
        expires_at = (now + timedelta(hours=SESSION_EXPIRY_HOURS)).isoformat()
        
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at, last_accessed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, user_id, created_at, expires_at, created_at),
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
        return token
    finally:
        conn.close()


def validate_session_token(token: str) -> bool:
    """
    Validate a session token.
    
    Updates last_accessed time if valid.
    
    Returns:
        True if token is valid and not expired
    """
    conn = _get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        # Check if session exists and is not expired
        cursor = conn.execute(
            """
            SELECT s.token, s.user_id, s.expires_at, u.is_active
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ? AND u.is_active = 1
            """,
            (token, now),
        )
        row = cursor.fetchone()
        
        if row:
            # Update last accessed time
            conn.execute(
                "UPDATE sessions SET last_accessed = ? WHERE token = ?",
                (now, token),
            )
            conn.commit()
            if hasattr(conn, 'sync'):
                conn.sync()
            return True
        return False
    finally:
        conn.close()


def get_user_by_session_token(token: str) -> Optional[dict[str, Any]]:
    """
    Get the user associated with a session token.
    
    Returns None if token is invalid or expired.
    """
    conn = _get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        cursor = conn.execute(
            """
            SELECT u.id, u.username, u.email, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ? AND u.is_active = 1
            """,
            (token, now),
        )
        row = cursor.fetchone()
        
        if row:
            # Update last accessed time
            conn.execute(
                "UPDATE sessions SET last_accessed = ? WHERE token = ?",
                (now, token),
            )
            conn.commit()
            if hasattr(conn, 'sync'):
                conn.sync()
            return _row_to_dict(cursor, row)
        return None
    finally:
        conn.close()


def revoke_session(token: str) -> None:
    """Revoke (delete) a session token."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
    finally:
        conn.close()


def revoke_all_user_sessions(user_id: int) -> None:
    """Revoke all sessions for a user (logout everywhere)."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
    finally:
        conn.close()


def cleanup_expired_sessions() -> int:
    """
    Remove expired sessions from the database.
    
    Returns:
        Number of sessions cleaned up
    """
    conn = _get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= ?",
            (now,),
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
        return cursor.rowcount
    finally:
        conn.close()


def list_users(limit: int = 100) -> list[dict[str, Any]]:
    """List all active users (for admin purposes)."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, username, email, created_at FROM users WHERE is_active = 1 ORDER BY id LIMIT ?",
            (limit,),
        )
        return [_row_to_dict(cursor, row) for row in cursor.fetchall()]
    finally:
        conn.close()


def user_exists(username: str = None, email: str = None) -> bool:
    """Check if a username or email already exists."""
    conn = _get_connection()
    try:
        if username:
            cursor = conn.execute(
                "SELECT 1 FROM users WHERE username = ? AND is_active = 1",
                (username,),
            )
        elif email:
            cursor = conn.execute(
                "SELECT 1 FROM users WHERE email = ? AND is_active = 1",
                (email,),
            )
        else:
            return False
        return cursor.fetchone() is not None
    finally:
        conn.close()


def log_login_attempt(username: str, success: bool, ip_address: Optional[str] = None) -> None:
    """Log a login attempt for brute-force monitoring."""
    conn = _get_connection()
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO login_attempts (username, timestamp, success, ip_address) VALUES (?, ?, ?, ?)",
            (username, timestamp, 1 if success else 0, ip_address),
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
    finally:
        conn.close()


def get_recent_failed_attempts(username: str, minutes: int = 10) -> int:
    """Count failed login attempts for a user in the last X minutes."""
    conn = _get_connection()
    try:
        since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM login_attempts WHERE username = ? AND success = 0 AND timestamp > ?",
            (username, since),
        )
        return cursor.fetchone()[0]
    finally:
        conn.close()


def save_user_data(user_id: int, data_type: str, file_name: str, content_json: str) -> None:
    """Save or update JSON data for a user."""
    conn = _get_connection()
    try:
        updated_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO user_data (user_id, data_type, file_name, content_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, data_type, file_name) DO UPDATE SET
                content_json = excluded.content_json,
                updated_at = excluded.updated_at
            """,
            (user_id, data_type, file_name, content_json, updated_at)
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
    finally:
        conn.close()


def load_user_data(user_id: int, data_type: str, file_name: str) -> Optional[str]:
    """Load JSON data for a user. Returns None if not found."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT content_json FROM user_data WHERE user_id = ? AND data_type = ? AND file_name = ?",
            (user_id, data_type, file_name)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        return None
    finally:
        conn.close()


def list_user_data(user_id: int, data_type: str) -> list[str]:
    """List all file names for a user and data type."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT file_name FROM user_data WHERE user_id = ? AND data_type = ?",
            (user_id, data_type)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def delete_user_data(user_id: int, data_type: str, file_name: str) -> bool:
    """Delete specific data for a user. Returns True if deleted."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM user_data WHERE user_id = ? AND data_type = ? AND file_name = ?",
            (user_id, data_type, file_name)
        )
        conn.commit()
        if hasattr(conn, 'sync'):
            conn.sync()
        return cursor.rowcount > 0
    finally:
        conn.close()