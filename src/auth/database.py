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


def _get_db_path() -> Path:
    """Get the database path, creating parent directories if needed."""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return AUTH_DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with proper settings."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
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

            -- Index for faster session lookups
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
            ON sessions(user_id);
            
            -- Index for session cleanup
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
            ON sessions(expires_at);
        """)
        conn.commit()
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
            return dict(row)
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
            return dict(row)
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
            return dict(row)
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
            return dict(row)
        return None
    finally:
        conn.close()


def revoke_session(token: str) -> None:
    """Revoke (delete) a session token."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


def revoke_all_user_sessions(user_id: int) -> None:
    """Revoke all sessions for a user (logout everywhere)."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
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
        return [dict(row) for row in cursor.fetchall()]
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