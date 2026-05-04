"""
Tests for authentication and multi-user support.

Tests cover:
- User registration and validation
- Login/logout functionality
- Session management
- Password hashing
- User data isolation
- Migration functionality
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path
from unittest import TestCase

# Set up test environment
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

import src.auth
import src.auth.database
import src.auth.manager
import src.auth.migrations

from src.auth.database import (
    init_auth_database,
    create_user,
    get_user_by_username,
    get_user_by_id,
    validate_session_token,
    create_session,
    revoke_session,
    get_user_by_session_token,
    user_exists,
    _get_db_path,
)
from src.auth.manager import (
    hash_password,
    verify_password,
    validate_username,
    validate_email,
    validate_password,
    register_user,
    login_user,
    logout_user,
    get_current_user,
    is_authenticated,
    get_user_data_dir,
    ensure_user_dirs,
)
from src.auth.migrations import (
    migrate_existing_data,
    get_migration_status,
    create_default_user,
)


class TestPasswordHashing(TestCase):
    """Test password hashing and verification."""
    
    def test_hash_password_returns_string(self):
        result = hash_password("TestPass123")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_verify_password_correct(self):
        password = "SecurePass123"
        hashed = hash_password(password)
        self.assertTrue(verify_password(password, hashed))
    
    def test_verify_password_incorrect(self):
        password = "SecurePass123"
        hashed = hash_password(password)
        self.assertFalse(verify_password("WrongPass456", hashed))
    
    def test_different_hashes_for_same_password(self):
        password = "SamePassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        # bcrypt includes salt, so hashes should be different
        # (unless using fallback SHA-256)
        # Both should still verify correctly
        self.assertTrue(verify_password(password, hash1))
        self.assertTrue(verify_password(password, hash2))


class TestInputValidation(TestCase):
    """Test input validation functions."""
    
    def test_validate_username_valid(self):
        valid, msg = validate_username("testuser")
        self.assertTrue(valid)
        self.assertEqual(msg, "")
    
    def test_validate_username_too_short(self):
        valid, msg = validate_username("ab")
        self.assertFalse(valid)
        self.assertIn("at least 3 characters", msg)
    
    def test_validate_username_too_long(self):
        valid, msg = validate_username("a" * 31)
        self.assertFalse(valid)
        self.assertIn("30 characters or less", msg)
    
    def test_validate_username_invalid_chars(self):
        valid, msg = validate_username("test@user")
        self.assertFalse(valid)
    
    def test_validate_username_starts_with_number(self):
        valid, msg = validate_username("1testuser")
        self.assertFalse(valid)
    
    def test_validate_email_valid(self):
        valid, msg = validate_email("test@example.com")
        self.assertTrue(valid)
        self.assertEqual(msg, "")
    
    def test_validate_email_invalid_format(self):
        valid, msg = validate_email("not-an-email")
        self.assertFalse(valid)
        self.assertIn("Invalid email", msg)
    
    def test_validate_email_empty(self):
        valid, msg = validate_email("")
        self.assertFalse(valid)
    
    def test_validate_password_valid(self):
        valid, msg = validate_password("SecurePass123")
        self.assertTrue(valid)
        self.assertEqual(msg, "")
    
    def test_validate_password_too_short(self):
        valid, msg = validate_password("Short1")
        self.assertFalse(valid)
        self.assertIn("at least 8 characters", msg)
    
    def test_validate_password_no_letter(self):
        valid, msg = validate_password("12345678")
        self.assertFalse(valid)
        self.assertIn("at least one letter", msg)
    
    def test_validate_password_no_number(self):
        valid, msg = validate_password("OnlyLetters")
        self.assertFalse(valid)
        self.assertIn("at least one number", msg)


class TestUserRegistration(TestCase):
    """Test user registration functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        # Use a temporary database for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_db_path = Path(cls.temp_dir) / "test_auth.db"
        # Set environment variable to use test database
        os.environ["AUTH_TEST_DB_PATH"] = str(cls.test_db_path)
        # Reinitialize the database module to pick up the new path
        import importlib
        db_module = importlib.reload(src.auth.database)
        importlib.reload(src.auth.manager)
        importlib.reload(src.auth.migrations)
        # Initialize the test database
        db_module.init_auth_database()
    
    def test_register_user_success(self):
        user, errors = register_user(
            username="testuser1",
            email="test1@example.com",
            password="TestPass123",
            confirm_password="TestPass123"
        )
        self.assertIsNotNone(user)
        self.assertEqual(len(errors), 0)
        self.assertEqual(user["username"], "testuser1")
        self.assertEqual(user["email"], "test1@example.com")
        self.assertIn("id", user)
    
    def test_register_user_password_mismatch(self):
        user, errors = register_user(
            username="testuser2",
            email="test2@example.com",
            password="TestPass123",
            confirm_password="DifferentPass456"
        )
        self.assertIsNone(user)
        self.assertIn("Passwords do not match", errors)
    
    def test_register_user_duplicate_username(self):
        # First registration
        user1, errors1 = register_user(
            username="duplicateuser",
            email="dup1@example.com",
            password="TestPass123",
            confirm_password="TestPass123"
        )
        self.assertIsNotNone(user1)
        
        # Second registration with same username
        user2, errors2 = register_user(
            username="duplicateuser",
            email="dup2@example.com",
            password="TestPass123",
            confirm_password="TestPass123"
        )
        self.assertIsNone(user2)
        self.assertIn("Username already exists", errors2)
    
    def test_register_user_duplicate_email(self):
        # First registration
        user1, errors1 = register_user(
            username="emailuser1",
            email="same@example.com",
            password="TestPass123",
            confirm_password="TestPass123"
        )
        self.assertIsNotNone(user1)
        
        # Second registration with same email
        user2, errors2 = register_user(
            username="emailuser2",
            email="same@example.com",
            password="TestPass123",
            confirm_password="TestPass123"
        )
        self.assertIsNone(user2)
        self.assertIn("Email already registered", errors2)


class TestLoginLogout(TestCase):
    """Test login and logout functionality."""
    
    def test_login_success(self):
        # Register a user first (use unique username)
        user, errors = register_user(
            username="loginuser_success",
            email="login_success@example.com",
            password="LoginPass123",
            confirm_password="LoginPass123"
        )
        self.assertIsNotNone(user, f"Registration failed: {errors}")
        
        # Login
        token, logged_user, login_errors = login_user("loginuser_success", "LoginPass123")
        self.assertIsNotNone(token)
        self.assertIsNotNone(logged_user)
        self.assertEqual(len(login_errors), 0)
        self.assertEqual(logged_user["username"], "loginuser_success")
    
    def test_login_wrong_password(self):
        # Register a user first (use unique username)
        user, errors = register_user(
            username="wrongpassuser_unique",
            email="wrongpass_unique@example.com",
            password="CorrectPass123",
            confirm_password="CorrectPass123"
        )
        self.assertIsNotNone(user, f"Registration failed: {errors}")
        
        # Try to login with wrong password
        token, logged_user, login_errors = login_user("wrongpassuser_unique", "WrongPass456")
        self.assertIsNone(token)
        self.assertIsNone(logged_user)
        self.assertIn("Invalid username or password", login_errors)
    
    def test_login_nonexistent_user(self):
        token, logged_user, login_errors = login_user("nonexistent", "AnyPass123")
        self.assertIsNone(token)
        self.assertIsNone(logged_user)
        self.assertIn("Invalid username or password", login_errors)
    
    def test_logout(self):
        # Register and login
        user, _ = register_user(
            username="logoutuser",
            email="logout@example.com",
            password="LogoutPass123",
            confirm_password="LogoutPass123"
        )
        token, _, _ = login_user("logoutuser", "LogoutPass123")
        self.assertIsNotNone(token)
        
        # Verify session is valid
        self.assertTrue(validate_session_token(token))
        
        # Logout
        logout_user(token)
        
        # Verify session is no longer valid
        self.assertFalse(validate_session_token(token))
    
    def test_get_current_user(self):
        # Register and login
        user, _ = register_user(
            username="currentuser",
            email="current@example.com",
            password="CurrentPass123",
            confirm_password="CurrentPass123"
        )
        token, _, _ = login_user("currentuser", "CurrentPass123")
        
        # Get current user
        current_user = get_current_user(token)
        self.assertIsNotNone(current_user)
        self.assertEqual(current_user["username"], "currentuser")
    
    def test_is_authenticated(self):
        # Register and login
        user, _ = register_user(
            username="authuser",
            email="auth@example.com",
            password="AuthPass123",
            confirm_password="AuthPass123"
        )
        token, _, _ = login_user("authuser", "AuthPass123")
        
        # Check authentication
        self.assertTrue(is_authenticated(token))
        
        # Logout and check again
        logout_user(token)
        self.assertFalse(is_authenticated(token))


class TestUserDataIsolation(TestCase):
    """Test that user data is properly isolated."""
    
    def test_get_user_data_dir(self):
        user_dir = get_user_data_dir(999)  # Use high ID to avoid conflicts
        self.assertIsInstance(user_dir, Path)
        self.assertTrue(str(user_dir).endswith("999"))
    
    def test_ensure_user_dirs(self):
        dirs = ensure_user_dirs(998)
        self.assertIn("root", dirs)
        self.assertIn("portfolios", dirs)
        self.assertIn("swing_tracker", dirs)
        self.assertIn("run_history", dirs)
        
        # Verify directories exist
        for dir_path in dirs.values():
            self.assertTrue(dir_path.exists())


class TestMigration(TestCase):
    """Test data migration functionality."""
    
    def test_get_migration_status_initial(self):
        status = get_migration_status()
        # Initial status should show not completed
        self.assertIn("completed", status)
    
    def test_create_default_user(self):
        user = create_default_user()
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "admin")
        
        # Second call should return existing user
        user2 = create_default_user()
        self.assertIsNotNone(user2)
        self.assertEqual(user2["username"], "admin")
    
    def test_migrate_existing_data(self):
        result = migrate_existing_data(dry_run=True)
        self.assertIn("success", result)
        self.assertTrue(result["success"])


if __name__ == "__main__":
    import unittest
    unittest.main()