# Multi-User Authentication Setup Guide

This document explains how to set up and use the multi-user authentication system in the Quant Platform.

## Overview

The authentication system provides:
- **User registration** with email and password
- **Secure login/logout** with session management
- **Data isolation** - each user's data is stored separately
- **Migration** of existing single-user data to the new system

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The main new dependency is `bcrypt>=4.0.0` for secure password hashing.

### 2. First Run (Automatic Migration)

When you first run the Streamlit app after adding the auth system, it will automatically:

1. Create the authentication database (`data/auth.db`)
2. Create a default admin user
3. Migrate any existing data to the admin user's directory

**Default credentials:**
- Username: `admin`
- Password: `admin123`

⚠️ **Important:** Change the default password after first login!

### 3. Run the Application

```bash
streamlit run ui/streamlit_app.py
```

You'll be presented with a login screen. Use the default credentials or register a new account.

## User Management

### Registration

1. Click "Create Account" on the login screen
2. Fill in username, email, and password
3. Password requirements:
   - At least 8 characters
   - At least one letter
   - At least one number

### Login

1. Enter your username and password
2. Click "Sign In"
3. Session persists for 24 hours

### Logout

Click the "🚪 Logout" button in the sidebar.

## Data Storage

### Directory Structure

After migration, user data is organized as:

```
data/
├── auth.db                    # Authentication database
├── .migration_completed       # Migration marker file
├── cache/
│   └── market_data.db         # Shared market data cache
├── users/
│   └── {user_id}/
│       ├── portfolios/        # User's portfolios
│       ├── swing_tracker/     # User's swing trades
│       └── run_history/       # User's analysis history
├── portfolios/                # Legacy (pre-migration)
├── swing_tracker/             # Legacy (pre-migration)
└── run_history/               # Legacy (pre-migration)
```

### Data Isolation

- Each user can only access their own data
- Portfolios, swing trades, and analysis history are completely separated
- Market data cache is shared (same stock prices for all users)

## API Usage

### Using Auth in Your Code

```python
from src.auth import (
    register_user,
    login_user,
    logout_user,
    get_current_user,
    is_authenticated,
    get_user_data_dir,
    ensure_user_dirs,
)

# Register a new user
user, errors = register_user("username", "email@example.com", "password123")
if errors:
    print(f"Registration failed: {errors}")

# Login
token, user, errors = login_user("username", "password123")
if token:
    print(f"Logged in as {user['username']}")

# Get current user
user = get_current_user(token)
if user:
    print(f"User ID: {user['id']}")

# Get user-specific data directory
user_dir = get_user_data_dir(user['id'])
dirs = ensure_user_dirs(user['id'])
print(f"Portfolio dir: {dirs['portfolios']}")

# Logout
logout_user(token)
```

### Using User-Specific Data Paths

The data layer functions now accept an optional `user_id` parameter:

```python
from src.portfolio_tracker.manager import load_portfolio, save_portfolio, list_portfolios

# Load user's portfolio
portfolio = load_portfolio("my_portfolio", user_id=current_user_id)

# Save to user's directory
save_portfolio(portfolio, "my_portfolio", user_id=current_user_id)

# List user's portfolios
portfolios = list_portfolios(user_id=current_user_id)
```

## Streamlit Integration

### In Your Streamlit App

```python
import streamlit as st
from ui.auth_page import (
    init_multi_user_mode,
    get_current_user_id,
    render_logout_button,
    render_user_info,
)

# Initialize auth system
user_id = init_multi_user_mode()

# Check if user is logged in
if user_id is None:
    st.stop()  # Will show login screen

# Show user info and logout button in sidebar
render_user_info()
render_logout_button()

# Use user_id for data operations
current_portfolio = load_portfolio("default", user_id=user_id)
```

## Migration

### Manual Migration

If you need to run migration manually:

```python
from src.auth import migrate_existing_data

# Dry run (preview what would be migrated)
result = migrate_existing_data(dry_run=True)
print(result)

# Actual migration
result = migrate_existing_data(dry_run=False)
print(result)
```

### Migration is Idempotent

- Safe to run multiple times
- Won't duplicate data
- Skips already-migrated files

### Rollback Migration

```python
from src.auth.migrations import rollback_migration

# Remove migration marker (allows re-migration)
rollback_migration()
```

⚠️ **Warning:** This only removes the marker file, not the migrated data.

## Testing

Run the test suite:

```bash
pytest tests/test_auth.py -v
```

Tests cover:
- Password hashing and verification
- Input validation
- User registration
- Login/logout
- Session management
- User data isolation
- Migration functionality

## Security Considerations

1. **Password Storage**: Passwords are hashed with bcrypt (industry standard)
2. **Session Security**: Sessions use cryptographically secure random tokens
3. **Session Expiry**: Sessions expire after 24 hours
4. **Input Validation**: All inputs are validated before processing
5. **SQL Injection Protection**: Parameterized queries prevent SQL injection

## Troubleshooting

### "Database locked" error

Close any other applications using the database, or delete `data/auth.db` to reset (this will delete all users).

### Can't log in after migration

The default user credentials are `admin` / `admin123`. If you've forgotten your password, you'll need to reset the database.

### Data not appearing after migration

Check that the migration completed successfully:
```python
from src.auth.migrations import get_migration_status
print(get_migration_status())
```

### Session expires too quickly

Increase session expiry in `src/auth/database.py`:
```python
SESSION_EXPIRY_HOURS = 48  # Change from 24 to 48
```

## Support

For issues or questions:
1. Check the test file `tests/test_auth.py` for usage examples
2. Review the docstrings in `src/auth/` modules
3. Open an issue on the project repository