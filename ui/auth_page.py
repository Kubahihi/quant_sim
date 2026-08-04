"""
Authentication UI for Streamlit.

Provides login, registration, and session management UI components.
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from typing import Any

import streamlit as st


# ---- Session state keys ----
AUTH_TOKEN_KEY = "auth_token"
AUTH_USER_KEY = "auth_user"
AUTH_INIT_KEY = "auth_initialized"
AuthStorageInitializer = Callable[[], dict[str, Any]]


def _init_auth_system():
    """Initialize the auth system (run once at startup)."""
    if not st.session_state.get(AUTH_INIT_KEY):
        from src.auth import init_auth_database

        init_auth_database()
        st.session_state[AUTH_INIT_KEY] = True


def render_login_form(
    prepare_auth_storage: AuthStorageInitializer | None = None,
) -> None:
    """Render the login form."""
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            """
            <div class="dashboard-hero" style="margin-top:3rem;">
                <div class="dashboard-kicker">Secure workspace</div>
                <h2>Welcome back.</h2>
                <p>Sign in to continue to portfolio analytics, decision tools, and reporting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                key="login_username",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            login_clicked = st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
            )
            show_register = st.form_submit_button(
                "Create an account",
                use_container_width=True,
            )

            if login_clicked:
                if prepare_auth_storage is not None:
                    prepare_auth_storage()

                from src.auth import login_user

                token, user, errors = login_user(username, password)
                if errors:
                    for error in errors:
                        st.error(error)
                elif token and user:
                    st.session_state[AUTH_TOKEN_KEY] = token
                    st.session_state[AUTH_USER_KEY] = user
                    st.success(f"Welcome back, {user['username']}!")
                    st.rerun()

            if show_register:
                st.session_state["show_register"] = True
                st.rerun()

        if st.session_state.get("show_register"):
            render_register_form(prepare_auth_storage=prepare_auth_storage)


def render_register_form(
    prepare_auth_storage: AuthStorageInitializer | None = None,
) -> None:
    """Render the registration form."""
    st.markdown("---")
    st.subheader("Create your account")
    st.caption("Use a unique username and a password with at least eight characters.")
    
    with st.form("register_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            reg_username = st.text_input("Username *", placeholder="3-30 characters", key="reg_username")
            email = st.text_input("Email *", placeholder="your@email.com", key="reg_email")
        with col2:
            password = st.text_input("Password *", type="password", placeholder="Min 8 chars, 1 letter, 1 number", key="reg_password")
            confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Re-enter password", key="reg_confirm_password")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            register_clicked = st.form_submit_button("Create account", type="primary", use_container_width=True)
        with col2:
            back_to_login = st.form_submit_button("Back to sign in", use_container_width=True)
        
        if register_clicked:
            if prepare_auth_storage is not None:
                prepare_auth_storage()

            from src.auth import register_user

            user, errors = register_user(reg_username, email, password, confirm_password)
            if errors:
                for error in errors:
                    st.error(error)
            elif user:
                st.success(f"Account created for {user['username']}! Please log in.")
                st.session_state["show_register"] = False
                st.rerun()
        
        if back_to_login:
            st.session_state["show_register"] = False
            st.rerun()


def render_logout_button() -> bool:
    """
    Render a logout button in the sidebar.
    
    Returns True if logout was clicked.
    """
    if st.sidebar.button("Sign out", use_container_width=True):
        from src.auth import logout_user

        token = st.session_state.get(AUTH_TOKEN_KEY)
        if token:
            logout_user(token)
        st.session_state.pop(AUTH_TOKEN_KEY, None)
        st.session_state.pop(AUTH_USER_KEY, None)
        st.success("Logged out successfully!")
        st.rerun()
        return True
    return False


def render_user_info() -> None:
    """Render current user info in the sidebar."""
    user = st.session_state.get(AUTH_USER_KEY)
    if user:
        username = escape(str(user.get("username", "User")))
        initial = username[:1].upper()
        st.sidebar.markdown(
            f"""
            <div class="qp-brand" style="padding:0.1rem 0;">
                <div class="qp-brand-mark" style="width:2rem;height:2rem;">{initial}</div>
                <div class="qp-brand-copy">
                    <strong>{username}</strong>
                    <span>Authenticated session</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def check_auth() -> bool:
    """
    Check if user is authenticated. Call this at the start of your app.
    
    Returns True if authenticated, False otherwise.
    If not authenticated, renders login form and stops execution.
    """
    _init_auth_system()
    
    token = st.session_state.get(AUTH_TOKEN_KEY)
    
    if token:
        from src.auth import get_current_user

        # get_current_user validates the token; avoid doing the same remote
        # database lookup once through is_authenticated and then again here.
        user = get_current_user(token)
        if user:
            st.session_state[AUTH_USER_KEY] = user
            return True
    
    # Not authenticated - show login
    render_login_form()
    st.stop()
    return False


def get_current_user_id() -> int | None:
    """Get the current user's ID, or None if not authenticated."""
    user = st.session_state.get(AUTH_USER_KEY)
    if user:
        return user.get("id")
    return None


def require_auth(func):
    """
    Decorator to require authentication for a page/function.
    
    Usage:
        @st.cache_data
        @require_auth
        def get_user_data():
            ...
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get(AUTH_TOKEN_KEY):
            st.error("Authentication required")
            st.stop()
            return None
        return func(*args, **kwargs)
    return wrapper


def render_migration_info() -> None:
    """Render migration status info (for admin)."""
    from src.auth.migrations import get_migration_status
    
    status = get_migration_status()
    if status.get("completed"):
        st.sidebar.caption(f"Migration completed")
        st.sidebar.caption(f"   at {status.get('migrated_at', 'unknown')}")
    else:
        st.sidebar.caption("Migration pending")


def init_multi_user_mode() -> int | None:
    """
    Initialize multi-user mode and run migration if needed.
    
    Call this at the start of your Streamlit app to:
    1. Initialize auth database
    2. Run migration if needed
    3. Return current user ID if authenticated
    
    Returns:
        Current user ID if authenticated, None otherwise
    """
    _init_auth_system()
    
    # Run migration if needed
    from src.auth import get_current_user, migrate_existing_data
    from src.auth.migrations import get_migration_status
    status = get_migration_status()
    if not status.get("completed"):
        # Auto-run migration on first startup
        result = migrate_existing_data()
        if result.get("success") and not result.get("already_migrated"):
            st.info(f" Multi-user system initialized! Default user: admin / REMOVED")
            st.info(f"   {result.get('files_migrated', {}).get('total', 0)} files migrated.")
    
    # Check authentication
    token = st.session_state.get(AUTH_TOKEN_KEY)
    if token:
        user = get_current_user(token)
        if user:
            st.session_state[AUTH_USER_KEY] = user
            return user.get("id")
    
    return None
