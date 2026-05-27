"""
API Authentication module.

Provides authentication decorators and utilities for API endpoints.
Integrates with the existing auth system from src.auth.manager.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from .config import APIConfig


# Header name for API token authentication
API_TOKEN_HEADER = "X-API-Token"

# Config placeholder (will be set by create_app)
_api_config: Optional[APIConfig] = None


def set_api_config(config: APIConfig) -> None:
    """Set the API configuration for auth module."""
    global _api_config
    _api_config = config


def get_api_config() -> Optional[APIConfig]:
    """Get the current API configuration."""
    return _api_config


def get_user_from_token(token: str) -> Optional[dict[str, Any]]:
    """
    Validate an API token and return the associated user.
    
    This function integrates with the existing auth.manager module
    to validate session tokens.
    
    Args:
        token: The API token/session token to validate
    
    Returns:
        User dict if valid, None otherwise
    """
    if not token:
        return None
    
    try:
        from src.auth.manager import get_current_user
        user = get_current_user(token)
        return user
    except Exception:
        return None


def _resolve_token_header(config: Optional[APIConfig]) -> str:
    """Resolve configured token header with safe fallback."""
    if config is not None and config.token_header:
        return str(config.token_header)
    return API_TOKEN_HEADER


def require_auth(f: Callable) -> Callable:
    """
    Decorator to require authentication for an endpoint.
    
    When applied to an endpoint handler, it checks for a valid
    API token in the request headers. If authentication fails,
    returns a 401 Unauthorized response.
    
    The user object is passed to the handler as the 'user' keyword argument.
    
    Usage:
        @require_auth
        def my_endpoint(user=None):
            # user will be None if auth is disabled or token is invalid
            pass
    """
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # Import Flask here to avoid dependency if not using Flask
        from flask import request
        
        config = get_api_config()
        
        # If auth is disabled, call the function without user
        if config is None or not config.auth_enabled:
            return f(*args, user=None, **kwargs)
        
        # Get token from header
        token = request.headers.get(_resolve_token_header(config))
        
        if not token:
            from flask import jsonify
            timestamp = _utc_iso()
            return jsonify({
                "success": False,
                "error": "Authentication required",
                "error_code": "auth_required",
                "timestamp": timestamp,
                "updatedAt": timestamp,
            }), 401
        
        # Validate token
        user = get_user_from_token(token)
        
        if user is None:
            from flask import jsonify
            timestamp = _utc_iso()
            return jsonify({
                "success": False,
                "error": "Invalid or expired token",
                "error_code": "invalid_token",
                "timestamp": timestamp,
                "updatedAt": timestamp,
            }), 401
        
        # Pass user to the handler
        return f(*args, user=user, **kwargs)
    
    return decorated_function


def optional_auth(f: Callable) -> Callable:
    """
    Decorator for endpoints where authentication is optional.
    
    If a valid token is provided, the user object is passed to the handler.
    If no token or an invalid token is provided, the handler is called
    with user=None.
    
    When auth is disabled, the default_user_id from config is used
    for data access.
    """
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        from flask import request
        
        config = get_api_config()
        user = None
        
        if config is not None and config.auth_enabled:
            token = request.headers.get(_resolve_token_header(config))
            if token:
                user = get_user_from_token(token)
        elif config is not None and config.default_user_id:
            # When auth is disabled, use default user
            try:
                from src.auth.database import get_user_by_id
                user = get_user_by_id(config.default_user_id)
            except Exception:
                pass
        
        return f(*args, user=user, **kwargs)
    
    return decorated_function


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def generate_api_token(user_id: int) -> str:
    """
    Generate a new API token for a user.
    
    This creates a session token using the existing auth system.
    
    Args:
        user_id: The ID of the user to generate a token for
    
    Returns:
        The generated token string
    """
    try:
        from src.auth.database import init_auth_database, create_session
        
        init_auth_database()
        token = create_session(user_id)
        
        return token
    except Exception:
        return ""


def revoke_api_token(token: str) -> bool:
    """
    Revoke an API token.
    
    Args:
        token: The token to revoke
    
    Returns:
        True if revoked successfully, False otherwise
    """
    try:
        from src.auth.database import revoke_session
        revoke_session(token)
        return True
    except Exception:
        return False
