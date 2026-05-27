"""
Quant Sim API Module

Provides a RESTful API for external tools (like iOS Scriptable widgets)
to consume portfolio data, signals, and analytics from Quant Sim.
"""

from .config import APIConfig
from .responses import APIResponse, ErrorResponse, SuccessResponse
from .auth import require_auth, get_user_from_token, API_TOKEN_HEADER
from .handlers import (
    handle_summary,
    handle_portfolio,
    handle_positions,
    handle_watchlist,
    handle_signals,
    handle_recent_trades,
    handle_risk,
    handle_overview,
)
from .routes import register_routes, create_app

__all__ = [
    "APIConfig",
    "APIResponse",
    "ErrorResponse",
    "SuccessResponse",
    "require_auth",
    "get_user_from_token",
    "API_TOKEN_HEADER",
    "handle_summary",
    "handle_portfolio",
    "handle_positions",
    "handle_watchlist",
    "handle_signals",
    "handle_recent_trades",
    "handle_risk",
    "handle_overview",
    "register_routes",
    "create_app",
]