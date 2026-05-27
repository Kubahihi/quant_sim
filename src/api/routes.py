"""
API Routes module.

Defines all API routes and creates the Flask application
with proper middleware and configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from flask import Flask, jsonify, request

from .config import APIConfig
from .auth import set_api_config, require_auth
from .responses import APIResponse
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


def _json_response(response_obj: Any, status_code: int = 200):
    """
    Convert an APIResponse to a JSON response with proper headers.
    
    Args:
        response_obj: APIResponse object or dict
        status_code: HTTP status code
    
    Returns:
        Flask response tuple (jsonify_data, status_code, headers)
    """
    if hasattr(response_obj, "to_dict"):
        data = response_obj.to_dict()
    elif isinstance(response_obj, dict):
        data = response_obj
    else:
        data = {"success": False, "error": "Unexpected response type"}
    
    return (
        jsonify(data),
        status_code,
        {"Content-Type": "application/json"},
    )


def _not_found_error(error):
    """Handle 404 errors with JSON response."""
    response = APIResponse.error("Endpoint not found", "not_found", 404)
    payload = response.to_dict()
    payload["path"] = request.path
    return jsonify(payload), 404


def _internal_error(error):
    """Handle 500 errors with JSON response."""
    return jsonify(APIResponse.error("Internal server error", "internal_error", 500).to_dict()), 500


def create_app(config: Optional[APIConfig] = None) -> Flask:
    """
    Create and configure the Flask API application.
    
    Args:
        config: API configuration. If None, loads from config/settings.yaml
    
    Returns:
        Configured Flask application
    """
    if config is None:
        config = APIConfig.from_yaml()
    
    # Set config for auth module
    set_api_config(config)
    
    app = Flask(__name__)
    app.config["DEBUG"] = config.debug
    
    # Register error handlers
    app.register_error_handler(404, _not_found_error)
    app.register_error_handler(500, _internal_error)
    
    # Add CORS headers
    @app.after_request
    def add_cors_headers(response):
        if config.cors_enabled:
            allowed_origins = ",".join(config.cors_origins) if config.cors_origins else "*"
            response.headers["Access-Control-Allow-Origin"] = allowed_origins
            response.headers["Access-Control-Allow-Headers"] = f"Content-Type, {config.token_header}, Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Content-Type"] = "application/json"
        return response
    
    # Register API routes
    register_routes(app, config)
    
    return app


def register_routes(app: Flask, config: APIConfig) -> None:
    """
    Register all API routes with the Flask application.
    
    Args:
        app: Flask application instance
        config: API configuration
    """
    api_prefix = config.api_prefix
    
    # Health check endpoint (no auth required)
    @app.route(f"{api_prefix}/health")
    def health_check():
        """
        GET /api/v1/health
        
        Health check endpoint. Returns API status and version.
        
        Sample response:
        {
            "success": true,
            "timestamp": "2026-05-22T20:00:00Z",
            "data": {
                "status": "healthy",
                "version": "1.0.0",
                "api_version": "v1"
            }
        }
        """
        response = APIResponse.ok({
            "status": "healthy",
            "version": "1.0.0",
            "api_version": config.version,
        })
        return _json_response(response)
    
    # Summary endpoint
    @app.route(f"{api_prefix}/summary")
    @require_auth
    def api_summary(user=None):
        """GET /api/v1/summary - Portfolio summary"""
        response = handle_summary(user)
        return _json_response(response)
    
    # Portfolio endpoint
    @app.route(f"{api_prefix}/portfolio")
    @require_auth
    def api_portfolio(user=None):
        """GET /api/v1/portfolio - Full portfolio with positions"""
        response = handle_portfolio(user)
        return _json_response(response)
    
    # Positions endpoint
    @app.route(f"{api_prefix}/positions")
    @require_auth
    def api_positions(user=None):
        """GET /api/v1/positions - List of positions"""
        response = handle_positions(user)
        return _json_response(response)
    
    # Watchlist endpoint
    @app.route(f"{api_prefix}/watchlist")
    @require_auth
    def api_watchlist(user=None):
        """GET /api/v1/watchlist - Watchlist with prices"""
        response = handle_watchlist(user)
        return _json_response(response)
    
    # Signals endpoint
    @app.route(f"{api_prefix}/signals")
    @require_auth
    def api_signals(user=None):
        """GET /api/v1/signals - Active alerts and signals"""
        response = handle_signals(user)
        return _json_response(response)
    
    # Recent trades endpoint
    @app.route(f"{api_prefix}/trades/recent")
    @require_auth
    def api_recent_trades(user=None):
        """GET /api/v1/trades/recent - Recent closed trades"""
        response = handle_recent_trades(user)
        return _json_response(response)
    
    # Risk endpoint
    @app.route(f"{api_prefix}/risk")
    @require_auth
    def api_risk(user=None):
        """GET /api/v1/risk - Risk metrics"""
        response = handle_risk(user)
        return _json_response(response)
    
    # Overview endpoint
    @app.route(f"{api_prefix}/overview")
    @require_auth
    def api_overview(user=None):
        """GET /api/v1/overview - Dashboard overview"""
        response = handle_overview(user)
        return _json_response(response)
    
    # Auth endpoint for generating API tokens
    @app.route(f"{api_prefix}/auth/token", methods=["POST"])
    def api_generate_token():
        """
        POST /api/v1/auth/token
        
        Generate an API token for authenticated users.
        Requires username and password in request body.
        
        Request body:
        {
            "username": "user",
            "password": "password"
        }
        
        Sample response:
        {
            "success": true,
            "timestamp": "2026-05-22T20:00:00Z",
            "data": {
                "token": "abc123...",
                "expires_in": 86400
            }
        }
        """
        from src.auth.manager import login_user
        
        data = request.get_json()
        if not data:
            return _json_response(
                APIResponse.error("Request body required", "bad_request", 400),
                400,
            )
        
        username = data.get("username", "").strip()
        password = data.get("password", "")
        
        if not username or not password:
            return _json_response(
                APIResponse.error("Username and password required", "bad_request", 400),
                400,
            )
        
        token, user, errors = login_user(username, password)
        
        if not token:
            return _json_response(
                APIResponse.error(" | ".join(errors), "auth_failed", 401),
                401,
            )
        
        return _json_response(APIResponse.ok({
            "token": token,
            "expires_in": 86400,  # 24 hours
            "user": {
                "id": user.get("id"),
                "username": user.get("username"),
            },
        }))
    
    # Root API info endpoint
    @app.route(f"{api_prefix}/")
    def api_info():
        """
        GET /api/v1/
        
        API information and available endpoints.
        """
        response = APIResponse.ok({
            "name": "Quant Sim API",
            "version": config.version,
            "endpoints": [
                {"path": "/health", "method": "GET", "description": "Health check"},
                {"path": "/summary", "method": "GET", "description": "Portfolio summary"},
                {"path": "/portfolio", "method": "GET", "description": "Full portfolio"},
                {"path": "/positions", "method": "GET", "description": "Position list"},
                {"path": "/watchlist", "method": "GET", "description": "Watchlist"},
                {"path": "/signals", "method": "GET", "description": "Active signals/alerts"},
                {"path": "/trades/recent", "method": "GET", "description": "Recent trades"},
                {"path": "/risk", "method": "GET", "description": "Risk metrics"},
                {"path": "/overview", "method": "GET", "description": "Dashboard overview"},
                {"path": "/auth/token", "method": "POST", "description": "Generate auth token"},
            ],
        })
        return _json_response(response)
