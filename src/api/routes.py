"""
API Routes module.

Defines all API routes and creates the Flask application
with proper middleware and configuration.
"""

from __future__ import annotations

from collections import OrderedDict, deque
import logging
import math
import re
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from flask import Flask, current_app, g, jsonify, request

from .config import APIConfig
from .auth import set_api_config, require_auth
from .responses import APIResponse
from .readiness import ReadinessProbe
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


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class _InMemoryRateLimiter:
    """Small, process-local sliding-window limiter scoped to one Flask app."""

    _MAX_BUCKETS = 10_000

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_window < 1:
            raise ValueError("rate_limit_requests must be positive when rate limiting is enabled.")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("rate_limit_window must be positive when rate limiting is enabled.")

        self.requests_per_window = int(requests_per_window)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, key: str) -> tuple[bool, int]:
        """Record an allowed request or return the seconds until retry."""
        now = self._clock()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.pop(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.requests_per_window:
                self._buckets[key] = bucket
                retry_after = max(1, math.ceil(bucket[0] + self.window_seconds - now))
                return False, retry_after

            bucket.append(now)
            self._buckets[key] = bucket
            if len(self._buckets) > self._MAX_BUCKETS:
                self._buckets.popitem(last=False)
            return True, 0


def _request_id() -> str:
    """Return the correlation id assigned by the app's request hook."""
    request_id = getattr(g, "request_id", None)
    return str(request_id) if request_id else uuid4().hex


def _response_status(response_obj: Any, data: dict[str, Any], fallback: int) -> int:
    """Resolve an APIResponse status code while retaining a safe fallback."""
    if isinstance(response_obj, APIResponse):
        raw_status = data.get("meta", {}).get("status_code")
        try:
            resolved = int(raw_status)
        except (TypeError, ValueError):
            resolved = fallback
        if 100 <= resolved <= 599:
            return resolved
    return fallback


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
    
    resolved_status = _response_status(response_obj, data, status_code)
    if not data.get("success", False):
        response_meta = data.setdefault("meta", {})
        response_meta.setdefault("request_id", _request_id())

    if resolved_status >= 500:
        current_app.logger.error(
            "API request failed request_id=%s method=%s path=%s error_code=%s",
            _request_id(),
            request.method,
            request.path,
            data.get("error_code", "internal_error"),
        )
        data["error"] = "Internal server error"

    return (
        jsonify(data),
        resolved_status,
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
    request_id = _request_id()
    current_app.logger.error(
        "Unhandled API exception request_id=%s method=%s path=%s exception_type=%s",
        request_id,
        request.method,
        request.path,
        type(error).__name__,
    )
    payload = APIResponse.error("Internal server error", "internal_error", 500).to_dict()
    payload.setdefault("meta", {})["request_id"] = request_id
    return jsonify(payload), 500


def create_app(
    config: Optional[APIConfig] = None,
    readiness_probe: Callable[[], dict[str, Any]] | None = None,
) -> Flask:
    """
    Create and configure the Flask API application.
    
    Args:
        config: API configuration. If None, loads from config/settings.yaml
        readiness_probe: Optional dependency probe override, primarily for tests.
    
    Returns:
        Configured Flask application
    """
    if config is None:
        config = APIConfig.from_yaml()

    config.validate()

    # Set config for auth module
    set_api_config(config)
    
    app = Flask(__name__)
    app.config["DEBUG"] = config.debug
    app.extensions["quant_sim_api_config"] = config
    if readiness_probe is None:
        readiness_probe = ReadinessProbe(config, logger=app.logger)
    app.extensions["quant_sim_readiness_probe"] = readiness_probe
    if config.environment == "production":
        app.logger.setLevel(logging.INFO)

    rate_limiter: _InMemoryRateLimiter | None = None
    if config.rate_limit_enabled:
        rate_limiter = _InMemoryRateLimiter(
            config.rate_limit_requests,
            config.rate_limit_window,
        )
        app.extensions["quant_sim_rate_limiter"] = rate_limiter

    @app.before_request
    def prepare_request_context():
        g.request_started_at = time.perf_counter()
        supplied_request_id = str(request.headers.get("X-Request-ID", "")).strip()
        g.request_id = (
            supplied_request_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid4().hex
        )

        if (
            rate_limiter is None
            or request.method == "OPTIONS"
            or not request.path.startswith(config.api_prefix)
            or request.path in {
                f"{config.api_prefix}/health",
                f"{config.api_prefix}/ready",
            }
        ):
            return None

        client_address = request.remote_addr or "unknown"
        endpoint = request.endpoint or request.path
        allowed, retry_after = rate_limiter.check(f"{client_address}:{endpoint}")
        if allowed:
            return None

        response, response_status, headers = _json_response(
            APIResponse.error("Rate limit exceeded", "rate_limited", 429)
        )
        headers["Retry-After"] = str(retry_after)
        return response, response_status, headers
    
    # Register error handlers
    app.register_error_handler(404, _not_found_error)
    app.register_error_handler(500, _internal_error)
    
    # Add CORS headers
    @app.after_request
    def add_cors_headers(response):
        response.headers["X-Request-ID"] = _request_id()
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if config.environment == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        if config.cors_enabled:
            configured_origins = set(config.cors_origins)
            request_origin = request.headers.get("Origin")
            allowed_origin = None
            if "*" in configured_origins:
                allowed_origin = "*"
            elif request_origin and request_origin in configured_origins:
                allowed_origin = request_origin

            if allowed_origin is not None:
                response.headers["Access-Control-Allow-Origin"] = allowed_origin
                response.headers["Access-Control-Allow-Headers"] = (
                    f"Content-Type, {config.token_header}, Authorization"
                )
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                if allowed_origin != "*":
                    response.vary.add("Origin")
        response.headers["Content-Type"] = "application/json"
        started_at = getattr(g, "request_started_at", None)
        duration_ms = (
            max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
            if started_at is not None
            else 0.0
        )
        current_app.logger.info(
            "api_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            _request_id(),
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        return response
    
    # Register API routes
    register_routes(app, config, readiness_probe)
    
    return app


def register_routes(
    app: Flask,
    config: APIConfig,
    readiness_probe: Callable[[], dict[str, Any]],
) -> None:
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

    # Readiness endpoint (no auth required)
    @app.route(f"{api_prefix}/ready")
    def readiness_check():
        """GET /api/v1/ready - Verify the database and storage dependencies."""
        result = readiness_probe()
        if result.get("ready") is True:
            return _json_response(APIResponse.ok(result))

        current_app.logger.warning(
            "API is not ready request_id=%s checks=%s",
            _request_id(),
            {
                name: check.get("status", "unhealthy")
                for name, check in result.get("checks", {}).items()
                if isinstance(check, dict)
            },
        )
        response = APIResponse.error(
            "Service is not ready",
            "not_ready",
            503,
        ).to_dict()
        response["data"] = result
        response.setdefault("meta", {})["request_id"] = _request_id()
        return jsonify(response), 503, {"Content-Type": "application/json"}
    
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
        
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return _json_response(
                APIResponse.error("Request body required", "bad_request", 400),
                400,
            )
        
        username_value = data.get("username", "")
        password_value = data.get("password", "")
        username = username_value.strip() if isinstance(username_value, str) else ""
        password = password_value if isinstance(password_value, str) else ""
        
        if not username or not password:
            return _json_response(
                APIResponse.error("Username and password required", "bad_request", 400),
                400,
            )
        
        token, user, errors = login_user(
            username,
            password,
            ip_address=request.remote_addr,
        )
        
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
                {"path": "/ready", "method": "GET", "description": "Dependency readiness"},
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
