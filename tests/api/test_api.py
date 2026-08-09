"""
Tests for the Quant Sim API.

Run with: python -m pytest tests/api/test_api.py -v
"""

from __future__ import annotations

import json
import logging
import pytest

from pathlib import Path

import src.api.routes as api_routes
from src.api.config import APIConfig, APIConfigurationError
from src.api.routes import create_app
from src.api.responses import APIResponse, make_paginated_response
from src.utils import environment as runtime_environment


@pytest.fixture
def app():
    """Create a test Flask app with auth disabled."""
    config = APIConfig(
        host="127.0.0.1",
        port=5555,
        debug=True,
        auth_enabled=False,
        rate_limit_enabled=False,
    )
    app = create_app(config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def app_auth_enabled():
    """Create a test Flask app with auth enabled."""
    config = APIConfig(
        host="127.0.0.1",
        port=5556,
        debug=True,
        auth_enabled=True,
        rate_limit_enabled=False,
    )
    app = create_app(config)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def client_auth_enabled(app_auth_enabled):
    """Create a test client with auth enabled config."""
    return app_auth_enabled.test_client()


class TestAPIResponse:
    """Tests for APIResponse class."""
    
    def test_ok_response(self):
        response = APIResponse.ok({"key": "value"})
        assert response.success is True
        assert response.data == {"key": "value"}
        assert response.error is None
    
    def test_error_response(self):
        response = APIResponse.error("Something went wrong", "test_error", 400)
        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.error_code == "test_error"
    
    def test_to_dict_success(self):
        response = APIResponse.ok({"data": 123}, {"page": 1})
        result = response.to_dict()
        assert result["success"] is True
        assert result["data"] == {"data": 123}
        assert result["meta"] == {"page": 1}
        assert "timestamp" in result
        assert "updatedAt" in result
    
    def test_to_dict_error(self):
        response = APIResponse.error("Error message", "err_code", 500)
        result = response.to_dict()
        assert result["success"] is False
        assert result["error"] == "Error message"
        assert result["error_code"] == "err_code"
        assert result["meta"]["status_code"] == 500


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert "version" in data["data"]
        assert "updatedAt" in data
    
    def test_health_check_content_type(self, client):
        response = client.get("/api/v1/health")
        assert response.content_type == "application/json"


class TestAPIInfo:
    """Tests for the API info endpoint."""
    
    def test_api_info(self, client):
        response = client.get("/api/v1/")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert data["data"]["name"] == "Quant Sim API"
        assert isinstance(data["data"]["endpoints"], list)
        assert len(data["data"]["endpoints"]) > 0


class TestSummaryEndpoint:
    """Tests for the summary endpoint."""
    
    def test_summary_returns_valid_structure(self, client):
        response = client.get("/api/v1/summary")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "total_value" in data["data"]
        assert "positions_count" in data["data"]


class TestPortfolioEndpoint:
    """Tests for the portfolio endpoint."""
    
    def test_portfolio_returns_valid_structure(self, client):
        response = client.get("/api/v1/portfolio")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "positions" in data["data"]
        assert isinstance(data["data"]["positions"], list)


class TestPositionsEndpoint:
    """Tests for the positions endpoint."""
    
    def test_positions_returns_list(self, client):
        response = client.get("/api/v1/positions")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)


class TestWatchlistEndpoint:
    """Tests for the watchlist endpoint."""
    
    def test_watchlist_returns_list(self, client):
        response = client.get("/api/v1/watchlist")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)


class TestSignalsEndpoint:
    """Tests for the signals endpoint."""
    
    def test_signals_returns_valid_structure(self, client):
        response = client.get("/api/v1/signals")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "active_trades" in data["data"]
        assert "alerts" in data["data"]


class TestRecentTradesEndpoint:
    """Tests for the recent trades endpoint."""
    
    def test_recent_trades_returns_list(self, client):
        response = client.get("/api/v1/trades/recent")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)


class TestRiskEndpoint:
    """Tests for the risk endpoint."""
    
    def test_risk_returns_valid_structure(self, client):
        response = client.get("/api/v1/risk")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "risk_flags" in data["data"]
        assert isinstance(data["data"]["risk_flags"], list)


class TestOverviewEndpoint:
    """Tests for the overview endpoint."""
    
    def test_overview_returns_valid_structure(self, client):
        response = client.get("/api/v1/overview")
        assert response.status_code == 200
        
        data = response.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "portfolio" in data["data"]
        assert "trading" in data["data"]
        assert "market" in data["data"]


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_returns_json(self, client):
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        data = response.get_json()
        assert data["success"] is False
        assert data["error_code"] == "not_found"
    
    def test_cors_headers(self, client):
        response = client.get("/api/v1/health")
        assert response.headers.get("Access-Control-Allow-Origin") == "*"

    def test_security_headers_and_request_metrics_are_emitted(self, client, caplog):
        with caplog.at_level(logging.INFO):
            response = client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "health-probe-1"},
            )

        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Pragma"] == "no-cache"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Strict-Transport-Security" not in response.headers
        assert "request_id=health-probe-1" in caplog.text
        assert "path=/api/v1/health" in caplog.text
        assert "duration_ms=" in caplog.text

    def test_handler_5xx_uses_http_status_and_hides_internal_detail(
        self,
        client,
        monkeypatch,
        caplog,
    ):
        internal_detail = "database failed at C:/private/auth.db"
        monkeypatch.setattr(
            api_routes,
            "handle_summary",
            lambda _user=None: APIResponse.error(
                internal_detail,
                "summary_error",
                500,
            ),
        )

        with caplog.at_level(logging.ERROR):
            response = client.get(
                "/api/v1/summary",
                headers={"X-Request-ID": "test-correlation-123"},
            )

        payload = response.get_json()
        assert response.status_code == 500
        assert payload["success"] is False
        assert payload["error"] == "Internal server error"
        assert internal_detail not in response.get_data(as_text=True)
        assert payload["error_code"] == "summary_error"
        assert payload["meta"]["status_code"] == 500
        assert payload["meta"]["request_id"] == "test-correlation-123"
        assert response.headers["X-Request-ID"] == "test-correlation-123"
        assert internal_detail not in caplog.text
        assert "error_code=summary_error" in caplog.text

    def test_cors_echoes_only_an_allowed_origin(self):
        cors_app = create_app(APIConfig(
            auth_enabled=False,
            rate_limit_enabled=False,
            cors_enabled=True,
            cors_origins=["https://first.example", "https://second.example"],
        ))
        cors_client = cors_app.test_client()

        allowed = cors_client.get(
            "/api/v1/health",
            headers={"Origin": "https://second.example"},
        )
        denied = cors_client.get(
            "/api/v1/health",
            headers={"Origin": "https://attacker.example"},
        )

        assert allowed.headers["Access-Control-Allow-Origin"] == "https://second.example"
        assert "Origin" in allowed.headers.get("Vary", "")
        assert "Access-Control-Allow-Origin" not in denied.headers


class TestRateLimiting:
    """Rate limiting must be effective and isolated to each Flask app."""

    @staticmethod
    def _create_limited_app(monkeypatch, *, request_limit: int = 2):
        import src.auth.manager as auth_manager

        monkeypatch.setattr(
            auth_manager,
            "login_user",
            lambda _username, _password, ip_address=None: (
                None,
                None,
                ["Invalid username or password"],
            ),
        )
        return create_app(APIConfig(
            auth_enabled=True,
            rate_limit_enabled=True,
            rate_limit_requests=request_limit,
            rate_limit_window=60,
            cors_enabled=False,
        ))

    def test_auth_token_rate_limit_returns_429_and_retry_after(self, monkeypatch):
        limited_app = self._create_limited_app(monkeypatch, request_limit=2)
        limited_client = limited_app.test_client()
        request_kwargs = {
            "json": {"username": "probe", "password": "Wrong123"},
            "environ_base": {"REMOTE_ADDR": "192.0.2.10"},
        }

        first = limited_client.post("/api/v1/auth/token", **request_kwargs)
        second = limited_client.post("/api/v1/auth/token", **request_kwargs)
        blocked = limited_client.post("/api/v1/auth/token", **request_kwargs)
        other_client = limited_client.post(
            "/api/v1/auth/token",
            json=request_kwargs["json"],
            environ_base={"REMOTE_ADDR": "192.0.2.11"},
        )

        assert first.status_code == 401
        assert second.status_code == 401
        assert blocked.status_code == 429
        assert blocked.get_json()["error_code"] == "rate_limited"
        assert int(blocked.headers["Retry-After"]) >= 1
        assert other_client.status_code == 401

    def test_options_does_not_consume_rate_limit(self, monkeypatch):
        limited_app = self._create_limited_app(monkeypatch, request_limit=1)
        limited_client = limited_app.test_client()
        address = {"REMOTE_ADDR": "192.0.2.20"}

        for _ in range(3):
            response = limited_client.options(
                "/api/v1/auth/token",
                environ_base=address,
            )
            assert response.status_code == 200

        first_post = limited_client.post(
            "/api/v1/auth/token",
            json={"username": "probe", "password": "Wrong123"},
            environ_base=address,
        )
        assert first_post.status_code == 401

    def test_rate_limit_state_is_scoped_per_app(self, monkeypatch):
        first_app = self._create_limited_app(monkeypatch, request_limit=1)
        second_app = self._create_limited_app(monkeypatch, request_limit=1)
        first_client = first_app.test_client()
        second_client = second_app.test_client()
        request_kwargs = {
            "json": {"username": "probe", "password": "Wrong123"},
            "environ_base": {"REMOTE_ADDR": "192.0.2.30"},
        }

        assert first_app.extensions["quant_sim_rate_limiter"] is not second_app.extensions[
            "quant_sim_rate_limiter"
        ]
        assert first_client.post("/api/v1/auth/token", **request_kwargs).status_code == 401
        assert first_client.post("/api/v1/auth/token", **request_kwargs).status_code == 429
        assert second_client.post("/api/v1/auth/token", **request_kwargs).status_code == 401


class TestPagination:
    """Tests for pagination helper."""
    
    def test_make_paginated_response(self):
        data = [{"id": 1}, {"id": 2}]
        response = make_paginated_response(data, total=50, page=1, per_page=10)
        
        assert response.success is True
        assert response.meta["page"] == 1
        assert response.meta["per_page"] == 10
        assert response.meta["total"] == 50
        assert response.meta["total_pages"] == 5
        assert response.meta["has_next"] is True
        assert response.meta["has_prev"] is False


class TestAuthEndpoint:
    """Tests for the auth token endpoint."""
    
    def test_auth_requires_credentials(self, client):
        response = client.post("/api/v1/auth/token", json={})
        assert response.status_code == 400
        
        data = response.get_json()
        assert data["success"] is False

    def test_auth_rejects_malformed_json_as_json_error(self, client):
        response = client.post(
            "/api/v1/auth/token",
            data=b'{"username":',
            content_type="application/json",
        )

        assert response.status_code == 400
        assert response.is_json
        assert response.get_json()["error_code"] == "bad_request"

    def test_auth_rejects_non_string_credentials(self, client):
        response = client.post(
            "/api/v1/auth/token",
            json={"username": ["user"], "password": {"secret": True}},
        )

        assert response.status_code == 400
        assert response.get_json()["error_code"] == "bad_request"
    
    def test_auth_rejects_invalid_credentials(self, client):
        response = client.post("/api/v1/auth/token", json={
            "username": "invalid_user",
            "password": "wrong_password"
        })
        assert response.status_code == 401
        
        data = response.get_json()
        assert data["success"] is False


class TestProtectedEndpoints:
    """Tests for token protection on data endpoints."""

    def test_summary_requires_token_when_auth_enabled(self, client_auth_enabled):
        response = client_auth_enabled.get("/api/v1/summary")
        assert response.status_code == 401
        data = response.get_json()
        assert data["success"] is False
        assert data["error_code"] == "auth_required"

    def test_auth_configuration_is_isolated_per_app(self):
        protected_app = create_app(APIConfig(auth_enabled=True, rate_limit_enabled=False))
        open_app = create_app(APIConfig(auth_enabled=False, rate_limit_enabled=False))
        protected_app.config["TESTING"] = True
        open_app.config["TESTING"] = True

        assert open_app.test_client().get("/api/v1/summary").status_code == 200
        assert protected_app.test_client().get("/api/v1/summary").status_code == 401


class TestAPIConfig:
    """Tests for API configuration loading."""

    def test_config_loads_extended_fields(self, tmp_path: Path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            """
api:
  base_path: "/mobile-api"
  version: "v9"
  host: "127.0.0.1"
  port: 9090
  token_header: "X-Mobile-Token"
  auth_enabled: false
  default_user_id: 7
  cors_enabled: false
  cors_origins: ["https://example.com"]
""".strip(),
            encoding="utf-8",
        )

        config = APIConfig.from_yaml(config_file)
        assert config.base_path == "/mobile-api"
        assert config.version == "v9"
        assert config.host == "127.0.0.1"
        assert config.port == 9090
        assert config.token_header == "X-Mobile-Token"
        assert config.auth_enabled is False
        assert config.default_user_id == 7
        assert config.cors_enabled is False
        assert config.cors_origins == ["https://example.com"]

    def test_config_reads_environment_from_streamlit_secret(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUANT_SIM_ENV", raising=False)
        monkeypatch.setattr(
            runtime_environment,
            "_streamlit_environment",
            lambda: "production",
        )
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "api:\n  cors_enabled: false\n  cors_origins: []\n",
            encoding="utf-8",
        )

        config = APIConfig.from_yaml(config_file)

        assert config.environment == "production"

    @pytest.mark.parametrize(
        ("overrides", "expected_issue"),
        [
            ({"debug": True}, "debug mode"),
            ({"auth_enabled": False}, "authentication"),
            ({"rate_limit_enabled": False}, "rate limiting"),
            ({"cors_origins": ["*"]}, "wildcard CORS"),
            ({"cors_origins": []}, "explicit CORS origin"),
            ({"default_user_id": 1}, "default_user_id"),
        ],
    )
    def test_production_config_rejects_unsafe_defaults(self, overrides, expected_issue):
        values = {
            "environment": "production",
            "auth_enabled": True,
            "rate_limit_enabled": True,
            "cors_enabled": True,
            "cors_origins": ["https://dashboard.example"],
            "default_user_id": None,
        }
        values.update(overrides)

        with pytest.raises(APIConfigurationError, match=expected_issue):
            create_app(APIConfig(**values))

    def test_production_config_accepts_explicit_secure_settings(self):
        app = create_app(APIConfig(
            environment="production",
            auth_enabled=True,
            rate_limit_enabled=True,
            cors_enabled=True,
            cors_origins=["https://dashboard.example"],
        ))

        assert app.extensions["quant_sim_api_config"].environment == "production"
        assert app.logger.isEnabledFor(logging.INFO)
        response = app.test_client().get("/api/v1/health")
        assert response.headers["Strict-Transport-Security"] == (
            "max-age=31536000; includeSubDomains"
        )

    def test_empty_cors_origin_list_is_preserved(self, tmp_path: Path):
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(
            "api:\n  cors_enabled: false\n  cors_origins: []\n",
            encoding="utf-8",
        )

        config = APIConfig.from_yaml(config_file)

        assert config.cors_enabled is False
        assert config.cors_origins == []

    def test_enabled_cors_with_empty_origin_list_does_not_open_wildcard(self):
        app = create_app(APIConfig(
            cors_enabled=True,
            cors_origins=[],
            auth_enabled=False,
            rate_limit_enabled=False,
        ))

        response = app.test_client().get(
            "/api/v1/health",
            headers={"Origin": "https://attacker.example"},
        )

        assert "Access-Control-Allow-Origin" not in response.headers
