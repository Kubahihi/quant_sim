"""
Tests for the Quant Sim API.

Run with: python -m pytest tests/api/test_api.py -v
"""

from __future__ import annotations

import json
import pytest

from pathlib import Path

from src.api.config import APIConfig
from src.api.routes import create_app
from src.api.responses import APIResponse, make_paginated_response


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
