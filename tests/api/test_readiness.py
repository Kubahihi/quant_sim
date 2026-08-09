"""Readiness probe and endpoint tests."""

from __future__ import annotations

import json
import logging

from src.api.config import APIConfig
from src.api.readiness import ReadinessProbe
from src.api.routes import create_app


def _test_config(**overrides) -> APIConfig:
    values = {
        "host": "127.0.0.1",
        "auth_enabled": False,
        "rate_limit_enabled": False,
        "cors_enabled": False,
    }
    values.update(overrides)
    return APIConfig(**values)


def test_probe_caches_dependency_checks_until_ttl_expires():
    now = [100.0]
    calls = {"database": 0, "storage": 0}

    def check_database() -> bool:
        calls["database"] += 1
        return True

    def check_storage() -> bool:
        calls["storage"] += 1
        return True

    probe = ReadinessProbe(
        _test_config(),
        database_check=check_database,
        storage_check=check_storage,
        ttl_seconds=30,
        clock=lambda: now[0],
    )

    first = probe()
    second = probe()

    assert first["ready"] is True
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["checked_at"] == first["checked_at"]
    assert calls == {"database": 1, "storage": 1}

    now[0] += 31
    refreshed = probe()

    assert refreshed["cached"] is False
    assert calls == {"database": 2, "storage": 2}


def test_probe_hides_exception_details_from_public_result(caplog):
    secret_detail = "token=do-not-return endpoint=C:/private/auth.db"

    def failing_database() -> bool:
        raise RuntimeError(secret_detail)

    probe = ReadinessProbe(
        _test_config(),
        database_check=failing_database,
        storage_check=lambda: True,
        ttl_seconds=0,
    )

    with caplog.at_level(logging.ERROR):
        result = probe()

    assert result["ready"] is False
    assert result["checks"] == {
        "database": {"status": "unhealthy"},
        "storage": {"status": "healthy"},
    }
    assert secret_detail not in json.dumps(result)
    assert secret_detail not in caplog.text
    assert "exception_type=RuntimeError" in caplog.text


def test_ready_endpoint_returns_200_for_healthy_dependencies():
    result = {
        "ready": True,
        "checked_at": "2026-08-09T12:00:00+00:00",
        "cached": False,
        "checks": {
            "database": {"status": "healthy"},
            "storage": {"status": "healthy"},
        },
    }
    app = create_app(_test_config(), readiness_probe=lambda: result)

    response = app.test_client().get("/api/v1/ready")

    assert response.status_code == 200
    assert response.get_json()["data"] == result


def test_ready_endpoint_returns_safe_503_for_unhealthy_dependencies(caplog):
    secret_detail = "secret-access-key"

    def failing_database() -> bool:
        raise RuntimeError(secret_detail)

    config = _test_config()
    probe = ReadinessProbe(
        config,
        database_check=failing_database,
        storage_check=lambda: True,
        ttl_seconds=0,
    )
    app = create_app(config, readiness_probe=probe)

    with caplog.at_level(logging.ERROR):
        response = app.test_client().get("/api/v1/ready")

    payload = response.get_json()
    assert response.status_code == 503
    assert payload["success"] is False
    assert payload["error"] == "Service is not ready"
    assert payload["error_code"] == "not_ready"
    assert payload["data"]["ready"] is False
    assert payload["data"]["checks"]["database"] == {"status": "unhealthy"}
    assert payload["meta"]["request_id"] == response.headers["X-Request-ID"]
    assert secret_detail not in response.get_data(as_text=True)
    assert secret_detail not in caplog.text


def test_health_and_ready_endpoints_are_not_rate_limited():
    result = {
        "ready": True,
        "checked_at": "2026-08-09T12:00:00+00:00",
        "cached": True,
        "checks": {},
    }
    app = create_app(
        _test_config(rate_limit_enabled=True, rate_limit_requests=1),
        readiness_probe=lambda: result,
    )
    client = app.test_client()

    for _ in range(3):
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/ready").status_code == 200
