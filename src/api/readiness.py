"""Cached readiness checks for external API dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
import threading
import time
from typing import Any, Callable

from .config import APIConfig


ReadinessCheck = Callable[[], bool]


class ReadinessProbe:
    """Check critical dependencies while limiting remote probe traffic."""

    def __init__(
        self,
        config: APIConfig,
        *,
        database_check: ReadinessCheck | None = None,
        storage_check: ReadinessCheck | None = None,
        ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds < 0:
            raise ValueError("Readiness cache TTL must be a non-negative finite number.")

        self._config = config
        self._database_check = database_check or self._check_database
        self._storage_check = storage_check or self._check_storage
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._cached_result: dict[str, Any] | None = None
        self._cache_expires_at = 0.0

    def __call__(self) -> dict[str, Any]:
        """Return the cached readiness result or perform fresh checks."""
        with self._lock:
            now = self._clock()
            if self._cached_result is not None and now < self._cache_expires_at:
                return self._public_result(self._cached_result, cached=True)

            checks = {
                "database": {"status": self._run_check("database", self._database_check)},
                "storage": {"status": self._run_check("storage", self._storage_check)},
            }
            result = {
                "ready": all(check["status"] == "healthy" for check in checks.values()),
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "checks": checks,
            }
            self._cached_result = result
            self._cache_expires_at = self._clock() + self._ttl_seconds
            return self._public_result(result, cached=False)

    def invalidate(self) -> None:
        """Force the next call to perform fresh dependency checks."""
        with self._lock:
            self._cached_result = None
            self._cache_expires_at = 0.0

    def _run_check(self, component: str, check: ReadinessCheck) -> str:
        try:
            return "healthy" if check() is True else "unhealthy"
        except Exception as error:
            self._logger.error(
                "Readiness check failed component=%s exception_type=%s",
                component,
                type(error).__name__,
            )
            return "unhealthy"

    def _check_database(self) -> bool:
        from src.auth.database import AUTH_DB_PATH, get_db_connection

        connection = get_db_connection(AUTH_DB_PATH)
        try:
            row = connection.execute("SELECT 1").fetchone()
            return row is not None
        finally:
            connection.close()

    def _check_storage(self) -> bool:
        from src.storage.backend import get_storage_backend

        result = get_storage_backend().health_check()
        return isinstance(result, dict) and result.get("status") == "healthy"

    @staticmethod
    def _public_result(result: dict[str, Any], *, cached: bool) -> dict[str, Any]:
        """Copy only the intentionally public, non-sensitive readiness fields."""
        return {
            "ready": bool(result["ready"]),
            "checked_at": str(result["checked_at"]),
            "cached": cached,
            "checks": {
                name: {"status": str(check["status"])}
                for name, check in result["checks"].items()
            },
        }
