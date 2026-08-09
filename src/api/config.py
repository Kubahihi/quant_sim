"""
API Configuration module.

Handles API settings including base URL, port, authentication,
and rate limiting configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.environment import resolve_environment


class APIConfigurationError(ValueError):
    """Raised when API settings are unsafe or internally inconsistent."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("Invalid API configuration: " + "; ".join(self.issues))


def _as_bool(value: Any, default: bool) -> bool:
    """Parse bool-like values from env/config."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class APIConfig:
    """
    Configuration for the Quant Sim API.
    
    Settings are loaded from config/settings.yaml with
    environment variable overrides.
    """
    
    # Server settings
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # API settings
    base_path: str = "/api"
    version: str = "v1"
    
    # Authentication settings
    auth_enabled: bool = True
    token_header: str = "X-API-Token"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60  # per minute
    rate_limit_window: int = 60    # seconds
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    
    # Data paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    
    # Default user for unauthenticated requests (if auth disabled)
    default_user_id: int | None = None
    
    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "APIConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
        else:
            config_path = Path(config_path)
        
        config_data: dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data and "api" in yaml_data:
                        config_data = yaml_data["api"]
            except Exception:
                pass
        
        # Override with environment variables
        environment = str(resolve_environment())
        host = os.getenv("API_HOST", config_data.get("host", "0.0.0.0"))
        port = int(os.getenv("API_PORT", config_data.get("port", 8080)))
        debug = _as_bool(os.getenv("API_DEBUG", config_data.get("debug", False)), False)
        base_path = str(os.getenv("API_BASE_PATH", config_data.get("base_path", "/api")))
        version = str(os.getenv("API_VERSION", config_data.get("version", "v1")))
        auth_enabled = _as_bool(os.getenv("API_AUTH_ENABLED", config_data.get("auth_enabled", True)), True)
        token_header = str(os.getenv("API_TOKEN_HEADER", config_data.get("token_header", "X-API-Token")))
        rate_limit_enabled = _as_bool(os.getenv("API_RATE_LIMIT", config_data.get("rate_limit_enabled", True)), True)
        rate_limit_requests = int(os.getenv("API_RATE_LIMIT_REQUESTS", config_data.get("rate_limit_requests", 60)))
        rate_limit_window = int(os.getenv("API_RATE_LIMIT_WINDOW", config_data.get("rate_limit_window", 60)))
        cors_enabled = _as_bool(os.getenv("API_CORS_ENABLED", config_data.get("cors_enabled", True)), True)
        cors_origins_raw = os.getenv("API_CORS_ORIGINS")
        if cors_origins_raw:
            cors_origins = [item.strip() for item in cors_origins_raw.split(",") if item.strip()]
        else:
            configured_origins = config_data.get("cors_origins", ["*"])
            if isinstance(configured_origins, list):
                cors_origins = [str(item) for item in configured_origins]
            else:
                cors_origins = [str(configured_origins)]

        default_user_id_raw = os.getenv("API_DEFAULT_USER_ID", config_data.get("default_user_id"))
        default_user_id = int(default_user_id_raw) if default_user_id_raw not in (None, "") else None
        
        return cls(
            environment=environment,
            host=host,
            port=port,
            debug=debug,
            base_path=base_path,
            version=version,
            auth_enabled=auth_enabled,
            token_header=token_header,
            rate_limit_enabled=rate_limit_enabled,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window=rate_limit_window,
            cors_enabled=cors_enabled,
            cors_origins=cors_origins,
            default_user_id=default_user_id,
        )
    
    @property
    def api_prefix(self) -> str:
        """Get the full API prefix path."""
        return f"{self.base_path}/{self.version}"

    def validate(self) -> None:
        """Reject invalid settings and unsafe production defaults."""
        issues: list[str] = []
        environment = str(self.environment or "development").strip().lower()
        self.environment = environment

        if environment not in {"development", "test", "production"}:
            issues.append("environment must be development, test, or production")
        if not 1 <= int(self.port) <= 65_535:
            issues.append("port must be between 1 and 65535")
        if not str(self.base_path).startswith("/"):
            issues.append("base_path must start with '/'")
        if not str(self.version).strip():
            issues.append("version must not be empty")
        if self.rate_limit_enabled and int(self.rate_limit_requests) < 1:
            issues.append("rate_limit_requests must be positive")
        if self.rate_limit_enabled and int(self.rate_limit_window) < 1:
            issues.append("rate_limit_window must be positive")

        if environment == "production":
            if self.debug:
                issues.append("debug mode must be disabled in production")
            if not self.auth_enabled:
                issues.append("authentication must be enabled in production")
            if not self.rate_limit_enabled:
                issues.append("rate limiting must be enabled in production")
            if self.cors_enabled and "*" in set(self.cors_origins or []):
                issues.append("wildcard CORS origins are not allowed in production")
            if self.cors_enabled and not self.cors_origins:
                issues.append("at least one explicit CORS origin is required when CORS is enabled")
            if self.default_user_id is not None:
                issues.append("default_user_id is not allowed in production")

        if issues:
            raise APIConfigurationError(issues)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "base_path": self.base_path,
            "version": self.version,
            "auth_enabled": self.auth_enabled,
            "token_header": self.token_header,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
        }
