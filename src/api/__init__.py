"""
Quant Sim API Module

Provides a RESTful API for external tools (like iOS Scriptable widgets)
to consume portfolio data, signals, and analytics from Quant Sim.

Flask and all handler dependencies are imported lazily so that this package
can be imported (e.g. to access ``APIConfig``) without requiring Flask to be
installed.
"""
from __future__ import annotations

__all__ = [
    "APIConfig",
    "APIConfigurationError",
    "ReadinessProbe",
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

_SUBMODULE_MAP: dict[str, str] = {
    "APIConfig":           "src.api.config",
    "APIConfigurationError": "src.api.config",
    "ReadinessProbe":       "src.api.readiness",
    "APIResponse":         "src.api.responses",
    "ErrorResponse":       "src.api.responses",
    "SuccessResponse":     "src.api.responses",
    "require_auth":        "src.api.auth",
    "get_user_from_token": "src.api.auth",
    "API_TOKEN_HEADER":    "src.api.auth",
    "handle_summary":      "src.api.handlers",
    "handle_portfolio":    "src.api.handlers",
    "handle_positions":    "src.api.handlers",
    "handle_watchlist":    "src.api.handlers",
    "handle_signals":      "src.api.handlers",
    "handle_recent_trades": "src.api.handlers",
    "handle_risk":         "src.api.handlers",
    "handle_overview":     "src.api.handlers",
    "register_routes":     "src.api.routes",
    "create_app":          "src.api.routes",
}


def __getattr__(name: str):
    if name not in _SUBMODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(_SUBMODULE_MAP[name])
    obj = getattr(module, name)
    globals()[name] = obj
    return obj
