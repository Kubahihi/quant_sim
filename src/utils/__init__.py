"""Utility package with a lazy public API.

Importing a focused helper such as :mod:`src.utils.environment` happens on
every application start.  Eagerly importing the YAML configuration reader and
Loguru there made those optional dependencies part of the critical startup
path even when they were never used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "EnvironmentConfigurationError": "src.utils.environment",
    "is_production_environment": "src.utils.environment",
    "resolve_environment": "src.utils.environment",
    "load_config": "src.utils.config",
    "setup_logger": "src.utils.logger",
}

__all__ = [
    "EnvironmentConfigurationError",
    "is_production_environment",
    "load_config",
    "resolve_environment",
    "setup_logger",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
