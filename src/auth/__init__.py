"""Authentication API with lazy backend imports.

Importing a lightweight authentication submodule (for example the fixed
Wharton credential contract) must not initialize the database, session manager,
and migration stack. Public exports retain the original ``src.auth`` API and
are resolved on first use.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "init_auth_database": "src.auth.database",
    "create_user": "src.auth.database",
    "get_user_by_username": "src.auth.database",
    "get_user_by_id": "src.auth.database",
    "validate_session_token": "src.auth.database",
    "create_session": "src.auth.database",
    "revoke_session": "src.auth.database",
    "cleanup_expired_sessions": "src.auth.database",
    "get_user_by_session_token": "src.auth.database",
    "register_user": "src.auth.manager",
    "login_user": "src.auth.manager",
    "logout_user": "src.auth.manager",
    "get_current_user": "src.auth.manager",
    "is_authenticated": "src.auth.manager",
    "get_user_data_dir": "src.auth.manager",
    "ensure_user_dirs": "src.auth.manager",
    "migrate_existing_data": "src.auth.migrations",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
