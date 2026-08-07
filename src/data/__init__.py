"""
Data package — market data fetching, caching, and validation.

Heavy optional dependencies (yfinance) are imported lazily so that importing
from this package (e.g. ``from src.data import PriceValidator``) never forces
yfinance to load if it is not installed or not needed by the caller.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# PEP 562 — lazy module-level __getattr__ keeps the public API identical
# to an eager import while deferring the actual work until first use.
__all__ = [
    "YahooFetcher",
    "CacheManager",
    "PriceValidator",
    "build_universe_snapshot",
    "load_universe_snapshot",
    "load_universe_metadata",
    "refresh_universe_if_stale",
    "get_universe",
]

_SUBMODULE_MAP: dict[str, str] = {
    "YahooFetcher":               "src.data.fetchers.yahoo_fetcher",
    "CacheManager":               "src.data.cache_manager",
    "PriceValidator":             "src.data.validators",
    "build_universe_snapshot":    "src.data.stock_universe",
    "load_universe_snapshot":     "src.data.stock_universe",
    "load_universe_metadata":     "src.data.stock_universe",
    "refresh_universe_if_stale":  "src.data.stock_universe",
    "get_universe":               "src.data.stock_universe",
}


def __getattr__(name: str):
    if name not in _SUBMODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(_SUBMODULE_MAP[name])
    obj = getattr(module, name)
    # Cache on the package so subsequent accesses are O(1).
    globals()[name] = obj
    return obj
