"""
Portfolio tracker package — position management, live valuation, and
pre-trade analysis.

yfinance (pulled in by manager.py) and all heavy analytics modules are
imported lazily so that importing from this package never blocks on network
libraries.
"""
from __future__ import annotations

__all__ = [
    "load_portfolio",
    "save_portfolio",
    "add_position",
    "remove_position",
    "update_position",
    "compute_live_values",
    "generate_rebalance_suggestions",
    "list_portfolios",
    "build_competition_strategy_snapshot",
    "simulate_trade_plan",
    "analyze_pretrade_impact",
    "assess_behavioral_profile",
]

_SUBMODULE_MAP: dict[str, str] = {
    "load_portfolio":                    "src.portfolio_tracker.manager",
    "save_portfolio":                    "src.portfolio_tracker.manager",
    "add_position":                      "src.portfolio_tracker.manager",
    "remove_position":                   "src.portfolio_tracker.manager",
    "update_position":                   "src.portfolio_tracker.manager",
    "compute_live_values":               "src.portfolio_tracker.manager",
    "generate_rebalance_suggestions":    "src.portfolio_tracker.manager",
    "list_portfolios":                   "src.portfolio_tracker.manager",
    "build_competition_strategy_snapshot": "src.portfolio_tracker.pretrade_analysis",
    "simulate_trade_plan":               "src.portfolio_tracker.pretrade_analysis",
    "analyze_pretrade_impact":           "src.portfolio_tracker.pretrade_analysis",
    "assess_behavioral_profile":         "src.portfolio_tracker.client_behavior",
}


def __getattr__(name: str):
    if name not in _SUBMODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(_SUBMODULE_MAP[name])
    obj = getattr(module, name)
    globals()[name] = obj
    return obj
