"""Portfolio optimization public API with lazy module loading.

Importing the package itself stays lightweight. Numerical backends such as
SciPy and CVXPY are loaded only when their corresponding optimizer is used.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "optimize_minimum_variance": ".minimum_variance",
    "optimize_maximum_sharpe": ".maximum_sharpe",
    "optimize_cost_aware_rebalance": ".cost_aware_rebalance",
    "calculate_efficient_frontier": ".efficient_frontier",
    "calculate_portfolio_statistics": ".efficient_frontier",
    "sample_portfolio_cloud": ".efficient_frontier",
    "PortfolioEstimates": ".estimators",
    "clean_returns": ".estimators",
    "estimate_portfolio_inputs": ".estimators",
    "estimate_black_litterman_inputs": ".estimators",
    "resolve_portfolio_estimates": ".estimators",
    "run_optimization_walk_forward": ".walk_forward",
    "GroupConstraint": ".constraint_sets",
    "PortfolioConstraintSet": ".constraint_sets",
    "build_constraint_report": ".constraint_sets",
    "build_constraint_set": ".constraint_sets",
    "validate_constraint_solution": ".constraint_sets",
    "SUPPORTED_OBJECTIVES": ".engine",
    "optimize_portfolio": ".engine",
    "build_execution_plan": ".execution",
    "estimate_trade_costs": ".execution",
    "parse_tax_lots": ".execution",
    "align_point_in_time_membership": ".universe",
    "parse_point_in_time_membership": ".universe",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)
