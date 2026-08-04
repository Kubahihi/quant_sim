from .returns import calculate_returns, calculate_cumulative_returns
from .risk_metrics import calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown
from .correlation import calculate_correlation_matrix, calculate_covariance_matrix
from .portfolio_metrics import (
    calculate_portfolio_daily_returns,
    calculate_concentration_metrics,
    calculate_average_correlation,
    calculate_portfolio_core_metrics,
    build_portfolio_timeseries,
)
from .benchmark import (
    calculate_active_risk_metrics,
    calculate_return_contribution,
    calculate_risk_contribution,
)
from .model_validation import (
    build_model_validation_report,
    distribution_diagnostics,
    moving_block_bootstrap_intervals,
)
from .scoring import (
    build_deterministic_fallback_review,
    compute_weighted_factor_score,
    evaluate_portfolio_score,
)
from .advanced.runner import run_advanced_models
from .scenario_playground import (
    build_role_exposure_table,
    build_scenario_suite,
    classify_asset_role,
    list_scenario_presets,
    run_scenario_preset,
)
from .modular import (
    build_news_analysis,
    build_news_rows_for_ui,
    build_summary,
    compare_runs,
    list_run_records,
    load_run_record,
    run_model_bundle,
    run_quant_stack,
    run_signal_bundle,
)
from importlib import import_module


_LAZY_EXPORTS = {
    "DCF_SCHEMA_VERSION": ".dcf",
    "build_dcf_sensitivity": ".dcf",
    "build_multistage_dcf_scenarios": ".dcf",
    "calculate_multistage_dcf": ".dcf",
    "calculate_wacc": ".dcf",
    "default_multistage_dcf_assumptions": ".dcf",
    "prepare_dcf_inputs": ".dcf",
    "solve_reverse_dcf": ".dcf",
    "COMMODITY_CATALOG": ".commodity_analysis",
    "COMMODITY_SYMBOLS": ".commodity_analysis",
    "build_cumulative_index": ".commodity_analysis",
    "build_price_shock_table": ".commodity_analysis",
    "build_return_correlation": ".commodity_analysis",
    "calculate_commodity_metrics": ".commodity_analysis",
    "commodity_catalog_frame": ".commodity_analysis",
    "FX_USD_QUOTES": ".currency_risk",
    "SUPPORTED_CURRENCIES": ".currency_risk",
    "aggregate_currency_exposure": ".currency_risk",
    "build_fx_rate_history": ".currency_risk",
    "build_fx_stress_table": ".currency_risk",
    "calculate_fx_risk": ".currency_risk",
    "optimize_currency_hedges": ".currency_risk",
    "required_fx_symbols": ".currency_risk",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "calculate_returns",
    "calculate_cumulative_returns",
    "calculate_volatility",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_correlation_matrix",
    "calculate_covariance_matrix",
    "calculate_portfolio_daily_returns",
    "calculate_concentration_metrics",
    "calculate_average_correlation",
    "calculate_portfolio_core_metrics",
    "build_portfolio_timeseries",
    "calculate_active_risk_metrics",
    "calculate_return_contribution",
    "calculate_risk_contribution",
    "build_model_validation_report",
    "distribution_diagnostics",
    "moving_block_bootstrap_intervals",
    "evaluate_portfolio_score",
    "build_deterministic_fallback_review",
    "compute_weighted_factor_score",
    "run_advanced_models",
    "classify_asset_role",
    "build_role_exposure_table",
    "run_scenario_preset",
    "build_scenario_suite",
    "list_scenario_presets",
    "run_model_bundle",
    "run_signal_bundle",
    "build_summary",
    "build_news_analysis",
    "build_news_rows_for_ui",
    "run_quant_stack",
    "list_run_records",
    "load_run_record",
    "compare_runs",
    "DCF_SCHEMA_VERSION",
    "calculate_wacc",
    "prepare_dcf_inputs",
    "default_multistage_dcf_assumptions",
    "calculate_multistage_dcf",
    "build_multistage_dcf_scenarios",
    "build_dcf_sensitivity",
    "solve_reverse_dcf",
    "COMMODITY_CATALOG",
    "COMMODITY_SYMBOLS",
    "commodity_catalog_frame",
    "calculate_commodity_metrics",
    "build_cumulative_index",
    "build_return_correlation",
    "build_price_shock_table",
    "FX_USD_QUOTES",
    "SUPPORTED_CURRENCIES",
    "required_fx_symbols",
    "build_fx_rate_history",
    "aggregate_currency_exposure",
    "calculate_fx_risk",
    "optimize_currency_hedges",
    "build_fx_stress_table",
]
