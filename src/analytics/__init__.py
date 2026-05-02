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
from .scoring import (
    build_deterministic_fallback_review,
    compute_weighted_factor_score,
    evaluate_portfolio_score,
)
from .advanced import run_advanced_models
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
]
