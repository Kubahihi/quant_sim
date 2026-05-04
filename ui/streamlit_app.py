from __future__ import annotations

from datetime import date, datetime, timedelta
from html import escape
from io import BytesIO
from pathlib import Path
import importlib
import inspect
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Guard against accidentally reusing a non-local `src` package from site-packages.
for module_name, module_obj in list(sys.modules.items()):
    if module_name != "src" and not module_name.startswith("src."):
        continue
    module_file = getattr(module_obj, "__file__", None)
    if not module_file:
        continue
    resolved = str(Path(module_file).resolve())
    if not resolved.startswith(PROJECT_ROOT):
        sys.modules.pop(module_name, None)

from ui.economics_questions import render_economics_questions_section
from ui.dashboard_shell import (
    PAGE_LABELS,
    DashboardPreferences,
    inject_dashboard_styles,
    render_dashboard_preferences,
)
from ui.auth_page import (
    init_multi_user_mode,
    get_current_user_id,
    render_logout_button,
    render_user_info,
)
from src.ai import generate_ai_review, resolve_groq_api_key
from src.analytics import (
    build_deterministic_fallback_review,
    build_news_rows_for_ui,
    build_portfolio_timeseries,
    compare_runs,
    calculate_average_correlation,
    calculate_concentration_metrics,
    calculate_correlation_matrix,
    list_run_records,
    load_run_record,
    calculate_portfolio_core_metrics,
    calculate_portfolio_daily_returns,
    evaluate_portfolio_score,
    run_advanced_models,
    run_quant_stack,
)
from src.analytics.scenario_playground import build_scenario_suite, list_scenario_presets
from src.analytics.risk_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)
from src.analytics.returns import calculate_annualized_return
import src.data.universe_enrichment as universe_enrichment_module
import src.data.universe_sources as universe_sources_module
import src.data.stock_universe as stock_universe_module
from src.data.fetchers.yahoo_fetcher import YahooFetcher
from src.optimization import (
    calculate_efficient_frontier,
    calculate_portfolio_statistics,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
    sample_portfolio_cloud,
)
from src.reporting import (
    export_full_report_json,
    export_portfolio_data_csv,
    generate_pdf_report,
)
from src.simulation import run_monte_carlo_simulation
from src.stock_picker.ai_filter import apply_ai_query, parse_ai_query
import src.stock_picker.screener as stock_screener_module
from src.stock_picker.screener import (
    apply_classic_filters,
    apply_technical_indicators,
    calculate_quant_score,
    rank_stocks,
)
from src.portfolio_tracker.manager import (
    add_position,
    compute_live_values,
    generate_rebalance_suggestions,
    list_portfolios,
    load_portfolio,
    remove_position,
    save_portfolio,
    update_position,
)
from src.swing_tracker import (
    build_discipline_overview,
    build_stop_rationale,
    calculate_position_size_for_trade,
    calculate_stop_loss,
    classify_setup_type,
    close_trade as close_swing_trade,
    create_trade,
    generate_stop_rationale as generate_ai_stop_rationale,
    historical_trade_rows,
    open_trade_rows,
    refresh_and_persist as refresh_swing_trades,
    resolve_swing_tracker_api_key,
    save_trade_book as save_swing_trade_book,
    summarize_post_trade_review,
    summarize_trade_thesis,
    trades_to_rows,
    upsert_trade as upsert_swing_trade,
)
from src.visualization.charts_2d import (
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_drawdown,
    plot_efficient_frontier,
    plot_monte_carlo_fan,
)
from src.visualization.charts_3d import (
    plot_monte_carlo_percentile_surface,
    plot_portfolio_tradeoff_3d,
)
from src.visualization.cockpit_charts import (
    plot_asset_stress_impact,
    plot_crisis_playback,
    plot_phase_timeline,
    plot_scenario_atlas,
    plot_scenario_fingerprint,
    plot_scenario_shock_map,
)


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "VTI",
    "GLD",
    "BND",
]


st.set_page_config(page_title="Quant Platform", layout="wide", page_icon=":bar_chart:")
inject_dashboard_styles()

# ---- Authentication initialization ----
# First, check if user is already logged in
token = st.session_state.get("auth_token")
user_id = None

if token:
    from src.auth import is_authenticated, get_current_user
    if is_authenticated(token):
        user = get_current_user(token)
        if user:
            user_id = user.get("id")
            st.session_state["auth_user"] = user

# If not authenticated, show login screen
if user_id is None:
    from ui.auth_page import render_login_form
    
    # Initialize auth database if needed
    from src.auth import init_auth_database
    init_auth_database()
    
    # Run migration if needed
    from src.auth.migrations import get_migration_status, migrate_existing_data
    status = get_migration_status()
    if not status.get("completed"):
        result = migrate_existing_data()
        if result.get("success") and not result.get("already_migrated"):
            st.info(f"🎉 Multi-user system initialized! Default user: admin / admin123")
            st.info(f"   {result.get('files_migrated', {}).get('total', 0)} files migrated.")
    
    # Show login form and stop
    render_login_form()
    st.stop()

# Show user info and logout in sidebar if logged in
with st.sidebar:
    render_user_info()
    render_logout_button()
    st.markdown("---")

st.title("Quant Platform v0.4")
st.caption(
    "Portfolio evaluator with deterministic scoring, AI review via Groq, "
    "and multi-page PDF export."
)
st.markdown("---")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data_cached(
    symbols: Tuple[str, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch close prices with Streamlit cache to reduce repeated API calls."""
    fetcher = YahooFetcher()
    return fetcher.fetch_close_prices(list(symbols), start_date, end_date)


def _parse_tickers(raw_text: str) -> Tuple[List[str], List[str]]:
    tickers = [line.strip().upper() for line in raw_text.splitlines() if line.strip()]
    errors: List[str] = []

    if not tickers:
        errors.append("Ticker list is empty.")
        return [], errors

    duplicates = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicates:
        errors.append(f"Duplicate tickers found: {', '.join(duplicates)}.")

    return tickers, errors


def _parse_weights(raw_text: str, n_tickers: int) -> Tuple[np.ndarray, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if n_tickers == 0:
        errors.append("Cannot parse weights without tickers.")
        return np.array([]), warnings, errors

    if not raw_text.strip():
        equal_weight = np.array([100.0 / n_tickers] * n_tickers, dtype=float)
        warnings.append("Weights were empty; equal weighting was applied.")
        return equal_weight, warnings, errors

    weight_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(weight_lines) != n_tickers:
        errors.append(
            f"Number of weights ({len(weight_lines)}) must match number of tickers ({n_tickers})."
        )
        return np.array([]), warnings, errors

    parsed_weights: List[float] = []
    for index, line in enumerate(weight_lines, start=1):
        cleaned = line.replace("%", "").replace(",", ".").strip()
        try:
            value = float(cleaned)
        except ValueError:
            errors.append(f"Weight on line {index} is not a valid number: '{line}'.")
            continue

        if value < 0:
            errors.append(f"Weight on line {index} is negative: '{line}'.")
        parsed_weights.append(value)

    if errors:
        return np.array([]), warnings, errors

    weights = np.array(parsed_weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        errors.append("Weight sum must be greater than 0.")
        return np.array([]), warnings, errors

    if not np.isclose(total, 100.0, atol=0.01):
        weights = weights / total * 100.0
        warnings.append(f"Weights were normalized from {total:.2f}% to 100.00%.")

    return weights, warnings, errors


def _align_prices_and_weights(
    prices: pd.DataFrame,
    tickers: List[str],
    weights_pct: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    warnings: List[str] = []
    available_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]

    if not available_tickers:
        raise ValueError("No valid market data was fetched for the selected tickers.")

    aligned_prices = prices[available_tickers].dropna(how="all")
    if aligned_prices.empty:
        raise ValueError("Fetched data is empty after alignment.")

    weight_by_ticker = dict(zip(tickers, weights_pct, strict=False))
    aligned_pct = np.array([weight_by_ticker[ticker] for ticker in available_tickers], dtype=float)

    if aligned_pct.sum() <= 0:
        raise ValueError("Weights for tickers with data are zero.")

    if missing_tickers:
        warnings.append(
            f"Missing market data for: {', '.join(missing_tickers)}. "
            "Remaining weights were renormalized."
        )

    if not np.isclose(aligned_pct.sum(), 100.0, atol=0.01):
        original_sum = float(aligned_pct.sum())
        aligned_pct = aligned_pct / original_sum * 100.0
        warnings.append(
            f"Weights were re-normalized for available tickers "
            f"({original_sum:.2f}% -> 100.00%)."
        )

    return aligned_prices, aligned_pct / 100.0, missing_tickers, warnings


def _build_ai_payload(
    tickers: List[str],
    weights: np.ndarray,
    metrics: Dict[str, float],
    score_result: Dict[str, Any],
    flags: List[str],
    risk_profile: str,
    horizon_days: int,
) -> Dict[str, Any]:
    return {
        "goal": "Evaluate portfolio quality and practical risk controls.",
        "risk_profile": risk_profile,
        "horizon_days": horizon_days,
        "tickers": tickers,
        "weights_pct": [round(float(weight * 100.0), 2) for weight in weights],
        "metrics": {
            "annualized_return": round(float(metrics.get("annualized_return", 0.0)), 6),
            "volatility": round(float(metrics.get("volatility", 0.0)), 6),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 4),
            "max_drawdown": round(float(metrics.get("max_drawdown", 0.0)), 6),
            "avg_correlation": round(float(metrics.get("avg_correlation", 0.0)), 4),
            "concentration_hhi": round(float(metrics.get("hhi", 0.0)), 4),
            "effective_holdings": round(float(metrics.get("effective_holdings", 0.0)), 4),
            "max_weight": round(float(metrics.get("max_weight", 0.0)), 4),
        },
        "score": int(score_result.get("score", 0)),
        "rating": score_result.get("rating", "N/A"),
        "flags": flags,
    }


def _create_simulation_percentiles(price_paths: np.ndarray) -> pd.DataFrame:
    days = np.arange(price_paths.shape[0])
    return pd.DataFrame({
        "day": days,
        "p5": np.percentile(price_paths, 5, axis=1),
        "p25": np.percentile(price_paths, 25, axis=1),
        "p50": np.percentile(price_paths, 50, axis=1),
        "p75": np.percentile(price_paths, 75, axis=1),
        "p95": np.percentile(price_paths, 95, axis=1),
    })


def _model_signals_from_outputs(model_outputs: Dict[str, Any]) -> Dict[str, float]:
    lr_metrics = model_outputs.get("linear_regression", {}).get("metrics", {})
    arima_metrics = model_outputs.get("arima", {}).get("metrics", {})
    garch_metrics = model_outputs.get("garch", {}).get("metrics", {})

    return {
        "lr_expected_annual_return": float(lr_metrics.get("expected_annual_return", 0.0) or 0.0),
        "arima_next_return": float(arima_metrics.get("next_period_return_forecast", 0.0) or 0.0),
        "garch_annualized_volatility": float(garch_metrics.get("volatility_annualized", 0.0) or 0.0),
    }


def _build_report_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    serialized_frontier = []
    for point in result["frontier"]:
        serialized_frontier.append({
            "return": float(point.get("return", 0.0)),
            "volatility": float(point.get("volatility", 0.0)),
            "sharpe_ratio": float(point.get("sharpe_ratio", 0.0)),
            "diversification_score": float(point.get("diversification_score", 0.0)),
            "effective_holdings": float(point.get("effective_holdings", 0.0)),
            "max_weight": float(point.get("max_weight", 0.0)),
            "top_holdings": point.get("top_holdings", ""),
            "weights": [float(value) for value in point.get("weights", [])],
        })

    return {
        "inputs": {
            "tickers": result["tickers"],
            "weights_pct": [round(float(weight * 100), 2) for weight in result["weights"]],
            "horizon_days": result["horizon_days"],
            "risk_profile": result["risk_profile"],
            "risk_free_rate": result["risk_free_rate"],
            "date_range": {
                "start": result["start_date"].isoformat(),
                "end": result["end_date"].isoformat(),
            },
        },
        "metrics": result["metrics"],
        "score": {
            "score": result["score_result"]["score"],
            "rating": result["score_result"]["rating"],
            "breakdown": result["score_result"]["breakdown"],
        },
        "flags": result["score_result"]["flags"],
        "correlation_matrix": result["correlation_matrix"],
        "portfolio_timeseries": result["portfolio_timeseries"],
        "simulation": result["simulation_stats"],
        "simulation_percentiles": result["simulation_percentiles"],
        "frontier_points": serialized_frontier,
        "ai_review": result["ai_review"],
        "advanced_models": result.get("advanced_models", {}),
        "models": {
            name: item.to_dict() for name, item in result.get("model_results", {}).items()
        },
        "signals": {
            name: item.to_dict() for name, item in result.get("signal_results", {}).items()
        },
        "summary_layer": result.get("summary_result").to_dict() if result.get("summary_result") else {},
        "news_layer": result.get("news_result").to_dict() if result.get("news_result") else {},
        "backtest_layer": {
            "metrics": result.get("backtest_result", {}).get("metrics", {}),
            "lookahead_safe": bool(result.get("backtest_result", {}).get("lookahead_safe", False)),
        },
        "recommendation": result["ai_review"].get("verdict", ""),
    }


def _build_pdf_figures(result: Dict[str, Any]) -> Dict[str, Any]:
    portfolio_returns = result["portfolio_returns"]
    returns = result["returns"]
    corr_matrix = result["correlation_matrix"]
    price_paths = result["price_paths"]
    frontier = result["frontier"]

    figures: Dict[str, Any] = {}
    figures["cumulative"] = plot_cumulative_returns(
        pd.DataFrame({"Portfolio": portfolio_returns}),
        title="Portfolio Cumulative Return",
    )
    figures["drawdown"] = plot_drawdown(portfolio_returns, title="Portfolio Drawdown")
    figures["correlation"] = plot_correlation_heatmap(corr_matrix, title="Correlation Matrix")
    figures["monte_carlo"] = plot_monte_carlo_fan(price_paths, title="Monte Carlo Fan Chart")

    if frontier:
        figures["frontier"] = plot_efficient_frontier(frontier, title="Efficient Frontier")

    asset_figure = plot_cumulative_returns(returns, title="Asset Cumulative Returns")
    figures["assets"] = asset_figure
    return figures


def _analysis_export_cache_key(result: Dict[str, Any]) -> str:
    run_record = result.get("run_record")
    run_id = getattr(run_record, "run_id", None)
    if run_id:
        return f"exports::{run_id}"

    tickers_key = ",".join(str(ticker) for ticker in result.get("tickers", []))
    date_key = f"{result.get('start_date', '')}:{result.get('end_date', '')}"
    return f"exports::{tickers_key}::{date_key}::{int(result.get('horizon_days', 0))}"


def _prepare_export_artifacts(result: Dict[str, Any]) -> Dict[str, Any]:
    cache = st.session_state.setdefault("_prepared_export_artifacts", {})
    cache_key = _analysis_export_cache_key(result)
    if cache_key in cache:
        return cache[cache_key]

    report_payload = _build_report_payload(result)
    artifacts: Dict[str, Any] = {
        "report_payload": report_payload,
        "pdf_bytes": None,
        "csv_bytes": None,
        "json_bytes": None,
        "errors": {},
    }

    pdf_figures: Dict[str, Any] = {}
    try:
        pdf_figures = _build_pdf_figures(result)
        pdf_buffer = generate_pdf_report(report_payload, pdf_figures)
        artifacts["pdf_bytes"] = pdf_buffer.getvalue()
    except Exception as exc:
        artifacts["errors"]["pdf"] = str(exc)
    finally:
        for figure in pdf_figures.values():
            try:
                plt.close(figure)
            except Exception:
                pass

    try:
        artifacts["csv_bytes"] = export_portfolio_data_csv(report_payload)
    except Exception as exc:
        artifacts["errors"]["csv"] = str(exc)

    try:
        artifacts["json_bytes"] = export_full_report_json(report_payload)
    except Exception as exc:
        artifacts["errors"]["json"] = str(exc)

    cache[cache_key] = artifacts
    return artifacts


def _render_export_actions(
    analysis_result: Dict[str, Any],
    title: str,
    body: str,
    compact: bool = False,
    key_namespace: str = "exports",
) -> None:
    export_artifacts = _prepare_export_artifacts(analysis_result)
    errors = dict(export_artifacts.get("errors", {}))
    pdf_bytes = export_artifacts.get("pdf_bytes")
    csv_bytes = export_artifacts.get("csv_bytes")
    json_bytes = export_artifacts.get("json_bytes")
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown(f"### {title}")
    st.caption(body)

    if not compact:
        status_cols = st.columns(3)
        status_cols[0].metric("PDF Report", "Ready" if pdf_bytes is not None else "Issue")
        status_cols[1].metric("CSV Export", "Ready" if csv_bytes is not None else "Issue")
        status_cols[2].metric("JSON Export", "Ready" if json_bytes is not None else "Issue")

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes or b"",
            file_name=f"portfolio_report_{timestamp_suffix}.pdf",
            mime="application/pdf",
            disabled=pdf_bytes is None,
            help=errors.get("pdf", "Charts, score, simulation, and commentary in one document."),
            use_container_width=True,
            key=f"{key_namespace}_download_pdf",
        )
    with export_col2:
        st.download_button(
            "Download Data CSV",
            data=csv_bytes or b"",
            file_name=f"portfolio_data_{timestamp_suffix}.csv",
            mime="text/csv",
            disabled=csv_bytes is None,
            help=errors.get("csv", "Portfolio data export for spreadsheet work."),
            use_container_width=True,
            key=f"{key_namespace}_download_csv",
        )
    with export_col3:
        st.download_button(
            "Download Full JSON",
            data=json_bytes or b"",
            file_name=f"portfolio_full_report_{timestamp_suffix}.json",
            mime="application/json",
            disabled=json_bytes is None,
            help=errors.get("json", "Complete analysis payload for automation or archiving."),
            use_container_width=True,
            key=f"{key_namespace}_download_json",
        )

    if errors:
        with st.expander("Export diagnostics", expanded=not compact):
            if "pdf" in errors:
                st.warning(f"PDF export issue: {errors['pdf']}")
            if "csv" in errors:
                st.warning(f"CSV export issue: {errors['csv']}")
            if "json" in errors:
                st.warning(f"JSON export issue: {errors['json']}")


def _compute_analysis(
    tickers: List[str],
    weights_pct: np.ndarray,
    start_date: date,
    end_date: date,
    risk_free_rate: float,
    risk_profile: str,
    n_simulations: int,
    horizon_days: int,
    portfolio_samples: int,
) -> Dict[str, Any]:
    prices = fetch_market_data_cached(tuple(tickers), start_date, end_date)
    if prices.empty:
        raise ValueError("No market data fetched. Please check tickers and date range.")

    aligned_prices, weights, missing_tickers, alignment_warnings = _align_prices_and_weights(
        prices=prices,
        tickers=tickers,
        weights_pct=weights_pct,
    )

    returns = aligned_prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Not enough historical data after cleaning to compute returns.")
    if returns.shape[1] < 2:
        raise ValueError("At least 2 assets with valid data are required.")

    portfolio_returns = calculate_portfolio_daily_returns(returns, weights)
    core_metrics = calculate_portfolio_core_metrics(portfolio_returns, risk_free_rate=risk_free_rate)
    corr_matrix = calculate_correlation_matrix(returns)
    concentration = calculate_concentration_metrics(weights)
    avg_corr = calculate_average_correlation(corr_matrix)

    metrics = {
        **core_metrics,
        **concentration,
        "avg_correlation": avg_corr,
    }

    advanced_models = run_advanced_models(
        returns=portfolio_returns,
        forecast_periods=min(10, max(3, horizon_days // 63)),
        returns_df=returns,
    )
    model_signals = _model_signals_from_outputs(advanced_models)

    score_result = evaluate_portfolio_score(
        metrics=metrics,
        concentration=concentration,
        avg_correlation=avg_corr,
        n_assets=returns.shape[1],
        risk_profile=risk_profile,
        model_signals=model_signals,
    )
    fallback_review = build_deterministic_fallback_review(score_result, metrics)

    ai_payload = _build_ai_payload(
        tickers=returns.columns.tolist(),
        weights=weights,
        metrics=metrics,
        score_result=score_result,
        flags=score_result["flags"],
        risk_profile=risk_profile,
        horizon_days=horizon_days,
    )

    try:
        streamlit_secrets = st.secrets
    except Exception:
        streamlit_secrets = None

    api_key = resolve_groq_api_key(streamlit_secrets)
    news_api_key = ""
    if streamlit_secrets is not None:
        try:
            news_api_key = str(
                streamlit_secrets.get("NEWSAPI_KEY")
                or streamlit_secrets.get("NEWS_API_KEY")
                or ""
            ).strip()
        except Exception:
            news_api_key = ""
    ai_review = generate_ai_review(ai_payload, api_key=api_key)
    ai_messages: List[str] = []
    if not ai_review.get("available", False):
        error_detail = ai_review.get("error", "AI service unavailable.")
        ai_messages.append(f"AI fallback active: {error_detail}")
        ai_review = {
            **fallback_review,
            "available": False,
            "source_detail": error_detail,
        }
    else:
        ai_review["available"] = True

    expected_return = float(np.clip(metrics["annualized_return"], -0.90, 2.5))
    volatility = float(max(metrics["volatility"], 1e-6))
    price_paths, simulation_stats = run_monte_carlo_simulation(
        current_value=100000.0,
        expected_return=expected_return,
        volatility=volatility,
        time_horizon=horizon_days,
        n_simulations=n_simulations,
    )
    simulation_percentiles = _create_simulation_percentiles(price_paths)

    min_var_result = optimize_minimum_variance(returns)
    max_sharpe_result = optimize_maximum_sharpe(returns, risk_free_rate=risk_free_rate)
    frontier = calculate_efficient_frontier(returns, n_points=30)

    portfolio_cloud = sample_portfolio_cloud(
        returns,
        n_samples=portfolio_samples,
        risk_free_rate=risk_free_rate,
    )

    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252

    def build_portfolio_marker(
        name: str,
        marker_weights: np.ndarray,
        expected_return_value: float,
        volatility_value: float,
        sharpe_ratio: float,
    ) -> Dict[str, Any]:
        marker_metrics = calculate_portfolio_statistics(
            weights=marker_weights,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=risk_free_rate,
            symbols=returns.columns.tolist(),
        )
        return {
            "name": name,
            "expected_return": expected_return_value,
            "volatility": volatility_value,
            "sharpe_ratio": sharpe_ratio,
            "diversification_score": marker_metrics["diversification_score"],
            "effective_holdings": marker_metrics["effective_holdings"],
            "max_weight": marker_metrics["max_weight"],
            "top_holdings": marker_metrics["top_holdings"],
        }

    highlighted_portfolios: List[Dict[str, Any]] = []

    user_stats = calculate_portfolio_statistics(
        weights=weights,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        symbols=returns.columns.tolist(),
    )
    highlighted_portfolios.append(
        build_portfolio_marker(
            "User Portfolio",
            marker_weights=weights,
            expected_return_value=user_stats["return"],
            volatility_value=user_stats["volatility"],
            sharpe_ratio=user_stats["sharpe_ratio"],
        )
    )

    if min_var_result.get("success"):
        highlighted_portfolios.append(
            build_portfolio_marker(
                "Min Variance",
                marker_weights=np.array(min_var_result["weights"]),
                expected_return_value=float(min_var_result["expected_return"]),
                volatility_value=float(min_var_result["volatility"]),
                sharpe_ratio=float(min_var_result["sharpe_ratio"]),
            )
        )

    if max_sharpe_result.get("success"):
        highlighted_portfolios.append(
            build_portfolio_marker(
                "Max Sharpe",
                marker_weights=np.array(max_sharpe_result["weights"]),
                expected_return_value=float(max_sharpe_result["expected_return"]),
                volatility_value=float(max_sharpe_result["volatility"]),
                sharpe_ratio=float(max_sharpe_result["sharpe_ratio"]),
            )
        )

    asset_metrics_data = []
    for symbol in returns.columns:
        asset_returns = returns[symbol]
        asset_metrics_data.append({
            "Symbol": symbol,
            "Ann. Return": calculate_annualized_return(asset_returns),
            "Volatility": calculate_volatility(asset_returns),
            "Sharpe": calculate_sharpe_ratio(asset_returns, risk_free_rate),
            "Max DD": calculate_max_drawdown(asset_returns),
        })
    asset_metrics_df = pd.DataFrame(asset_metrics_data)

    portfolio_timeseries = build_portfolio_timeseries(portfolio_returns, initial_value=100.0)

    quant_stack = run_quant_stack(
        portfolio_returns=portfolio_returns,
        returns_df=returns,
        config={
            "tickers": returns.columns.tolist(),
            "weights": [float(value) for value in weights],
            "start_date": start_date,
            "end_date": end_date,
            "risk_profile": risk_profile,
            "risk_free_rate": risk_free_rate,
            "horizon_days": horizon_days,
            "portfolio_metrics": metrics,
            "news_max_items": 120,
            "news_api_key": news_api_key,
            "sector_keywords": ["macro", "rates", "inflation", "earnings", "volatility"],
        },
    )

    return {
        "tickers": returns.columns.tolist(),
        "weights": weights,
        "start_date": start_date,
        "end_date": end_date,
        "horizon_days": horizon_days,
        "risk_profile": risk_profile,
        "risk_free_rate": risk_free_rate,
        "prices": aligned_prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "portfolio_timeseries": portfolio_timeseries,
        "metrics": metrics,
        "score_result": score_result,
        "ai_review": ai_review,
        "ai_messages": ai_messages,
        "correlation_matrix": corr_matrix,
        "min_var_result": min_var_result,
        "max_sharpe_result": max_sharpe_result,
        "frontier": frontier,
        "portfolio_cloud": portfolio_cloud,
        "highlighted_portfolios": highlighted_portfolios,
        "price_paths": price_paths,
        "simulation_stats": simulation_stats,
        "simulation_percentiles": simulation_percentiles,
        "asset_metrics_df": asset_metrics_df,
        "advanced_models": advanced_models,
        "model_results": quant_stack["models"],
        "signal_results": quant_stack["signals"],
        "summary_result": quant_stack["summary"],
        "news_result": quant_stack["news"],
        "backtest_result": quant_stack["backtest"],
        "run_record": quant_stack["run_record"],
        "history_path": quant_stack["history_path"],
        "history_records": list_run_records(limit=40),
        "warnings": alignment_warnings,
        "missing_tickers": missing_tickers,
    }


SCREENER_DISPLAY_COLUMNS = [
    ("Rank", "Rank"),
    ("Ticker", "Ticker"),
    ("Company", "Company"),
    ("Sector", "Sector"),
    ("Industry", "Industry"),
    ("Exchange", "Exchange"),
    ("MarketCap", "MarketCap"),
    ("Beta", "Beta"),
    ("PE", "P/E"),
    ("ForwardPE", "Forward P/E"),
    ("PEG", "PEG"),
    ("ROE", "ROE"),
    ("ROA", "ROA"),
    ("DividendYield", "DividendYield"),
    ("Return52W", "52w Return"),
    ("RSI", "RSI"),
    ("MACD", "MACD"),
    ("Volatility", "Volatility"),
    ("Drawdown", "Drawdown"),
    ("QuantScore", "QuantScore"),
    ("AvgVolume", "AvgVolume"),
    ("Price", "Price"),
]


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_ticker_history_cached(symbol: str, period: str = "1y") -> pd.DataFrame:
    history = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if isinstance(history.columns, pd.MultiIndex):
        level_0 = history.columns.get_level_values(0)
        level_1 = history.columns.get_level_values(1)
        if symbol in level_1:
            history = history.xs(symbol, axis=1, level=1, drop_level=True)
        elif symbol in level_0:
            history = history[symbol]
    return history


def _dataframe_to_excel_bytes(dataframe: pd.DataFrame) -> bytes | None:
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Screener")
        output.seek(0)
        return output.getvalue()
    except Exception:
        return None


def _numeric_bounds(series: pd.Series, default_min: float, default_max: float) -> Tuple[float, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float(default_min), float(default_max)

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return float(default_min), float(default_max)

    if min_value == max_value:
        max_value = min_value + max(1.0, abs(min_value) * 0.05)
    return min_value, max_value


def _rounded_slider_range(
    minimum: float,
    maximum: float,
    precision: int = 2,
    floor_at_zero: bool = False,
) -> tuple[float, float]:
    low = round(float(minimum), precision)
    high = round(float(maximum), precision)
    if floor_at_zero:
        low = max(0.0, low)
        high = max(low + (10 ** -precision), high)
    if low >= high:
        high = low + max(10 ** -precision, abs(low) * 0.05, 1.0 if floor_at_zero else 0.1)
        high = round(high, precision)
    return low, high


def _portfolio_positions_dataframe(portfolio: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in portfolio.get("positions", []):
        if not isinstance(item, dict):
            continue
        rows.append({
            "Ticker": str(item.get("ticker", "")).upper(),
            "Shares": float(item.get("shares", 0.0) or 0.0),
            "CostBasis": float(item.get("cost_basis")) if item.get("cost_basis") not in (None, "") else np.nan,
            "TargetWeight": float(item.get("target_weight")) if item.get("target_weight") not in (None, "") else np.nan,
            "Notes": str(item.get("notes", "") or ""),
        })
    return pd.DataFrame(rows)


def _positions_from_editor(edited_df: pd.DataFrame) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    if edited_df.empty:
        return positions

    for _, row in edited_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        shares_raw = pd.to_numeric(row.get("Shares"), errors="coerce")
        shares = 0.0 if pd.isna(shares_raw) else float(shares_raw)
        if shares == 0.0:
            continue

        cost_basis_raw = pd.to_numeric(row.get("CostBasis"), errors="coerce")
        target_weight_raw = pd.to_numeric(row.get("TargetWeight"), errors="coerce")

        positions.append({
            "ticker": ticker,
            "shares": shares,
            "cost_basis": (None if pd.isna(cost_basis_raw) else float(cost_basis_raw)),
            "target_weight": (None if pd.isna(target_weight_raw) else float(target_weight_raw)),
            "notes": str(row.get("Notes", "") or "").strip(),
        })
    return positions


def _invalidate_portfolio_live_snapshot() -> None:
    st.session_state.pop("portfolio_live_holdings", None)
    st.session_state.pop("portfolio_live_summary", None)


def _portfolio_cost_summary(portfolio: Dict[str, Any]) -> Dict[str, float]:
    positions_df = _portfolio_positions_dataframe(portfolio)
    if positions_df.empty:
        return {"count": 0.0, "cost_value": 0.0}

    cost_value = (
        pd.to_numeric(positions_df["Shares"], errors="coerce")
        * pd.to_numeric(positions_df["CostBasis"], errors="coerce")
    ).sum(min_count=1)
    return {
        "count": float(len(positions_df)),
        "cost_value": float(cost_value if pd.notna(cost_value) else 0.0),
    }


def _render_sidebar_portfolio_summary() -> None:
    portfolio = st.session_state.get("current_portfolio")
    if not isinstance(portfolio, dict):
        return

    summary = _portfolio_cost_summary(portfolio)
    if summary["count"] <= 0:
        return

    st.markdown("---")
    st.caption("Portfolio Tracker Snapshot")
    c1, c2 = st.columns(2)
    c1.metric("Holdings", f"{int(summary['count'])}")
    c2.metric("Cost Value", f"${summary['cost_value']:,.0f}")


def _is_universe_snapshot_stale(metadata: Dict[str, Any], max_age_hours: int) -> bool:
    last_refresh_raw = str(metadata.get("last_refresh", "")).strip()
    if not last_refresh_raw:
        return True

    try:
        last_refresh = datetime.fromisoformat(last_refresh_raw.replace("Z", "+00:00"))
    except Exception:
        return True

    now_utc = datetime.now().astimezone(last_refresh.tzinfo)
    age_seconds = (now_utc - last_refresh).total_seconds()
    return age_seconds > float(max_age_hours) * 3600.0


def _extract_filter_options(series: pd.Series) -> list[str]:
    normalized = series.fillna("").astype(str).str.strip()
    known = sorted([value for value in normalized.unique().tolist() if value and value.lower() not in {"nan", "none"}])
    has_unknown = bool((normalized == "").any())
    if has_unknown or not known:
        known.append("Unknown")
    return known


def _is_default_range(
    selected: tuple[float, float],
    full_range: tuple[float, float],
    tolerance: float = 1e-8,
) -> bool:
    return (
        abs(float(selected[0]) - float(full_range[0])) <= tolerance * max(1.0, abs(float(full_range[0])))
        and abs(float(selected[1]) - float(full_range[1])) <= tolerance * max(1.0, abs(float(full_range[1])))
    )


def _parse_targets_input(raw_text: str) -> list[float]:
    if not raw_text.strip():
        return []
    values: list[float] = []
    for item in raw_text.replace(";", ",").split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if pd.notna(parsed) and float(parsed) > 0:
            values.append(float(parsed))
    return values


def _format_compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"

    numeric = float(value)
    abs_value = abs(numeric)
    if abs_value >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{numeric / 1_000:.1f}K"
    return f"{numeric:.0f}"


def _coverage_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.notna().mean())


def _build_screenable_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    if universe_df.empty:
        return universe_df.copy()

    output = universe_df.copy()
    source_series = output["Source"].fillna("").astype(str).str.strip()
    price_series = pd.to_numeric(output["Price"], errors="coerce")
    sec_only_mask = source_series.eq("sec_company_tickers")

    core = output.loc[~(sec_only_mask & price_series.isna())].copy()
    return core.reset_index(drop=True)


def _build_universe_health_rows(universe_df: pd.DataFrame) -> pd.DataFrame:
    health_specs = [
        ("Price", "Live price snapshot"),
        ("AvgVolume", "Liquidity coverage"),
        ("MarketCap", "Size coverage"),
        ("Sector", "Classification"),
        ("Industry", "Sub-industry classification"),
        ("PE", "Trailing valuation"),
        ("ForwardPE", "Forward valuation"),
        ("ROE", "Quality"),
        ("RevenueGrowth", "Growth"),
        ("DividendYield", "Income"),
        ("Return52W", "Momentum"),
    ]
    rows: list[dict[str, Any]] = []
    total_rows = max(1, len(universe_df))
    for column, label in health_specs:
        if column not in universe_df.columns:
            continue
        non_null = int(universe_df[column].notna().sum())
        coverage = non_null / float(total_rows)
        rows.append({
            "Field": column,
            "Purpose": label,
            "Coverage": f"{coverage:.1%}",
            "Filled Rows": non_null,
        })
    return pd.DataFrame(rows)


def _reload_stock_picker_modules() -> Tuple[Any, Any]:
    importlib.reload(universe_sources_module)
    importlib.reload(universe_enrichment_module)
    refreshed_universe_module = importlib.reload(stock_universe_module)
    refreshed_screener_module = importlib.reload(stock_screener_module)
    return refreshed_universe_module, refreshed_screener_module


def _render_universe_overview(
    universe_df: pd.DataFrame,
    screenable_universe_df: pd.DataFrame,
    metadata: Dict[str, Any],
    snapshot_stale: bool,
) -> None:
    st.markdown("### Universe Health")

    raw_symbols_total = int(len(universe_df))
    symbols_total = int(len(screenable_universe_df))
    excluded_count = max(0, raw_symbols_total - symbols_total)
    priced_count = int(screenable_universe_df["Price"].notna().sum()) if "Price" in screenable_universe_df.columns else 0
    sector_count = int(screenable_universe_df["Sector"].notna().sum()) if "Sector" in screenable_universe_df.columns else 0
    fundamentals_mask = pd.Series(False, index=screenable_universe_df.index)
    for column in ["PE", "ForwardPE", "ROE", "RevenueGrowth", "DividendYield"]:
        if column in screenable_universe_df.columns:
            fundamentals_mask = fundamentals_mask | screenable_universe_df[column].notna()
    fundamentals_count = int(fundamentals_mask.sum()) if len(screenable_universe_df) else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe Size", f"{symbols_total:,}")
    m2.metric("Price Coverage", f"{_coverage_ratio(screenable_universe_df['Price']) if 'Price' in screenable_universe_df.columns else 0.0:.1%}", f"{priced_count:,} priced")
    m3.metric("Fundamental Coverage", f"{(fundamentals_count / max(1, symbols_total)):.1%}", f"{fundamentals_count:,} with key fields")
    m4.metric("Sector Coverage", f"{(sector_count / max(1, symbols_total)):.1%}", f"{sector_count:,} classified")

    left_col, right_col = st.columns([1.35, 1.65])
    with left_col:
        if excluded_count > 0:
            st.caption(
                f"Raw candidates: {raw_symbols_total:,}. "
                f"Screenable core: {symbols_total:,}. "
                f"Excluded SEC-only symbols without price: {excluded_count:,}."
            )
        health_rows = _build_universe_health_rows(screenable_universe_df)
        if not health_rows.empty:
            st.dataframe(health_rows, use_container_width=True, hide_index=True)
        status = str(metadata.get("status", "unknown")).strip().lower()
        last_refresh = str(metadata.get("last_refresh", "")).strip()
        if last_refresh:
            st.caption(f"Last refresh (UTC): {last_refresh}")
        if snapshot_stale:
            st.warning("Snapshot is stale. You can still screen on cached data, but a refresh is recommended.")
        elif status == "fallback":
            st.warning("Last refresh fell back to cached data. Coverage may be partial.")
        else:
            st.caption("Snapshot looks ready for screening.")

    with right_col:
        exchange_counts = (
            screenable_universe_df["Exchange"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").value_counts().head(10)
            if "Exchange" in screenable_universe_df.columns
            else pd.Series(dtype=int)
        )
        if not exchange_counts.empty:
            st.caption("Exchange mix")
            st.bar_chart(exchange_counts)

        fallback_reason = str(metadata.get("fallback_reason", "")).strip()
        if fallback_reason:
            st.caption(f"Refresh diagnostic: {fallback_reason}")


def _render_selectable_results(results: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if results.empty:
        st.info("No matching stocks found with the current filters.")
        return pd.DataFrame()

    view = results.copy()
    for internal_name, _ in SCREENER_DISPLAY_COLUMNS:
        if internal_name not in view.columns:
            view[internal_name] = np.nan

    max_rows_cap = min(1000, max(1, len(view)))
    max_rows = st.slider(
        "Rows to display",
        min_value=1,
        max_value=max_rows_cap,
        value=min(250, max_rows_cap),
        key=f"{key_prefix}_rows_to_display",
    )
    display_view = pd.DataFrame({
        display_name: view[internal_name]
        for internal_name, display_name in SCREENER_DISPLAY_COLUMNS
    }).head(max_rows).copy()
    display_view.insert(0, "Select", False)

    edited = st.data_editor(
        display_view,
        use_container_width=True,
        hide_index=True,
        key=f"{key_prefix}_table_editor",
        disabled=[column for column in display_view.columns if column != "Select"],
        num_rows="fixed",
    )
    selected_tickers = (
        edited.loc[edited["Select"], "Ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .drop_duplicates()
        .tolist()
    )
    if not selected_tickers:
        return pd.DataFrame(columns=results.columns)
    return results[results["Ticker"].astype(str).str.upper().isin(selected_tickers)].copy()


def _render_screener_bulk_actions(
    selected_rows: pd.DataFrame,
    full_results: pd.DataFrame,
    key_prefix: str,
) -> None:
    st.markdown("### Bulk Actions")
    s1, s2, s3 = st.columns(3)
    s1.metric("Selected", int(len(selected_rows)))
    s2.metric("Result Set", int(len(full_results)))
    median_market_cap = (
        _format_compact_number(pd.to_numeric(selected_rows.get("MarketCap"), errors="coerce").median())
        if not selected_rows.empty and "MarketCap" in selected_rows.columns
        else "-"
    )
    s3.metric("Median Market Cap", median_market_cap)

    action_tabs = st.tabs(["Portfolio", "Export", "Inspect"])

    with action_tabs[0]:
        shares_to_add = st.number_input(
            "Shares per selected ticker",
            min_value=0.01,
            value=1.0,
            step=0.25,
            key=f"{key_prefix}_shares_to_add",
        )
        add_clicked = st.button(
            "Add selected to Portfolio",
            key=f"{key_prefix}_add_to_portfolio",
            disabled=selected_rows.empty,
            use_container_width=True,
        )
        if add_clicked and not selected_rows.empty:
            portfolio = dict(st.session_state.get("current_portfolio", load_portfolio("default")))
            for _, row in selected_rows.iterrows():
                ticker = str(row.get("Ticker", "")).strip().upper()
                if not ticker:
                    continue
                price_raw = pd.to_numeric(row.get("Price"), errors="coerce")
                cost_basis = None if pd.isna(price_raw) else float(price_raw)
                portfolio = add_position(
                    portfolio,
                    ticker=ticker,
                    shares=float(shares_to_add),
                    cost_basis=cost_basis,
                )
            st.session_state["current_portfolio"] = portfolio
            _invalidate_portfolio_live_snapshot()
            st.success(f"Added {len(selected_rows)} symbols into the current portfolio.")
        if selected_rows.empty:
            st.caption("Select rows in the result table to push them into the portfolio tracker.")

    with action_tabs[1]:
        export_source = st.radio(
            "Export scope",
            options=["Selected rows", "All results"],
            horizontal=True,
            key=f"{key_prefix}_export_scope",
        )
        export_df = selected_rows if (export_source == "Selected rows" and not selected_rows.empty) else full_results
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        excel_bytes = _dataframe_to_excel_bytes(export_df)

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "Export CSV",
                data=csv_bytes,
                file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"{key_prefix}_download_csv",
            )
        with e2:
            if excel_bytes is None:
                st.info("Excel export unavailable (openpyxl missing).")
            else:
                st.download_button(
                    "Export Excel",
                    data=excel_bytes,
                    file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"{key_prefix}_download_excel",
                )

    with action_tabs[2]:
        if selected_rows.empty:
            st.info("Select at least one row to inspect a candidate in more detail.")
        else:
            selected_tickers = selected_rows["Ticker"].dropna().astype(str).str.upper().tolist()
            selected_ticker = st.selectbox(
                "Inspect selected ticker",
                options=selected_tickers,
                key=f"{key_prefix}_inspect_ticker",
            )
            history = _fetch_ticker_history_cached(selected_ticker)
            if history.empty:
                st.warning(f"Could not load historical data for {selected_ticker}.")
            else:
                close_column = "Adj Close" if "Adj Close" in history.columns else "Close"
                if close_column not in history.columns:
                    st.warning(f"Price column not available for {selected_ticker}.")
                else:
                    close_prices = pd.to_numeric(history[close_column], errors="coerce").dropna()
                    returns = close_prices.pct_change().dropna()
                    selected_view = selected_rows[
                        selected_rows["Ticker"].astype(str).str.upper() == selected_ticker
                    ].head(1)

                    if not selected_view.empty:
                        record = selected_view.iloc[0]
                        price_value = pd.to_numeric(record.get("Price"), errors="coerce")
                        market_cap_value = pd.to_numeric(record.get("MarketCap"), errors="coerce")
                        return_52w_value = pd.to_numeric(record.get("Return52W"), errors="coerce")
                        quant_score_value = pd.to_numeric(record.get("QuantScore"), errors="coerce")
                        i1, i2, i3, i4 = st.columns(4)
                        i1.metric("Price", f"${float(price_value):.2f}" if pd.notna(price_value) else "-")
                        i2.metric("Market Cap", _format_compact_number(market_cap_value))
                        i3.metric("52w Return", f"{float(return_52w_value):.2%}" if pd.notna(return_52w_value) else "-")
                        i4.metric("Quant Score", f"{float(quant_score_value):.1f}" if pd.notna(quant_score_value) else "-")
                        st.caption(
                            " | ".join(
                                [
                                    str(record.get("Company", "") or selected_ticker),
                                    str(record.get("Sector", "") or "Unknown sector"),
                                    str(record.get("Industry", "") or "Unknown industry"),
                                ]
                            )
                        )

                    if returns.empty:
                        st.warning(f"Insufficient data for quick analysis of {selected_ticker}.")
                    else:
                        q1, q2, q3, q4 = st.columns(4)
                        q1.metric("Annualized Return", f"{calculate_annualized_return(returns):.2%}")
                        q2.metric("Volatility", f"{calculate_volatility(returns):.2%}")
                        q3.metric("Sharpe", f"{calculate_sharpe_ratio(returns):.3f}")
                        q4.metric("Max Drawdown", f"{calculate_max_drawdown(returns):.2%}")
                    st.line_chart(close_prices.rename(selected_ticker))


def _render_stock_picker_tab() -> None:
    st.subheader("Stock Screener")
    st.caption(
        "Daily cached equity universe with staged filtering: broad cached filters first, "
        "technical enrichment only for the shortlist."
    )
    universe_api = stock_universe_module

    refresh_col, age_col, auto_col, status_col = st.columns([1, 1, 1.1, 2])
    with age_col:
        max_age_hours = st.number_input(
            "Universe max age (hours)",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
            key="stock_picker_max_age_hours",
        )
    with auto_col:
        auto_refresh_stale = st.toggle(
            "Auto-refresh stale snapshot",
            value=False,
            key="stock_picker_auto_refresh_stale",
            help="When enabled, stale snapshots are rebuilt automatically before screening.",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Refresh Universe",
            key="stock_picker_refresh_universe",
            use_container_width=True,
        )

    if refresh_clicked:
        try:
            universe_api, _ = _reload_stock_picker_modules()
        except Exception:
            universe_api = stock_universe_module

        universe_api.get_universe.clear()
        universe_api.load_universe_snapshot.clear()
        universe_api.load_universe_metadata.clear()
        progress_bar = st.empty().progress(0.0, text="Starting universe refresh...")

        def _on_universe_progress(event: Dict[str, Any]) -> None:
            progress_value = float(event.get("progress", 0.0) or 0.0)
            message = str(event.get("message", "Refreshing universe..."))
            current = event.get("current")
            total = event.get("total")
            if current is not None and total is not None and int(total) > 0:
                message = f"{message} ({int(current)}/{int(total)})"
            progress_bar.progress(max(0.0, min(1.0, progress_value)), text=message)

        try:
            universe_df = universe_api.refresh_universe_if_stale(
                max_age_hours=int(max_age_hours),
                force_refresh=True,
                progress_callback=_on_universe_progress,
            )
        except TypeError:
            try:
                progress_bar.progress(0.05, text="Reloading universe pipeline for compatibility...")
                universe_api, _ = _reload_stock_picker_modules()
                refreshed_fn = universe_api.refresh_universe_if_stale
                signature = inspect.signature(refreshed_fn)
                if "progress_callback" in signature.parameters:
                    universe_df = refreshed_fn(
                        max_age_hours=int(max_age_hours),
                        force_refresh=True,
                        progress_callback=_on_universe_progress,
                    )
                else:
                    progress_bar.progress(
                        0.1,
                        text="Compatibility mode: refreshing without live progress details...",
                    )
                    universe_df = refreshed_fn(
                        max_age_hours=int(max_age_hours),
                        force_refresh=True,
                    )
            except Exception:
                progress_bar.progress(
                    0.1,
                    text="Compatibility mode: refreshing without live progress details...",
                )
                universe_df = universe_api.refresh_universe_if_stale(
                    max_age_hours=int(max_age_hours),
                    force_refresh=True,
                )

        universe_api.load_universe_snapshot.clear()
        universe_api.load_universe_metadata.clear()
        st.session_state.pop("stock_picker_results_classic", None)
        st.session_state.pop("stock_picker_results_ai", None)
        st.session_state.pop("stock_picker_classic_info", None)
        st.session_state.pop("stock_picker_ai_explanation", None)
        st.session_state.pop("stock_picker_ai_parsed", None)

        refresh_metadata = universe_api.load_universe_metadata()
        refresh_failed = bool(universe_df.empty) or str(refresh_metadata.get("status", "")).lower() == "fallback"
        if refresh_failed:
            progress_bar.progress(1.0, text="Universe refresh finished with fallback")
            fallback_reason = str(refresh_metadata.get("fallback_reason", "")).strip()
            if fallback_reason:
                st.error(f"Universe refresh failed: {fallback_reason}")
        else:
            progress_bar.progress(1.0, text="Universe refresh completed")
    else:
        universe_df = universe_api.load_universe_snapshot()

    metadata = universe_api.load_universe_metadata()
    snapshot_stale = _is_universe_snapshot_stale(metadata, int(max_age_hours))
    if auto_refresh_stale and snapshot_stale and not refresh_clicked:
        with st.spinner("Snapshot is stale, rebuilding universe..."):
            universe_df = universe_api.refresh_universe_if_stale(
                max_age_hours=int(max_age_hours),
                force_refresh=False,
            )
        universe_api.load_universe_snapshot.clear()
        universe_api.load_universe_metadata.clear()
        metadata = universe_api.load_universe_metadata()
        snapshot_stale = _is_universe_snapshot_stale(metadata, int(max_age_hours))

    with status_col:
        st.write(
            f"Snapshot status: **{metadata.get('status', 'unknown')}** | "
            f"Rows: **{metadata.get('rows', len(universe_df))}**"
        )
        last_refresh = str(metadata.get("last_refresh", ""))
        if last_refresh:
            st.caption(f"Last refresh (UTC): {last_refresh}")
        if snapshot_stale:
            st.caption("Working from a stale snapshot is allowed, but results may lag current listings.")

    if universe_df.empty:
        st.warning("Universe snapshot is empty. Use 'Refresh Universe' to build it.")
        return

    screenable_universe_df = _build_screenable_universe(universe_df)
    _render_universe_overview(universe_df, screenable_universe_df, metadata, snapshot_stale)
    screener_tabs = st.tabs(["Classic Screen", "AI Query"])

    with screener_tabs[0]:
        filter_col, result_col = st.columns([1.15, 1.85])
        with filter_col:
            st.markdown("### Filter Builder")
            st.caption("Build the broad cached filter first, then rank and enrich only the shortlist.")

            market_cap_series = pd.to_numeric(screenable_universe_df["MarketCap"], errors="coerce") / 1e9
            beta_series = pd.to_numeric(screenable_universe_df["Beta"], errors="coerce")
            price_series = pd.to_numeric(screenable_universe_df["Price"], errors="coerce")

            mcap_min, mcap_max = _numeric_bounds(market_cap_series, default_min=0.0, default_max=5000.0)
            beta_min, beta_max = _numeric_bounds(beta_series, default_min=-1.0, default_max=5.0)
            price_min, price_max = _numeric_bounds(price_series, default_min=0.5, default_max=2000.0)

            sectors = _extract_filter_options(screenable_universe_df["Sector"])
            industries = _extract_filter_options(screenable_universe_df["Industry"])
            exchanges = _extract_filter_options(screenable_universe_df["Exchange"])

            with st.form("classic_screener_form", clear_on_submit=False):
                builder_tabs = st.tabs(["Universe", "Fundamentals", "Ranking"])

                with builder_tabs[0]:
                    full_market_cap_range_b = _rounded_slider_range(
                        minimum=mcap_min,
                        maximum=max(mcap_max, max(1.0, mcap_min + 1.0)),
                        precision=2,
                        floor_at_zero=True,
                    )
                    market_cap_range_b = st.slider(
                        "Market Cap range ($B)",
                        min_value=full_market_cap_range_b[0],
                        max_value=full_market_cap_range_b[1],
                        value=full_market_cap_range_b,
                        step=0.25,
                    )
                    full_beta_range = _rounded_slider_range(
                        minimum=beta_min,
                        maximum=beta_max,
                        precision=2,
                    )
                    beta_range = st.slider(
                        "Beta range",
                        min_value=full_beta_range[0],
                        max_value=full_beta_range[1],
                        value=full_beta_range,
                        step=0.05,
                    )
                    full_price_range = _rounded_slider_range(
                        minimum=price_min,
                        maximum=max(price_max, max(1.0, price_min + 1.0)),
                        precision=2,
                        floor_at_zero=True,
                    )
                    price_range = st.slider(
                        "Price range ($)",
                        min_value=full_price_range[0],
                        max_value=full_price_range[1],
                        value=full_price_range,
                        step=0.25,
                    )
                    min_avg_volume = st.number_input(
                        "Minimum average volume",
                        min_value=0.0,
                        value=0.0,
                        step=50_000.0,
                    )
                    selected_exchanges = st.multiselect("Exchanges", exchanges)
                    selected_sectors = st.multiselect("Sectors", sectors)
                    selected_industries = st.multiselect("Industries", industries)
                    liquidity_prefilter = st.checkbox("Enable liquidity prefilter", value=False)

                with builder_tabs[1]:
                    f1, f2 = st.columns(2)
                    with f1:
                        use_valuation = st.checkbox("Apply valuation filters", value=False)
                        pe_max = st.number_input("Max P/E", min_value=0.0, value=35.0, step=1.0)
                        forward_pe_max = st.number_input("Max Forward P/E", min_value=0.0, value=35.0, step=1.0)
                        peg_max = st.number_input("Max PEG", min_value=0.0, value=3.0, step=0.1)

                        use_growth = st.checkbox("Apply growth filters", value=False)
                        revenue_growth_min = st.number_input("Min Revenue Growth", value=0.05, step=0.01, format="%.3f")
                        earnings_growth_min = st.number_input("Min Earnings Growth", value=0.05, step=0.01, format="%.3f")

                    with f2:
                        use_quality = st.checkbox("Apply quality filters", value=False)
                        roe_min = st.number_input("Min ROE", value=0.05, step=0.01, format="%.3f")
                        roa_min = st.number_input("Min ROA", value=0.02, step=0.01, format="%.3f")

                        use_momentum = st.checkbox("Apply momentum filters", value=False)
                        return_52w_min = st.number_input("Min 52w Return", value=0.0, step=0.02, format="%.3f")
                        use_dividend = st.checkbox("Apply dividend filters", value=False)
                        dividend_yield_min = st.number_input("Min Dividend Yield", value=0.0, step=0.005, format="%.4f")

                with builder_tabs[2]:
                    r1, r2 = st.columns(2)
                    with r1:
                        value_weight = st.slider("Value weight", 0.0, 3.0, 1.0, 0.1)
                        growth_weight = st.slider("Growth weight", 0.0, 3.0, 1.0, 0.1)
                        quality_weight = st.slider("Quality weight", 0.0, 3.0, 1.0, 0.1)
                    with r2:
                        momentum_weight = st.slider("Momentum weight", 0.0, 3.0, 1.0, 0.1)
                        stability_weight = st.slider("Stability weight", 0.0, 3.0, 1.0, 0.1)
                        dividend_weight = st.slider("Dividend weight", 0.0, 3.0, 0.5, 0.1)
                    technical_limit = st.slider("Technical indicator shortlist size", 25, 400, 150, 25)
                    st.caption("Technical indicators are computed only for the top-ranked shortlist.")

                run_classic = st.form_submit_button(
                    "Run Classic Screen",
                    type="primary",
                    use_container_width=True,
                )

            if run_classic:
                try:
                    screener_api = importlib.reload(stock_screener_module)
                except Exception:
                    screener_api = stock_screener_module

                valuation_filters = {}
                if use_valuation:
                    valuation_filters = {
                        "PE": (None, float(pe_max)),
                        "ForwardPE": (None, float(forward_pe_max)),
                        "PEG": (None, float(peg_max)),
                    }

                growth_filters = {}
                if use_growth:
                    growth_filters = {
                        "RevenueGrowth": (float(revenue_growth_min), None),
                        "EarningsGrowth": (float(earnings_growth_min), None),
                    }

                quality_filters = {}
                if use_quality:
                    quality_filters = {
                        "ROE": (float(roe_min), None),
                        "ROA": (float(roa_min), None),
                    }

                momentum_filters = {}
                if use_momentum:
                    momentum_filters = {"Return52W": (float(return_52w_min), None)}

                dividend_filters = {}
                if use_dividend:
                    dividend_filters = {"DividendYield": (float(dividend_yield_min), None)}

                active_market_cap_range = (
                    None
                    if _is_default_range(
                        (float(market_cap_range_b[0]), float(market_cap_range_b[1])),
                        full_market_cap_range_b,
                    )
                    else (float(market_cap_range_b[0]) * 1e9, float(market_cap_range_b[1]) * 1e9)
                )
                active_beta_range = (
                    None
                    if _is_default_range((float(beta_range[0]), float(beta_range[1])), full_beta_range)
                    else (float(beta_range[0]), float(beta_range[1]))
                )
                active_price_range = (
                    None
                    if _is_default_range((float(price_range[0]), float(price_range[1])), full_price_range)
                    else (float(price_range[0]), float(price_range[1]))
                )
                active_min_avg_volume = float(min_avg_volume) if float(min_avg_volume) > 0 else None

                try:
                    filtered = screener_api.apply_classic_filters(
                        df=screenable_universe_df,
                        market_cap_range=active_market_cap_range,
                        sectors=selected_sectors,
                        industries=selected_industries,
                        exchanges=selected_exchanges,
                        beta_range=active_beta_range,
                        price_range=active_price_range,
                        min_avg_volume=active_min_avg_volume,
                        valuation_filters=valuation_filters,
                        growth_filters=growth_filters,
                        quality_filters=quality_filters,
                        momentum_filters=momentum_filters,
                        dividend_filters=dividend_filters,
                        liquidity_prefilter=bool(liquidity_prefilter),
                    )
                except Exception as exc:
                    message = str(exc)
                    if (
                        ".str accessor" not in message
                        and "_chunked" not in message
                        and not isinstance(exc, NameError)
                    ):
                        raise
                    screener_api = importlib.reload(stock_screener_module)
                    filtered = screener_api.apply_classic_filters(
                        df=screenable_universe_df,
                        market_cap_range=active_market_cap_range,
                        sectors=selected_sectors,
                        industries=selected_industries,
                        exchanges=selected_exchanges,
                        beta_range=active_beta_range,
                        price_range=active_price_range,
                        min_avg_volume=active_min_avg_volume,
                        valuation_filters=valuation_filters,
                        growth_filters=growth_filters,
                        quality_filters=quality_filters,
                        momentum_filters=momentum_filters,
                        dividend_filters=dividend_filters,
                        liquidity_prefilter=bool(liquidity_prefilter),
                    )

                weighted = {
                    "value": value_weight,
                    "growth": growth_weight,
                    "quality": quality_weight,
                    "momentum": momentum_weight,
                    "stability": stability_weight,
                    "dividend": dividend_weight,
                }

                ranked = screener_api.rank_stocks(
                    screener_api.calculate_quant_score(filtered, weighted),
                    sort_by="QuantScore",
                    ascending=False,
                )
                shortlist_size = min(int(technical_limit), len(ranked))
                if shortlist_size > 0:
                    with st.spinner("Calculating technical indicators for shortlist..."):
                        try:
                            technical = screener_api.apply_technical_indicators(ranked.head(shortlist_size))
                        except Exception as exc:
                            message = str(exc)
                            if (
                                ".str accessor" not in message
                                and "_chunked" not in message
                                and not isinstance(exc, NameError)
                            ):
                                raise
                            screener_api = importlib.reload(stock_screener_module)
                            technical = screener_api.apply_technical_indicators(ranked.head(shortlist_size))
                    technical_cols = [
                        column
                        for column in ["Ticker", "RSI", "MACD", "Volatility", "Drawdown"]
                        if column in technical.columns
                    ]
                    ranked = ranked.merge(technical[technical_cols], on="Ticker", how="left")
                    ranked = screener_api.rank_stocks(
                        screener_api.calculate_quant_score(ranked, weighted),
                        sort_by="QuantScore",
                        ascending=False,
                    )

                st.session_state["stock_picker_results_classic"] = ranked
                st.session_state["stock_picker_classic_info"] = (
                    f"Stage 1: {len(filtered):,} matches | "
                    f"Stage 2: indicators on top {shortlist_size:,} symbols"
                )
                if filtered.empty:
                    st.warning(
                        "No matches with current active filters. Tip: keep the broad universe ranges open "
                        "and tighten only the factors that truly matter for this pass."
                    )

        with result_col:
            st.markdown("### Shortlist")
            if st.session_state.get("stock_picker_classic_info"):
                st.caption(st.session_state["stock_picker_classic_info"])
            classic_results = st.session_state.get("stock_picker_results_classic", pd.DataFrame())
            if not classic_results.empty:
                qscore_series = pd.to_numeric(
                    classic_results["QuantScore"] if "QuantScore" in classic_results.columns else pd.Series(dtype=float),
                    errors="coerce",
                )
                volume_series = pd.to_numeric(
                    classic_results["AvgVolume"] if "AvgVolume" in classic_results.columns else pd.Series(dtype=float),
                    errors="coerce",
                )
                priced_ratio = _coverage_ratio(classic_results["Price"]) if "Price" in classic_results.columns else 0.0
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Matches", f"{len(classic_results):,}")
                r2.metric("Median Quant Score", f"{qscore_series.median():.1f}" if qscore_series.notna().any() else "-")
                r3.metric("Median Avg Volume", _format_compact_number(volume_series.median()) if volume_series.notna().any() else "-")
                r4.metric("Price Coverage", f"{priced_ratio:.1%}")

            selected_rows = _render_selectable_results(classic_results, "classic")
            if not classic_results.empty:
                _render_screener_bulk_actions(selected_rows, classic_results, "classic")

    with screener_tabs[1]:
        ai_col, result_col = st.columns([1.05, 1.95])
        with ai_col:
            st.markdown("### Natural Language Query")
            query = st.text_area(
                "Describe your stock screen in plain English",
                height=180,
                placeholder=(
                    "Example: Find profitable large-cap semiconductor stocks with low debt, "
                    "strong ROE, and positive momentum."
                ),
                key="stock_picker_ai_query",
            )
            analyze_clicked = st.button(
                "Analyze with Groq",
                type="primary",
                key="stock_picker_ai_run",
                use_container_width=True,
            )
            if analyze_clicked:
                with st.spinner("Parsing request and applying AI filters..."):
                    parsed = parse_ai_query(query)
                    results, explanation = apply_ai_query(query, screenable_universe_df, parsed_query=parsed)
                    st.session_state["stock_picker_ai_parsed"] = parsed
                    st.session_state["stock_picker_ai_explanation"] = explanation
                    st.session_state["stock_picker_results_ai"] = results

            explanation_text = st.session_state.get("stock_picker_ai_explanation")
            if explanation_text:
                st.info(explanation_text)

        with result_col:
            parsed_payload = st.session_state.get("stock_picker_ai_parsed")
            if parsed_payload:
                with st.expander("Parsed Filters (JSON)", expanded=False):
                    st.json(parsed_payload)

            ai_results = st.session_state.get("stock_picker_results_ai", pd.DataFrame())
            st.markdown("### AI Results")
            if not ai_results.empty:
                a1, a2, a3 = st.columns(3)
                a1.metric("Matches", f"{len(ai_results):,}")
                a2.metric("Priced", f"{_coverage_ratio(ai_results['Price']) if 'Price' in ai_results.columns else 0.0:.1%}")
                a3.metric(
                    "Median Market Cap",
                    _format_compact_number(pd.to_numeric(ai_results.get("MarketCap"), errors="coerce").median())
                    if "MarketCap" in ai_results.columns
                    else "-",
                )
            selected_rows = _render_selectable_results(ai_results, "ai")
            if not ai_results.empty:
                _render_screener_bulk_actions(selected_rows, ai_results, "ai")


def _render_portfolio_tracker_tab() -> None:
    st.subheader("Portfolio Tracker")
    st.caption("Holdings are persisted as JSON files under data/portfolios/ and synced with session_state.")

    # Get current user_id for data isolation
    user_id = get_current_user_id()

    available_portfolios = list_portfolios(user_id=user_id)
    current_name = str(st.session_state.get("current_portfolio_name", "default"))
    if current_name not in available_portfolios:
        available_portfolios = [current_name, *available_portfolios]

    c1, c2 = st.columns([2, 1])
    selected_name = c1.selectbox(
        "Saved portfolios",
        options=available_portfolios if available_portfolios else ["default"],
        index=0,
        key="portfolio_tracker_selected_name",
    )
    if c2.button("Load Portfolio", key="portfolio_tracker_load", use_container_width=True):
        st.session_state["current_portfolio"] = load_portfolio(selected_name, user_id=user_id)
        st.session_state["current_portfolio_name"] = selected_name
        _invalidate_portfolio_live_snapshot()
        st.success(f"Loaded portfolio '{selected_name}'.")

    save_as_name = st.text_input(
        "Save as portfolio name",
        value=current_name,
        key="portfolio_tracker_save_as_name",
    )
    s1, s2 = st.columns(2)
    if s1.button("Save Current Portfolio", key="portfolio_tracker_save_current", use_container_width=True):
        save_portfolio(st.session_state["current_portfolio"], st.session_state.get("current_portfolio_name", current_name), user_id=user_id)
        st.success("Portfolio saved.")
    if s2.button("Save As New JSON", key="portfolio_tracker_save_as", use_container_width=True):
        save_portfolio(st.session_state["current_portfolio"], save_as_name, user_id=user_id)
        st.session_state["current_portfolio_name"] = save_as_name
        st.success(f"Saved portfolio as '{save_as_name}'.")

    with st.expander("Add Position", expanded=False):
        a1, a2 = st.columns(2)
        new_ticker = a1.text_input("Ticker", key="portfolio_add_ticker").strip().upper()
        new_shares = a2.number_input("Shares", min_value=0.01, value=1.0, step=0.25, key="portfolio_add_shares")
        b1, b2 = st.columns(2)
        new_cost = b1.number_input("Cost basis (optional)", min_value=0.0, value=0.0, step=0.1, key="portfolio_add_cost")
        new_target = b2.number_input("Target weight (optional)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="portfolio_add_target")
        if st.button("Add / Update Position", key="portfolio_add_button", use_container_width=True):
            cost_basis = None if new_cost <= 0 else float(new_cost)
            target_weight = None if new_target <= 0 else float(new_target)
            st.session_state["current_portfolio"] = add_position(
                st.session_state["current_portfolio"],
                ticker=new_ticker,
                shares=float(new_shares),
                cost_basis=cost_basis,
                target_weight=target_weight,
            )
            _invalidate_portfolio_live_snapshot()
            st.success(f"Position for {new_ticker} was added/updated.")

    portfolio = st.session_state.get("current_portfolio", load_portfolio("default", user_id=user_id))
    positions_df = _portfolio_positions_dataframe(portfolio)

    st.markdown("### Holdings")
    if positions_df.empty:
        st.info("No positions yet. Add holdings from Stock Picker or manually here.")
    else:
        edited_positions = st.data_editor(
            positions_df,
            use_container_width=True,
            hide_index=True,
            key="portfolio_positions_editor",
            num_rows="dynamic",
        )

        u1, u2 = st.columns(2)
        if u1.button("Apply Edits", key="portfolio_apply_edits", use_container_width=True):
            updated_portfolio = dict(st.session_state["current_portfolio"])
            updated_portfolio["positions"] = _positions_from_editor(edited_positions)
            st.session_state["current_portfolio"] = updated_portfolio
            _invalidate_portfolio_live_snapshot()
            st.success("Position edits applied.")

        removable = sorted(edited_positions["Ticker"].dropna().astype(str).str.upper().unique().tolist())
        to_remove = u2.multiselect("Remove tickers", removable, key="portfolio_remove_tickers")
        if st.button("Remove Selected Tickers", key="portfolio_remove_button", disabled=(len(to_remove) == 0), use_container_width=True):
            updated = st.session_state["current_portfolio"]
            for ticker in to_remove:
                updated = remove_position(updated, ticker)
            st.session_state["current_portfolio"] = updated
            _invalidate_portfolio_live_snapshot()
            st.success(f"Removed {len(to_remove)} ticker(s).")

    if st.button("Refresh Live Valuation", key="portfolio_refresh_live", use_container_width=True) or "portfolio_live_holdings" not in st.session_state:
        with st.spinner("Fetching latest prices for holdings..."):
            holdings, summary = compute_live_values(st.session_state["current_portfolio"])
        st.session_state["portfolio_live_holdings"] = holdings
        st.session_state["portfolio_live_summary"] = summary

    holdings = st.session_state.get("portfolio_live_holdings", pd.DataFrame())
    summary = st.session_state.get("portfolio_live_summary", {})

    if isinstance(summary, dict):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Market Value", f"${float(summary.get('TotalMarketValue', 0.0)):,.0f}")
        v2.metric("Cost Value", f"${float(summary.get('TotalCostValue', 0.0)):,.0f}")
        v3.metric("P&L", f"${float(summary.get('TotalPnL', 0.0)):,.0f}")
        v4.metric("Priced Positions", int(summary.get("PricedPositions", 0.0)))

    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        st.dataframe(holdings, use_container_width=True, hide_index=True)
        tolerance = st.slider(
            "Rebalance tolerance",
            min_value=0.01,
            max_value=0.10,
            value=0.03,
            step=0.005,
            key="portfolio_rebalance_tolerance",
        )
        suggestions = generate_rebalance_suggestions(holdings, tolerance=float(tolerance))
        st.markdown("### Rebalance Suggestions")
        if suggestions.empty:
            st.success("Current weights are within tolerance or no target weights were provided.")
        else:
            st.dataframe(suggestions, use_container_width=True, hide_index=True)


def _render_swing_tracker_tab() -> None:
    st.subheader("Swing Tracker")
    st.caption(
        "Discretionary swing-trade journal with stop rationale discipline, "
        "time-stop monitoring, lifecycle tracking, and post-trade review."
    )

    # Get current user_id for data isolation
    user_id = get_current_user_id()

    trades = refresh_swing_trades(user_id=user_id)
    overview = build_discipline_overview(trades)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Trades", int(overview.get("total_trades", 0.0)))
    s2.metric("Open", int(overview.get("open_trades", 0.0)))
    s3.metric("Overdue", int(overview.get("overdue_trades", 0.0)))
    s4.metric("Win Rate", f"{float(overview.get('win_rate', 0.0)):.1%}")
    s5.metric("Avg Discipline", f"{float(overview.get('avg_discipline_score', 0.0)):.1f}")

    try:
        streamlit_secrets = st.secrets
    except Exception:
        streamlit_secrets = None
    ai_api_key = resolve_swing_tracker_api_key(streamlit_secrets=streamlit_secrets)

    swing_tabs = st.tabs(["New Plan", "Open Trades", "Close Trade", "History", "Analytics"])

    with swing_tabs[0]:
        st.markdown("### Create New Trade Plan")
        p1, p2, p3 = st.columns(3)
        account_size = p1.number_input(
            "Account Size ($)",
            min_value=1_000.0,
            value=100_000.0,
            step=1_000.0,
            key="swing_new_account_size",
        )
        risk_percent = p2.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="swing_new_risk_percent",
        )
        initial_status = p3.selectbox(
            "Initial Status",
            options=["planned", "open"],
            index=0,
            key="swing_new_initial_status",
        )

        t1, t2, t3 = st.columns(3)
        ticker = t1.text_input("Ticker", key="swing_new_ticker").strip().upper()
        direction = t2.selectbox("Direction", options=["long", "short"], key="swing_new_direction")
        setup_type = t3.text_input("Setup Type", key="swing_new_setup_type").strip()
        thesis = st.text_area(
            "Thesis",
            height=120,
            key="swing_new_thesis",
            placeholder="Why does this setup have edge, and what invalidates it?",
        )

        x1, x2, x3 = st.columns(3)
        entry_price = x1.number_input(
            "Planned Entry",
            min_value=0.01,
            value=100.0,
            step=0.01,
            key="swing_new_entry_price",
        )
        stop_type = x2.selectbox(
            "Stop Type",
            options=["structural", "atr", "fixed_risk", "time_stop"],
            key="swing_new_stop_type",
        )
        trade_entry_date = x3.date_input(
            "Entry Date (for open status)",
            value=datetime.now().date(),
            key="swing_new_entry_date",
        )

        structural_price: float | None = None
        atr_value: float | None = None
        atr_multiple: float | None = None
        fixed_risk_percent: float | None = None
        manual_stop_loss: float | None = None

        if stop_type == "structural":
            structural_price = st.number_input(
                "Structural Stop Level",
                min_value=0.01,
                value=max(0.01, float(entry_price) * 0.95),
                step=0.01,
                key="swing_new_structural_price",
            )
        elif stop_type == "atr":
            a1, a2 = st.columns(2)
            atr_value = a1.number_input(
                "ATR Value",
                min_value=0.0001,
                value=max(0.01, float(entry_price) * 0.02),
                step=0.01,
                key="swing_new_atr_value",
            )
            atr_multiple = a2.number_input(
                "ATR Multiple",
                min_value=0.25,
                value=2.0,
                step=0.25,
                key="swing_new_atr_multiple",
            )
        elif stop_type == "fixed_risk":
            fixed_risk_percent = st.number_input(
                "Fixed Risk Stop (%)",
                min_value=0.1,
                max_value=25.0,
                value=5.0,
                step=0.1,
                key="swing_new_fixed_risk_pct",
            )
        else:
            z1, z2 = st.columns(2)
            fixed_risk_percent = z1.number_input(
                "Fallback Fixed Risk Stop (%)",
                min_value=0.0,
                max_value=25.0,
                value=3.0,
                step=0.1,
                key="swing_new_time_stop_risk_pct",
                help="Used only if manual hard stop is left blank.",
            )
            manual_stop_loss = z2.number_input(
                "Manual Hard Stop (optional)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key="swing_new_manual_stop",
            )
            manual_stop_loss = None if float(manual_stop_loss) <= 0 else float(manual_stop_loss)

        y1, y2, y3 = st.columns(3)
        target_price_raw = y1.number_input(
            "Primary Target",
            min_value=0.0,
            value=max(0.0, float(entry_price) * 1.1),
            step=0.01,
            key="swing_new_target_price",
        )
        time_stop_days = int(
            y2.number_input(
                "Time Stop (days)",
                min_value=1,
                value=8,
                step=1,
                key="swing_new_time_stop_days",
            )
        )
        planned_holding_days = int(
            y3.number_input(
                "Planned Holding (days)",
                min_value=1,
                value=10,
                step=1,
                key="swing_new_planned_holding_days",
            )
        )

        targets_text = st.text_input(
            "Additional Targets (comma-separated)",
            key="swing_new_targets_text",
            placeholder="Example: 112.5, 118.0",
        )
        notes = st.text_area("Planning Notes", height=100, key="swing_new_notes")

        computed_stop_loss: float | None = None
        stop_error = ""
        try:
            computed_stop_loss = calculate_stop_loss(
                direction=direction,
                entry_price=float(entry_price),
                stop_type=stop_type,
                structural_price=structural_price,
                atr_value=atr_value,
                atr_multiple=atr_multiple,
                fixed_risk_percent=fixed_risk_percent,
                manual_stop_loss=manual_stop_loss,
            )
        except Exception as exc:
            stop_error = str(exc)

        computed_position_size: float | None = None
        sizing_error = ""
        if computed_stop_loss is not None:
            try:
                computed_position_size = calculate_position_size_for_trade(
                    account_size=float(account_size),
                    risk_percent=float(risk_percent),
                    entry_price=float(entry_price),
                    stop_loss=float(computed_stop_loss),
                )
            except Exception as exc:
                sizing_error = str(exc)

        d1, d2, d3 = st.columns(3)
        d1.metric("Computed Stop", "-" if computed_stop_loss is None else f"{computed_stop_loss:.2f}")
        d2.metric(
            "Position Size",
            "-" if computed_position_size is None else f"{computed_position_size:,.2f}",
        )
        risk_amount = 0.0
        if computed_stop_loss is not None and computed_position_size is not None:
            risk_amount = abs(float(entry_price) - float(computed_stop_loss)) * float(computed_position_size)
        d3.metric("Risk Amount", f"${risk_amount:,.2f}")

        if stop_error:
            st.error(stop_error)
        if sizing_error:
            st.error(sizing_error)

        deterministic_rationale = ""
        if computed_stop_loss is not None:
            deterministic_rationale = build_stop_rationale(
                direction=direction,
                stop_type=stop_type,
                entry_price=float(entry_price),
                stop_loss=float(computed_stop_loss),
                time_stop_days=int(time_stop_days),
                atr_value=atr_value,
                atr_multiple=atr_multiple,
                fixed_risk_percent=fixed_risk_percent,
            )

        with st.expander("Groq AI Helper (Optional)", expanded=False):
            if not ai_api_key:
                st.info("GROQ_API_KEY not found. AI helper is disabled; deterministic flow remains active.")

            ai1, ai2, ai3 = st.columns(3)
            if ai1.button("Summarize Thesis", key="swing_ai_thesis_btn", use_container_width=True):
                st.session_state["swing_ai_thesis_result"] = summarize_trade_thesis(
                    ticker=ticker,
                    direction=direction,
                    thesis=thesis,
                    api_key=ai_api_key,
                )
            if ai2.button("Classify Setup", key="swing_ai_setup_btn", use_container_width=True):
                st.session_state["swing_ai_setup_result"] = classify_setup_type(
                    ticker=ticker,
                    direction=direction,
                    thesis=thesis,
                    api_key=ai_api_key,
                )
            if ai3.button(
                "Draft Stop Rationale",
                key="swing_ai_stop_btn",
                use_container_width=True,
                disabled=(computed_stop_loss is None),
            ):
                stop_payload = {
                    "ticker": ticker,
                    "direction": direction,
                    "entry_price": float(entry_price),
                    "stop_type": stop_type,
                    "stop_loss": computed_stop_loss,
                    "time_stop_days": int(time_stop_days),
                    "planned_holding_days": int(planned_holding_days),
                    "thesis": thesis,
                }
                stop_result = generate_ai_stop_rationale(
                    trade_payload=stop_payload,
                    api_key=ai_api_key,
                )
                st.session_state["swing_ai_stop_result"] = stop_result
                if stop_result.get("available"):
                    st.session_state["swing_new_stop_rationale"] = str(
                        stop_result.get("stop_rationale_summary", "")
                    )

            thesis_ai = st.session_state.get("swing_ai_thesis_result")
            if isinstance(thesis_ai, dict):
                st.caption("Thesis summary payload")
                st.json({
                    "available": thesis_ai.get("available", False),
                    "thesis_summary": thesis_ai.get("thesis_summary", ""),
                    "risk_highlights": thesis_ai.get("risk_highlights", []),
                    "execution_focus": thesis_ai.get("execution_focus", []),
                    "error": thesis_ai.get("error", ""),
                })

            setup_ai = st.session_state.get("swing_ai_setup_result")
            if isinstance(setup_ai, dict):
                st.caption("Setup classification payload")
                st.json({
                    "available": setup_ai.get("available", False),
                    "setup_type": setup_ai.get("setup_type", ""),
                    "confidence": setup_ai.get("confidence", 0.0),
                    "reasoning_tags": setup_ai.get("reasoning_tags", []),
                    "error": setup_ai.get("error", ""),
                })

            stop_ai = st.session_state.get("swing_ai_stop_result")
            if isinstance(stop_ai, dict):
                st.caption("Stop rationale payload")
                st.json({
                    "available": stop_ai.get("available", False),
                    "stop_rationale_summary": stop_ai.get("stop_rationale_summary", ""),
                    "invalidators": stop_ai.get("invalidators", []),
                    "checklist": stop_ai.get("checklist", []),
                    "time_stop_rule": stop_ai.get("time_stop_rule", ""),
                    "error": stop_ai.get("error", ""),
                })

        if "swing_new_stop_rationale" not in st.session_state:
            st.session_state["swing_new_stop_rationale"] = deterministic_rationale
        elif (
            deterministic_rationale
            and not str(st.session_state.get("swing_new_stop_rationale", "")).strip()
        ):
            st.session_state["swing_new_stop_rationale"] = deterministic_rationale

        stop_rationale = st.text_area(
            "Stop-Loss Rationale",
            height=120,
            key="swing_new_stop_rationale",
        )

        if deterministic_rationale:
            st.caption(f"Deterministic rationale helper: {deterministic_rationale}")

        if st.button(
            "Create Trade Plan",
            type="primary",
            key="swing_create_trade_btn",
            use_container_width=True,
        ):
            if computed_stop_loss is None:
                st.error("Cannot create trade: stop-loss is invalid.")
            elif computed_position_size is None or computed_position_size <= 0:
                st.error("Cannot create trade: position size calculation failed.")
            else:
                primary_target = None if float(target_price_raw) <= 0 else float(target_price_raw)
                target_list = _parse_targets_input(targets_text)
                if primary_target is not None:
                    target_list = [primary_target, *[item for item in target_list if item != primary_target]]

                entry_date_value = trade_entry_date if initial_status == "open" else None
                try:
                    trade = create_trade(
                        ticker=ticker,
                        direction=direction,
                        setup_type=setup_type or "unspecified",
                        thesis=thesis,
                        entry_price=float(entry_price),
                        stop_loss=float(computed_stop_loss),
                        stop_type=stop_type,
                        stop_rationale=stop_rationale,
                        target_price=primary_target,
                        targets=target_list,
                        time_stop_days=int(time_stop_days),
                        planned_holding_days=int(planned_holding_days),
                        risk_percent=float(risk_percent),
                        position_size=float(computed_position_size),
                        status=initial_status,
                        entry_date=entry_date_value,
                        notes=notes,
                    )
                    updated_trades = upsert_swing_trade(trades, trade)
                    save_swing_trade_book(updated_trades, user_id=user_id)
                    st.success(f"Trade plan created for {trade.ticker} ({trade.id}).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Trade plan validation failed: {exc}")

    with swing_tabs[1]:
        st.markdown("### Open Trades")
        active_trades = open_trade_rows(trades)
        planned_trades = [item for item in trades if item.status == "planned"]
        if not active_trades:
            st.info("No open trades right now.")
        else:
            active_df = pd.DataFrame(trades_to_rows(active_trades))
            active_df = active_df.sort_values(
                by=["Status", "ActualHoldDays"],
                ascending=[True, False],
            )
            active_cols = [
                "ID",
                "Ticker",
                "Direction",
                "Setup",
                "Status",
                "EntryDate",
                "Entry",
                "Stop",
                "Target",
                "PlannedHoldDays",
                "TimeStopDays",
                "ActualHoldDays",
                "HoldDelta",
                "PositionSize",
                "Notional",
                "RiskAmount",
                "DisciplineScore",
            ]
            active_df = active_df[[column for column in active_cols if column in active_df.columns]]

            o1, o2, o3 = st.columns(3)
            o1.metric("Open Trades", int(overview.get("open_trades", 0.0)))
            o2.metric("Overdue Trades", int(overview.get("overdue_trades", 0.0)))
            o3.metric(
                "Capital Trapped (Overdue)",
                f"${float(overview.get('capital_trapped_overdue', 0.0)):,.0f}",
            )

            def _highlight_overdue(row: pd.Series) -> list[str]:
                if str(row.get("Status", "")).lower() == "overdue":
                    return ["background-color: #ffe6e6"] * len(row)
                return [""] * len(row)

            st.dataframe(
                active_df.style.apply(_highlight_overdue, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            overdue_symbols = sorted(
                active_df.loc[active_df["Status"] == "overdue", "Ticker"]
                .dropna()
                .astype(str)
                .str.upper()
                .unique()
                .tolist()
            )
            if overdue_symbols:
                st.warning(f"Overdue trades needing action: {', '.join(overdue_symbols)}")

        if planned_trades:
            st.markdown("### Planned (Not Open Yet)")
            planned_df = pd.DataFrame(trades_to_rows(planned_trades))
            planned_cols = [
                "ID",
                "Ticker",
                "Direction",
                "Setup",
                "Status",
                "Entry",
                "Stop",
                "Target",
                "PlannedHoldDays",
                "TimeStopDays",
                "RiskPct",
                "PositionSize",
                "StopType",
                "StopRationale",
            ]
            planned_df = planned_df[[column for column in planned_cols if column in planned_df.columns]]
            st.dataframe(planned_df, use_container_width=True, hide_index=True)

    with swing_tabs[2]:
        st.markdown("### Close Trade")
        closable = open_trade_rows(trades)
        if not closable:
            st.info("No open or overdue trades to close.")
        else:
            close_options = {
                f"{trade.id} | {trade.ticker} | {trade.status}": trade.id for trade in closable
            }
            selected_label = st.selectbox(
                "Select trade",
                options=list(close_options.keys()),
                key="swing_close_selected_trade",
            )
            selected_id = close_options[selected_label]
            selected_trade = next(item for item in closable if item.id == selected_id)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", f"{selected_trade.entry_price:.2f}")
            c2.metric("Stop", f"{selected_trade.stop_loss:.2f}")
            c3.metric("Position", f"{selected_trade.position_size:,.2f}")
            c4.metric("Current Status", selected_trade.status)

            q1, q2 = st.columns(2)
            exit_price = q1.number_input(
                "Exit Price",
                min_value=0.01,
                value=float(selected_trade.entry_price),
                step=0.01,
                key="swing_close_exit_price",
            )
            exit_date = q2.date_input(
                "Exit Date",
                value=datetime.now().date(),
                key="swing_close_exit_date",
            )

            r1, r2 = st.columns(2)
            exit_reason = r1.selectbox(
                "Exit Reason",
                options=["target_hit", "stop_loss", "time_stop", "manual_exit", "thesis_invalidated"],
                key="swing_close_exit_reason",
            )
            final_status = r2.selectbox(
                "Final Status",
                options=["closed", "invalidated"],
                index=0,
                key="swing_close_final_status",
            )
            close_notes = st.text_area("Post-Trade Notes", height=120, key="swing_close_notes")

            if st.button(
                "Summarize Review (AI)",
                key="swing_ai_post_review_btn",
                use_container_width=True,
            ):
                st.session_state["swing_ai_post_review_result"] = summarize_post_trade_review(
                    trade_payload=selected_trade.to_dict(),
                    review_notes=close_notes,
                    api_key=ai_api_key,
                )

            post_ai = st.session_state.get("swing_ai_post_review_result")
            if isinstance(post_ai, dict):
                st.caption("Post-trade review payload")
                st.json({
                    "available": post_ai.get("available", False),
                    "review_summary": post_ai.get("review_summary", ""),
                    "discipline_observations": post_ai.get("discipline_observations", []),
                    "process_improvements": post_ai.get("process_improvements", []),
                    "error": post_ai.get("error", ""),
                })

            if st.button(
                "Close Selected Trade",
                type="primary",
                key="swing_close_trade_btn",
                use_container_width=True,
            ):
                try:
                    updated_trades, closed_trade = close_swing_trade(
                        trades,
                        trade_id=selected_trade.id,
                        exit_price=float(exit_price),
                        exit_date=exit_date,
                        exit_reason=exit_reason,
                        notes=close_notes,
                        final_status=final_status,
                    )
                    save_swing_trade_book(updated_trades, user_id=user_id)
                    st.success(
                        f"Closed {closed_trade.ticker} | "
                        f"PnL: ${float(closed_trade.realized_pnl or 0.0):,.2f} | "
                        f"R: {float(closed_trade.realized_r_multiple or 0.0):.2f}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to close trade: {exc}")

    with swing_tabs[3]:
        st.markdown("### Historical Trades")
        history_trades = historical_trade_rows(trades)
        if not history_trades:
            st.info("No closed or invalidated trades yet.")
        else:
            history_df = pd.DataFrame(trades_to_rows(history_trades))
            history_df = history_df.sort_values(
                by=["ExitDate", "EntryDate"],
                ascending=[False, False],
            )
            history_cols = [
                "ID",
                "Ticker",
                "Direction",
                "Setup",
                "Status",
                "EntryDate",
                "ExitDate",
                "Entry",
                "Exit",
                "RealizedPnL",
                "RMultiple",
                "PlannedHoldDays",
                "ActualHoldDays",
                "HoldDelta",
                "DisciplineScore",
                "ExitReason",
                "Notes",
            ]
            history_df = history_df[[column for column in history_cols if column in history_df.columns]]
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            recent_row = history_df.iloc[0]
            st.markdown("### Recent Closed Trade Review")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Ticker", str(recent_row.get("Ticker", "")))
            rc2.metric("PnL", f"${float(recent_row.get('RealizedPnL', 0.0) or 0.0):,.2f}")
            rc3.metric("R Multiple", f"{float(recent_row.get('RMultiple', 0.0) or 0.0):.2f}")
            rc4.metric("Discipline", f"{float(recent_row.get('DisciplineScore', 0.0) or 0.0):.1f}")
            st.write(f"**Exit Reason:** {recent_row.get('ExitReason', '-')}")
            st.write(f"**Notes:** {recent_row.get('Notes', '-')}")

    with swing_tabs[4]:
        st.markdown("### Analytics")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Closed Trades", int(overview.get("closed_trades", 0.0)))
        a2.metric("Win Rate", f"{float(overview.get('win_rate', 0.0)):.1%}")
        a3.metric("Avg R Multiple", f"{float(overview.get('avg_r_multiple', 0.0)):.2f}")
        a4.metric(
            "Trapped Capital",
            f"${float(overview.get('capital_trapped_overdue', 0.0)):,.0f}",
        )

        all_rows = pd.DataFrame(trades_to_rows(trades))
        if all_rows.empty:
            st.info("Create a few trades to unlock analytics.")
        else:
            status_counts = (
                all_rows["Status"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Count")
            )
            st.markdown("#### Trade Status Distribution")
            st.dataframe(status_counts, use_container_width=True, hide_index=True)
            st.bar_chart(status_counts.set_index("Status"))

            discipline_table = (
                all_rows.groupby("Status", dropna=False)["DisciplineScore"]
                .mean()
                .reset_index()
                .rename(columns={"DisciplineScore": "AvgDisciplineScore"})
                .sort_values(by="AvgDisciplineScore", ascending=False)
            )
            st.markdown("#### Discipline Overview")
            st.dataframe(discipline_table, use_container_width=True, hide_index=True)

            holding_table = all_rows[all_rows["Status"].isin(["closed", "invalidated"])].copy()
            if not holding_table.empty:
                st.markdown("#### Planned vs Actual Holding")
                holding_table = holding_table[
                    [
                        "Ticker",
                        "Status",
                        "PlannedHoldDays",
                        "TimeStopDays",
                        "ActualHoldDays",
                        "HoldDelta",
                        "RMultiple",
                        "DisciplineScore",
                    ]
                ].sort_values(by="HoldDelta", ascending=False)
                st.dataframe(holding_table, use_container_width=True, hide_index=True)

            overdue_table = all_rows[all_rows["Status"] == "overdue"].copy()
            if not overdue_table.empty:
                st.markdown("#### Overdue Trade Focus")
                overdue_table = overdue_table[
                    [
                        "Ticker",
                        "EntryDate",
                        "ActualHoldDays",
                        "PlannedHoldDays",
                        "TimeStopDays",
                        "HoldDelta",
                        "Notional",
                        "RiskAmount",
                    ]
                ].sort_values(by="ActualHoldDays", ascending=False)
                st.dataframe(overdue_table, use_container_width=True, hide_index=True)


def _render_dashboard_note(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='dashboard-note'>"
            f"<strong>{escape(title)}</strong>"
            f"<span>{escape(body)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_empty_dashboard_state(preferences: DashboardPreferences) -> None:
    st.markdown(
        """
        <div class="dashboard-hero">
            <div class="dashboard-kicker">Modular dashboard</div>
            <h2>Run portfolio analysis only when you need it.</h2>
            <p>
                Keep the workspace available for screening, portfolio tracking,
                and swing-trade workflows even before the first analysis run.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_dashboard_note(
        "How the new layout works",
        "Portfolio inputs stay in the sidebar, while the page itself is split into dedicated"
        " workspaces for overview, analytics, portfolio lab, tools, and reporting.",
    )
    if preferences.show_workspace_when_empty:
        _render_workspace_hub(None)
    else:
        st.info("Turn on 'Keep workspace visible before first run' in the sidebar to open the tool hub here.")


def _render_dashboard_hero(analysis_result: Dict[str, Any], preferences: DashboardPreferences) -> None:
    summary_result = analysis_result.get("summary_result")
    tickers = [str(item) for item in analysis_result.get("tickers", [])]
    ticker_preview = ", ".join(tickers[:6])
    if len(tickers) > 6:
        ticker_preview = f"{ticker_preview} +{len(tickers) - 6}"

    regime_label = str(getattr(summary_result, "regime_label", "analysis ready"))
    confidence = float(getattr(summary_result, "confidence", 0.0))
    composite_score = float(getattr(summary_result, "composite_score", 0.0))
    run_record = analysis_result.get("run_record")
    run_id = getattr(run_record, "run_id", "-")

    hero_title = f"{analysis_result.get('risk_profile', 'balanced').title()} portfolio cockpit"
    hero_body = (
        f"Tracking {len(tickers)} assets across {int(analysis_result.get('horizon_days', 252))} days. "
        f"Use the {preferences.preset} preset or hide sections from the sidebar when you want a lighter view."
    )

    st.markdown(
        (
            "<div class='dashboard-hero'>"
            "<div class='dashboard-kicker'>Quant Platform</div>"
            f"<h2>{escape(hero_title)}</h2>"
            f"<p>{escape(hero_body)}</p>"
            "<div class='dashboard-badge-row'>"
            f"<span class='dashboard-badge'>Run: {escape(str(run_id))}</span>"
            f"<span class='dashboard-badge'>Regime: {escape(regime_label)}</span>"
            f"<span class='dashboard-badge'>Confidence: {confidence:.2f}</span>"
            f"<span class='dashboard-badge'>Composite: {composite_score:.2f}</span>"
            f"<span class='dashboard-badge'>Assets: {len(tickers)}</span>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    meta_col, universe_col = st.columns([1.05, 1.35])
    with meta_col:
        st.caption("Current run")
        st.write(f"**Date range:** {analysis_result['start_date']} to {analysis_result['end_date']}")
        st.write(f"**Risk-free rate:** {analysis_result['risk_free_rate']:.3f}")
        st.write(f"**Visible sections:** {', '.join(PAGE_LABELS[key] for key in preferences.visible_pages)}")
    with universe_col:
        st.caption("Tracked universe")
        st.write(ticker_preview or "No tickers loaded.")

    _render_export_actions(
        analysis_result,
        title="Quick Exports",
        body=(
            "Core report downloads stay pinned here, so PDF, CSV, and JSON are always available"
            " without digging through the dashboard."
        ),
        compact=True,
        key_namespace="hero_exports",
    )


def _render_workspace_hub(analysis_result: Dict[str, Any] | None) -> None:
    st.subheader("Workspace Hub")
    st.caption(
        "Operational tools live here so screening, portfolio maintenance, and trade journaling"
        " stay available without cluttering the analysis pages."
    )
    workspace_tabs = st.tabs(["Stock Picker", "Portfolio Tracker", "Swing Tracker", "Economics Coach"])
    with workspace_tabs[0]:
        _render_stock_picker_tab()
    with workspace_tabs[1]:
        _render_portfolio_tracker_tab()
    with workspace_tabs[2]:
        _render_swing_tracker_tab()
    with workspace_tabs[3]:
        if analysis_result is None:
            st.info("Run one portfolio analysis first to generate context-aware economics questions.")
        else:
            render_economics_questions_section(analysis_result)


def _render_overview_page(analysis_result: Dict[str, Any]) -> None:
    metrics = analysis_result["metrics"]
    score_result = analysis_result["score_result"]
    summary_result = analysis_result.get("summary_result")

    st.subheader("Executive Overview")
    _render_dashboard_note(
        "One place for the main answer",
        "This overview keeps the score, regime, risk flags, and positioning together so you"
        " can decide quickly whether to inspect deeper analytical layers.",
    )

    row1 = st.columns(4)
    row1[0].metric("Deterministic Score", f"{score_result['score']} / 100")
    row1[1].metric("Rating", str(score_result["rating"]))
    row1[2].metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    row1[3].metric("Volatility", f"{metrics['volatility']:.2%}")

    row2 = st.columns(4)
    row2[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
    row2[1].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    row2[2].metric("Average Correlation", f"{metrics['avg_correlation']:.3f}")
    row2[3].metric("Effective Holdings", f"{metrics['effective_holdings']:.2f}")

    left_col, right_col = st.columns([1.25, 1.0])
    with left_col:
        st.markdown("### Regime and signals")
        if summary_result:
            st.write(f"**Regime:** {getattr(summary_result, 'regime_label', 'neutral')}")
            st.write(f"**Confidence:** {float(getattr(summary_result, 'confidence', 0.0)):.3f}")
            st.write(
                f"**Expected return view:** "
                f"{float(getattr(summary_result, 'expected_return_view', 0.0)):.2%}"
            )
            st.write(
                f"**Expected risk view:** "
                f"{float(getattr(summary_result, 'expected_risk_view', 0.0)):.2%}"
            )
            regime_interpretation = str(getattr(summary_result, "regime_interpretation", "")).strip()
            if regime_interpretation:
                st.caption(regime_interpretation)
            for line in getattr(summary_result, "highlights", []):
                st.write(f"- {line}")
            strongest = getattr(summary_result, "strongest_signals", [])
            if strongest:
                st.caption("Strongest signals")
                st.dataframe(pd.DataFrame(strongest), use_container_width=True, hide_index=True)
        else:
            st.info("Summary layer is not available for this run.")

    with right_col:
        st.markdown("### Risk flags")
        rendered_alerts: set[str] = set()
        if score_result["flags"]:
            for flag in score_result["flags"]:
                normalized_flag = str(flag).strip()
                if normalized_flag and normalized_flag not in rendered_alerts:
                    st.warning(normalized_flag)
                    rendered_alerts.add(normalized_flag)
        else:
            st.success("No critical deterministic flags detected.")

        if summary_result:
            for warning in getattr(summary_result, "warnings", []):
                normalized_warning = str(warning).strip()
                if normalized_warning and normalized_warning not in rendered_alerts:
                    st.warning(normalized_warning)
                    rendered_alerts.add(normalized_warning)
            for flag in getattr(summary_result, "risk_flags", []):
                normalized_risk_flag = str(flag).strip()
                if normalized_risk_flag and normalized_risk_flag not in rendered_alerts:
                    st.warning(normalized_risk_flag)
                    rendered_alerts.add(normalized_risk_flag)
            recent_changes = getattr(summary_result, "recent_changes", [])
            if recent_changes:
                st.caption("Recent changes vs prior run")
                for change in recent_changes:
                    st.write(f"- {change}")

    allocation_col, compare_col = st.columns([1.0, 1.25])
    with allocation_col:
        st.markdown("### Current allocation")
        weights_df = pd.DataFrame(
            {
                "Ticker": analysis_result["tickers"],
                "Weight": [f"{float(weight):.2%}" for weight in analysis_result["weights"]],
            }
        )
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        if score_result["breakdown"]:
            with st.expander("Score breakdown", expanded=False):
                score_breakdown_df = pd.DataFrame(score_result["breakdown"]).rename(
                    columns={"rule": "Rule", "penalty": "Penalty", "detail": "Detail"}
                )
                st.dataframe(score_breakdown_df, use_container_width=True, hide_index=True)

    with compare_col:
        st.markdown("### Portfolio vs optimized alternatives")
        highlighted_rows = []
        for item in analysis_result.get("highlighted_portfolios", []):
            highlighted_rows.append(
                {
                    "Portfolio": item.get("name", ""),
                    "Expected Return": float(item.get("expected_return", 0.0)),
                    "Volatility": float(item.get("volatility", 0.0)),
                    "Sharpe": float(item.get("sharpe_ratio", 0.0)),
                    "Effective Holdings": float(item.get("effective_holdings", 0.0)),
                    "Max Weight": float(item.get("max_weight", 0.0)),
                }
            )
        if highlighted_rows:
            st.dataframe(pd.DataFrame(highlighted_rows), use_container_width=True, hide_index=True)
        top_news = getattr(summary_result, "top_relevant_news", []) if summary_result else []
        if top_news:
            st.caption("Top relevant news")
            st.dataframe(pd.DataFrame(top_news), use_container_width=True, hide_index=True)


def _render_decision_cockpit_page(analysis_result: Dict[str, Any]) -> None:
    st.subheader("Decision Cockpit")
    _render_dashboard_note(
        "Stress the book before the market does",
        "This playground replays your current portfolio through deterministic extreme scenarios"
        " so you can see which shock hurts most, how deep the path gets, and what action deserves attention first.",
    )

    summary_result = analysis_result.get("summary_result")
    preset_metadata = {item["name"]: item for item in list_scenario_presets()}
    control_col0, control_col1, control_col2, control_col3 = st.columns([1.15, 1.0, 1.0, 1.2])
    with control_col0:
        library_focus = st.selectbox(
            "Scenario library",
            options=["Historical Crises", "Synthetic Stress", "All Scenarios"],
            index=0,
            key="cockpit_library_focus",
        )
    with control_col1:
        severity = st.slider(
            "Stress severity",
            min_value=0.60,
            max_value=2.00,
            value=1.00,
            step=0.05,
            key="cockpit_stress_severity",
        )
    with control_col2:
        horizon_days = st.select_slider(
            "Scenario horizon",
            options=[10, 20, 30, 45, 60, 90],
            value=30,
            key="cockpit_horizon_days",
        )
    with control_col3:
        initial_value = st.number_input(
            "Capital base",
            min_value=10_000.0,
            value=100_000.0,
            step=10_000.0,
            key="cockpit_initial_value",
        )
    st.caption(
        "Historical crises are analog replays: the app keeps your current holdings and recent return structure,"
        " then layers crisis-shaped phase shocks on top."
    )

    scenario_suite = build_scenario_suite(
        returns_df=analysis_result["returns"],
        tickers=analysis_result["tickers"],
        weights=analysis_result["weights"],
        severity=float(severity),
        initial_value=float(initial_value),
        horizon_override=int(horizon_days),
    )
    summary_rows = scenario_suite["rows"].reset_index(drop=True)
    if library_focus == "Historical Crises":
        summary_rows = summary_rows[summary_rows["Category"] == "Historical Crisis"].reset_index(drop=True)
    elif library_focus == "Synthetic Stress":
        summary_rows = summary_rows[summary_rows["Category"] == "Synthetic Stress"].reset_index(drop=True)
    if summary_rows.empty:
        st.info("Scenario engine is unavailable for this run.")
        return

    scenario_options = summary_rows["Scenario"].tolist()
    current_selection = st.session_state.get("cockpit_selected_scenario")
    if current_selection not in scenario_options:
        st.session_state["cockpit_selected_scenario"] = scenario_options[0]

    selector_col, cue_col = st.columns([1.45, 1.0])
    with selector_col:
        selected_name = st.selectbox(
            "Replay a scenario",
            options=scenario_options,
            index=scenario_options.index(st.session_state["cockpit_selected_scenario"]),
            key="cockpit_selected_scenario",
        )
        selected_meta = preset_metadata.get(selected_name, {})
        st.caption(
            f"{selected_meta.get('category', '')} | {selected_meta.get('era', '')} | "
            f"default horizon {int(selected_meta.get('horizon_days', horizon_days))} days"
        )
    with cue_col:
        st.caption("Playback modes")
        st.write("- Animated crisis replay with play / pause controls")
        st.write("- Phase-by-phase shock heatmap")
        st.write("- Fingerprint view for stress profile comparison")

    worst_row = summary_rows.iloc[0]
    selected_scenario = scenario_suite["scenarios"][selected_name]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Most Fragile Scenario", str(worst_row["Scenario"]))
    m2.metric("Worst Stress Return", f"{float(worst_row['Total Return']):.2%}")
    m3.metric("Worst Stress Drawdown", f"{float(worst_row['Max Drawdown']):.2%}")
    m4.metric("Stress Gap vs Baseline", f"{float(worst_row['Stress Gap']):+.2%}")

    expected_risk = (
        float(getattr(summary_result, "expected_risk_view", analysis_result["metrics"]["volatility"]))
        if summary_result
        else float(analysis_result["metrics"]["volatility"])
    )
    if float(worst_row["Total Return"]) <= -0.15:
        cockpit_message = (
            "Current construction is vulnerable to at least one extreme regime. Priority should be"
            " hedge sizing, concentration checks, and pre-committing what gets trimmed first."
        )
    elif expected_risk > 0.18:
        cockpit_message = (
            "Base risk is already elevated, so even moderate scenario stress can matter. Use the worst path"
            " to decide where rebalancing or protection would help most."
        )
    else:
        cockpit_message = (
            "Stress losses look contained relative to the current return/risk profile. The useful next step"
            " is testing whether the same resilience holds after concentration or view changes."
        )
    _render_dashboard_note("Decision brief", cockpit_message)

    st.markdown("### Crisis Atlas")
    st.plotly_chart(
        plot_scenario_atlas(summary_rows, highlight_scenario=selected_name),
        use_container_width=True,
    )

    scenario_table = summary_rows.copy()
    scenario_table["Final Value"] = scenario_table["Final Value"].map(lambda value: f"${value:,.0f}")
    scenario_table["Total Return"] = scenario_table["Total Return"].map(lambda value: f"{value:.2%}")
    scenario_table["Max Drawdown"] = scenario_table["Max Drawdown"].map(lambda value: f"{value:.2%}")
    scenario_table["Worst Day"] = scenario_table["Worst Day"].map(lambda value: f"{value:.2%}")
    scenario_table["Stress Gap"] = scenario_table["Stress Gap"].map(lambda value: f"{value:+.2%}")
    st.markdown("### Extreme Scenario Scoreboard")
    st.dataframe(
        scenario_table[
            [
                "Scenario",
                "Category",
                "Era",
                "Horizon",
                "Action Cue",
                "Final Value",
                "Total Return",
                "Max Drawdown",
                "Worst Day",
                "Stress Gap",
                "Days Underwater",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    detail_col, exposure_col = st.columns([1.45, 1.0])
    with detail_col:
        st.markdown(f"### Replay Studio: {selected_name}")
        st.caption(preset_metadata.get(selected_name, {}).get("description", selected_scenario["description"]))
    with exposure_col:
        stressed_stats = selected_scenario["stressed_stats"]
        d1, d2 = st.columns(2)
        d1.metric("Action Cue", selected_scenario["action_cue"])
        d2.metric("Underwater Days", int(stressed_stats["days_underwater"]))
        d3, d4 = st.columns(2)
        d3.metric("Worst Day", f"{float(stressed_stats['worst_day']):.2%}")
        d4.metric(
            "Recovery Day",
            "-" if stressed_stats["recovery_day"] is None else int(stressed_stats["recovery_day"]),
        )
        st.write(f"**Playbook:** {selected_scenario['playbook']}")

    cockpit_tabs = st.tabs(["Playback", "Shock Map", "Fingerprint", "Impact Board"])
    with cockpit_tabs[0]:
        st.caption("Press Play to replay the path like a mini crisis filmstrip.")
        st.plotly_chart(
            plot_crisis_playback(
                scenario_name=selected_name,
                baseline_path=selected_scenario["baseline_path"],
                stressed_path=selected_scenario["stressed_path"],
                daily_phase_labels=selected_scenario["daily_phase_labels"],
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_phase_timeline(selected_name, selected_scenario["phase_table"]),
            use_container_width=True,
        )

    with cockpit_tabs[1]:
        shock_col, phase_col = st.columns([1.2, 1.0])
        with shock_col:
            st.plotly_chart(
                plot_scenario_shock_map(
                    selected_scenario["shock_map"],
                    title=f"{selected_name} Shock Map",
                ),
                use_container_width=True,
            )
        with phase_col:
            phase_view = selected_scenario["phase_table"].copy()
            if not phase_view.empty:
                phase_view["Vol Multiplier"] = phase_view["Vol Multiplier"].map(lambda value: f"{value:.2f}x")
            st.dataframe(phase_view, use_container_width=True, hide_index=True)
            drawdown_frame = pd.DataFrame(
                {
                    "Baseline Drawdown": selected_scenario["baseline_drawdown"],
                    "Scenario Drawdown": selected_scenario["stressed_drawdown"],
                }
            )
            st.line_chart(drawdown_frame)

    with cockpit_tabs[2]:
        fingerprint_col, role_col = st.columns([1.0, 1.0])
        with fingerprint_col:
            st.plotly_chart(
                plot_scenario_fingerprint(
                    scenario_name=selected_name,
                    stressed_stats=selected_scenario["stressed_stats"],
                    baseline_stats=selected_scenario["baseline_stats"],
                    horizon_days=int(selected_scenario["horizon_days"]),
                ),
                use_container_width=True,
            )
        with role_col:
            st.markdown("#### Role exposure map")
            st.dataframe(selected_scenario["role_exposures"], use_container_width=True, hide_index=True)
            path_frame = pd.DataFrame(
                {
                    "Baseline": selected_scenario["baseline_path"],
                    "Scenario": selected_scenario["stressed_path"],
                }
            )
            st.line_chart(path_frame)

    with cockpit_tabs[3]:
        impact_col, asset_col = st.columns([1.1, 0.9])
        with impact_col:
            st.plotly_chart(
                plot_asset_stress_impact(selected_scenario["asset_impact_proxy"]),
                use_container_width=True,
            )
        with asset_col:
            impact_view = (
                selected_scenario["asset_impact_proxy"]
                .sort_values()
                .rename_axis("Ticker")
                .reset_index(name="Stress Gap")
            )
            impact_view["Stress Gap"] = impact_view["Stress Gap"].map(lambda value: f"${value:,.0f}")
            st.dataframe(impact_view, use_container_width=True, hide_index=True)

    with st.expander("Scenario assumptions", expanded=False):
        assumptions_rows = []
        for item in list_scenario_presets():
            assumptions_rows.append(
                {
                    "Scenario": item["name"],
                    "Category": item["category"],
                    "Era": item["era"],
                    "Default Horizon": int(item["horizon_days"]),
                    "Description": item["description"],
                    "Playbook": item["playbook"],
                }
            )
        st.dataframe(pd.DataFrame(assumptions_rows), use_container_width=True, hide_index=True)
        st.caption(
            "These scenarios use a deterministic replay of recent asset returns and then layer"
            " role-specific shocks on top. They are designed for decision support, not probabilistic forecasting."
        )


def _render_analysis_lab_page(analysis_result: Dict[str, Any], show_raw_tables: bool) -> None:
    model_results = analysis_result.get("model_results", {})
    signal_results = analysis_result.get("signal_results", {})
    backtest_result = analysis_result.get("backtest_result", {})
    news_result = analysis_result.get("news_result")
    history_records = analysis_result.get("history_records", [])
    advanced_models = analysis_result.get("advanced_models", {})

    st.subheader("Analysis Lab")
    analysis_tabs = st.tabs(["Data", "Models", "Signals", "Backtest", "News", "History", "Compare"])

    with analysis_tabs[0]:
        st.caption(
            f"Run record: {analysis_result.get('run_record').run_id if analysis_result.get('run_record') else '-'}"
        )
        portfolio_timeseries = analysis_result["portfolio_timeseries"]
        if isinstance(portfolio_timeseries, pd.DataFrame):
            if "value" in portfolio_timeseries.columns:
                portfolio_value_series = portfolio_timeseries["value"]
            else:
                portfolio_value_series = portfolio_timeseries.iloc[:, 0]
        else:
            portfolio_value_series = portfolio_timeseries
        st.line_chart(portfolio_value_series.rename("Portfolio Value"))
        if show_raw_tables:
            data_tab1, data_tab2 = st.tabs(["Prices", "Returns"])
            with data_tab1:
                st.dataframe(analysis_result["prices"].tail(20), use_container_width=True)
            with data_tab2:
                st.dataframe(analysis_result["returns"].tail(20), use_container_width=True)

    with analysis_tabs[1]:
        model_rows: List[Dict[str, Any]] = []
        for model_name, model in model_results.items():
            metrics_map = getattr(model, "metrics", {})
            model_rows.append(
                {
                    "Model": model_name,
                    "Family": getattr(model, "family", ""),
                    "Available": bool(getattr(model, "available", False)),
                    "Confidence": float(getattr(model, "confidence", 0.0)),
                    "Primary Metric": float(
                        metrics_map.get(
                            "expected_annual_return",
                            metrics_map.get(
                                "posterior_annual_return",
                                metrics_map.get(
                                    "posterior_expected_annual_return",
                                    metrics_map.get("volatility_annualized", 0.0),
                                ),
                            ),
                        )
                    ),
                    "Band Width": float(
                        metrics_map.get(
                            "posterior_interval_width",
                            metrics_map.get("forecast_spread", 0.0),
                        )
                    ),
                    "Error": getattr(model, "error", ""),
                }
            )

        if model_rows:
            st.markdown("### Quant stack model outputs")
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
            confidence_frame = pd.DataFrame(model_rows)[["Model", "Confidence"]].set_index("Model")
            st.caption("Model confidence")
            st.bar_chart(confidence_frame)

        advanced_rows: List[Dict[str, Any]] = []
        for model_name, result in advanced_models.items():
            if result.get("available", False):
                metrics_map = result.get("metrics", {})
                advanced_rows.append(
                    {
                        "Model": model_name,
                        "Status": "ok",
                        "Signal 1": round(
                            float(
                                metrics_map.get(
                                    "expected_annual_return",
                                    metrics_map.get(
                                        "next_period_return_forecast",
                                        metrics_map.get("conditional_volatility", 0.0),
                                    ),
                                )
                                or 0.0
                            ),
                            6,
                        ),
                        "Signal 2": round(
                            float(
                                metrics_map.get(
                                    "trend_slope_daily",
                                    metrics_map.get(
                                        "forecast_confidence",
                                        metrics_map.get("volatility_annualized", 0.0),
                                    ),
                                )
                                or 0.0
                            ),
                            6,
                        ),
                        "Confidence": round(float(metrics_map.get("confidence", 0.0) or 0.0), 4),
                        "Error": "",
                    }
                )
            else:
                advanced_rows.append(
                    {
                        "Model": model_name,
                        "Status": "unavailable",
                        "Signal 1": np.nan,
                        "Signal 2": np.nan,
                        "Confidence": np.nan,
                        "Error": result.get("error", ""),
                    }
                )

        if advanced_rows:
            with st.expander("Legacy advanced model compatibility view", expanded=False):
                st.caption(
                    "Same underlying model layer rendered in the older dictionary schema for compatibility checks."
                )
                st.dataframe(pd.DataFrame(advanced_rows), use_container_width=True, hide_index=True)

    with analysis_tabs[2]:
        signal_rows: List[Dict[str, Any]] = []
        for signal_name, signal in signal_results.items():
            signal_rows.append(
                {
                    "Signal": signal_name,
                    "Family": getattr(signal, "family", ""),
                    "Direction": getattr(signal, "direction", "neutral"),
                    "Score": float(getattr(signal, "score", 0.0)),
                    "Confidence": float(getattr(signal, "confidence", 0.0)),
                    "Available": bool(getattr(signal, "available", False)),
                    "Error": getattr(signal, "error", ""),
                }
            )
        if signal_rows:
            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No signal outputs are available for this run.")

    with analysis_tabs[3]:
        bt_metrics = backtest_result.get("metrics", {})
        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        bt_col1.metric("Total Return", f"{bt_metrics.get('total_return', 0.0):.2%}")
        bt_col2.metric("Volatility", f"{bt_metrics.get('volatility', 0.0):.2%}")
        bt_col3.metric("Sharpe", f"{bt_metrics.get('sharpe', 0.0):.3f}")
        bt_col4.metric("Max DD", f"{bt_metrics.get('max_drawdown', 0.0):.2%}")

        equity_curve = backtest_result.get("equity_curve")
        drawdown_series = backtest_result.get("drawdown")
        if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            st.line_chart(equity_curve.rename("equity_curve"))
        if isinstance(drawdown_series, pd.Series) and not drawdown_series.empty:
            st.line_chart(drawdown_series.rename("drawdown"))
        st.caption(f"No-look-ahead safe: {bool(backtest_result.get('lookahead_safe', False))}")

    with analysis_tabs[4]:
        if news_result and getattr(news_result, "available", False):
            n_col1, n_col2 = st.columns(2)
            n_col1.metric(
                "Aggregate sentiment",
                f"{float(getattr(news_result, 'sentiment_score', 0.0)):+.3f}",
            )
            n_col2.metric(
                "Sentiment dispersion",
                f"{float(getattr(news_result, 'sentiment_dispersion', 0.0)):.3f}",
            )
            st.caption(
                f"Provider used: {str(getattr(news_result, 'context', {}).get('provider_used', 'unknown'))}"
            )

            fetch_errors = list(getattr(news_result, "context", {}).get("fetch_errors", []))
            if fetch_errors:
                with st.expander("News fetch diagnostics", expanded=False):
                    for msg in fetch_errors:
                        st.write(f"- {msg}")

            news_rows = build_news_rows_for_ui(news_result)
            if news_rows:
                news_df = pd.DataFrame(news_rows)

                def _sentiment_color_style(value: Any) -> str:
                    if value == "green":
                        return "background-color: #d5f5e3; color: #1e8449;"
                    if value == "red":
                        return "background-color: #fadbd8; color: #922b21;"
                    return "background-color: #fcf3cf; color: #7d6608;"

                styled = news_df.style.map(_sentiment_color_style, subset=["Sentiment Color"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.info("No relevant news items found for this run.")
        else:
            st.info(
                f"News layer unavailable: "
                f"{getattr(news_result, 'error', 'n/a') if news_result else 'n/a'}"
            )

    with analysis_tabs[5]:
        history_rows = [
            {
                "Run ID": item.get("run_id", ""),
                "Timestamp": item.get("timestamp", ""),
                "Tickers": ", ".join(item.get("universe", [])),
                "Regime": item.get("summary", {}).get("regime_label", ""),
                "Composite": item.get("summary", {}).get("composite_score", 0.0),
            }
            for item in history_records
        ]
        if history_rows:
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No historical runs stored yet.")

    with analysis_tabs[6]:
        run_ids = [item.get("run_id") for item in history_records if item.get("run_id")]
        if len(run_ids) >= 2:
            left_run = st.selectbox("Base run", options=run_ids, index=min(1, len(run_ids) - 1))
            right_run = st.selectbox("Compare run", options=run_ids, index=0)
            if left_run and right_run and left_run != right_run:
                try:
                    left_data = load_run_record(left_run)
                    right_data = load_run_record(right_run)
                    comparison = compare_runs(left_data, right_data)
                    st.dataframe(
                        pd.DataFrame([comparison["metric_diff"]]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if comparison.get("summary_diff"):
                        st.caption("Summary deltas")
                        st.dataframe(
                            pd.DataFrame([comparison["summary_diff"]]),
                            use_container_width=True,
                            hide_index=True,
                        )
                except Exception as exc:
                    st.warning(f"Comparison failed: {exc}")
        else:
            st.info("Need at least two saved runs for comparison.")


def _render_portfolio_lab_page(analysis_result: Dict[str, Any]) -> None:
    metrics = analysis_result["metrics"]
    portfolio_returns = analysis_result["portfolio_returns"]
    returns = analysis_result["returns"]
    corr_matrix = analysis_result["correlation_matrix"]
    frontier = analysis_result["frontier"]
    min_var_result = analysis_result["min_var_result"]
    max_sharpe_result = analysis_result["max_sharpe_result"]
    price_paths = analysis_result["price_paths"]
    simulation_stats = analysis_result["simulation_stats"]
    asset_metrics_df = analysis_result["asset_metrics_df"]

    st.subheader("Portfolio Lab")
    lab_tabs = st.tabs(["Performance", "Optimization", "Simulation", "Assets"])

    with lab_tabs[0]:
        portfolio_cumulative_fig = plot_cumulative_returns(
            pd.DataFrame({"Portfolio": portfolio_returns}),
            title="Portfolio Cumulative Return",
        )
        st.pyplot(portfolio_cumulative_fig)

        asset_cumulative_fig = plot_cumulative_returns(returns, title="Asset Cumulative Returns")
        with st.expander("Show cumulative returns for individual assets", expanded=False):
            st.pyplot(asset_cumulative_fig)

        drawdown_fig = plot_drawdown(portfolio_returns, title="Portfolio Drawdown")
        st.pyplot(drawdown_fig)

        corr_fig = plot_correlation_heatmap(corr_matrix, title="Correlation Matrix")
        st.pyplot(corr_fig)
        st.dataframe(corr_matrix.round(3), use_container_width=True)

    with lab_tabs[1]:
        compare_metrics = st.columns(4)
        compare_metrics[0].metric("Current Sharpe", f"{metrics['sharpe_ratio']:.3f}")
        compare_metrics[1].metric("Min Var Return", f"{min_var_result.get('expected_return', 0.0):.2%}")
        compare_metrics[2].metric("Max Sharpe Return", f"{max_sharpe_result.get('expected_return', 0.0):.2%}")
        compare_metrics[3].metric("Sampled Portfolios", f"{len(analysis_result['portfolio_cloud'])}")

        opt_tabs = st.tabs(["Minimum Variance", "Maximum Sharpe", "Efficient Frontier", "3D Lab"])
        with opt_tabs[0]:
            if min_var_result["success"]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Return", f"{min_var_result['expected_return']:.2%}")
                c2.metric("Volatility", f"{min_var_result['volatility']:.2%}")
                c3.metric("Sharpe Ratio", f"{min_var_result['sharpe_ratio']:.3f}")
                weights_df = pd.DataFrame(
                    {
                        "Symbol": min_var_result["symbols"],
                        "Weight": [f"{weight:.2%}" for weight in min_var_result["weights"]],
                    }
                )
                st.dataframe(weights_df, use_container_width=True)
            else:
                st.warning("Minimum variance optimization did not converge.")

        with opt_tabs[1]:
            if max_sharpe_result["success"]:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Return", f"{max_sharpe_result['expected_return']:.2%}")
                c2.metric("Volatility", f"{max_sharpe_result['volatility']:.2%}")
                c3.metric("Sharpe Ratio", f"{max_sharpe_result['sharpe_ratio']:.3f}")
                weights_df = pd.DataFrame(
                    {
                        "Symbol": max_sharpe_result["symbols"],
                        "Weight": [f"{weight:.2%}" for weight in max_sharpe_result["weights"]],
                    }
                )
                st.dataframe(weights_df, use_container_width=True)
            else:
                st.warning("Maximum Sharpe optimization did not converge.")

        with opt_tabs[2]:
            if frontier:
                frontier_fig = plot_efficient_frontier(frontier, title="Efficient Frontier")
                st.pyplot(frontier_fig)
            else:
                st.warning("No efficient frontier points were generated.")

        with opt_tabs[3]:
            try:
                tradeoff_fig = plot_portfolio_tradeoff_3d(
                    portfolio_cloud=analysis_result["portfolio_cloud"],
                    frontier_points=frontier,
                    highlighted_portfolios=analysis_result["highlighted_portfolios"],
                )
                st.plotly_chart(tradeoff_fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"3D portfolio view unavailable: {exc}")

    with lab_tabs[2]:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mean final value", f"${simulation_stats['mean']:,.0f}")
        mc2.metric("Median final value", f"${simulation_stats['median']:,.0f}")
        mc3.metric("5th percentile", f"${simulation_stats['percentile_5']:,.0f}")
        mc4.metric("95th percentile", f"${simulation_stats['percentile_95']:,.0f}")

        monte_carlo_fig = plot_monte_carlo_fan(price_paths)
        st.pyplot(monte_carlo_fig)

        with st.expander("Simulation percentile table", expanded=False):
            st.dataframe(analysis_result["simulation_percentiles"].round(2), use_container_width=True)

        try:
            surface_fig = plot_monte_carlo_percentile_surface(price_paths)
            st.plotly_chart(surface_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"3D scenario surface unavailable: {exc}")

    with lab_tabs[3]:
        asset_metrics_view = asset_metrics_df.copy()
        asset_metrics_view["Ann. Return"] = asset_metrics_view["Ann. Return"].map(
            lambda value: f"{value:.2%}"
        )
        asset_metrics_view["Volatility"] = asset_metrics_view["Volatility"].map(
            lambda value: f"{value:.2%}"
        )
        asset_metrics_view["Sharpe"] = asset_metrics_view["Sharpe"].map(lambda value: f"{value:.3f}")
        asset_metrics_view["Max DD"] = asset_metrics_view["Max DD"].map(lambda value: f"{value:.2%}")
        st.dataframe(asset_metrics_view, use_container_width=True)


def _render_reports_page(analysis_result: Dict[str, Any]) -> None:
    ai_review = analysis_result["ai_review"]

    st.subheader("Reports")
    report_tabs = st.tabs(["AI Commentary", "Export Center"])

    with report_tabs[0]:
        if ai_review.get("available", False):
            st.success("Groq AI review generated successfully.")
            if ai_review.get("json_mode_error"):
                st.caption(f"JSON mode fallback used: {ai_review['json_mode_error']}")
        else:
            st.info("AI review unavailable. Showing deterministic fallback text.")
            if ai_review.get("source_detail"):
                st.error(f"AI detail: {ai_review['source_detail']}")

        st.markdown(f"**Summary:** {ai_review.get('summary', '-')}")
        st.markdown(f"**Main Risks:** {ai_review.get('risks', '-')}")
        st.markdown(f"**Improvement Suggestions:** {ai_review.get('improvements', '-')}")
        st.markdown(f"**Final Evaluation:** {ai_review.get('verdict', '-')}")

        if ai_review.get("available", False) and ai_review.get("raw_response"):
            with st.expander("AI raw response", expanded=False):
                st.code(ai_review["raw_response"], language="json")

    with report_tabs[1]:
        _render_export_actions(
            analysis_result,
            title="Export Center",
            body=(
                "Download a polished PDF deck, raw CSV data, or the full JSON payload for this"
                " exact analysis snapshot."
            ),
            compact=False,
            key_namespace="reports_exports",
        )


def _render_modular_dashboard(
    analysis_result: Dict[str, Any],
    preferences: DashboardPreferences,
) -> None:
    _render_dashboard_hero(analysis_result, preferences)

    for warning in analysis_result.get("warnings", []):
        st.warning(warning)

    for ai_message in analysis_result.get("ai_messages", []):
        st.info(ai_message)

    page_keys = [key for key in PAGE_LABELS if key in preferences.visible_pages]
    page_tabs = st.tabs([PAGE_LABELS[key] for key in page_keys])

    for tab, page_key in zip(page_tabs, page_keys, strict=False):
        with tab:
            if page_key == "overview":
                _render_overview_page(analysis_result)
            elif page_key == "cockpit":
                _render_decision_cockpit_page(analysis_result)
            elif page_key == "analysis":
                _render_analysis_lab_page(analysis_result, preferences.show_raw_tables)
            elif page_key == "portfolio_lab":
                _render_portfolio_lab_page(analysis_result)
            elif page_key == "workspace":
                _render_workspace_hub(analysis_result)
            elif page_key == "reports":
                _render_reports_page(analysis_result)


if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "current_portfolio_name" not in st.session_state:
    st.session_state["current_portfolio_name"] = "default"
if "current_portfolio" not in st.session_state:
    st.session_state["current_portfolio"] = load_portfolio(st.session_state["current_portfolio_name"])


with st.sidebar:
    st.header("Portfolio configuration")

    tickers_input = st.text_area(
        "Tickers (one per line)",
        value="\n".join(DEFAULT_TICKERS),
        height=140,
    )

    weights_input = st.text_area(
        "Weights in % (one per line, optional)",
        value="",
        height=140,
        help="If empty, equal weights are used automatically.",
    )

    st.subheader("Data range")
    end_date = st.date_input("End date", value=datetime.now().date())
    start_date = st.date_input(
        "Start date",
        value=(datetime.now() - timedelta(days=365 * 2)).date(),
    )

    risk_free_rate = st.slider(
        "Risk-free rate",
        min_value=0.0,
        max_value=0.10,
        value=0.03,
        step=0.005,
        format="%.3f",
    )

    risk_profile = st.selectbox(
        "Risk profile",
        options=["conservative", "balanced", "aggressive"],
        index=1,
    )

    horizon_days = st.slider(
        "Investment horizon (days)",
        min_value=30,
        max_value=252 * 5,
        value=252,
        step=30,
    )

    n_simulations = st.slider(
        "Monte Carlo simulations",
        min_value=200,
        max_value=5000,
        value=1200,
        step=100,
    )

    portfolio_samples = st.slider(
        "3D sampled portfolios",
        min_value=500,
        max_value=6000,
        value=2500,
        step=250,
    )

    run_clicked = st.button("Evaluate Portfolio", type="primary", use_container_width=True)
    _render_sidebar_portfolio_summary()
    dashboard_preferences = render_dashboard_preferences(
        has_analysis=st.session_state.get("analysis_result") is not None
    )


if run_clicked:
    tickers, ticker_errors = _parse_tickers(tickers_input)
    weights_pct, weight_warnings, weight_errors = _parse_weights(weights_input, len(tickers))

    input_errors = [*ticker_errors, *weight_errors]
    for warning in weight_warnings:
        st.warning(warning)

    if start_date >= end_date:
        input_errors.append("Start date must be before end date.")

    if input_errors:
        for item in input_errors:
            st.error(item)
        st.stop()

    with st.spinner("Running full portfolio analysis..."):
        try:
            analysis_result = _compute_analysis(
                tickers=tickers,
                weights_pct=weights_pct,
                start_date=start_date,
                end_date=end_date,
                risk_free_rate=risk_free_rate,
                risk_profile=risk_profile,
                n_simulations=n_simulations,
                horizon_days=horizon_days,
                portfolio_samples=portfolio_samples,
            )
            st.session_state["analysis_result"] = analysis_result
            st.rerun()
        except Exception as exc:
            st.error(f"Portfolio evaluation failed: {exc}")
            st.stop()


analysis_result = st.session_state.get("analysis_result")

if analysis_result is None:
    _render_empty_dashboard_state(dashboard_preferences)
    st.stop()

_render_modular_dashboard(analysis_result, dashboard_preferences)
