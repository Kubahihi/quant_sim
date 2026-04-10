from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.ai import generate_ai_review, resolve_groq_api_key
from src.analytics import (
    build_deterministic_fallback_review,
    build_portfolio_timeseries,
    calculate_average_correlation,
    calculate_concentration_metrics,
    calculate_correlation_matrix,
    calculate_portfolio_core_metrics,
    calculate_portfolio_daily_returns,
    evaluate_portfolio_score,
    run_advanced_models,
)
from src.analytics.risk_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)
from src.analytics.returns import calculate_annualized_return
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


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "VTI",
    "GLD",
    "BND",
]


st.set_page_config(page_title="Quant Platform", layout="wide", page_icon=":bar_chart:")
st.title("Quant Platform v0.2")
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
        "warnings": alignment_warnings,
        "missing_tickers": missing_tickers,
    }


if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None


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
        except Exception as exc:
            st.error(f"Portfolio evaluation failed: {exc}")
            st.stop()


analysis_result = st.session_state.get("analysis_result")

if analysis_result is None:
    st.info("Configure the portfolio in the sidebar and click 'Evaluate Portfolio'.")
    st.stop()


for warning in analysis_result.get("warnings", []):
    st.warning(warning)

for ai_message in analysis_result.get("ai_messages", []):
    st.info(ai_message)


metrics = analysis_result["metrics"]
score_result = analysis_result["score_result"]
portfolio_returns = analysis_result["portfolio_returns"]
corr_matrix = analysis_result["correlation_matrix"]
frontier = analysis_result["frontier"]
min_var_result = analysis_result["min_var_result"]
max_sharpe_result = analysis_result["max_sharpe_result"]
price_paths = analysis_result["price_paths"]
simulation_stats = analysis_result["simulation_stats"]
ai_review = analysis_result["ai_review"]
returns = analysis_result["returns"]
asset_metrics_df = analysis_result["asset_metrics_df"]
advanced_models = analysis_result.get("advanced_models", {})


st.header("Portfolio score")
col1, col2, col3 = st.columns(3)
col1.metric("Deterministic score", f"{score_result['score']} / 100")
col2.metric("Rating", str(score_result["rating"]))
col3.metric("Flags", len(score_result["flags"]))
st.progress(int(score_result["score"]))

if score_result["flags"]:
    for flag in score_result["flags"]:
        st.warning(flag)
else:
    st.success("No critical deterministic flags detected.")

if score_result["breakdown"]:
    score_breakdown_df = pd.DataFrame(score_result["breakdown"]).rename(columns={
        "rule": "Rule",
        "penalty": "Penalty",
        "detail": "Detail",
    })
    st.caption("Score breakdown")
    st.dataframe(score_breakdown_df, use_container_width=True, hide_index=True)
else:
    st.caption("No score penalties were triggered.")

st.markdown("---")


st.header("Metrics overview")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
metric_col2.metric("Volatility", f"{metrics['volatility']:.2%}")
metric_col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
metric_col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
metric_col5.metric("Average Daily Return", f"{metrics['daily_return_mean']:.3%}")
metric_col6.metric("Concentration (HHI)", f"{metrics['hhi']:.3f}")
metric_col7.metric("Effective Holdings", f"{metrics['effective_holdings']:.2f}")
metric_col8.metric("Average Correlation", f"{metrics['avg_correlation']:.3f}")

st.markdown("---")


st.header("Advanced Models (V0.3 / V0.4)")
advanced_rows: List[Dict[str, Any]] = []
for model_name, result in advanced_models.items():
    if result.get("available", False):
        metrics_map = result.get("metrics", {})
        advanced_rows.append({
            "Model": model_name,
            "Status": "ok",
            "Signal 1": round(float(metrics_map.get("expected_annual_return", metrics_map.get("next_period_return_forecast", metrics_map.get("conditional_volatility", 0.0))) or 0.0), 6),
            "Signal 2": round(float(metrics_map.get("trend_slope_daily", metrics_map.get("forecast_confidence", metrics_map.get("volatility_annualized", 0.0))) or 0.0), 6),
            "Confidence": round(float(metrics_map.get("confidence", 0.0) or 0.0), 4),
            "Error": "",
        })
    else:
        advanced_rows.append({
            "Model": model_name,
            "Status": "unavailable",
            "Signal 1": np.nan,
            "Signal 2": np.nan,
            "Confidence": np.nan,
            "Error": result.get("error", ""),
        })

if advanced_rows:
    st.dataframe(pd.DataFrame(advanced_rows), use_container_width=True, hide_index=True)

st.markdown("---")


st.header("Portfolio performance")
portfolio_cumulative_fig = plot_cumulative_returns(
    pd.DataFrame({"Portfolio": portfolio_returns}),
    title="Portfolio Cumulative Return",
)
st.pyplot(portfolio_cumulative_fig)

asset_cumulative_fig = plot_cumulative_returns(returns, title="Asset Cumulative Returns")
with st.expander("Show cumulative returns for individual assets", expanded=False):
    st.pyplot(asset_cumulative_fig)


st.header("Drawdown")
drawdown_fig = plot_drawdown(portfolio_returns, title="Portfolio Drawdown")
st.pyplot(drawdown_fig)


st.header("Correlation matrix")
corr_fig = plot_correlation_heatmap(corr_matrix, title="Correlation Matrix")
st.pyplot(corr_fig)
st.dataframe(corr_matrix.round(3), use_container_width=True)

st.markdown("---")


st.header("Portfolio optimization")
tab1, tab2, tab3 = st.tabs(["Minimum Variance", "Maximum Sharpe", "Efficient Frontier"])

with tab1:
    if min_var_result["success"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{min_var_result['expected_return']:.2%}")
        c2.metric("Volatility", f"{min_var_result['volatility']:.2%}")
        c3.metric("Sharpe Ratio", f"{min_var_result['sharpe_ratio']:.3f}")
        weights_df = pd.DataFrame({
            "Symbol": min_var_result["symbols"],
            "Weight": [f"{weight:.2%}" for weight in min_var_result["weights"]],
        })
        st.dataframe(weights_df, use_container_width=True)
    else:
        st.warning("Minimum variance optimization did not converge.")

with tab2:
    if max_sharpe_result["success"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Return", f"{max_sharpe_result['expected_return']:.2%}")
        c2.metric("Volatility", f"{max_sharpe_result['volatility']:.2%}")
        c3.metric("Sharpe Ratio", f"{max_sharpe_result['sharpe_ratio']:.3f}")
        weights_df = pd.DataFrame({
            "Symbol": max_sharpe_result["symbols"],
            "Weight": [f"{weight:.2%}" for weight in max_sharpe_result["weights"]],
        })
        st.dataframe(weights_df, use_container_width=True)
    else:
        st.warning("Maximum Sharpe optimization did not converge.")

with tab3:
    if frontier:
        frontier_fig = plot_efficient_frontier(frontier, title="Efficient Frontier")
        st.pyplot(frontier_fig)
    else:
        st.warning("No efficient frontier points were generated.")

st.markdown("---")


st.header("3D visualizations")
viz_tab1, viz_tab2 = st.tabs(["Portfolio Lab", "Scenario Surface"])

with viz_tab1:
    try:
        tradeoff_fig = plot_portfolio_tradeoff_3d(
            portfolio_cloud=analysis_result["portfolio_cloud"],
            frontier_points=frontier,
            highlighted_portfolios=analysis_result["highlighted_portfolios"],
        )
        st.plotly_chart(tradeoff_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"3D portfolio view unavailable: {exc}")

with viz_tab2:
    try:
        surface_fig = plot_monte_carlo_percentile_surface(price_paths)
        st.plotly_chart(surface_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"3D scenario surface unavailable: {exc}")

st.markdown("---")


st.header("Monte Carlo simulation")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Mean final value", f"${simulation_stats['mean']:,.0f}")
mc2.metric("Median final value", f"${simulation_stats['median']:,.0f}")
mc3.metric("5th percentile", f"${simulation_stats['percentile_5']:,.0f}")
mc4.metric("95th percentile", f"${simulation_stats['percentile_95']:,.0f}")

monte_carlo_fig = plot_monte_carlo_fan(price_paths)
st.pyplot(monte_carlo_fig)

with st.expander("Simulation percentile table", expanded=False):
    st.dataframe(analysis_result["simulation_percentiles"].round(2), use_container_width=True)

st.markdown("---")


st.header("Individual asset metrics")
asset_metrics_view = asset_metrics_df.copy()
asset_metrics_view["Ann. Return"] = asset_metrics_view["Ann. Return"].map(lambda value: f"{value:.2%}")
asset_metrics_view["Volatility"] = asset_metrics_view["Volatility"].map(lambda value: f"{value:.2%}")
asset_metrics_view["Sharpe"] = asset_metrics_view["Sharpe"].map(lambda value: f"{value:.3f}")
asset_metrics_view["Max DD"] = asset_metrics_view["Max DD"].map(lambda value: f"{value:.2%}")
st.dataframe(asset_metrics_view, use_container_width=True)

st.markdown("---")


st.header("AI Commentary")
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

st.markdown("---")


st.header("Export panel")
report_payload = _build_report_payload(analysis_result)
pdf_bytes: bytes | None = None
csv_bytes: bytes | None = None
json_bytes: bytes | None = None

try:
    pdf_figures = _build_pdf_figures(analysis_result)
    pdf_buffer = generate_pdf_report(report_payload, pdf_figures)
    pdf_bytes = pdf_buffer.getvalue()
    for figure in pdf_figures.values():
        plt.close(figure)
except Exception as exc:
    st.error(f"PDF export could not be prepared: {exc}")

try:
    csv_bytes = export_portfolio_data_csv(report_payload)
except Exception as exc:
    st.error(f"Data CSV export failed: {exc}")

try:
    json_bytes = export_full_report_json(report_payload)
except Exception as exc:
    st.error(f"Full JSON export failed: {exc}")

export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    if pdf_bytes is not None:
        st.download_button(
            "Export PDF",
            data=pdf_bytes,
            file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
with export_col2:
    if csv_bytes is not None:
        st.download_button(
            "Export Data (CSV)",
            data=csv_bytes,
            file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
with export_col3:
    if json_bytes is not None:
        st.download_button(
            "Export Full Report (JSON)",
            data=json_bytes,
            file_name=f"portfolio_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
