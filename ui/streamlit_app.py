from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))

from ui.economics_questions import render_economics_questions_section
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
from src.analytics.risk_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)
from src.analytics.returns import calculate_annualized_return
from src.data.stock_universe import get_universe, load_universe_metadata, load_universe_snapshot
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


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "VTI",
    "GLD",
    "BND",
]


st.set_page_config(page_title="Quant Platform", layout="wide", page_icon=":bar_chart:")
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

    selected_tickers = selected_rows["Ticker"].astype(str).tolist() if not selected_rows.empty else []
    selected_ticker = selected_tickers[0] if selected_tickers else ""

    q1, q2 = st.columns(2)
    if q1.button(
        "Quick Analyze selected ticker",
        disabled=(not selected_ticker),
        key=f"{key_prefix}_quick_analyze",
        use_container_width=True,
    ) and selected_ticker:
        history = _fetch_ticker_history_cached(selected_ticker)
        if history.empty or "Close" not in history.columns:
            st.warning(f"Could not load historical data for {selected_ticker}.")
        else:
            returns = pd.to_numeric(history["Close"], errors="coerce").pct_change().dropna()
            if returns.empty:
                st.warning(f"Insufficient data for quick analysis of {selected_ticker}.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Annualized Return", f"{calculate_annualized_return(returns):.2%}")
                c2.metric("Volatility", f"{calculate_volatility(returns):.2%}")
                c3.metric("Sharpe", f"{calculate_sharpe_ratio(returns):.3f}")
                c4.metric("Max Drawdown", f"{calculate_max_drawdown(returns):.2%}")

    if q2.button(
        "Quick Chart selected ticker",
        disabled=(not selected_ticker),
        key=f"{key_prefix}_quick_chart",
        use_container_width=True,
    ) and selected_ticker:
        history = _fetch_ticker_history_cached(selected_ticker)
        if history.empty:
            st.warning(f"Could not load chart data for {selected_ticker}.")
        else:
            close_column = "Adj Close" if "Adj Close" in history.columns else "Close"
            if close_column in history.columns:
                st.line_chart(history[close_column].rename(selected_ticker))
            else:
                st.warning(f"Price column not available for {selected_ticker}.")


def _render_stock_picker_tab() -> None:
    st.subheader("Stock Picker")
    st.caption("Two-stage screener: cheap full-universe filtering first, expensive indicators only on shortlist.")

    refresh_col, age_col, auto_col, status_col = st.columns([1, 1, 1.2, 2])
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
            help="When enabled, stale universe snapshots are rebuilt automatically (can block the app rerun).",
        )
    with refresh_col:
        refresh_clicked = st.button(
            "Refresh Universe",
            key="stock_picker_refresh_universe",
            use_container_width=True,
        )

    if refresh_clicked:
        get_universe.clear()
        load_universe_snapshot.clear()
        load_universe_metadata.clear()
        with st.spinner("Refreshing universe snapshot... this can take longer."):
            universe_df = get_universe(max_age_hours=int(max_age_hours), force_refresh=True)
        load_universe_snapshot.clear()
        load_universe_metadata.clear()
    else:
        universe_df = load_universe_snapshot()

    metadata = load_universe_metadata()
    snapshot_stale = _is_universe_snapshot_stale(metadata, int(max_age_hours))
    if auto_refresh_stale and snapshot_stale and not refresh_clicked:
        with st.spinner("Snapshot is stale, rebuilding universe..."):
            universe_df = get_universe(max_age_hours=int(max_age_hours), force_refresh=False)
        load_universe_snapshot.clear()
        load_universe_metadata.clear()
        metadata = load_universe_metadata()
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
            st.caption("Snapshot is stale. You can continue with cached data or run manual refresh.")

    if universe_df.empty:
        st.warning("Universe snapshot is empty. Use 'Refresh Universe' to build it.")
        return

    mode = st.radio(
        "Screener mode",
        options=["Classic Filter Mode", "AI Natural Language Mode"],
        horizontal=True,
        key="stock_picker_mode",
    )

    if mode == "Classic Filter Mode":
        filter_col, result_col = st.columns([1.2, 2.2])
        with filter_col:
            mcap_min, mcap_max = _numeric_bounds(universe_df["MarketCap"] / 1e9, default_min=0.0, default_max=5000.0)
            beta_min, beta_max = _numeric_bounds(universe_df["Beta"], default_min=-1.0, default_max=5.0)
            price_min, price_max = _numeric_bounds(universe_df["Price"], default_min=0.5, default_max=2000.0)
            avg_volume_min, _ = _numeric_bounds(universe_df["AvgVolume"], default_min=0.0, default_max=10_000_000.0)

            sectors = sorted([value for value in universe_df["Sector"].dropna().astype(str).unique() if value.strip()])
            industries = sorted([value for value in universe_df["Industry"].dropna().astype(str).unique() if value.strip()])
            exchanges = sorted([value for value in universe_df["Exchange"].dropna().astype(str).unique() if value.strip()])

            with st.form("classic_screener_form", clear_on_submit=False):
                st.markdown("### Universe Filters")
                market_cap_range_b = st.slider(
                    "Market Cap range ($B)",
                    min_value=float(max(0.0, mcap_min)),
                    max_value=float(max(mcap_max, max(1.0, mcap_min + 1.0))),
                    value=(float(max(0.0, mcap_min)), float(max(mcap_max, max(1.0, mcap_min + 1.0)))),
                )
                beta_range = st.slider(
                    "Beta range",
                    min_value=float(beta_min),
                    max_value=float(beta_max),
                    value=(float(beta_min), float(beta_max)),
                )
                price_range = st.slider(
                    "Price range ($)",
                    min_value=float(max(0.0, price_min)),
                    max_value=float(max(price_max, max(1.0, price_min + 1.0))),
                    value=(float(max(0.0, price_min)), float(max(price_max, max(1.0, price_min + 1.0)))),
                )
                min_avg_volume = st.number_input(
                    "Minimum average volume",
                    min_value=0.0,
                    value=float(max(0.0, avg_volume_min)),
                    step=50_000.0,
                )
                selected_sectors = st.multiselect("Sectors", sectors)
                selected_industries = st.multiselect("Industries", industries)
                selected_exchanges = st.multiselect("Exchanges", exchanges)
                liquidity_prefilter = st.checkbox("Enable liquidity prefilter", value=True)

                with st.expander("Valuation Filters", expanded=False):
                    use_valuation = st.checkbox("Apply valuation filters", value=False)
                    pe_max = st.number_input("Max P/E", min_value=0.0, value=35.0, step=1.0)
                    forward_pe_max = st.number_input("Max Forward P/E", min_value=0.0, value=35.0, step=1.0)
                    peg_max = st.number_input("Max PEG", min_value=0.0, value=3.0, step=0.1)

                with st.expander("Growth Filters", expanded=False):
                    use_growth = st.checkbox("Apply growth filters", value=False)
                    revenue_growth_min = st.number_input("Min Revenue Growth", value=0.05, step=0.01, format="%.3f")
                    earnings_growth_min = st.number_input("Min Earnings Growth", value=0.05, step=0.01, format="%.3f")

                with st.expander("Quality Filters", expanded=False):
                    use_quality = st.checkbox("Apply quality filters", value=False)
                    roe_min = st.number_input("Min ROE", value=0.05, step=0.01, format="%.3f")
                    roa_min = st.number_input("Min ROA", value=0.02, step=0.01, format="%.3f")

                with st.expander("Momentum & Dividend Filters", expanded=False):
                    use_momentum = st.checkbox("Apply momentum filters", value=False)
                    return_52w_min = st.number_input("Min 52w Return", value=0.0, step=0.02, format="%.3f")
                    use_dividend = st.checkbox("Apply dividend filters", value=False)
                    dividend_yield_min = st.number_input("Min Dividend Yield", value=0.0, step=0.005, format="%.4f")

                with st.expander("Ranking Weights", expanded=False):
                    value_weight = st.slider("Value weight", 0.0, 3.0, 1.0, 0.1)
                    growth_weight = st.slider("Growth weight", 0.0, 3.0, 1.0, 0.1)
                    quality_weight = st.slider("Quality weight", 0.0, 3.0, 1.0, 0.1)
                    momentum_weight = st.slider("Momentum weight", 0.0, 3.0, 1.0, 0.1)
                    stability_weight = st.slider("Stability weight", 0.0, 3.0, 1.0, 0.1)
                    dividend_weight = st.slider("Dividend weight", 0.0, 3.0, 0.5, 0.1)

                technical_limit = st.slider("Technical indicator shortlist size", 25, 400, 150, 25)
                run_classic = st.form_submit_button("Run Classic Screen", type="primary", use_container_width=True)

            if run_classic:
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

                filtered = apply_classic_filters(
                    df=universe_df,
                    market_cap_range=(market_cap_range_b[0] * 1e9, market_cap_range_b[1] * 1e9),
                    sectors=selected_sectors,
                    industries=selected_industries,
                    exchanges=selected_exchanges,
                    beta_range=beta_range,
                    price_range=price_range,
                    min_avg_volume=float(min_avg_volume),
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

                ranked = rank_stocks(calculate_quant_score(filtered, weighted), sort_by="QuantScore", ascending=False)
                shortlist_size = min(int(technical_limit), len(ranked))
                if shortlist_size > 0:
                    with st.spinner("Calculating technical indicators for shortlist..."):
                        technical = apply_technical_indicators(ranked.head(shortlist_size))
                    technical_cols = [column for column in ["Ticker", "RSI", "MACD", "Volatility", "Drawdown"] if column in technical.columns]
                    ranked = ranked.merge(technical[technical_cols], on="Ticker", how="left")
                    ranked = rank_stocks(calculate_quant_score(ranked, weighted), sort_by="QuantScore", ascending=False)

                st.session_state["stock_picker_results_classic"] = ranked
                st.session_state["stock_picker_classic_info"] = (
                    f"Stage 1: {len(filtered):,} matches | "
                    f"Stage 2: indicators on top {shortlist_size:,} symbols"
                )

        with result_col:
            st.markdown("### Classic Results")
            if st.session_state.get("stock_picker_classic_info"):
                st.caption(st.session_state["stock_picker_classic_info"])
            classic_results = st.session_state.get("stock_picker_results_classic", pd.DataFrame())
            selected_rows = _render_selectable_results(classic_results, "classic")
            if not classic_results.empty:
                _render_screener_bulk_actions(selected_rows, classic_results, "classic")

    else:
        query = st.text_area(
            "Describe your stock screen in plain English",
            height=160,
            placeholder="Example: Find profitable large-cap semiconductor stocks with low debt, strong ROE, and positive momentum.",
            key="stock_picker_ai_query",
        )
        analyze_clicked = st.button("Analyze with Groq", type="primary", key="stock_picker_ai_run", use_container_width=True)
        if analyze_clicked:
            with st.spinner("Parsing request and applying AI filters..."):
                parsed = parse_ai_query(query)
                results, explanation = apply_ai_query(query, universe_df, parsed_query=parsed)
                st.session_state["stock_picker_ai_parsed"] = parsed
                st.session_state["stock_picker_ai_explanation"] = explanation
                st.session_state["stock_picker_results_ai"] = results

        parsed_payload = st.session_state.get("stock_picker_ai_parsed")
        if parsed_payload:
            with st.expander("Parsed Filters (JSON)", expanded=True):
                st.json(parsed_payload)
        explanation_text = st.session_state.get("stock_picker_ai_explanation")
        if explanation_text:
            st.info(explanation_text)

        ai_results = st.session_state.get("stock_picker_results_ai", pd.DataFrame())
        st.markdown("### AI Results")
        selected_rows = _render_selectable_results(ai_results, "ai")
        if not ai_results.empty:
            _render_screener_bulk_actions(selected_rows, ai_results, "ai")


def _render_portfolio_tracker_tab() -> None:
    st.subheader("Portfolio Tracker")
    st.caption("Holdings are persisted as JSON files under data/portfolios/ and synced with session_state.")

    available_portfolios = list_portfolios()
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
        st.session_state["current_portfolio"] = load_portfolio(selected_name)
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
        save_portfolio(st.session_state["current_portfolio"], st.session_state.get("current_portfolio_name", current_name))
        st.success("Portfolio saved.")
    if s2.button("Save As New JSON", key="portfolio_tracker_save_as", use_container_width=True):
        save_portfolio(st.session_state["current_portfolio"], save_as_name)
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

    portfolio = st.session_state.get("current_portfolio", load_portfolio("default"))
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

    trades = refresh_swing_trades()
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
                    save_swing_trade_book(updated_trades)
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
                    save_swing_trade_book(updated_trades)
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
    standalone_tabs = st.tabs(["Stock Picker", "Portfolio Tracker", "Swing Tracker"])
    with standalone_tabs[0]:
        _render_stock_picker_tab()
    with standalone_tabs[1]:
        _render_portfolio_tracker_tab()
    with standalone_tabs[2]:
        _render_swing_tracker_tab()
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
model_results = analysis_result.get("model_results", {})
signal_results = analysis_result.get("signal_results", {})
summary_result = analysis_result.get("summary_result")
news_result = analysis_result.get("news_result")
backtest_result = analysis_result.get("backtest_result", {})
history_records = analysis_result.get("history_records", [])


st.header("Modular Dashboard")
dashboard_tabs = st.tabs(
    [
        "Data",
        "Models",
        "Signals",
        "Backtest",
        "News",
        "Summary",
        "History",
        "Compare",
        "Stock Picker",
        "Portfolio Tracker",
        "Swing Tracker",
    ]
)

with dashboard_tabs[0]:
    st.subheader("Data snapshot")
    st.caption(f"Run record: {analysis_result.get('run_record').run_id if analysis_result.get('run_record') else '-'}")
    st.dataframe(analysis_result["prices"].tail(20), use_container_width=True)
    st.dataframe(analysis_result["returns"].tail(20), use_container_width=True)

with dashboard_tabs[1]:
    st.subheader("Model outputs")
    model_rows: List[Dict[str, Any]] = []
    for model_name, model in model_results.items():
        metrics_map = getattr(model, "metrics", {})
        model_rows.append({
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
            "Band Width": float(metrics_map.get("posterior_interval_width", metrics_map.get("forecast_spread", 0.0))),
            "Error": getattr(model, "error", ""),
        })
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
        confidence_frame = pd.DataFrame(model_rows)[["Model", "Confidence"]].set_index("Model")
        st.caption("Model confidence")
        st.bar_chart(confidence_frame)

with dashboard_tabs[2]:
    st.subheader("Signal outputs")
    signal_rows: List[Dict[str, Any]] = []
    for signal_name, signal in signal_results.items():
        signal_rows.append({
            "Signal": signal_name,
            "Family": getattr(signal, "family", ""),
            "Direction": getattr(signal, "direction", "neutral"),
            "Score": float(getattr(signal, "score", 0.0)),
            "Confidence": float(getattr(signal, "confidence", 0.0)),
            "Available": bool(getattr(signal, "available", False)),
            "Error": getattr(signal, "error", ""),
        })
    if signal_rows:
        st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

with dashboard_tabs[3]:
    st.subheader("Deterministic backtest")
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

with dashboard_tabs[4]:
    st.subheader("News relevance")
    if news_result and getattr(news_result, "available", False):
        n_col1, n_col2 = st.columns(2)
        n_col1.metric("Aggregate sentiment", f"{float(getattr(news_result, 'sentiment_score', 0.0)):+.3f}")
        n_col2.metric("Sentiment dispersion", f"{float(getattr(news_result, 'sentiment_dispersion', 0.0)):.3f}")
        st.caption(f"Provider used: {str(getattr(news_result, 'context', {}).get('provider_used', 'unknown'))}")

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
        st.info(f"News layer unavailable: {getattr(news_result, 'error', 'n/a') if news_result else 'n/a'}")

with dashboard_tabs[5]:
    st.subheader("Summary layer")
    if summary_result:
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Composite Score", f"{getattr(summary_result, 'composite_score', 0.0):.3f}")
        s_col2.metric("Regime", str(getattr(summary_result, "regime_label", "neutral")))
        s_col3.metric("Confidence", f"{getattr(summary_result, 'confidence', 0.0):.3f}")
        s_col4.metric("News Sentiment", f"{float(getattr(summary_result, 'news_sentiment', 0.0)):+.3f}")
        st.metric("News Sentiment Dispersion", f"{float(getattr(summary_result, 'news_sentiment_dispersion', 0.0)):.3f}")
        for line in getattr(summary_result, "highlights", []):
            st.write(f"- {line}")
        st.caption(str(getattr(summary_result, "regime_interpretation", "")))
        st.write(f"Expected return view: {float(getattr(summary_result, 'expected_return_view', 0.0)):.2%}")
        st.write(f"Expected risk view: {float(getattr(summary_result, 'expected_risk_view', 0.0)):.2%}")
        st.write(f"Drawdown implication: {getattr(summary_result, 'drawdown_implication', '')}")
        st.write(f"Volatility implication: {getattr(summary_result, 'volatility_implication', '')}")
        st.write(f"News implication: {getattr(summary_result, 'news_implication', '')}")
        strongest = getattr(summary_result, "strongest_signals", [])
        if strongest:
            st.caption("Strongest signals")
            st.dataframe(pd.DataFrame(strongest), use_container_width=True, hide_index=True)
        top_news = getattr(summary_result, "top_relevant_news", [])
        if top_news:
            st.caption("Top 3 relevant news")
            st.dataframe(pd.DataFrame(top_news), use_container_width=True, hide_index=True)
        changes = getattr(summary_result, "recent_changes", [])
        if changes:
            st.caption("Recent changes vs prior run")
            for change in changes:
                st.write(f"- {change}")
        for warning in getattr(summary_result, "warnings", []):
            st.warning(warning)
        for flag in getattr(summary_result, "risk_flags", []):
            st.warning(flag)

with dashboard_tabs[6]:
    st.subheader("Run history")
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

with dashboard_tabs[7]:
    st.subheader("Run comparison")
    run_ids = [item.get("run_id") for item in history_records if item.get("run_id")]
    if len(run_ids) >= 2:
        left_run = st.selectbox("Base run", options=run_ids, index=min(1, len(run_ids) - 1))
        right_run = st.selectbox("Compare run", options=run_ids, index=0)
        if left_run and right_run and left_run != right_run:
            try:
                left_data = load_run_record(left_run)
                right_data = load_run_record(right_run)
                comparison = compare_runs(left_data, right_data)
                st.dataframe(pd.DataFrame([comparison["metric_diff"]]), use_container_width=True, hide_index=True)
                if comparison.get("summary_diff"):
                    st.caption("Summary deltas")
                    st.dataframe(pd.DataFrame([comparison["summary_diff"]]), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Comparison failed: {exc}")
    else:
        st.info("Need at least two saved runs for comparison.")

with dashboard_tabs[8]:
    _render_stock_picker_tab()

with dashboard_tabs[9]:
    _render_portfolio_tracker_tab()

with dashboard_tabs[10]:
    _render_swing_tracker_tab()

st.markdown("---")

render_economics_questions_section(analysis_result)

st.markdown("---")

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
