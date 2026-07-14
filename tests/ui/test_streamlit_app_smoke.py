from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.analytics.modular.results import NewsResult, RunRecord, SummaryResult


matplotlib.use("Agg")


APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"


def _sample_prices(symbols: list[str], periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    rows: dict[str, np.ndarray] = {}
    for index, symbol in enumerate(symbols, start=1):
        drift = 0.0005 + index * 0.00004
        seasonal = np.sin(np.linspace(0, 4, periods)) * (0.0015 + index * 0.0001)
        returns = drift + seasonal
        rows[symbol] = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(rows, index=dates)


def _fake_quant_stack(tmp_path: Path):
    def _runner(portfolio_returns: pd.Series, returns_df: pd.DataFrame, config: dict, history_dir: str = "data/run_history") -> dict:
        summary = SummaryResult(
            generated_at=datetime.now(timezone.utc).isoformat(),
            composite_score=0.12,
            regime_label="neutral",
            confidence=0.58,
            highlights=["Neutral regime with moderate conviction."],
            model_snapshot={},
            signal_snapshot={},
            risk_flags=["Signals are weak: composite score is close to neutral."],
            strongest_signals=[],
            agreement_score=0.62,
            disagreement_score=0.38,
            uncertainty=0.42,
            expected_return_view=0.11,
            expected_risk_view=0.13,
            regime_interpretation="Signals are mixed but stable.",
            drawdown_implication="Drawdown profile is manageable.",
            volatility_implication="Volatility remains within expected bounds.",
            recent_changes=[],
            news_sentiment=0.0,
            news_sentiment_dispersion=0.0,
            top_relevant_news=[],
            news_implication="No major news pressure in the smoke test context.",
            warnings=[
                "Signals are weak: composite score is close to neutral.",
                "Signals are weak: composite score is close to neutral.",
            ],
        )
        news = NewsResult(
            available=True,
            items=[],
            context={"provider_used": "test", "relevance_coverage": 0.0},
            sentiment_score=0.0,
            sentiment_dispersion=0.0,
        )
        run_record = RunRecord.now(
            run_id="run_test_ui_001",
            config=config,
            universe=config.get("tickers", []),
            date_range={
                "start": str(config.get("start_date", "")),
                "end": str(config.get("end_date", "")),
            },
            outputs={},
            metrics={},
            summary=summary.to_dict(),
            news=news.to_dict(),
            sentiment={},
        )
        backtest_index = portfolio_returns.index[: min(5, len(portfolio_returns))]
        backtest_series = pd.Series(np.linspace(1.0, 1.04, len(backtest_index)), index=backtest_index)
        drawdown_series = pd.Series(np.linspace(0.0, -0.02, len(backtest_index)), index=backtest_index)
        return {
            "models": {},
            "signals": {},
            "summary": summary,
            "news": news,
            "backtest": {
                "metrics": {
                    "total_return": 0.04,
                    "volatility": 0.10,
                    "sharpe": 0.40,
                    "max_drawdown": -0.02,
                },
                "equity_curve": backtest_series,
                "drawdown": drawdown_series,
                "lookahead_safe": True,
            },
            "run_record": run_record,
            "history_path": str(tmp_path / "run_test_ui_001.json"),
        }

    return _runner


def test_streamlit_app_shows_workspace_by_default():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert any(item.value == "Workspace Hub" for item in at.subheader)


def test_wharton_cockpit_exposes_rules_and_portfolio_tabs():
    at = AppTest.from_file(str(APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Wharton Cockpit"
    at.session_state["wharton_user_profile_v2"] = {
        "id": 1,
        "username": "Jakub",
        "role": "Captain/Quant",
        "primary_module": "Quant Engine",
    }
    at.session_state["wharton_company_analysis_v1"] = {
        "MSFT": {
            "ticker": "MSFT",
            "fetched_at": "2026-07-14T12:00:00+00:00",
            "info": {
                "longName": "Microsoft Corporation",
                "longBusinessSummary": "Software and cloud services company.",
                "currentPrice": 100.0,
                "marketCap": 1_000_000_000_000,
                "freeCashflow": 10_000_000_000,
                "sharesOutstanding": 1_000_000_000,
                "totalCash": 5_000_000_000,
                "totalDebt": 1_000_000_000,
                "revenueGrowth": 0.10,
                "earningsGrowth": 0.12,
                "grossMargins": 0.60,
                "operatingMargins": 0.30,
                "returnOnEquity": 0.25,
            },
            "metrics": {"currentPrice": 100.0, "marketCap": 1_000_000_000_000},
            "officers": [{"name": "Test CEO", "title": "CEO"}],
            "history": pd.DataFrame({"Close": [90.0, 100.0]}),
            "news": [],
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
            "quarterly_income_statement": pd.DataFrame(),
            "quarterly_balance_sheet": pd.DataFrame(),
            "quarterly_cash_flow": pd.DataFrame(),
        }
    }
    at.run(timeout=60)

    assert len(at.exception) == 0
    tab_labels = [tab.label for tab in at.tabs]
    assert "Assignment & Rules" in tab_labels
    assert "Portfolio Tracker" in tab_labels
    assert "Company Analysis" in tab_labels

    assignment_tab = next(tab for tab in at.tabs if tab.label == "Assignment & Rules")
    portfolio_tab = next(tab for tab in at.tabs if tab.label == "Portfolio Tracker")
    company_tab = next(tab for tab in at.tabs if tab.label == "Company Analysis")

    assignment_headings = [item.value for item in assignment_tab.markdown]
    portfolio_headings = [item.value for item in portfolio_tab.markdown]
    company_headings = [item.value for item in company_tab.markdown]
    assert any("Assignment & Rules" in value for value in assignment_headings)
    assert any("Portfolio Tracker" in value for value in portfolio_headings)
    assert any("Company Analysis" in value for value in company_headings)
    assert not any("Company Analysis" in value for value in portfolio_headings)


def test_streamlit_app_evaluate_flow_renders_both_export_sections(monkeypatch, tmp_path):
    import src.ai
    import src.analytics
    import src.data.fetchers.yahoo_fetcher
    import src.optimization
    import src.simulation

    monkeypatch.setattr(
        src.data.fetchers.yahoo_fetcher.YahooFetcher,
        "fetch_close_prices",
        lambda self, symbols, start_date, end_date: _sample_prices(list(symbols)),
    )
    monkeypatch.setattr(
        src.ai,
        "generate_ai_review",
        lambda payload, api_key=None: {
            "available": False,
            "error": "Smoke test fallback",
        },
    )
    monkeypatch.setattr(src.analytics, "run_advanced_models", lambda returns, forecast_periods, returns_df: {})
    monkeypatch.setattr(src.analytics, "run_quant_stack", _fake_quant_stack(tmp_path))
    monkeypatch.setattr(src.analytics, "list_run_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        src.optimization,
        "optimize_minimum_variance",
        lambda returns: {
            "success": True,
            "symbols": list(returns.columns),
            "weights": [1.0 / len(returns.columns)] * len(returns.columns),
            "expected_return": 0.09,
            "volatility": 0.12,
            "sharpe_ratio": 0.50,
        },
    )
    monkeypatch.setattr(
        src.optimization,
        "optimize_maximum_sharpe",
        lambda returns, risk_free_rate=0.0: {
            "success": True,
            "symbols": list(returns.columns),
            "weights": [1.0 / len(returns.columns)] * len(returns.columns),
            "expected_return": 0.11,
            "volatility": 0.13,
            "sharpe_ratio": 0.62,
        },
    )
    monkeypatch.setattr(src.optimization, "calculate_efficient_frontier", lambda returns, n_points=30: [])
    monkeypatch.setattr(src.optimization, "sample_portfolio_cloud", lambda returns, n_samples, risk_free_rate=0.0: [])
    monkeypatch.setattr(
        src.simulation,
        "run_monte_carlo_simulation",
        lambda current_value, expected_return, volatility, time_horizon, n_simulations: (
            np.array(
                [
                    [100000.0, 100000.0, 100000.0],
                    [101500.0, 100800.0, 102200.0],
                    [102300.0, 101200.0, 103100.0],
                    [103100.0, 101900.0, 104000.0],
                ]
            ),
            {
                "mean": 103000.0,
                "median": 103100.0,
                "percentile_5": 101000.0,
                "percentile_95": 104000.0,
            },
        ),
    )

    at = AppTest.from_file(str(APP_PATH))
    at.session_state["dashboard_layout_preset"] = "Focused"
    at.session_state["dashboard_layout_preset_auto"] = False
    at.session_state["dashboard_layout_preset_applied"] = "Focused"
    at.session_state["dashboard_visible_pages_selector"] = ["overview", "reports"]
    at.run(timeout=60)

    next(button for button in at.button if button.label == "Evaluate Portfolio").click()
    at.run(timeout=60)

    assert len(at.exception) == 0
    markdown_values = [item.value for item in at.markdown]
    assert "### Quick Exports" in markdown_values
    assert "### Export Center" in markdown_values
    warning_values = [item.value for item in at.warning]
    assert warning_values.count("Signals are weak: composite score is close to neutral.") == 1
