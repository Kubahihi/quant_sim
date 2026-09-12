from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.analytics.modular.results import NewsResult, RunRecord, SummaryResult


matplotlib.use("Agg")


APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"
QUANT_APP_PATH = Path(__file__).resolve().parents[2] / "ui" / "quant_platform.py"


def _enable_test_auto_login(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_SIM_ENV", "test")
    monkeypatch.setenv("QUANT_SIM_TEST_AUTO_LOGIN", "1")


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
    def _runner(
        portfolio_returns: pd.Series,
        returns_df: pd.DataFrame,
        config: dict,
        history_dir: str = "data/run_history",
        precomputed_models=None,
        user_id=None,
        team_connection_factory=None,
    ) -> dict:
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


def test_pytest_context_alone_does_not_bypass_login(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "development")
    monkeypatch.delenv("QUANT_SIM_TEST_AUTO_LOGIN", raising=False)

    at = AppTest.from_file(str(APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Quant Platform"
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert any("Welcome back." in item.value for item in at.markdown)
    assert not any(item.value == "Workspace Hub" for item in at.subheader)


def test_streamlit_app_defaults_to_wharton_cockpit(monkeypatch):
    _enable_test_auto_login(monkeypatch)
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert any("Wharton Cockpit" in item.value for item in at.markdown)
    assert not any(item.value == "Workspace Hub" for item in at.subheader)


def test_judge_session_is_forced_to_isolated_wharton_view(monkeypatch):
    from ui.pages import wharton_dash

    profile = {
        "id": 5,
        "username": "judge",
        "role": "Judge",
        "primary_module": "Judge View",
    }
    monkeypatch.setattr(wharton_dash, "init_db", lambda: None)
    monkeypatch.setattr(wharton_dash, "_get_current_profile", lambda: profile)
    monkeypatch.setattr(wharton_dash, "_render_header", lambda profile: None)
    monkeypatch.setattr(
        wharton_dash,
        "_render_judge_view",
        lambda profile: st.markdown("JUDGE_VIEW_SENTINEL"),
    )
    monkeypatch.setattr(
        wharton_dash,
        "_render_cockpit_navigation",
        lambda *args, **kwargs: pytest.fail("judge reached team navigation"),
    )

    at = AppTest.from_file(str(APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Quant Platform"
    at.session_state["wharton_user_profile_v2"] = profile
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert at.session_state["quant_sim_workspace_route"] == "Wharton Cockpit"
    assert any(item.value == "JUDGE_VIEW_SENTINEL" for item in at.markdown)
    assert not any(item.label == "Choose workspace" for item in at.radio)
    assert not any(item.value == "Workspace Hub" for item in at.subheader)


def test_direct_quant_entrypoint_keeps_judge_in_read_only_view(monkeypatch):
    from ui.pages import wharton_dash

    profile = {
        "id": 5,
        "username": "judge",
        "role": "Judge",
        "primary_module": "Judge View",
    }
    monkeypatch.setattr(wharton_dash, "init_db", lambda: None)
    monkeypatch.setattr(wharton_dash, "_get_current_profile", lambda: profile)
    monkeypatch.setattr(wharton_dash, "_render_header", lambda profile: None)
    monkeypatch.setattr(
        wharton_dash,
        "_render_judge_view",
        lambda profile: st.markdown("DIRECT_JUDGE_VIEW_SENTINEL"),
    )
    monkeypatch.setattr(
        wharton_dash,
        "_render_cockpit_navigation",
        lambda *args, **kwargs: pytest.fail("judge reached team navigation"),
    )

    at = AppTest.from_file(str(QUANT_APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Quant Platform"
    at.session_state["wharton_user_profile_v2"] = profile
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert at.session_state["quant_sim_workspace_route"] == "Wharton Cockpit"
    assert any(item.value == "DIRECT_JUDGE_VIEW_SENTINEL" for item in at.markdown)
    assert not any(item.label == "Choose workspace" for item in at.radio)
    assert not any(item.value == "Workspace Hub" for item in at.subheader)


def test_streamlit_app_loads_default_portfolio_only_once(monkeypatch):
    _enable_test_auto_login(monkeypatch)
    import src.portfolio_tracker.manager

    load_calls: list[tuple[str, object]] = []

    def fake_load_portfolio(name="default", user_id=None):
        load_calls.append((name, user_id))
        return {"name": name, "positions": []}

    monkeypatch.setattr(
        src.portfolio_tracker.manager,
        "load_portfolio",
        fake_load_portfolio,
    )

    at = AppTest.from_file(str(APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Quant Platform"
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert load_calls == [("default", None)]


def test_wharton_cockpit_groups_and_lazily_renders_panels(monkeypatch, tmp_path):
    from ui.pages import wharton_dash

    monkeypatch.setenv("QUANT_SIM_ENV", "test")
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setattr(wharton_dash, "DB_PATH", tmp_path / "wharton.db")
    monkeypatch.setattr(wharton_dash, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(wharton_dash, "resolve_wharton_credentials", lambda *args, **kwargs: dict.fromkeys(wharton_dash.REQUIRED_WHARTON_USERS, "smoke-only-password"))
    monkeypatch.setattr(wharton_dash, "resolve_wharton_judge_credential", lambda *args, **kwargs: None)
    wharton_dash.init_db()
    profile = wharton_dash.authenticate_user("Jakub", "smoke-only-password")
    assert profile is not None

    automatic_peer_info = {
        "ORCL": {
            "shortName": "Oracle",
            "sector": "Technology",
            "industry": "Software - Infrastructure",
            "marketCap": 450_000_000_000,
            "operatingMargins": 0.30,
            "revenueGrowth": 0.08,
            "forwardPE": 24.0,
        },
        "CRM": {
            "shortName": "Salesforce",
            "sector": "Technology",
            "industry": "Software - Application",
            "marketCap": 300_000_000_000,
            "operatingMargins": 0.20,
            "revenueGrowth": 0.11,
            "forwardPE": 26.0,
        },
    }
    enrichment_calls: list[str] = []
    monkeypatch.setattr(
        wharton_dash,
        "_discover_automatic_peers_cached",
        lambda ticker, target_info, max_peers=6: enrichment_calls.append("peers") or {
            "available": True,
            "source": "Smoke-test fundamentals",
            "peers": [],
            "info": automatic_peer_info,
            "failures": [],
        },
    )
    monkeypatch.setattr(
        wharton_dash,
        "_fetch_ai_dcf_assumptions_cached",
        lambda ticker, evidence: enrichment_calls.append("dcf") or {"available": False, "source": "smoke_test"},
    )
    at = AppTest.from_file(str(APP_PATH))
    at.session_state["quant_sim_workspace_route"] = "Wharton Cockpit"
    at.session_state["wharton_user_profile_v2"] = profile
    at.session_state["wharton_company_analysis_v1"] = {
        "MSFT": {
            "ticker": "MSFT",
            "fetched_at": "2026-07-14T12:00:00+00:00",
            "info": {
                "longName": "Microsoft Corporation",
                "longBusinessSummary": "Software and cloud services company.",
                "currency": "USD",
                "financialCurrency": "USD",
                "currentPrice": 100.0,
                "marketCap": 1_000_000_000_000,
                "freeCashflow": 10_000_000_000,
                "sector": "Technology",
                "industry": "Software - Infrastructure",
                "beta": 1.0,
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
            "geographic_revenue": {
                "available": True,
                "currency": "USD",
                "source_name": "Test annual report",
                "source_url": "https://example.com/annual-report",
                "report_date": "2025-06-30",
                "analysis": {
                    "available": True,
                    "score": 3,
                    "max_score": 5,
                    "label": "Moderately diversified",
                    "top_region": "United States",
                    "top_region_share": 0.55,
                    "effective_regions": 1.98,
                    "warning": "Test warning",
                    "interpretation": "Test interpretation",
                    "strengths": [],
                    "risks": [],
                    "rows": [
                        {"region": "United States", "revenue": 55.0, "share": 0.55, "strategic_importance": "Core"},
                        {"region": "Europe", "revenue": 45.0, "share": 0.45, "strategic_importance": "Material"},
                    ],
                },
            },
        }
    }
    at.session_state["company_macro_snapshot_v5_MSFT_USA"] = {
        "available": True,
        "economy_code": "USA",
        "economy_name": "United States",
        "reference_year": 2024,
        "fetched_at": "2026-07-15T00:00:00+00:00",
        "source_url": "https://api.worldbank.org/v2/country/USA/indicator/test",
        "indicators": {
            "FP.CPI.TOTL.ZG": {
                "label": "Inflation",
                "latest_value": 2.5,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 2.5}],
            },
            "GC.DOD.TOTL.GD.ZS": {
                "label": "Central government debt",
                "latest_value": 55.0,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 55.0}],
            },
            "NY.GDP.MKTP.KD.ZG": {
                "label": "Real GDP growth",
                "latest_value": 3.2,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 3.2}],
            },
            "SL.UEM.TOTL.ZS": {
                "label": "Unemployment",
                "latest_value": 4.0,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 4.0}],
            },
            "FR.INR.RINR": {
                "label": "Real interest rate",
                "latest_value": 2.0,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 2.0}],
            },
            "BN.CAB.XOKA.GD.ZS": {
                "label": "Current account balance",
                "latest_value": -2.0,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": -2.0}],
            },
            "FR.INR.LEND": {
                "label": "Lending interest rate",
                "latest_value": 6.0,
                "latest_year": 2024,
                "series": [{"year": 2024, "value": 6.0}],
            },
        },
    }
    at.run(timeout=60)

    assert len(at.exception) == 0
    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    assert area_selector.options == [
        "Home", "Client & Policy", "Research", "Decisions", "Portfolio", "Deliverables"
    ]
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Overview & Tasks", "Competition Readiness"]

    panel_selector.set_value("Competition Readiness").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Competition Readiness" in item.value for item in at.markdown)
    readiness_tabs = [tab.label for tab in at.tabs]
    assert "Red Team & AI Audit" in readiness_tabs
    assert "Report & Pitch" in readiness_tabs

    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    area_selector.set_value("Client & Policy").run(timeout=60)
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Mandate & Strategy"]
    assert any("Mandate & Strategy" in item.value for item in at.markdown)
    strategy_tab_labels = [tab.label for tab in at.tabs]
    assert strategy_tab_labels == [
        "Client Mandate", "Behavioral Profile", "Strategy Rulebook", "Alignment & Drift"
    ]
    assert "Thesis Monitor" not in strategy_tab_labels
    assert "Decision Journal" not in strategy_tab_labels

    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    area_selector.set_value("Research").run(timeout=60)
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Research Workspace", "Security Dossiers"]
    research_view = next(item for item in at.radio if item.label == "Research view")
    assert research_view.options == ["Stock Screener", "Company Analysis", "Fixed Income", "Real Assets"]
    research_view.set_value("Company Analysis").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Company Analysis" in item.value for item in at.markdown)

    nested_tab_labels = [tab.label for tab in at.tabs]
    assert nested_tab_labels == ["Overview", "Financials", "Valuation", "Business & risks", "Evidence"]
    assert enrichment_calls == []
    assert not any(item.label == "Initial FCFF Growth (%)" for item in at.number_input)
    assert not any(item.label == "Comparable companies" for item in at.multiselect)

    at.session_state["company_section_tab_MSFT"] = "Business & risks"
    at.session_state["company_detail_view_MSFT_3"] = "Industry & Peers"
    at.run(timeout=60)
    assert len(at.exception) == 0
    assert enrichment_calls == ["peers"]
    assert any(item.label == "Comparable companies" for item in at.multiselect)
    assert any("Automatically selected competitors" in item.value for item in at.markdown)
    peers = next(item for item in at.multiselect if item.label == "Comparable companies")
    peers.set_value(["ORCL"]).run(timeout=60)

    at.session_state["company_section_tab_MSFT"] = "Valuation"
    at.run(timeout=60)
    assert len(at.exception) == 0
    assert enrichment_calls == ["peers", "peers", "dcf"]
    assert any(item.label == "Normalized FCFF (billions)" for item in at.number_input)
    assert any(item.label == "Initial FCFF Growth (%)" for item in at.number_input)
    assert any(item.label == "Competitive Fade (years)" for item in at.number_input)
    assert any("What Must Be True?" in item.value for item in at.markdown)

    growth_input = next(item for item in at.number_input if item.label == "Initial FCFF Growth (%)")
    growth_input.set_value(17.0).run(timeout=60)
    growth_input = next(item for item in at.number_input if item.label == "Initial FCFF Growth (%)")
    assert growth_input.value == 17.0

    calls_before_overview = list(enrichment_calls)
    at.session_state["company_section_tab_MSFT"] = "Overview"
    at.run(timeout=60)
    assert len(at.exception) == 0
    assert enrichment_calls == calls_before_overview
    assert at.session_state["dcf_growth_MSFT"] == 17.0
    at.session_state["company_section_tab_MSFT"] = "Valuation"
    at.run(timeout=60)
    growth_input = next(item for item in at.number_input if item.label == "Initial FCFF Growth (%)")
    assert growth_input.value == 17.0

    at.session_state["company_section_tab_MSFT"] = "Financials"
    at.session_state["company_detail_view_MSFT_1"] = "Revenue by Region"
    at.run(timeout=60)
    region_view = next(item for item in at.radio if item.label == "Regional analysis view")
    assert region_view.options == ["Revenue Exposure", "Macro Drill-down"]
    region_view.set_value("Macro Drill-down").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Regional Macro Drill-down" in item.value for item in at.markdown)
    assert any(item.label == "Macro resilience (2024)" for item in at.metric)

    for label in (
        "Evidence & Sources", "Financial Statements", "Management",
        "Moat, Track Record & Risks", "All Metrics", "Industry & Peers",
    ):
        group, subindex = {
            "Evidence & Sources": ("Evidence", None),
            "Financial Statements": ("Financials", 1),
            "Management": ("Business & risks", 3),
            "Moat, Track Record & Risks": ("Business & risks", 3),
            "All Metrics": ("Financials", 1),
            "Industry & Peers": ("Business & risks", 3),
        }[label]
        at.session_state["company_section_tab_MSFT"] = group
        if subindex is not None:
            at.session_state[f"company_detail_view_MSFT_{subindex}"] = label
        at.run(timeout=60)
        assert len(at.exception) == 0, label
        if label == "Financial Statements":
            at.session_state["company_financials_tab_MSFT"] = "Quarterly"
            at.run(timeout=60)
            assert len(at.exception) == 0
            assert any("Quarterly Income Statement" in item.value for item in at.markdown)
    peers = next(item for item in at.multiselect if item.label == "Comparable companies")
    assert peers.value == ["ORCL"]
    assert at.session_state["dcf_growth_MSFT"] == 17.0

    research_view = next(item for item in at.radio if item.label == "Research view")
    research_view.set_value("Fixed Income").run(timeout=60)
    assert len(at.exception) == 0
    assert any(item.label == "Canonical Security Dossier" for item in at.selectbox)
    assert any(item.label == "Instrument type" for item in at.radio)

    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    panel_selector.set_value("Security Dossiers").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Security Dossiers" in item.value for item in at.markdown)

    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    area_selector.set_value("Decisions").run(timeout=60)
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Investment Committee"]
    assert len(at.exception) == 0
    assert any("Investment Committee" in item.value for item in at.markdown)

    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    area_selector.set_value("Portfolio").run(timeout=60)
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Portfolio Overview", "WInS & Reconciliation", "Risk & Scenarios"]
    assert any("Portfolio Overview" in item.value for item in at.markdown)
    assert not any("Company Analysis" in item.value for item in at.markdown)
    assert not any(item.label == "WInS positions snapshot" for item in at.file_uploader)

    panel_selector.set_value("WInS & Reconciliation").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Live Portfolio & Data Reliability" in item.value for item in at.markdown)
    assert [item.label for item in at.file_uploader].count(
        "WInS positions CSV or Excel"
    ) == 1
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    panel_selector.set_value("Risk & Scenarios").run(timeout=60)
    analytics_view = next(item for item in at.radio if item.label == "Analytics view")
    analytics_view.set_value("FX & Hedging").run(timeout=60)
    assert len(at.exception) == 0
    assert any("Currency Risk & Hedging" in item.value for item in at.markdown)
    assert any(item.label == "Reporting currency" for item in at.selectbox)

    area_selector = next(item for item in at.sidebar.radio if item.label == "Workspace area")
    area_selector.set_value("Deliverables").run(timeout=60)
    panel_selector = next(item for item in at.selectbox if item.label == "Active panel")
    assert panel_selector.options == ["Report & Pitch", "Rules & Compliance"]


@pytest.mark.parametrize(
    ("benchmark", "expected_market_data_calls"),
    [
        ("AAPL", [("AAPL", "MSFT", "VTI", "GLD", "BND")]),
        (
            "SPY",
            [("AAPL", "MSFT", "VTI", "GLD", "BND"), ("SPY",)],
        ),
    ],
)
def test_streamlit_app_evaluate_flow_renders_both_export_sections(
    monkeypatch,
    tmp_path,
    benchmark,
    expected_market_data_calls,
):
    _enable_test_auto_login(monkeypatch)
    st.cache_data.clear()
    import src.ai
    import src.analytics
    import src.data.fetchers.yahoo_fetcher
    import src.optimization
    import src.simulation

    market_data_calls: list[tuple[str, ...]] = []

    def fake_fetch_close_prices(self, symbols, start_date, end_date):
        requested = tuple(symbols)
        market_data_calls.append(requested)
        return _sample_prices(list(requested))

    monkeypatch.setattr(
        src.data.fetchers.yahoo_fetcher.YahooFetcher,
        "fetch_close_prices",
        fake_fetch_close_prices,
    )
    monkeypatch.setattr(
        src.ai,
        "generate_ai_review",
        lambda payload, api_key=None: {
            "available": False,
            "error": "Smoke test fallback",
        },
    )
    monkeypatch.setattr(
        src.analytics,
        "run_advanced_models_with_bundle",
        lambda returns, forecast_periods, returns_df, model_context=None: ({}, {}),
    )
    monkeypatch.setattr(src.analytics, "run_quant_stack", _fake_quant_stack(tmp_path))
    monkeypatch.setattr(src.analytics, "list_run_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        src.optimization,
        "optimize_minimum_variance",
        lambda returns, **kwargs: {
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
        lambda returns, risk_free_rate=0.0, **kwargs: {
            "success": True,
            "symbols": list(returns.columns),
            "weights": [1.0 / len(returns.columns)] * len(returns.columns),
            "expected_return": 0.11,
            "volatility": 0.13,
            "sharpe_ratio": 0.62,
        },
    )
    monkeypatch.setattr(
        src.optimization,
        "calculate_efficient_frontier",
        lambda returns, n_points=30, **kwargs: [],
    )
    monkeypatch.setattr(
        src.optimization,
        "sample_portfolio_cloud",
        lambda returns, n_samples, risk_free_rate=0.0, **kwargs: [],
    )
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
    at.session_state["quant_sim_workspace_route"] = "Quant Platform"
    at.session_state["dashboard_layout_preset"] = "Focused"
    at.session_state["dashboard_layout_preset_auto"] = False
    at.session_state["dashboard_layout_preset_applied"] = "Focused"
    at.session_state["dashboard_visible_pages_selector"] = ["overview", "reports"]
    at.run(timeout=60)

    next(item for item in at.text_input if item.label == "Benchmark").set_value(benchmark)
    next(button for button in at.button if button.label == "Evaluate Portfolio").click()
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert market_data_calls == expected_market_data_calls
    markdown_values = [item.value for item in at.markdown]
    assert "### Quick Exports" in markdown_values
    warning_values = [item.value for item in at.warning]
    assert warning_values.count("Signals are weak: composite score is close to neutral.") == 1

    page_selector = next(item for item in at.radio if item.label == "Analysis workspace")
    page_selector.set_value("reports").run(timeout=60)
    assert len(at.exception) == 0
    markdown_values = [item.value for item in at.markdown]
    assert "### Quick Exports" in markdown_values
    assert "### Export Center" in markdown_values
