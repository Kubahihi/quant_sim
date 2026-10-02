from io import BytesIO
from pathlib import Path

import pytest
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = "from ui.custom_history import render_custom_history\nrender_custom_history()"
WHARTON_APP = """
from ui.pages.wharton_dash import _render_risk_scenarios_workspace
_render_risk_scenarios_workspace({'username': 'Jakub'}, {})
"""


def test_empty_import_workspace():
    app = AppTest.from_string(APP).run()
    assert not app.exception
    assert app.title[0].value == "Custom data analysis"
    assert len(app.info) == 1


def test_quant_platform_custom_source_route(monkeypatch):
    monkeypatch.setenv("QUANT_SIM_ENV", "test")
    monkeypatch.setenv("QUANT_SIM_TEST_AUTO_LOGIN", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"))
    app.session_state["quant_sim_workspace_route"] = "Quant Platform"
    app.session_state["quant_history_source"] = "Custom data"
    app.run(timeout=60)
    assert not app.exception
    assert any(x.value == "Custom data analysis" for x in app.title)
    app.radio(key="quant_history_source").set_value("Market data").run(timeout=60)
    assert not app.exception
    assert any(x.label == "Evaluate Portfolio" for x in app.button)


@pytest.mark.parametrize("entrypoint", [APP, WHARTON_APP], ids=["quant", "wharton"])
def test_import_analyze_and_invalidate_changed_settings(monkeypatch, entrypoint):
    data = BytesIO(b"Date,A,B\n2024-01-31,0.02,0.01\n2024-02-29,-0.01,0.03\n2024-03-31,0.03,-0.02\n2024-04-30,0.01,0.02\n")
    data.name = "history.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: data)
    app = AppTest.from_string(entrypoint)
    app.session_state["wharton_risk_scenarios_view"] = "Custom data"
    competition_result = {"generated_at": "unchanged", "tickers": ["COMPETITION"]}
    app.session_state["wharton_quant_result"] = competition_result
    app.run()
    next(x for x in app.selectbox if x.label == "Observation frequency").set_value("Monthly")
    next(x for x in app.text_input if x.label == "Data currency").set_value("USD")
    app.run()
    assert not app.exception
    assert not app.error
    next(x for x in app.button if x.label == "Analyze imported data").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 5
    assert app.session_state["wharton_quant_result"] == competition_result
    next(x for x in app.number_input if x.label == "Annual risk-free rate (%)").set_value(4.0).run()
    assert not app.exception
    assert not app.metric
    assert any("Inputs have changed" in x.value for x in app.info)


@pytest.mark.parametrize("entrypoint", [APP, WHARTON_APP], ids=["quant", "wharton"])
@pytest.mark.parametrize("quote_currency", ["USD", "EUR"])
def test_mixed_history_download_analyze_and_invalidate(monkeypatch, entrypoint, quote_currency):
    from ui import custom_history
    from src.data.mixed_history import MarketHistory

    data = BytesIO(b"Date,A,B\n2023-12-29,,\n2024-01-31,0.02,0.01\n2024-02-29,-0.01,0.03\n2024-03-31,0.03,-0.02\n2024-04-30,0.01,0.02\n")
    data.name = "history.csv"
    calls = []

    def download(tickers, start, end, *, base_currency):
        assert base_currency == "USD"
        calls.append((tickers, start, end))
        prices = pd.DataFrame({"SPY": [100, 105, 102, 108, 110]}, index=pd.to_datetime(["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]))
        fx = pd.DataFrame({"EURUSD=X": [1.0, 1.1, 1.05, 1.08, 1.1]}, index=prices.index) if quote_currency == "EUR" else pd.DataFrame()
        return MarketHistory(prices, {"SPY": quote_currency}, "2026-09-26T00:00:00+00:00", fx)

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: data)
    monkeypatch.setattr(custom_history, "download_market_history", download)
    app = AppTest.from_string(entrypoint)
    app.session_state["wharton_risk_scenarios_view"] = "Custom data"
    app.run()
    next(x for x in app.selectbox if x.label == "Observation frequency").set_value("Monthly")
    next(x for x in app.text_input if x.label == "Data currency").set_value("USD")
    app.checkbox(key="custom_add_market").check().run()
    app.text_area(key="custom_market_tickers").set_value("SPY").run()
    assert not app.exception
    assert not calls
    assert not app.metric
    app.button(key="custom_download_market").click().run()
    assert not app.exception
    assert not app.error
    assert len(calls) == 1
    assert any("Common history:" in item.value for item in app.info)
    next(x for x in app.button if x.label == "Analyze imported data").click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 5
    saved = app.session_state["custom_history_result"][1]
    assert list(saved["assets"].index) == ["A", "B", "SPY"]
    assert len(saved["portfolio_returns"]) == 4
    market_return = 1.05 * 1.1 - 1 if quote_currency == "EUR" else 0.05
    assert saved["portfolio_returns"].iloc[0] == pytest.approx((0.02 + 0.01 + market_return) / 3)
    assert len(calls) == 1  # Analysis and widget reruns reuse the explicit download.
    next(x for x in app.text_input if x.label == "Data currency").set_value("EUR").run()
    assert not app.exception
    assert not app.metric
    assert any("Download market data to check coverage" in item.value for item in app.info)
    next(x for x in app.text_input if x.label == "Data currency").set_value("USD").run()
    assert len(calls) == 1
    app.text_area(key="custom_market_tickers").set_value("AGG").run()
    assert not app.exception
    assert not app.metric
    assert len(calls) == 1
    assert any("Download market data to check coverage" in item.value for item in app.info)
