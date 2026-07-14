from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.company_analysis import (
    analyze_moat,
    analyze_track_record,
    build_dcf_scenarios,
    calculate_dcf,
    classify_biography_sections,
    fetch_management_biography,
    format_statement,
)


def _sample_info() -> dict:
    return {
        "freeCashflow": 10_000_000_000,
        "revenueGrowth": 0.10,
        "earningsGrowth": 0.12,
        "grossMargins": 0.60,
        "operatingMargins": 0.25,
        "returnOnEquity": 0.30,
        "marketCap": 500_000_000_000,
        "totalCash": 20_000_000_000,
        "totalDebt": 5_000_000_000,
        "sharesOutstanding": 1_000_000_000,
        "currentPrice": 100.0,
    }


def test_dcf_calculates_equity_value_and_per_share_value():
    result = calculate_dcf(
        free_cash_flow=10_000_000_000,
        growth_rate=0.08,
        discount_rate=0.10,
        terminal_growth_rate=0.025,
        years=5,
        cash=20_000_000_000,
        debt=5_000_000_000,
        shares_outstanding=1_000_000_000,
        current_price=100.0,
    )

    assert result["available"] is True
    assert len(result["projected"]) == 5
    assert result["equity_value"] > result["enterprise_value"]
    assert result["fair_value_per_share"] > 0
    assert result["upside_pct"] == pytest.approx(result["fair_value_per_share"] / 100 - 1)


def test_dcf_rejects_invalid_cash_flow_and_terminal_assumptions():
    negative = calculate_dcf(-1, 0.05, 0.10, 0.02, 5, 0, 0, 10)
    invalid_spread = calculate_dcf(100, 0.05, 0.02, 0.03, 5, 0, 0, 10)

    assert negative["available"] is False
    assert "positive" in negative["error"]
    assert invalid_spread["available"] is False
    assert "higher" in invalid_spread["error"]


def test_scenarios_order_bear_base_bull_fair_values():
    scenarios = build_dcf_scenarios(_sample_info())

    assert list(scenarios) == ["Bear", "Base", "Bull"]
    assert scenarios["Bear"]["fair_value_per_share"] < scenarios["Base"]["fair_value_per_share"]
    assert scenarios["Base"]["fair_value_per_share"] < scenarios["Bull"]["fair_value_per_share"]


def test_moat_and_track_record_are_evidence_based():
    info = _sample_info()
    history = pd.DataFrame({"Close": [100.0, 120.0, 90.0, 150.0]})

    moat = analyze_moat(info)
    track = analyze_track_record(info, history)

    assert moat["score"] == moat["max_score"]
    assert moat["label"] == "Wide moat signal"
    assert any("revenue growth" in item.lower() for item in track["successes"])
    assert any("share-price return" in item.lower() for item in track["successes"])


def test_statement_formatter_transposes_dates_and_scales_to_millions():
    statement = pd.DataFrame(
        {pd.Timestamp("2025-12-31"): [2_000_000, 500_000]},
        index=["Total Revenue", "Net Income"],
    )

    result = format_statement(statement)

    assert list(result.columns) == ["2025-12-31"]
    assert result.loc["Total Revenue", "2025-12-31"] == pytest.approx(2.0)


def test_biography_sections_find_education_and_career_topics():
    sections = [
        {"line": "Early life and education", "index": "1"},
        {"line": "Career", "index": "2"},
        {"line": "Awards and recognition", "index": "3"},
    ]

    result = classify_biography_sections(sections)

    assert result["education"] == ["1"]
    assert result["career"] == ["2"]


def test_management_biography_returns_sourced_education_and_career(monkeypatch):
    def fake_request(params: dict) -> dict:
        if params.get("list") == "search":
            return {
                "query": {
                    "search": [
                        {
                            "title": "Satya Nadella",
                            "snippet": "Indian-American business executive and CEO of Microsoft",
                        }
                    ]
                }
            }
        if params.get("prop") == "sections":
            return {
                "parse": {
                    "sections": [
                        {"line": "Early life and education", "index": "1"},
                        {"line": "Career", "index": "2"},
                    ]
                }
            }
        if params.get("prop") == "text" and params.get("section") == "1":
            return {"parse": {"text": {"*": "<p>Nadella earned degrees in engineering, computer science, and business.</p>"}}}
        if params.get("prop") == "text" and params.get("section") == "2":
            return {"parse": {"text": {"*": "<p>He joined Microsoft in 1992 and became CEO in 2014.</p>"}}}
        return {"query": {"pages": {"1": {"extract": "Satya Nadella is the CEO of Microsoft."}}}}

    monkeypatch.setattr("src.analytics.company_analysis._wikipedia_request", fake_request)

    result = fetch_management_biography("Satya Nadella", "Microsoft")

    assert result["available"] is True
    assert "engineering" in result["education"].lower()
    assert "joined microsoft" in result["career"].lower()
    assert result["source_url"] == "https://en.wikipedia.org/wiki/Satya_Nadella"
