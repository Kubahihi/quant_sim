from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.company_analysis import (
    _extract_geographic_revenue_from_ixbrl,
    _imf_world_gdp_weighted_series,
    analyze_geographic_revenue,
    analyze_macro_snapshot,
    analyze_moat,
    analyze_track_record,
    build_dcf_scenarios,
    calculate_dcf,
    classify_biography_sections,
    fetch_imf_current_account_balance,
    fetch_imf_general_government_debt,
    fetch_macro_snapshot,
    fetch_management_biography,
    format_statement,
    infer_macro_economy,
)


SAMPLE_IXBRL = b'''<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:xbrli="http://www.xbrl.org/2003/instance"
      xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
  <body>
    <ix:resources>
      <xbrli:unit id="USD"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
      <xbrli:context id="FY">
        <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier></xbrli:entity>
        <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
      </xbrli:context>
      <xbrli:context id="US">
        <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier><xbrli:segment>
          <xbrldi:explicitMember dimension="srt:StatementGeographicalAxis">example:UnitedStatesMember</xbrldi:explicitMember>
        </xbrli:segment></xbrli:entity>
        <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
      </xbrli:context>
      <xbrli:context id="EU">
        <xbrli:entity><xbrli:identifier scheme="test">1</xbrli:identifier><xbrli:segment>
          <xbrldi:explicitMember dimension="srt:StatementGeographicalAxis">example:EuropeMember</xbrldi:explicitMember>
        </xbrli:segment></xbrli:entity>
        <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
      </xbrli:context>
    </ix:resources>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="FY" unitRef="USD" scale="6">1,000</ix:nonFraction>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="US" unitRef="USD" scale="6">600</ix:nonFraction>
    <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" contextRef="EU" unitRef="USD" scale="6">300</ix:nonFraction>
  </body>
</html>'''


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


def test_ixbrl_geographic_revenue_extraction_adds_honest_residual_bucket():
    result = _extract_geographic_revenue_from_ixbrl(SAMPLE_IXBRL, report_date="2024-12-31")

    assert result["available"] is True
    assert result["currency"] == "USD"
    assert result["coverage_ratio"] == pytest.approx(0.9)
    assert result["rows"] == [
        {"region": "United States", "revenue": 600_000_000.0},
        {"region": "Europe", "revenue": 300_000_000.0},
        {"region": "Not separately disclosed", "revenue": 100_000_000.0},
    ]


def test_geographic_analysis_rates_concentration_and_region_importance():
    result = analyze_geographic_revenue(
        [
            {"region": "North America", "revenue": 60},
            {"region": "Europe", "revenue": 25},
            {"region": "Asia", "revenue": 15},
        ]
    )

    assert result["available"] is True
    assert result["score"] == 3
    assert result["top_region"] == "North America"
    assert result["top_region_share"] == pytest.approx(0.60)
    assert result["rows"][0]["strategic_importance"] == "Core"
    assert any("North America" in risk for risk in result["risks"])


def test_geographic_analysis_requires_two_positive_regions():
    result = analyze_geographic_revenue([{"region": "Europe", "revenue": 100}])

    assert result["available"] is False


def test_geographic_analysis_marks_broad_international_bucket_as_limited_detail():
    result = analyze_geographic_revenue(
        [
            {"region": "United States", "revenue": 51},
            {"region": "Non-US / International", "revenue": 49},
        ]
    )

    assert result["score"] == 3
    assert result["limited_granularity"] is True
    assert "limited detail" in result["label"].lower()


def test_infer_macro_economy_maps_filing_labels_to_editable_proxies():
    assert infer_macro_economy("United States") == "USA"
    assert infer_macro_economy("US") == "USA"
    assert infer_macro_economy("Europe, Middle East and Africa") == "WLD"
    assert infer_macro_economy("Asia Pacific") == "EAS"
    assert infer_macro_economy("International") == "WLD"


def test_fetch_macro_snapshot_parses_latest_world_bank_observations(monkeypatch):
    observations = []
    sample_values = {
        "FP.CPI.TOTL.ZG": {"2023": 3.4, "2024": 2.5},
        "GC.DOD.TOTL.GD.ZS": {"2024": 58.0},
        "NY.GDP.MKTP.KD.ZG": {"2024": 3.2},
        "SL.UEM.TOTL.ZS": {"2024": 4.0},
        "FR.INR.RINR": {"2024": 2.0},
        "BN.CAB.XOKA.GD.ZS": {"2024": -2.0},
        "FR.INR.LEND": {"2024": 6.0},
    }
    for indicator_code, values in sample_values.items():
        for year, value in values.items():
            observations.append({
                "indicator": {"id": indicator_code, "value": indicator_code},
                "country": {"id": "US", "value": "United States"},
                "countryiso3code": "USA",
                "date": year,
                "value": value,
            })

    monkeypatch.setattr(
        "src.analytics.company_analysis._world_bank_request",
        lambda url: [{"page": 1, "pages": 1}, observations],
    )
    result = fetch_macro_snapshot("USA", start_year=2023, end_year=2024)

    assert result["available"] is True
    assert result["economy_name"] == "United States"
    assert result["available_indicator_count"] == 7
    assert result["indicators"]["FP.CPI.TOTL.ZG"]["latest_value"] == pytest.approx(2.5)
    assert result["indicators"]["FP.CPI.TOTL.ZG"]["latest_year"] == 2024
    assert [point["year"] for point in result["indicators"]["FP.CPI.TOTL.ZG"]["series"]] == [2023, 2024]
    assert "date=2023:2024" in result["source_url"]


def test_macro_snapshot_uses_only_fixed_2024_observations(monkeypatch):
    observations = [
        {
            "indicator": {"id": "FP.CPI.TOTL.ZG", "value": "Inflation"},
            "country": {"id": "JP", "value": "Japan"},
            "date": year,
            "value": value,
        }
        for year, value in (("2023", 3.2), ("2025", 2.1))
    ]
    monkeypatch.setattr(
        "src.analytics.company_analysis._world_bank_request",
        lambda url: [{"page": 1, "pages": 1}, observations],
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis.fetch_imf_general_government_debt",
        lambda code, start, end: {"available": False},
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis.fetch_imf_current_account_balance",
        lambda code, start, end: {"available": False},
    )

    result = fetch_macro_snapshot("JPN", start_year=2023, end_year=2025)
    inflation = result["indicators"]["FP.CPI.TOTL.ZG"]

    assert result["reference_year"] == 2024
    assert inflation["latest_value"] is None
    assert inflation["latest_year"] is None
    assert [point["year"] for point in inflation["series"]] == [2023, 2025]
    assert "earlier or later years are not substituted" in result["warning"]


def test_imf_debt_fallback_uses_latest_historical_year_not_forecast(monkeypatch):
    monkeypatch.setattr(
        "src.analytics.company_analysis._latest_dbnomics_imf_weo_dataset",
        lambda: "WEO:2025-04",
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis._dbnomics_request",
        lambda url: {
            "series": {
                "docs": [{
                    "period": ["2022", "2023", "2024", "2025", "2026"],
                    "value": [66.1, 63.5, 62.2, 64.0, 65.0],
                }]
            }
        },
    )

    result = fetch_imf_general_government_debt("DEU", 2020, 2026)

    assert result["available"] is True
    assert result["latest_year"] == 2024
    assert result["latest_value"] == pytest.approx(62.2)
    assert result["definition"] == "General government gross debt (% of GDP)"
    assert result["is_fallback"] is True


def test_imf_current_account_fallback_uses_percent_of_gdp_series(monkeypatch):
    requested_urls = []
    monkeypatch.setattr(
        "src.analytics.company_analysis._latest_dbnomics_imf_weo_dataset",
        lambda: "WEO:2025-04",
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis._dbnomics_request",
        lambda url: requested_urls.append(url) or {
            "series": {"docs": [{"period": ["2023", "2024", "2025"], "value": [-0.7, -0.4, -0.2]}]}
        },
    )

    result = fetch_imf_current_account_balance("WLD", 2020, 2026)

    assert result["available"] is True
    assert result["latest_year"] == 2024
    assert result["latest_value"] == pytest.approx(-0.4)
    assert "WEOAGG:2025-04" in requested_urls[0]
    assert "001.BCA_NGDPD.pcent_gdp" in requested_urls[0]


def test_world_imf_fallback_calculates_gdp_weighted_country_ratio(monkeypatch):
    ratio_docs = [
        {"series_code": "AAA.GGXWDG_NGDP", "period": ["2024"], "value": [50.0]},
        {"series_code": "BBB.GGXWDG_NGDP", "period": ["2024"], "value": [100.0]},
    ]
    gdp_docs = [
        {"series_code": "AAA.NGDPD", "period": ["2024"], "value": [100.0]},
        {"series_code": "BBB.NGDPD", "period": ["2024"], "value": [300.0]},
    ]
    monkeypatch.setattr(
        "src.analytics.company_analysis._dbnomics_imf_country_series",
        lambda dataset, subject: gdp_docs if subject == "NGDPD" else ratio_docs,
    )

    result = _imf_world_gdp_weighted_series("WEO:2025-04", "GGXWDG_NGDP", 2024, 2024)

    assert result == [{"year": 2024, "value": pytest.approx(87.5)}]


def test_macro_snapshot_fills_missing_world_bank_debt_from_imf(monkeypatch):
    observations = [{
        "indicator": {"id": "FP.CPI.TOTL.ZG", "value": "Inflation"},
        "country": {"id": "DE", "value": "Germany"},
        "date": "2024",
        "value": 2.3,
    }]
    monkeypatch.setattr(
        "src.analytics.company_analysis._world_bank_request",
        lambda url: [{"page": 1, "pages": 1}, observations],
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis.fetch_imf_general_government_debt",
        lambda code, start, end: {
            "available": True,
            "latest_value": 62.2,
            "latest_year": 2024,
            "series": [{"year": 2024, "value": 62.2}],
            "source_name": "IMF World Economic Outlook 2025-04 via DBnomics",
            "source_url": "https://db.nomics.world/example",
            "definition": "General government gross debt (% of GDP)",
            "is_fallback": True,
        },
    )
    monkeypatch.setattr(
        "src.analytics.company_analysis.fetch_imf_current_account_balance",
        lambda code, start, end: {"available": False, "error": "not available in fixture"},
    )

    result = fetch_macro_snapshot("DEU", start_year=2020, end_year=2026)
    debt = result["indicators"]["GC.DOD.TOTL.GD.ZS"]

    assert debt["latest_value"] == pytest.approx(62.2)
    assert debt["label"] == "General government gross debt"
    assert debt["is_fallback"] is True
    assert result["uses_imf_debt_fallback"] is True
    assert "IMF WEO" in result["warning"]


def test_macro_score_is_weighted_and_reports_data_coverage():
    snapshot = {
        "indicators": {
            "FP.CPI.TOTL.ZG": {"latest_value": 2.5, "latest_year": 2024},
            "GC.DOD.TOTL.GD.ZS": {"latest_value": 55.0, "latest_year": 2023},
            "NY.GDP.MKTP.KD.ZG": {"latest_value": 3.2, "latest_year": 2024},
            "SL.UEM.TOTL.ZS": {"latest_value": 4.0, "latest_year": 2024},
            "FR.INR.RINR": {"latest_value": 2.0, "latest_year": 2023},
            "BN.CAB.XOKA.GD.ZS": {"latest_value": -2.0, "latest_year": 2024},
            "FR.INR.LEND": {"latest_value": 6.0, "latest_year": 2024},
        }
    }

    result = analyze_macro_snapshot(snapshot)

    assert result["available"] is True
    assert result["score"] == pytest.approx(98.75)
    assert result["label"] == "Resilient"
    assert result["data_coverage"] == pytest.approx(1.0)
    lending = next(item for item in result["components"] if item["indicator_code"] == "FR.INR.LEND")
    assert lending["component_score"] is None
    assert lending["weight"] == 0


def test_macro_score_flags_low_coverage_without_treating_missing_data_as_zero():
    result = analyze_macro_snapshot({
        "indicators": {"FP.CPI.TOTL.ZG": {"latest_value": 2.0, "latest_year": 2024}}
    })

    assert result["score"] == pytest.approx(100.0)
    assert result["data_coverage"] == pytest.approx(0.25)
    assert any("low-confidence" in risk for risk in result["risks"])
