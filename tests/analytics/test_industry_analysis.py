from __future__ import annotations

import pytest

from src.analytics.industry_analysis import (
    PeerMetricSpec,
    analyze_peer_comparison,
    build_industry_analysis,
    compare_company_to_peers,
    synthesize_porter_five_forces,
    synthesize_swot,
)


def _metric(result: dict, key: str) -> dict:
    return next(row for row in result["metrics"] if row["key"] == key)


def test_peer_comparison_covers_all_four_categories_and_respects_direction():
    company = {
        "trailingPE": 10.0,
        "revenueGrowth": 0.20,
        "operatingMargins": 0.25,
        "debtToEquity": 0.20,
    }
    peers = {
        "A": {"trailingPE": 12.0, "revenueGrowth": 0.05, "operatingMargins": 0.20, "debtToEquity": 0.40},
        "B": {"trailingPE": 15.0, "revenueGrowth": 0.10, "operatingMargins": 0.20, "debtToEquity": 0.50},
        "C": {"trailingPE": 18.0, "revenueGrowth": 0.15, "operatingMargins": 0.22, "debtToEquity": 0.60},
        "D": {"trailingPE": 1000.0, "revenueGrowth": 0.10, "operatingMargins": 0.18, "debtToEquity": 0.70},
    }

    result = analyze_peer_comparison(company, peers, company_name="Target")

    assert result["available"] is True
    assert result["peer_count"] == 4
    assert result["category_coverage_pct"] == 100.0
    assert result["score"] == 100.0
    assert result["rating"] == "Strong versus peers"
    assert _metric(result, "pe_ratio")["peer_median"] == pytest.approx(16.5)
    assert _metric(result, "pe_ratio")["relative_flag"] == "favorable"
    assert _metric(result, "debt_to_equity")["direction"] == "lower_is_better"
    assert _metric(result, "revenue_growth")["direction"] == "higher_is_better"


def test_peer_median_is_robust_to_an_extreme_peer_and_exposes_quartiles():
    result = analyze_peer_comparison(
        {"pe_ratio": 14},
        {"A": {"pe_ratio": 10}, "B": {"pe_ratio": 12}, "C": {"pe_ratio": 16}, "Outlier": {"pe_ratio": 5000}},
    )

    row = _metric(result, "pe_ratio")
    assert row["peer_median"] == pytest.approx(14.0)
    assert row["peer_q1"] == pytest.approx(11.5)
    assert row["peer_q3"] == pytest.approx(1262.0)
    assert row["desirability_percentile"] == pytest.approx(50.0)
    assert row["relative_flag"] == "in_line"


def test_equal_values_receive_a_neutral_midrank_percentile():
    result = compare_company_to_peers(
        {"forward_pe": 15},
        [{"ticker": "A", "forward_pe": 15}, {"ticker": "B", "forward_pe": 15}, {"ticker": "C", "forward_pe": 15}],
    )

    row = _metric(result, "forward_pe")
    assert row["raw_percentile"] == 50.0
    assert row["desirability_percentile"] == 50.0
    assert row["relative_flag"] == "in_line"
    assert result["score"] == 50.0


def test_missing_invalid_and_insufficient_values_are_reported_not_imputed():
    result = analyze_peer_comparison(
        {"pe_ratio": -4, "revenue_growth": 0.10},
        {
            "A": {"pe_ratio": 12, "revenue_growth": 0.08},
            "B": {"pe_ratio": float("inf"), "revenue_growth": "N/A"},
            "C": {"pe_ratio": 20, "revenue_growth": 0.12},
        },
    )

    assert result["available"] is False
    assert result["score"] is None
    assert result["confidence"] == "Insufficient"
    assert _metric(result, "pe_ratio")["missing_reason"] == "company_value_missing"
    assert _metric(result, "revenue_growth")["missing_reason"] == "insufficient_peer_values"
    assert all(flag["severity"] == "data_gap" for flag in result["flags"])


def test_percent_strings_and_nested_metric_payloads_are_supported():
    result = analyze_peer_comparison(
        {"metrics": {"revenueGrowth": "20%"}},
        [
            {"ticker": "A", "metrics": {"revenueGrowth": "5%"}},
            {"symbol": "B", "metrics": {"revenueGrowth": "10%"}},
            {"name": "C", "metrics": {"revenueGrowth": "15%"}},
        ],
    )

    row = _metric(result, "revenue_growth")
    assert row["company_value"] == pytest.approx(0.2)
    assert row["peer_median"] == pytest.approx(0.1)
    assert [peer["name"] for peer in row["peer_values"]] == ["A", "B", "C"]


def test_custom_metric_spec_produces_transparent_custom_category_score():
    custom = (
        PeerMetricSpec(
            key="retention",
            label="Customer retention",
            category="business_quality",
            higher_is_better=True,
            aliases=("retentionRate",),
        ),
    )
    result = analyze_peer_comparison(
        {"retentionRate": 0.95},
        {"A": {"retention": 0.80}, "B": {"retention": 0.85}, "C": {"retention": 0.90}},
        metric_specs=custom,
    )

    assert result["score"] == 100.0
    assert result["categories"]["business_quality"]["raw_score"] == 100.0
    assert result["methodology"]["minimum_peer_count"] == 3


def test_porter_synthesis_uses_only_supplied_ratings_and_evidence():
    result = synthesize_porter_five_forces(
        {
            "rivalry": {"rating": "high", "evidence": ["Five similarly sized competitors."]},
            "supplier_power": 2,
            "industry_name": "Example industry",
        }
    )

    assert result["available"] is True
    assert result["completed_forces"] == 2
    assert result["coverage_pct"] == 40.0
    assert result["average_pressure"] == 3.5
    assert result["evidence_coverage_pct"] == 50.0
    assert len(result["gaps"]) == 3
    assert result["unknown_input_keys"] == ["industry_name"]
    rivalry = next(row for row in result["forces"] if row["key"] == "competitive_rivalry")
    entrants = next(row for row in result["forces"] if row["key"] == "threat_of_new_entrants")
    assert rivalry["evidence"] == ["Five similarly sized competitors."]
    assert entrants["rating"] is None
    assert entrants["evidence"] == []


def test_invalid_porter_rating_is_left_unassessed():
    result = synthesize_porter_five_forces(
        {"competitive_rivalry": 8, "buyer_power": {"rating": "unknown", "notes": "Not researched"}}
    )

    assert result["available"] is False
    assert result["coverage_pct"] == 0.0
    assert result["industry_attractiveness_score"] is None
    assert all(row["assessed"] is False for row in result["forces"])


def test_swot_cleans_deduplicates_and_never_fills_missing_quadrants():
    result = synthesize_swot(
        {
            "strengths": "- Recurring revenue\n- Recurring revenue\n* Strong retention",
            "weakness": ["Customer concentration"],
            "opportunities": [],
            "extra_context": "Do not turn this into a SWOT claim",
        }
    )

    assert result["available"] is True
    assert result["coverage_pct"] == 50.0
    assert result["item_count"] == 3
    assert result["quadrants"]["strengths"]["items"] == ["Recurring revenue", "Strong retention"]
    assert result["quadrants"]["weaknesses"]["items"] == ["Customer concentration"]
    assert result["quadrants"]["opportunities"]["items"] == []
    assert result["quadrants"]["threats"]["items"] == []
    assert result["unknown_input_keys"] == ["extra_context"]


def test_combined_payload_is_ui_ready_and_preserves_section_availability():
    result = build_industry_analysis(
        {"operating_margin": 0.25},
        {"A": {"operating_margin": 0.10}, "B": {"operating_margin": 0.15}, "C": {"operating_margin": 0.20}},
        company_name="Target",
        porter_assessments={"buyer_power": "moderate"},
        swot_assessments={"threats": ["Regulatory change"]},
    )

    assert result["available"] is True
    assert result["peer_comparison"]["company_name"] == "Target"
    assert result["porter_five_forces"]["completed_forces"] == 1
    assert result["swot"]["item_count"] == 1
    assert "user-entered" in result["input_policy"]
