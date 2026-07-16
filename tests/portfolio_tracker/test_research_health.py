from datetime import date

from src.portfolio_tracker.research_health import assess_research_health


def test_research_health_reports_factual_coverage_and_review_queue():
    result = assess_research_health(
        ["AAA", "BBB"],
        theses=[
            {
                "ticker": "AAA",
                "next_review_at": "2026-07-20",
                "payload": {"investment_thesis": "Evidence-backed thesis"},
            }
        ],
        sources=[
            {
                "ticker": "AAA",
                "primary_source": True,
                "verified_by": "Jakub",
                "accessed_at": "2026-07-10",
            }
        ],
        catalysts=[
            {
                "ticker": "AAA",
                "status": "expected",
                "window_start": "2026-08-01",
                "window_end": "2026-08-05",
            },
            {
                "ticker": "BBB",
                "status": "expected",
                "window_start": "2026-06-01",
                "window_end": "2026-06-03",
            },
        ],
        price_observations={
            "AAA": {"observed_at": "2026-07-16", "source": "Yahoo Finance"},
            "BBB": {"observed_at": "2026-07-10", "source": "Manual"},
        },
        as_of=date(2026, 7, 16),
    )

    assert result["summary"]["ticker_count"] == 2
    assert result["summary"]["fresh_price_coverage_pct"] == 50.0
    assert result["summary"]["thesis_coverage_pct"] == 50.0
    assert result["summary"]["evidence_coverage_pct"] == 50.0
    assert result["summary"]["primary_source_coverage_pct"] == 50.0
    assert result["summary"]["overdue_catalyst_count"] == 1
    bbb = next(row for row in result["tickers"] if row["ticker"] == "BBB")
    assert bbb["price_status"] == "review_due"
    assert bbb["thesis_status"] == "missing"
    assert bbb["evidence_status"] == "missing"
    assert {item["area"] for item in result["review_queue"] if item["ticker"] == "BBB"} == {
        "price", "thesis", "evidence", "catalyst"
    }


def test_macro_is_year_labelled_and_no_composite_investment_score_is_returned():
    result = assess_research_health([], as_of=date(2026, 7, 16))

    assert "reference-year locked" in result["macro_policy"].lower()
    assert "score" not in result
    assert result["summary"]["ticker_count"] == 0


def test_future_dates_are_not_silently_treated_as_fresh():
    result = assess_research_health(
        ["AAA"],
        price_observations={"AAA": "2026-07-17"},
        as_of=date(2026, 7, 16),
    )

    assert result["tickers"][0]["price_status"] == "future_date"
    assert result["review_queue"][0]["area"] == "price"
