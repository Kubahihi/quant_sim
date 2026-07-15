from __future__ import annotations

from copy import deepcopy

import pytest

from src.portfolio_tracker.wins_reconciliation import (
    normalize_wins_rows,
    reconcile_wins_positions,
)


def test_normalize_wins_aliases_and_aggregate_duplicate_tickers():
    rows = [
        {
            "Symbol": " aapl ",
            "Shares": "10",
            "Cost Basis": "$1,000.00",
            "Price": "$110.00",
            "Security Type": "Stock",
        },
        {
            "Ticker Symbol": "AAPL",
            "Qty": 5,
            "Total Cost": "$600",
            "Current Value": "$650",
            "Asset Type": "Stock",
        },
    ]

    assert normalize_wins_rows(rows) == [
        {
            "ticker": "AAPL",
            "security_type": "Stock",
            "quantity": 15.0,
            "unit_cost": pytest.approx(1_600 / 15),
            "total_cost": 1_600.0,
            "current_price": pytest.approx(1_750 / 15),
            "current_value": 1_750.0,
            "source_rows": 2,
        }
    ]


def test_mapping_input_is_supported_without_mutation():
    rows = {
        "msft": {
            "units": "(2)",
            "average cost": "$300",
            "market value": "($650)",
            "instrument type": "Equity",
        }
    }
    original = deepcopy(rows)

    result = normalize_wins_rows(rows)

    assert rows == original
    assert result[0]["ticker"] == "MSFT"
    assert result[0]["quantity"] == -2
    assert result[0]["total_cost"] == -600
    assert result[0]["current_value"] == -650
    assert result[0]["current_price"] == 325


def test_incomplete_duplicate_data_does_not_report_partial_total_as_exact():
    result = normalize_wins_rows(
        [
            {"ticker": "AAA", "quantity": 2, "cost": 20, "current value": 24},
            {"ticker": "AAA", "quantity": 3},
        ]
    )[0]

    assert result["quantity"] == 5
    assert result["total_cost"] is None
    assert result["unit_cost"] is None
    assert result["current_value"] is None
    assert result["current_price"] is None


def test_reconcile_exact_snapshot_aggregates_tracker_and_ignores_closed_positions():
    wins = [
        {
            "symbol": "AAA",
            "shares": 10,
            "cost": 1_000,
            "current value": 1_200,
            "security type": "Stock",
        },
        {
            "symbol": "BBB",
            "shares": 5,
            "cost": 250,
            "current value": 275,
            "security type": "ETF",
        },
    ]
    tracked = [
        {
            "ticker": "AAA",
            "quantity": 4,
            "entry_price": 100,
            "last_price": 120,
            "security_type": "stock",
            "status": "open",
        },
        {
            "ticker": "AAA",
            "quantity": 6,
            "entry_price": 100,
            "last_price": 120,
            "security_type": "Stock",
        },
        {
            "ticker": "BBB",
            "quantity": 5,
            "entry_price": 50,
            "last_price": 55,
            "security_type": "ETF",
        },
        {
            "ticker": "OLD",
            "quantity": 99,
            "entry_price": 10,
            "last_price": 10,
            "security_type": "Stock",
            "status": "closed",
        },
    ]

    result = reconcile_wins_positions(wins, tracked)

    assert result["status"] == "reconciled"
    assert result["is_reconciled"] is True
    assert result["coverage_pct"] == 100
    assert result["summary"] == {
        "wins_positions": 2,
        "tracked_open_positions": 2,
        "matched_positions": 2,
        "exact_matches": 2,
        "mismatched_positions": 0,
        "partial_positions": 0,
        "missing_positions": 0,
        "extra_positions": 0,
        "coverage_pct": 100.0,
        "two_way_coverage_pct": 100.0,
    }
    assert result["totals"]["difference"] == {
        "quantity": 0,
        "total_cost": 0,
        "current_value": 0,
    }
    assert all(row["status"] == "matched" for row in result["matched"])


def test_reconcile_returns_value_differences_missing_extra_and_both_coverages():
    wins = [
        {
            "symbol": "AAA",
            "shares": 11,
            "cost": 1_000,
            "current value": 1_210,
            "type": "Stock",
        },
        {
            "symbol": "EXTRA",
            "shares": 2,
            "cost": 40,
            "current value": 50,
            "type": "Stock",
        },
    ]
    tracked = [
        {
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 100,
            "last_price": 120,
            "security_type": "Stock",
        },
        {
            "ticker": "MISSING",
            "quantity": 3,
            "entry_price": 20,
            "last_price": 25,
            "security_type": "Stock",
        },
    ]

    result = reconcile_wins_positions(wins, tracked)

    assert result["status"] == "differences"
    assert result["is_reconciled"] is False
    assert result["coverage_pct"] == 50
    assert result["two_way_coverage_pct"] == pytest.approx(100 / 3)
    assert result["summary"]["mismatched_positions"] == 1
    assert result["matched"][0]["quantity_difference"] == 1
    assert result["matched"][0]["cost_difference"] == 0
    assert result["matched"][0]["value_difference"] == 10
    assert result["missing"][0]["ticker"] == "MISSING"
    assert result["missing"][0]["quantity_difference"] == -3
    assert result["missing"][0]["cost_difference"] == -60
    assert result["missing"][0]["value_difference"] == -75
    assert result["extra"][0]["ticker"] == "EXTRA"
    assert result["extra"][0]["quantity_difference"] == 2
    assert result["extra"][0]["cost_difference"] == 40
    assert result["extra"][0]["value_difference"] == 50
    assert result["totals"]["difference"] == {
        "quantity": 0,
        "total_cost": -20,
        "current_value": -15,
    }


def test_missing_comparable_value_produces_partial_not_false_match():
    wins = [
        {"ticker": "AAA", "quantity": 10, "cost": 1_000, "security type": "Stock"}
    ]
    tracked = [
        {
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 100,
            "security_type": "Stock",
        }
    ]

    result = reconcile_wins_positions(wins, tracked)

    assert result["status"] == "partial"
    assert result["matched"][0]["field_matches"] == {
        "quantity": True,
        "total_cost": True,
        "current_value": None,
    }
    assert result["totals"]["wins"]["current_value"] is None
    assert result["totals"]["difference"]["current_value"] is None


def test_tolerances_are_applied_only_to_status_and_raw_difference_is_preserved():
    wins = [
        {
            "ticker": "AAA",
            "quantity": 10.000001,
            "cost": 1_000.004,
            "current value": 1_100.009,
            "type": "Stock",
        }
    ]
    tracked = [
        {
            "ticker": "AAA",
            "quantity": 10,
            "entry_price": 100,
            "last_price": 110,
            "type": "Stock",
        }
    ]

    result = reconcile_wins_positions(
        wins,
        tracked,
        quantity_tolerance=0.00001,
        currency_tolerance=0.01,
    )

    assert result["status"] == "reconciled"
    assert result["matched"][0]["quantity_difference"] == pytest.approx(0.000001)
    assert result["matched"][0]["cost_difference"] == pytest.approx(0.004)
    assert result["matched"][0]["value_difference"] == pytest.approx(0.009)


def test_empty_inputs_return_explicit_no_data_status():
    result = reconcile_wins_positions([], [])

    assert result["status"] == "no_data"
    assert result["is_reconciled"] is False
    assert result["coverage_pct"] == 100
    assert result["matched"] == []
    assert result["missing"] == []
    assert result["extra"] == []


def test_closed_wins_rows_are_excluded_from_open_snapshot():
    rows = [
        {"ticker": "OPEN", "quantity": 2, "status": "Open"},
        {"ticker": "OLD", "quantity": 3, "status": "Closed"},
        {"ticker": "SOLD", "quantity": 4, "position status": "Sold"},
    ]

    assert [row["ticker"] for row in normalize_wins_rows(rows)] == ["OPEN"]


def test_equivalent_security_type_labels_do_not_create_false_difference():
    result = reconcile_wins_positions(
        [{"ticker": "AAA", "quantity": 10, "cost": 1_000, "current value": 1_100, "type": "Common Stock"}],
        [{"ticker": "AAA", "quantity": 10, "entry_price": 100, "last_price": 110, "security_type": "Equity"}],
    )

    assert result["status"] == "reconciled"
    assert result["matched"][0]["security_type_match"] is True
