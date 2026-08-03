from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.portfolio_tracker.manual_bond_quant import (
    build_manual_bond_metrics_table,
    build_manual_bond_proxy_returns,
    combine_hybrid_weights,
    parse_manual_bond_rows,
)


def _row(**overrides):
    values = {
        "Identifier": "US0000000001",
        "Weight %": 25.0,
        "Clean Price": 99.0,
        "Face Value": 1_000.0,
        "Quantity": 10.0,
        "Coupon %": 5.0,
        "Maturity": "2031-08-03",
        "Coupon Frequency": 2,
        "YTM %": 5.25,
        "Modified Duration": 4.2,
        "Convexity": 22.0,
        "Annual Volatility %": 6.0,
        "Proxy Ticker": "IEF",
        "Issuer": "Example Corp",
        "Credit Rating": "A",
    }
    values.update(overrides)
    return values


def test_manual_grid_row_is_normalized_and_valued_as_an_individual_bond():
    bonds = parse_manual_bond_rows([_row()], as_of="2026-08-03")

    assert len(bonds) == 1
    bond = bonds[0]
    assert bond["ticker"] == "BOND:US0000000001"
    assert bond["quant_weight"] == pytest.approx(0.25)
    assert bond["coupon_rate"] == pytest.approx(0.05)
    assert bond["yield_to_maturity"] == pytest.approx(0.0525)
    assert bond["annual_volatility"] == pytest.approx(0.06)

    table = build_manual_bond_metrics_table(bonds, as_of="2026-08-03")
    assert table.loc[0, "YieldToWorst"] == pytest.approx(0.0525)
    assert table.loc[0, "ModifiedDuration"] == pytest.approx(4.2)
    assert table.loc[0, "ProxyTicker"] == "IEF"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"Proxy Ticker": ""}, "Proxy Ticker is required"),
        ({"Maturity": "2026-01-01"}, "Maturity must be after"),
        ({"Annual Volatility %": 0.0}, "Annual Volatility % must be greater"),
        ({"Weight %": 101.0}, "Weight % must be greater"),
    ],
)
def test_invalid_manual_bond_rows_fail_with_actionable_messages(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        parse_manual_bond_rows([_row(**overrides)], as_of="2026-08-03")


def test_proxy_series_matches_entered_mean_and_volatility_without_randomness():
    bonds = parse_manual_bond_rows([_row()], as_of="2026-08-03")
    index = pd.date_range("2025-01-01", periods=300, freq="B")
    proxy = pd.DataFrame(
        {"IEF": np.sin(np.arange(300) / 11.0) * 0.004 + np.cos(np.arange(300) / 7.0) * 0.002},
        index=index,
    )

    first = build_manual_bond_proxy_returns(bonds, proxy, as_of="2026-08-03")
    second = build_manual_bond_proxy_returns(bonds, proxy, as_of="2026-08-03")

    pd.testing.assert_frame_equal(first, second)
    series = first["BOND:US0000000001"]
    assert float(series.mean()) * 252 == pytest.approx(0.0525, abs=1e-12)
    assert float(series.std(ddof=1)) * math.sqrt(252) == pytest.approx(0.06, abs=1e-12)
    assert float(series.corr(proxy["IEF"])) == pytest.approx(1.0)


def test_callable_terms_and_expected_credit_loss_flow_into_proxy_expected_return():
    bonds = parse_manual_bond_rows(
        [_row(**{
            "Callable": True,
            "First Call Date": "2028-08-03",
            "Call Price": 100.0,
            "Default Probability %": 2.0,
            "Recovery Rate %": 40.0,
        })],
        as_of="2026-08-03",
    )
    metrics = build_manual_bond_metrics_table(bonds, as_of="2026-08-03")

    assert bonds[0]["callable"] == 1
    assert metrics.loc[0, "ExpectedLossRate"] == pytest.approx(0.012)
    assert metrics.loc[0, "ProxyExpectedReturn"] == pytest.approx(
        metrics.loc[0, "YieldToWorst"] - 0.012
    )


def test_hybrid_weights_reserve_explicit_bond_weight_and_scale_market_sleeve():
    bonds = parse_manual_bond_rows([_row(**{"Weight %": 25.0})], as_of="2026-08-03")
    columns = ["SPY", "GLD", "BOND:US0000000001"]

    weights = combine_hybrid_weights(
        ["SPY", "GLD"], [0.6, 0.4], bonds, columns,
    )

    assert weights.tolist() == pytest.approx([0.45, 0.30, 0.25])
    assert float(weights.sum()) == pytest.approx(1.0)


def test_manual_bond_only_portfolio_requires_weights_to_total_100_percent():
    partial = parse_manual_bond_rows([_row(**{"Weight %": 25.0})], as_of="2026-08-03")
    with pytest.raises(ValueError, match="must total 100%"):
        combine_hybrid_weights([], [], partial, ["BOND:US0000000001"])

    full = parse_manual_bond_rows([_row(**{"Weight %": 100.0})], as_of="2026-08-03")
    result = combine_hybrid_weights([], [], full, ["BOND:US0000000001"])
    assert result.tolist() == pytest.approx([1.0])
