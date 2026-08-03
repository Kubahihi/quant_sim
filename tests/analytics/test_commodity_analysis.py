from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.commodity_analysis import (
    COMMODITY_SYMBOLS,
    build_cumulative_index,
    build_price_shock_table,
    build_return_correlation,
    calculate_commodity_metrics,
    commodity_catalog_frame,
)


def test_catalog_contains_diversified_etf_and_futures_proxies():
    catalog = commodity_catalog_frame()

    assert {"Ticker", "Instrument", "Group", "Vehicle"}.issubset(catalog.columns)
    assert {"DBC", "GLD", "USO", "DBA", "GC=F", "CL=F"}.issubset(COMMODITY_SYMBOLS)
    assert {"Broad basket", "Precious metals", "Energy", "Agriculture"}.issubset(set(catalog["Group"]))


def test_metrics_rebasing_and_correlations_use_price_history():
    index = pd.bdate_range("2024-01-01", periods=300)
    shared_returns = 0.0008 + 0.0004 * np.sin(np.arange(300) / 9.0)
    prices = pd.DataFrame(
        {
            "DBC": 100.0 * np.cumprod(1.0 + shared_returns),
            "GLD": 150.0 * np.cumprod(1.0 + shared_returns),
        },
        index=index,
    )

    metrics = calculate_commodity_metrics(prices)
    indexed = build_cumulative_index(prices)
    correlation = build_return_correlation(prices)

    assert metrics["Ticker"].tolist() == ["DBC", "GLD"]
    assert metrics.loc[0, "Return1M"] > 0
    assert metrics.loc[0, "Return12M"] > metrics.loc[0, "Return1M"]
    assert math.isclose(indexed["DBC"].iloc[0], 100.0)
    assert correlation.loc["DBC", "GLD"] == pytest.approx(1.0)


def test_price_shock_table_scales_by_units_multiplier_and_fx():
    table = build_price_shock_table(
        80.0,
        2.0,
        contract_multiplier=1_000.0,
        fx_to_usd=1.25,
        shocks=[-0.10, 0.10],
    )

    assert table["PositionValueUSD"].tolist() == pytest.approx([180_000.0, 220_000.0])
    assert table["PnLUSD"].tolist() == pytest.approx([-20_000.0, 20_000.0])


@pytest.mark.parametrize("field,value", [("current_price", 0), ("units", -1), ("contract_multiplier", 0), ("fx_to_usd", np.nan)])
def test_price_shock_table_rejects_invalid_position_inputs(field, value):
    kwargs = {"current_price": 80.0, "units": 1.0, "contract_multiplier": 1.0, "fx_to_usd": 1.0}
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        build_price_shock_table(**kwargs)
