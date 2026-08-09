import numpy as np
import pytest

from src.portfolio_tracker import manager


def test_live_value_pnl_uses_cost_of_priced_positions_only(monkeypatch):
    portfolio = {
        "positions": [
            {"ticker": "AAPL", "shares": 10, "cost_basis": 100},
            {"ticker": "MSFT", "shares": 5, "cost_basis": 200},
        ]
    }
    monkeypatch.setattr(manager, "_fetch_latest_prices", lambda tickers: {"AAPL": 110.0})

    holdings, summary = manager.compute_live_values(portfolio)

    priced = holdings.loc[holdings["Ticker"] == "AAPL"].iloc[0]
    unpriced = holdings.loc[holdings["Ticker"] == "MSFT"].iloc[0]
    assert priced["MarketValue"] == pytest.approx(1_100.0)
    assert priced["PnL"] == pytest.approx(100.0)
    assert np.isnan(unpriced["MarketValue"])
    assert np.isnan(unpriced["PnL"])

    assert summary["TotalMarketValue"] == pytest.approx(1_100.0)
    assert summary["TotalCostValue"] == pytest.approx(2_000.0)
    assert summary["PricedCostValue"] == pytest.approx(1_000.0)
    assert summary["TotalPnL"] == pytest.approx(100.0)
    assert summary["UnpricedCostValue"] == pytest.approx(1_000.0)
    assert summary["PricedPositions"] == 1.0
    assert summary["TotalPositions"] == 2.0
    assert summary["UnpricedPositions"] == 1.0
    assert summary["PriceCoveragePct"] == pytest.approx(0.5)
    assert summary["PnlCoveragePct"] == pytest.approx(0.5)
    assert summary["PartialCoverage"] is True


def test_live_value_pnl_excludes_quoted_position_without_cost_basis(monkeypatch):
    portfolio = {
        "positions": [
            {"ticker": "AAPL", "shares": 10, "cost_basis": None},
        ]
    }
    monkeypatch.setattr(manager, "_fetch_latest_prices", lambda tickers: {"AAPL": 110.0})

    holdings, summary = manager.compute_live_values(portfolio)

    assert holdings.iloc[0]["MarketValue"] == pytest.approx(1_100.0)
    assert np.isnan(holdings.iloc[0]["PnL"])
    assert summary["TotalMarketValue"] == pytest.approx(1_100.0)
    assert summary["TotalPnL"] == pytest.approx(0.0)
    assert summary["PnlCoveredPositions"] == 0.0
    assert summary["PnlCoveragePct"] == 0.0
    assert summary["MissingCostPositions"] == 1.0
    assert summary["MissingCostMarketValue"] == pytest.approx(1_100.0)
    assert summary["PartialCoverage"] is True
