from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.portfolio_tracker.manual_bond_quant import parse_manual_bond_rows
from ui.pages import wharton_dash


def test_quant_run_combines_market_asset_and_manual_individual_bond(monkeypatch):
    import src.analytics as analytics
    import src.simulation as simulation

    index = pd.date_range("2024-01-02", periods=520, freq="B")
    steps = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * np.exp(0.00035 * steps + 0.012 * np.sin(steps / 13.0)),
            "IEF": 100.0 * np.exp(0.00012 * steps + 0.006 * np.cos(steps / 17.0)),
        },
        index=index,
    )

    def fake_prices(symbols, start_date, end_date):
        return prices[[symbol for symbol in symbols if symbol in prices.columns]].copy()

    shared_estimates = SimpleNamespace(
        mean_returns=prices.pct_change().dropna().mean().to_numpy(dtype=float) * 252.0,
        covariance=prices.pct_change().dropna().cov().to_numpy(dtype=float) * 252.0,
    )
    optimizer_estimates = []

    monkeypatch.setattr(wharton_dash, "_fetch_close_prices_cached", fake_prices)
    monkeypatch.setattr(
        wharton_dash,
        "_fetch_market_data_cached",
        lambda symbols, start_date, end_date: {
            "prices": fake_prices(symbols, start_date, end_date),
            "average_daily_dollar_volume": {
                symbol: 10_000_000.0 for symbol in symbols if symbol in prices.columns
            },
            "adv_history": pd.DataFrame(
                {
                    symbol: 10_000_000.0
                    for symbol in symbols if symbol in prices.columns
                },
                index=index,
            ),
        },
    )
    def fake_optimization(returns, **kwargs):
        optimizer_estimates.append(kwargs.get("portfolio_estimates"))
        weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])
        return {
            "weights": weights,
            "symbols": returns.columns.tolist(),
            "expected_return": float(returns.mean().dot(weights) * 252.0),
            "volatility": float(np.sqrt(weights @ returns.cov().to_numpy() @ weights) * np.sqrt(252.0)),
            "sharpe_ratio": 0.0,
            "turnover": 0.0,
            "success": True,
        }

    fake_optimization_module = SimpleNamespace(
        optimize_minimum_variance=fake_optimization,
        optimize_maximum_sharpe=fake_optimization,
        optimize_cost_aware_rebalance=fake_optimization,
        calculate_efficient_frontier=lambda returns, **kwargs: [],
        sample_portfolio_cloud=lambda returns, **kwargs: pd.DataFrame(),
        run_optimization_walk_forward=lambda returns, **kwargs: {
            "success": True,
            "metrics": {},
            "equal_weight_metrics": {},
        },
        estimate_portfolio_inputs=lambda returns: shared_estimates,
        calculate_portfolio_statistics=lambda weights, mean_returns, cov_matrix, **kwargs: {
            "return": float(np.asarray(weights) @ mean_returns),
            "volatility": float(
                np.sqrt(np.asarray(weights) @ cov_matrix @ np.asarray(weights))
            ),
            "sharpe_ratio": 0.0,
        },
        optimize_portfolio=lambda returns, current_weights, **kwargs: {
            **fake_optimization(returns),
            "objective": kwargs.get("objective", "maximum_utility"),
            "current_weights": np.asarray(current_weights, dtype=float),
            "historical_cvar_daily": 0.0,
            "transaction_cost_drag": 0.0,
            "constraint_report": [],
            "warnings": [],
        },
        build_execution_plan=lambda *args, **kwargs: {
            "success": True,
            "trades": [],
            "warnings": [],
            "holding_count": len(args[0]),
            "cash": 0.0,
            "cash_weight": 0.0,
            "total_execution_cost": 0.0,
            "total_execution_cost_drag": 0.0,
            "estimated_tax": 0.0,
            "tracking_difference_l1": 0.0,
        },
    )
    monkeypatch.setattr(
        wharton_dash,
        "_load_quant_modules",
        lambda: {
            "analytics": analytics,
            "optimization": fake_optimization_module,
            "simulation": simulation,
            "yahoo_fetcher": SimpleNamespace(),
        },
    )
    monkeypatch.setattr(
        wharton_dash,
        "_load_modular_pipeline",
        lambda: SimpleNamespace(run_quant_stack=lambda **kwargs: {"backtest": {}}),
    )
    bonds = parse_manual_bond_rows(
        [{
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
        }],
        as_of="2026-08-03",
    )

    result = wharton_dash._compute_quant_run(
        tickers=["SPY"],
        weights=np.asarray([1.0]),
        benchmark_ticker="SPY",
        start_date=date(2024, 1, 2),
        end_date=date(2026, 8, 3),
        risk_free_rate=0.03,
        current_value=100_000.0,
        max_weight=0.34,
        turnover_limit=0.50,
        transaction_cost_bps=10.0,
        risk_aversion=3.0,
        simulation_days=30,
        n_simulations=200,
        random_seed=42,
        jump_intensity=1.0,
        jump_mean=-0.05,
        jump_volatility=0.08,
        manual_bonds=bonds,
        strategy_rulebook={"min_cash_weight": 0.10},
    )

    assert result["tickers"] == ["SPY", "BOND:US0000000001"]
    assert result["weights"].tolist() == pytest.approx([0.75, 0.25])
    assert result["security_types"]["BOND:US0000000001"] == "Bond"
    assert result["manual_bond_metrics"].loc[0, "AllocatedMarketValueUSD"] == pytest.approx(25_000.0)
    assert result["manual_bond_metrics"].loc[0, "DV01USD"] == pytest.approx(10.5)
    assert result["returns"].columns.tolist() == ["SPY", "BOND:US0000000001"]
    assert result["inputs"]["manual_bond_count"] == 1
    assert result["inputs"]["max_weight"] == pytest.approx(0.50)
    assert result["inputs"]["mandate_max_weight"] == pytest.approx(0.34)
    assert result["inputs"]["synthetic_cash_proxy"] is True
    assert optimizer_estimates[:3] == [shared_estimates] * 3
    assert result["mandate_aware"]["symbols"][-1] == "CASH"
    assert any("synthetic" in warning for warning in result["mandate_aware"]["warnings"])
    assert result["quant_stack"] is None

    enriched = wharton_dash._run_quant_research_stack(result)
    assert enriched["quant_stack"] == {"backtest": {}}
    assert enriched["runtime"]["research_stack_seconds"] >= 0.0

    simulated = wharton_dash._run_quant_simulations(enriched)
    assert simulated["price_paths"] is None
    assert simulated["adv_price_paths"] is None
    assert simulated["simulation_artifacts"]["sample_paths"].shape == (31, 50)
    assert simulated["advanced_simulation_artifacts"]["sample_paths"].shape == (31, 50)
    assert simulated["simulation_artifacts"]["terminal_values"].shape == (200,)
    assert simulated["advanced_simulation_artifacts"]["terminal_values"].shape == (200,)
    assert wharton_dash._simulations_ready(simulated) is True
    assert simulated["simulation_stats"]["random_seed"] == 42
    assert simulated["runtime"]["simulation_seconds"] >= 0.0


def test_compact_simulation_artifacts_preserve_displayed_distribution() -> None:
    paths = np.random.default_rng(818).lognormal(size=(91, 1_000))

    artifacts = wharton_dash._compact_simulation_paths(paths)

    assert artifacts["path_count"] == 1_000
    assert artifacts["period_count"] == 90
    assert artifacts["sample_paths"].shape == (91, 50)
    assert np.array_equal(artifacts["terminal_values"], paths[-1])
    for percentile in wharton_dash._SIMULATION_DISPLAY_PERCENTILES:
        assert np.array_equal(
            artifacts["percentile_paths"][f"p{percentile}"],
            np.percentile(paths, percentile, axis=1),
        )

    compact_bytes = (
        artifacts["terminal_values"].nbytes
        + artifacts["sample_paths"].nbytes
        + sum(values.nbytes for values in artifacts["percentile_paths"].values())
    )
    assert compact_bytes < paths.nbytes * 0.10


def test_legacy_simulation_matrix_is_compacted_and_released() -> None:
    paths = np.random.default_rng(17).lognormal(size=(21, 100))
    result = {"price_paths": paths, "simulation_artifacts": None}

    artifacts = wharton_dash._simulation_artifacts(result)

    assert artifacts is result["simulation_artifacts"]
    assert result["price_paths"] is None
    assert np.array_equal(artifacts["terminal_values"], paths[-1])


def test_compact_artifacts_render_in_baseline_and_advanced_views(monkeypatch) -> None:
    class MetricColumn:
        def metric(self, *_args, **_kwargs):
            return None

    class StreamlitStub:
        def __init__(self):
            self.plotly_calls = 0
            self.warnings: list[str] = []

        def columns(self, count):
            return [MetricColumn() for _ in range(count)]

        def markdown(self, *_args, **_kwargs):
            return None

        def plotly_chart(self, *_args, **_kwargs):
            self.plotly_calls += 1

        def line_chart(self, *_args, **_kwargs):
            return None

        def caption(self, *_args, **_kwargs):
            return None

        def warning(self, message, **_kwargs):
            self.warnings.append(str(message))

        def info(self, *_args, **_kwargs):
            return None

    paths = np.random.default_rng(27).lognormal(size=(31, 200)) * 100_000.0
    artifacts = wharton_dash._compact_simulation_paths(paths)
    terminal = paths[-1]
    stats = {
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "std": float(np.std(terminal)),
        "percentile_5": float(np.percentile(terminal, 5)),
        "percentile_95": float(np.percentile(terminal, 95)),
    }
    result = {
        "inputs": {
            "current_value": 100_000.0,
            "n_simulations": 200,
            "simulation_days": 30,
            "random_seed": 27,
        },
        "simulation_stats": stats,
        "adv_simulation_stats": stats,
        "simulation_artifacts": artifacts,
        "advanced_simulation_artifacts": artifacts,
        "price_paths": None,
        "adv_price_paths": None,
    }
    streamlit_stub = StreamlitStub()
    monkeypatch.setattr(wharton_dash, "st", streamlit_stub)

    wharton_dash._render_monte_carlo(result)
    wharton_dash._render_advanced_monte_carlo(result)

    assert streamlit_stub.plotly_calls == 6
    assert streamlit_stub.warnings == []
