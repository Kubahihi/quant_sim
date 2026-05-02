from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.modular.models import run_model_bundle
from src.analytics.scenario_playground import (
    build_role_exposure_table,
    build_scenario_suite,
    classify_asset_role,
    list_scenario_presets,
    run_scenario_preset,
)
from src.visualization.cockpit_charts import (
    plot_asset_stress_impact,
    plot_crisis_playback,
    plot_phase_timeline,
    plot_scenario_atlas,
    plot_scenario_fingerprint,
    plot_scenario_shock_map,
)


def _sample_returns(n: int = 180) -> pd.Series:
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0.0004, scale=0.01, size=n)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.Series(data, index=idx)


def _sample_returns_df(n: int = 180) -> pd.DataFrame:
    base = _sample_returns(n)
    return pd.DataFrame(
        {
            "AAPL": base,
            "BND": base * 0.2 + 0.0001,
            "GLD": base * -0.1 + 0.0002,
            "BTC": base * 1.8 - 0.0003,
        }
    )


def test_lasso_zero_signal_confidence_stays_low():
    flat = pd.Series(np.zeros(40), index=pd.date_range("2024-01-01", periods=40, freq="B"))
    models = run_model_bundle(flat, context={"returns_df": pd.DataFrame({"AAA": flat, "BBB": flat})})
    lasso = models["lasso"]

    assert lasso.available is True
    assert lasso.metrics["sparse_beta"] == 0.0
    assert lasso.confidence <= 0.05


def test_kalman_confidence_is_not_artificially_unity():
    flat = pd.Series(np.zeros(40), index=pd.date_range("2024-01-01", periods=40, freq="B"))
    models = run_model_bundle(flat, context={"returns_df": pd.DataFrame({"AAA": flat, "BBB": flat})})
    kalman = models["kalman_filter"]

    assert kalman.available is True
    assert kalman.confidence < 0.85


def test_black_litterman_without_views_is_reference_mode():
    returns_df = _sample_returns_df()
    series = returns_df.mean(axis=1)
    models = run_model_bundle(
        series,
        context={"returns_df": returns_df, "market_weights": [0.4, 0.3, 0.2, 0.1]},
    )
    black_litterman = models["black_litterman"]

    assert black_litterman.available is True
    assert black_litterman.metrics["passive_reference_only"] == 1.0
    assert black_litterman.metrics["view_count"] == 0.0
    assert black_litterman.confidence <= 0.35


def test_scenario_suite_builds_extreme_playground_outputs():
    returns_df = _sample_returns_df()
    suite = build_scenario_suite(
        returns_df=returns_df,
        tickers=list(returns_df.columns),
        weights=np.array([0.4, 0.3, 0.2, 0.1]),
        severity=1.2,
        initial_value=150_000.0,
        horizon_override=30,
    )

    assert not suite["rows"].empty
    assert set(suite["scenarios"].keys()) == {item["name"] for item in list_scenario_presets()}

    flash_crash = suite["scenarios"]["Flash Crash"]
    assert flash_crash["stressed_stats"]["final_value"] > 0
    assert flash_crash["baseline_path"].iloc[0] == 150_000.0
    assert flash_crash["stressed_path"].iloc[0] == 150_000.0
    assert len(flash_crash["asset_impact_proxy"]) == returns_df.shape[1]


def test_asset_role_classification_and_exposure_table():
    assert classify_asset_role("BND") == "bond"
    assert classify_asset_role("GLD") == "gold"
    assert classify_asset_role("BTC") == "crypto"
    assert classify_asset_role("AAPL") == "equity"

    exposure = build_role_exposure_table(["AAPL", "BND", "GLD"], np.array([0.5, 0.3, 0.2]))
    assert list(exposure["Role"]) == ["equity", "bond", "gold"]


def test_historical_crisis_presets_and_phase_outputs_exist():
    preset_names = {item["name"] for item in list_scenario_presets()}
    assert {
        "2008 Global Financial Crisis",
        "Dot-Com Bust",
        "1970s Stagflation Spiral",
        "1929 Great Depression Analog",
    }.issubset(preset_names)

    returns_df = _sample_returns_df()
    scenario = run_scenario_preset(
        returns_df=returns_df,
        tickers=list(returns_df.columns),
        weights=np.array([0.4, 0.3, 0.2, 0.1]),
        preset_name="2008 Global Financial Crisis",
        severity=1.0,
        initial_value=100_000.0,
        horizon_override=60,
    )

    assert scenario["category"] == "Historical Crisis"
    assert scenario["era"] == "2008-2009"
    assert not scenario["phase_table"].empty
    assert not scenario["shock_map"].empty
    assert len(scenario["daily_phase_labels"]) == 60


def test_cockpit_visualizations_build_plotly_outputs():
    returns_df = _sample_returns_df()
    suite = build_scenario_suite(
        returns_df=returns_df,
        tickers=list(returns_df.columns),
        weights=np.array([0.4, 0.3, 0.2, 0.1]),
        severity=1.0,
        initial_value=100_000.0,
        horizon_override=30,
    )
    selected = suite["scenarios"]["2008 Global Financial Crisis"]

    atlas_fig = plot_scenario_atlas(suite["rows"], highlight_scenario="2008 Global Financial Crisis")
    playback_fig = plot_crisis_playback(
        scenario_name="2008 Global Financial Crisis",
        baseline_path=selected["baseline_path"],
        stressed_path=selected["stressed_path"],
        daily_phase_labels=selected["daily_phase_labels"],
    )
    timeline_fig = plot_phase_timeline("2008 Global Financial Crisis", selected["phase_table"])
    heatmap_fig = plot_scenario_shock_map(selected["shock_map"])
    fingerprint_fig = plot_scenario_fingerprint(
        scenario_name="2008 Global Financial Crisis",
        stressed_stats=selected["stressed_stats"],
        baseline_stats=selected["baseline_stats"],
        horizon_days=selected["horizon_days"],
    )
    impact_fig = plot_asset_stress_impact(selected["asset_impact_proxy"])

    assert len(atlas_fig.data) >= 1
    assert len(playback_fig.frames) == len(selected["baseline_path"])
    assert playback_fig.layout.updatemenus[0].buttons[2].label == "Reset to start"
    assert len(timeline_fig.data) == len(selected["phase_table"])
    assert len(heatmap_fig.data) == 1
    assert len(fingerprint_fig.data) == 2
    assert len(impact_fig.data) == 1
