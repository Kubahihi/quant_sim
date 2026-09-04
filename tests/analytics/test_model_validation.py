from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.model_validation import (
    build_model_validation_report,
    distribution_diagnostics,
    moving_block_bootstrap_intervals,
)
from src.analytics.modular.backtest import walk_forward_baseline_backtest
from src.simulation.monte_carlo import (
    run_advanced_monte_carlo_simulation,
    run_monte_carlo_simulation,
)


def _returns(n: int = 756) -> pd.Series:
    rng = np.random.default_rng(123)
    values = rng.standard_t(df=5, size=n) * 0.009 + 0.0003
    return pd.Series(values, index=pd.date_range("2022-01-03", periods=n, freq="B"))


def test_block_bootstrap_intervals_are_seeded_and_ordered():
    series = _returns(504)
    first = moving_block_bootstrap_intervals(series, n_bootstrap=200, random_seed=7)
    second = moving_block_bootstrap_intervals(series, n_bootstrap=200, random_seed=7)

    assert first == second
    assert set(first) == {"annualized_return", "volatility", "sharpe_ratio", "var_95", "cvar_95"}
    for interval in first.values():
        assert interval["ci_low"] <= interval["ci_high"]


def test_block_bootstrap_rejects_invalid_annual_risk_free_rate():
    with np.testing.assert_raises_regex(ValueError, "greater than -100%"):
        moving_block_bootstrap_intervals(
            _returns(100),
            risk_free_rate=-1.0,
            n_bootstrap=100,
        )


def test_distribution_diagnostics_detect_fat_tails_without_scipy_dependency():
    diagnostics = distribution_diagnostics(_returns(2000))
    assert diagnostics["excess_kurtosis"] > 0
    assert 0.0 <= diagnostics["normality_p_value"] <= 1.0


def test_validation_report_does_not_claim_predictive_accuracy():
    series = _returns()
    _, simulation = run_monte_carlo_simulation(
        current_value=100_000,
        expected_return=float(series.mean() * 252),
        volatility=float(series.std() * np.sqrt(252)),
        time_horizon=252,
        n_simulations=5_000,
        random_seed=42,
    )
    backtest = walk_forward_baseline_backtest(series)
    report = build_model_validation_report(
        series,
        simulation_stats=simulation,
        backtest=backtest,
        n_bootstrap=200,
    )

    assert 0.0 <= report["methodology_score"] <= 100.0
    assert report["predictive_accuracy_measured"] is False
    assert report["generated_with"]["backtest_validation_type"] == "walk_forward_causal_baseline"
    assert any("full model ensemble" in item.lower() for item in report["limitations"])


def test_validation_report_reuses_rolling_optimizer_and_snapshot_evidence():
    series = _returns()
    optimization_validation = {
        "success": True,
        "causal": True,
        "validation_type": "point_in_time_rolling_reoptimization_out_of_sample",
        "survivorship_bias_controlled": True,
        "windows": [{"window_id": 1}, {"window_id": 2}],
        "metrics": {"transaction_cost_drag": 0.002, "annualized_return": 0.08},
        "equal_weight_metrics": {
            "transaction_cost_drag": 0.001,
            "annualized_return": 0.06,
        },
    }

    report = build_model_validation_report(
        series,
        backtest=walk_forward_baseline_backtest(series),
        optimization_validation=optimization_validation,
        data_snapshot_id="wins-42",
        n_bootstrap=200,
    )

    oos_gate = next(item for item in report["gates"] if item["gate"] == "Out-of-sample process")
    assert oos_gate["status"] == "pass"
    assert oos_gate["points"] == 18.0
    evidence = {item["key"]: item for item in report["accuracy_evidence"]["checks"]}
    assert evidence["causal_oos"]["status"] == "pass"
    assert evidence["after_costs"]["status"] == "pass"
    assert evidence["comparator"]["status"] == "pass"
    assert evidence["point_in_time_universe"]["status"] == "pass"
    assert evidence["frozen_snapshot"]["status"] == "pass"
    assert evidence["full_ensemble"]["status"] == "partial"
    assert evidence["untouched_holdout"]["status"] == "gap"
    assert report["accuracy_evidence"]["decision_use"] == "decision_support_only"
    assert report["generated_with"]["data_snapshot_id"] == "wins-42"


def test_failed_rolling_window_is_only_partial_accuracy_evidence():
    series = _returns()
    report = build_model_validation_report(
        series,
        backtest=walk_forward_baseline_backtest(series),
        optimization_validation={
            "success": False,
            "causal": True,
            "validation_type": "rolling_reoptimization_out_of_sample",
            "survivorship_bias_controlled": True,
            "windows": [{"window_id": 1}, {"window_id": 2, "success": False}],
            "metrics": {"transaction_cost_drag": 0.002},
            "equal_weight_metrics": {"annualized_return": 0.06},
        },
        n_bootstrap=200,
    )

    oos_gate = next(item for item in report["gates"] if item["gate"] == "Out-of-sample process")
    causal_check = next(
        item
        for item in report["accuracy_evidence"]["checks"]
        if item["key"] == "causal_oos"
    )
    assert oos_gate["status"] == "warning"
    assert oos_gate["points"] == 10.0
    assert causal_check["status"] == "partial"


def test_small_sample_bootstrap_is_not_awarded_full_uncertainty_credit():
    report = build_model_validation_report(_returns(20), n_bootstrap=100)

    gate = next(item for item in report["gates"] if item["gate"] == "Parameter uncertainty")
    assert gate["status"] == "fail"
    assert gate["points"] == 3.0


def test_missing_tail_diagnostics_cannot_receive_full_simulation_credit():
    report = build_model_validation_report(
        _returns(504),
        simulation_stats={
            "model": "geometric_brownian_motion",
            "relative_standard_error_mean": 0.001,
            "random_seed": 7,
        },
        n_bootstrap=100,
    )

    gate = next(
        item for item in report["gates"]
        if item["gate"] == "Simulation convergence"
    )
    assert gate["status"] == "warning"
    assert gate["points"] <= 8.0
    assert "not reported" in gate["evidence"].lower()


def test_merton_model_metadata_is_recognized_but_not_claimed_as_complete_validation():
    series = _returns(504)
    _, simulation = run_advanced_monte_carlo_simulation(
        current_value=100_000.0,
        expected_return=0.08,
        volatility=0.20,
        n_simulations=2_000,
        random_seed=12,
    )
    report = build_model_validation_report(
        series,
        simulation_stats=simulation,
        n_bootstrap=100,
    )

    gate = next(item for item in report["gates"] if item["gate"] == "Distribution/model risk")
    assert gate["status"] == "warning"
    assert gate["points"] > 0.0
    assert "Merton" in gate["evidence"]
    assert any(
        "Merton jump diffusion" in item
        for item in report["presentation_caveats"]
    )
    assert not any(
        item.startswith("GBM") for item in report["presentation_caveats"]
    )


def test_zero_cost_placeholder_is_not_full_after_cost_evidence():
    series = _returns(504)
    report = build_model_validation_report(
        series,
        backtest={
            "lookahead_safe": True,
            "scope": "Causal baseline.",
            "parameters": {"transaction_cost_bps": 0.0},
        },
        n_bootstrap=100,
    )

    check = next(
        item
        for item in report["accuracy_evidence"]["checks"]
        if item["key"] == "after_costs"
    )
    assert check["status"] == "partial"
    assert "zero" in check["evidence"].lower()
