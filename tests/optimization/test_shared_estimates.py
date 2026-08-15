from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    calculate_efficient_frontier,
    clean_returns,
    estimate_black_litterman_inputs,
    estimate_portfolio_inputs,
    optimize_cost_aware_rebalance,
    optimize_maximum_sharpe,
    optimize_minimum_variance,
    sample_portfolio_cloud,
)


def _sample_returns(periods: int = 220, assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    common = rng.normal(0.00035, 0.006, size=(periods, 1))
    residual = rng.normal(0.00015, 0.009, size=(periods, assets))
    return pd.DataFrame(
        common + residual,
        columns=[f"A{index}" for index in range(assets)],
    )


def _legacy_clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(returns).copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if frame.empty:
        raise ValueError("returns are empty after cleaning.")
    if frame.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")
    return frame.astype(float)


def test_clean_returns_numeric_fast_path_matches_legacy_for_non_finite_rows(
    monkeypatch,
):
    returns = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03, 0.04, 0.05],
            "B": [0.02, 0.03, np.inf, 0.05, 0.06],
            "C": [0.03, 0.04, 0.05, -np.inf, 0.07],
        },
        index=[10, 20, 30, 40, 50],
    )
    expected = _legacy_clean_returns(returns)

    def reject_apply(*_args, **_kwargs):
        raise AssertionError("numeric clean_returns input must bypass DataFrame.apply")

    monkeypatch.setattr(pd.DataFrame, "apply", reject_apply)
    actual = clean_returns(returns)

    pd.testing.assert_frame_equal(actual, expected)


def test_clean_returns_nullable_numeric_fast_path_matches_legacy(monkeypatch):
    returns = pd.DataFrame(
        {
            "float": pd.Series([0.01, pd.NA, 0.03, 0.04], dtype="Float64"),
            "integer": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "flag": pd.Series([True, False, True, False], dtype="boolean"),
        }
    )
    expected = _legacy_clean_returns(returns)

    def reject_apply(*_args, **_kwargs):
        raise AssertionError("nullable numeric input must bypass DataFrame.apply")

    monkeypatch.setattr(pd.DataFrame, "apply", reject_apply)
    actual = clean_returns(returns)

    pd.testing.assert_frame_equal(actual, expected)


def test_clean_returns_mixed_string_fallback_matches_legacy(monkeypatch):
    returns = pd.DataFrame(
        {
            "numeric": pd.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype="Float64"),
            "text": ["0.02", "bad", "0.04", None, "0.06"],
            "object": [1, "2", 3.0, 4, 5],
        },
        index=[11, 12, 13, 14, 15],
    )
    expected = _legacy_clean_returns(returns)
    original_apply = pd.DataFrame.apply
    apply_calls = 0

    def track_apply(self, *args, **kwargs):
        nonlocal apply_calls
        apply_calls += 1
        return original_apply(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "apply", track_apply)
    actual = clean_returns(returns)

    assert apply_calls == 1
    pd.testing.assert_frame_equal(actual, expected)


def test_shared_estimates_are_finite_psd_and_auditable():
    returns = _sample_returns()
    returns.iloc[0, 0] = np.inf
    returns.iloc[1, 1] = np.nan

    estimates = estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=0.35,
        return_shrinkage=0.60,
    )

    assert estimates.observations == len(returns) - 2
    assert np.all(np.isfinite(estimates.mean_returns))
    assert np.min(np.linalg.eigvalsh(estimates.covariance)) > 0
    assert estimates.metadata()["covariance_shrinkage"] == pytest.approx(0.35)
    assert estimates.metadata()["return_shrinkage"] == pytest.approx(0.60)


def test_optimizers_use_the_same_default_estimation_contract():
    returns = _sample_returns()
    current = np.full(returns.shape[1], 1.0 / returns.shape[1])

    minimum_variance = optimize_minimum_variance(returns)
    maximum_sharpe = optimize_maximum_sharpe(returns)
    cost_aware = optimize_cost_aware_rebalance(
        returns,
        current,
        max_weight=0.50,
        turnover_limit=1.0,
    )
    frontier = calculate_efficient_frontier(returns, n_points=12)
    cloud = sample_portfolio_cloud(returns, n_samples=50)

    metadata = minimum_variance["estimation"]
    assert maximum_sharpe["estimation"] == metadata
    assert cost_aware["estimation"] == metadata
    assert frontier[0]["estimation"] == metadata
    assert cloud.attrs["estimation"] == metadata


def test_precomputed_estimates_preserve_results_and_reject_misalignment():
    returns = _sample_returns()
    estimates = estimate_portfolio_inputs(returns)

    direct = optimize_maximum_sharpe(returns, max_weight=0.40)
    reused = optimize_maximum_sharpe(
        returns,
        max_weight=0.40,
        portfolio_estimates=estimates,
    )

    assert reused["success"] is True
    assert np.array_equal(reused["weights"], direct["weights"])
    assert reused["estimation"] == direct["estimation"]
    with pytest.raises(ValueError, match="symbols must match"):
        optimize_minimum_variance(
            returns[list(reversed(returns.columns))],
            portfolio_estimates=estimates,
        )


def test_frontier_starts_at_global_minimum_variance_and_uses_supplied_rate():
    returns = _sample_returns()
    risk_free_rate = 0.0175
    minimum_variance = optimize_minimum_variance(
        returns,
        risk_free_rate=risk_free_rate,
    )
    frontier = calculate_efficient_frontier(
        returns,
        n_points=18,
        risk_free_rate=risk_free_rate,
    )

    first = frontier[0]
    assert first["volatility"] == pytest.approx(
        minimum_variance["volatility"], abs=1e-8
    )
    assert np.allclose(first["weights"], minimum_variance["weights"], atol=1e-7)
    assert all(
        float(point["volatility"]) >= float(first["volatility"]) - 1e-8
        for point in frontier
    )
    assert np.all(np.diff([float(point["return"]) for point in frontier]) >= -1e-8)
    assert first["sharpe_ratio"] == pytest.approx(
        (float(first["return"]) - risk_free_rate) / float(first["volatility"])
    )


def test_portfolio_cloud_respects_position_cap():
    returns = _sample_returns(assets=5)
    cloud = sample_portfolio_cloud(
        returns,
        n_samples=250,
        max_weight=0.30,
    )

    assert float(cloud["max_weight"].max()) <= 0.30 + 1e-10


def test_black_litterman_view_space_solve_matches_precision_space_reference(
    monkeypatch,
):
    returns = _sample_returns(periods=480, assets=18)
    symbols = list(returns.columns)
    market_weights = np.linspace(1.0, 2.0, len(symbols))
    market_weights /= market_weights.sum()
    views = {"A1": 0.08, "A7": 0.04, "A15": -0.02}
    confidences = {"A1": 0.25, "A7": 0.60, "A15": 0.90}
    risk_aversion = 3.25
    tau = 0.07
    covariance_shrinkage = 0.17
    return_shrinkage = 0.63

    estimates = estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    covariance = estimates.covariance
    equilibrium = risk_aversion * covariance @ market_weights
    view_symbols = [symbol for symbol in symbols if symbol in views]
    pick = np.zeros((len(view_symbols), len(symbols)), dtype=float)
    view_returns = np.asarray([views[symbol] for symbol in view_symbols])
    scaled_covariance = tau * covariance
    omega_diagonal = np.empty(len(view_symbols), dtype=float)
    for row, symbol in enumerate(view_symbols):
        pick[row, symbols.index(symbol)] = 1.0
        base_uncertainty = float(pick[row] @ scaled_covariance @ pick[row])
        confidence = confidences[symbol]
        omega_diagonal[row] = max(
            base_uncertainty * (1.0 - confidence) / confidence,
            1e-12,
        )

    inverse_scaled_covariance = np.linalg.pinv(scaled_covariance)
    inverse_omega = np.diag(1.0 / omega_diagonal)
    reference_covariance = np.linalg.pinv(
        inverse_scaled_covariance + pick.T @ inverse_omega @ pick
    )
    reference = reference_covariance @ (
        inverse_scaled_covariance @ equilibrium
        + pick.T @ inverse_omega @ view_returns
    )

    def reject_pseudoinverse(*_args, **_kwargs):
        raise AssertionError("Black-Litterman update must solve in view space")

    monkeypatch.setattr(np.linalg, "pinv", reject_pseudoinverse)
    optimized = estimate_black_litterman_inputs(
        returns,
        market_weights=market_weights,
        views=views,
        view_confidences=confidences,
        risk_aversion=risk_aversion,
        tau=tau,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )

    np.testing.assert_allclose(optimized.mean_returns, reference, rtol=1e-8, atol=1e-10)


def test_portfolio_cloud_variance_matches_direct_quadratic_form():
    returns = _sample_returns(periods=320, assets=13)
    estimates = estimate_portfolio_inputs(returns)
    n_samples = 1024
    random_seed = 20260815
    weights = np.random.default_rng(random_seed).dirichlet(
        np.ones(len(estimates.symbols)),
        size=n_samples,
    )
    reference_variance = np.einsum(
        "ij,jk,ik->i",
        weights,
        estimates.covariance,
        weights,
    )

    cloud = sample_portfolio_cloud(
        returns,
        n_samples=n_samples,
        random_seed=random_seed,
        portfolio_estimates=estimates,
    )

    np.testing.assert_allclose(
        cloud["volatility"].to_numpy(),
        np.sqrt(np.clip(reference_variance, a_min=0.0, a_max=None)),
        rtol=1e-13,
        atol=1e-15,
    )
