import numpy as np
import pandas as pd
import pytest

from src.analytics.modular.robustness_validation import (
    calculate_dsr,
    calculate_psr,
    create_walk_forward_splits,
    run_walk_forward_validation,
)


@pytest.fixture
def dummy_index():
    regular = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    # Deliberate calendar gaps prove that split sizes are observation counts.
    return regular.delete([4, 7, 15, 31, 63])


def test_create_walk_forward_splits(dummy_index):
    splits = create_walk_forward_splits(dummy_index, train_days=300, test_days=100, step_days=200)

    assert len(splits) == 3

    first_train, first_test = splits[0]

    assert len(first_train) == 300
    assert len(first_test) == 100
    assert first_train.equals(dummy_index[:300])
    assert first_test.equals(dummy_index[300:400])

    second_train, _ = splits[1]
    assert second_train[0] == dummy_index[200]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("train_days", 0),
        ("test_days", -1),
        ("step_days", 0.5),
        ("step_days", True),
    ],
)
def test_create_walk_forward_splits_requires_positive_integer_sizes(
    dummy_index,
    parameter,
    value,
):
    arguments = {"train_days": 10, "test_days": 5, "step_days": 5}
    arguments[parameter] = value

    with pytest.raises(ValueError, match="positive integer"):
        create_walk_forward_splits(dummy_index, **arguments)


@pytest.mark.parametrize(
    "invalid_index",
    [
        pd.Index([1, 1, 2]),
        pd.Index([2, 1, 3]),
    ],
)
def test_create_walk_forward_splits_rejects_ambiguous_index(invalid_index):
    with pytest.raises(ValueError):
        create_walk_forward_splits(invalid_index, train_days=1, test_days=1, step_days=1)


def test_calculate_psr():
    # Normal returns with positive mean should have positive PSR
    np.random.seed(42)
    returns = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=252))
    psr = calculate_psr(returns, benchmark_sr=0.0)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.5  # Since mean is positive
    
    # Negative returns should have PSR < 0.5
    returns_neg = pd.Series(np.random.normal(loc=-0.001, scale=0.01, size=252))
    psr_neg = calculate_psr(returns_neg, benchmark_sr=0.0)
    assert psr_neg < 0.5


def test_calculate_psr_converts_effective_annual_risk_free_rate_to_daily():
    annual_rf = 0.10
    daily_rf = np.expm1(np.log1p(annual_rf) / 252)
    symmetric_noise = np.tile([-0.01, -0.005, 0.005, 0.01], 63)
    returns = pd.Series(daily_rf + symmetric_noise)

    assert calculate_psr(returns, risk_free_rate=annual_rf) == pytest.approx(0.5, abs=1e-12)


def test_calculate_dsr():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=252))
    
    dsr_1_trial = calculate_dsr(returns, num_trials=1, variance_trials=0.1)
    dsr_100_trials = calculate_dsr(returns, num_trials=100, variance_trials=0.1)
    
    # More trials mean lower DSR because of data snooping penalty
    assert dsr_100_trials < dsr_1_trial


@pytest.mark.parametrize(
    ("num_trials", "variance_trials"),
    [(0, 0.1), (1.5, 0.1), (True, 0.1), (2, -0.1), (2, np.inf)],
)
def test_calculate_dsr_rejects_invalid_trial_parameters(num_trials, variance_trials):
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])
    with pytest.raises(ValueError):
        calculate_dsr(returns, num_trials=num_trials, variance_trials=variance_trials)


def test_run_walk_forward_validation(dummy_index):
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(loc=0.0005, scale=0.01, size=len(dummy_index)),
        index=dummy_index,
    )

    results = run_walk_forward_validation(
        returns,
        train_days=300,
        test_days=100,
        step_days=150,
        num_trials=5
    )

    assert "windows" in results
    assert "aggregate_evaluation_returns" in results
    assert "aggregate_oos_returns" not in results
    assert "metrics" in results

    metrics = results["metrics"]
    assert "psr" in metrics
    assert "dsr" in metrics
    assert "evaluation_sharpe" in metrics
    assert "oos_sharpe" not in metrics

    assert len(results["windows"]) > 0
    assert results["validation_type"] == "rolling_fixed_returns_segmentation"
    assert results["strategy_refit_performed"] is False
    assert results["supports_strategy_oos_claim"] is False
    assert metrics["dsr"] is None
    assert "not calculated" in metrics["dsr_interpretation"].lower()


@pytest.mark.parametrize(
    "invalid_returns",
    [
        pd.Series([0.01, np.nan, 0.02]),
        pd.Series([0.01, np.inf, 0.02]),
        pd.Series([0.01, -1.01, 0.02]),
        pd.Series([0.01, 0.02, 0.03], index=[1, 1, 2]),
        pd.Series([0.01, 0.02, 0.03], index=[2, 1, 3]),
    ],
)
def test_robustness_validation_fails_closed_on_invalid_returns(invalid_returns):
    with pytest.raises(ValueError):
        run_walk_forward_validation(
            invalid_returns,
            train_days=1,
            test_days=1,
            step_days=1,
        )
