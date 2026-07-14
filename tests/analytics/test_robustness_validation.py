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
    return pd.date_range(start="2020-01-01", periods=1000, freq="D")


def test_create_walk_forward_splits(dummy_index):
    splits = create_walk_forward_splits(dummy_index, train_days=300, test_days=100, step_days=200)
    
    assert len(splits) > 0
    
    first_train, first_test = splits[0]
    
    # Train should be roughly 300 days
    assert (first_train[-1] - first_train[0]).days <= 300
    
    # Test should immediately follow train
    assert first_train[-1] < first_test[0]
    
    # Test should be roughly 100 days
    assert (first_test[-1] - first_test[0]).days <= 100
    
    # Second split train should start 200 days after first split train
    if len(splits) > 1:
        second_train, _ = splits[1]
        assert (second_train[0] - first_train[0]).days == 200


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


def test_calculate_dsr():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(loc=0.001, scale=0.01, size=252))
    
    dsr_1_trial = calculate_dsr(returns, num_trials=1, variance_trials=0.1)
    dsr_100_trials = calculate_dsr(returns, num_trials=100, variance_trials=0.1)
    
    # More trials mean lower DSR because of data snooping penalty
    assert dsr_100_trials < dsr_1_trial


def test_run_walk_forward_validation(dummy_index):
    np.random.seed(42)
    returns = pd.Series(np.random.normal(loc=0.0005, scale=0.01, size=1000), index=dummy_index)
    
    results = run_walk_forward_validation(
        returns,
        train_days=300,
        test_days=100,
        step_days=150,
        num_trials=5
    )
    
    assert "windows" in results
    assert "aggregate_oos_returns" in results
    assert "metrics" in results
    
    metrics = results["metrics"]
    assert "psr" in metrics
    assert "dsr" in metrics
    assert "oos_sharpe" in metrics
    
    assert len(results["windows"]) > 0
