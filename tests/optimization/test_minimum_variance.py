import numpy as np
import pandas as pd
import pytest

from src.optimization.minimum_variance import optimize_minimum_variance

@pytest.fixture
def sample_returns():
    np.random.seed(42)
    data = np.random.normal(0.001, 0.02, (100, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])

def test_sharpe_ratio(sample_returns):
    result = optimize_minimum_variance(sample_returns, risk_free_rate=0.03)
    assert result["success"]
    
    weights = result["weights"]
    returns_mean = sample_returns.mean()
    expected_ret = (weights @ returns_mean) * 252
    
    cov_matrix = sample_returns.cov().values
    variance = weights.T @ cov_matrix @ weights
    volatility = np.sqrt(variance) * np.sqrt(252)
    
    expected_sharpe = (expected_ret - 0.03) / volatility
    assert np.isclose(result["sharpe_ratio"], expected_sharpe)

def test_allow_short_with_max_weight(sample_returns):
    result = optimize_minimum_variance(sample_returns, allow_short=True, max_weight=0.5)
    assert result["success"]
    
    weights = result["weights"]
    # It should allow negative weights down to -0.5 and positive up to 0.5
    assert np.all(weights >= -0.5 - 1e-5)
    assert np.all(weights <= 0.5 + 1e-5)
    
def test_clean_returns_validation():
    # Test NaN removal
    data = pd.DataFrame({"A": [0.01, 0.02, np.nan, 0.04], "B": [0.02, -0.01, 0.03, 0.01]})
    result = optimize_minimum_variance(data)
    assert result["success"]
    assert len(result["symbols"]) == 2
    
    # Test insufficient observations
    bad_data = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(ValueError, match="returns must contain at least two observations"):
        optimize_minimum_variance(bad_data)

def test_zero_volatility_edge_case():
    # Create returns with 0 volatility (constant returns)
    data = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [0.01, 0.01, 0.01, 0.01]})
    result = optimize_minimum_variance(data, risk_free_rate=0.0)
    assert result["success"]
    assert np.isclose(result["volatility"], 0.0)
    assert result["sharpe_ratio"] == 0
