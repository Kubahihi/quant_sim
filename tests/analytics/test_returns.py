from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.returns import (
    calculate_annualized_return,
    calculate_cumulative_returns,
)


def test_annualized_return_compounds_simple_returns_in_log_space():
    returns = pd.Series([0.10, -0.05, 0.02])
    expected = np.expm1(np.log1p(returns.to_numpy()).sum() * 252 / 3)

    assert calculate_annualized_return(returns) == pytest.approx(expected)


@pytest.mark.parametrize("bad_return", [np.nan, np.inf, -1.0, -1.1])
def test_return_aggregation_rejects_invalid_simple_returns(bad_return):
    returns = pd.Series([0.01, bad_return])

    with pytest.raises(ValueError, match="finite and greater"):
        calculate_annualized_return(returns)
    with pytest.raises(ValueError, match="finite and greater"):
        calculate_cumulative_returns(returns)


def test_annualized_return_requires_positive_integer_frequency():
    with pytest.raises(ValueError, match="positive integer"):
        calculate_annualized_return(pd.Series([0.01]), periods_per_year=0)
