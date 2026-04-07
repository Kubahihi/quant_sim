import pandas as pd
import numpy as np
from typing import Optional


def calculate_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate correlation matrix"""
    return returns.corr(method=method)


def calculate_covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate covariance matrix"""
    return returns.cov()


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """Calculate beta vs market"""
    covariance = asset_returns.cov(market_returns)
    market_variance = market_returns.var()
    
    if market_variance == 0:
        return 0.0
    
    return float(covariance / market_variance)


def calculate_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """Calculate Jensen's alpha"""
    from .returns import calculate_annualized_return
    
    beta = calculate_beta(asset_returns, market_returns)
    
    asset_ann_return = calculate_annualized_return(asset_returns, periods_per_year)
    market_ann_return = calculate_annualized_return(market_returns, periods_per_year)
    
    expected_return = risk_free_rate + beta * (market_ann_return - risk_free_rate)
    alpha = asset_ann_return - expected_return
    
    return float(alpha)
