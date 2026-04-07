import pandas as pd
import numpy as np
from typing import Dict


def calculate_contribution_to_risk(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
) -> np.ndarray:
    """Calculate contribution to risk for each asset"""
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_vol = np.sqrt(portfolio_variance)
    
    if portfolio_vol == 0:
        return np.zeros_like(weights)
    
    marginal_contrib = cov_matrix @ weights
    contrib_to_risk = weights * marginal_contrib / portfolio_vol
    
    return contrib_to_risk


def calculate_contribution_to_return(
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> np.ndarray:
    """Calculate contribution to return for each asset"""
    mean_returns = returns.mean()
    portfolio_return = weights @ mean_returns
    
    if portfolio_return == 0:
        return np.zeros_like(weights)
    
    contrib = weights * mean_returns / portfolio_return
    return contrib


def generate_contribution_report(
    symbols: list[str],
    weights: np.ndarray,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Generate comprehensive contribution analysis report"""
    cov_matrix = returns.cov()
    
    contrib_risk = calculate_contribution_to_risk(weights, cov_matrix)
    contrib_return = calculate_contribution_to_return(weights, returns)
    
    report = pd.DataFrame({
        "symbol": symbols,
        "weight": weights,
        "contribution_to_risk": contrib_risk,
        "contribution_to_return": contrib_return,
    })
    
    return report
