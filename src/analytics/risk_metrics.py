import pandas as pd
import numpy as np
from typing import Optional


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
    annualize: bool = True,
) -> float:
    """Calculate volatility (standard deviation of returns)"""
    vol = returns.std()
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return float(vol)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio (uses downside deviation)"""
    excess_returns = returns - risk_free_rate / periods_per_year
    
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return float(drawdown.min())


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)"""
    from .returns import calculate_annualized_return
    
    ann_return = calculate_annualized_return(returns, periods_per_year)
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return 0.0
    
    return float(ann_return / abs(max_dd))


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Value at Risk (historical method)"""
    return float(-np.percentile(returns, (1 - confidence_level) * 100))


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    var = calculate_var(returns, confidence_level)
    return float(-returns[returns <= -var].mean())
