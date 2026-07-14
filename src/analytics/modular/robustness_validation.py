from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger

from src.analytics.returns import calculate_annualized_return
from src.analytics.risk_metrics import calculate_sharpe_ratio

TRADING_DAYS = 252


def create_walk_forward_splits(
    index: pd.DatetimeIndex, 
    train_days: int, 
    test_days: int, 
    step_days: int
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Splits a DatetimeIndex into rolling walk-forward train and test windows.
    Returns a list of tuples: (train_indices, test_indices).
    """
    splits = []
    if len(index) == 0:
        return splits
        
    start_date = index[0]
    end_date = index[-1]
    
    current_train_start = start_date
    
    while True:
        current_train_end = current_train_start + pd.Timedelta(days=train_days)
        current_test_end = current_train_end + pd.Timedelta(days=test_days)
        
        if current_test_end > end_date:
            break
            
        train_mask = (index >= current_train_start) & (index < current_train_end)
        test_mask = (index >= current_train_end) & (index < current_test_end)
        
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            splits.append((index[train_mask], index[test_mask]))
            
        current_train_start += pd.Timedelta(days=step_days)
        
    return splits


def calculate_psr(
    returns: pd.Series, 
    benchmark_sr: float = 0.0, 
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculates Probabilistic Sharpe Ratio (PSR).
    benchmark_sr is assumed to be in the same frequency as returns (usually daily).
    """
    if len(returns) < 3:
        return 0.0
        
    std_dev = float(returns.std())
    if pd.isna(std_dev) or std_dev == 0:
        return 0.0
        
    sr_daily = (float(returns.mean()) - risk_free_rate) / std_dev
    
    skewness = float(returns.skew())
    kurtosis = float(returns.kurtosis()) # Fisher's kurtosis (excess kurtosis)
    
    n = len(returns)
    
    # Adjust excess kurtosis to Pearson's kurtosis for the formula
    kurt = kurtosis + 3 
    
    denominator = np.sqrt(max(1e-10, 1 - skewness * sr_daily + ((kurt - 1) / 4) * (sr_daily ** 2)))
    psr_stat = ((sr_daily - benchmark_sr) * np.sqrt(n - 1)) / denominator
    
    return float(stats.norm.cdf(psr_stat))


def calculate_dsr(
    returns: pd.Series, 
    num_trials: int, 
    variance_trials: float,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculates Deflated Sharpe Ratio (DSR).
    variance_trials should be the variance of the daily Sharpe Ratios across trials.
    """
    if num_trials < 1:
        return 0.0
    if num_trials == 1:
        return calculate_psr(returns, benchmark_sr=0.0, risk_free_rate=risk_free_rate)
        
    euler_gamma = np.euler_gamma
    
    z1 = float(stats.norm.ppf(1 - 1 / num_trials))
    z2 = float(stats.norm.ppf(1 - 1 / (num_trials * np.e)))
    
    emsr = np.sqrt(max(0.0, variance_trials)) * ((1 - euler_gamma) * z1 + euler_gamma * z2)
    
    return calculate_psr(returns, benchmark_sr=emsr, risk_free_rate=risk_free_rate)


def run_walk_forward_validation(
    portfolio_returns: pd.Series,
    train_days: int = 1095, 
    test_days: int = 180,   
    step_days: int = 90,    
    num_trials: int = 1,
    risk_free_rate: float = 0.0,
) -> Dict[str, object]:
    """
    Runs walk-forward validation and calculates robustness metrics.
    """
    portfolio_returns = portfolio_returns.dropna()
    splits = create_walk_forward_splits(portfolio_returns.index, train_days, test_days, step_days)
    
    windows = []
    oos_returns_list = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        train_ret = portfolio_returns.loc[train_idx]
        test_ret = portfolio_returns.loc[test_idx]
        
        train_sr = calculate_sharpe_ratio(train_ret, risk_free_rate, TRADING_DAYS)
        test_sr = calculate_sharpe_ratio(test_ret, risk_free_rate, TRADING_DAYS)
        train_ann = calculate_annualized_return(train_ret, TRADING_DAYS)
        test_ann = calculate_annualized_return(test_ret, TRADING_DAYS)
        
        windows.append({
            "window_id": i + 1,
            "train_start": train_idx[0].strftime('%Y-%m-%d'),
            "train_end": train_idx[-1].strftime('%Y-%m-%d'),
            "test_start": test_idx[0].strftime('%Y-%m-%d'),
            "test_end": test_idx[-1].strftime('%Y-%m-%d'),
            "train_sharpe": train_sr,
            "test_sharpe": test_sr,
            "train_return": train_ann,
            "test_return": test_ann,
        })
        
        oos_returns_list.append(test_ret)
        
    if not oos_returns_list:
        return {
            "windows": [],
            "aggregate_oos_returns": pd.Series(dtype=float),
            "metrics": {
                "psr": 0.0,
                "dsr": 0.0,
                "oos_sharpe": 0.0,
                "oos_annualized_return": 0.0,
                "psr_interpretation": "Not enough data for walk-forward splits.",
                "dsr_interpretation": "Not enough data for walk-forward splits.",
            }
        }
        
    agg_oos_returns = pd.concat(oos_returns_list)
    agg_oos_returns = agg_oos_returns[~agg_oos_returns.index.duplicated(keep='first')].sort_index()
    
    oos_sharpe = calculate_sharpe_ratio(agg_oos_returns, risk_free_rate, TRADING_DAYS)
    oos_ann_ret = calculate_annualized_return(agg_oos_returns, TRADING_DAYS)
    
    psr = calculate_psr(agg_oos_returns, risk_free_rate=risk_free_rate)
    
    window_srs = [w["test_sharpe"] for w in windows]
    variance_trials = float(np.var(window_srs)) if len(window_srs) > 1 else 0.5
    var_daily_trials = variance_trials / TRADING_DAYS
    
    dsr = calculate_dsr(agg_oos_returns, num_trials, var_daily_trials, risk_free_rate=risk_free_rate)
    
    psr_pct = psr * 100
    if psr > 0.95:
        psr_text = f"Strong confidence ({psr_pct:.1f}%) that the true Sharpe ratio is > 0."
    elif psr > 0.8:
        psr_text = f"Moderate confidence ({psr_pct:.1f}%) that the true Sharpe ratio is > 0."
    else:
        psr_text = f"Low confidence ({psr_pct:.1f}%) that the true Sharpe ratio is > 0. The strategy might be overfitted."
        
    dsr_pct = dsr * 100
    if dsr > 0.95:
        dsr_text = f"Strategy survives data-snooping test with {dsr_pct:.1f}% confidence."
    else:
        dsr_text = f"Potential overfitting detected (DSR {dsr_pct:.1f}%). High risk of false discovery given {num_trials} trials."

    return {
        "windows": windows,
        "aggregate_oos_returns": agg_oos_returns,
        "metrics": {
            "psr": psr,
            "dsr": dsr,
            "oos_sharpe": oos_sharpe,
            "oos_annualized_return": oos_ann_ret,
            "psr_interpretation": psr_text,
            "dsr_interpretation": dsr_text,
        }
    }
