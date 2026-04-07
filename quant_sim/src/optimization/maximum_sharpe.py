import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict
from loguru import logger


def optimize_maximum_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """Optimize for maximum Sharpe ratio portfolio"""
    n_assets = returns.shape[1]
    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    
    def negative_sharpe(weights):
        portfolio_return = weights @ mean_returns
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_vol == 0:
            return 1e10
        
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return -sharpe
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    if allow_short:
        bounds = [(-1.0, 1.0) for _ in range(n_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    if max_weight is not None:
        bounds = [(0.0, max_weight) for _ in range(n_assets)]
    
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    
    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")
    
    optimal_weights = result.x
    portfolio_return = optimal_weights @ mean_returns
    portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        "weights": optimal_weights,
        "symbols": returns.columns.tolist(),
        "expected_return": portfolio_return,
        "volatility": portfolio_vol,
        "sharpe_ratio": sharpe,
        "success": result.success,
    }
