import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict
from loguru import logger


def optimize_minimum_variance(
    returns: pd.DataFrame,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """Optimize for minimum variance portfolio"""
    n_assets = returns.shape[1]
    cov_matrix = returns.cov().values
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    if allow_short:
        bounds = [(-1.0, 1.0) for _ in range(n_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    if max_weight is not None:
        bounds = [(0.0, max_weight) for _ in range(n_assets)]
    
    initial_weights = np.array([1.0 / n_assets] * n_assets)
    
    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    
    if not result.success:
        logger.warning(f"Optimization failed: {result.message}")
    
    optimal_weights = result.x
    optimal_variance = result.fun
    optimal_volatility = np.sqrt(optimal_variance) * np.sqrt(252)
    
    mean_returns = returns.mean()
    portfolio_return = (optimal_weights @ mean_returns) * 252
    
    return {
        "weights": optimal_weights,
        "symbols": returns.columns.tolist(),
        "expected_return": portfolio_return,
        "volatility": optimal_volatility,
        "sharpe_ratio": portfolio_return / optimal_volatility if optimal_volatility > 0 else 0,
        "success": result.success,
    }
