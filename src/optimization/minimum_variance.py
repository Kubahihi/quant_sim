import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict
from loguru import logger

from ._shared import _clean_returns


def optimize_minimum_variance(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, any]:
    """Optimize for minimum variance portfolio"""
    clean = _clean_returns(returns)
    n_assets = clean.shape[1]
    cov_matrix = clean.cov().values
    
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0
    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        upper_bound = float(max_weight)
        if allow_short:
            lower_bound = -upper_bound

    bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]
    
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
    
    mean_returns = clean.mean()
    portfolio_return = (optimal_weights @ mean_returns) * 252
    
    return {
        "weights": optimal_weights,
        "symbols": clean.columns.tolist(),
        "expected_return": float(portfolio_return),
        "volatility": float(optimal_volatility),
        "sharpe_ratio": float((portfolio_return - risk_free_rate) / optimal_volatility) if optimal_volatility > 0 else 0,
        "success": bool(result.success),
    }
