from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional


TRADING_DAYS = 252.0


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame(returns).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        raise ValueError("returns are empty after cleaning.")
    if clean.shape[1] < 1:
        raise ValueError("returns must contain at least one asset.")
    if clean.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")
    return clean.astype(float)


def _shrink_expected_returns(sample_mean: np.ndarray, shrinkage: float) -> np.ndarray:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    grand_mean = float(np.mean(sample_mean))
    return (1.0 - shrinkage) * sample_mean + shrinkage * grand_mean


def _shrink_covariance(sample_cov: np.ndarray, shrinkage: float) -> np.ndarray:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    variances = np.clip(np.diag(sample_cov), a_min=0.0, a_max=None)
    avg_variance = float(np.mean(variances)) if variances.size else 0.0
    target = np.eye(sample_cov.shape[0], dtype=float) * avg_variance
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * target
    return (shrunk + shrunk.T) / 2.0


def optimize_maximum_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    covariance_shrinkage: float = 0.25,
    return_shrinkage: float = 0.50,
) -> Dict[str, any]:
    """Optimize for maximum Sharpe ratio using conservative input estimates.

    Direct sample mean/covariance estimates are very noisy for portfolio
    construction.  The optimizer therefore uses simple shrinkage by default:
    expected returns are pulled toward the cross-sectional mean and covariance
    is pulled toward a diagonal matrix with average variance.
    """
    clean = _clean_returns(returns)
    n_assets = clean.shape[1]
    sample_mean_returns = clean.mean().values * TRADING_DAYS
    sample_cov_matrix = clean.cov().values * TRADING_DAYS
    mean_returns = _shrink_expected_returns(sample_mean_returns, return_shrinkage)
    cov_matrix = _shrink_covariance(sample_cov_matrix, covariance_shrinkage)
    
    def negative_sharpe(weights):
        portfolio_return = weights @ mean_returns
        portfolio_variance = float(weights.T @ cov_matrix @ weights)
        portfolio_vol = np.sqrt(max(portfolio_variance, 0.0))
        
        if portfolio_vol == 0:
            return 1e10
        
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return -sharpe
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0
    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")
        upper_bound = float(max_weight)
        if not allow_short and upper_bound * n_assets < 1.0:
            relaxed = 1.0 / n_assets
            logger.warning(
                f"max_weight={max_weight} is infeasible for {n_assets} assets; relaxing to {relaxed:.6f}"
            )
            upper_bound = relaxed
        if allow_short:
            lower_bound = -upper_bound

    bounds = [(lower_bound, upper_bound) for _ in range(n_assets)]
    
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
    
    optimal_weights = np.asarray(result.x, dtype=float)
    portfolio_return = optimal_weights @ mean_returns
    portfolio_vol = np.sqrt(max(float(optimal_weights.T @ cov_matrix @ optimal_weights), 0.0))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        "weights": optimal_weights,
        "symbols": clean.columns.tolist(),
        "expected_return": float(portfolio_return),
        "volatility": float(portfolio_vol),
        "sharpe_ratio": float(sharpe),
        "success": bool(result.success),
        "message": str(result.message),
        "estimation": {
            "method": "shrunk_mean_shrunk_covariance",
            "observations": int(clean.shape[0]),
            "covariance_shrinkage": float(np.clip(covariance_shrinkage, 0.0, 1.0)),
            "return_shrinkage": float(np.clip(return_shrinkage, 0.0, 1.0)),
            "sample_expected_returns": {
                symbol: float(value)
                for symbol, value in zip(clean.columns.tolist(), sample_mean_returns, strict=False)
            },
            "shrunk_expected_returns": {
                symbol: float(value)
                for symbol, value in zip(clean.columns.tolist(), mean_returns, strict=False)
            },
        },
    }
