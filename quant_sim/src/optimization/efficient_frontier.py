import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from loguru import logger

TRADING_DAYS = 252


def _calculate_diversification_metrics(weights: np.ndarray) -> tuple[float, float]:
    """Return normalized diversification score and effective holdings."""
    concentration = float(np.sum(np.square(weights)))
    if concentration <= 0:
        return 0.0, 0.0

    effective_holdings = 1.0 / concentration
    diversification_score = effective_holdings / len(weights)
    return diversification_score, effective_holdings


def _format_top_holdings(
    weights: np.ndarray,
    symbols: List[str],
    top_n: int = 3,
) -> str:
    """Create a compact hover-friendly summary of the largest positions."""
    ranked_idx = np.argsort(weights)[::-1][:top_n]
    return ", ".join(
        f"{symbols[idx]} {weights[idx]:.0%}"
        for idx in ranked_idx
        if weights[idx] > 0
    )


def calculate_portfolio_statistics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.03,
    symbols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Calculate portfolio metrics used across optimizers and visualizations."""
    portfolio_return = float(weights @ mean_returns)
    portfolio_volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe_ratio = (
        (portfolio_return - risk_free_rate) / portfolio_volatility
        if portfolio_volatility > 0
        else 0.0
    )
    diversification_score, effective_holdings = _calculate_diversification_metrics(weights)

    metrics = {
        "return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "diversification_score": diversification_score,
        "effective_holdings": effective_holdings,
        "max_weight": float(np.max(weights)),
    }

    if symbols is not None:
        metrics["top_holdings"] = _format_top_holdings(weights, symbols)

    return metrics


def sample_portfolio_cloud(
    returns: pd.DataFrame,
    n_samples: int = 2500,
    risk_free_rate: float = 0.03,
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Sample a large set of long-only portfolios for 3D visualization."""
    n_assets = returns.shape[1]
    mean_returns = returns.mean().values * TRADING_DAYS
    cov_matrix = returns.cov().values * TRADING_DAYS
    rng = np.random.default_rng(random_seed)

    weights = rng.dirichlet(np.ones(n_assets), size=n_samples)
    portfolio_returns = weights @ mean_returns
    portfolio_variance = np.einsum("ij,jk,ik->i", weights, cov_matrix, weights)
    portfolio_volatility = np.sqrt(np.clip(portfolio_variance, a_min=0.0, a_max=None))
    sharpe_ratio = np.divide(
        portfolio_returns - risk_free_rate,
        portfolio_volatility,
        out=np.zeros_like(portfolio_returns),
        where=portfolio_volatility > 0,
    )

    concentration = np.sum(np.square(weights), axis=1)
    effective_holdings = np.divide(
        1.0,
        concentration,
        out=np.zeros_like(concentration),
        where=concentration > 0,
    )
    diversification_score = effective_holdings / n_assets
    max_weight = np.max(weights, axis=1)
    symbols = returns.columns.tolist()
    top_holdings = [
        _format_top_holdings(sample_weights, symbols)
        for sample_weights in weights
    ]

    cloud = pd.DataFrame({
        "expected_return": portfolio_returns,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "diversification_score": diversification_score,
        "effective_holdings": effective_holdings,
        "max_weight": max_weight,
        "top_holdings": top_holdings,
    })

    logger.info(f"Sampled {len(cloud)} portfolios for 3D trade-off visualization")
    return cloud


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    allow_short: bool = False,
) -> List[Dict]:
    """Calculate efficient frontier points"""
    n_assets = returns.shape[1]
    mean_returns = returns.mean().values * TRADING_DAYS
    cov_matrix = returns.cov().values * TRADING_DAYS
    symbols = returns.columns.tolist()
    
    def portfolio_metrics(weights):
        ret = weights @ mean_returns
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return ret, vol
    
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    if allow_short:
        bounds = [(-1.0, 1.0) for _ in range(n_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier_points = []
    
    for target_return in target_returns:
        target_constraints = constraints + [
            {"type": "eq", "fun": lambda w: w @ mean_returns - target_return}
        ]
        
        result = minimize(
            portfolio_volatility,
            np.array([1.0 / n_assets] * n_assets),
            method="SLSQP",
            bounds=bounds,
            constraints=target_constraints,
            options={"maxiter": 1000, "disp": False},
        )
        
        if result.success:
            weights = result.x
            ret, vol = portfolio_metrics(weights)
            metrics = calculate_portfolio_statistics(
                weights=weights,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                risk_free_rate=0.0,
                symbols=symbols,
            )
            
            frontier_points.append({
                "weights": weights,
                "return": ret,
                "volatility": vol,
                "sharpe_ratio": metrics["sharpe_ratio"],
                "diversification_score": metrics["diversification_score"],
                "effective_holdings": metrics["effective_holdings"],
                "max_weight": metrics["max_weight"],
                "top_holdings": metrics["top_holdings"],
            })
    
    logger.info(f"Calculated {len(frontier_points)} efficient frontier points")
    return frontier_points
