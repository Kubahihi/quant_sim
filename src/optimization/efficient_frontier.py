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
    """Calculate efficient frontier points.

    The frontier consists of a sequence of closely related quadratic
    programs.  Supplying the exact objective/constraint gradients and using
    each solution to initialize the next target avoids the repeated finite-
    difference work of solving every point from equal weights.
    """
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2.")

    clean_returns = (
        pd.DataFrame(returns)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    if clean_returns.empty or clean_returns.shape[1] == 0:
        raise ValueError(
            "returns must contain finite observations for at least one asset."
        )
    if clean_returns.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")

    clean_returns = clean_returns.astype(float)
    n_assets = clean_returns.shape[1]
    mean_returns = clean_returns.mean().to_numpy(dtype=float) * TRADING_DAYS
    cov_matrix = clean_returns.cov().to_numpy(dtype=float) * TRADING_DAYS
    # Numerical noise can make a sample covariance microscopically
    # asymmetric.  A symmetric matrix keeps the analytic gradient exact.
    cov_matrix = (cov_matrix + cov_matrix.T) * 0.5
    symbols = clean_returns.columns.tolist()
    
    def portfolio_metrics(weights):
        ret = float(weights @ mean_returns)
        variance = float(weights @ cov_matrix @ weights)
        vol = float(np.sqrt(max(variance, 0.0)))
        return ret, vol

    if n_assets == 1:
        weights = np.ones(1, dtype=float)
        ret, vol = portfolio_metrics(weights)
        metrics = calculate_portfolio_statistics(
            weights=weights,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=0.0,
            symbols=symbols,
        )
        logger.info("Calculated the single feasible portfolio for one asset")
        return [{
            "weights": weights,
            "return": ret,
            "volatility": vol,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "diversification_score": metrics["diversification_score"],
            "effective_holdings": metrics["effective_holdings"],
            "max_weight": metrics["max_weight"],
            "top_holdings": metrics["top_holdings"],
        }]
    
    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ cov_matrix @ weights)

    def portfolio_variance_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_matrix @ weights)

    ones = np.ones(n_assets, dtype=float)
    fully_invested_constraint = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda _weights: ones,
    }
    
    if allow_short:
        bounds = [(-1.0, 1.0) for _ in range(n_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier_points = []
    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    
    for target_return in target_returns:
        target_constraint = {
            "type": "eq",
            "fun": lambda weights, target=target_return: float(
                weights @ mean_returns - target
            ),
            "jac": lambda _weights: mean_returns,
        }
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            jac=portfolio_variance_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=(fully_invested_constraint, target_constraint),
            options={"maxiter": 1000, "disp": False},
        )
        
        if result.success:
            weights = np.asarray(result.x, dtype=float)
            # Adjacent target-return problems have adjacent solutions.
            initial_weights = weights
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
