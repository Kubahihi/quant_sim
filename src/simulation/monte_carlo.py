import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger


def run_monte_carlo_simulation(
    current_value: float,
    expected_return: float,
    volatility: float,
    time_horizon: int = 252,
    n_simulations: int = 1000,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Run Monte Carlo simulation using Geometric Brownian Motion"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    dt = 1 / 252
    
    drift = (expected_return - 0.5 * volatility ** 2) * dt
    diffusion = volatility * np.sqrt(dt)
    
    random_shocks = np.random.normal(0, 1, (time_horizon, n_simulations))
    
    returns = drift + diffusion * random_shocks
    
    price_paths = np.zeros((time_horizon + 1, n_simulations))
    price_paths[0] = current_value
    
    for t in range(1, time_horizon + 1):
        price_paths[t] = price_paths[t - 1] * np.exp(returns[t - 1])
    
    final_values = price_paths[-1]
    
    statistics = {
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "std": float(np.std(final_values)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "percentile_5": float(np.percentile(final_values, 5)),
        "percentile_25": float(np.percentile(final_values, 25)),
        "percentile_75": float(np.percentile(final_values, 75)),
        "percentile_95": float(np.percentile(final_values, 95)),
    }
    
    logger.info(f"Ran {n_simulations} simulations over {time_horizon} periods")
    
    return price_paths, statistics


def calculate_percentile_paths(
    price_paths: np.ndarray,
    percentiles: list[float] = [5, 25, 50, 75, 95],
) -> pd.DataFrame:
    """Calculate percentile paths from simulation results"""
    percentile_data = {}
    
    for p in percentiles:
        percentile_data[f"p{p}"] = np.percentile(price_paths, p, axis=1)
    
    return pd.DataFrame(percentile_data)


def run_advanced_monte_carlo_simulation(
    current_value: float,
    expected_return: float,
    volatility: float,
    time_horizon: int = 252,
    n_simulations: int = 1000,
    jump_intensity: float = 1.5,
    jump_mean: float = -0.05,
    jump_volatility: float = 0.08,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Run Advanced Monte Carlo simulation using Merton Jump Diffusion.
    Fully vectorized for maximum performance.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    dt = 1 / 252
    
    jump_compensator = jump_intensity * (np.exp(jump_mean + 0.5 * jump_volatility ** 2) - 1)
    drift = (expected_return - 0.5 * volatility ** 2 - jump_compensator) * dt
    diffusion = volatility * np.sqrt(dt)
    
    z_shocks = np.random.normal(0, 1, (time_horizon, n_simulations))
    gbm_returns = drift + diffusion * z_shocks
    
    n_jumps = np.random.poisson(jump_intensity * dt, (time_horizon, n_simulations))
    
    jump_returns = np.zeros_like(gbm_returns)
    mask = n_jumps > 0
    jump_returns[mask] = np.random.normal(
        n_jumps[mask] * jump_mean,
        np.sqrt(n_jumps[mask]) * jump_volatility
    )
    
    total_returns = gbm_returns + jump_returns
    
    cumulative_returns = np.cumsum(total_returns, axis=0)
    cumulative_returns = np.vstack([np.zeros(n_simulations), cumulative_returns])
    
    price_paths = current_value * np.exp(cumulative_returns)
    
    final_values = price_paths[-1]
    
    statistics = {
        "mean": float(np.mean(final_values)),
        "median": float(np.median(final_values)),
        "std": float(np.std(final_values)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "percentile_5": float(np.percentile(final_values, 5)),
        "percentile_25": float(np.percentile(final_values, 25)),
        "percentile_75": float(np.percentile(final_values, 75)),
        "percentile_95": float(np.percentile(final_values, 95)),
    }
    
    logger.info(f"Ran {n_simulations} advanced (Merton Jump Diffusion) simulations over {time_horizon} periods")
    
    return price_paths, statistics
