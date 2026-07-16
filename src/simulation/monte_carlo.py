from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


TRADING_DAYS = 252.0


def _validate_simulation_inputs(
    current_value: float,
    expected_return: float,
    volatility: float,
    time_horizon: int,
    n_simulations: int,
) -> None:
    numeric_inputs = {
        "current_value": current_value,
        "expected_return": expected_return,
        "volatility": volatility,
    }
    for name, value in numeric_inputs.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if current_value <= 0:
        raise ValueError("current_value must be positive.")
    if volatility < 0:
        raise ValueError("volatility must be non-negative.")
    if not isinstance(time_horizon, (int, np.integer)) or time_horizon <= 0:
        raise ValueError("time_horizon must be a positive integer.")
    if not isinstance(n_simulations, (int, np.integer)) or n_simulations < 2:
        raise ValueError("n_simulations must be an integer of at least 2.")


def _terminal_path_statistics(
    final_values: np.ndarray,
    current_value: float,
    n_simulations: int,
) -> dict:
    mean = float(np.mean(final_values))
    std = float(np.std(final_values, ddof=1))
    standard_error = float(std / np.sqrt(n_simulations))
    percentile_5 = float(np.percentile(final_values, 5))
    tail_5 = final_values[final_values <= percentile_5]
    expected_shortfall_value_95 = float(np.mean(tail_5)) if tail_5.size else percentile_5
    return {
        "mean": mean,
        "median": float(np.median(final_values)),
        "std": std,
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
        "percentile_5": percentile_5,
        "percentile_25": float(np.percentile(final_values, 25)),
        "percentile_75": float(np.percentile(final_values, 75)),
        "percentile_95": float(np.percentile(final_values, 95)),
        "probability_of_loss": float(np.mean(final_values < current_value)),
        "value_at_risk_95_loss": float(current_value - percentile_5),
        "expected_shortfall_95_loss": float(current_value - expected_shortfall_value_95),
        "expected_shortfall_95_value": expected_shortfall_value_95,
        "standard_error_mean": standard_error,
        "relative_standard_error_mean": float(standard_error / abs(mean)) if mean else 0.0,
        "mean_ci_95": (
            float(mean - 1.96 * standard_error),
            float(mean + 1.96 * standard_error),
        ),
    }


def run_monte_carlo_simulation(
    current_value: float,
    expected_return: float,
    volatility: float,
    time_horizon: int = 252,
    n_simulations: int = 1000,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Simulate portfolio values with geometric Brownian motion.

    ``expected_return`` is the annualized arithmetic drift and ``volatility``
    is annualized standard deviation.  The returned diagnostics quantify
    Monte Carlo sampling error; they do not quantify parameter or model risk.

    A local random generator is used so a seeded run is reproducible without
    mutating NumPy's process-wide random state.
    """
    _validate_simulation_inputs(
        current_value=current_value,
        expected_return=expected_return,
        volatility=volatility,
        time_horizon=time_horizon,
        n_simulations=n_simulations,
    )

    dt = 1.0 / TRADING_DAYS
    drift = (float(expected_return) - 0.5 * float(volatility) ** 2) * dt
    diffusion = float(volatility) * np.sqrt(dt)

    rng = np.random.default_rng(random_seed)
    random_shocks = rng.standard_normal((int(time_horizon), int(n_simulations)))
    log_increments = drift + diffusion * random_shocks
    cumulative_log_returns = np.vstack(
        [
            np.zeros((1, int(n_simulations)), dtype=float),
            np.cumsum(log_increments, axis=0),
        ]
    )
    price_paths = float(current_value) * np.exp(cumulative_log_returns)
    final_values = price_paths[-1]

    horizon_years = float(time_horizon) / TRADING_DAYS
    analytic_mean = float(current_value * np.exp(expected_return * horizon_years))

    statistics = {
        **_terminal_path_statistics(final_values, current_value, int(n_simulations)),
        "analytic_mean": analytic_mean,
        "mean_convergence_gap": float((float(np.mean(final_values)) - analytic_mean) / analytic_mean) if analytic_mean else 0.0,
        "model": "geometric_brownian_motion",
        "expected_return_input": float(expected_return),
        "volatility_input": float(volatility),
        "time_horizon": int(time_horizon),
        "n_simulations": int(n_simulations),
        "random_seed": int(random_seed) if random_seed is not None else None,
        "assumptions": [
            "constant annual drift and volatility",
            "independent normally distributed log-return shocks",
            "continuous paths with no jumps, liquidity constraints, or transaction costs",
        ],
    }

    logger.info(f"Ran {n_simulations} simulations over {time_horizon} periods")
    return price_paths, statistics


def calculate_percentile_paths(
    price_paths: np.ndarray,
    percentiles: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Calculate percentile paths from simulation results"""
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    paths = np.asarray(price_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] == 0:
        raise ValueError("price_paths must be a non-empty 2D array.")
    if not np.isfinite(paths).all():
        raise ValueError("price_paths must contain only finite values.")

    percentile_data = {}
    for p in percentiles:
        if not 0 <= float(p) <= 100:
            raise ValueError("percentiles must be between 0 and 100.")
        percentile_data[f"p{p}"] = np.percentile(paths, p, axis=1)

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
    """Simulate portfolio values with Merton jump diffusion.

    A local random generator is used so seeded advanced simulations are
    reproducible without mutating NumPy's process-wide random state.
    """
    _validate_simulation_inputs(
        current_value=current_value,
        expected_return=expected_return,
        volatility=volatility,
        time_horizon=time_horizon,
        n_simulations=n_simulations,
    )
    for name, value in {
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_volatility": jump_volatility,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if jump_intensity < 0:
        raise ValueError("jump_intensity must be non-negative.")
    if jump_volatility < 0:
        raise ValueError("jump_volatility must be non-negative.")
    
    dt = 1.0 / TRADING_DAYS
    rng = np.random.default_rng(random_seed)
    
    jump_compensator = jump_intensity * (np.exp(jump_mean + 0.5 * jump_volatility ** 2) - 1)
    drift = (expected_return - 0.5 * volatility ** 2 - jump_compensator) * dt
    diffusion = volatility * np.sqrt(dt)
    
    z_shocks = rng.standard_normal((int(time_horizon), int(n_simulations)))
    gbm_returns = drift + diffusion * z_shocks
    
    n_jumps = rng.poisson(jump_intensity * dt, (int(time_horizon), int(n_simulations)))
    
    jump_returns = np.zeros_like(gbm_returns)
    mask = n_jumps > 0
    jump_returns[mask] = rng.normal(
        n_jumps[mask] * jump_mean,
        np.sqrt(n_jumps[mask]) * jump_volatility
    )
    
    total_returns = gbm_returns + jump_returns
    
    cumulative_returns = np.cumsum(total_returns, axis=0)
    cumulative_returns = np.vstack([np.zeros(int(n_simulations)), cumulative_returns])
    
    price_paths = float(current_value) * np.exp(cumulative_returns)
    
    final_values = price_paths[-1]
    horizon_years = float(time_horizon) / TRADING_DAYS
    analytic_mean = float(current_value * np.exp(expected_return * horizon_years))
    
    statistics = {
        **_terminal_path_statistics(final_values, float(current_value), int(n_simulations)),
        "analytic_mean": analytic_mean,
        "mean_convergence_gap": float((float(np.mean(final_values)) - analytic_mean) / analytic_mean) if analytic_mean else 0.0,
        "model": "merton_jump_diffusion",
        "expected_return_input": float(expected_return),
        "volatility_input": float(volatility),
        "jump_intensity_input": float(jump_intensity),
        "jump_mean_input": float(jump_mean),
        "jump_volatility_input": float(jump_volatility),
        "realized_average_jumps_per_path": float(np.mean(np.sum(n_jumps, axis=0))),
        "time_horizon": int(time_horizon),
        "n_simulations": int(n_simulations),
        "random_seed": int(random_seed) if random_seed is not None else None,
        "assumptions": [
            "constant annual drift, volatility, and jump parameters",
            "independent normally distributed diffusion shocks",
            "Poisson jump arrivals with normally distributed jump sizes",
            "no liquidity constraints, taxes, transaction costs, or stochastic volatility",
        ],
    }
    
    logger.info(f"Ran {n_simulations} advanced (Merton Jump Diffusion) simulations over {time_horizon} periods")
    
    return price_paths, statistics
