from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


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

    trading_days = 252.0
    dt = 1.0 / trading_days
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

    mean = float(np.mean(final_values))
    std = float(np.std(final_values, ddof=1))
    standard_error = float(std / np.sqrt(n_simulations))
    mean_ci_95 = (
        float(mean - 1.96 * standard_error),
        float(mean + 1.96 * standard_error),
    )
    percentile_5 = float(np.percentile(final_values, 5))
    tail_5 = final_values[final_values <= percentile_5]
    expected_shortfall_value_95 = float(np.mean(tail_5)) if tail_5.size else percentile_5
    horizon_years = float(time_horizon) / trading_days
    analytic_mean = float(current_value * np.exp(expected_return * horizon_years))

    statistics = {
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
        "mean_ci_95": mean_ci_95,
        "analytic_mean": analytic_mean,
        "mean_convergence_gap": float((mean - analytic_mean) / analytic_mean) if analytic_mean else 0.0,
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
