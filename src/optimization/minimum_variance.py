from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Any, Dict, Optional
from loguru import logger

from ._shared import _clean_returns


TRADING_DAYS = 252.0

# Volatility values below this threshold are treated as zero for Sharpe
# computation.  Without this guard a singular covariance matrix (e.g. from
# constant returns) produces floating-point residual variance ~1e-30 which,
# after annualisation, yields a Sharpe ratio of order 1e+15.
_VOLATILITY_EPS: float = 1e-8


def optimize_minimum_variance(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.03,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Optimize for the minimum variance portfolio.

    Parameters
    ----------
    returns:
        Daily returns DataFrame.  Rows with NaN/inf are dropped before
        optimisation.
    risk_free_rate:
        Annualised risk-free rate used for the Sharpe ratio only.
    allow_short:
        If True, weights may be negative down to ``-max_weight``
        (or ``-1.0`` if *max_weight* is None).
    max_weight:
        Per-asset weight upper bound (must be > 0).  When ``allow_short=True``
        the lower bound mirrors this as ``-max_weight``.

    Returns
    -------
    dict with keys:
        ``weights``, ``symbols``, ``expected_return``, ``volatility``,
        ``sharpe_ratio``, ``success``, ``message``.

    Raises
    ------
    ValueError
        On invalid inputs (detected before optimisation).
    """
    clean = _clean_returns(returns)
    n_assets = clean.shape[1]

    # Annualise the covariance matrix consistently with maximum_sharpe.py.
    cov_matrix = clean.cov().to_numpy(dtype=float) * TRADING_DAYS
    # Numerical noise can make a sample covariance microscopically asymmetric.
    cov_matrix = (cov_matrix + cov_matrix.T) * 0.5

    # Validate covariance matrix: all diagonal entries must be non-negative.
    if np.any(np.diag(cov_matrix) < 0):
        raise ValueError(
            "Covariance matrix has negative diagonal entries — returns data is invalid."
        )

    # ---------- bound construction ----------
    lower_bound = -1.0 if allow_short else 0.0
    upper_bound = 1.0

    if max_weight is not None:
        if max_weight <= 0:
            raise ValueError(f"max_weight must be positive, got {max_weight!r}.")
        upper_bound = float(max_weight)
        if allow_short:
            lower_bound = -upper_bound

    # Feasibility check: the sum-to-one constraint requires
    # n_assets * upper_bound >= 1.0 for long-only portfolios.
    if not allow_short and upper_bound * n_assets < 1.0:
        relaxed = 1.0 / n_assets
        logger.warning(
            f"max_weight={max_weight} is infeasible for {n_assets} assets "
            f"(n_assets * max_weight = {upper_bound * n_assets:.6f} < 1.0); "
            f"relaxing upper bound to {relaxed:.6f}"
        )
        upper_bound = relaxed

    bounds = [(lower_bound, upper_bound)] * n_assets

    # ---------- objective + gradient ----------
    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ cov_matrix @ weights)

    def portfolio_variance_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_matrix @ weights)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda _w: np.ones(n_assets, dtype=float),
        }
    ]

    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)

    result = minimize(
        portfolio_variance,
        initial_weights,
        jac=portfolio_variance_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "disp": False},
    )

    if not result.success:
        logger.warning(f"Minimum-variance optimisation did not converge: {result.message}")

    optimal_weights = np.asarray(result.x, dtype=float)

    # Guard against tiny negative variance from floating-point noise.
    optimal_variance = float(result.fun)
    optimal_volatility = float(np.sqrt(max(optimal_variance, 0.0)))

    mean_returns_annualised = clean.mean().to_numpy(dtype=float) * TRADING_DAYS
    portfolio_return = float(optimal_weights @ mean_returns_annualised)

    sharpe = (
        (portfolio_return - risk_free_rate) / optimal_volatility
        if optimal_volatility > _VOLATILITY_EPS
        else 0.0
    )

    return {
        "weights":          optimal_weights,
        "symbols":          clean.columns.tolist(),
        "expected_return":  portfolio_return,
        "volatility":       optimal_volatility,
        "sharpe_ratio":     sharpe,
        "success":          bool(result.success),
        "message":          str(result.message),
    }
