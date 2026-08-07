from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0
DEFAULT_COVARIANCE_SHRINKAGE = 0.25
DEFAULT_RETURN_SHRINKAGE = 0.50


@dataclass(frozen=True)
class PortfolioEstimates:
    """A single, auditable set of inputs shared by every optimizer."""

    returns: pd.DataFrame
    symbols: tuple[str, ...]
    mean_returns: np.ndarray
    covariance: np.ndarray
    sample_mean_returns: np.ndarray
    sample_covariance: np.ndarray
    observations: int
    trading_days: float
    covariance_shrinkage: float
    return_shrinkage: float
    covariance_eigenvalue_floor: float
    expected_return_method: str = "shrunk_historical"
    expected_return_details: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        method = (
            "shrunk_mean_shrunk_covariance"
            if self.expected_return_method == "shrunk_historical"
            else f"{self.expected_return_method}_shrunk_covariance"
        )
        return {
            "method": method,
            "expected_return_method": self.expected_return_method,
            "observations": self.observations,
            "assets": len(self.symbols),
            "trading_days": self.trading_days,
            "covariance_shrinkage": self.covariance_shrinkage,
            "return_shrinkage": self.return_shrinkage,
            "covariance_eigenvalue_floor": self.covariance_eigenvalue_floor,
            "sample_expected_returns": {
                symbol: float(value)
                for symbol, value in zip(
                    self.symbols, self.sample_mean_returns, strict=False
                )
            },
            "shrunk_expected_returns": {
                symbol: float(value)
                for symbol, value in zip(self.symbols, self.mean_returns, strict=False)
            },
            "expected_return_details": self.expected_return_details,
        }


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a finite, numeric, complete-case matrix suitable for estimation."""
    frame = pd.DataFrame(returns).copy()
    if frame.shape[1] < 1:
        raise ValueError("returns are empty after cleaning.")
    if frame.columns.has_duplicates:
        raise ValueError("returns columns must be unique.")

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if frame.empty:
        raise ValueError("returns are empty after cleaning.")
    if frame.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")
    return frame.astype(float)


def _validate_shrinkage(value: float, name: str) -> float:
    shrinkage = float(value)
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return shrinkage


def _shrink_expected_returns(
    sample_mean: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    grand_mean = float(np.mean(sample_mean))
    return (1.0 - shrinkage) * sample_mean + shrinkage * grand_mean


def _shrink_covariance(
    sample_covariance: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    variances = np.clip(np.diag(sample_covariance), a_min=0.0, a_max=None)
    average_variance = float(np.mean(variances)) if variances.size else 0.0
    target = np.eye(sample_covariance.shape[0], dtype=float) * average_variance
    shrunk = (1.0 - shrinkage) * sample_covariance + shrinkage * target
    return (shrunk + shrunk.T) * 0.5


def _repair_covariance(covariance: np.ndarray) -> tuple[np.ndarray, float]:
    """Project numerical covariance noise onto the PSD cone."""
    symmetric = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = scale * 1e-10
    repaired = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return (repaired + repaired.T) * 0.5, floor


def estimate_portfolio_inputs(
    returns: pd.DataFrame,
    *,
    trading_days: float = TRADING_DAYS,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
) -> PortfolioEstimates:
    """Estimate annualized portfolio inputs once using conservative defaults."""
    annualization = float(trading_days)
    if not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("trading_days must be positive.")
    covariance_alpha = _validate_shrinkage(
        covariance_shrinkage, "covariance_shrinkage"
    )
    return_alpha = _validate_shrinkage(return_shrinkage, "return_shrinkage")

    clean = clean_returns(returns)
    sample_mean = clean.mean().to_numpy(dtype=float) * annualization
    sample_covariance = clean.cov().to_numpy(dtype=float) * annualization
    sample_covariance = (sample_covariance + sample_covariance.T) * 0.5

    mean_returns = _shrink_expected_returns(sample_mean, return_alpha)
    covariance = _shrink_covariance(sample_covariance, covariance_alpha)
    covariance, eigenvalue_floor = _repair_covariance(covariance)

    if not np.all(np.isfinite(mean_returns)) or not np.all(np.isfinite(covariance)):
        raise ValueError("portfolio estimates contain non-finite values.")

    return PortfolioEstimates(
        returns=clean,
        symbols=tuple(str(column) for column in clean.columns),
        mean_returns=mean_returns,
        covariance=covariance,
        sample_mean_returns=sample_mean,
        sample_covariance=sample_covariance,
        observations=int(clean.shape[0]),
        trading_days=annualization,
        covariance_shrinkage=covariance_alpha,
        return_shrinkage=return_alpha,
        covariance_eigenvalue_floor=float(eigenvalue_floor),
    )


def resolve_portfolio_estimates(
    returns: pd.DataFrame,
    *,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
) -> PortfolioEstimates:
    """Reuse precomputed inputs only when they exactly match the asset order."""
    estimates = portfolio_estimates or estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    input_symbols = tuple(str(column) for column in pd.DataFrame(returns).columns)
    if estimates.symbols != input_symbols:
        raise ValueError("portfolio_estimates symbols must match return columns in order.")
    return estimates


def _aligned_weights(
    values: Sequence[float] | np.ndarray | Mapping[str, float],
    symbols: Sequence[str],
    name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = [symbol for symbol in symbols if symbol not in values]
        if missing:
            raise ValueError(f"{name} is missing values for: {', '.join(missing)}.")
        vector = np.asarray([values[symbol] for symbol in symbols], dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size != len(symbols):
        raise ValueError(f"{name} length must match assets.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite.")
    return vector


def estimate_black_litterman_inputs(
    returns: pd.DataFrame,
    *,
    market_weights: Sequence[float] | np.ndarray | Mapping[str, float],
    views: Optional[Mapping[str, float]] = None,
    view_confidences: Optional[Mapping[str, float]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
) -> PortfolioEstimates:
    """Return shared covariance with Black-Litterman posterior expected returns."""
    estimates = estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
    )
    symbols = list(estimates.symbols)
    weights = _aligned_weights(market_weights, symbols, "market_weights")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("market_weights must be non-negative with a positive sum.")
    weights = weights / float(weights.sum())
    delta = float(risk_aversion)
    uncertainty_scale = float(tau)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("risk_aversion must be positive.")
    if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0:
        raise ValueError("tau must be positive.")

    covariance = estimates.covariance
    equilibrium = delta * covariance @ weights
    supplied_views = dict(views or {})
    unknown = [symbol for symbol in supplied_views if symbol not in symbols]
    if unknown:
        raise ValueError(f"views contain unknown assets: {', '.join(unknown)}.")
    confidence_map = dict(view_confidences or {})

    if supplied_views:
        view_symbols = [symbol for symbol in symbols if symbol in supplied_views]
        pick = np.zeros((len(view_symbols), len(symbols)), dtype=float)
        view_returns = np.zeros(len(view_symbols), dtype=float)
        omega_diagonal = np.zeros(len(view_symbols), dtype=float)
        resolved_confidences: dict[str, float] = {}
        scaled_covariance = uncertainty_scale * covariance
        for row, symbol in enumerate(view_symbols):
            index = symbols.index(symbol)
            pick[row, index] = 1.0
            view_returns[row] = float(supplied_views[symbol])
            confidence = float(confidence_map.get(symbol, 0.50))
            if not np.isfinite(confidence) or not 0.0 < confidence <= 1.0:
                raise ValueError(f"view confidence for {symbol} must be in (0, 1].")
            resolved_confidences[symbol] = confidence
            base_uncertainty = float(pick[row] @ scaled_covariance @ pick[row])
            omega_diagonal[row] = max(
                base_uncertainty * (1.0 - confidence) / confidence,
                1e-12,
            )
        inverse_scaled_covariance = np.linalg.pinv(scaled_covariance)
        inverse_omega = np.diag(1.0 / omega_diagonal)
        posterior_covariance = np.linalg.pinv(
            inverse_scaled_covariance + pick.T @ inverse_omega @ pick
        )
        posterior = posterior_covariance @ (
            inverse_scaled_covariance @ equilibrium
            + pick.T @ inverse_omega @ view_returns
        )
    else:
        view_symbols = []
        resolved_confidences = {}
        posterior = equilibrium

    details = {
        "risk_aversion": delta,
        "tau": uncertainty_scale,
        "market_weights": {
            symbol: float(value)
            for symbol, value in zip(symbols, weights, strict=False)
        },
        "equilibrium_returns": {
            symbol: float(value)
            for symbol, value in zip(symbols, equilibrium, strict=False)
        },
        "views": {symbol: float(supplied_views[symbol]) for symbol in view_symbols},
        "view_confidences": resolved_confidences,
    }
    return replace(
        estimates,
        mean_returns=np.asarray(posterior, dtype=float),
        expected_return_method="black_litterman",
        expected_return_details=details,
    )
