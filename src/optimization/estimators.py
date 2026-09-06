from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from src.utils.rates import annual_effective_to_arithmetic


TRADING_DAYS = 252.0
DEFAULT_COVARIANCE_SHRINKAGE = 0.25
DEFAULT_RETURN_SHRINKAGE = 0.50

# These names are an explicit modelling convention in QuantSim, not an
# inference from a flat price history.  A different constant-return asset must
# be opted in through ``deterministic_assets``; otherwise the estimator fails
# closed instead of turning a stale risky series into a zero-risk investment.
_DECLARED_CASH_SYMBOLS = {
    "CASH",
    "USD",
    "EUR",
    "GBP",
    "CZK",
    "JPY",
    "CHF",
    "RISK FREE",
    "RISK-FREE",
    "RISK_FREE",
}


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
    deterministic_assets: tuple[str, ...] = ()

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
            "return_convention": "annualized_arithmetic_simple_return",
            "covariance_shrinkage": self.covariance_shrinkage,
            "return_shrinkage": self.return_shrinkage,
            "covariance_eigenvalue_floor": self.covariance_eigenvalue_floor,
            "deterministic_assets": list(self.deterministic_assets),
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

    real_numeric = all(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
        for dtype in frame.dtypes
    )
    if real_numeric:
        values = frame.to_numpy(dtype=float, na_value=np.nan)
        frame = frame.iloc[np.isfinite(values).all(axis=1)]
    else:
        # Preserve coercion semantics for object, string, categorical, and
        # other mixed inputs that may contain numeric text.
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if frame.empty:
        raise ValueError("returns are empty after cleaning.")
    if frame.shape[0] < 2:
        raise ValueError("returns must contain at least two observations.")
    frame = frame.astype(float)
    if bool((frame.to_numpy(dtype=float) < -1.0).any()):
        raise ValueError("Simple returns must be at least -100%.")
    return frame


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


def _constant_column_mask(returns: pd.DataFrame) -> np.ndarray:
    """Identify columns with no economically or numerically observable variation."""
    values = returns.to_numpy(dtype=float)
    reference = values[[0], :]
    scale = np.maximum(np.max(np.abs(values), axis=0), 1.0)
    tolerance = np.finfo(float).eps * scale * 16.0
    return np.max(np.abs(values - reference), axis=0) <= tolerance


def _deterministic_column_mask(
    returns: pd.DataFrame,
    deterministic_assets: Optional[Sequence[str]],
) -> np.ndarray:
    """Resolve explicitly declared, actually constant cash/risk-free columns.

    Constancy alone is not evidence that an asset is risk-free: a suspended,
    stale, or sparsely priced risky security can also have a flat return
    history.  Project-standard cash labels count as an explicit declaration;
    callers can opt in another label with ``deterministic_assets``.
    """
    symbols = tuple(str(column) for column in returns.columns)
    symbol_lookup = {symbol.casefold(): symbol for symbol in symbols}
    requested = {
        str(symbol).strip().casefold()
        for symbol in (deterministic_assets or ())
        if str(symbol).strip()
    }
    unknown = sorted(
        symbol for symbol in requested if symbol not in symbol_lookup
    )
    if unknown:
        raise ValueError(
            "deterministic_assets contains unknown return columns: "
            + ", ".join(unknown)
            + "."
        )

    conventionally_declared = {
        symbol.casefold()
        for symbol in symbols
        if symbol.strip().upper() in _DECLARED_CASH_SYMBOLS
    }
    declared = requested | conventionally_declared
    constant = _constant_column_mask(returns)
    requested_mask = np.asarray(
        [symbol.casefold() in requested for symbol in symbols],
        dtype=bool,
    )
    declared_mask = np.asarray(
        [symbol.casefold() in declared for symbol in symbols],
        dtype=bool,
    )

    declared_but_variable = [
        symbols[index]
        for index in np.flatnonzero(requested_mask & ~constant)
    ]
    if declared_but_variable:
        raise ValueError(
            "Explicit deterministic_assets must have a constant periodic return: "
            + ", ".join(declared_but_variable)
            + "."
        )

    unexplained_constant = [
        symbols[index]
        for index in np.flatnonzero(constant & ~declared_mask)
    ]
    if unexplained_constant:
        raise ValueError(
            "Constant return columns are not assumed risk-free. Identify verified "
            "cash/risk-free assets with deterministic_assets or repair stale data: "
            + ", ".join(unexplained_constant)
            + "."
        )
    return constant & declared_mask


def estimate_portfolio_inputs(
    returns: pd.DataFrame,
    *,
    trading_days: float = TRADING_DAYS,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    deterministic_assets: Optional[Sequence[str]] = None,
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
    deterministic = _deterministic_column_mask(clean, deterministic_assets)
    sample_mean = clean.mean().to_numpy(dtype=float) * annualization
    # Cash and risky assets must use the same arithmetic annualization. The
    # effective annual risk-free input is converted separately before Sharpe.
    sample_covariance = clean.cov().to_numpy(dtype=float) * annualization
    sample_covariance = (sample_covariance + sample_covariance.T) * 0.5

    # A synthetic cash/risk-free column is often represented by an exactly
    # constant daily return. Shrinking it toward risky assets invents both
    # return and volatility. Preserve deterministic columns and estimate only
    # the stochastic block.
    stochastic = ~deterministic
    sample_covariance[deterministic, :] = 0.0
    sample_covariance[:, deterministic] = 0.0

    mean_returns = sample_mean.copy()
    covariance = np.zeros_like(sample_covariance)
    eigenvalue_floor = 0.0
    if np.any(stochastic):
        stochastic_indices = np.flatnonzero(stochastic)
        mean_returns[stochastic] = _shrink_expected_returns(
            sample_mean[stochastic], return_alpha
        )
        stochastic_covariance = sample_covariance[
            np.ix_(stochastic_indices, stochastic_indices)
        ]
        stochastic_covariance = _shrink_covariance(
            stochastic_covariance, covariance_alpha
        )
        stochastic_covariance, eigenvalue_floor = _repair_covariance(
            stochastic_covariance
        )
        covariance[np.ix_(stochastic_indices, stochastic_indices)] = (
            stochastic_covariance
        )

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
        deterministic_assets=tuple(
            str(clean.columns[index]) for index in np.flatnonzero(deterministic)
        ),
    )


def resolve_portfolio_estimates(
    returns: pd.DataFrame,
    *,
    portfolio_estimates: Optional[PortfolioEstimates] = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
) -> PortfolioEstimates:
    """Reuse precomputed inputs only for the exact cleaned return sample."""
    if portfolio_estimates is None:
        return estimate_portfolio_inputs(
            returns,
            covariance_shrinkage=covariance_shrinkage,
            return_shrinkage=return_shrinkage,
        )

    estimates = portfolio_estimates
    candidate = pd.DataFrame(returns)
    input_symbols = tuple(str(column) for column in candidate.columns)
    if estimates.symbols != input_symbols:
        raise ValueError("portfolio_estimates symbols must match return columns in order.")

    # The fast path avoids cleaning again when a caller passes the estimator's
    # own complete-case frame. Otherwise compare against the same cleaning
    # contract used to create the estimates. Matching symbols alone is unsafe:
    # it can leak a full-sample estimate into a training fold.
    cleaned = (
        candidate if candidate.equals(estimates.returns) else clean_returns(candidate)
    )
    exact_match = (
        cleaned.index.equals(estimates.returns.index)
        and cleaned.columns.equals(estimates.returns.columns)
        and np.array_equal(
            cleaned.to_numpy(dtype=float),
            estimates.returns.to_numpy(dtype=float),
        )
    )
    if not exact_match:
        raise ValueError(
            "portfolio_estimates must match the exact cleaned returns index and values."
        )
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
    risk_free_rate: float = 0.0,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    return_shrinkage: float = DEFAULT_RETURN_SHRINKAGE,
    deterministic_assets: Optional[Sequence[str]] = None,
) -> PortfolioEstimates:
    """Return Black-Litterman total returns from absolute total-return views.

    Reverse optimization produces annualized arithmetic excess returns.
    ``risk_free_rate`` is an effective annual rate, converted to annualized
    arithmetic units before addition to the prior. Views are absolute annualized
    arithmetic simple returns, and Sharpe subtracts the converted rate once.
    """
    estimates = estimate_portfolio_inputs(
        returns,
        covariance_shrinkage=covariance_shrinkage,
        return_shrinkage=return_shrinkage,
        deterministic_assets=deterministic_assets,
    )
    symbols = list(estimates.symbols)
    weights = _aligned_weights(market_weights, symbols, "market_weights")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("market_weights must be non-negative with a positive sum.")
    weights = weights / float(weights.sum())
    delta = float(risk_aversion)
    uncertainty_scale = float(tau)
    risk_free = float(risk_free_rate)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("risk_aversion must be positive.")
    if not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0:
        raise ValueError("tau must be positive.")
    if not np.isfinite(risk_free) or risk_free <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -1.")

    covariance = estimates.covariance
    equilibrium_excess = delta * covariance @ weights
    arithmetic_risk_free = annual_effective_to_arithmetic(risk_free, estimates.trading_days)
    equilibrium = arithmetic_risk_free + equilibrium_excess
    supplied_views = dict(views or {})
    unknown = [symbol for symbol in supplied_views if symbol not in symbols]
    if unknown:
        raise ValueError(f"views contain unknown assets: {', '.join(unknown)}.")
    confidence_map = dict(view_confidences or {})
    unused_confidences = [
        symbol for symbol in confidence_map if symbol not in supplied_views
    ]
    if unused_confidences:
        raise ValueError(
            "view_confidences contain assets without views: "
            + ", ".join(unused_confidences)
            + "."
        )

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
            if not np.isfinite(view_returns[row]):
                raise ValueError(f"view return for {symbol} must be finite.")
            confidence = float(confidence_map.get(symbol, 0.50))
            if not np.isfinite(confidence) or not 0.0 < confidence <= 1.0:
                raise ValueError(f"view confidence for {symbol} must be in (0, 1].")
            resolved_confidences[symbol] = confidence
            base_uncertainty = float(pick[row] @ scaled_covariance @ pick[row])
            omega_diagonal[row] = max(
                base_uncertainty * (1.0 - confidence) / confidence,
                1e-12,
            )
        # Apply the Woodbury form of the Black-Litterman update.  Solving in
        # view space avoids two N x N pseudoinverses when only K assets have
        # views (normally K is much smaller than N).
        view_covariance = pick @ scaled_covariance @ pick.T
        view_covariance.flat[:: len(view_symbols) + 1] += omega_diagonal
        view_adjustment = np.linalg.solve(
            view_covariance,
            view_returns - pick @ equilibrium,
        )
        posterior = equilibrium + scaled_covariance @ pick.T @ view_adjustment
    else:
        view_symbols = []
        resolved_confidences = {}
        posterior = equilibrium

    details = {
        "risk_aversion": delta,
        "tau": uncertainty_scale,
        "risk_free_rate": risk_free,
        "annualized_arithmetic_risk_free_rate": arithmetic_risk_free,
        "return_convention": "annualized_arithmetic_simple_return",
        "market_weights": {
            symbol: float(value)
            for symbol, value in zip(symbols, weights, strict=False)
        },
        "equilibrium_returns": {
            symbol: float(value)
            for symbol, value in zip(symbols, equilibrium, strict=False)
        },
        "equilibrium_total_returns": {
            symbol: float(value)
            for symbol, value in zip(symbols, equilibrium, strict=False)
        },
        "equilibrium_excess_returns": {
            symbol: float(value)
            for symbol, value in zip(symbols, equilibrium_excess, strict=False)
        },
        "prior_return_type": "annual_total_return",
        "view_return_type": "absolute_annual_total_return",
        "views": {symbol: float(supplied_views[symbol]) for symbol in view_symbols},
        "view_confidences": resolved_confidences,
    }
    return replace(
        estimates,
        mean_returns=np.asarray(posterior, dtype=float),
        expected_return_method="black_litterman",
        expected_return_details=details,
    )
