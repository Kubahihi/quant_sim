import pandas as pd
import numpy as np
from typing import Optional


def calculate_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate correlation matrix"""
    return returns.corr(method=method)


def calculate_covariance_matrix(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate covariance matrix"""
    return returns.cov()


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """Calculate beta vs market"""
    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
    ).dropna(how="any")
    if len(aligned) < 2:
        return 0.0
    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Return series must contain only finite values.")
    covariance = aligned["asset"].cov(aligned["market"])
    market_variance = aligned["market"].var()
    
    if market_variance == 0:
        return 0.0
    
    return float(covariance / market_variance)


def calculate_alpha(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """Calculate arithmetic annualized Jensen alpha from periodic excess returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -100%.")

    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
    ).dropna(how="any")
    if aligned.shape[0] < 2:
        return 0.0

    asset = aligned["asset"].astype(float)
    market = aligned["market"].astype(float)
    if not np.isfinite(asset.to_numpy()).all() or not np.isfinite(market.to_numpy()).all():
        raise ValueError("Return series must contain only finite values.")

    periodic_rf = float(np.expm1(np.log1p(risk_free_rate) / periods_per_year))
    beta = calculate_beta(asset, market)
    alpha_periodic = (asset - periodic_rf).mean() - beta * (market - periodic_rf).mean()
    return float(alpha_periodic * periods_per_year)
