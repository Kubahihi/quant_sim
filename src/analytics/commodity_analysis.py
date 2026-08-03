from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# Exchange-traded proxies and Yahoo Finance continuous futures are kept separate
# in the UI so users can see whether they are analysing an investable vehicle or
# a futures-price proxy. Contract multipliers are deliberately not hard-coded:
# they vary by contract and must be verified for the instrument being traded.
COMMODITY_CATALOG: tuple[dict[str, str], ...] = (
    {"ticker": "DBC", "name": "Invesco DB Commodity Index Tracking Fund", "group": "Broad basket", "vehicle": "ETF"},
    {"ticker": "PDBC", "name": "Invesco Optimum Yield Diversified Commodity Strategy", "group": "Broad basket", "vehicle": "ETF"},
    {"ticker": "GSG", "name": "iShares S&P GSCI Commodity-Indexed Trust", "group": "Broad basket", "vehicle": "ETF"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "group": "Precious metals", "vehicle": "ETF"},
    {"ticker": "SLV", "name": "iShares Silver Trust", "group": "Precious metals", "vehicle": "ETF"},
    {"ticker": "CPER", "name": "United States Copper Index Fund", "group": "Industrial metals", "vehicle": "ETF"},
    {"ticker": "USO", "name": "United States Oil Fund", "group": "Energy", "vehicle": "ETF"},
    {"ticker": "BNO", "name": "United States Brent Oil Fund", "group": "Energy", "vehicle": "ETF"},
    {"ticker": "UNG", "name": "United States Natural Gas Fund", "group": "Energy", "vehicle": "ETF"},
    {"ticker": "DBA", "name": "Invesco DB Agriculture Fund", "group": "Agriculture", "vehicle": "ETF"},
    {"ticker": "CORN", "name": "Teucrium Corn Fund", "group": "Agriculture", "vehicle": "ETF"},
    {"ticker": "WEAT", "name": "Teucrium Wheat Fund", "group": "Agriculture", "vehicle": "ETF"},
    {"ticker": "SOYB", "name": "Teucrium Soybean Fund", "group": "Agriculture", "vehicle": "ETF"},
    {"ticker": "GC=F", "name": "Gold continuous futures", "group": "Precious metals", "vehicle": "Futures proxy"},
    {"ticker": "SI=F", "name": "Silver continuous futures", "group": "Precious metals", "vehicle": "Futures proxy"},
    {"ticker": "HG=F", "name": "Copper continuous futures", "group": "Industrial metals", "vehicle": "Futures proxy"},
    {"ticker": "CL=F", "name": "WTI crude oil continuous futures", "group": "Energy", "vehicle": "Futures proxy"},
    {"ticker": "BZ=F", "name": "Brent crude oil continuous futures", "group": "Energy", "vehicle": "Futures proxy"},
    {"ticker": "NG=F", "name": "Natural gas continuous futures", "group": "Energy", "vehicle": "Futures proxy"},
    {"ticker": "ZC=F", "name": "Corn continuous futures", "group": "Agriculture", "vehicle": "Futures proxy"},
    {"ticker": "ZW=F", "name": "Wheat continuous futures", "group": "Agriculture", "vehicle": "Futures proxy"},
    {"ticker": "ZS=F", "name": "Soybean continuous futures", "group": "Agriculture", "vehicle": "Futures proxy"},
)

COMMODITY_SYMBOLS = frozenset(item["ticker"] for item in COMMODITY_CATALOG)
_CATALOG_BY_TICKER = {item["ticker"]: item for item in COMMODITY_CATALOG}


def commodity_catalog_frame() -> pd.DataFrame:
    """Return a display-ready copy of the supported commodity starter universe."""
    return pd.DataFrame(COMMODITY_CATALOG).rename(
        columns={"ticker": "Ticker", "name": "Instrument", "group": "Group", "vehicle": "Vehicle"}
    )


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        return pd.DataFrame(index=prices.index)
    clean = prices.copy()
    clean.columns = [str(column).strip().upper() for column in clean.columns]
    clean = clean.loc[:, ~clean.columns.duplicated(keep="first")]
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).sort_index().ffill()
    return clean.dropna(axis=1, how="all")


def calculate_commodity_metrics(
    prices: pd.DataFrame,
    *,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Calculate comparable return and risk metrics for commodity price proxies."""
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    if not np.isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be finite.")

    clean = _clean_prices(prices)
    rows: list[dict[str, object]] = []
    for ticker in clean.columns:
        series = clean[ticker].dropna()
        if series.empty:
            continue
        daily_returns = series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        observations = int(series.size)
        years = max((observations - 1) / float(trading_days), 0.0)
        annualized_return = (
            float((series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0)
            if years > 0 and series.iloc[0] > 0 and series.iloc[-1] >= 0
            else np.nan
        )
        annualized_volatility = (
            float(daily_returns.std(ddof=1) * np.sqrt(trading_days))
            if daily_returns.size >= 2
            else np.nan
        )
        sharpe = (
            (annualized_return - float(annual_risk_free_rate)) / annualized_volatility
            if np.isfinite(annualized_return)
            and np.isfinite(annualized_volatility)
            and annualized_volatility > 0
            else np.nan
        )
        running_peak = series.cummax()
        drawdowns = series.div(running_peak).sub(1.0)

        def horizon_return(days: int) -> float:
            if observations <= days or series.iloc[-days - 1] == 0:
                return np.nan
            return float(series.iloc[-1] / series.iloc[-days - 1] - 1.0)

        metadata = _CATALOG_BY_TICKER.get(ticker, {})
        rows.append(
            {
                "Ticker": ticker,
                "Instrument": metadata.get("name", "Custom commodity instrument"),
                "Group": metadata.get("group", "Custom"),
                "Vehicle": metadata.get("vehicle", "Custom ticker"),
                "LastPrice": float(series.iloc[-1]),
                "Return1M": horizon_return(21),
                "Return3M": horizon_return(63),
                "Return12M": horizon_return(252),
                "AnnualizedReturn": annualized_return,
                "AnnualizedVolatility": annualized_volatility,
                "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
                "MaxDrawdown": float(drawdowns.min()),
                "Observations": observations,
            }
        )
    return pd.DataFrame(rows)


def build_cumulative_index(prices: pd.DataFrame, *, base: float = 100.0) -> pd.DataFrame:
    """Rebase each price series independently for a comparable performance chart."""
    if not np.isfinite(base) or base <= 0:
        raise ValueError("base must be a positive finite number.")
    clean = _clean_prices(prices)
    indexed = pd.DataFrame(index=clean.index)
    for ticker in clean.columns:
        series = clean[ticker].dropna()
        if series.empty or series.iloc[0] == 0:
            continue
        indexed[ticker] = clean[ticker] / float(series.iloc[0]) * float(base)
    return indexed.dropna(how="all")


def build_return_correlation(prices: pd.DataFrame, *, min_periods: int = 20) -> pd.DataFrame:
    """Return the pairwise daily-return correlation matrix."""
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1.")
    clean = _clean_prices(prices)
    returns = clean.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.corr(min_periods=min_periods)


def build_price_shock_table(
    current_price: float,
    units: float,
    *,
    contract_multiplier: float = 1.0,
    fx_to_usd: float = 1.0,
    shocks: Iterable[float] = (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Calculate direct mark-to-market P/L for explicit commodity price shocks."""
    values = {
        "current_price": current_price,
        "units": units,
        "contract_multiplier": contract_multiplier,
        "fx_to_usd": fx_to_usd,
    }
    normalized: dict[str, float] = {}
    for name, value in values.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        normalized[name] = numeric

    shock_values = [float(value) for value in shocks]
    if not shock_values:
        raise ValueError("At least one shock is required.")
    if any(not np.isfinite(value) or value <= -1.0 for value in shock_values):
        raise ValueError("Shocks must be finite and greater than -100%.")

    price = normalized["current_price"]
    notional = (
        price
        * normalized["units"]
        * normalized["contract_multiplier"]
        * normalized["fx_to_usd"]
    )
    return pd.DataFrame(
        [
            {
                "Shock": shock,
                "StressedPrice": price * (1.0 + shock),
                "PositionValueUSD": notional * (1.0 + shock),
                "PnLUSD": notional * shock,
            }
            for shock in shock_values
        ]
    )
