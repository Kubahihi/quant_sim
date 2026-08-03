from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# Yahoo symbols are converted to the common convention "USD per one unit of
# currency". Cross rates can then be produced consistently for any supported
# reporting currency without relying on the provider's pair orientation.
FX_USD_QUOTES: dict[str, tuple[str, bool]] = {
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "AUD": ("AUDUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "JPY": ("JPY=X", True),
    "CHF": ("CHF=X", True),
    "CAD": ("CAD=X", True),
    "CZK": ("CZK=X", True),
    "PLN": ("PLN=X", True),
    "SEK": ("SEK=X", True),
    "NOK": ("NOK=X", True),
    "DKK": ("DKK=X", True),
    "CNY": ("CNY=X", True),
    "HKD": ("HKD=X", True),
    "SGD": ("SGD=X", True),
}

SUPPORTED_CURRENCIES: tuple[str, ...] = ("USD", *FX_USD_QUOTES.keys())


def _currency_code(value: object) -> str:
    code = str(value or "").strip().upper()
    if code not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported currency {code or '<empty>'}. "
            f"Supported currencies: {', '.join(SUPPORTED_CURRENCIES)}."
        )
    return code


def required_fx_symbols(currencies: Iterable[str], base_currency: str = "USD") -> tuple[str, ...]:
    """Return the minimal Yahoo symbol set needed to construct the cross rates."""
    base = _currency_code(base_currency)
    requested = {_currency_code(currency) for currency in currencies}
    requested.add(base)
    symbols = {
        FX_USD_QUOTES[currency][0]
        for currency in requested
        if currency != "USD"
    }
    return tuple(sorted(symbols))


def build_fx_rate_history(
    market_prices: pd.DataFrame,
    currencies: Iterable[str],
    *,
    base_currency: str = "USD",
) -> pd.DataFrame:
    """Build rates expressed as units of base currency per foreign currency.

    A positive return therefore always means that the foreign currency
    strengthened against the reporting currency and increased an unhedged long
    foreign asset's value in the reporting currency.
    """
    if not isinstance(market_prices, pd.DataFrame):
        raise TypeError("market_prices must be a pandas DataFrame.")
    base = _currency_code(base_currency)
    requested = list(dict.fromkeys(_currency_code(currency) for currency in currencies))
    if base not in requested:
        requested.append(base)

    clean = market_prices.copy()
    clean.columns = [str(column).strip().upper() for column in clean.columns]
    clean = clean.loc[:, ~clean.columns.duplicated(keep="first")]
    clean = clean.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.sort_index().ffill()

    missing = [symbol for symbol in required_fx_symbols(requested, base) if symbol not in clean.columns]
    if missing:
        raise ValueError(f"Missing FX market history for: {', '.join(missing)}.")

    usd_value: dict[str, pd.Series] = {}
    template = pd.Series(1.0, index=clean.index, dtype=float)
    usd_value["USD"] = template
    for currency in set(requested):
        if currency == "USD":
            continue
        symbol, invert = FX_USD_QUOTES[currency]
        quote = clean[symbol].where(clean[symbol] > 0)
        usd_value[currency] = 1.0 / quote if invert else quote

    rates = pd.DataFrame(index=clean.index)
    base_usd = usd_value[base]
    for currency in requested:
        rates[currency] = usd_value[currency] / base_usd
    rates[base] = 1.0
    return rates.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def aggregate_currency_exposure(
    positions: pd.DataFrame,
    current_rates: Mapping[str, float] | pd.Series,
    *,
    base_currency: str = "USD",
) -> pd.DataFrame:
    """Aggregate asset-level local market values into currency exposure."""
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame.")
    required = {"Currency", "MarketValueLocal"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"positions is missing columns: {', '.join(sorted(missing))}.")
    base = _currency_code(base_currency)
    clean = positions.copy()
    clean["Currency"] = clean["Currency"].map(_currency_code)
    clean["MarketValueLocal"] = pd.to_numeric(clean["MarketValueLocal"], errors="coerce")
    if clean["MarketValueLocal"].isna().any() or not np.isfinite(clean["MarketValueLocal"]).all():
        raise ValueError("MarketValueLocal must contain finite numbers.")
    rates = {str(key).strip().upper(): float(value) for key, value in dict(current_rates).items()}
    rates[base] = 1.0
    needed = sorted(set(clean["Currency"]))
    invalid_rates = [currency for currency in needed if currency not in rates or not np.isfinite(rates[currency]) or rates[currency] <= 0]
    if invalid_rates:
        raise ValueError(f"Missing or invalid current rates for: {', '.join(invalid_rates)}.")

    clean["RateToBase"] = clean["Currency"].map(rates)
    clean["ExposureBase"] = clean["MarketValueLocal"] * clean["RateToBase"]
    clean["GrossRowExposureBase"] = clean["ExposureBase"].abs()
    grouped = (
        clean.groupby("Currency", as_index=False, sort=True)
        .agg(
            LocalMarketValue=("MarketValueLocal", "sum"),
            RateToBase=("RateToBase", "last"),
            NetExposureBase=("ExposureBase", "sum"),
            GrossExposureBase=("GrossRowExposureBase", "sum"),
            PositionCount=("Currency", "size"),
        )
    )
    gross = float(grouped["GrossExposureBase"].sum())
    grouped["GrossShare"] = grouped["GrossExposureBase"] / gross if gross > 0 else 0.0
    grouped["BaseCurrency"] = base
    return grouped


def _aligned_fx_inputs(
    exposure_table: pd.DataFrame,
    rate_history: pd.DataFrame,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    if not isinstance(exposure_table, pd.DataFrame) or "Currency" not in exposure_table or "NetExposureBase" not in exposure_table:
        raise ValueError("exposure_table must contain Currency and NetExposureBase.")
    if not isinstance(rate_history, pd.DataFrame):
        raise TypeError("rate_history must be a pandas DataFrame.")
    base = str(exposure_table.get("BaseCurrency", pd.Series([""])).iloc[0]).upper()
    currencies = [
        str(row.Currency).upper()
        for row in exposure_table.itertuples(index=False)
        if str(row.Currency).upper() != base and abs(float(row.NetExposureBase)) > 1e-12
    ]
    missing = [currency for currency in currencies if currency not in rate_history.columns]
    if missing:
        raise ValueError(f"rate_history is missing currencies: {', '.join(missing)}.")
    exposure_by_currency = exposure_table.set_index("Currency")["NetExposureBase"].astype(float)
    exposures = exposure_by_currency.reindex(currencies).to_numpy(dtype=float)
    returns = (
        rate_history[currencies]
        .apply(pd.to_numeric, errors="coerce")
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    return currencies, exposures, returns


def calculate_fx_risk(
    exposure_table: pd.DataFrame,
    rate_history: pd.DataFrame,
    *,
    confidence: float = 0.95,
    trading_days: int = 252,
) -> dict[str, object]:
    """Calculate standalone FX volatility, VaR, expected shortfall and contributions."""
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    currencies, exposures, returns = _aligned_fx_inputs(exposure_table, rate_history)
    if currencies and len(returns) < 2:
        raise ValueError("At least two aligned FX return observations are required.")

    if not currencies:
        empty = pd.DataFrame(columns=["Currency", "NetExposureBase", "AnnualizedRiskContributionBase", "RiskContributionPct"])
        return {
            "DailyVolatilityBase": 0.0,
            "AnnualizedVolatilityBase": 0.0,
            "ParametricVaRBase": 0.0,
            "HistoricalVaRBase": 0.0,
            "ExpectedShortfallBase": 0.0,
            "Observations": 0,
            "DailyPnL": pd.Series(dtype=float),
            "Contributions": empty,
        }

    covariance = returns.cov().to_numpy(dtype=float)
    daily_variance = max(float(exposures @ covariance @ exposures), 0.0)
    daily_vol = float(np.sqrt(daily_variance))
    annual_vol = daily_vol * float(np.sqrt(trading_days))
    daily_pnl = returns.mul(exposures, axis=1).sum(axis=1)
    loss_quantile = float(daily_pnl.quantile(1.0 - confidence))
    historical_var = max(-loss_quantile, 0.0)
    tail = daily_pnl[daily_pnl <= loss_quantile]
    expected_shortfall = max(-float(tail.mean()), 0.0) if not tail.empty else historical_var
    parametric_var = NormalDist().inv_cdf(confidence) * daily_vol

    if daily_vol > 0:
        component_daily = exposures * (covariance @ exposures) / daily_vol
        component_annual = component_daily * np.sqrt(trading_days)
        contribution_pct = component_annual / annual_vol
    else:
        component_annual = np.zeros_like(exposures)
        contribution_pct = np.zeros_like(exposures)
    contributions = pd.DataFrame(
        {
            "Currency": currencies,
            "NetExposureBase": exposures,
            "AnnualizedRiskContributionBase": component_annual,
            "RiskContributionPct": contribution_pct,
        }
    )
    return {
        "DailyVolatilityBase": daily_vol,
        "AnnualizedVolatilityBase": annual_vol,
        "ParametricVaRBase": float(parametric_var),
        "HistoricalVaRBase": historical_var,
        "ExpectedShortfallBase": expected_shortfall,
        "Observations": int(len(returns)),
        "DailyPnL": daily_pnl,
        "Contributions": contributions,
    }


def optimize_currency_hedges(
    exposure_table: pd.DataFrame,
    rate_history: pd.DataFrame,
    *,
    annual_cost_bps: float | Mapping[str, float] = 10.0,
    risk_aversion: float = 1.0,
    max_hedge_ratio: float = 1.0,
    trading_days: int = 252,
) -> dict[str, object]:
    """Optimize currency hedge ratios against risk and annual hedge cost.

    The objective is annualized FX variance as a fraction of gross portfolio
    exposure, multiplied by ``risk_aversion``, plus annual hedge cost as a
    fraction of gross exposure. Ratios are long-only hedges of the existing
    exposure and cannot reverse it.
    """
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be a non-negative finite number.")
    if not np.isfinite(max_hedge_ratio) or not 0 <= max_hedge_ratio <= 1:
        raise ValueError("max_hedge_ratio must be between 0 and 1.")
    currencies, exposures, returns = _aligned_fx_inputs(exposure_table, rate_history)
    if currencies and len(returns) < 2:
        raise ValueError("At least two aligned FX return observations are required.")
    if not currencies:
        return {
            "Plan": pd.DataFrame(),
            "BeforeAnnualVolatilityBase": 0.0,
            "AfterAnnualVolatilityBase": 0.0,
            "RiskReductionPct": 0.0,
            "EstimatedAnnualCostBase": 0.0,
            "Success": True,
        }

    if isinstance(annual_cost_bps, Mapping):
        costs_bps = np.asarray([float(annual_cost_bps.get(currency, 0.0)) for currency in currencies])
    else:
        costs_bps = np.full(len(currencies), float(annual_cost_bps), dtype=float)
    if not np.isfinite(costs_bps).all() or (costs_bps < 0).any():
        raise ValueError("annual_cost_bps must be non-negative and finite.")
    gross = float(exposure_table["GrossExposureBase"].abs().sum())
    if gross <= 0:
        gross = float(np.abs(exposures).sum()) or 1.0
    annual_covariance = returns.cov().to_numpy(dtype=float) * float(trading_days)

    def objective(ratios: np.ndarray) -> float:
        residual = exposures * (1.0 - ratios)
        variance_fraction = max(float(residual @ annual_covariance @ residual), 0.0) / (gross * gross)
        cost_fraction = float(np.sum(np.abs(exposures) * ratios * costs_bps / 10_000.0)) / gross
        return float(risk_aversion) * variance_fraction + cost_fraction

    result = minimize(
        objective,
        x0=np.zeros(len(currencies), dtype=float),
        method="SLSQP",
        bounds=[(0.0, float(max_hedge_ratio)) for _ in currencies],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise ValueError(f"Currency hedge optimization failed: {result.message}")
    ratios = np.clip(np.asarray(result.x, dtype=float), 0.0, float(max_hedge_ratio))
    residual = exposures * (1.0 - ratios)
    before_vol = float(np.sqrt(max(float(exposures @ annual_covariance @ exposures), 0.0)))
    after_vol = float(np.sqrt(max(float(residual @ annual_covariance @ residual), 0.0)))
    costs = np.abs(exposures) * ratios * costs_bps / 10_000.0
    rates = exposure_table.set_index("Currency")["RateToBase"].astype(float)

    plan = pd.DataFrame(
        {
            "Currency": currencies,
            "NetExposureBase": exposures,
            "HedgeRatio": ratios,
            "HedgeNotionalBase": np.abs(exposures) * ratios,
            "HedgeNotionalLocal": np.abs(exposures) * ratios / rates.reindex(currencies).to_numpy(dtype=float),
            "Direction": [
                f"Sell {currency} / Buy {str(exposure_table['BaseCurrency'].iloc[0]).upper()}"
                if exposure > 0
                else f"Buy {currency} / Sell {str(exposure_table['BaseCurrency'].iloc[0]).upper()}"
                for currency, exposure in zip(currencies, exposures)
            ],
            "ResidualExposureBase": residual,
            "AnnualCostBps": costs_bps,
            "EstimatedAnnualCostBase": costs,
        }
    )
    return {
        "Plan": plan,
        "BeforeAnnualVolatilityBase": before_vol,
        "AfterAnnualVolatilityBase": after_vol,
        "RiskReductionPct": 1.0 - after_vol / before_vol if before_vol > 0 else 0.0,
        "EstimatedAnnualCostBase": float(costs.sum()),
        "Success": True,
        "ObjectiveValue": float(result.fun),
    }


def build_fx_stress_table(
    exposure_table: pd.DataFrame,
    shocks: Mapping[str, float],
    *,
    hedge_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply deterministic FX shocks to current and residual exposures."""
    if not isinstance(shocks, Mapping):
        raise TypeError("shocks must be a currency-to-return mapping.")
    residuals: dict[str, float] = {}
    if isinstance(hedge_plan, pd.DataFrame) and not hedge_plan.empty:
        needed = {"Currency", "ResidualExposureBase"}
        if not needed.issubset(hedge_plan.columns):
            raise ValueError("hedge_plan must contain Currency and ResidualExposureBase.")
        residuals = hedge_plan.set_index("Currency")["ResidualExposureBase"].astype(float).to_dict()

    rows: list[dict[str, object]] = []
    for row in exposure_table.itertuples(index=False):
        currency = str(row.Currency).upper()
        shock = float(shocks.get(currency, 0.0))
        if not np.isfinite(shock) or shock <= -1.0:
            raise ValueError(f"Shock for {currency} must be finite and greater than -100%.")
        exposure = float(row.NetExposureBase)
        residual = float(residuals.get(currency, exposure))
        rows.append(
            {
                "Currency": currency,
                "Shock": shock,
                "NetExposureBase": exposure,
                "UnhedgedPnLBase": exposure * shock,
                "ResidualExposureBase": residual,
                "HedgedPnLBase": residual * shock,
                "HedgeBenefitBase": (exposure - residual) * shock,
            }
        )
    return pd.DataFrame(rows)
