"""Manual individual-bond adapter for the mixed-asset Quant Engine.

Individual bonds usually do not have a reliable Yahoo ticker history.  This
adapter values their contractual terms directly and creates a deterministic
historical risk proxy from a user-selected traded bond ETF.  The proxy is
re-scaled to the bond's entered annual volatility and shifted so its arithmetic
annual mean equals yield to worst less entered expected credit loss.  It is an approximation for covariance and
scenario work, never a reconstructed transaction-price history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .bond_analytics import calculate_bond_metrics


MANUAL_BOND_PREFIX = "BOND:"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not (isinstance(value, float) and math.isnan(value))
    return _text(value).casefold() in {"1", "true", "yes", "y", "callable"}


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    normalized = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        if name.strip().casefold() in normalized:
            return normalized[name.strip().casefold()]
    return None


def _symbol(identifier: str) -> str:
    clean = re.sub(r"[^A-Z0-9._-]+", "-", identifier.upper()).strip("-")
    return f"{MANUAL_BOND_PREFIX}{clean}"


def parse_manual_bond_rows(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    *,
    as_of: Any = None,
) -> list[dict[str, Any]]:
    """Validate editable-grid rows and return normalized analytical records."""
    settlement = _as_date(as_of) or date.today()
    if isinstance(rows, pd.DataFrame):
        records = rows.to_dict("records")
    else:
        records = list(rows or [])
    bonds: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            continue
        identifier = _text(_field(raw, "Identifier", "identifier", "ISIN", "ticker")).upper()
        if not identifier:
            continue
        label = f"Manual bond row {index} ({identifier})"
        symbol = _symbol(identifier)
        if symbol in seen:
            errors.append(f"{label}: duplicate identifier.")
            continue
        seen.add(symbol)

        weight_pct = _finite(_field(raw, "Weight %", "weight_pct", "weight"))
        clean_price = _finite(_field(raw, "Clean Price", "clean_price", "price"))
        face_value = _finite(_field(raw, "Face Value", "face_value"))
        quantity = _finite(_field(raw, "Quantity", "quantity"))
        coupon_pct = _finite(_field(raw, "Coupon %", "coupon_pct", "coupon"))
        ytm_pct = _finite(_field(raw, "YTM %", "ytm_pct", "ytm"))
        duration = _finite(_field(raw, "Modified Duration", "modified_duration", "duration"))
        convexity = _finite(_field(raw, "Convexity", "convexity"))
        annual_vol_pct = _finite(_field(raw, "Annual Volatility %", "annual_volatility_pct", "annual_volatility"))
        frequency_value = _finite(_field(raw, "Coupon Frequency", "coupon_frequency", "frequency"))
        frequency = int(frequency_value if frequency_value is not None else 2)
        maturity = _as_date(_field(raw, "Maturity", "maturity_date", "maturity"))
        proxy = _text(_field(raw, "Proxy Ticker", "proxy_ticker", "proxy")).upper()
        callable_bond = _truthy(_field(raw, "Callable", "callable"))
        call_date = _as_date(_field(raw, "First Call Date", "call_date"))
        call_price = _finite(_field(raw, "Call Price", "call_price"))
        default_probability_pct = _finite(_field(raw, "Default Probability %", "default_probability_pct", "default_probability"))
        recovery_rate_pct = _finite(_field(raw, "Recovery Rate %", "recovery_rate_pct", "recovery_rate"))

        if weight_pct is None or weight_pct <= 0 or weight_pct > 100:
            errors.append(f"{label}: Weight % must be greater than 0 and at most 100.")
        if clean_price is None or clean_price <= 0:
            errors.append(f"{label}: Clean Price must be positive.")
        if face_value is None:
            face_value = 1_000.0
        if face_value <= 0:
            errors.append(f"{label}: Face Value must be positive.")
        if quantity is None:
            quantity = 1.0
        if quantity <= 0:
            errors.append(f"{label}: Quantity must be positive.")
        if coupon_pct is None:
            coupon_pct = 0.0
        if coupon_pct < 0 or coupon_pct > 100:
            errors.append(f"{label}: Coupon % must be between 0 and 100.")
        if ytm_pct is not None and ytm_pct <= -95:
            errors.append(f"{label}: YTM % must be greater than -95.")
        if duration is not None and duration <= 0:
            errors.append(f"{label}: Modified Duration must be positive when supplied.")
        if convexity is not None and convexity < 0:
            errors.append(f"{label}: Convexity cannot be negative.")
        if annual_vol_pct is None or annual_vol_pct <= 0 or annual_vol_pct > 100:
            errors.append(f"{label}: Annual Volatility % must be greater than 0 and at most 100.")
        if maturity is None or maturity <= settlement:
            errors.append(f"{label}: Maturity must be after the Quant Engine end date.")
        if frequency not in {1, 2, 4, 12}:
            errors.append(f"{label}: Coupon Frequency must be 1, 2, 4, or 12.")
        if not proxy:
            errors.append(f"{label}: Proxy Ticker is required for covariance estimation.")
        if callable_bond and (call_date is None or maturity is None or not (settlement < call_date < maturity)):
            errors.append(f"{label}: First Call Date must be after the end date and before maturity.")
        if callable_bond and (call_price is None or call_price <= 0):
            errors.append(f"{label}: Call Price must be positive for a callable bond.")
        if default_probability_pct is None:
            default_probability_pct = 0.0
        if default_probability_pct < 0 or default_probability_pct > 100:
            errors.append(f"{label}: Default Probability % must be between 0 and 100.")
        if recovery_rate_pct is None:
            recovery_rate_pct = 40.0
        if recovery_rate_pct < 0 or recovery_rate_pct > 100:
            errors.append(f"{label}: Recovery Rate % must be between 0 and 100.")

        if any(message.startswith(f"{label}:") for message in errors):
            continue
        bond = {
            "id": symbol,
            "ticker": symbol,
            "display_identifier": identifier,
            "security_type": "Bond",
            "bond_instrument_type": "individual",
            "bond_category": _text(_field(raw, "Category", "bond_category")) or "Unspecified",
            "isin": identifier if len(identifier) == 12 else "",
            "issuer": _text(_field(raw, "Issuer", "issuer")),
            "currency": _text(_field(raw, "Currency", "currency")).upper() or "USD",
            "quantity": float(quantity),
            "face_value": float(face_value),
            "entry_price": float(clean_price),
            "last_price": float(clean_price),
            "accrued_interest": 0.0,
            "coupon_rate": float(coupon_pct) / 100.0,
            "coupon_frequency": frequency,
            "maturity_date": maturity.isoformat(),
            "yield_to_maturity": float(ytm_pct) / 100.0 if ytm_pct is not None else None,
            "modified_duration": float(duration) if duration is not None else None,
            "convexity": float(convexity) if convexity is not None else None,
            "credit_rating": _text(_field(raw, "Credit Rating", "credit_rating")),
            "callable": int(callable_bond),
            "call_date": call_date.isoformat() if callable_bond and call_date else None,
            "call_price": float(call_price) if callable_bond and call_price is not None else None,
            "default_probability": float(default_probability_pct) / 100.0,
            "recovery_rate": float(recovery_rate_pct) / 100.0,
            "fx_rate_to_usd": 1.0,
            "valuation_source": "Manual Quant Engine input",
            "price_observed_at": settlement.isoformat(),
            "quant_weight": float(weight_pct) / 100.0,
            "proxy_ticker": proxy,
            "annual_volatility": float(annual_vol_pct) / 100.0,
            "status": "open",
        }
        metrics = calculate_bond_metrics(bond, clean_price, as_of=settlement)
        if metrics.get("yield_to_worst") is None:
            errors.append(f"{label}: YTM could not be calculated; supply a valid YTM or contractual terms.")
            continue
        if metrics.get("modified_duration") is None:
            errors.append(f"{label}: duration could not be calculated; supply Modified Duration.")
            continue
        bonds.append(bond)
    if errors:
        raise ValueError(" ".join(errors))
    return bonds


def build_manual_bond_proxy_returns(
    bonds: Sequence[Mapping[str, Any]],
    proxy_returns: pd.DataFrame,
    *,
    as_of: Any = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Create deterministic mean/volatility-calibrated proxy return series."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    outputs: dict[str, pd.Series] = {}
    for raw in bonds:
        bond = dict(raw)
        symbol = _text(bond.get("ticker"))
        proxy = _text(bond.get("proxy_ticker")).upper()
        if not symbol or not proxy:
            raise ValueError("Every manual bond needs a synthetic symbol and proxy ticker.")
        if proxy not in proxy_returns.columns:
            raise ValueError(f"Proxy data is unavailable for {proxy} ({bond.get('display_identifier') or symbol}).")
        series = pd.to_numeric(proxy_returns[proxy], errors="coerce").dropna()
        if len(series) < 3:
            raise ValueError(f"Proxy {proxy} has insufficient return history.")
        proxy_std = float(series.std(ddof=1))
        if not math.isfinite(proxy_std) or proxy_std <= 0:
            raise ValueError(f"Proxy {proxy} has zero or invalid volatility.")
        target_vol = _finite(bond.get("annual_volatility"))
        if target_vol is None or target_vol <= 0:
            raise ValueError(f"Annual volatility is missing for {bond.get('display_identifier') or symbol}.")
        metrics = calculate_bond_metrics(bond, bond.get("last_price"), as_of=as_of)
        yield_to_worst = _finite(metrics.get("yield_to_worst"))
        if yield_to_worst is None:
            raise ValueError(f"Yield to worst is unavailable for {bond.get('display_identifier') or symbol}.")
        expected_loss_rate = _finite(metrics.get("expected_loss_rate")) or 0.0
        expected_return = yield_to_worst - expected_loss_rate
        daily_vol = target_vol / math.sqrt(float(periods_per_year))
        standardized = (series - float(series.mean())) / proxy_std
        outputs[symbol] = standardized * daily_vol + expected_return / float(periods_per_year)
    return pd.DataFrame(outputs).dropna(how="any")


def combine_hybrid_weights(
    market_tickers: Sequence[str],
    market_relative_weights: Sequence[float],
    manual_bonds: Sequence[Mapping[str, Any]],
    asset_columns: Sequence[str],
) -> np.ndarray:
    """Combine explicit bond weights with relative weights for the market sleeve."""
    market = [str(item).upper() for item in market_tickers]
    raw_market = np.asarray(market_relative_weights, dtype=float)
    if len(market) != raw_market.size:
        raise ValueError("Market weights must match the market tickers.")
    bond_weights = {
        _text(item.get("ticker")): float(_finite(item.get("quant_weight")) or 0.0)
        for item in manual_bonds
    }
    bond_total = float(sum(bond_weights.values()))
    if bond_total > 1.0 + 1e-9:
        raise ValueError("Manual bond weights exceed 100%.")
    available_market = [ticker for ticker in market if ticker in asset_columns]
    if available_market:
        relative = pd.Series(raw_market, index=market, dtype=float).reindex(available_market).fillna(0.0)
        if float(relative.sum()) <= 0:
            raise ValueError("Available market tickers have zero total weight.")
        market_map = (relative / float(relative.sum()) * max(1.0 - bond_total, 0.0)).to_dict()
    else:
        market_map = {}
        if manual_bonds and not math.isclose(bond_total, 1.0, abs_tol=0.001):
            raise ValueError("When the portfolio contains only manual bonds, their weights must total 100%.")
    combined = np.asarray(
        [bond_weights.get(str(column), market_map.get(str(column).upper(), 0.0)) for column in asset_columns],
        dtype=float,
    )
    if combined.size == 0 or float(combined.sum()) <= 0:
        raise ValueError("The combined portfolio has no positive weights.")
    return combined / float(combined.sum())


def build_manual_bond_metrics_table(
    bonds: Sequence[Mapping[str, Any]],
    *,
    as_of: Any = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bond in bonds:
        metrics = calculate_bond_metrics(bond, bond.get("last_price"), as_of=as_of)
        rows.append({
            "Identifier": bond.get("display_identifier") or bond.get("ticker"),
            "Issuer": bond.get("issuer") or "Unassigned",
            "Weight": bond.get("quant_weight"),
            "CleanPrice": metrics.get("clean_price"),
            "YieldToWorst": metrics.get("yield_to_worst"),
            "ExpectedLossRate": metrics.get("expected_loss_rate"),
            "ProxyExpectedReturn": (
                (_finite(metrics.get("yield_to_worst")) or 0.0)
                - (_finite(metrics.get("expected_loss_rate")) or 0.0)
            ),
            "ModifiedDuration": metrics.get("modified_duration"),
            "Convexity": metrics.get("convexity"),
            "DV01USD": metrics.get("dv01_usd"),
            "Maturity": metrics.get("maturity_date"),
            "AnnualVolatilityAssumption": bond.get("annual_volatility"),
            "ProxyTicker": bond.get("proxy_ticker"),
            "CreditRating": bond.get("credit_rating") or "Unrated",
        })
    return pd.DataFrame(rows)


__all__ = [
    "MANUAL_BOND_PREFIX", "build_manual_bond_metrics_table",
    "build_manual_bond_proxy_returns", "combine_hybrid_weights",
    "parse_manual_bond_rows",
]
