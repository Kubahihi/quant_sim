"""Deterministic deployment policy for already-selected securities; no forecasts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.price_alignment import normalize_daily_series
from src.analytics.returns import calculate_returns
from src.analytics.risk_metrics import calculate_rolling_volatility, calculate_volatility
from src.analytics.technical import calculate_rsi


DEFAULT_WEIGHTS = dict(trend=.25, momentum=.20, volatility=.20, drawdown=.15,
                       relative_strength=.20)


@dataclass(frozen=True)
class DeploymentBand:
    minimum: float
    rating: str
    fraction: float
    action: str


DEFAULT_BANDS = (
    DeploymentBand(0, "Unfavorable", .10, "Delay most deployment and require review."),
    DeploymentBand(40, "Neutral / Cautious", .35, "Use phased deployment."),
    DeploymentBand(60, "Moderately Favorable", .70, "Phase in the remaining allocation."),
    DeploymentBand(80, "Favorable", 1., "Deploy approximately the intended allocation."),
)


@dataclass
class EntryQualityConfig:
    weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    bands: tuple[DeploymentBand, ...] = DEFAULT_BANDS
    max_stale_days: int = 7

    def validate(self):
        values = list(self.weights.values())
        if (set(self.weights) != set(DEFAULT_WEIGHTS) or
                not all(np.isfinite(v) and v >= 0 for v in values) or
                not np.isclose(sum(values), 1., atol=1e-8, rtol=0)):
            raise ValueError("Weights must contain all five components, be nonnegative and sum to 1.")
        if (not self.bands or self.bands[0].minimum != 0 or
                any(not np.isfinite(b.minimum) or not 0 <= b.minimum <= 100 or
                    not np.isfinite(b.fraction) or not 0 <= b.fraction <= 1 for b in self.bands) or
                any(a.minimum >= b.minimum or a.fraction > b.fraction
                    for a, b in zip(self.bands, self.bands[1:]))):
            raise ValueError("Deployment bands must start at zero and increase monotonically.")
        if not isinstance(self.max_stale_days, int) or self.max_stale_days < 0:
            raise ValueError("max_stale_days must be a nonnegative integer.")


@dataclass
class EntryComponent:
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    explanation: str = "Insufficient usable history."
    warnings: list[str] = field(default_factory=list)


@dataclass
class EntryQualityResult:
    ticker: str
    benchmark: str
    as_of: str
    entry_score: float | None
    rating: str
    recommended_deployment_pct: float | None
    target_weight: float | None
    recommended_initial_weight: float | None
    remaining_undeployed_weight: float | None
    components: dict[str, EntryComponent]
    warnings: list[str]
    data_quality: dict
    explanation: str
    action: str
    weights: dict[str, float]
    methodology_version: str = "1.0"

    def to_dict(self):
        return asdict(self)


def _day(value):
    return pd.Timestamp(value).tz_localize(None).normalize()


def _prepare(prices, as_of, label, max_stale_days):
    """Reject corrupt input rather than filling prices or compressing missing sessions."""
    warnings = []
    if prices is None or len(prices) == 0:
        return pd.Series(dtype=float), [f"{label}: missing price data."]
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        return pd.Series(dtype=float), [f"{label}: dated daily Series required."]
    if prices.index.hasnans:
        return pd.Series(dtype=float), [f"{label}: invalid session dates."]
    prices = prices.copy()
    prices.index = prices.index.tz_localize(None).normalize()
    prices = prices.loc[prices.index <= as_of]  # truncate BEFORE quality checks/calculations
    try:
        prices = normalize_daily_series(prices)
    except ValueError as exc:
        return pd.Series(dtype=float), [f"{label}: {exc}"]
    prices = pd.to_numeric(prices, errors="coerce")
    if prices.empty:
        return prices, [f"{label}: no observations at or before the evaluation date."]
    if prices.index.hasnans or not np.isfinite(prices).all() or (prices <= 0).any():
        warnings.append(f"{label}: missing, nonfinite or nonpositive observations; repair the source.")
    if (as_of - prices.index[-1]).days > max_stale_days:
        warnings.append(f"{label}: stale data (last session {prices.index[-1].date()}).")
    if prices.index.to_series().diff().dt.days.gt(7).any():
        warnings.append(f"{label}: gaps longer than seven calendar days; daily continuity unverified.")
    if len(prices) > 2 and prices.index.to_series().diff().dt.days.median() > 3:
        warnings.append(f"{label}: observations are not daily session data.")
    if warnings:
        return pd.Series(dtype=float), warnings
    return prices.astype(float), warnings


def _bounded(x):
    return float(50 + 50 * np.tanh(x))


def _component(score, metrics, explanation, warnings=()):
    return EntryComponent(float(np.clip(score, 0, 100)),
                          {k: float(v) for k, v in metrics.items()}, explanation, list(warnings))


def calculate_entry_quality(ticker: str, prices: pd.Series | None,
                            benchmark_prices: pd.Series | None, *, benchmark: str = "SPY",
                            target_weight: float | None = None, as_of=None,
                            config: EntryQualityConfig | None = None) -> EntryQualityResult:
    """Use daily adjusted closes. Explicit as_of makes historical evaluation reproducible.

    All components are required, even when assigned zero weight. No redistribution.
    A missing component suppresses both composite and allocation recommendations.
    """
    config = config or EntryQualityConfig()
    config.validate()
    if target_weight is not None and (not np.isfinite(target_weight) or not 0 <= target_weight <= 1):
        raise ValueError("target_weight must be a fraction between 0 and 1.")
    date = _day(as_of if as_of is not None else pd.Timestamp.now(tz="UTC"))
    if pd.isna(date):
        raise ValueError("as_of must be a valid date.")
    p, warnings = _prepare(prices, date, ticker, config.max_stale_days)
    b, bw = _prepare(benchmark_prices, date, benchmark, config.max_stale_days)
    warnings.extend(bw)
    components = {name: EntryComponent() for name in DEFAULT_WEIGHTS}
    r = calculate_returns(p)
    if len(p) >= 252:
        vol = calculate_volatility(r.tail(20))
        daily = max(calculate_volatility(r.tail(63), annualize=False), .005)
        ma50, ma200 = p.rolling(50).mean(), p.rolling(200).mean()
        d50, d200 = p.iloc[-1] / ma50.iloc[-1] - 1, p.iloc[-1] / ma200.iloc[-1] - 1
        cross = ma50.iloc[-1] / ma200.iloc[-1] - 1
        slope50 = ma50.iloc[-1] / ma50.iloc[-21] - 1
        slope200 = ma200.iloc[-1] / ma200.iloc[-21] - 1
        extension = max(0., d50 / (daily * np.sqrt(50)) - 2)
        trend = np.mean([_bounded(d50 / (daily * np.sqrt(50))),
                         _bounded(d200 / (daily * np.sqrt(200))),
                         _bounded(cross / (daily * np.sqrt(150))),
                         _bounded(slope50 / (daily * np.sqrt(20))),
                         _bounded(slope200 / (daily * np.sqrt(20)))]) - min(30, 15 * extension)
        components['trend'] = _component(trend, dict(distance_sma50=d50, distance_sma200=d200,
            sma50_vs_sma200=cross, slope50_20d=slope50, slope200_20d=slope200,
            extension_penalty=min(30, 15 * extension)),
            f"Price is {d50:+.1%} versus SMA50 and {d200:+.1%} versus SMA200; "
            f"20-session SMA50 slope is {slope50:+.1%}. Extension penalty: {min(30, 15 * extension):.1f} points.")

        rsi = calculate_rsi(p)
        ret21, ret63 = p.iloc[-1] / p.iloc[-22] - 1, p.iloc[-1] / p.iloc[-64] - 1
        rsi_score = np.interp(rsi, [0, 30, 55, 70, 80, 100], [0, 25, 100, 100, 50, 0])
        momentum = .35 * _bounded(ret21 / (daily * np.sqrt(21))) + .35 * _bounded(ret63 / (daily * np.sqrt(63))) + .30 * rsi_score
        components['momentum'] = _component(momentum, dict(return_21d=ret21, return_63d=ret63, rsi14=rsi),
            f"1M/3M returns are {ret21:+.1%}/{ret63:+.1%}; RSI(14) is {rsi:.0f}" +
            (", with an extreme-momentum penalty." if rsi > 70 else "."))

        rolling = calculate_rolling_volatility(r, window=20)
        past = rolling.iloc[:-1].dropna().tail(756)
        percentile = float(((past < vol).sum() + .5 * (past == vol).sum()) / len(past))
        median = float(past.median())
        ratio = vol / max(median, .01)
        vs = .7 * (100 - 80 * percentile) + .3 * (100 / (1 + (vol / .40) ** 2))
        vs -= min(30, max(0, ratio - 1.5) * 20)
        if (r.tail(20).abs() < 1e-12).sum() >= 15 or vol < .005:
            components['volatility'] = EntryComponent(metrics={'realized_volatility_20d': vol},
                explanation="Near-static prices make entry risk unreliable.",
                warnings=["Near-static recent history; volatility component unavailable."])
        else:
            components['volatility'] = _component(vs, dict(realized_volatility_20d=vol,
                historical_percentile=percentile, historical_median=median, volatility_ratio=ratio),
                f"20-day annualized volatility is {vol:.1%}, at historical percentile {percentile:.0%} "
                f"and {ratio:.2f} times its trailing median.",
                ["Less than one year of prior volatility windows."] if len(past) < 252 else [])

        high, low = p.tail(252).max(), p.tail(252).min()
        dd = p.iloc[-1] / high - 1
        location = (p.iloc[-1] - low) / (high - low) if high > low else .5
        base = np.interp(dd, [-1, -.5, -.2, -.08, -.03, 0], [0, 5, 35, 90, 100, 85])
        location_score = base * (.25 + .75 * components['trend'].score / 100)
        components['drawdown'] = _component(location_score, dict(drawdown_252d=dd, range_location_252d=location),
            f"Price is {dd:.1%} below its 252-session closing high; range location is {location:.0%}. "
            "The location score is scaled by trend quality.")

    if len(p) >= 127 and not b.empty:
        window = p.tail(127)
        aligned = b.reindex(window.index)
        if aligned.notna().all():
            excess = {f'excess_{h}d': float(window.iloc[-1] / window.iloc[-h-1] - aligned.iloc[-1] / aligned.iloc[-h-1]) for h in (21, 63, 126)}
            active = calculate_returns(window) - calculate_returns(aligned)
            scale = max(calculate_volatility(active, annualize=False), .005)
            rs = np.mean([_bounded(excess[f'excess_{h}d'] / (scale * np.sqrt(h))) for h in (21, 63, 126)])
            components['relative_strength'] = _component(rs, excess,
                f"1M/3M/6M excess returns versus {benchmark}: " +
                "/".join(f"{v:+.1%}" for v in excess.values()) + ".")
        else:
            components['relative_strength'].warnings.append("Benchmark does not cover all last 127 security sessions; no filling applied.")

    for name, component in components.items():
        warnings.extend(component.warnings)
        if component.score is None:
            warnings.append(f"{name}: component unavailable.")
    complete = all(c.score is not None for c in components.values())
    score = float(sum(config.weights[k] * c.score for k, c in components.items())) if complete else None
    band = next((b for b in reversed(config.bands) if score is not None and score >= b.minimum), None)
    fraction = band.fraction if band else None
    initial = target_weight * fraction if target_weight is not None and fraction is not None else None
    return EntryQualityResult(ticker, benchmark, str(date.date()), score,
        band.rating if band else "Incomplete", fraction, target_weight, initial,
        target_weight - initial if initial is not None else None, components, list(dict.fromkeys(warnings)),
        dict(complete=complete, confidence="limited" if warnings or not complete else "standard",
             confidence_meaning="Data sufficiency only; not predictive confidence.",
             security_observations=len(p), benchmark_observations=len(b),
             last_security_session=str(p.index[-1].date()) if not p.empty else None),
        f"{ticker}: " + " ".join(c.explanation for c in components.values()),
        band.action if band else "Review data before making a deployment decision.", dict(config.weights))
