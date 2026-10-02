"""Experimental historical-analogue estimate of time to an observable condition.

This estimates condition arrival, not a profitable entry date. Never places orders
or changes the approved deployment policy.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.analytics.entry_quality import _prepare, calculate_entry_quality
from src.analytics.deployment import day


@dataclass(frozen=True)
class TimingConfig:
    horizon: int = 20
    min_cases: int = 12
    max_cases: int = 40
    max_distance: float = 1.
    score_threshold: float = 80.
    trend_threshold: float = 60.
    volatility_threshold: float = 50.

    def validate(self):
        if any(type(v) is not int for v in (self.horizon, self.min_cases, self.max_cases)):
            raise ValueError('Horizon and episode counts must be integers.')
        if self.horizon != 20:
            raise ValueError('Timing v1 uses a fixed 20-session horizon.')
        if not 5 <= self.min_cases <= self.max_cases <= 100:
            raise ValueError('Use 5–100 cases with minimum no greater than maximum.')
        if not np.isfinite(self.max_distance) or not 0 < self.max_distance <= 2:
            raise ValueError('Similarity distance must be finite and in (0, 2].')
        if any(not np.isfinite(x) or not 0 <= x <= 100 for x in
               (self.score_threshold, self.trend_threshold, self.volatility_threshold)):
            raise ValueError('Condition thresholds must be between 0 and 100.')


@dataclass
class TimingEstimate:
    as_of: str
    status: str
    condition: str
    sample_count: int = 0
    probabilities: dict = field(default_factory=dict)
    no_hit_probability: float | None = None
    median_sessions: int | None = None
    cases: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    version: str = '1.0-experimental'

    def to_dict(self):
        return asdict(self)


def _features(result):
    if result.entry_score is None:
        return None
    return np.array([result.entry_score, result.components['trend'].score,
                     result.components['volatility'].score], dtype=float)


def _eligible(features, config):
    return (features is not None and features[0] >= config.score_threshold and
            features[1] >= config.trend_threshold and features[2] >= config.volatility_threshold)


def summarize_analogues(features, dates, current, *, as_of, config):
    """Select by starting-state similarity BEFORE reading subsequent outcomes.

    Complete, non-overlapping 20-session outcome windows only. The current
    observation is excluded from all historical outcome windows (one-session embargo).
    """
    config.validate()
    result = TimingEstimate(str(day(as_of).date()), 'insufficient',
        f'Entry ≥ {config.score_threshold:g}, trend ≥ {config.trend_threshold:g}, '
        f'volatility score ≥ {config.volatility_threshold:g}', config=asdict(config),
        warnings=['Experimental analogue frequencies, not calibrated probabilities or evidence of better returns.',
                  'Outcome windows do not overlap; market regimes and shared estimation histories remain dependent.'])
    if _eligible(current, config):
        result.status = 'ready'
        return result
    candidates = []
    for i in range(len(features) - config.horizon - 1):
        feature = features[i]
        if feature is None or _eligible(feature, config) or any(x is None for x in features[i + 1:i + config.horizon + 1]):
            continue
        distance = float(np.sqrt(np.mean(((feature - current) / 20.) ** 2)))
        if distance <= config.max_distance:
            candidates.append((distance, i))
    selected = []
    for distance, i in sorted(candidates):
        if all(abs(i - j) > config.horizon for _, j in selected):
            selected.append((distance, i))
        if len(selected) == config.max_cases:
            break
    waits = []
    for distance, i in selected:
        wait = next((h for h in range(1, config.horizon + 1)
                     if _eligible(features[i + h], config)), None)
        waits.append(wait if wait is not None else config.horizon + 1)
        result.cases.append(dict(date=str(pd.Timestamp(dates[i]).date()), distance=distance,
                                 first_hit_sessions=wait, observed_sessions=config.horizon))
    result.sample_count = len(waits)
    if len(waits) < config.min_cases:
        result.warnings.append(f'Only {len(waits)} comparable episodes; at least {config.min_cases} required. No timing estimate shown.')
        return result
    result.status = 'estimated'
    result.probabilities = {str(h): float(np.mean(np.array(waits) <= h)) for h in (5, 10, 20)}
    result.no_hit_probability = float(np.mean(np.array(waits) > config.horizon))
    result.median_sessions = next((h for h in range(1, config.horizon + 1)
                                  if np.mean(np.array(waits) <= h) >= .5), None)
    return result


def estimate_deployment_timing(ticker, prices, benchmark_prices, *, benchmark='SPY', as_of,
                               config=None):
    config = config or TimingConfig()
    config.validate()
    date = day(as_of)
    p, warnings = _prepare(prices, date, ticker, 7)
    b, benchmark_warnings = _prepare(benchmark_prices, date, benchmark, 7)
    current = calculate_entry_quality(ticker, p, b, benchmark=benchmark, as_of=date)
    feature = _features(current)
    if feature is None:
        return TimingEstimate(str(date.date()), 'unavailable', 'Complete Entry Quality assessment required.',
                              warnings=warnings + benchmark_warnings + current.warnings, config=asdict(config))
    if _eligible(feature, config):
        return summarize_analogues([], [], feature, as_of=date, config=config)
    # Reuse the exact public score implementation at every historical date.
    dates = p.index[251:]
    features = [_features(calculate_entry_quality(ticker, p, b, benchmark=benchmark, as_of=d)) for d in dates]
    return summarize_analogues(features, dates, feature, as_of=date, config=config)
