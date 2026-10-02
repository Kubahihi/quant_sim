"""Walk-forward condition-arrival calibration, not a timing-strategy backtest."""
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from src.analytics.deployment import day
from src.analytics.entry_quality import _prepare, calculate_entry_quality
from src.analytics.deployment_timing import TimingConfig, _features, _eligible, summarize_analogues


@dataclass
class TimingValidation:
    observations: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict


def _base_rate(features, config):
    """Past-only unconditional frequency on non-overlapping eligible starting states."""
    waits, last = [], -config.horizon - 1
    for j in range(len(features) - config.horizon - 1):
        if j - last <= config.horizon or features[j] is None or _eligible(features[j], config):
            continue
        future = features[j + 1:j + config.horizon + 1]
        if any(f is None for f in future):
            continue
        waits.append(next((h for h, f in enumerate(future, 1) if _eligible(f, config)), config.horizon + 1))
        last = j
    if len(waits) < config.min_cases:
        return None
    return {str(h): float(np.mean(np.array(waits) <= h)) for h in (5, 10, 20)}


def validate_timing_features(features, dates, *, evaluation_start, evaluation_end, config=None):
    """Pure feature-level diagnostic. Each forecast sees only features through its origin."""
    config = config or TimingConfig()
    config.validate()
    start, end = day(evaluation_start), day(evaluation_end)
    if start > end:
        raise ValueError('Evaluation start must not follow its end.')
    dates = pd.DatetimeIndex(dates)
    if len(dates) != len(features) or dates.has_duplicates or dates.hasnans or not dates.is_monotonic_increasing:
        raise ValueError('Ordered unique dates matching features are required.')
    rows = []
    candidates = [i for i, d in enumerate(dates) if start <= d <= end]
    for i in candidates[::config.horizon + 1]:
        row = dict(date=str(dates[i].date()), status='unavailable', sample_count=0)
        if features[i] is None:
            rows.append(row)
            continue
        past = features[:i + 1]
        estimate = summarize_analogues(past, dates[:i + 1], features[i], as_of=dates[i], config=config)
        row.update(status=estimate.status, sample_count=estimate.sample_count)
        if estimate.status != 'estimated':
            rows.append(row)
            continue
        baseline = _base_rate(past, config)
        for h in (5, 10, 20):
            row[f'forecast_{h}'] = estimate.probabilities[str(h)]
            row[f'baseline_{h}'] = baseline[str(h)] if baseline is not None else np.nan
            row[f'outcome_{h}'] = np.nan
            # Outcomes are evaluated only AFTER producing and recording the forecast.
            observed = features[i + 1:i + h + 1]
            if len(observed) == h and all(f is not None for f in observed):
                row[f'outcome_{h}'] = int(any(_eligible(f, config) for f in observed))
        rows.append(row)
    obs = pd.DataFrame(rows)
    summaries = []
    for h in (5, 10, 20):
        columns = [f'forecast_{h}', f'baseline_{h}', f'outcome_{h}']
        usable = obs.reindex(columns=columns).dropna()
        if usable.empty:
            summaries.append(dict(horizon=h, evaluated=0, brier=np.nan, baseline_brier=np.nan,
                                  brier_improvement=np.nan, mean_forecast=np.nan, observed_frequency=np.nan))
            continue
        forecast, baseline, actual = (usable[c] for c in columns)
        brier = float(((forecast - actual) ** 2).mean())
        base_brier = float(((baseline - actual) ** 2).mean())
        summaries.append(dict(horizon=h, evaluated=len(usable), brier=brier, baseline_brier=base_brier,
            brier_improvement=base_brier - brier, mean_forecast=float(forecast.mean()), observed_frequency=float(actual.mean())))
    return TimingValidation(obs, pd.DataFrame(summaries), dict(
        evaluation_start=str(start.date()), evaluation_end=str(end.date()), config=asdict(config),
        requested_origins=len(rows), estimable_origins=sum(r['status'] == 'estimated' for r in rows),
        interpretation='Lower Brier is better; positive improvement is relative to a past-only base rate. '
                       'No significance or profitability claim follows from this diagnostic.',
        selection='Origins every 21 observed sessions. Incomplete, already-met and insufficient cases are retained.',
        limitations='Retrospective walk-forward diagnostic, not proof of prospectively held-out validation. '
                    'Threshold selection, price revisions, shared histories and small samples limit inference. '
                    'This does not execute a strategy or automatically unlock timing for competition use.'))


def validate_deployment_timing(ticker, prices, benchmark_prices, *, benchmark='SPY', evaluation_start,
                               evaluation_end, data_as_of, config=None):
    """Compute each historical feature from then-available input, retaining missing outcomes."""
    config = config or TimingConfig()
    config.validate()
    cutoff = day(data_as_of)
    p, warnings = _prepare(prices, cutoff, ticker, 7)
    b, bw = _prepare(benchmark_prices, cutoff, benchmark, 7)
    if warnings or bw or len(p) < 252:
        raise ValueError('; '.join(warnings + bw) or 'At least 252 daily security observations required.')
    if day(evaluation_end) > cutoff:
        raise ValueError('Evaluation end cannot exceed the data cutoff.')
    dates = p.index[251:]
    features = [_features(calculate_entry_quality(ticker, p, b, benchmark=benchmark, as_of=d)) for d in dates]
    result = validate_timing_features(features, dates, evaluation_start=evaluation_start,
                                      evaluation_end=evaluation_end, config=config)
    result.metadata.update(ticker=ticker, benchmark=benchmark, data_as_of=str(cutoff.date()),
        security_sha256=sha256(p.to_csv().encode()).hexdigest(), benchmark_sha256=sha256(b.to_csv().encode()).hexdigest())
    return result
