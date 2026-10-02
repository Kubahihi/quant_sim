import numpy as np
import pandas as pd
import pytest

from src.analytics.deployment_timing import TimingConfig, estimate_deployment_timing, summarize_analogues


def test_nonarrival_counts_and_nonoverlapping_cases():
    features = [np.array([60., 60., 60.]) for _ in range(253)]
    for k in range(0, 12, 2):
        features[k * 21 + 3] = np.array([85., 70., 70.])
    dates = pd.bdate_range('2020-01-01', periods=len(features))
    result = summarize_analogues(features, dates, np.array([60., 60., 60.]),
                                as_of=dates[-1], config=TimingConfig())
    assert result.status == 'estimated'
    assert result.sample_count == 12
    assert result.probabilities == {'5': .5, '10': .5, '20': .5}
    assert result.no_hit_probability == .5
    assert result.median_sessions == 3
    indexes = sorted(dates.get_loc(c['date']) for c in result.cases)
    assert all(b - a > 20 for a, b in zip(indexes, indexes[1:]))


def test_insufficient_sample_suppresses_estimates():
    features = [np.array([60., 60., 60.])] * 100
    result = summarize_analogues(features, pd.bdate_range('2020-01-01', periods=100),
        features[-1], as_of='2021-01-01', config=TimingConfig())
    assert result.status == 'insufficient'
    assert not result.probabilities
    assert result.no_hit_probability is None


def test_no_hits_are_not_dropped_or_given_a_finite_median():
    features = [np.array([60., 60., 60.])] * 253
    result = summarize_analogues(features, pd.bdate_range('2020-01-01', periods=253),
        features[-1], as_of='2021-01-01', config=TimingConfig())
    assert result.no_hit_probability == 1
    assert result.probabilities['20'] == 0
    assert result.median_sessions is None


def test_ready_is_not_a_future_forecast():
    result = summarize_analogues([], [], np.array([85., 70., 70.]), as_of='2026-01-01', config=TimingConfig())
    assert result.status == 'ready'
    assert not result.probabilities


def test_future_prices_cannot_change_estimate():
    index = pd.bdate_range('2022-01-03', periods=550)
    p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(8).normal(0, .01, len(index)))), index=index)
    b = p.copy()
    original = estimate_deployment_timing('TEST', p, b, as_of=index[500])
    p.iloc[501:] = np.nan
    b.iloc[501:] *= 100
    actual = estimate_deployment_timing('TEST', p, b, as_of=index[500])
    assert actual.to_dict() == original.to_dict()
    assert all(pd.Timestamp(c['date']) < index[480] for c in actual.cases)


def test_missing_benchmark_does_not_create_timing():
    p = pd.Series(np.linspace(100, 150, 400), index=pd.bdate_range('2020-01-01', periods=400))
    result = estimate_deployment_timing('TEST', p, None, as_of=p.index[-1])
    assert result.status == 'unavailable'
    assert result.warnings


def test_invalid_config():
    with pytest.raises(ValueError):
        TimingConfig(min_cases=1).validate()
