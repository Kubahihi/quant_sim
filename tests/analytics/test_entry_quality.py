import json

import numpy as np
import pandas as pd
import pytest

from src.analytics.entry_quality import (
    DEFAULT_WEIGHTS, DEFAULT_BANDS, DeploymentBand, EntryQualityConfig, calculate_entry_quality,
)
from src.analytics.entry_quality_validation import validate_entry_quality
from src.analytics.technical import calculate_rsi


def history(drift=.001, noise=.01, n=800, seed=7):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, noise, n))),
                     index=pd.bdate_range('2022-01-03', periods=n))


def evaluate(p, b=None, **kwargs):
    return calculate_entry_quality('TEST', p, history(seed=2) if b is None else b,
                                   as_of=p.index[-1], **kwargs)


@pytest.mark.parametrize('seed', range(12))
def test_bounded_deterministic_and_serializable(seed):
    p = history(seed=seed)
    a = evaluate(p)
    assert 0 <= a.entry_score <= 100
    assert all(0 <= c.score <= 100 for c in a.components.values())
    assert a.to_dict() == evaluate(p).to_dict()
    json.dumps(a.to_dict(), allow_nan=False)
    assert a.entry_score == pytest.approx(sum(a.weights[k] * c.score for k, c in a.components.items()))


@pytest.mark.parametrize('weights', [{}, {**DEFAULT_WEIGHTS, 'trend': .5},
    {**DEFAULT_WEIGHTS, 'trend': -.25}, {**DEFAULT_WEIGHTS, 'trend': float('nan')}])
def test_bad_weights(weights):
    with pytest.raises(ValueError, match='Weights'):
        evaluate(history(), config=EntryQualityConfig(weights=weights))


def test_custom_weights_and_bands():
    result = evaluate(history(), config=EntryQualityConfig(
        weights=dict(trend=1., momentum=0., volatility=0., drawdown=0., relative_strength=0.),
        bands=(DeploymentBand(0, 'Custom', .6, 'Stage remainder.'),)))
    assert result.entry_score == result.components['trend'].score
    assert result.recommended_deployment_pct == .6


@pytest.mark.parametrize('n', [0, 5, 100, 251])
def test_short_history(n):
    result = calculate_entry_quality('IPO', history(n=n), history(), as_of='2025-01-24')
    assert result.entry_score is None
    assert result.recommended_deployment_pct is None
    assert result.warnings


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0, -10])
def test_invalid_observations(bad):
    p = history()
    p.iloc[-5] = bad
    assert evaluate(p).entry_score is None


def test_benchmark_missing_misaligned_and_stale():
    p = history()
    for b in (pd.Series(dtype=float), history().drop(p.index[-20]), history().iloc[:-20]):
        result = evaluate(p, b)
        assert result.entry_score is None
        assert result.components['relative_strength'].score is None
        assert result.warnings
    result = calculate_entry_quality('TEST', p, None, as_of=p.index[-1])
    assert result.rating == 'Incomplete'


def test_stale_duplicate_gap_and_flat_data():
    p = history()
    stale = calculate_entry_quality('TEST', p, p, as_of=p.index[-1] + pd.Timedelta(days=9))
    assert stale.entry_score is None
    for invalid in (pd.concat([p, p.tail(1)]), p.drop(p.index[100:110]), p * 0 + 100):
        assert evaluate(invalid).entry_score is None


def test_weekly_and_invalid_dates_are_rejected():
    p = history()
    p.index = pd.date_range('2010-01-01', periods=len(p), freq='W')
    assert evaluate(p, p).entry_score is None
    p.index = pd.DatetimeIndex([pd.NaT, *p.index[1:]])
    assert evaluate(p, p).entry_score is None


def test_shared_rsi_preserves_ordinary_screener_values():
    from src.stock_picker.screener import _compute_rsi

    p = history()
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = -delta.clip(upper=0).rolling(14).mean().iloc[-1]
    assert _compute_rsi(p) == pytest.approx(100 - 100 / (1 + gain / loss))


def test_trends_and_collapse_location():
    up, down = evaluate(history(drift=.003)), evaluate(history(drift=-.003))
    assert up.components['trend'].score > down.components['trend'].score
    assert up.entry_score > down.entry_score
    assert down.components['drawdown'].score < 20


def test_rsi_extremes():
    assert calculate_rsi(pd.Series(np.arange(30.))) == 100
    assert calculate_rsi(pd.Series(-np.arange(30.))) == 0
    assert calculate_rsi(pd.Series(np.ones(30))) == 50
    p = history()
    p.iloc[-15:] = p.iloc[-16] * np.exp(np.arange(1, 16) * .01)
    result = evaluate(p)
    assert result.components['momentum'].metrics['rsi14'] == 100
    assert result.components['momentum'].score <= 70


def test_extreme_volatility_penalty():
    p = history()
    extreme = p.copy()
    extreme.iloc[-20:] = p.iloc[-21] * np.exp(np.cumsum(np.tile([.2, -.2], 10)))
    assert evaluate(extreme).components['volatility'].score < evaluate(p).components['volatility'].score


def test_target_weight_and_invalid_config():
    result = evaluate(history(), target_weight=.05)
    assert result.recommended_initial_weight == pytest.approx(.05 * result.recommended_deployment_pct)
    assert result.remaining_undeployed_weight + result.recommended_initial_weight == pytest.approx(.05)
    assert evaluate(history()).recommended_initial_weight is None
    for weight in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            evaluate(history(), target_weight=weight)
    with pytest.raises(ValueError):
        evaluate(history(), config=EntryQualityConfig(bands=tuple(reversed(DEFAULT_BANDS))))


def test_future_observations_cannot_change_historical_score():
    p, b = history(), history(seed=2)
    date = p.index[400]
    expected = calculate_entry_quality('TEST', p, b, as_of=date).to_dict()
    p.iloc[401:] = np.nan
    b.iloc[401:] *= 100
    assert calculate_entry_quality('TEST', p, b, as_of=date).to_dict() == expected
    assert calculate_entry_quality('TEST', p.loc[:date], b.loc[:date], as_of=date).to_dict() == expected


def test_validation_next_close_outcomes_and_censoring():
    p, b = history(), history(seed=2)
    date = p.index[400]
    result = validate_entry_quality('TEST', p, b, dates=[date, p.index[-1]])
    row = result.observations.iloc[0]
    assert row.entry_date == p.index[401]
    assert row.return_21d == pytest.approx(p.iloc[422] / p.iloc[401] - 1)
    assert result.observations.iloc[-1].filter(like='return_').isna().all()
    score = row.entry_score
    p.iloc[401:] *= 1.5
    changed = validate_entry_quality('TEST', p, b, dates=[date])
    assert changed.observations.iloc[0].entry_score == score
    assert result.summary[('return_21d', 'count')].sum() == 1


def test_validation_drawdown_includes_first_loss():
    p = history()
    date = p.index[400]
    p.iloc[401] = 100
    p.iloc[402:528] = 80
    row = validate_entry_quality('TEST', p, history(seed=2), dates=[date]).observations.iloc[0]
    assert row.max_drawdown_21d == pytest.approx(-.2)
    assert row.downside_deviation_21d > 0
