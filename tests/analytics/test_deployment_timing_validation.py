import numpy as np
import pandas as pd
import pytest

from src.analytics.deployment_timing import TimingConfig
from src.analytics.deployment_timing_validation import validate_timing_features


def fixture():
    features = [np.array([60., 60., 60.]) for _ in range(700)]
    for k in range(0, 32, 2):
        features[k * 21 + 3] = np.array([85., 70., 70.])
    return features, pd.bdate_range('2020-01-01', periods=len(features))


def test_forecasts_are_past_only_but_outcomes_change():
    features, dates = fixture()
    a = validate_timing_features(features, dates, evaluation_start=dates[420], evaluation_end=dates[420])
    features[421:441] = [np.array([60., 60., 60.])] * 20
    b = validate_timing_features(features, dates, evaluation_start=dates[420], evaluation_end=dates[420])
    row_a, row_b = a.observations.iloc[0], b.observations.iloc[0]
    assert row_a.status == 'estimated'
    for h in (5, 10, 20):
        assert row_a[f'forecast_{h}'] == row_b[f'forecast_{h}']
        assert row_a[f'baseline_{h}'] == row_b[f'baseline_{h}']
        assert row_a[f'outcome_{h}'] == 1
        assert row_b[f'outcome_{h}'] == 0


def test_brier_score_and_base_rate_use_the_same_evaluable_origins():
    features, dates = fixture()
    result = validate_timing_features(features, dates, evaluation_start=dates[420], evaluation_end=dates[600])
    for h in (5, 10, 20):
        obs = result.observations.dropna(subset=[f'forecast_{h}', f'baseline_{h}', f'outcome_{h}'])
        summary = result.summary.set_index('horizon').loc[h]
        assert summary.evaluated == len(obs)
        assert summary.brier == pytest.approx(((obs[f'forecast_{h}'] - obs[f'outcome_{h}']) ** 2).mean())
    assert 'not proof' in result.metadata['limitations']


def test_incomplete_outcomes_are_not_counted_as_failures():
    features, dates = fixture()
    result = validate_timing_features(features, dates, evaluation_start=dates[-1], evaluation_end=dates[-1])
    assert len(result.observations) == 1
    assert result.summary.evaluated.eq(0).all()
    assert result.summary.brier.isna().all()


def test_empty_evaluation_and_invalid_configuration():
    features, dates = fixture()
    result = validate_timing_features(features, dates, evaluation_start='2030-01-01', evaluation_end='2030-02-01')
    assert result.metadata['requested_origins'] == 0
    assert result.summary.evaluated.eq(0).all()
    with pytest.raises(ValueError):
        TimingConfig(min_cases=5.5).validate()
